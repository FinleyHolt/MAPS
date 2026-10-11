"""Server-discovered endpoint facts: one probe, one owner, fail loud.

``EndpointSpec`` is THE authority for every served-LLM and
perception-service endpoint fact — which model is served, how big its
context window is, which service URLs answer. The facts are DISCOVERED
from the live endpoints by :func:`discover` (``GET {url}/v1/models`` +
per-service ``GET {url}/health``) exactly once per process, at startup,
before any inference or perception call. Client-side guesses (a YAML
model name, a hardcoded window constant, an env pin) do not exist; the
only human inputs are the URLs themselves (WHERE to connect) and — for
offline/replay or a non-vLLM endpoint — an explicit, loud pin funneled
through the same type.

Any divergence raises :class:`EndpointDiscoveryError` at process start,
naming the URL and exactly what was seen — never a silent per-call
404/400 cascade minutes into episode 0.

Usage at a process entry point (after the compute-profile env applies)::

    from maps.llm.endpoint_spec import discover_from_env, install_spec
    install_spec(discover_from_env())

Hot-path consumers receive the spec by explicit injection (the compiler
threads it into the reasoner); the few genuinely off-tree consumers read
the installed singleton back via :func:`spec`, which raises when no
install happened (a forgotten install fails loud, never defaults).

Escape hatches (explicit, never inferred):

  ``SEARCH_ENDPOINT_OFFLINE=1``
      Offline / replay: no endpoint to probe. :func:`discover` then
      REQUIRES an explicit pin (``SEARCH_ENDPOINT_SPEC_OVERRIDE``) —
      an offline run cannot probe, so it must declare the window it
      emulates.
  ``SEARCH_ENDPOINT_SPEC_OVERRIDE=<model_id>:<max_model_len>``
      Endpoint serving several models (or omitting ``max_model_len``).
      When the endpoint IS reachable the override's ``model_id`` is still
      cross-checked against the served ids — the override never bypasses
      the live check.
  ``SEARCH_REASONER_MODELS_URL=<url>``
      Where the served ids are listed, for an endpoint that lists them
      somewhere other than ``{SEARCH_REASONER_URL}/models``: Bedrock's
      bedrock-mantle serves GPT-5.6 under ``/openai/v1`` and its model
      list under ``/v1/models``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import time
import urllib.request
from dataclasses import dataclass

log = logging.getLogger(__name__)


# Probe timeout per URL. Discovery runs once at startup; a slow answer is
# tolerated, a hung one is not.
_PROBE_TIMEOUT_S = 15.0
# A transport failure is asked again before it is a verdict: the arm's ground
# box relays these ports over a link that re-forms when the box changes
# networks, and one closed connection during that would otherwise fail a Fly
# next outright.
_PROBE_TRIES = 3
_PROBE_RETRY_S = 3.0

# Output reserve cap: tokens held back for the current turn's thinking +
# completion. ``min(cap, window // 4)`` keeps a small window from going
# negative (the 262144-assumption bug: 32768 - 40000 < 0).
_OUTPUT_RESERVE_CAP = 40000

# Defensive floor on the request ``max_tokens`` pin: a known Qwen3
# thinking-completion size, so a small window cannot truncate a long
# thinking turn into a malformed/empty tool call. On windows >= 32768 the
# floor equals ``window // 4`` and never binds.
_MIN_OUTPUT_TOKENS = 8192

# Fixed slack between the compaction input budget and the served window,
# absorbing client-vs-server chat-template token skew so a one-token
# undercount never trips a server-side length 400.
_WINDOW_MARGIN_TOKENS = 1024

# The reasoner cannot operate below this input budget (case f).
_MIN_INPUT_BUDGET = 8000


class EndpointDiscoveryError(RuntimeError):
    """An endpoint fact could not be discovered, or diverged from the
    declared/derived expectation. Raised at process start; uncaught it
    exits the process in seconds, naming the URL and what was seen."""


@dataclass(frozen=True)
class EndpointOverride:
    """Explicit model pin for offline/replay or a non-vLLM endpoint.

    Canonical env form: ``SEARCH_ENDPOINT_SPEC_OVERRIDE=<model_id>:<max_model_len>``
    (parsed by :func:`parse_override`)."""

    model_id: str
    max_model_len: int


def parse_override(raw: str) -> EndpointOverride:
    """Parse the ``<model_id>:<max_model_len>`` override env format."""
    text = (raw or "").strip()
    model_id, sep, window = text.rpartition(":")
    if not sep or not model_id.strip() or not window.strip().isdigit():
        raise EndpointDiscoveryError(
            f"SEARCH_ENDPOINT_SPEC_OVERRIDE={raw!r} is malformed; expected "
            f"'<model_id>:<max_model_len>' (e.g. 'qwen36-35b-a3b:131072')"
        )
    return EndpointOverride(
        model_id=model_id.strip(), max_model_len=int(window.strip())
    )


@dataclass(frozen=True)
class EndpointSpec:
    """The discovered endpoint facts, frozen for the process lifetime.

    ``base_url`` is THE OpenAI-style base (typically ending in ``/v1``)
    discovery probed — one URL; the engine load-balances its own replicas
    behind it. ``sam_url`` may be a comma-list of replicas, frozen verbatim
    as probed."""

    base_url: str
    model_id: str
    max_model_len: int
    tokenizer_name: str
    output_reserve: int
    sam_url: str = ""

    @property
    def input_budget(self) -> int:
        """Compaction ceiling: prompt tokens must fit under this so that
        ``prompt + max_output_tokens`` can never exceed the served window.

        Subtracts ``max_output_tokens``, which is what the request actually
        pins, NOT ``output_reserve``. The two differ below a 32,768 window,
        where the ``_MIN_OUTPUT_TOKENS`` floor exceeds ``maxlen // 4``:
        subtracting the reserve there would hand compaction a ceiling the
        server rejects (at 24,576, budget 17,408 plus a pinned 8,192 is
        25,600 against the window).

        A no-op wherever the reserve already clears the floor, which is every
        window at or above 32,768 and so every served model in the fleet.
        """
        return (self.max_model_len - self.max_output_tokens
                - _WINDOW_MARGIN_TOKENS)

    @property
    def max_output_tokens(self) -> int:
        """The request ``max_tokens`` pin: the output reserve, floored at a
        known thinking-completion size (see ``_MIN_OUTPUT_TOKENS``)."""
        return max(self.output_reserve, _MIN_OUTPUT_TOKENS)


def _output_reserve_for(max_model_len: int) -> int:
    return min(_OUTPUT_RESERVE_CAP, max_model_len // 4)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def _get_json(url: str, *, bearer: str | None = None) -> object:
    """GET ``url`` and parse the JSON body; raise EndpointDiscoveryError on
    transport failure, non-200, or a non-JSON body. ``bearer`` rides an
    Authorization header — hosted OpenAI-compatible endpoints auth their
    /v1/models route, mirroring the chat path's bearer."""
    req = urllib.request.Request(url)
    if bearer:
        req.add_header("Authorization", f"Bearer {bearer}")
    for attempt in range(1, _PROBE_TRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT_S) as resp:
                status = resp.status
                body = resp.read()
            break
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
            if attempt == _PROBE_TRIES:
                raise EndpointDiscoveryError(
                    f"endpoint unreachable: GET {url} failed ({exc}) after {_PROBE_TRIES} tries"
                ) from exc
            log.warning("discovery: GET %s failed (%s); asking again in %.0f s", url, exc, _PROBE_RETRY_S)
            time.sleep(_PROBE_RETRY_S)
    if not (200 <= status < 300):
        raise EndpointDiscoveryError(
            f"endpoint unhealthy: GET {url} returned HTTP {status}"
        )
    try:
        return json.loads(body)
    except ValueError as exc:
        raise EndpointDiscoveryError(
            f"endpoint returned non-JSON: GET {url} -> {body[:120]!r}"
        ) from exc


def models_url(base_url: str) -> str:
    """Where the reasoner at ``base_url`` lists its served models: discovery
    and every later health probe ask the same URL, so a probe can never
    fail on an endpoint discovery accepted."""
    explicit = os.environ.get("SEARCH_REASONER_MODELS_URL", "").strip()
    if explicit:
        return explicit
    base = base_url.rstrip("/")
    return f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"


def _probe_models(base_url: str) -> list[dict]:
    """``GET {base_url}/v1/models`` -> the ``data`` model list (non-empty)."""
    payload = _get_json(
        models_url(base_url),
        bearer=os.environ.get("SEARCH_REASONER_API_KEY") or None,
    )
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        raise EndpointDiscoveryError(
            f"{models_url(base_url)} returned no served models "
            f"(data={data!r}) — the server is up but serves nothing"
        )
    return [m for m in data if isinstance(m, dict)]


def _facts_from_endpoint(
    base_url: str, override: EndpointOverride | None
) -> tuple[str, int]:
    """One endpoint's ``(model_id, max_model_len)``."""
    models = _probe_models(base_url)
    ids = [str(m.get("id") or "") for m in models]
    if override is not None:
        # The override never bypasses the live check: the pinned id must be
        # among the served ids.
        if override.model_id not in ids:
            raise EndpointDiscoveryError(
                f"SEARCH_ENDPOINT_SPEC_OVERRIDE names {override.model_id!r} "
                f"but {models_url(base_url)} serves {ids} — the pin does "
                f"not match the live endpoint"
            )
        served_len = next(m for m in models if m.get("id") == override.model_id).get("max_model_len")
        if served_len and int(served_len) != override.max_model_len:
            raise EndpointDiscoveryError(
                f"SEARCH_ENDPOINT_SPEC_OVERRIDE pins max_model_len="
                f"{override.max_model_len} but {base_url} reports "
                f"{served_len} for {override.model_id!r}"
            )
        return override.model_id, override.max_model_len
    if len(models) != 1:
        raise EndpointDiscoveryError(
            f"{models_url(base_url)} serves {len(models)} models ({ids}); "
            f"refusing to guess — set SEARCH_ENDPOINT_SPEC_OVERRIDE="
            f"'<model_id>:<max_model_len>' to pin one"
        )
    model = models[0]
    model_id = str(model.get("id") or "")
    max_model_len = model.get("max_model_len")
    if not model_id:
        raise EndpointDiscoveryError(
            f"{models_url(base_url)} model entry has no id: {model!r}"
        )
    if not max_model_len:
        raise EndpointDiscoveryError(
            f"{models_url(base_url)} reports no max_model_len for "
            f"{model_id!r} — cannot derive a token budget. If this is a "
            f"non-vLLM endpoint, set SEARCH_ENDPOINT_SPEC_OVERRIDE="
            f"'<model_id>:<max_model_len>'"
        )
    return model_id, int(max_model_len)


def _probe_sam_health(urls: str) -> None:
    """Probe every SAM replica's ``/health`` (comma-list tolerated). A payload
    carrying a ``service`` field must name sam: a stale or foreign server
    answering 200 on the right port aborts instead of producing a
    silently-wrong green run. A payload without the field is
    reachability-checked only."""
    for url in (u.strip().rstrip("/") for u in urls.split(",") if u.strip()):
        try:
            payload = _get_json(f"{url}/health")
        except EndpointDiscoveryError as exc:
            raise EndpointDiscoveryError(
                f"service 'sam' failed its /health probe: {exc}"
            ) from exc
        identity = str(payload.get("service") or "").lower() if isinstance(payload, dict) else ""
        if identity and "sam" not in identity:
            raise EndpointDiscoveryError(
                f"service 'sam' at {url} answered /health as {identity!r} — a "
                f"reachable-but-WRONG server; expected an identity containing 'sam'"
            )


#: The turns and tool a Reasoner's transcript carries, in the shape the
#: compactor sizes them (tool-call arguments parsed to a mapping): discovery
#: renders it once, so a template that cannot render any of them is refused
#: before a run starts.
_TEMPLATE_PROBE = {
    "messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "reasoning_content": "r", "tool_calls": [
            {"id": "c0", "type": "function", "function": {"name": "run", "arguments": {"code": "c"}}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "t"},
    ],
    "tools": [{"type": "function", "function": {
        "name": "run", "description": "d",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}}],
}


def _require_tokenizer(model_id: str) -> str:
    """Load the tokenizer for ``model_id`` and render its chat template once
    over :data:`_TEMPLATE_PROBE` (case g).

    A tokenizer that cannot load, or whose chat template cannot render that
    transcript, aborts at startup naming the repo, instead of leaving the
    compactor unable to size a prompt. Lazy import keeps
    ``spec()``/``install_spec()`` readers free of the transformers import."""
    from maps.llm.tokenizer import _hf_tokenizer_name, _tokenizer

    tokenizer_name = _hf_tokenizer_name(model_id)
    try:
        tok = _tokenizer(model_id)
    except RuntimeError as exc:
        raise EndpointDiscoveryError(str(exc)) from exc
    try:
        tok.apply_chat_template(_TEMPLATE_PROBE["messages"], tools=_TEMPLATE_PROBE["tools"],
                                add_generation_prompt=True, tokenize=False)
    except Exception as exc:  # noqa: BLE001 - any render failure refuses the tokenizer
        raise EndpointDiscoveryError(
            f"tokenizer {tokenizer_name!r} for served model {model_id!r} cannot render its chat "
            f"template ({type(exc).__name__}: {exc}). Point SEARCH_COMPACT_TOKENIZER_NAME at a "
            f"repo or directory that carries the model's chat template."
        ) from exc
    return tokenizer_name


# ---------------------------------------------------------------------------
# discover / install / read-back
# ---------------------------------------------------------------------------


def discover(
    base_url: str,
    sam_url: str = "",
    *,
    offline: bool = False,
    override: EndpointOverride | None = None,
) -> EndpointSpec:
    """Discover the endpoint facts, or raise :class:`EndpointDiscoveryError`.

    ``base_url``: THE reasoner endpoint, ``SEARCH_REASONER_URL``, the one
    legit human input. Exactly one URL: multi-GPU throughput is the
    engine's own data parallelism behind this single endpoint, so a
    comma-list is refused loud instead of routed client-side.

    ``sam_url``: the SAM replicas the run posts to (comma-list); empty is
    skipped, otherwise each must answer ``/health`` with a matching
    identity.

    ``offline=True`` skips all probes and REQUIRES ``override``.
    """
    url = str(base_url or "").strip().rstrip("/")
    if "," in url:
        raise EndpointDiscoveryError(
            f"SEARCH_REASONER_URL={base_url!r} is a comma-list — the "
            f"reasoner endpoint is ONE URL: the serving engine load-balances "
            f"its own data-parallel replicas behind it, and no "
            f"client-side endpoint picking exists"
        )
    sam_url = str(sam_url or "").strip()

    if offline:
        if override is None:
            raise EndpointDiscoveryError(
                "SEARCH_ENDPOINT_OFFLINE=1 but no explicit pin: an offline "
                "run cannot probe, so it must declare the endpoint it "
                "emulates via SEARCH_ENDPOINT_SPEC_OVERRIDE="
                "'<model_id>:<max_model_len>'"
            )
        model_id, max_model_len = override.model_id, override.max_model_len
    else:
        if not url:
            raise EndpointDiscoveryError(
                "no reasoner endpoint to probe: SEARCH_REASONER_URL is empty"
            )
        model_id, max_model_len = _facts_from_endpoint(url, override)
        if sam_url:
            _probe_sam_health(sam_url)

    from maps.llm.model_cards import card_for, check_effort  # noqa: PLC0415

    try:
        card_for(model_id)  # the sampling envelope: refused at startup, not mid-mission
    except ValueError as exc:
        raise EndpointDiscoveryError(str(exc)) from exc
    effort = os.environ.get("SEARCH_SAMPLING_REASONING_EFFORT", "").strip()
    if effort:
        try:
            check_effort(model_id, effort)
        except ValueError as exc:
            raise EndpointDiscoveryError(f"SEARCH_SAMPLING_REASONING_EFFORT: {exc}") from exc
    output_reserve = _output_reserve_for(max_model_len)
    tokenizer_name = _require_tokenizer(model_id)

    spec = EndpointSpec(
        base_url=url,
        model_id=model_id,
        max_model_len=max_model_len,
        tokenizer_name=tokenizer_name,
        output_reserve=output_reserve,
        sam_url=sam_url,
    )
    # Checked off the SPEC, never off a second spelling of the arithmetic.
    # Recomputing the budget inline writes the property's formula twice, and
    # the two drift the moment one of them changes. One source, so a fix
    # cannot land in half the places that use it.
    if spec.input_budget < _MIN_INPUT_BUDGET:
        raise EndpointDiscoveryError(
            f"served window max_model_len={max_model_len} leaves "
            f"input_budget={spec.input_budget} after max_output_tokens="
            f"{spec.max_output_tokens} + margin={_WINDOW_MARGIN_TOKENS} — "
            f"below the {_MIN_INPUT_BUDGET}-token reasoner minimum"
        )
    return spec


def discover_from_env() -> EndpointSpec:
    """The one canonical env -> :func:`discover` glue for process entry
    points (``run_benchmark.main``, ``bench_fullset``, ``launch``).

    Reads ``SEARCH_REASONER_URL`` + the SAM replicas, honors the explicit
    ``SEARCH_ENDPOINT_OFFLINE`` / ``SEARCH_ENDPOINT_SPEC_OVERRIDE`` escape
    hatches."""

    from maps.perception.sam.detect_client import sam_endpoints  # noqa: PLC0415

    override_raw = os.environ.get("SEARCH_ENDPOINT_SPEC_OVERRIDE", "").strip()
    try:
        sam_url = ",".join(sam_endpoints())  # the replicas the run posts to
    except ValueError as exc:
        raise EndpointDiscoveryError(str(exc)) from exc
    return discover(
        os.environ.get("SEARCH_REASONER_URL", "http://127.0.0.1:8001/v1"),
        sam_url,
        offline=os.environ.get("SEARCH_ENDPOINT_OFFLINE", "").strip() == "1",
        override=parse_override(override_raw) if override_raw else None,
    )


_SPEC: EndpointSpec | None = None
_SPEC_LOCK = threading.Lock()


def install_spec(new_spec: EndpointSpec) -> EndpointSpec:
    """Install the process-wide spec, exactly once.

    A second install with identical fields is a no-op (idempotent entry
    points); a second install with DIFFERENT fields raises — two divergent
    reads physically cannot coexist in one process."""
    global _SPEC
    with _SPEC_LOCK:
        if _SPEC is None:
            _SPEC = new_spec
            log.info(
                "endpoint spec installed: model_id=%s max_model_len=%d "
                "input_budget=%d max_output_tokens=%d base_url=%s",
                new_spec.model_id,
                new_spec.max_model_len,
                new_spec.input_budget,
                new_spec.max_output_tokens,
                new_spec.base_url,
            )
        elif _SPEC != new_spec:
            raise EndpointDiscoveryError(
                f"divergent second install_spec: installed={_SPEC} "
                f"new={new_spec}"
            )
        return _SPEC


def spec() -> EndpointSpec:
    """The installed spec. Raises when no install happened — a forgotten
    ``install_spec(discover_from_env())`` at the process entry point fails
    loud instead of silently defaulting to some model."""
    if _SPEC is None:
        raise EndpointDiscoveryError(
            "no EndpointSpec installed: the process entry point must run "
            "install_spec(discover_from_env()) before any consumer reads "
            "spec()"
        )
    return _SPEC


def spec_installed() -> bool:
    """Whether an install happened — for surfaces that legitimately run
    LLM-free (``--no-llm``) and record the served model as absent."""
    return _SPEC is not None
