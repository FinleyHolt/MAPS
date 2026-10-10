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
      the live check. A LoRA arm pins SGLang's ``<base>:<adapter>`` form
      (e.g. ``qwen36-35b-a3b:qwen36-35b-a3b-lora:262144``): both
      components must be served ids, and requests then carry the full
      colon form, which is what routes them through the adapter.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import time
import urllib.request
from collections.abc import Mapping, Sequence
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
    behind it. Service URLs may be comma-lists of replicas; they are
    frozen verbatim as probed."""

    base_url: str
    model_id: str
    max_model_len: int
    tokenizer_name: str
    output_reserve: int
    sam_url: str = ""
    sam_video_url: str = ""
    # Whether the endpoint reported its own ``max_model_len`` on
    # ``/v1/models``. That is the tell for a dedicated engine we
    # launched: a self-hosted vLLM declares its window, while a hosted
    # multi-tenant endpoint does not (which is the whole reason
    # ``SEARCH_ENDPOINT_SPEC_OVERRIDE`` exists). Callers read it to
    # decide whether a first call pays engine cold-start cost worth
    # pre-paying. False on an offline run: nothing was probed.
    server_declared_window: bool = True

    @property
    def input_budget(self) -> int:
        """Compaction ceiling: prompt tokens must fit under this so that
        ``prompt + max_output_tokens`` can never exceed the served window.

        Subtracts ``max_output_tokens``, which is what the request actually
        pins, NOT ``output_reserve``. The two differ below a 32,768 window,
        where the ``_MIN_OUTPUT_TOKENS`` floor exceeds ``maxlen // 4`` -- and
        there the old arithmetic handed compaction a ceiling the server would
        reject, so a transcript compacted exactly to budget 400s on length:
        at 24,576, budget 17,408 plus a pinned 8,192 is 25,600 against a
        24,576 window. The docstring said "modulo the floor on tiny windows"
        where it meant "wrong on tiny windows".

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


def _split_urls(raw: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = [p for chunk in raw for p in str(chunk).split(",")]
    return tuple(u.strip().rstrip("/") for u in parts if u.strip())


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


def _models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"


def _probe_models(base_url: str) -> list[dict]:
    """``GET {base_url}/v1/models`` -> the ``data`` model list (non-empty)."""
    payload = _get_json(
        _models_url(base_url),
        bearer=os.environ.get("SEARCH_REASONER_API_KEY") or None,
    )
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        raise EndpointDiscoveryError(
            f"{_models_url(base_url)} returned no served models "
            f"(data={data!r}) — the server is up but serves nothing"
        )
    return [m for m in data if isinstance(m, dict)]


def _facts_from_endpoint(
    base_url: str, override: EndpointOverride | None
) -> tuple[str, int, bool]:
    """One endpoint's (model_id, max_model_len, server_declared_window).

    The third element records whether the server stated its own window,
    which distinguishes a dedicated engine from a hosted endpoint — see
    :class:`EndpointSpec`.
    """
    models = _probe_models(base_url)
    ids = [str(m.get("id") or "") for m in models]
    if override is not None:
        # The override never bypasses the live check: the pinned id must be
        # among the served ids. A ``base:adapter`` pin (SGLang's OpenAI form
        # for routing a request through a served LoRA adapter — a BARE
        # adapter name in ``model`` silently serves the base) validates both
        # components; the window check reads the BASE entry, since adapter
        # entries carry ``max_model_len: null``.
        anchor_id = override.model_id
        if override.model_id not in ids:
            base, sep, adapter = override.model_id.partition(":")
            if not (sep and base in ids and adapter in ids):
                raise EndpointDiscoveryError(
                    f"SEARCH_ENDPOINT_SPEC_OVERRIDE names {override.model_id!r} "
                    f"but {_models_url(base_url)} serves {ids} — the pin does "
                    f"not match the live endpoint (a LoRA pin must be "
                    f"'<base>:<adapter>' with both ids served)"
                )
            anchor_id = base
        served = next(m for m in models if m.get("id") == anchor_id)
        served_len = served.get("max_model_len")
        if served_len and int(served_len) != override.max_model_len:
            raise EndpointDiscoveryError(
                f"SEARCH_ENDPOINT_SPEC_OVERRIDE pins max_model_len="
                f"{override.max_model_len} but {base_url} reports "
                f"{served_len} for {override.model_id!r}"
            )
        return override.model_id, override.max_model_len, bool(served_len)
    if len(models) != 1:
        raise EndpointDiscoveryError(
            f"{_models_url(base_url)} serves {len(models)} models ({ids}); "
            f"refusing to guess — set SEARCH_ENDPOINT_SPEC_OVERRIDE="
            f"'<model_id>:<max_model_len>' to pin one"
        )
    model = models[0]
    model_id = str(model.get("id") or "")
    max_model_len = model.get("max_model_len")
    if not model_id:
        raise EndpointDiscoveryError(
            f"{_models_url(base_url)} model entry has no id: {model!r}"
        )
    if not max_model_len:
        raise EndpointDiscoveryError(
            f"{_models_url(base_url)} reports no max_model_len for "
            f"{model_id!r} — cannot derive a token budget. If this is a "
            f"non-vLLM endpoint, set SEARCH_ENDPOINT_SPEC_OVERRIDE="
            f"'<model_id>:<max_model_len>'"
        )
    return model_id, int(max_model_len), True


# Per-service identity tokens: when a service's /health payload carries a
# ``service`` field, it must contain one of these substrings — a stale or
# foreign server answering 200 on the right port aborts instead of
# producing a silently-wrong green run. A payload without the field is
# reachability-checked only.
_SERVICE_IDENTITY_TOKENS: dict[str, tuple[str, ...]] = {
    "sam": ("sam",),
    "sam_video": ("sam",),
}


def _probe_service_health(name: str, urls: str) -> None:
    """Probe every replica of one service URL (comma-list tolerated)."""
    for url in _split_urls(urls):
        try:
            payload = _get_json(f"{url}/health")
        except EndpointDiscoveryError as exc:
            raise EndpointDiscoveryError(
                f"service {name!r} failed its /health probe: {exc}"
            ) from exc
        identity = ""
        if isinstance(payload, dict):
            identity = str(payload.get("service") or "").lower()
        expected = _SERVICE_IDENTITY_TOKENS.get(name, ())
        if identity and expected and not any(t in identity for t in expected):
            raise EndpointDiscoveryError(
                f"service {name!r} at {url} answered /health as "
                f"{identity!r} — a reachable-but-WRONG server; expected an "
                f"identity containing one of {expected}"
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
    service_urls: Mapping[str, str] | None = None,
    *,
    offline: bool = False,
    override: EndpointOverride | None = None,
) -> EndpointSpec:
    """Discover the endpoint facts, or raise :class:`EndpointDiscoveryError`.

    ``base_url``: THE reasoner endpoint, ``SEARCH_REASONER_URL``, the one
    legit human input. Exactly one URL: multi-GPU throughput is the
    engine's own data parallelism behind this single endpoint, so a
    comma-list is refused loud instead of routed client-side.

    ``service_urls``: ``{"sam"|"sam_video": url}``; empty
    values are skipped, every non-empty one must answer ``/health`` with a
    matching identity (comma-lists of replicas tolerated).

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
    services = {k: str(v or "").strip() for k, v in (service_urls or {}).items()}

    if offline:
        if override is None:
            raise EndpointDiscoveryError(
                "SEARCH_ENDPOINT_OFFLINE=1 but no explicit pin: an offline "
                "run cannot probe, so it must declare the endpoint it "
                "emulates via SEARCH_ENDPOINT_SPEC_OVERRIDE="
                "'<model_id>:<max_model_len>'"
            )
        # Nothing was probed, so nothing was declared.
        model_id, max_model_len = override.model_id, override.max_model_len
        server_declared_window = False
    else:
        if not url:
            raise EndpointDiscoveryError(
                "no reasoner endpoint to probe: SEARCH_REASONER_URL is empty"
            )
        model_id, max_model_len, server_declared_window = _facts_from_endpoint(
            url, override
        )
        for name, value in services.items():
            if value:
                _probe_service_health(name, value)

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
        sam_url=services.get("sam", ""),
        sam_video_url=services.get("sam_video", ""),
        server_declared_window=server_declared_window,
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

    Reads ``SEARCH_REASONER_URL`` + the service URL keys, honors the explicit
    ``SEARCH_ENDPOINT_OFFLINE`` / ``SEARCH_ENDPOINT_SPEC_OVERRIDE`` escape
    hatches."""

    override_raw = os.environ.get("SEARCH_ENDPOINT_SPEC_OVERRIDE", "").strip()
    return discover(
        os.environ.get("SEARCH_REASONER_URL", "http://127.0.0.1:8001/v1"),
        {
            "sam": os.environ.get("SAM_SERVER_URL", ""),
            "sam_video": os.environ.get("SAM_VIDEO_SERVER_URL", ""),
        },
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
