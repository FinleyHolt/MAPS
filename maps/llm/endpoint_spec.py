"""Server-discovered endpoint facts, probed once at process start.

:func:`discover` reads the served model and its window off
``GET {url}/v1/models`` and checks each SAM replica's ``/health``; any
divergence raises :class:`EndpointDiscoveryError` naming the URL and what was
seen. An entry point runs ``install_spec(discover_from_env())``; the reasoner
gets the spec injected, and other readers call :func:`spec`, which raises when
nothing was installed.

Explicit pins:

  ``SEARCH_ENDPOINT_OFFLINE=1``
      Offline / replay: probe nothing; requires the override below.
  ``SEARCH_ENDPOINT_SPEC_OVERRIDE=<model_id>:<max_model_len>``
      An endpoint serving several models or stating no window. A reachable
      endpoint must still serve the pinned id.
  ``SEARCH_REASONER_MODELS_URL=<url>``
      Where the served ids are listed when not at ``{SEARCH_REASONER_URL}/models``
      (Bedrock's bedrock-mantle lists them under ``/v1/models``).
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


_PROBE_TIMEOUT_S = 15.0
# A transport failure is asked again before it is a verdict: a relayed link
# that re-forms drops one connection.
_PROBE_TRIES = 3
_PROBE_RETRY_S = 3.0

# Output reserve cap: tokens held back for the current turn's thinking +
# completion, ``min(cap, window // 4)``.
_OUTPUT_RESERVE_CAP = 40000

# Slack between the compaction budget and the window, for client-vs-server
# token-count skew.
_WINDOW_MARGIN_TOKENS = 1024

# The reasoner cannot operate below this input budget.
_MIN_INPUT_BUDGET = 8000


class EndpointDiscoveryError(RuntimeError):
    """An endpoint fact could not be discovered, or diverged from the pin."""


@dataclass(frozen=True)
class EndpointOverride:
    """``SEARCH_ENDPOINT_SPEC_OVERRIDE=<model_id>:<max_model_len>``, parsed."""

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
    """The discovered endpoint facts, frozen for the process. ``base_url`` is
    the one OpenAI-style base probed; ``sam_url`` the SAM replicas'
    comma-list, verbatim."""

    base_url: str
    model_id: str
    max_model_len: int
    tokenizer_name: str
    output_reserve: int
    sam_url: str = ""

    @property
    def input_budget(self) -> int:
        """Compaction ceiling: a prompt under it plus ``max_output_tokens``
        fits the served window."""
        return (self.max_model_len - self.max_output_tokens
                - _WINDOW_MARGIN_TOKENS)

    @property
    def max_output_tokens(self) -> int:
        """The request's output cap: the output reserve."""
        return self.output_reserve


def _output_reserve_for(max_model_len: int) -> int:
    return min(_OUTPUT_RESERVE_CAP, max_model_len // 4)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def _get_json(url: str, *, bearer: str | None = None) -> object:
    """GET ``url`` (``bearer`` as an Authorization header) and parse the JSON
    body; raise EndpointDiscoveryError on transport failure, non-2xx, or a
    non-JSON body."""
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
    """Where the reasoner at ``base_url`` lists its served models, for
    discovery and every later probe alike."""
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
    """Probe every SAM replica's ``/health``; a payload whose ``service``
    field does not name sam (a foreign server on the port) raises."""
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
#: compactor sizes them; discovery renders it once.
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
    """Load the tokenizer for ``model_id`` and render its chat template over
    :data:`_TEMPLATE_PROBE`; either failing raises naming the repo. The
    import is local so spec readers never load transformers."""
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

    ``base_url`` is one URL (a comma-list is refused: the engine balances its
    own replicas); ``sam_url`` the replicas to health-check (empty skips);
    ``offline=True`` probes nothing and requires ``override``.
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
    if spec.input_budget < _MIN_INPUT_BUDGET:
        raise EndpointDiscoveryError(
            f"served window max_model_len={max_model_len} leaves "
            f"input_budget={spec.input_budget} after max_output_tokens="
            f"{spec.max_output_tokens} + margin={_WINDOW_MARGIN_TOKENS} — "
            f"below the {_MIN_INPUT_BUDGET}-token reasoner minimum"
        )
    return spec


def discover_from_env() -> EndpointSpec:
    """:func:`discover` over ``SEARCH_REASONER_URL``, the SAM replicas and
    the ``SEARCH_ENDPOINT_OFFLINE`` / ``SEARCH_ENDPOINT_SPEC_OVERRIDE`` pins."""

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
    """Install the process-wide spec; a second install must be identical."""
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
    """The installed spec; raises when nothing was installed."""
    if _SPEC is None:
        raise EndpointDiscoveryError(
            "no EndpointSpec installed: the process entry point must run "
            "install_spec(discover_from_env()) before any consumer reads "
            "spec()"
        )
    return _SPEC


def spec_installed() -> bool:
    """Whether a spec was installed (an LLM-free run records no model)."""
    return _SPEC is not None
