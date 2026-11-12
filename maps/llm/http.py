"""Raw HTTP client for OpenAI-compatible chat-completions, and the flight
transcript every call is mirrored into.

The request body is exactly the caller's kwargs (no SDK). A 5xx or transport
error burns ``_MAX_RETRIES``; a 429/402 waits on its own budget, up to
``_RATE_LIMIT_BUDGET_S`` of waiting, honouring ``Retry-After`` (a hosted
endpoint bills no refused request, so a saturated limit costs wall clock, not
the episode); any other 4xx raises at once, as does a read, write or pool
timeout (the mission loop's backoff decides whether to send the turn again). A
call with a ``cancel_event`` streams (SSE), and the event shuts its socket
down and raises :class:`CallCanceled`; an answer that lands as the event fires
is never handed back or recorded.

``<workdir>/flight/llm/transcript.jsonl`` stores the conversation once, under
the mission's own directory (a caller's ``workdir``): per-call message
deltas, params written once per distinct body, image parts as sha refs.

With an image store named (``Endpoint.image_store_url``, the run's
``MAPS_IMAGE_STORE_URL``: a store standing one fast hop from
the reasoner, for a slow link), each inline image crosses the wire once: its
bytes are PUT to ``<store>/looks/<sha>`` and the request names it
``sha256:<sha>``, which the store expands back to the same data URL beside
the reasoner. The model reads the same bytes; the transcript keeps the body
as built. A 409 naming shas the store does not hold sends them again; a
PUT that cannot connect takes the call's transport retries, and a store
fault that outlasts them, or any other, raises :class:`ImageStoreError`.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import json
import logging
import socket
import threading
import time
import uuid
from pathlib import Path

import httpx

from maps.settings import Endpoint

from . import endpoint_spec, openai_wire
from .endpoint_spec import RATE_LIMIT_STATUS
from .model_cards import speaks_responses

log = logging.getLogger(__name__)


# Timeouts this layer does not re-send (the server may still hold the request);
# the mission loop's backoff owns that retry. A connect timeout is retried here:
# nothing was accepted.
_NO_RETRY_TIMEOUTS = (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)

# Ceiling on one rate-limit wait, however long the advertised ``Retry-After``.
_RATE_LIMIT_MAX_BACKOFF_S = 60.0

# The fault retries, the rate-limit wait per call (s), the backoff base (s).
_MAX_RETRIES = 2
_RATE_LIMIT_BUDGET_S = 1800.0
_BACKOFF_S = 1.5


def _retry_after_delay(headers: httpx.Headers, default_s: float) -> float:
    """Seconds to wait before re-sending: Azure's ``retry-after-ms``, else
    ``Retry-After`` in its delta-seconds form, else ``default_s``, held to
    1 s .. ``_RATE_LIMIT_MAX_BACKOFF_S`` (the floor: an endpoint answering
    "retry now" forever still spends the budget). The HTTP-date form, rare
    on JSON APIs, takes the default."""
    try:
        delay = float(headers["retry-after-ms"]) / 1000.0
    except (KeyError, ValueError):
        try:
            delay = float(headers.get("retry-after", ""))
        except ValueError:
            delay = default_s
    return max(1.0, min(_RATE_LIMIT_MAX_BACKOFF_S, delay))


def _backoff(delay: float, cancel_event: threading.Event | None) -> None:
    """Wait out one retry backoff; a cancel ends the wait, and the loop's
    pre-dispatch check then raises."""
    if cancel_event is None:
        time.sleep(delay)
    else:
        cancel_event.wait(delay)


class CallCanceled(Exception):
    """A request aborted via its ``cancel_event``; never retried."""


#: How often an in-flight call's watch looks at its cancel event.
_CANCEL_POLL_S = 0.25

#: The connect phase's own bound: a cancel cannot reach a socket still
#: connecting, and a SYN nothing answers otherwise waits out the kernel's
#: retries (~130 s).
_CONNECT_TIMEOUT_S = 15.0


class _CancelWatch:
    """Aborts one in-flight request the moment its cancel event is set: it
    takes the socket off the transport's ``trace`` extension as it connects
    and shuts it down on cancel, which wakes a blocked read (closing the
    client from another thread does not) and frees the serve's slot."""

    def __init__(self, cancel_event: threading.Event) -> None:
        self._cancel = cancel_event
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._sockets: list[socket.socket] = []
        self._aborted = False
        threading.Thread(target=self._watch, name="llm-cancel-watch", daemon=True).start()

    def trace(self, name: str, info: dict) -> None:
        if name not in ("connection.connect_tcp.complete", "connection.start_tls.complete"):
            return
        stream = info.get("return_value")
        sock = stream.get_extra_info("socket") if stream is not None else None
        if sock is None:
            return
        with self._lock:
            self._sockets.append(sock)
            if self._aborted:
                _shutdown(sock)

    def _watch(self) -> None:
        while not self._done.is_set():
            if self._cancel.wait(_CANCEL_POLL_S):
                with self._lock:
                    self._aborted = True
                    for sock in self._sockets:
                        _shutdown(sock)
                return

    def close(self) -> None:
        self._done.set()


def _shutdown(sock: socket.socket) -> None:
    with contextlib.suppress(OSError):  # already closed
        sock.shutdown(socket.SHUT_RDWR)


# ---------------------------------------------------------------------------
# Flight log
# ---------------------------------------------------------------------------

# Said once per process, so the warning is not buried under one line per turn.
_WARNED_NO_WORKDIR = False


def _resolved_flight_log_dir(workdir: str | Path | None) -> Path | None:
    """``<workdir>/flight/llm`` for the mission's directory, or None without
    one (a unit test), warned once so an unrecorded mission cannot pass silently."""
    workdir = str(workdir).strip() if workdir is not None else ""
    if not workdir:
        global _WARNED_NO_WORKDIR
        if not _WARNED_NO_WORKDIR:
            _WARNED_NO_WORKDIR = True
            log.warning(
                "no mission workdir: this process is making LLM "
                "calls that NOTHING is recording. Expected for a unit test; "
                "a flying mission names one and gets flight/llm/transcript.jsonl."
            )
        return None
    return Path(workdir).expanduser() / "flight" / "llm"


# ponytail: one process appends to a transcript, so a module lock serialises
# its lines; a multi-process runner needs a file per process and a merge.
_FLIGHT_LOCK = threading.Lock()
# flight_dir -> the previous request's scrubbed-message digests, for
# write-time delta encoding. Keyed by dir so sequential episodes in one
# process cannot cross-contaminate.
_FLIGHT_THREAD_DIGESTS: dict[str, list[str]] = {}
# flight_dir -> params shas already written to that transcript.
_FLIGHT_PARAMS_SEEN: dict[str, set[str]] = {}


def _map_image_parts(messages: list, fn) -> list:
    """``messages`` with each data-URL image part replaced by
    ``fn(part, media_type, raw)``, ``raw`` its decoded bytes (None when its
    base64 does not decode). The input is not mutated."""
    out: list = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            out.append(msg)
            continue
        parts: list = []
        changed = False
        for part in content:
            url = (
                part.get("image_url", {}).get("url", "")
                if isinstance(part, dict) and part.get("type") == "image_url"
                else ""
            )
            if not (isinstance(url, str) and url.startswith("data:")):
                parts.append(part)
                continue
            head, _, b64 = url.partition(",")
            try:
                raw = base64.b64decode(b64)
            except (ValueError, binascii.Error):
                raw = None
            parts.append(fn(part, head[len("data:") :].split(";")[0] or "image/jpeg", raw))
            changed = True
        out.append({**msg, "content": parts} if changed else msg)
    return out


def _image_ref(part: dict, media_type: str, raw: bytes | None) -> dict:
    """The transcript's ref for one image part: the sha of its decoded bytes,
    or of its base64 text when that does not decode."""
    if raw is None:
        raw = part["image_url"]["url"].partition(",")[2].encode("utf-8")
    return {
        "type": "image_ref",
        "media_type": media_type,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


# shas the store holds; a 409 from it drops the ones it names.
_IMAGE_STORE_HELD: set[str] = set()


class ImageStoreError(ValueError):
    """The image store refused or dropped a PUT. Never an LLM status: a PUT
    that could not connect takes the call's transport retries, and no other
    store fault meets its retries or status handling. A ``ValueError``, so
    the mission loop retries the turn (``maps.mission.serve_refusals._RETRIABLE_LLM_ERRORS``)."""


def _put_image(url: str, raw: bytes, media_type: str, cancel_event: threading.Event | None) -> None:
    """PUT one image's bytes to the store, its Content-Type the one header; the
    cancel event aborts it as it does the chat call."""
    watch = _CancelWatch(cancel_event) if cancel_event is not None else None
    try:
        with httpx.Client(timeout=httpx.Timeout(180.0, connect=_CONNECT_TIMEOUT_S)) as client:
            client.put(
                url,
                content=raw,
                headers={"Content-Type": media_type},
                extensions={"trace": watch.trace} if watch is not None else None,
            ).raise_for_status()
    except httpx.HTTPError as exc:
        if cancel_event is not None and cancel_event.is_set():
            raise CallCanceled("canceled during an image-store PUT") from None
        raise ImageStoreError(f"image store PUT {url} failed: {exc}") from exc
    finally:
        if watch is not None:
            watch.close()


def _wire_body(body: dict, cancel_event: threading.Event | None, image_store_url: str) -> dict:
    """``body`` as it crosses the wire: with an image store set
    (``image_store_url``, the run's ``MAPS_IMAGE_STORE_URL``), every
    data-URL image part rides as ``sha256:<sha>``, its bytes PUT once; a
    part whose base64 does not decode rides inline."""
    base = image_store_url
    if not base or not isinstance(body.get("messages"), list):
        return body

    def by_sha(part: dict, media_type: str, raw: bytes | None) -> dict:
        if raw is None:
            return part
        sha = hashlib.sha256(raw).hexdigest()
        if sha not in _IMAGE_STORE_HELD:
            _put_image(f"{base}/looks/{sha}", raw, media_type, cancel_event)
            _IMAGE_STORE_HELD.add(sha)
        return {**part, "image_url": {**part["image_url"], "url": f"sha256:{sha}"}}

    return {**body, "messages": _map_image_parts(body["messages"], by_sha)}


def _forget_missing(resp: httpx.Response) -> bool:
    """A 409 from the image store names the shas it does not hold (an
    eviction, a restart): forget them, so the next send PUTs them."""
    try:
        missing = resp.json()["missing"]
    except Exception:
        return False
    _IMAGE_STORE_HELD.difference_update(missing)
    return True


#: The reasoner's record of a run: every request, response and its params, one a line.
TRANSCRIPT_JSONL = "transcript.jsonl"


def _append_flight_record(directory: Path, record: dict) -> None:
    """Append one record line to ``<directory>/transcript.jsonl``, flushed. A
    filesystem error is logged and swallowed: recording never fails a call."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / TRANSCRIPT_JSONL).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()
    except OSError as exc:
        log.warning("flight log write failed at %s: %s", directory, exc)


def _flight_request_record(
    directory: Path,
    request_id: str,
    body: dict,
) -> None:
    """Append the params (if new) and request records for one dispatch, as it
    fires, before the stream flags join the body. The full request is
    ``previous[:n_prefix] + new_messages``; ``params`` (the body minus
    messages) is written once per distinct sha."""
    with _FLIGHT_LOCK:
        try:
            params = {k: v for k, v in body.items() if k != "messages"}
            params_sha = hashlib.sha256(
                json.dumps(params, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:16]
            seen = _FLIGHT_PARAMS_SEEN.setdefault(str(directory), set())
            if params_sha not in seen:
                _append_flight_record(
                    directory,
                    {"kind": "params", "sha": params_sha, "params": params},
                )
                seen.add(params_sha)

            scrubbed = _map_image_parts(body.get("messages") or [], _image_ref)
            digests = [
                hashlib.sha256(
                    json.dumps(m, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()[:16]
                for m in scrubbed
            ]
            prev = _FLIGHT_THREAD_DIGESTS.get(str(directory)) or []
            n_prefix = 0
            for a, b in zip(prev, digests):
                if a != b:
                    break
                n_prefix += 1
            _FLIGHT_THREAD_DIGESTS[str(directory)] = digests
            _append_flight_record(
                directory,
                {
                    "kind": "request",
                    "request_id": request_id,
                    "created_at": time.time(),
                    "params_sha": params_sha,
                    "n_prefix": n_prefix,
                    "n_messages": len(scrubbed),
                    "new_messages": scrubbed[n_prefix:],
                },
            )
        except Exception as exc:
            log.warning("flight request record failed at %s: %s", directory, exc)


def _flight_response_record(
    directory: Path,
    request_id: str,
    response_json: dict,
) -> None:
    """Append the response record that closes one request_id."""
    with _FLIGHT_LOCK:
        _append_flight_record(
            directory,
            {
                "kind": "response",
                "request_id": request_id,
                "ts": time.time(),
                "response": response_json,
            },
        )


def _request_headers(api_key: str | None, routing_key: str | None = None) -> dict[str, str]:
    """Headers for one POST. ``routing_key`` (the mission's: the runner names
    one per episode) rides as an affinity header so a key-affine router keeps
    an episode on one worker's prefix cache."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if routing_key:
        headers["x-smg-routing-key"] = routing_key
    return headers


# A stream of nothing but empty deltas is a hung generation: it ends after
# _EMPTY_STREAM_S or _EMPTY_STREAM_CHUNKS, whichever first, in a ReadTimeout
# whose text starts with _EMPTY_STREAM_ERROR.
_EMPTY_STREAM_S = 90.0
_EMPTY_STREAM_CHUNKS = 1200
_EMPTY_STREAM_ERROR = "LLM stream emitted only empty deltas"


def stream_chat_completions(
    url: str,
    headers: dict[str, str],
    body: dict,
    *,
    timeout: float,
    cancel_event: threading.Event | None = None,
) -> dict:
    """POST with ``stream: true`` and reassemble the SSE deltas into the
    non-streaming response shape. Streaming exists for cancellation. A stream
    that ends with neither a finish_reason nor ``[DONE]`` was cut short and
    raises ``httpx.RemoteProtocolError``, a transport fault."""
    aggregated_text: list[str] = []
    aggregated_reasoning: list[str] = []
    aggregated_tool_calls: dict[int, dict] = {}
    finish_reason: str | None = None
    usage: dict | None = None
    model: str | None = None
    role: str | None = None
    saw_done = False
    empty_chunk_count = 0
    first_empty_at: float | None = None

    watch = _CancelWatch(cancel_event) if cancel_event is not None else None
    try:
        with (
            httpx.Client(timeout=httpx.Timeout(timeout, connect=_CONNECT_TIMEOUT_S)) as client,
            client.stream(
                "POST",
                url,
                headers=headers,
                json=body,
                extensions={"trace": watch.trace} if watch is not None else None,
            ) as resp,
        ):
            if resp.status_code >= 400:
                resp.read()  # the body is legible after the stream closes
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                # a chunk that arrived just before the cancel fired
                if cancel_event is not None and cancel_event.is_set():
                    raise CallCanceled("canceled mid-stream")
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                payload = raw_line[len("data:") :].strip()
                if payload == "[DONE]":
                    saw_done = True
                    continue
                if not payload:
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    if event.get("usage"):
                        usage = event["usage"]
                    if event.get("model"):
                        model = event["model"]
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                if "role" in delta and role is None:
                    role = delta["role"]
                content_delta = delta.get("content")
                if content_delta:
                    aggregated_text.append(content_delta)
                # The serve names the span ``reasoning`` or ``reasoning_content``.
                reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning_delta:
                    aggregated_reasoning.append(reasoning_delta)
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = aggregated_tool_calls.setdefault(
                        idx,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
                fin = choice.get("finish_reason")
                if fin is not None:
                    finish_reason = fin
                if (
                    content_delta
                    or reasoning_delta
                    or delta.get("tool_calls")
                    or fin is not None
                    or delta.get("role")
                ):
                    empty_chunk_count = 0
                    first_empty_at = None
                    continue
                empty_chunk_count += 1
                first_empty_at = first_empty_at or time.monotonic()
                if (
                    time.monotonic() - first_empty_at >= _EMPTY_STREAM_S
                    or empty_chunk_count >= _EMPTY_STREAM_CHUNKS
                ):
                    raise httpx.ReadTimeout(f"{_EMPTY_STREAM_ERROR} for {empty_chunk_count} chunks")
            if finish_reason is None and not saw_done:
                raise httpx.RemoteProtocolError("LLM stream ended without a finish_reason")
    except httpx.HTTPError:
        if cancel_event is not None and cancel_event.is_set():
            raise CallCanceled("canceled in flight") from None
        raise
    finally:
        if watch is not None:
            watch.close()

    message: dict = {"role": role or "assistant", "content": "".join(aggregated_text)}
    if aggregated_reasoning:
        message["reasoning"] = "".join(aggregated_reasoning)
    if aggregated_tool_calls:
        message["tool_calls"] = [aggregated_tool_calls[k] for k in sorted(aggregated_tool_calls)]
    response_json: dict = {
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason or "stop"}],
    }
    if usage is not None:
        response_json["usage"] = usage
    if model is not None:
        response_json["model"] = model
    return response_json


#: The stream events that close a Responses stream with its whole response.
_RESPONSES_TERMINAL = ("response.completed", "response.incomplete")


def stream_responses(
    url: str,
    headers: dict[str, str],
    body: dict,
    *,
    timeout: float,
    cancel_event: threading.Event | None = None,
) -> dict:
    """POST a Responses body with ``stream: true`` and return the closing
    event's response in the chat shape. Streaming exists for cancellation; a
    stream that fails or ends with no closing event is a transport fault."""
    final: dict | None = None
    watch = _CancelWatch(cancel_event) if cancel_event is not None else None
    try:
        with (
            httpx.Client(timeout=httpx.Timeout(timeout, connect=_CONNECT_TIMEOUT_S)) as client,
            client.stream(
                "POST",
                url,
                headers=headers,
                json=body,
                extensions={"trace": watch.trace} if watch is not None else None,
            ) as resp,
        ):
            if resp.status_code >= 400:
                resp.read()  # the body is legible after the stream closes
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if cancel_event is not None and cancel_event.is_set():
                    raise CallCanceled("canceled mid-stream")
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                try:
                    event = json.loads(raw_line[len("data:") :].strip())
                except json.JSONDecodeError:
                    continue
                kind = event.get("type")
                if kind in _RESPONSES_TERMINAL:
                    final = event.get("response") or {}
                elif kind in ("response.failed", "error"):
                    raise httpx.RemoteProtocolError(
                        f"Responses stream {kind}: {json.dumps(event)[:300]}"
                    )
            if final is None:
                raise httpx.RemoteProtocolError("Responses stream ended without a closing event")
    except httpx.HTTPError:
        if cancel_event is not None and cancel_event.is_set():
            raise CallCanceled("canceled in flight") from None
        raise
    finally:
        if watch is not None:
            watch.close()
    return openai_wire.from_responses(final)


def _raise_on_served_model_mismatch(
    base_url: str, requested: str | None, refusal: str, endpoint: Endpoint
) -> None:
    """On a 404, re-probe the served ids and raise when the request's model is
    no longer among them (the server behind the URL changed since discovery).
    Returns when the probe fails or the model is still served."""
    if not requested:
        return
    try:
        served = [str(m.get("id")) for m in endpoint_spec.probe_models(base_url, endpoint)]
    except endpoint_spec.EndpointDiscoveryError:
        return
    if requested not in served:
        raise RuntimeError(
            f"served-model mismatch: this process discovered {requested!r} "
            f"at startup but {base_url} now serves {served} "
            f"(chat completion returned 404: {refusal[:200]!r}). The server "
            f"behind the URL changed mid-run; drain and restart."
        )


#: The sampling keys a refused request's summary names (a serve whose template
#: carries no effort field answers 400 on one).
_SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "max_tokens",
    "max_completion_tokens",
    "reasoning_effort",
)


def chat_completions_raw(
    base_url: str,
    api_key: str | None,
    *,
    timeout: float,
    cancel_event: threading.Event | None = None,
    workdir: str | Path | None = None,
    routing_key: str | None = None,
    endpoint: Endpoint = Endpoint(),
    **kwargs,
) -> dict:
    """Send one chat completion and return the response JSON, retrying as
    the module docstring says. ``kwargs`` is the request body. ``workdir`` is
    the mission's directory, whose ``flight/llm/transcript.jsonl`` records the
    call; ``routing_key`` its affinity key; ``endpoint`` the run's settings
    (the image store, and the models URL a 404 is checked against)."""
    body = kwargs
    flight_dir = _resolved_flight_log_dir(workdir)
    # A Responses card takes the same body translated at the wire, and its
    # answer comes back in the chat shape (maps.llm.openai_wire).
    responses = speaks_responses(body.get("model"))
    url = base_url.rstrip("/") + (openai_wire.RESPONSES_PATH if responses else "/chat/completions")
    headers = _request_headers(api_key, routing_key)

    request_id = uuid.uuid4().hex
    if flight_dir is not None:
        _flight_request_record(flight_dir, request_id, body)

    use_stream = cancel_event is not None
    if use_stream:
        # without it a stream carries no token counts
        body["stream_options"] = {"include_usage": True}
        body["stream"] = True

    attempt = 0
    rate_limit_attempt = 0
    rate_limit_waited = 0.0
    wire = body
    while True:
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise CallCanceled("canceled before dispatch")
            wire = _wire_body(body, cancel_event, endpoint.image_store_url)
            if responses:
                wire = openai_wire.to_responses(wire)
            if use_stream:
                response_json = (stream_responses if responses else stream_chat_completions)(
                    url, headers, wire, timeout=timeout, cancel_event=cancel_event
                )
            else:
                with httpx.Client(
                    timeout=httpx.Timeout(timeout, connect=_CONNECT_TIMEOUT_S)
                ) as client:
                    r = client.post(url, headers=headers, json=wire)
                r.raise_for_status()
                response_json = r.json()
                if responses:
                    response_json = openai_wire.from_responses(response_json)
            if cancel_event is not None and cancel_event.is_set():
                # The answer landed as the caller gave up on it: never used.
                raise CallCanceled("canceled as the answer landed")
            if flight_dir is not None:
                _flight_response_record(flight_dir, request_id, response_json)
            return response_json
        except CallCanceled:
            raise
        except _NO_RETRY_TIMEOUTS as e:
            # Raised to the caller, whose backoff decides on a re-send.
            if str(e).startswith(_EMPTY_STREAM_ERROR):
                log.error("%s; not re-sent here.", e)
                raise
            log.error(
                "LLM %s after %.0fs; not re-sent here (the server may still "
                "hold this request). Raise SEARCH_REASONER_TIMEOUT_S if this "
                "is queueing, not a hang.",
                type(e).__name__,
                timeout,
            )
            raise
        except (httpx.RequestError, httpx.HTTPStatusError, ImageStoreError) as e:
            if isinstance(e, ImageStoreError) and not isinstance(
                e.__cause__, (httpx.ConnectError, httpx.ConnectTimeout)
            ):
                raise
            status = e.response.status_code if isinstance(e, httpx.HTTPStatusError) else None
            if status in RATE_LIMIT_STATUS:
                delay = _retry_after_delay(
                    e.response.headers, _BACKOFF_S * (2 ** min(rate_limit_attempt, 6))
                )
                if rate_limit_waited + delay > _RATE_LIMIT_BUDGET_S:
                    log.error(
                        "LLM endpoint %d after %d rate-limit retries (%.0fs "
                        "waited); giving up on this call",
                        status,
                        rate_limit_attempt,
                        rate_limit_waited,
                    )
                    raise
                rate_limit_attempt += 1
                rate_limit_waited += delay
                log.warning(
                    "LLM endpoint %d (rate limit/quota) on rate-limit "
                    "retry %d (%.0f/%.0fs waited); waiting %.1fs",
                    status,
                    rate_limit_attempt,
                    rate_limit_waited,
                    _RATE_LIMIT_BUDGET_S,
                    delay,
                )
                _backoff(delay, cancel_event)
                continue
            if (
                status == 409
                and wire is not body
                and attempt < _MAX_RETRIES
                and _forget_missing(e.response)
            ):
                attempt += 1
                continue
            if status == 404:
                _raise_on_served_model_mismatch(
                    base_url, body.get("model"), e.response.text, endpoint
                )
            if status is not None and status < 500:
                msgs = body.get("messages") or []
                log.error("LLM endpoint %d response body: %s", status, e.response.text[:4000])
                log.error(
                    "LLM endpoint %d request summary: body_keys=%s, "
                    "n_tools=%d, sampling=%r, model=%r, n_messages=%d, roles=%s",
                    status,
                    sorted(body),
                    len(body.get("tools") or []),
                    {k: body[k] for k in _SAMPLING_KEYS if k in body},
                    body.get("model"),
                    len(msgs),
                    [m.get("role") for m in msgs],
                )
                raise
            if attempt >= _MAX_RETRIES:
                raise
            log.warning(
                "LLM transport error on attempt %d/%d: %s; retrying in %.1fs",
                attempt + 1,
                _MAX_RETRIES + 1,
                e,
                _BACKOFF_S * (2**attempt),
            )
            _backoff(_BACKOFF_S * (2**attempt), cancel_event)
            attempt += 1
