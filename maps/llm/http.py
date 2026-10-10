"""Raw HTTP client for OpenAI-compatible chat-completions.

Every stage of the parse-first compiler goes through this client rather
than the OpenAI SDK, so the request body is exactly what
`maps.llm.sampling` builds — a flat set of top-level keys that both a
self-hosted vLLM and a hosted endpoint honour. No LLM logic here; just
the transport.

Retries are split by cause. A 5xx or a transport error is a transient
fault and burns the `max_retries` budget. A 429/402 is a capacity or
quota event with its own, more patient budget (`max_rate_limit_retries`,
honouring `Retry-After`) — hosted endpoints rate-limit routinely, and
spending the transient budget on that would end an episode mid-compile.
Every other 4xx is a malformed-request bug and is surfaced immediately.

A content-addressed request/response cache (record + replay modes) lets
benchmarks and CI gate a full episode without a live vLLM. Configure
via `set_cache_mode(...)` (the runner's `--record-llm` / `--replay-llm`
flags). In replay mode a cache
miss raises `CacheMissError` immediately; there is no silent fall-back
to a network call.

A call that carries a ``cancel_event`` negotiates an SSE stream from
vLLM rather than a blocking POST. When the event fires the call's TCP
socket is shut down, so a read blocked on a silent serve returns at once
and the serve frees the KV-cache slot, and the call raises
:class:`CallCanceled`. An answer that lands as the event fires is never
handed back, metered or recorded.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import logging
import os
import socket
import threading
import time
import uuid
from pathlib import Path

import httpx

from .net_meter import (
    NetMeter,
    cached_tokens_of,
    completion_tokens_of,
    prompt_tokens_of,
    reasoning_tokens_of,
    request_kind,
    serialized_size,
)


# 4xx codes that mean "not now" rather than "not ever": 429 is the
# rate limit, 402 the credit/quota check. Both clear on their own, so
# they get a re-send instead of the immediate raise every other 4xx
# earns. A self-hosted vLLM emits neither; a hosted endpoint emits 429
# routinely under the compile's parallel fan-out.
_RATE_LIMIT_STATUS = frozenset({402, 429})

# Timeouts that are NOT retried: the server already has the request and is
# still working on it, so a re-send duplicates the work instead of replacing
# it. ``ConnectTimeout`` is deliberately absent -- nothing was accepted there,
# so retrying costs the server nothing.
_NO_RETRY_TIMEOUTS = (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)

# Ceiling on one rate-limit wait. A provider may advertise a long
# ``Retry-After``; past this the run is better off failing loud than
# silently parking a reasoner worker.
_RATE_LIMIT_MAX_BACKOFF_S = 60.0


def _retry_after_delay(headers: object, default_s: float) -> float:
    """Seconds to wait before re-sending, honouring ``Retry-After``.

    Only the delta-seconds form is read; the HTTP-date form is rare on
    JSON APIs and falls through to the caller's exponential backoff.
    """
    raw = None
    if headers is not None:
        try:
            raw = headers.get("retry-after")  # type: ignore[union-attr]
        except (AttributeError, TypeError):
            raw = None
    if raw:
        try:
            return max(0.0, min(_RATE_LIMIT_MAX_BACKOFF_S, float(str(raw).strip())))
        except (TypeError, ValueError):
            pass
    return min(_RATE_LIMIT_MAX_BACKOFF_S, default_s)


class CallCanceled(Exception):
    """Raised when a request is aborted via its ``cancel_event``.

    Distinct from generic timeouts / connection errors: the caller
    asked for the call to be torn down, and the worker should not
    retry the request.
    """


class DegenerateCompletion(ValueError):
    """The serve returned a completion carrying no content and no tool
    call because generation collapsed, not because the model had nothing
    to say.

    A SERVE fault, and the distinction is worth a class because the two
    look identical on the wire. A model's own empty turn is a dropped
    message the loop repairs and may end the mission on. A collapse is
    the sampler reading a broken logit vector: one sm_103 serve stack
    returned `!` (token id 0) x 40,000 on 21% of xhigh requests while the
    same checkpoint on L40S returned 0/320, and the longest LEGITIMATE
    completion either serve produced was ~7.3k tokens. Retrying is right
    (a fresh generation usually sticks); repairing is not, because the
    model was never asked anything.

    ``ValueError`` so it lands in ``reasoner_loop._RETRIABLE_LLM_ERRORS``
    without widening that tuple, alongside the transport-level decode
    failure it is the generation-side twin of.
    """


#: How often an in-flight call's watch looks at its cancel event.
_CANCEL_POLL_S = 0.25


class _CancelWatch:
    """Aborts one in-flight request the moment its cancel event is set.

    The request runs on a blocking socket, and closing a client from another
    thread does not wake a blocked read. The watch takes the socket off the
    transport's ``trace`` extension as it connects and shuts it down on
    cancel: the read returns at once and the serve sees the connection drop
    and frees the slot.
    """

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
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass  # already closed

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request/response cache
# ---------------------------------------------------------------------------


class CacheMissError(RuntimeError):
    """Raised when replay mode cannot find a cached response for a request."""


_CACHE_MODE: str = "off"  # "off" | "record" | "replay"
_CACHE_DIR: Path | None = None


def set_cache_mode(mode: str, directory: str | Path | None = None) -> None:
    """Switch the cache between off, record, and replay.

    In ``record`` mode every real HTTP response is written to
    ``<directory>/<hash>.json`` as ``{"request": body, "response":
    raw_json}``. In ``replay`` mode the cached response is returned
    directly and a missing key raises ``CacheMissError`` — there is no
    live fall-back.
    """
    global _CACHE_MODE, _CACHE_DIR
    if mode not in ("off", "record", "replay"):
        raise ValueError(f"cache mode must be off|record|replay, got {mode!r}")
    if mode != "off" and directory is None:
        raise ValueError(f"cache mode {mode!r} requires a directory")
    _CACHE_MODE = mode
    _CACHE_DIR = Path(directory) if directory is not None else None
    if _CACHE_DIR is not None and mode == "record":
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _resolved_cache_mode() -> tuple[str, Path | None]:
    """The effective cache mode, as ``set_cache_mode`` last left it."""
    return _CACHE_MODE, _CACHE_DIR


def _cache_key(body: dict) -> str:
    """Content-address a request body to a 16-char hex digest."""
    raw = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _cache_path(directory: Path, key: str) -> Path:
    return directory / f"{key}.json"


# ---------------------------------------------------------------------------
# Dashboard flight log
# ---------------------------------------------------------------------------


def _resolved_flight_log_dir() -> Path | None:
    """Where this mission's flight log goes: ``<workdir>/flight/llm``.

    Every chat-completions round-trip is mirrored into
    ``transcript.jsonl`` there -- an append-only record stream that stores
    the conversation ONCE (per-call message deltas + params dedup + image
    parts as sha refs) instead of a full re-serialization per call, and
    captures direct-HTTP and cache-replay calls alike so a reader works
    regardless of how the response was obtained. It is what the dashboard
    replay and nearly every reader under ``benchmarks/citynav/analysis/``
    actually read.

    RECORDING IS NOT OPTIONAL, and there is no environment variable that
    turns it off. There used to be: the flight log only ran when
    ``SEARCH_LLM_FLIGHT_LOG`` named a directory, five callers set it, and
    the sixth -- the FTE station -- did not. Twenty missions flew and left
    no reasoning and no tool output behind, with nothing to recover them
    from, because the switch defaulted to off and a mission that records
    nothing looks exactly like one that has nothing to say. A run's own
    record is not a feature a caller opts into.

    The one thing still configurable is WHERE a mission writes, and that
    has a single name already: ``SEARCH_COMPILE_WORKDIR``, which every
    caller that flies anything sets, and which all five of the deleted
    ``SEARCH_LLM_FLIGHT_LOG`` assignments were derived from verbatim.

    No workdir means no mission -- a bare transport in a unit test, not a
    flight -- so there is nothing to record and nowhere to put it. That
    case says so at warning rather than passing silently, because "no
    mission is running" and "a mission is running unrecorded" are the two
    readings of an empty transcript and only one of them is fine.
    """
    workdir = os.environ.get("SEARCH_COMPILE_WORKDIR", "").strip()
    if not workdir:
        global _WARNED_NO_WORKDIR
        if not _WARNED_NO_WORKDIR:
            _WARNED_NO_WORKDIR = True
            log.warning(
                "SEARCH_COMPILE_WORKDIR is unset: this process is making LLM "
                "calls that NOTHING is recording. Expected for a unit test; "
                "a flying mission sets it and gets flight/llm/transcript.jsonl."
            )
        return None
    return Path(workdir).expanduser() / "flight" / "llm"


# Said once per process. A per-call warning would bury the run's own log
# under one line per turn, which is how a real warning stops being read.
_WARNED_NO_WORKDIR = False


# Diagnostic-only kwargs mirrored into the flight-log transcript. They never
# enter the HTTP body or the cache key.
#
# The quadruple is DEAD FLEXIBILITY, kept only because the transcript's
# delta encoding keys its threads on it. There is one reasoner on a ReAct
# loop and no subagents of any kind, so every request a run writes carries
# the same tuple — measured: 292 of 292 requests tagged `reasoner`. Reading
# it as "which agent asked" describes an architecture this build does not
# have.
_TRANSPORT_META_KEYS = (
    "transport_kind",
    "transport_subagent_id",
    "transport_clause_id",
    "transport_agent_idx",
)

# The transcript is a SINGLE-PROCESS append stream: the reasoner and the
# cache-replay path both write from the runner process, so one module lock
# serializes lines. A second process appending to the same file would
# interleave records undetected — if a multi-process runner ever appears,
# the transcript needs a per-process file + a merge, not a bigger lock.
_FLIGHT_LOCK = threading.Lock()
# (flight_dir, transport quadruple) -> the previous request's scrubbed-message
# digests, for write-time delta encoding. Keyed by dir so sequential episodes
# in one process cannot cross-contaminate.
_FLIGHT_THREAD_DIGESTS: dict[tuple, list[str]] = {}
# flight_dir -> params shas already written to that transcript.
_FLIGHT_PARAMS_SEEN: dict[str, set[str]] = {}


def _scrub_image_parts(messages: list) -> list:
    """Message list with every data-URL image part replaced by a sha ref.

    ``{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,…"}}``
    becomes ``{"type": "image_ref", "media_type", "sha256", "bytes"}``. The
    sha is of the DECODED bytes and matches the look manifest's per-crop
    ``sha256``, so a transcript image part resolves to its crop record (and
    re-renders) by hash. Non-data URLs pass through. Never mutates the input
    — the HTTP body and the cache key see the original."""
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
            media_type = head[len("data:"):].split(";")[0] or "image/jpeg"
            try:
                raw = base64.b64decode(b64)
            except (ValueError, binascii.Error):
                raw = b64.encode("utf-8")  # undecodable: fingerprint as-is
            sha = hashlib.sha256(raw).hexdigest()
            parts.append({
                "type": "image_ref",
                "media_type": media_type,
                "sha256": sha,
                "bytes": len(raw),
            })
            changed = True
        out.append({**msg, "content": parts} if changed else msg)
    return out


def _append_flight_record(directory: Path, record: dict) -> None:
    """Append one record line to ``<directory>/transcript.jsonl``.

    Filesystem errors are logged at warning and swallowed: flight-log writes
    must never fail an LLM call. The flush makes each line visible to the
    dashboard's per-request reader immediately."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "transcript.jsonl").open(
            "a", encoding="utf-8"
        ) as fh:
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()
    except OSError as exc:
        log.warning("flight log write failed at %s: %s", directory, exc)


def _flight_request_record(
    directory: Path,
    request_id: str,
    body: dict,
    transport_meta: dict | None,
) -> None:
    """Append the params (if new) + request records for one dispatch.

    The flight log's "pending" pre-write: the dashboard sees the round the
    moment the call fires, before the response lands. ``new_messages`` is
    the delta against the same thread's previous call (write-time LCP over
    scrubbed-message digests); the reader reconstructs the full request as
    ``history[:n_prefix] + new_messages``, exactly. ``params`` (the body
    minus messages and transport flags — tools, response_format, sampling)
    is written once per distinct sha, which removes the other per-call
    re-serialization."""
    with _FLIGHT_LOCK:
        try:
            meta = {
                "kind": str(transport_meta.get("transport_kind") or "")
                if transport_meta else "",
                "subagent_id": (transport_meta or {}).get(
                    "transport_subagent_id"),
                "clause_id": (transport_meta or {}).get("transport_clause_id"),
                "agent_idx": (transport_meta or {}).get("transport_agent_idx"),
            }
            params = {
                k: v for k, v in body.items()
                if k not in ("messages", "stream", "stream_options")
            }
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

            scrubbed = _scrub_image_parts(body.get("messages") or [])
            digests = [
                hashlib.sha256(
                    json.dumps(m, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()[:16]
                for m in scrubbed
            ]
            thread_key = (
                str(directory), meta["kind"], meta["subagent_id"],
                meta["clause_id"], meta["agent_idx"],
            )
            prev = _FLIGHT_THREAD_DIGESTS.get(thread_key) or []
            n_prefix = 0
            for a, b in zip(prev, digests):
                if a != b:
                    break
                n_prefix += 1
            _FLIGHT_THREAD_DIGESTS[thread_key] = digests
            _append_flight_record(directory, {
                "kind": "request",
                "request_id": request_id,
                "created_at": time.time(),
                "transport": {k: v for k, v in meta.items() if v is not None},
                "params_sha": params_sha,
                "n_prefix": n_prefix,
                "n_messages": len(scrubbed),
                "new_messages": scrubbed[n_prefix:],
            })
        except Exception as exc:  # noqa: BLE001 — never fail the LLM call
            log.warning("flight request record failed at %s: %s",
                        directory, exc)


def _flight_response_record(
    directory: Path,
    request_id: str,
    response_json: dict,
) -> None:
    """Append the response record that closes one request_id."""
    with _FLIGHT_LOCK:
        _append_flight_record(directory, {
            "kind": "response",
            "request_id": request_id,
            "ts": time.time(),
            "response": response_json,
        })


class _RawChoiceMessage:
    """Minimal stand-in for an OpenAI ChatCompletionMessage."""

    __slots__ = ("content", "tool_calls", "reasoning")

    def __init__(self, d: dict):
        self.content = d.get("content")
        # vLLM v0.21.0 renamed the wire field from ``reasoning_content`` ->
        # ``reasoning`` (PR #27752, RFC #27755). Pre-0.21 deployments and
        # some non-vLLM backends still emit ``reasoning_content``, so read
        # both with ``reasoning`` first. The OpenCode-style interleaved
        # round-trip (sending prior thinking back on subsequent requests)
        # sees a non-None value on the hot path either way.
        self.reasoning = d.get("reasoning") or d.get("reasoning_content")
        raw_tcs = d.get("tool_calls") or []
        self.tool_calls = [_RawToolCall(tc) for tc in raw_tcs]


class _RawToolCall:
    __slots__ = ("id", "type", "function")

    def __init__(self, d: dict):
        self.id = d.get("id")
        self.type = d.get("type", "function")
        self.function = _RawFn(d.get("function", {}))


class _RawFn:
    __slots__ = ("name", "arguments")

    def __init__(self, d: dict):
        self.name = d.get("name", "")
        self.arguments = d.get("arguments", "")


class _RawChoice:
    __slots__ = ("message", "finish_reason")

    def __init__(self, d: dict):
        self.message = _RawChoiceMessage(d.get("message", {}))
        self.finish_reason = d.get("finish_reason", "")


class _RawResponse:
    __slots__ = ("choices",)

    def __init__(self, d: dict):
        self.choices = [_RawChoice(c) for c in d.get("choices", [])]


def _request_headers(api_key: str | None) -> dict[str, str]:
    """Headers for one chat-completions POST.

    ``SEARCH_LLM_ROUTING_KEY`` (set per episode by the runner) rides as
    an opaque affinity header — a key-affine router (sglang-router
    ``--policy manual``) pins every call carrying the same key to one
    worker, so an episode's growing context stays with its radix cache.
    ``SEARCH_LLM_ROUTING_KEY_HEADER`` renames the header for a router
    with a different spelling. Unset = no header: the client carries no
    router fact and single-instance serves see today's requests. Read
    per call, not cached — the runner rewrites the key each episode.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    routing_key = os.environ.get("SEARCH_LLM_ROUTING_KEY")
    if routing_key:
        header_name = (
            os.environ.get("SEARCH_LLM_ROUTING_KEY_HEADER") or "x-smg-routing-key"
        )
        headers[header_name] = routing_key
    return headers


def _stream_chat_completions(
    url: str,
    headers: dict[str, str],
    body: dict,
    *,
    timeout: float,
    cancel_event: threading.Event | None = None,
) -> dict:
    """POST with ``stream: true``, return the aggregated response.

    Reassembles the SSE deltas into the same ``{"choices": [...],
    "usage": {...}}`` shape vLLM returns for non-streaming calls so
    every downstream consumer (cache, parser, _RawResponse) is
    unaffected. Streaming exists for CANCELLATION (the per-chunk hook
    that tears down the TCP socket), not for logging — the flight
    transcript records the request at dispatch and the response when it
    lands, and stores no per-token stream.
    """
    streamed_body = {**body, "stream": True}
    aggregated_text: list[str] = []
    aggregated_reasoning: list[str] = []
    aggregated_tool_calls: dict[int, dict] = {}
    finish_reason: str | None = None
    usage: dict | None = None
    model: str | None = None
    role: str | None = None
    # A server that streams nothing but empty deltas is a hung generation;
    # give up on whichever bound trips first.
    empty_stream_timeout_s = 90.0
    empty_stream_max_chunks = 1200
    empty_chunk_count = 0
    first_empty_at: float | None = None

    # The cancel watch shuts the TCP socket down when the event fires; a
    # blocked read (the serve silent before its headers or between chunks)
    # returns at once instead of waiting out the read timeout.
    watch = _CancelWatch(cancel_event) if cancel_event is not None else None
    try:
        with httpx.Client(timeout=timeout) as client, client.stream(
            "POST", url, headers=headers, json=streamed_body,
            extensions={"trace": watch.trace} if watch is not None else None,
        ) as resp:
            if resp.status_code >= 400:
                resp.read()  # the body is legible after the stream closes
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                # Check cancellation per SSE chunk. The watch also
                # shuts the socket down from the side; this check
                # catches a chunk that arrived just before the
                # cancel fired.
                if cancel_event is not None and cancel_event.is_set():
                    raise CallCanceled("canceled mid-stream")
                # httpx normally yields str, but tests / older
                # clients may yield bytes; normalize so the SSE
                # parser stays simple.
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8", errors="replace")
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                payload = raw_line[len("data:") :].strip()
                if not payload or payload == "[DONE]":
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
                # vLLM v0.21+ emits the reasoning span on streaming
                # deltas under ``reasoning`` (PR #27752); older
                # backends used ``reasoning_content``. Aggregate
                # whichever shows up so the reassembled message
                # carries the ``<think>`` text on the streaming
                # path too (matches the non-streaming branch).
                reasoning_delta = delta.get("reasoning") or delta.get(
                    "reasoning_content"
                )
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
                has_signal = bool(
                    content_delta
                    or reasoning_delta
                    or delta.get("tool_calls")
                    or fin is not None
                    or delta.get("role")
                )
                if has_signal:
                    empty_chunk_count = 0
                    first_empty_at = None
                else:
                    empty_chunk_count += 1
                    now = time.monotonic()
                    if first_empty_at is None:
                        first_empty_at = now
                    timed_out = (
                        empty_stream_timeout_s > 0
                        and now - first_empty_at >= empty_stream_timeout_s
                    )
                    chunk_capped = (
                        empty_stream_max_chunks > 0
                        and empty_chunk_count >= empty_stream_max_chunks
                    )
                    if timed_out or chunk_capped:
                        raise httpx.ReadTimeout(
                            "LLM stream emitted only empty deltas "
                            f"for {empty_chunk_count} chunks"
                        )
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
        message["tool_calls"] = [
            aggregated_tool_calls[k] for k in sorted(aggregated_tool_calls)
        ]
    response_json: dict = {
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason or "stop"}
        ],
    }
    if usage is not None:
        response_json["usage"] = usage
    if model is not None:
        response_json["model"] = model
    return response_json


def _raise_on_served_model_mismatch(
    base_url: str, body: dict, r, api_key: str | None = None
) -> None:
    """On a 404 chat completion, re-probe ``{base_url}/models`` and abort
    LOUDLY when the request's model is no longer among the served ids —
    the server behind the URL changed since startup discovery (probe-once
    cannot see a same-URL restart). Falls through silently (the caller
    then raises the plain HTTP error) when the probe fails or the model
    is still served (a 404 for some other reason)."""
    requested = body.get("model")
    if not requested:
        return
    served: list[str] = []
    try:
        probe = httpx.get(
            base_url.rstrip("/") + "/models",
            timeout=10.0,
            headers=(
                {"Authorization": f"Bearer {api_key}"} if api_key else None
            ),
        )
        probe.raise_for_status()
        data = probe.json().get("data") or []
        served = [str(m.get("id")) for m in data if isinstance(m, dict)]
    except (httpx.HTTPError, ValueError, OSError):
        return
    if served and requested not in served:
        raise RuntimeError(
            f"served-model mismatch: this process discovered {requested!r} "
            f"at startup but {base_url} now serves {served} "
            f"(chat completion returned 404: {r.text[:200]!r}). The server "
            f"behind the URL changed mid-run — drain and restart."
        )


def _chat_completions_raw(
    base_url: str,
    api_key: str | None,
    *,
    timeout: float = 14400.0,  # 4h: only kill a truly-hung call (was 300s); the stage deadline is the real ceiling
    max_retries: int = 2,
    max_rate_limit_retries: int = 5,
    backoff_s: float = 1.5,
    cancel_event: threading.Event | None = None,
    **kwargs,
) -> _RawResponse:
    """Send a chat completion via raw HTTP.

    Retries transient 5xx responses up to `max_retries` times with exponential
    backoff. Request errors (connection refused, timeout) also retry.
    Rate-limit and quota responses (429/402) draw on a separate
    `max_rate_limit_retries` budget so a burst of them cannot consume the
    transient-fault budget; each wait honours `Retry-After` when present.
    Every other 4xx is surfaced immediately; it indicates a malformed request
    that retrying would not fix.

    Honors the cache mode set via :func:`set_cache_mode`:
    ``replay`` returns a cached
    response without touching the network; ``record`` makes the real
    call and writes the response to the cache directory.
    """
    transport_meta = {k: kwargs.pop(k, None) for k in _TRANSPORT_META_KEYS}
    body = {k: v for k, v in kwargs.items() if v is not None}
    # Content-address the request on this canonical body so record and
    # replay agree on the cache hash regardless of transport.
    _pristine_body = copy.deepcopy(body)
    mode, directory = _resolved_cache_mode()
    flight_dir = _resolved_flight_log_dir()

    if mode == "replay":
        assert directory is not None
        key = _cache_key(_pristine_body)
        path = _cache_path(directory, key)
        if not path.exists():
            raise CacheMissError(
                f"no cached response for request hash {key} at {path}. "
                f"Re-record the fixture or run without --replay-llm."
            )
        entry = json.loads(path.read_text())
        if flight_dir is not None:
            replay_rid = uuid.uuid4().hex
            _flight_request_record(
                flight_dir, replay_rid, _pristine_body, transport_meta
            )
            _flight_response_record(flight_dir, replay_rid, entry["response"])
        return _RawResponse(entry["response"])

    eff_base_url = base_url
    eff_api_key = api_key
    url = eff_base_url.rstrip("/") + "/chat/completions"
    headers = _request_headers(eff_api_key)

    # Layer 1: append the request record so the dashboard sees a "pending"
    # round the moment the call fires, not after the response lands.
    request_id = uuid.uuid4().hex
    if flight_dir is not None:
        _flight_request_record(flight_dir, request_id, body, transport_meta)

    # Layer 3: opt into SSE streaming when the env gate is set, gated by
    # an optional process-wide semaphore. Failed semaphore acquirers
    # fall back to the blocking POST so the run never stalls.
    #
    # A ``cancel_event`` argument also forces streaming on: the SSE
    # iterator gives us a per-chunk cancellation hook that actually
    # tears down the TCP socket when the event fires. The non-streaming
    # ``httpx.post`` path holds the read open until the server replies
    # or the read timeout expires — ``Client.close()`` releases the
    # pool but does not abort active reads — so cancellation there
    # wouldn't release vLLM's KV-cache slot until the request finished
    # anyway, defeating the whole point.
    use_stream = cancel_event is not None

    # Ask the server to emit the final ``usage`` block on streaming
    # responses — without this, Gemini/OpenAI-compat streams omit it
    # entirely and ``prompt_tokens_of`` / ``cached_tokens_of`` see
    # nothing. vLLM honors the same field. Merge into any pre-set
    # ``stream_options`` so a caller's flags (e.g. ``continuous_usage``)
    # aren't clobbered.
    if use_stream:
        existing_opts = body.get("stream_options")
        if not isinstance(existing_opts, dict):
            existing_opts = {}
        else:
            existing_opts = dict(existing_opts)
        existing_opts.setdefault("include_usage", True)
        body["stream_options"] = existing_opts
        body.setdefault("stream", True)
        # Watch vLLM #43713 here: the qwen3_xml STREAMING tool parser can emit
        # invalid JSON for several ``<function=...>`` blocks in one
        # ``<tool_call>``. The reasoner's own arg-repair
        # (``_normalize_tool_call_arguments``) is what absorbs it.

    # Two independent budgets. ``attempt`` counts transient faults (5xx,
    # connection errors); ``rate_limit_attempt`` counts 429/402 waits. A
    # single loop counter would let a rate-limit burst spend the fault
    # budget and end the episode on the next unrelated blip.
    attempt = 0
    rate_limit_attempt = 0
    while True:
        try:
            # Pre-call cancellation gate: don't even dial if the
            # caller already aborted (e.g. race deadline reached
            # before we got CPU time).
            if cancel_event is not None and cancel_event.is_set():
                raise CallCanceled("canceled before dispatch")
            ts_start = time.time()
            if use_stream:
                response_json = _stream_chat_completions(
                    url,
                    headers,
                    body,
                    timeout=timeout,
                    cancel_event=cancel_event,
                )
            else:
                with httpx.Client(timeout=timeout) as client:
                    r = client.post(url, headers=headers, json=body)
                if 500 <= r.status_code < 600 and attempt < max_retries:
                    log.warning(
                        "LLM endpoint %d on attempt %d/%d; retrying in %.1fs",
                        r.status_code,
                        attempt + 1,
                        max_retries + 1,
                        backoff_s * (2**attempt),
                    )
                    time.sleep(backoff_s * (2**attempt))
                    attempt += 1
                    continue
                if r.status_code == 404:
                    # Same-URL hot-swap guard: startup discovery is
                    # probe-once, so a server RESTARTED behind the same
                    # URL with a different model surfaces here. Re-check
                    # the served ids and abort naming the mismatch
                    # instead of churning silent per-call 404s.
                    _raise_on_served_model_mismatch(
                        eff_base_url, body, r, eff_api_key
                    )
                r.raise_for_status()
                response_json = r.json()
            if cancel_event is not None and cancel_event.is_set():
                # The answer landed as the caller gave up on it: never used.
                raise CallCanceled("canceled as the answer landed")
            NetMeter.instance().record(
                "reasoner_chat",
                serialized_size(body),
                serialized_size(response_json),
                ts_start,
                time.time(),
                kind=request_kind(body),
                completion_tokens=completion_tokens_of(response_json),
                prompt_tokens=prompt_tokens_of(response_json),
                reasoning_tokens=reasoning_tokens_of(response_json),
                cached_tokens=cached_tokens_of(response_json),
            )
            if flight_dir is not None:
                _flight_response_record(flight_dir, request_id, response_json)
            if mode == "record":
                assert directory is not None
                # Key + write the pristine body so the cache hash
                # is stable regardless of in-flight body mutation.
                key = _cache_key(_pristine_body)
                path = _cache_path(directory, key)
                path.write_text(
                    json.dumps(
                        {
                            "request": _pristine_body,
                            "response": response_json,
                        },
                        default=str,
                    )
                )
            return _RawResponse(response_json)
        except CallCanceled:
            # Surface immediately; cancellation is intentional and
            # retrying would just re-dispatch the same aborted call.
            raise
        except _NO_RETRY_TIMEOUTS as e:
            # A read/write/pool timeout means the request REACHED the server
            # and the server is still working on it -- this path is the
            # blocking POST (`cancel_event is None` on the bench), and as the
            # dispatch comment above says, closing the socket does not abort
            # the work. Re-sending is therefore not a retry, it is a SECOND
            # copy of a multi-megabyte multi-image request queued behind the
            # first, and it fires exactly when the server is already
            # saturated. That is congestion collapse, and it is MEASURED: on
            # one run at workers=128, calls timed out at the 240s
            # default (`SEARCH_REASONER_TIMEOUT_S` unset) and were re-sent
            # 2,253 + 1,944 times, so 128 clients held 407 requests open
            # server-side (`http_requests_active` 407 against
            # `num_running_reqs` 0/0/0) and mean TTFT ran 224s of which the
            # scheduler saw 0.27s. The lane completed 23 of 151 started
            # episodes and then nothing for 44 minutes.
            #
            # A CONNECT timeout is the opposite case and still retries below:
            # no request was accepted, so no server work exists to duplicate.
            log.error(
                "LLM %s after %.0fs; NOT retrying (the server still holds "
                "this request -- a re-send would duplicate it). Raise "
                "SEARCH_REASONER_TIMEOUT_S if this is queueing, not a hang.",
                type(e).__name__,
                timeout,
            )
            raise
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            status = (
                e.response.status_code
                if isinstance(e, httpx.HTTPStatusError)
                else None
            )
            if status in _RATE_LIMIT_STATUS:
                # Capacity/quota, not a malformed body: the same request
                # succeeds once the window resets. Both the streaming and
                # blocking paths land here, so this covers both.
                if rate_limit_attempt >= max_rate_limit_retries:
                    log.error(
                        "LLM endpoint %d after %d rate-limit retries; "
                        "giving up on this call",
                        status,
                        rate_limit_attempt,
                    )
                    raise
                delay = _retry_after_delay(
                    e.response.headers, backoff_s * (2**rate_limit_attempt)
                )
                rate_limit_attempt += 1
                log.warning(
                    "LLM endpoint %d (rate limit/quota) on rate-limit "
                    "retry %d/%d; waiting %.1fs",
                    status,
                    rate_limit_attempt,
                    max_rate_limit_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            if status is not None and status < 500:
                try:
                    resp_text = e.response.text
                except Exception:
                    resp_text = "<failed to read response body>"
                tools_list = body.get("tools") or []
                log.error(
                    "LLM endpoint %d response body: %s",
                    e.response.status_code,
                    resp_text[:4000],
                )
                msgs = body.get("messages") or []
                msg_sizes = []
                for i, m in enumerate(msgs):
                    c = m.get("content")
                    if isinstance(c, str):
                        n = len(c)
                    elif c is None:
                        n = 0
                    else:
                        n = len(json.dumps(c, default=str))
                    tcs = m.get("tool_calls") or []
                    tc_chars = sum(
                        len(((tc.get("function") or {}).get("arguments") or ""))
                        for tc in tcs
                    )
                    msg_sizes.append(
                        f"[{i}:{m.get('role')}]={n}c"
                        + (f"+{tc_chars}tc" if tc_chars else "")
                    )
                tools_chars = sum(
                    len(json.dumps(t, default=str)) for t in tools_list
                )
                log.error(
                    "LLM endpoint %d request summary: transport_kind=%s, "
                    "body_keys=%s, n_tools=%d (%dc), tool_choice=%r, "
                    "sampling=%r, model=%r, "
                    "n_messages=%d, msg_sizes=%s, total_msg_chars=%d",
                    e.response.status_code,
                    transport_meta.get("transport_kind"),
                    sorted(body.keys()),
                    len(tools_list),
                    tools_chars,
                    body.get("tool_choice"),
                    {
                        k: body.get(k)
                        for k in (
                            "temperature",
                            "top_p",
                            "top_k",
                            "presence_penalty",
                            "max_tokens",
                            # A serve whose template carries no effort
                            # field answers 400 on one, and this line is
                            # where that reads: the summary names the key
                            # and the value, so an arm asking for an
                            # effort the card does not take is a legible
                            # refusal rather than an unexplained
                            # malformed-request raise. The body head below
                            # cannot serve for it — ``messages`` is dumped
                            # first and swamps the 8000-char window.
                            "reasoning_effort",
                        )
                        if k in body
                    },
                    body.get("model"),
                    len(msgs),
                    " ".join(msg_sizes),
                    sum(
                        (
                            len(m["content"])
                            if isinstance(m.get("content"), str)
                            else 0
                        )
                        for m in msgs
                    ),
                )
                log.error(
                    "LLM endpoint %d request body (head): %s",
                    e.response.status_code,
                    json.dumps(body, indent=2, default=str)[:8000],
                )
                # A <500 that is not a rate limit is a malformed-request
                # bug, not a capacity event; surface it (retry won't fix
                # it).
                raise
            if attempt >= max_retries:
                raise
            log.warning(
                "LLM transport error on attempt %d/%d: %s; retrying in %.1fs",
                attempt + 1,
                max_retries + 1,
                e,
                backoff_s * (2**attempt),
            )
            time.sleep(backoff_s * (2**attempt))
            attempt += 1
