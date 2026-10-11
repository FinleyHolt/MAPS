"""Raw HTTP client for OpenAI-compatible chat-completions.

Every stage of the parse-first compiler goes through this client rather
than the OpenAI SDK, so the request body is exactly what
`maps.llm.model_cards.sampling_kwargs` builds — a flat set of top-level keys that both a
self-hosted vLLM and a hosted endpoint honour. No LLM logic here; just
the transport.

Retries are split by cause. A 5xx or a transport error is a transient
fault and burns the `_MAX_RETRIES` budget. A 429/402 is a capacity or
quota event with its own, more patient budget (`_MAX_RATE_LIMIT_RETRIES`,
honouring `Retry-After`) — hosted endpoints rate-limit routinely, and
spending the transient budget on that would end an episode mid-compile.
Every other 4xx is a malformed-request bug and is surfaced immediately.

A call that carries a ``cancel_event`` negotiates an SSE stream from
vLLM rather than a blocking POST. When the event fires the call's TCP
socket is shut down, so a read blocked on a silent serve returns at once
and the serve frees the KV-cache slot, and the call raises
:class:`CallCanceled`. An answer that lands as the event fires is never
handed back or recorded.
"""

from __future__ import annotations

import base64
import binascii
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

from . import endpoint_spec


# 4xx codes that mean "not now" rather than "not ever": 429 is the
# rate limit, 402 the credit/quota check. Both clear on their own, so
# they get a re-send instead of the immediate raise every other 4xx
# earns. A self-hosted vLLM emits neither; a hosted endpoint emits 429
# routinely under the compile's parallel fan-out.
_RATE_LIMIT_STATUS = frozenset({402, 429})
# Every 4xx a serve may clear on its own: the rate limit and quota checks,
# and 408 request timeout. Any other 4xx is a request the caller built wrong.
TRANSIENT_CLIENT_STATUS = _RATE_LIMIT_STATUS | {408}

# Timeouts that are NOT retried: the server already has the request and is
# still working on it, so a re-send duplicates the work instead of replacing
# it. ``ConnectTimeout`` is deliberately absent -- nothing was accepted there,
# so retrying costs the server nothing.
_NO_RETRY_TIMEOUTS = (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)

# Ceiling on one rate-limit wait. A provider may advertise a long
# ``Retry-After``; past this the run is better off failing loud than
# silently parking a reasoner worker.
_RATE_LIMIT_MAX_BACKOFF_S = 60.0

# The two retry budgets and the exponential backoff base, in seconds.
_MAX_RETRIES = 2
_MAX_RATE_LIMIT_RETRIES = 5
_BACKOFF_S = 1.5


def _retry_after_delay(headers: httpx.Headers, default_s: float) -> float:
    """Seconds to wait before re-sending: ``Retry-After`` in its delta-seconds
    form, else ``default_s``, capped. The HTTP-date form, rare on JSON APIs,
    takes the default."""
    try:
        delay = float(headers.get("retry-after", ""))
    except ValueError:
        delay = default_s
    return max(0.0, min(_RATE_LIMIT_MAX_BACKOFF_S, delay))


def _backoff(delay: float, cancel_event: threading.Event | None) -> None:
    """Wait out one retry backoff; a cancel ends the wait, and the loop's
    pre-dispatch check then raises."""
    if cancel_event is None:
        time.sleep(delay)
    else:
        cancel_event.wait(delay)


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
# Flight log
# ---------------------------------------------------------------------------


def _resolved_flight_log_dir() -> Path | None:
    """Where this mission's flight log goes: ``<workdir>/flight/llm``.

    Every chat-completions round-trip is mirrored into
    ``transcript.jsonl`` there -- an append-only record stream that stores
    the conversation ONCE (per-call message deltas + params dedup + image
    parts as sha refs) instead of a full re-serialization per call. It is
    what nearly every reader under ``benchmarks/citynav/analysis/`` reads.

    RECORDING IS NOT OPTIONAL, and no environment variable turns it off: a
    mission that records nothing looks exactly like one that has nothing
    to say. The one thing configurable is WHERE a mission writes:
    ``SEARCH_COMPILE_WORKDIR``, which every caller that flies anything sets.

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


# The transcript is a SINGLE-PROCESS append stream: every call writes from
# the runner process, so one module lock serializes lines. A second process appending to the same file would
# interleave records undetected — if a multi-process runner ever appears,
# the transcript needs a per-process file + a merge, not a bigger lock.
_FLIGHT_LOCK = threading.Lock()
# flight_dir -> the previous request's scrubbed-message digests, for
# write-time delta encoding. Keyed by dir so sequential episodes in one
# process cannot cross-contaminate.
_FLIGHT_THREAD_DIGESTS: dict[str, list[str]] = {}
# flight_dir -> params shas already written to that transcript.
_FLIGHT_PARAMS_SEEN: dict[str, set[str]] = {}


def _scrub_image_parts(messages: list) -> list:
    """Message list with every data-URL image part replaced by a sha ref.

    ``{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,…"}}``
    becomes ``{"type": "image_ref", "media_type", "sha256", "bytes"}``. The
    sha is of the DECODED bytes and matches the look manifest's per-crop
    ``sha256``, so a transcript image part resolves to its crop record (and
    re-renders) by hash. Non-data URLs pass through. Never mutates the input
    — the HTTP body sees the original."""
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
    must never fail an LLM call. The flush makes each line visible to a
    live reader immediately."""
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
) -> None:
    """Append the params (if new) + request records for one dispatch.

    The flight log's "pending" pre-write: a reader sees the round the
    moment the call fires, before the response lands. ``new_messages`` is
    the delta against the previous call (write-time LCP over
    scrubbed-message digests); the reader reconstructs the full request as
    ``history[:n_prefix] + new_messages``, exactly. ``params`` (the body
    minus messages and transport flags — tools, response_format, sampling)
    is written once per distinct sha, which removes the other per-call
    re-serialization."""
    with _FLIGHT_LOCK:
        try:
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
            prev = _FLIGHT_THREAD_DIGESTS.get(str(directory)) or []
            n_prefix = 0
            for a, b in zip(prev, digests):
                if a != b:
                    break
                n_prefix += 1
            _FLIGHT_THREAD_DIGESTS[str(directory)] = digests
            _append_flight_record(directory, {
                "kind": "request",
                "request_id": request_id,
                "created_at": time.time(),
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


def _request_headers(api_key: str | None) -> dict[str, str]:
    """Headers for one chat-completions POST.

    ``SEARCH_LLM_ROUTING_KEY`` (set per episode by the runner) rides as
    an opaque affinity header — a key-affine router (sglang-router
    ``--policy manual``) pins every call carrying the same key to one
    worker, so an episode's growing context stays with its radix cache.
    Unset = no header. Read per call: the runner rewrites the key each
    episode.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    routing_key = os.environ.get("SEARCH_LLM_ROUTING_KEY")
    if routing_key:
        headers["x-smg-routing-key"] = routing_key
    return headers


_EMPTY_STREAM_S = 90.0
_EMPTY_STREAM_CHUNKS = 1200


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
    "usage": {...}}`` shape vLLM returns for non-streaming calls.
    Streaming exists for CANCELLATION (the per-chunk hook that tears down
    the TCP socket), not for logging: the flight transcript stores no
    per-token stream.
    """
    aggregated_text: list[str] = []
    aggregated_reasoning: list[str] = []
    aggregated_tool_calls: dict[int, dict] = {}
    finish_reason: str | None = None
    usage: dict | None = None
    model: str | None = None
    role: str | None = None
    # A server that streams nothing but empty deltas is a hung generation;
    # give up after _EMPTY_STREAM_S or _EMPTY_STREAM_CHUNKS, whichever first.
    empty_chunk_count = 0
    first_empty_at: float | None = None

    # The cancel watch shuts the TCP socket down when the event fires; a
    # blocked read (the serve silent before its headers or between chunks)
    # returns at once instead of waiting out the read timeout.
    watch = _CancelWatch(cancel_event) if cancel_event is not None else None
    try:
        with httpx.Client(timeout=timeout) as client, client.stream(
            "POST", url, headers=headers, json=body,
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
                    continue
                empty_chunk_count += 1
                first_empty_at = first_empty_at or time.monotonic()
                if (time.monotonic() - first_empty_at >= _EMPTY_STREAM_S
                        or empty_chunk_count >= _EMPTY_STREAM_CHUNKS):
                    raise httpx.ReadTimeout(
                        f"LLM stream emitted only empty deltas for {empty_chunk_count} chunks")
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


def _raise_on_served_model_mismatch(base_url: str, requested: str | None, refusal: str) -> None:
    """On a 404 chat completion, re-probe the served ids and abort LOUDLY when
    the request's model is no longer among them: the server behind the URL
    changed since startup discovery (probe-once cannot see a same-URL
    restart). Returns (the caller raises the plain HTTP error) when the probe
    fails or the model is still served (a 404 for some other reason)."""
    if not requested:
        return
    try:
        served = [str(m.get("id")) for m in endpoint_spec._probe_models(base_url)]
    except endpoint_spec.EndpointDiscoveryError:
        return
    if requested not in served:
        raise RuntimeError(
            f"served-model mismatch: this process discovered {requested!r} "
            f"at startup but {base_url} now serves {served} "
            f"(chat completion returned 404: {refusal[:200]!r}). The server "
            f"behind the URL changed mid-run — drain and restart."
        )


#: The sampling keys a refused request's summary names (a serve whose template
#: carries no effort field answers 400 on one).
_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "max_tokens", "max_completion_tokens", "reasoning_effort")


def _chat_completions_raw(
    base_url: str,
    api_key: str | None,
    *,
    timeout: float,
    cancel_event: threading.Event | None = None,
    **kwargs,
) -> dict:
    """Send a chat completion via raw HTTP and return the response JSON.

    A 5xx or a transport error (connection refused, connect timeout) burns
    `_MAX_RETRIES` with exponential backoff. Rate-limit and quota responses
    (429/402) draw on a separate `_MAX_RATE_LIMIT_RETRIES` budget, each wait
    honouring `Retry-After`. Every other 4xx is surfaced immediately; it
    indicates a malformed request that retrying would not fix.
    """
    body = kwargs
    flight_dir = _resolved_flight_log_dir()
    url = base_url.rstrip("/") + "/chat/completions"
    headers = _request_headers(api_key)

    # The request record lands the moment the call fires, so a call that
    # never returns stays a pending round in the transcript.
    request_id = uuid.uuid4().hex
    if flight_dir is not None:
        _flight_request_record(flight_dir, request_id, body)

    # A ``cancel_event`` streams: the SSE path is where the cancel watch
    # shuts the socket down (see ``_stream_chat_completions``).
    use_stream = cancel_event is not None
    if use_stream:
        # The final ``usage`` block: without it OpenAI-compatible streams
        # omit token counts and the transcript's response carries none.
        body["stream_options"] = {"include_usage": True}
        body["stream"] = True
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
            if use_stream:
                response_json = _stream_chat_completions(
                    url, headers, body, timeout=timeout, cancel_event=cancel_event)
            else:
                with httpx.Client(timeout=timeout) as client:
                    r = client.post(url, headers=headers, json=body)
                r.raise_for_status()
                response_json = r.json()
            if cancel_event is not None and cancel_event.is_set():
                # The answer landed as the caller gave up on it: never used.
                raise CallCanceled("canceled as the answer landed")
            if flight_dir is not None:
                _flight_response_record(flight_dir, request_id, response_json)
            return response_json
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
                if rate_limit_attempt >= _MAX_RATE_LIMIT_RETRIES:
                    log.error(
                        "LLM endpoint %d after %d rate-limit retries; "
                        "giving up on this call",
                        status,
                        rate_limit_attempt,
                    )
                    raise
                delay = _retry_after_delay(
                    e.response.headers, _BACKOFF_S * (2**rate_limit_attempt)
                )
                rate_limit_attempt += 1
                log.warning(
                    "LLM endpoint %d (rate limit/quota) on rate-limit "
                    "retry %d/%d; waiting %.1fs",
                    status,
                    rate_limit_attempt,
                    _MAX_RATE_LIMIT_RETRIES,
                    delay,
                )
                _backoff(delay, cancel_event)
                continue
            if status == 404:
                # Same-URL hot-swap guard: startup discovery is probe-once,
                # so a server RESTARTED behind the same URL with a different
                # model surfaces here, on either path.
                _raise_on_served_model_mismatch(base_url, body.get("model"), e.response.text)
            if status is not None and status < 500:
                # A <500 that is not a rate limit is a malformed-request
                # bug, not a capacity event; surface it (retry won't fix it).
                msgs = body.get("messages") or []
                log.error(
                    "LLM endpoint %d response body: %s", status, e.response.text[:4000])
                log.error(
                    "LLM endpoint %d request summary: body_keys=%s, "
                    "n_tools=%d, sampling=%r, model=%r, n_messages=%d, roles=%s",
                    status, sorted(body),
                    len(body.get("tools") or []), {k: body[k] for k in _SAMPLING_KEYS if k in body},
                    body.get("model"), len(msgs), [m.get("role") for m in msgs])
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
