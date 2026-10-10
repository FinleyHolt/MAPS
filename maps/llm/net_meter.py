"""Bytes-on-wire meter for outbound LLM calls.

Activated by ``NET_METER_LOG=<path>`` — when set, every recorded call
appends one JSON object to ``<path>``. When unset, ``record()`` is a
no-op so production runs pay nothing.

Each line:

    {"ts_start": 12345.678,    # time.time() at request submit
     "ts_end":   12345.912,    # time.time() at response receipt
     "endpoint": "reasoner_chat",  # caller-supplied label
     "kind":     "text",       # "text" | "image" — image-bearing calls
                               # carry base64 frames in the request body
     "up_bytes":   12345,      # serialized request body length
     "down_bytes":  4567,      # serialized response body length
     "prompt_tokens":   3421,  # from response.usage.prompt_tokens
                               # (0 if vLLM doesn't return usage)
     "completion_tokens": 256, # from response.usage.completion_tokens
                               # (0 if vLLM doesn't return usage)
     "reasoning_tokens": 128,  # from response.usage.reasoning_tokens
                               # (0 if absent — present on Qwen3 thinking
                               # mode, OpenAI o-series, etc.)
     "latency_s":   0.234,
     "stage":   "mission",          # nullable; set by the active
                                    # ``net_meter.stage(...)`` context
     "metadata": {"reasoner_idx": 0, ...}}  # nullable free-form

Built for sizing the bandwidth required to move text-LLM traffic from
the on-prem vLLM at NPS to a cloud endpoint, and for attributing latency
to compile stages so a 19-min episode can be split into "where did the
time go" buckets. Offline throughput analyzers consume this JSONL and
emit a summary report.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Iterator

from .tokenizer import _tokenizer

log = logging.getLogger(__name__)


_stage_local = threading.local()


class NetMeter:
    """Process-wide singleton that appends per-call records to a JSONL file."""

    _instance: "NetMeter | None" = None
    _instance_env: str | None = None
    _instance_lock = threading.Lock()

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._write_lock = threading.Lock()
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                log.warning("net_meter: cannot create %s: %s", path.parent, exc)
                self._path = None

    @property
    def enabled(self) -> bool:
        return self._path is not None

    @classmethod
    def instance(cls) -> "NetMeter":
        # Keyed on the env value, not latched at first use: the episode
        # server rewrites NET_METER_LOG per episode, and a latched path put
        # two episodes' calls in one file (652 meter files for 1007 workdirs
        # on the pulled fleet).
        with cls._instance_lock:
            raw = os.environ.get("NET_METER_LOG", "").strip()
            if cls._instance is None or raw != cls._instance_env:
                cls._instance = cls(Path(raw).expanduser() if raw else None)
                cls._instance_env = raw
            return cls._instance

    def record(
        self,
        endpoint: str,
        up_bytes: int,
        down_bytes: int,
        ts_start: float,
        ts_end: float,
        kind: str = "text",
        completion_tokens: int = 0,
        prompt_tokens: int = 0,
        reasoning_tokens: int = 0,
        cached_tokens: int | None = None,
        stage: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._path is None:
            return
        active_stage, active_meta = current_stage()
        record = {
            "ts_start": ts_start,
            "ts_end": ts_end,
            "endpoint": endpoint,
            "kind": kind,
            "up_bytes": int(up_bytes),
            "down_bytes": int(down_bytes),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "reasoning_tokens": int(reasoning_tokens),
            "latency_s": max(0.0, ts_end - ts_start),
            "stage": stage if stage is not None else active_stage,
        }
        if cached_tokens is not None:
            record["cached_tokens"] = int(cached_tokens)
        merged_meta: dict[str, Any] = {}
        if active_meta:
            merged_meta.update(active_meta)
        if metadata:
            merged_meta.update(metadata)
        if merged_meta:
            record["metadata"] = merged_meta
        line = json.dumps(record) + "\n"
        with self._write_lock:
            try:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
            except OSError as exc:
                log.warning("net_meter: write failed at %s: %s", self._path, exc)


@contextlib.contextmanager
def stage(name: str, **metadata: Any) -> Iterator[None]:
    """Tag every NetMeter record emitted inside this block with ``name``.

    Stacks: nested ``stage()`` blocks override the outer name (innermost
    wins) and merge metadata. Thread-local — workers running in parallel
    each see their own active stage. Outside any block, records carry
    ``stage=None``.
    """
    stack = getattr(_stage_local, "stack", None)
    if stack is None:
        stack = []
        _stage_local.stack = stack
    stack.append((name, dict(metadata)))
    try:
        yield
    finally:
        stack.pop()


def current_stage() -> tuple[str | None, dict[str, Any]]:
    """Return ``(name, merged_metadata)`` for the active stage stack.

    Metadata from outer frames is merged first; inner frames overwrite.
    Returns ``(None, {})`` when no stage is active.
    """
    stack = getattr(_stage_local, "stack", None)
    if not stack:
        return None, {}
    merged: dict[str, Any] = {}
    for _, meta in stack:
        merged.update(meta)
    name = stack[-1][0]
    return name, merged


def request_kind(body: dict) -> str:
    """Classify a chat-completions request body as ``"text"`` or ``"image"``.

    OpenAI/vLLM multimodal messages put image data in
    ``messages[*].content`` as a list of typed parts; an image part has
    ``type == "image_url"`` (or ``"image"``). Any image part anywhere in
    the conversation flips the request to ``"image"``. Tool-result
    messages are walked too — perception tools sometimes return images
    back to the model.
    """
    for msg in body.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            t = part.get("type")
            if t in ("image_url", "image", "input_image"):
                return "image"
    return "text"


def completion_tokens_of(response_json: dict) -> int:
    """Extract ``usage.completion_tokens`` from a chat-completions response.

    Returns 0 when the field is missing or non-numeric. vLLM normally
    populates ``usage`` but some streaming code paths may omit it.
    """
    return _usage_int(response_json, "completion_tokens")


def prompt_tokens_of(response_json: dict) -> int:
    """Extract ``usage.prompt_tokens`` from a chat-completions response.

    Returns 0 when the field is missing or non-numeric. Mirrors
    ``completion_tokens_of`` for context-size telemetry.
    """
    return _usage_int(response_json, "prompt_tokens")


def cached_tokens_of(response_json: dict) -> int | None:
    """The prompt tokens the serve read from its prefix cache,
    ``usage.prompt_tokens_details.cached_tokens``; ``None`` when the serve
    sends no ``prompt_tokens_details``, so a serve without cache reporting
    reads n/a rather than 0.
    """
    usage = response_json.get("usage") if isinstance(response_json, dict) else None
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    if not isinstance(details, dict):
        return None
    try:
        return int(details.get("cached_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def reasoning_tokens_of(response_json: dict) -> int:
    """Extract the reasoning-span token count from a chat-completions response.

    A reasoning serve routes the thinking span into
    ``choices[i].message.reasoning_content`` (or ``reasoning`` on some
    SDKs) and rolls its tokens into ``usage.completion_tokens``. Some
    serves also report ``usage.reasoning_tokens``; where they do, it is
    the exact count and beats tokenizing the text.

    **THE SPAN DECIDES, NOT THE USAGE FIELD.** A response carrying no
    thinking span reads 0 here whatever ``usage`` claims, and the order
    is that way round because the claim is not reliable: sglang 0.6.17
    serving Inkling reports ``reasoning_tokens = completion_tokens - 1``
    on a response whose ``reasoning_content`` is ``null`` — not zero, not
    absent, but one short of the entire generation. Measured on the
    effort ladder, where the arm served ``reasoning_effort=0.0`` produced
    341 responses, ZERO of them carrying a span, and 339 of them claiming
    to be almost entirely thinking. Preferring the field made the
    least-thinking arm read as the most-thinking one, and did it
    monotonically across the whole ladder, which is the shape that gets
    believed.

    Order of resolution:
      1. no ``reasoning`` / ``reasoning_content`` on any choice → 0.
      2. ``usage.reasoning_tokens`` when the serve reports one.
      3. tokenize the span with the same HF tokenizer the compactor
         uses, keyed off the response's ``model`` alias.
      4. 0 (no tokenizer available).

    Note: ``completion_tokens_of`` continues to return the wire
    ``usage.completion_tokens`` total (which includes the reasoning
    tokens). The two buckets overlap by design; ``reasoning`` is a
    diagnostic slice of ``completion``, not a disjoint addition. The
    runner's per-stage summary line reports both so readers can see
    what fraction of decode is thinking.
    """
    choices = response_json.get("choices") or []
    reasoning_texts: list[str] = []
    for ch in choices:
        if not isinstance(ch, dict):
            continue
        msg = ch.get("message") or {}
        if not isinstance(msg, dict):
            continue
        for field_name in ("reasoning", "reasoning_content"):
            val = msg.get(field_name)
            if isinstance(val, str) and val:
                reasoning_texts.append(val)
                break
    if not reasoning_texts:
        return 0
    direct = _usage_int(response_json, "reasoning_tokens")
    if direct > 0:
        return direct
    model_alias = response_json.get("model")
    if not isinstance(model_alias, str):
        return 0
    tok = _resolve_response_tokenizer(model_alias)
    if tok is None:
        return 0
    total = 0
    for text in reasoning_texts:
        try:
            ids = tok.encode(text, add_special_tokens=False)
        except Exception:  # noqa: BLE001 — tokenizer quirks; fall through silently
            continue
        total += len(ids)
    return total


def _resolve_response_tokenizer(model_alias: str):
    """Return the HF tokenizer for a vLLM model alias, or ``None`` if it
    can't be loaded."""
    try:
        return _tokenizer(model_alias)
    except Exception:  # noqa: BLE001 — robust against tokenizer load races
        return None


def _usage_int(response_json: dict, key: str) -> int:
    usage = response_json.get("usage") if isinstance(response_json, dict) else None
    if not isinstance(usage, dict):
        return 0
    val = usage.get(key)
    try:
        return int(val) if val is not None else 0
    except (TypeError, ValueError):
        return 0


def summarize_window(
    jsonl_path: Path,
    *,
    t_start: float | None = None,
    t_end: float | None = None,
) -> dict[str, Any]:
    """Aggregate NetMeter records in ``[t_start, t_end]`` into a summary dict.

    The window is inclusive on both ends; ``None`` means open. Records
    are bucketed by ``stage`` (``None`` rows go under the literal string
    ``"unstaged"``). Total counts cover *all* records in the window.

    Returned shape::

        {
          "n_calls": 47,
          "total_prompt_tokens": 312456,
          "total_completion_tokens": 8934,
          "total_reasoning_tokens": 49200,
          "total_latency_s": 184.2,
          "total_up_bytes": ...,
          "total_down_bytes": ...,
          "by_stage": {
            "round0": {
              "n": 5,
              "prompt_tokens": ...,
              "completion_tokens": ...,
              "reasoning_tokens": ...,
              "latency_s": ...,
              "kinds": {"text": 5, "image": 0},
            },
            ...
          },
          "by_kind": {"text": {...}, "image": {...}},
        }

    Returns an empty summary (zeroed) if the file is missing.
    """
    summary: dict[str, Any] = {
        "n_calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_reasoning_tokens": 0,
        "total_latency_s": 0.0,
        "total_up_bytes": 0,
        "total_down_bytes": 0,
        "by_stage": {},
        "by_kind": {},
    }
    if not jsonl_path.exists():
        return summary
    try:
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = float(rec.get("ts_start", 0.0))
                if t_start is not None and ts < t_start:
                    continue
                if t_end is not None and ts > t_end:
                    continue
                _accumulate(summary, rec)
    except OSError as exc:
        log.warning(
            "net_meter: summarize_window read failed at %s: %s", jsonl_path, exc
        )
    return summary


def _accumulate(summary: dict[str, Any], rec: dict[str, Any]) -> None:
    pt = int(rec.get("prompt_tokens") or 0)
    ct = int(rec.get("completion_tokens") or 0)
    rt = int(rec.get("reasoning_tokens") or 0)
    cached = rec.get("cached_tokens")
    lat = float(rec.get("latency_s") or 0.0)
    up = int(rec.get("up_bytes") or 0)
    down = int(rec.get("down_bytes") or 0)
    kind = rec.get("kind") or "text"
    stage = rec.get("stage") or "unstaged"
    summary["n_calls"] += 1
    summary["total_prompt_tokens"] += pt
    summary["total_completion_tokens"] += ct
    summary["total_reasoning_tokens"] += rt
    if cached is not None:
        summary["total_cached_tokens"] = summary.get("total_cached_tokens", 0) + int(cached)
    summary["total_latency_s"] += lat
    summary["total_up_bytes"] += up
    summary["total_down_bytes"] += down
    by_stage = summary["by_stage"].setdefault(
        stage,
        {
            "n": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "latency_s": 0.0,
            "kinds": {},
        },
    )
    by_stage["n"] += 1
    by_stage["prompt_tokens"] += pt
    by_stage["completion_tokens"] += ct
    by_stage["reasoning_tokens"] += rt
    if cached is not None:
        by_stage["cached_tokens"] = by_stage.get("cached_tokens", 0) + int(cached)
    by_stage["latency_s"] += lat
    by_stage["kinds"][kind] = by_stage["kinds"].get(kind, 0) + 1
    by_kind = summary["by_kind"].setdefault(
        kind,
        {
            "n": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "latency_s": 0.0,
        },
    )
    by_kind["n"] += 1
    by_kind["prompt_tokens"] += pt
    by_kind["completion_tokens"] += ct
    by_kind["reasoning_tokens"] += rt
    by_kind["latency_s"] += lat


def serialized_size(payload: dict) -> int:
    """Length in bytes of the canonical JSON serialization of a request/response.

    Matches the wire form vLLM sends and receives: UTF-8 JSON without
    insignificant whitespace, defaulting non-serializable objects to
    their string representation (consistent with how ``llm_http``
    persists its flight-log payloads).
    """
    return len(json.dumps(payload, default=str).encode("utf-8"))
