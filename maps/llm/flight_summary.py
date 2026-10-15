"""One episode's LLM activity, reduced to a block that rides its row.

Folds the flight dir's ``llm/transcript.jsonl`` into turns, requests, token
sums and latency. The cache meters (``cached_tokens``, ``cache_write_tokens``,
off ``usage.prompt_tokens_details``) and ``served_model`` (the responses'
``model``, a hosted provider's versioned name) are None when no response
reported them: a serve without cache reporting reads n/a, not a zero hit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .http import TRANSCRIPT_JSONL
from .model_cards import speaks_openai
from .tokenizer import load_tokenizer

log = logging.getLogger(__name__)

def _as_dict(v: Any) -> dict:
    """``v`` if it is a dict, else ``{}``: model-emitted JSON is untrusted in
    shape (an object field can arrive as a string)."""
    return v if isinstance(v, dict) else {}


def blank() -> dict:
    """The zero shape, written when the episode kept no transcript."""
    return {
        "turns": 0,
        "requests": 0,
        "unanswered": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "latency_s_sum": 0.0,
        "cached_tokens": None,
        "cache_write_tokens": None,
        "served_model": None,
    }


def reasoning_tokens_of(response_json: dict, *, model: str | None = None) -> int:
    """The reasoning-span token count of one response, a slice of
    ``usage.completion_tokens``.

    For an OpenAI-wire ``model`` (no span comes back) it is the provider's
    ``usage.completion_tokens_details.reasoning_tokens``. Otherwise the span
    decides, because a serve's ``usage.reasoning_tokens`` can claim thinking
    on a response that carries none: no ``reasoning``/``reasoning_content``
    reads 0; else ``usage.reasoning_tokens`` when positive; else the span
    tokenized with the response ``model``'s tokenizer; else 0.
    """
    usage = _as_dict(response_json.get("usage"))
    wire = speaks_openai(model)
    meter = (_as_dict(usage.get("completion_tokens_details")) if wire else usage).get("reasoning_tokens")
    direct = int(meter) if isinstance(meter, (int, float)) else 0
    if wire:
        return direct
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
    if direct > 0:
        return direct
    model_alias = response_json.get("model")
    if not isinstance(model_alias, str):
        return 0
    try:
        tok = load_tokenizer(model_alias)
    except Exception:  # noqa: BLE001 — no tokenizer reads 0
        return 0
    total = 0
    for text in reasoning_texts:
        try:
            ids = tok.encode(text, add_special_tokens=False)
        except Exception:  # noqa: BLE001 — tokenizer quirks; fall through silently
            continue
        total += len(ids)
    return total


def _scan_transcript(path: Path, out: dict) -> None:
    """Fold one transcript into ``out`` in a raw line scan (no request is
    rebuilt): ``params`` for the model, ``request`` for timestamps,
    ``response`` for the rest."""
    started: dict[str, float] = {}
    model: str | None = None
    with path.open(errors="replace") as fh:
        for line in fh:
            if '"kind":' not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:  # a live writer's truncated last line
                continue
            rec = _as_dict(rec)
            kind = rec.get("kind")
            if kind == "params":
                params = _as_dict(rec.get("params"))
                if isinstance(params.get("model"), str):
                    model = params["model"]
                continue
            if kind == "request":
                out["requests"] += 1
                ts = rec.get("created_at")
                if isinstance(ts, (int, float)):
                    started[str(rec.get("request_id"))] = float(ts)
                continue
            if kind != "response":
                continue

            out["turns"] += 1
            t0 = started.pop(str(rec.get("request_id")), None)
            t1 = rec.get("ts")
            if t0 is not None and isinstance(t1, (int, float)):
                out["latency_s_sum"] += float(t1) - t0

            body = _as_dict(rec.get("response"))
            out["reasoning_tokens"] += reasoning_tokens_of(body, model=model)
            usage = _as_dict(body.get("usage"))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                v = usage.get(key)
                if isinstance(v, (int, float)):
                    out[key] += int(v)
            details = _as_dict(usage.get("prompt_tokens_details"))
            for key in ("cached_tokens", "cache_write_tokens"):
                v = details.get(key)
                if isinstance(v, (int, float)):
                    out[key] = (out[key] or 0) + int(v)
            if isinstance(body.get("model"), str):
                out["served_model"] = body["model"]

    # requests never answered: a crash, a timeout or a kill mid-call
    out["unanswered"] = len(started)


def summarize(flight_dir: Path) -> dict:
    """One episode's LLM activity from its flight dir; :func:`blank` when it
    kept no transcript. Never raises: it runs on the episode's way out."""
    out = blank()
    transcript = Path(flight_dir) / "llm" / TRANSCRIPT_JSONL
    try:
        _scan_transcript(transcript, out)
    except OSError:
        return blank()
    except Exception as exc:  # noqa: BLE001 — never fail the episode
        log.warning("flight summary scan failed at %s: %s", transcript, exc)
        return blank()

    out["latency_s_sum"] = round(out["latency_s_sum"], 3)
    return out
