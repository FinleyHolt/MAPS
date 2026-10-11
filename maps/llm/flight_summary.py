"""One episode's LLM activity, reduced to a block that rides its row.

:func:`summarize` folds the flight dir's ``llm/transcript.jsonl`` into turns,
requests, token sums, latency, a histogram of the tool calls the model emitted
and the tools each ``params`` record declared.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .model_cards import speaks_openai
from .tokenizer import _tokenizer

log = logging.getLogger(__name__)

#: The histogram bucket of a tool name the model generated rather than
#: selected (leaked prose), counted rather than binned verbatim.
MALFORMED = "«malformed»"

_TOOL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,39}\Z")


def _tool_name(raw: Any) -> str:
    name = str(raw or "")
    return name if _TOOL_NAME.match(name) else MALFORMED


def _as_dict(v: Any) -> dict:
    """``v`` if it is a dict, else ``{}``: model-emitted JSON is untrusted in
    shape (an object field can arrive as a string)."""
    return v if isinstance(v, dict) else {}


def _declared_tools(params: dict) -> list[str]:
    """The tool names one params record declares, in order."""
    names = (_as_dict(_as_dict(t).get("function")).get("name")
             for t in params.get("tools") or [])
    return [n for n in names if n]


def _bump(d: dict, k: Any) -> None:
    if k:
        d[k] = d.get(k, 0) + 1


def blank() -> dict:
    """The zero shape, written when the episode kept no transcript."""
    return {
        "turns": 0,
        "requests": 0,
        "unanswered": 0,
        "prompt_tokens": 0,
        # the largest single prompt: how full the window ever got
        "prompt_tokens_max": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "latency_s_sum": 0.0,
        "latency_s_max": 0.0,
        "calls": {},
        "declared": {"tools": []},
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
        tok = _tokenizer(model_alias)
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
    rebuilt): ``params`` for the declared tools, ``request`` for timestamps,
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
                for t in _declared_tools(params):
                    if t not in out["declared"]["tools"]:
                        out["declared"]["tools"].append(t)
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
                lat = float(t1) - t0
                out["latency_s_sum"] += lat
                out["latency_s_max"] = max(out["latency_s_max"], lat)

            body = _as_dict(rec.get("response"))
            out["reasoning_tokens"] += reasoning_tokens_of(body, model=model)
            usage = _as_dict(body.get("usage"))
            for key, field in (
                ("prompt_tokens", "prompt_tokens"),
                ("completion_tokens", "completion_tokens"),
                ("total_tokens", "total_tokens"),
            ):
                v = usage.get(field)
                if isinstance(v, (int, float)):
                    out[key] += int(v)
            p = usage.get("prompt_tokens")
            if isinstance(p, (int, float)):
                out["prompt_tokens_max"] = max(out["prompt_tokens_max"], int(p))

            message = _as_dict(
                _as_dict((body.get("choices") or [{}])[0]).get("message")
            )
            for tc in message.get("tool_calls") or []:
                fn = _as_dict(_as_dict(tc).get("function"))
                _bump(out["calls"], _tool_name(fn.get("name")))

    # requests never answered: a crash, a timeout or a kill mid-call
    out["unanswered"] = len(started)


def summarize(flight_dir: Path) -> dict:
    """One episode's LLM activity from its flight dir; :func:`blank` when it
    kept no transcript. Never raises: it runs on the episode's way out."""
    out = blank()
    transcript = Path(flight_dir) / "llm" / "transcript.jsonl"
    try:
        _scan_transcript(transcript, out)
    except OSError:
        return blank()
    except Exception as exc:  # noqa: BLE001 — never fail the episode
        log.warning("flight summary scan failed at %s: %s", transcript, exc)
        return blank()

    out["latency_s_sum"] = round(out["latency_s_sum"], 3)
    out["latency_s_max"] = round(out["latency_s_max"], 3)
    return out
