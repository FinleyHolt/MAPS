"""One episode's LLM activity, reduced to a row-sized block.

The flight dir already holds everything: ``llm/transcript.jsonl`` records
every round-trip (:mod:`maps.llm.http` writes it) and ``net_meter.jsonl``
records per-call token/latency accounting (:mod:`maps.llm.net_meter`).
Both are written on every episode. Neither survives a rows-only telemetry
pull, and both cost a file walk per episode to read, so the population view
that wants "tokens and turns across 1000 episodes" could not have them.

:func:`summarize` collapses the pair into ~10 scalars plus a call
histogram and the declared tools, cheap enough to ride the row into
``episodes.jsonl``.

WHICH FILE OWNS WHICH NUMBER — one source per quantity, never two:

  TRANSCRIPT owns turns, requests, tokens, the tool calls the model
    emitted, and the DECLARED tools each ``params`` record carries.

  NET METER owns ``reasoning_tokens``. Whether it rides the wire is a
    backend fact, not a constant: measured over the pulled tree, 1055 of
    1761 episodes carry ``usage.reasoning_tokens`` (sglang) and the rest
    do not (vLLM 0.21 leaves it unset and rolls the span into
    ``completion_tokens``). ``net_meter.reasoning_tokens_of`` takes the
    wire value where it exists and re-tokenizes ``message.reasoning``
    where it does not, so the meter is a strict superset of the
    transcript — which is why summing the transcript's copy here would be
    the wrong number on half the fleet.

    ``reasoning_tokens`` is a SLICE of ``completion_tokens``, not an
    addition — the two overlap by design.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from . import net_meter

log = logging.getLogger(__name__)

#: A tool name the model generated rather than selected. Measured once in
#: 12,679 calls on one sglang serve: a name 163 characters long carrying
#: leaked prose ("act...`. Let me fix this. Actually, looking at the previous
#: successful call…"). Real data, not a parse failure — but binning it
#: verbatim puts a paragraph in the axis, and dropping it hides a generation
#: failure worth counting. It gets its own bucket.
MALFORMED = "«malformed»"

_TOOL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,39}\Z")


def _tool_name(raw: Any) -> str:
    name = str(raw or "")
    return name if _TOOL_NAME.match(name) else MALFORMED


def _as_dict(v: Any) -> dict:
    """Model-emitted JSON is untrusted in SHAPE, not just in presence.

    ``x.get(k) or {}`` guards a missing or null field and nothing else: a
    field declared as an object arrives as a string often enough to matter,
    and the next ``.get`` on it raises. This runs inside the episode,
    behind a catch-all, so an unguarded raise would return the zero shape
    and lose that episode's telemetry.
    """
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
    """The zero shape. Written verbatim when the episode kept no transcript,
    so a reader never has to distinguish "no block" from "empty block"."""
    return {
        "turns": 0,
        "requests": 0,
        "unanswered": 0,
        "prompt_tokens": 0,
        # Largest single-request prompt — peak context occupancy. The sums
        # above measure spend; this measures how full the window ever got.
        "prompt_tokens_max": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "latency_s_sum": 0.0,
        "latency_s_max": 0.0,
        "calls": {},
        "declared": {"tools": []},
    }


def _scan_transcript(path: Path, out: dict) -> None:
    """Fold one transcript into ``out``.

    A raw line scan: it never reconstructs a request's full message list
    from the per-thread deltas. Only ``params``
    (the declared vocabulary), ``request`` (timestamps, for latency and
    the unanswered count) and ``response`` matter.
    """
    started: dict[str, float] = {}
    with path.open(errors="replace") as fh:
        for line in fh:
            # Cheap reject before the JSON parse.
            if '"kind":' not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                # A truncated final line is a live writer mid-append, not
                # an error.
                continue
            rec = _as_dict(rec)
            kind = rec.get("kind")
            if kind == "params":
                for t in _declared_tools(_as_dict(rec.get("params"))):
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

    # Requests still open at EOF never got a response — a crash, a timeout,
    # or a run killed mid-call. Counting them is why ``requests`` is read at
    # all: ``turns`` alone cannot tell a clean episode from a truncated one.
    out["unanswered"] = len(started)


def summarize(flight_dir: Path) -> dict:
    """One episode's LLM activity from its flight dir.

    Returns :func:`blank` when the episode kept no transcript. NEVER
    raises: this runs on the episode's way out, and a summary is not
    worth losing a flown episode over.
    """
    out = blank()
    transcript = Path(flight_dir) / "llm" / "transcript.jsonl"
    try:
        _scan_transcript(transcript, out)
    except OSError:
        return blank()
    except Exception as exc:  # noqa: BLE001 — never fail the episode
        log.warning("flight summary scan failed at %s: %s", transcript, exc)
        return blank()

    try:
        meter = net_meter.summarize_window(Path(flight_dir) / "net_meter.jsonl")
        out["reasoning_tokens"] = int(meter.get("total_reasoning_tokens") or 0)
    except Exception as exc:  # noqa: BLE001 — never fail the episode
        log.warning("flight summary meter read failed at %s: %s", flight_dir, exc)

    out["latency_s_sum"] = round(out["latency_s_sum"], 3)
    out["latency_s_max"] = round(out["latency_s_max"], 3)
    return out
