"""One episode's LLM activity, reduced to a row-sized block.

The flight dir already holds everything: ``llm/transcript.jsonl`` records
every round-trip (:mod:`maps.llm.http` writes it) and ``net_meter.jsonl``
records per-call token/latency accounting (:mod:`maps.llm.net_meter`).
Both are written on every episode. Neither survives a rows-only telemetry
pull, and both cost a file walk per episode to read, so the population view
that wants "tokens and turns across 1000 episodes" could not have them.

:func:`summarize` collapses the pair into ~20 scalars plus three small
histograms, cheap enough to ride the row into ``episodes.jsonl``.

WHICH FILE OWNS WHICH NUMBER — one source per quantity, never two:

  TRANSCRIPT owns turns, requests, tokens, and everything the model
    emitted (tool calls, program ops, op args) plus the DECLARED
    vocabulary each ``params`` record carries. It covers cache-replay
    calls, which is why it and not the meter owns the counts.

  NET METER owns ``reasoning_tokens``. Whether it rides the wire is a
    backend fact, not a constant: measured over the pulled tree, 1055 of
    1761 episodes carry ``usage.reasoning_tokens`` (sglang) and the rest
    do not (vLLM 0.21 leaves it unset and rolls the span into
    ``completion_tokens``). ``net_meter.reasoning_tokens_of`` takes the
    wire value where it exists and re-tokenizes ``message.reasoning``
    where it does not, so the meter is a strict superset of the
    transcript — which is why summing the transcript's copy here would be
    the wrong number on half the fleet. It is skipped on the cache-replay
    path, so it reads 0 on a ``--replay-llm`` run while the counts do not.

    ``reasoning_tokens`` is a SLICE of ``completion_tokens``, not an
    addition — the two overlap by design.

DECLARED vs USED is the interesting set, and only the transcript can
state it: ``find`` declares a ``color`` arg, and whether the agent ever
supplies one is load-bearing (CLAUDE.md: colour is ORDERING-only).
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

#: The op list lives at ``analyze.parameters.program.ops.items``, as a
#: variant list when the schema discriminates on ``op``.
_ANALYZE = "analyze"


def _tool_name(raw: Any) -> str:
    name = str(raw or "")
    return name if _TOOL_NAME.match(name) else MALFORMED


def _as_dict(v: Any) -> dict:
    """Model-emitted JSON is untrusted in SHAPE, not just in presence.

    ``x.get(k) or {}`` guards a missing or null field and nothing else: a
    field declared as an object arrives as a string often enough to matter,
    and the next ``.get`` on it raises. Measured on the laptop hub — an
    ``analyze`` call whose ``program`` was the bare string form raised
    ``AttributeError: 'str' object has no attribute 'get'`` and took the
    WHOLE fleet listing to a 500 over one episode.

    Here the blast radius is different and worse: this runs inside the
    episode, behind a catch-all, so an unguarded raise would silently
    return the zero shape and lose that episode's telemetry with nothing
    logged as missing.
    """
    return v if isinstance(v, dict) else {}


def _variants(item: dict) -> list[dict]:
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(item.get(key), list):
            return [v for v in item[key] if isinstance(v, dict)]
    return [item] if item else []


def _declared(params: dict) -> dict:
    """``{"tools": [name...], "ops": {op: [arg...]}}`` from one params record."""
    tools: list[str] = []
    ops: dict[str, list[str]] = {}
    for t in params.get("tools") or []:
        fn = _as_dict(_as_dict(t).get("function"))
        name = fn.get("name")
        if not name:
            continue
        if name not in tools:
            tools.append(name)
        if name != _ANALYZE:
            continue
        props = _as_dict(_as_dict(fn.get("parameters")).get("properties"))
        items = _as_dict(
            _as_dict(_as_dict(props.get("program")).get("properties")).get("ops")
        ).get("items")
        for v in _variants(_as_dict(items)):
            vprops = _as_dict(v.get("properties"))
            op = _as_dict(vprops.get("op")).get("const")
            if not op:
                continue
            args = _as_dict(_as_dict(vprops.get("args")).get("properties"))
            ops[op] = sorted(args)
    return {"tools": tools, "ops": ops}


def _merge_declared(into: dict, more: dict) -> None:
    for t in more.get("tools") or []:
        if t not in into["tools"]:
            into["tools"].append(t)
    for op, args in (more.get("ops") or {}).items():
        into["ops"].setdefault(op, [])
        for a in args:
            if a not in into["ops"][op]:
                into["ops"][op].append(a)


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
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "latency_s_sum": 0.0,
        "latency_s_max": 0.0,
        "calls": {},
        "ops": {},
        "op_args": {},
        "declared": {"tools": [], "ops": {}},
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
                _merge_declared(out["declared"], _declared(_as_dict(rec.get("params"))))
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
            # vLLM reports the prefix-cache hit under ``prompt_tokens_details``;
            # some backends put it flat. Same quantity, two spellings.
            details = _as_dict(usage.get("prompt_tokens_details"))
            cached = details.get("cached_tokens", usage.get("cached_tokens"))
            if isinstance(cached, (int, float)):
                out["cached_tokens"] += int(cached)

            message = _as_dict(
                _as_dict((body.get("choices") or [{}])[0]).get("message")
            )
            for tc in message.get("tool_calls") or []:
                fn = _as_dict(_as_dict(tc).get("function"))
                name = _tool_name(fn.get("name"))
                _bump(out["calls"], name)
                if name != _ANALYZE:
                    continue
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    continue
                for op in _as_dict(_as_dict(args).get("program")).get("ops") or []:
                    if not isinstance(op, dict):
                        continue
                    op_name = op.get("op")
                    _bump(out["ops"], op_name)
                    if not op_name:
                        continue
                    per = out["op_args"].setdefault(op_name, {})
                    for a, val in _as_dict(op.get("args")).items():
                        # An arg present but null is not a use: the model
                        # emitted the key and declined the value, which is
                        # exactly the "ignored" case being measured.
                        if val is not None:
                            _bump(per, a)

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
