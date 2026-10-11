"""One episode's LLM activity, reduced to a row-sized block.

The flight dir's ``llm/transcript.jsonl`` records every round-trip
(:mod:`maps.llm.http` writes it). It does not survive a rows-only telemetry
pull, and it costs a file walk per episode to read, so the population view
that wants "tokens and turns across 1000 episodes" could not have it.

:func:`summarize` collapses it into ~10 scalars plus a call histogram and
the declared tools, cheap enough to ride the row into ``episodes.jsonl``:
turns, requests, tokens, the tool calls the model emitted, and the DECLARED
tools each ``params`` record carries.

``reasoning_tokens`` is not summed off ``usage``: whether
``usage.reasoning_tokens`` rides the wire is a backend fact (1055 of 1761
pulled episodes carry it, from sglang; vLLM 0.21 leaves it unset and rolls
the span into ``completion_tokens``). :func:`reasoning_tokens_of` takes the
wire value where it exists and re-tokenizes the response's thinking span
where it does not; a hosted OpenAI model returns no span, so its count is
the provider's ``usage`` meter. It is a SLICE of ``completion_tokens``, not
an addition.
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


def reasoning_tokens_of(response_json: dict, *, model: str | None = None) -> int:
    """Extract the reasoning-span token count from a chat-completions response.

    ``model`` is the requested model. On an OpenAI-wire card
    (``maps.llm.model_cards``) the span never comes back by design, so the
    rule below would read every response as zero thinking: the count there
    is ``usage.completion_tokens_details.reasoning_tokens``, the provider's
    own meter of the tokens it billed as reasoning.

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

    The count is a slice of ``usage.completion_tokens``, which includes
    the reasoning tokens; the two overlap by design.
    """
    if speaks_openai(model):
        details = _as_dict(_as_dict(response_json.get("usage")).get("completion_tokens_details"))
        return _usage_int({"usage": details}, "reasoning_tokens")
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


def _usage_int(response_json: dict, key: str) -> int:
    usage = response_json.get("usage") if isinstance(response_json, dict) else None
    if not isinstance(usage, dict):
        return 0
    val = usage.get(key)
    try:
        return int(val) if val is not None else 0
    except (TypeError, ValueError):
        return 0


def _scan_transcript(path: Path, out: dict) -> None:
    """Fold one transcript into ``out``.

    A raw line scan: it never reconstructs a request's full message list
    from the deltas. Only ``params``
    (the declared vocabulary), ``request`` (timestamps, for latency and
    the unanswered count) and ``response`` matter.
    """
    started: dict[str, float] = {}
    model: str | None = None
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

    out["latency_s_sum"] = round(out["latency_s_sum"], 3)
    out["latency_s_max"] = round(out["latency_s_max"], 3)
    return out
