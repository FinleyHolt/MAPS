"""Deterministic clear-to-census autocompact for the reasoning engine.

When the assembled prompt for the persistent reasoner would exceed the
served window's input budget, ``compact_transcript`` sheds stale image
crops and rewrites every SUPERSEDED tool-result dump (all but the most
recent, the live view) to its fids+census stub form, returning a
message list that fits. Tool dumps are re-derivable views over the live
world-state graph, so a stub loses nothing the agent cannot re-query, and
the stub says so. No LLM call is made on this path.

The analyze surface puts the dump at the top level of its ``analyze``
result; the mission surface wraps the same dump under ``result`` in its
``run`` result. Steps select tool results by SHAPE, never by tool name.

Steps, in order, re-estimating after each; the cheapest loss goes first and
re-derivable text goes before pixels:

  1. FOLD the oldest spent turns into one script block: the agent's own code
     verbatim plus the fids+census of what those turns found. The only step
     that changes message count; it leaves [system, instruction, block, live
     tail]. Bounded by the shortfall to the low-water mark.
  2. Clear stale dumps to fids+census oldest-first, keeping a compact per-fid
     color table (the color signal lives inside the ``rows`` render the trim
     drops). Stops at the low-water mark; the newest dump is never cleared
     here.
  3. Still over: drop the color tables from the stubs, oldest-first.
  4. Still over: degrade remaining tool dumps largest-first, the live dump
     last, to plain fids+census. Keeps every candidate fid.
  5. Still over: strip the findings from the block THIS fire created,
     keeping the code. Only the newest block: an earlier fire's block is
     already served prefix.
  6. Still over: evict stale image crops oldest-first, never the current
     turn's. Text-only transcripts skip it.
  7. Still over: thin the CURRENT turn's crops oldest-first, keeping the
     newest.
  8. Still over: raise ``CompactionInfeasible``.

APPEND-ONLY ACROSS FIRES. A block is inserted where its span was, and no later
fire rewrites or absorbs an earlier one, so the served prefix extends rather
than moves.

Below-budget calls return the input list itself (same ``id()``), keeping
the prefix cache byte-identical. The post-compaction sweep checks
``tool_call_id`` closure and coverage and raises
``CompactionInvariantError`` on either failure.

The budget is CALLER-SUPPLIED: ``target_input_tokens`` comes from the
server-discovered ``EndpointSpec.input_budget`` (``maps.llm.endpoint_spec``).
A prompt that cannot be measured (the tokenizer cannot load or its chat
template cannot render the transcript) raises rather than being sent
unmeasured. The tokenizer name comes from ``SEARCH_COMPACT_TOKENIZER_NAME``
(``maps.llm.tokenizer``).
"""

from __future__ import annotations

import json
import logging
import time

from maps.llm.tokenizer import _tokenizer
from maps.llm.vision_content import image_tokens_in, image_tokens_of
from maps.search.analyze_scene.geom_program import join_hints

log = logging.getLogger(__name__)


# At or below this many messages the transcript passes through without a
# token estimate.
_CHEAP_GATE_MSG_COUNT = 4

# The low-water mark sits this far below target, so a fire buys turns of
# growth before the next one (each fire is one prefix-cache invalidation).
# Roughly half a dense analyze dump.
_CLEAR_HEADROOM_TOKENS = 16384


class CompactionError(RuntimeError):
    """Base class for autocompact failures."""


class CompactionInfeasible(CompactionError):
    """No strategy can bring the prompt under target_input_tokens."""


class CompactionInvariantError(CompactionError):
    """Post-compaction sweep found a tool_call_id without its counterpart:
    a result with no earlier call, or a call with no result."""


# What an ``image_url`` part flattens to for tokenization, so a crop's base64
# never reaches the tokenizer. The placeholder is not the price:
# ``_estimate_input_tokens`` adds ``image_tokens_in`` back after the flatten.
_IMAGE_PART_PLACEHOLDER = "[image crop]"


def _flatten_multimodal_content(content: object) -> str:
    """Collapse an OpenAI multi-part content list to a compact text string,
    replacing each ``image_url`` part with ``_IMAGE_PART_PLACEHOLDER``. A plain
    string returns unchanged; a non-list/str falls back to ``json.dumps`` (dict
    tool results keep their prior serialization)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, default=str)
    parts: list[str] = []
    for p in content:
        if isinstance(p, dict):
            t = p.get("type")
            if t == "text":
                parts.append(str(p.get("text") or ""))
            elif t == "image_url" or "image" in str(t):
                parts.append(_IMAGE_PART_PLACEHOLDER)
            else:
                parts.append(str(p))
        else:
            parts.append(str(p))
    return " ".join(x for x in parts if x)


def _messages_for_tokenization(messages: list[dict]) -> list[dict]:
    """Return a copy of ``messages`` with tool_calls.function.arguments
    parsed to a dict AND multimodal list content flattened to text.

    ``arguments`` is a JSON string on the wire, and the chat template's
    ``tool_call.arguments | items`` filter needs a mapping; the server parses
    it before rendering, so the local estimate does the same. The original
    list is unchanged. Multi-part (vision) content flattens through
    ``_flatten_multimodal_content``.
    """
    out: list[dict] = []
    for m in messages:
        m2 = m
        content = m.get("content")
        if isinstance(content, list):
            m2 = {**m2, "content": _flatten_multimodal_content(content)}
        tcs = m.get("tool_calls")
        if tcs:
            new_tcs: list[dict] = []
            for tc in tcs:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        parsed = {}
                    new_fn = dict(fn)
                    new_fn["arguments"] = parsed
                    new_tcs.append({**tc, "function": new_fn})
                else:
                    new_tcs.append(tc)
            m2 = {**m2, "tool_calls": new_tcs}
        out.append(m2)
    return out


#: What replaces an evicted crop, so the model reads a removal rather than an
#: image it failed to see.
_EVICTED_CROP_TEXT = "[earlier crop evicted from context]"


def _evict_oldest_crops(
    messages: list[dict], *, shed_tokens: int
) -> tuple[list[dict], int, int]:
    """Drop the OLDEST image crops until ``shed_tokens`` are freed.

    Returns ``(messages, n_evicted, tokens_freed)``; the input list is never
    mutated and is returned unchanged when nothing is evicted. The LAST
    message carrying images (the current turn's crops) is never touched.
    Each crop's cost comes from its own pixels, so no re-tokenization runs
    per eviction.
    """
    last_with_images = -1
    for i, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in content
        ):
            last_with_images = i
    if last_with_images < 0:
        return messages, 0, 0

    out = list(messages)
    freed = evicted = 0
    for i in range(last_with_images):
        if freed >= shed_tokens:
            break
        content = out[i].get("content")
        if not isinstance(content, list):
            continue
        parts: list[dict] = []
        touched = False
        for part in content:
            cost = image_tokens_of(part)
            if cost and freed < shed_tokens:
                freed += cost
                evicted += 1
                touched = True
                parts.append({"type": "text", "text": _EVICTED_CROP_TEXT})
            else:
                parts.append(part)
        if touched:
            out[i] = {**out[i], "content": parts}
    return (out, evicted, freed) if evicted else (messages, 0, 0)


#: Marks a span the compactor replaced with the agent's own code; a later fire
#: recognises it and never folds it again.
_SCRIPT_BLOCK_HEADER = "[script so far]"

#: Separates the folded code from the census of what those turns found; the
#: writer and the step that strips it share it.
_FINDINGS_SEP = "\nwhat those turns found:\n"

def _protected_head(messages: list[dict]) -> int:
    """How many head messages are never droppable: through the first user
    message (the operator instruction), after the system prompt when one is
    served."""
    first_user = next((i for i, m in enumerate(messages) if m.get("role") == "user"), 0)
    return first_user + 1

#: Turn units the fold keeps verbatim at the tail, where the model is
#: mid-derivation.
_KEEP_TAIL_UNITS = 2


def _run_code_of(message: dict) -> list[str]:
    """The code an assistant turn ran, verbatim, one entry per tool call.

    The mission surface's one tool, ``run``, takes ``{"code": ...}``. A call
    whose arguments do not parse, or carry no code, contributes its raw
    argument string, so every action stays on the record.
    """
    out: list[str] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw = fn.get("arguments")
        if not isinstance(raw, str):
            continue
        try:
            code = (json.loads(raw) or {}).get("code")
        except (ValueError, TypeError):
            code = None
        out.append(code if isinstance(code, str) and code.strip() else raw)
    return out


def _turn_units(messages: list[dict]) -> list[tuple[int, int]]:
    """Contiguous ``[assistant-with-tool_calls, *its tool results]`` spans.

    A unit is the atom of dropping, so a call and its results always move
    together.
    """
    units: list[tuple[int, int]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            ids = {tc.get("id") for tc in msg["tool_calls"] if tc.get("id")}
            j = i + 1
            answered: set[str] = set()
            while (j < len(messages) and messages[j].get("role") == "tool"
                   and messages[j].get("tool_call_id") in ids):
                answered.add(messages[j]["tool_call_id"])
                j += 1
            # Only a unit whose results are all present and contiguous is
            # foldable: folding a short run would orphan the results that
            # stayed. The later steps still reduce those messages.
            if ids and answered == ids:
                units.append((i, j))
                i = j
                continue
            i = j
            continue
        i += 1
    return units


def _folded_findings(messages: list[dict], span: tuple[int, int]) -> list[str]:
    """The fids+census stubs of what a folded span FOUND.

    The code says what the agent asked; this says what came back, through the
    same degrade the stubbing steps apply, so a folded turn carries exactly
    what a stubbed one would.
    """
    out: list[str] = []
    for i in range(span[0], span[1]):
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        stub = _degrade_tool_result_content(content, keep_color=True)
        # None means nothing left to trim, which an already-stubbed dump
        # returns; its content is then carried as it stands, fids included.
        carried = stub if stub else (content if isinstance(content, str) else None)
        if carried:
            out.append(carried)
    return out


def _script_block(sources: list[str], n_turns: int,
                  findings: list[str] | None = None) -> dict:
    """The artifact: what the agent DID, in the agent's own bytes.

    Verbatim code is deterministic, costs no summariser call, and loses
    nothing in a retelling. The interpreter namespace persists for the whole
    mission (``python_surface``), so every name the folded turns bound is
    still live.
    """
    body = "\n".join(sources)
    found = "\n".join(findings or [])
    return {
        "role": "assistant",
        "content": (
            f"{_SCRIPT_BLOCK_HEADER} ({n_turns} earlier turns folded; the names "
            f"they bound are still live)\n```python\n{body}\n```"
            + (f"{_FINDINGS_SEP}{found}" if found else "")
        ),
    }


def _newest_script_block(messages: list[dict]) -> int | None:
    """Index of the LAST script block, or None. The only one a fire may edit."""
    for i in range(len(messages) - 1, -1, -1):
        if _is_script_block(messages[i]):
            return i
    return None


def _is_script_block(message: dict) -> bool:
    content = message.get("content")
    return (message.get("role") == "assistant" and isinstance(content, str)
            and content.startswith(_SCRIPT_BLOCK_HEADER))


def _drop_spent_turns(
    messages: list[dict], *, shed_tokens: int, model: str,
    tools: list[dict] | None = None,
) -> tuple[list[dict], int, int]:
    """Fold the OLDEST spent turns into one script block until ``shed_tokens``.

    Returns ``(messages, n_turns_folded, tokens_freed)``; the input list is
    never mutated and is returned unchanged when nothing is folded. Folding
    stops once ``shed_tokens`` are freed (the caller passes the shortfall to
    the low-water mark).

    The block is INSERTED where the folded span was, and an existing block is
    never rewritten or absorbed, so earlier blocks stay byte-identical and the
    served prefix extends instead of moving.
    """
    units = _turn_units(messages)
    head = _protected_head(messages)
    droppable = [
        (a, b) for (a, b) in units
        if a >= head and not _is_script_block(messages[a])
    ]
    if len(droppable) <= _KEEP_TAIL_UNITS:
        return messages, 0, 0
    candidates = droppable[:-_KEEP_TAIL_UNITS]

    before = _estimate_input_tokens(messages, model=model, tools=tools)

    folded: list[tuple[int, int]] = []
    sources: list[str] = []
    findings: list[str] = []
    out = messages
    freed = 0
    for a, b in candidates:
        folded.append((a, b))
        sources.extend(_run_code_of(messages[a]))
        findings.extend(_folded_findings(messages, (a, b)))
        head = messages[:folded[0][0]]
        tail = [m for i, m in enumerate(messages)
                if i >= folded[0][0] and not any(x <= i < y for x, y in folded)]
        out = head + [_script_block(sources, len(folded), findings)] + tail
        after = _estimate_input_tokens(out, model=model, tools=tools)
        freed = before - after
        # Bounded by the shortfall to the low-water mark: deep enough to buy
        # quiet turns, never the whole history for a marginal overflow.
        if freed >= shed_tokens:
            break
    if not folded or freed <= 0:
        return messages, 0, 0
    return out, len(folded), freed


def _evict_current_turn_crops(
    messages: list[dict], *, shed_tokens: int
) -> tuple[list[dict], int, int]:
    """Last resort: thin the CURRENT turn's own crops, oldest-first.

    Reached when the newest vision message alone overruns the window. The
    NEWEST crop always survives, so the turn never degenerates to a
    text-only prompt.
    """
    last_with_images = -1
    for i, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in content
        ):
            last_with_images = i
    if last_with_images < 0:
        return messages, 0, 0

    content = messages[last_with_images]["content"]
    crop_at = [
        j for j, p in enumerate(content)
        if isinstance(p, dict) and p.get("type") == "image_url"
    ]
    freed = evicted = 0
    parts = list(content)
    for j in crop_at[:-1]:  # never the newest
        if freed >= shed_tokens:
            break
        freed += image_tokens_of(parts[j])
        evicted += 1
        parts[j] = {"type": "text", "text": _EVICTED_CROP_TEXT}
    if not evicted:
        return messages, 0, 0
    out = list(messages)
    out[last_with_images] = {**out[last_with_images], "content": parts}
    return out, evicted, freed


def _estimate_input_tokens(
    messages: list[dict],
    *,
    model: str,
    tools: list[dict] | None = None,
) -> int:
    """Estimate the chat-templated input-token count vLLM will see.

    Uses ``tokenizer.apply_chat_template(... add_generation_prompt=True)``
    over ``_messages_for_tokenization``. A tokenizer that cannot load or a
    template that cannot render the transcript raises; discovery renders a
    transcript of every turn kind first, so neither happens mid-run.
    """
    tok = _tokenizer(model)
    text = tok.apply_chat_template(
        _messages_for_tokenization(messages),
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(text, str):
        text = "".join(text)
    ids = tok.encode(text, add_special_tokens=False)
    # The flatten replaced every crop with a placeholder; the server bills the
    # pixels against the same window as the text.
    return len(ids) + image_tokens_in(messages)


def _validate_invariants(messages: list[dict]) -> None:
    """Sweep ``messages`` for tool-protocol violations: CLOSURE (every tool
    result has an earlier call) and COVERAGE (every call has a result). Step
    1 drops messages, so both can arise; the server would answer either with
    a 400 mid-episode.
    """
    seen_ids: set[str] = set()
    answered: set[str] = set()
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                tc_id = tc.get("id")
                if tc_id:
                    seen_ids.add(tc_id)
            continue
        if role == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id not in seen_ids:
                raise CompactionInvariantError(
                    f"orphan tool message: tool_call_id={tc_id!r} has no "
                    f"matching earlier assistant tool_calls entry"
                )
            answered.add(tc_id)
    unanswered = seen_ids - answered
    if unanswered:
        raise CompactionInvariantError(
            f"unanswered tool_call(s): {sorted(unanswered)!r} announced with no "
            f"tool result -- a turn was dropped without its results, and the "
            f"server would 400 on this"
        )


# The verbose render keys of an op head (the markdown grid and its units
# line). The ``fids`` list carries the rank order and
# ``census``/``len``/``named``/``columns`` the census and schema, so dropping
# these trims a dump to fids+census without removing any candidate fid.
_TRIM_RENDER_KEYS = ("rows", "units", "grid", "legend")


def _iter_op_heads(obj: dict):
    """Yield the per-op head dicts wherever the payload nests them.

    The analyze payload is ``{"results": {"rN": head, ...}}``; a bare head
    or a top-level ``{op_ref: head}`` map also occur. The mission surface
    wraps the same dump under ``result``:
    ``{"said": ..., "result": {"results": {"near_res": head}}}``.
    """
    if any(k in obj for k in _TRIM_RENDER_KEYS):
        yield obj
        return
    wrapped = obj.get("result")
    if isinstance(wrapped, dict) and not isinstance(obj.get("results"), dict):
        yield from _iter_op_heads(wrapped)
        return
    inner = obj.get("results") if isinstance(obj.get("results"), dict) else obj
    for v in inner.values():
        if isinstance(v, dict) and any(k in v for k in _TRIM_RENDER_KEYS):
            yield v


def _rows_color_entries(rows: str) -> list[str]:
    """``fid=color[@conf]`` entries parsed from one ``rows`` markdown grid:
    the color cell (with its ``?`` uncertainty mark) and ``color_conf``,
    which live inside the ``rows`` render a trim drops."""
    entries: list[str] = []
    fid_i = col_i = conf_i = None
    for line in rows.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if "fid" in cells:  # a header row (repeats down the grid)
            fid_i = cells.index("fid")
            col_i = cells.index("color") if "color" in cells else None
            conf_i = cells.index("color_conf") if "color_conf" in cells else None
            continue
        if fid_i is None or col_i is None or fid_i >= len(cells):
            continue
        fid, col = cells[fid_i], cells[col_i] if col_i < len(cells) else "-"
        # skip blanks, "-" absent-cells AND the "---" markdown separator row
        if not fid.strip("-") or not col.strip("-"):
            continue
        conf = cells[conf_i] if conf_i is not None and conf_i < len(cells) else ""
        entries.append(f"{fid}={col}" + (f"@{conf}" if conf and conf != "-" else ""))
    return entries


def _degrade_tool_result_content(
    content: str, *, keep_color: bool = False
) -> str | None:
    """Rewrite ONE tool-result JSON string to its fids+census stub.

    Drops only the render keys from every op head; ``fids`` and every
    scalar/census key survive. The stub's hint says the graph is live and
    re-running analyze recovers the rows. ``keep_color`` keeps the per-fid
    color table parsed from the dropped ``rows``. Returns the new JSON
    string, or ``None`` when nothing is trimmable or the content is not the
    expected shape."""
    try:
        obj = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    changed = False
    for head in _iter_op_heads(obj):
        present = [k for k in _TRIM_RENDER_KEYS if k in head]
        if not present:
            continue
        colors: list[str] = []
        if keep_color and isinstance(head.get("rows"), str):
            colors = _rows_color_entries(head["rows"])
        for k in present:
            head.pop(k, None)
        if colors:
            head["colors"] = "; ".join(colors)
        changed = True
        # rN refs from an earlier round are dead, so the stub promises only
        # that the graph is re-queryable.
        note = ("rows trimmed to fids+census for token budget (fids "
                "preserved); the world-state graph is live — re-run analyze "
                "to re-derive the full rows.")
        head["hint"] = join_hints(head.get("hint"), note)
    if not changed:
        return None
    return json.dumps(obj, default=str)


def _degrade_oversized_tool_dumps(
    messages: list[dict],
    *,
    model: str,
    tools: list[dict] | None,
    target_input_tokens: int,
) -> tuple[list[dict], int]:
    """Trim the largest tool-result dumps to fids+census until the
    transcript fits ``target_input_tokens`` (or nothing more is trimmable).

    Returns ``(possibly-degraded messages, n_trimmed)``. A degraded message
    keeps its role / tool_call_id / name and a valid JSON content string, and
    every candidate fid. A message not in the expected shape stays verbatim.
    Biggest dumps trim first, so the fewest trims restore feasibility, except
    the NEWEST dump, the round's live evidence, which trims last."""
    idxs = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") == "tool"
        and _carries_dump(m.get("content"))
    ]
    newest = max(idxs) if idxs else None
    idxs.sort(key=lambda i: len(messages[i]["content"]), reverse=True)
    if newest is not None:
        idxs = [i for i in idxs if i != newest] + [newest]
    out = list(messages)
    n_trimmed = 0
    for i in idxs:
        trimmed = _degrade_tool_result_content(out[i]["content"])
        if trimmed is None:
            continue
        new_msg = dict(out[i])
        new_msg["content"] = trimmed
        out[i] = new_msg
        n_trimmed += 1
        est = _estimate_input_tokens(out, model=model, tools=tools)
        if est <= target_input_tokens:
            break
    return out, n_trimmed


def _carries_dump(content: object) -> bool:
    """Whether a tool-result body holds a head this module can trim.

    The SHAPE is the test, never the tool name: the analyze surface names its
    tool ``analyze`` and a mission names it ``run``, and callers protect the
    newest DUMP, so a non-dump result must not count as one.
    """
    if not isinstance(content, str):
        return False
    try:
        obj = json.loads(content)
    except (TypeError, ValueError):
        return False
    return isinstance(obj, dict) and any(_iter_op_heads(obj))


def _dump_result_indices(messages: list[dict]) -> list[int]:
    """Indices of the tool-result messages carrying re-derivable dumps."""
    return [
        i for i, m in enumerate(messages)
        if m.get("role") == "tool" and _carries_dump(m.get("content"))
    ]


def compact_transcript(
    messages: list[dict],
    *,
    target_input_tokens: int,
    model: str,
    tools: list[dict] | None = None,
    site: str = "unknown",
) -> list[dict]:
    """Return a possibly-compacted view of ``messages`` whose
    chat-templated input-token count is <= ``target_input_tokens``.

    Below budget: returns ``messages`` itself (same ``id()``). Above budget:
    the steps of the module docstring, in order. No LLM call is made.

    ``site`` identifies the call site for log lines (e.g.
    ``"reasoner.step_tool_call"``).

    Raises ``CompactionInfeasible`` when every step together cannot reach
    the target, and ``CompactionInvariantError`` if the post-compaction
    sweep finds a tool-protocol violation.
    """
    # 1. Cheap gate.
    if len(messages) <= _CHEAP_GATE_MSG_COUNT:
        return messages

    # 2. Token estimate.
    input_tokens = _estimate_input_tokens(messages, model=model, tools=tools)

    # 3. Budget check.
    if input_tokens <= target_input_tokens:
        return messages

    t0 = time.time()

    # Text goes before pixels: a stubbed dump costs a re-query, an evicted
    # crop costs a view the model cannot get back.
    #
    # 4. STEP 1 -- fold the spent turns into one script block. First, because
    # every other step rewrites content in place at an old index and leaves
    # the rewritten body to re-prefill behind the break; a fold leaves only
    # head+block+tail. Step 5 gives the block its own degrade.
    #
    # The low-water mark is never above target (tiny targets) and floored at
    # 8000, the reasoner's minimum viable input budget.
    low_water = max(target_input_tokens - _CLEAR_HEADROOM_TOKENS,
                    min(8000, target_input_tokens))
    folded_turns = 0
    out = messages
    output_tokens = input_tokens
    if input_tokens > target_input_tokens:
        out, folded_turns, freed_fold = _drop_spent_turns(
            out, shed_tokens=input_tokens - low_water, model=model, tools=tools,
        )
        if folded_turns:
            output_tokens = _estimate_input_tokens(out, model=model, tools=tools)
            log.info(
                "compaction[%s] folded %d spent turn(s) into a script block, "
                "freed ~%d tok -> %s/%d",
                site, folded_turns, freed_fold, output_tokens,
                target_input_tokens,
            )

    # 5. STEP 2 -- clear stale dumps (never the newest, the live view) to
    # their fids+census stubs oldest-first, color tables kept, stopping at the
    # low-water mark so the most recent dumps stay verbatim when the budget
    # allows.
    idxs = _dump_result_indices(out)
    stale = idxs[:-1]
    base = out
    out = list(out)
    cleared: list[int] = []
    colored: list[int] = []
    for i in stale:
        # Checked before the write, so a fold that already landed under the
        # low-water mark stubs nothing.
        if output_tokens <= low_water:
            break
        stub = _degrade_tool_result_content(out[i]["content"], keep_color=True)
        if stub is None:
            continue  # already stubbed / not the expected shape
        new_msg = dict(out[i])
        new_msg["content"] = stub
        out[i] = new_msg
        cleared.append(i)
        if '"colors":' in stub:
            colored.append(i)
        output_tokens = _estimate_input_tokens(out, model=model, tools=tools)
        if output_tokens <= low_water:
            break
    color_tables_kept = len(colored)

    # 6. STEP 3 -- still over target: strip the per-fid color tables from the
    # stubs oldest-first.
    if output_tokens > target_input_tokens:
        for i in colored:
            plain = _degrade_tool_result_content(
                base[i]["content"], keep_color=False)
            if plain is None:
                continue
            new_msg = dict(out[i])
            new_msg["content"] = plain
            out[i] = new_msg
            color_tables_kept -= 1
            output_tokens = _estimate_input_tokens(out, model=model, tools=tools)
            if output_tokens <= target_input_tokens:
                break

    # 7. STEP 4 -- degrade the remaining dumps largest-first to plain
    # fids+census, the live one last (_degrade_oversized_tool_dumps).
    degraded_dumps = 0
    if output_tokens > target_input_tokens:
        out, degraded_dumps = _degrade_oversized_tool_dumps(
            out,
            model=model,
            tools=tools,
            target_input_tokens=target_input_tokens,
        )
        if degraded_dumps:
            output_tokens = _estimate_input_tokens(out, model=model, tools=tools)

    # 8. STEP 5 -- strip the findings from the script block, keeping the
    # code. A block is not a tool result, so no step above reduces it.
    stripped_findings = False
    if output_tokens > target_input_tokens and folded_turns:
        # Only the block this fire created: earlier blocks are served prefix.
        i = _newest_script_block(out)
        if i is not None:
            # rsplit: findings are single-line json, so the separator cannot
            # occur inside them, but the agent's own code CAN contain it and
            # split()[0] would truncate the code at the model's own text.
            body = out[i]["content"].rsplit(_FINDINGS_SEP, 1)[0]
            if body != out[i]["content"]:
                out = list(out)
                out[i] = {**out[i], "content": body}
                stripped_findings = True
        if stripped_findings:
            output_tokens = _estimate_input_tokens(out, model=model, tools=tools)
            log.info("compaction[%s] stripped block findings -> %s/%d",
                     site, output_tokens, target_input_tokens)

    # 9. STEP 6 -- evict STALE image crops, oldest-first, never the message
    # carrying the current turn's. Text-only transcripts have nothing to
    # evict.
    n_evicted = 0
    if output_tokens > target_input_tokens:
        out, n_evicted, freed = _evict_oldest_crops(
            out, shed_tokens=output_tokens - target_input_tokens
        )
        if n_evicted:
            output_tokens = _estimate_input_tokens(out, model=model, tools=tools)
            log.info(
                "compaction[%s] evicted %d stale crop(s), freed ~%d tok -> %s/%d",
                site, n_evicted, freed, output_tokens, target_input_tokens,
            )

    # 10. STEP 7 -- thin the CURRENT turn's crops, oldest-first, keeping the
    # newest.
    evicted_current = 0
    if output_tokens > target_input_tokens:
        out, evicted_current, freed_current = _evict_current_turn_crops(
            out, shed_tokens=output_tokens - target_input_tokens
        )
        if evicted_current:
            output_tokens = _estimate_input_tokens(out, model=model, tools=tools)
            log.info(
                "compaction[%s] thinned the current turn: %d crop(s), "
                "freed ~%d tok -> %s/%d",
                site, evicted_current, freed_current, output_tokens,
                target_input_tokens,
            )

    latency = time.time() - t0
    if output_tokens > target_input_tokens:
        # Name the largest messages in the raise, so the residue's source is
        # in the log line.
        sizes = sorted(
            (
                (len(str(m.get("content") or "")), i,
                 str(m.get("role") or "?"), str(m.get("name") or ""))
                for i, m in enumerate(out)
            ),
            reverse=True,
        )[:5]
        anatomy = "; ".join(
            f"msg[{i}] {role}{('/' + name) if name else ''} "
            f"{b}B ~{head!r}"
            for b, i, role, name in sizes
            for head in [str(out[i].get("content") or "")[:60]]
        )
        raise CompactionInfeasible(
            f"cleared {len(cleared)} stale dump(s), degraded "
            f"{degraded_dumps} dump(s), folded {folded_turns} spent turn(s), "
            f"evicted {n_evicted} stale crop(s) AND thinned {evicted_current} "
            f"current-turn crop(s); tokens={output_tokens} still exceeds "
            f"target={target_input_tokens} (site={site}, "
            f"input_tokens={input_tokens}); largest: {anatomy}")

    # 11. Invariant validation: closure AND coverage.
    _validate_invariants(out)

    log.info(
        "[compactor] fired site=%s input_tokens=%d output_tokens=%d "
        "cleared_dumps=%d color_tables_kept=%d degraded_dumps=%d "
        "latency_s=%.2f",
        site,
        input_tokens,
        output_tokens,
        len(cleared),
        color_tables_kept,
        degraded_dumps,
        latency,
    )
    return out


__all__ = [
    "CompactionError",
    "CompactionInfeasible",
    "CompactionInvariantError",
    "compact_transcript",
]
