"""Fit the reasoner transcript under the served window's input budget, with
no model call.

``compact_transcript`` takes these steps in order, re-measuring after each and
stopping once the transcript fits; re-derivable text goes before pixels:

  1. fold the oldest spent turns into script blocks, one per run of
     adjacent turns in that run's place (the agent's code verbatim, plus the
     fids and census of what they found), keeping the head and the two
     newest turns, down to a low-water mark below target;
  2. stub stale dumps (all but the newest) to fids, census and a colour
     table, oldest first, down to the low-water mark;
  3. drop those colour tables;
  4. stub the remaining dumps largest first, the newest last;
  5. strip the findings from the blocks this fire made;
  6. evict stale crops oldest first, never the current turn's;
  7. thin the current turn's crops, keeping its newest;

and raises ``CompactionInfeasible`` when all of them fall short. A block is
never rewritten by a later fire, so the served prefix only grows. A
transcript under budget comes back as the same list.
"""

from __future__ import annotations

import json
import logging
import time

from maps.llm.endpoint_spec import _MIN_INPUT_BUDGET
from maps.llm.tokenizer import _tokenizer
from maps.llm.vision_content import image_tokens_in, image_tokens_of
from maps.search.analyze_scene.geom_program import join_hints

log = logging.getLogger(__name__)


# At or below this many messages the transcript passes through without a
# token estimate.
_CHEAP_GATE_MSG_COUNT = 4

# The low-water mark sits this far below target, so a fire buys turns of
# growth before the next one invalidates the prefix cache.
_CLEAR_HEADROOM_TOKENS = 16384


class CompactionInfeasible(RuntimeError):
    """No strategy can bring the prompt under target_input_tokens."""


class CompactionInvariantError(RuntimeError):
    """A tool result with no earlier call, or a call with no result."""


# What a crop flattens to for the tokenizer; its pixels are priced separately.
_IMAGE_PART_PLACEHOLDER = "[image crop]"


def _flatten_multimodal_content(content: list) -> str:
    """Multi-part content as one string, each image part a placeholder."""
    parts: list[str] = []
    for p in content:
        if isinstance(p, dict):
            t = p.get("type")
            if t == "text":
                parts.append(str(p.get("text") or ""))
            elif "image" in str(t):
                parts.append(_IMAGE_PART_PLACEHOLDER)
            else:
                parts.append(str(p))
        else:
            parts.append(str(p))
    return " ".join(x for x in parts if x)


def _parsed_arguments(tc: dict) -> dict:
    """``tc`` with string ``function.arguments`` parsed to a mapping (``{}``
    when empty or not JSON); any other tool call as itself."""
    fn = tc.get("function") or {}
    args = fn.get("arguments")
    if not isinstance(args, str):
        return tc
    try:
        parsed = json.loads(args) if args else {}
    except json.JSONDecodeError:
        parsed = {}
    return {**tc, "function": {**fn, "arguments": parsed}}


def _messages_for_tokenization(messages: list[dict]) -> list[dict]:
    """A copy of ``messages`` as the server renders them: tool-call
    arguments parsed to a mapping, multi-part content flattened."""
    out: list[dict] = []
    for m in messages:
        m2 = dict(m)
        if isinstance(m.get("content"), list):
            m2["content"] = _flatten_multimodal_content(m["content"])
        if m.get("tool_calls"):
            m2["tool_calls"] = [_parsed_arguments(tc) for tc in m["tool_calls"]]
        out.append(m2)
    return out


#: What replaces an evicted crop, so the model reads a removal rather than an
#: image it failed to see.
_EVICTED_CROP_TEXT = "[earlier crop evicted from context]"


def _with_content(message: dict, content) -> dict:
    """A copy of ``message`` carrying ``content``."""
    return {**message, "content": content}


def _evict_parts(out: list[dict], slots) -> None:
    """Replace each ``(message index, part index)`` crop of ``out`` with
    :data:`_EVICTED_CROP_TEXT`, in place."""
    for i, j in slots:
        parts = list(out[i]["content"])
        parts[j] = {"type": "text", "text": _EVICTED_CROP_TEXT}
        out[i] = _with_content(out[i], parts)


def _crop_slots(messages: list[dict]) -> list[tuple[int, int]]:
    """``(message index, part index)`` of every ``image_url`` part, in order."""
    return [
        (i, j)
        for i, message in enumerate(messages)
        if isinstance(message.get("content"), list)
        for j, part in enumerate(message["content"])
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]


def _evict_crops(
    messages: list[dict], *, shed_tokens: int, current: bool
) -> tuple[list[dict], int, int]:
    """``(messages, n_evicted, tokens_freed)``: crops replaced by
    :data:`_EVICTED_CROP_TEXT`, oldest first, until ``shed_tokens`` are freed;
    every priced crop before the current turn's message, or with ``current``
    that message's crops but its newest. Nothing evicted returns the input."""
    slots = _crop_slots(messages)
    if not slots:
        return messages, 0, 0
    last = slots[-1][0]
    if current:
        slots = [s for s in slots if s[0] == last][:-1]
    else:
        slots = [(i, j) for i, j in slots
                 if i < last and image_tokens_of(messages[i]["content"][j])]
    taken: list[tuple[int, int]] = []
    freed = 0
    for i, j in slots:
        if freed >= shed_tokens:
            break
        freed += image_tokens_of(messages[i]["content"][j])
        taken.append((i, j))
    if not taken:
        return messages, 0, 0
    out = list(messages)
    _evict_parts(out, taken)
    return out, len(taken), freed


#: How far under ``max_images`` an eviction goes: more than one ``look``
#: delivers (``parse_compile.views.LOOK_MAX_CROPS``, 16), so the next look
#: lands under the limit without evicting again.
_IMAGE_COUNT_HEADROOM = 20


def bound_image_count(messages: list[dict], *, max_images: int) -> list[dict]:
    """Past ``max_images`` image parts (a hosted endpoint's count limit),
    the oldest evicted down to ``max_images - _IMAGE_COUNT_HEADROOM``; within
    the limit, the input itself."""
    image_at = _crop_slots(messages)
    if len(image_at) <= max_images:
        return messages
    excess = len(image_at) - max(max_images - _IMAGE_COUNT_HEADROOM, 1)
    out = list(messages)
    _evict_parts(out, image_at[:excess])
    log.info("image bound: evicted the %d oldest of %d crops (limit %d)",
             excess, len(image_at), max_images)
    return out


#: Opens a script block; a later fire never folds one again.
_SCRIPT_BLOCK_HEADER = "[script so far]"

#: Separates a block's code from what its turns found.
_FINDINGS_SEP = "\nwhat those turns found:\n"


def _protected_head(messages: list[dict]) -> int:
    """How many head messages never fold: through the first user message."""
    first_user = next((i for i, m in enumerate(messages) if m.get("role") == "user"), 0)
    return first_user + 1


#: Turn units the fold keeps verbatim at the tail.
_KEEP_TAIL_UNITS = 2


def _run_code_of(message: dict) -> list[str]:
    """The ``code`` of each ``run`` call, or its raw arguments when they
    carry none."""
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
    """The ``[assistant call, *its results]`` spans whose results are all
    present and contiguous: a call folds with its results or not at all."""
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
            if ids and answered == ids:
                units.append((i, j))
            i = j
            continue
        i += 1
    return units


def _folded_findings(messages: list[dict], span: tuple[int, int]) -> list[str]:
    """The colour-keeping stubs of a span's tool results (an already
    stubbed result as it stands)."""
    out: list[str] = []
    for i in range(span[0], span[1]):
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        stub, _ = _degrade_tool_result_content(content, keep_color=True)
        carried = stub if stub else (content if isinstance(content, str) else None)
        if carried:
            out.append(carried)
    return out


def _script_block(sources: list[str], n_turns: int,
                  findings: list[str] | None = None) -> dict:
    """The script block: the folded turns' code verbatim (their names stay
    bound in the mission's namespace), then what they found."""
    body = "\n".join(sources)
    found = "\n".join(findings or [])
    return {
        "role": "assistant",
        "content": (
            f"{_SCRIPT_BLOCK_HEADER} ({n_turns} earlier "
            f"turn{'s' if n_turns != 1 else ''} folded; the names "
            f"they bound are still live)\n```python\n{body}\n```"
            + (f"{_FINDINGS_SEP}{found}" if found else "")
        ),
    }


def _folded(messages: list[dict], spans: list[tuple[int, int]]) -> list[dict]:
    """``messages`` with each run of adjacent ``spans`` replaced by one
    script block in its place; every message between runs stays."""
    runs: list[list[tuple[int, int]]] = []
    for a, b in spans:
        if runs and runs[-1][-1][1] == a:
            runs[-1].append((a, b))
        else:
            runs.append([(a, b)])
    out: list[dict] = []
    i = 0
    for run in runs:
        out += messages[i:run[0][0]]
        out.append(_script_block(
            [c for a, _ in run for c in _run_code_of(messages[a])], len(run),
            [f for span in run for f in _folded_findings(messages, span)]))
        i = run[-1][1]
    return out + messages[i:]


def _is_script_block(message: dict) -> bool:
    content = message.get("content")
    return (message.get("role") == "assistant" and isinstance(content, str)
            and content.startswith(_SCRIPT_BLOCK_HEADER))


def _drop_spent_turns(
    messages: list[dict], *, before: int, shed_tokens: int, model: str,
    tools: list[dict] | None,
) -> tuple[list[dict], int, int]:
    """``(messages, n_turns_folded, tokens_freed)``: the oldest foldable
    turns of ``messages`` (``before`` input tokens) folded into script
    blocks (:func:`_folded`) until ``shed_tokens`` are freed. Earlier blocks
    are never touched; nothing folded returns the input."""
    units = _turn_units(messages)
    head = _protected_head(messages)
    droppable = [(a, b) for (a, b) in units if a >= head]
    if len(droppable) <= _KEEP_TAIL_UNITS:
        return messages, 0, 0
    candidates = droppable[:-_KEEP_TAIL_UNITS]

    folded: list[tuple[int, int]] = []
    out = messages
    freed = 0
    for span in candidates:
        folded.append(span)
        out = _folded(messages, folded)
        after = _estimate_input_tokens(out, model=model, tools=tools)
        freed = before - after
        if freed >= shed_tokens:
            break
    if freed <= 0:
        return messages, 0, 0
    return out, len(folded), freed


def _estimate_input_tokens(
    messages: list[dict],
    *,
    model: str,
    tools: list[dict] | None = None,
) -> int:
    """The chat-templated input tokens the server will see, crops priced by
    their pixels; a tokenizer or template failure raises."""
    tok = _tokenizer(model)
    text = tok.apply_chat_template(
        _messages_for_tokenization(messages),
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    ids = tok.encode(text, add_special_tokens=False)
    return len(ids) + image_tokens_in(messages)


def _validate_invariants(messages: list[dict]) -> None:
    """Raise unless every tool result has an earlier call and every call a
    result; the server would refuse either."""
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


# The render keys a stub drops; ``fids`` and the census stay.
_TRIM_RENDER_KEYS = ("rows", "legend")


def _iter_op_heads(obj: dict):
    """The op heads of a served ``run`` result,
    ``{"result": {"results": {name: head, ...}}}``, that carry a render."""
    res = obj.get("result")
    results = res.get("results") if isinstance(res, dict) else None
    if isinstance(results, dict):
        for v in results.values():
            if isinstance(v, dict) and any(k in v for k in _TRIM_RENDER_KEYS):
                yield v


def _rows_color_entries(rows: str) -> list[str]:
    """``fid=color`` per data row of a ``rows`` grid."""
    entries: list[str] = []
    fid_i = col_i = None
    for line in rows.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if "fid" in cells:  # a header row (repeats down the grid)
            fid_i = cells.index("fid")
            col_i = cells.index("color") if "color" in cells else None
            continue
        if fid_i is None or col_i is None or fid_i >= len(cells):
            continue
        fid, col = cells[fid_i], cells[col_i] if col_i < len(cells) else "-"
        # skip blanks, "-" absent-cells AND the "---" markdown separator row
        if not fid.strip("-") or not col.strip("-"):
            continue
        entries.append(f"{fid}={col}")
    return entries


def _degrade_tool_result_content(
    content: str, *, keep_color: bool = False
) -> tuple[str | None, bool]:
    """``(stub, had_colors)``: one tool result with every head's render keys
    dropped (with ``keep_color``, a colour table kept) and a hint that the
    rows re-derive, ``None`` when there is nothing to trim; ``had_colors``
    when a colour table was kept."""
    try:
        obj = json.loads(content)
    except (TypeError, ValueError):
        return None, False
    if not isinstance(obj, dict):
        return None, False
    changed = had_colors = False
    for head in _iter_op_heads(obj):
        present = [k for k in _TRIM_RENDER_KEYS if k in head]
        colors: list[str] = []
        if keep_color and isinstance(head.get("rows"), str):
            colors = _rows_color_entries(head["rows"])
        for k in present:
            head.pop(k, None)
        if colors:
            head["colors"] = "; ".join(colors)
            had_colors = True
        changed = True
        note = ("rows trimmed to fids for token budget; the world-state "
                "graph is live: show the handle again to re-derive the full "
                "rows.")
        head["hint"] = join_hints(head.get("hint"), note)
    if not changed:
        return None, False
    return json.dumps(obj, default=str), had_colors


def _stub_until(out, idxs, *, keep_color, source, bound, tokens, model, tools):
    """``(stubbed, colored, tokens)``: each of ``idxs`` in turn rewritten in
    ``out`` as the stub of ``source[i]``, until ``tokens`` is at or under
    ``bound``; the indices stubbed, those that kept a colour table, and the
    new estimate."""
    stubbed: list[int] = []
    colored: list[int] = []
    for i in idxs:
        if tokens <= bound:
            break
        stub, had_colors = _degrade_tool_result_content(
            source[i]["content"], keep_color=keep_color)
        if stub is None:
            continue
        out[i] = _with_content(out[i], stub)
        stubbed.append(i)
        if had_colors:
            colored.append(i)
        tokens = _estimate_input_tokens(out, model=model, tools=tools)
    return stubbed, colored, tokens


def _carries_dump(content: object) -> bool:
    """Whether a tool-result body holds a head with a render to trim."""
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
) -> list[dict]:
    """``messages`` fitted under ``target_input_tokens`` by the steps of the
    module docstring; ``CompactionInvariantError`` if a step broke the tool
    protocol."""
    if len(messages) <= _CHEAP_GATE_MSG_COUNT:
        return messages

    input_tokens = _estimate_input_tokens(messages, model=model, tools=tools)

    if input_tokens <= target_input_tokens:
        return messages

    t0 = time.time()

    # Step 1 goes first: every other step rewrites an old message in place,
    # leaving it to re-prefill. The low-water mark is never above target and
    # floored at the least input budget a reasoner runs on.
    low_water = max(target_input_tokens - _CLEAR_HEADROOM_TOKENS,
                    min(_MIN_INPUT_BUDGET, target_input_tokens))
    output_tokens = input_tokens
    out, folded_turns, freed_fold = _drop_spent_turns(
        messages, before=input_tokens, shed_tokens=input_tokens - low_water,
        model=model, tools=tools,
    )
    if folded_turns:
        output_tokens = input_tokens - freed_fold
        log.info(
            "compaction folded %d spent turn(s) into a script block, "
            "freed ~%d tok -> %s/%d",
            folded_turns, freed_fold, output_tokens, target_input_tokens,
        )

    # Steps 2-4.
    base = out
    out = list(out)
    _est = {"model": model, "tools": tools}
    cleared, colored, output_tokens = _stub_until(
        out, _dump_result_indices(out)[:-1], keep_color=True, source=base,
        bound=low_water, tokens=output_tokens, **_est)
    plain, _, output_tokens = _stub_until(
        out, colored, keep_color=False, source=base,
        bound=target_input_tokens, tokens=output_tokens, **_est)
    color_tables_kept = len(colored) - len(plain)
    idxs = _dump_result_indices(out)
    # Largest first, the newest last.
    order = sorted(idxs, key=lambda i: len(out[i]["content"]), reverse=True)
    if idxs:
        order = [i for i in order if i != idxs[-1]] + [idxs[-1]]
    degraded, _, output_tokens = _stub_until(
        out, order, keep_color=False, source=out,
        bound=target_input_tokens, tokens=output_tokens, **_est)
    degraded_dumps = len(degraded)

    # Step 5, on the blocks this fire made: earlier blocks are served prefix.
    if output_tokens > target_input_tokens and folded_turns:
        kept = {id(m) for m in messages}
        stripped = False
        for i, m in enumerate(out):
            if not _is_script_block(m) or id(m) in kept:
                continue
            # rsplit: the agent's code may contain the separator, the
            # single-line findings cannot.
            body = m["content"].rsplit(_FINDINGS_SEP, 1)[0]
            if body != m["content"]:
                out[i] = _with_content(m, body)
                stripped = True
        if stripped:
            output_tokens = _estimate_input_tokens(out, model=model, tools=tools)
            log.info("compaction stripped block findings -> %s/%d",
                     output_tokens, target_input_tokens)

    # Step 6.
    n_evicted = 0
    if output_tokens > target_input_tokens:
        out, n_evicted, freed = _evict_crops(
            out, shed_tokens=output_tokens - target_input_tokens, current=False
        )
        if n_evicted:
            output_tokens = _estimate_input_tokens(out, model=model, tools=tools)
            log.info(
                "compaction evicted %d stale crop(s), freed ~%d tok -> %s/%d",
                n_evicted, freed, output_tokens, target_input_tokens,
            )

    # Step 7.
    evicted_current = 0
    if output_tokens > target_input_tokens:
        out, evicted_current, freed_current = _evict_crops(
            out, shed_tokens=output_tokens - target_input_tokens, current=True
        )
        if evicted_current:
            output_tokens = _estimate_input_tokens(out, model=model, tools=tools)
            log.info(
                "compaction thinned the current turn: %d crop(s), "
                "freed ~%d tok -> %s/%d",
                evicted_current, freed_current, output_tokens,
                target_input_tokens,
            )

    latency = time.time() - t0
    if output_tokens > target_input_tokens:
        # The raise names the largest messages.
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
            f"target={target_input_tokens} "
            f"(input_tokens={input_tokens}); largest: {anatomy}")

    _validate_invariants(out)

    log.info(
        "[compactor] fired input_tokens=%d output_tokens=%d "
        "cleared_dumps=%d color_tables_kept=%d degraded_dumps=%d "
        "latency_s=%.2f",
        input_tokens,
        output_tokens,
        len(cleared),
        color_tables_kept,
        degraded_dumps,
        latency,
    )
    return out
