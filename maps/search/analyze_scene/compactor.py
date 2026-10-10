"""Deterministic clear-to-census autocompact for the reasoning engine.

When the assembled prompt for the persistent reasoner would exceed the
served window's input budget, ``compact_transcript`` sheds stale image
crops and rewrites every SUPERSEDED tool-result dump (all but the most
recent — the live view) to its fids+census stub form, returning a
message list that fits. Tool dumps here are re-derivable views over the
live world-state graph (the agent can re-run the op-program at any
turn), so clearing a stale dump loses nothing the agent cannot recover —
the stub says so explicitly. No LLM call is made anywhere on this path.

Both surfaces are served: the analyze surface names its tool
``analyze`` and puts the dump at the top level, the mission surface
names it ``run`` and wraps the same dump under ``result``. Tiers select
tool results by SHAPE, never by tool name.

Why per-tool-result clearing: dense-scene overflow is
TAIL-resident — the last-6 messages alone hold 83-90k of the 91k fleet
target at the measured carry points, and analyze dumps are ~90% of a
dense transcript's tokens — so the reducer must reach individual tool
results wherever they sit, not compress a "middle". Clearing is also
cache-honest: stubs are deterministic bytes, so the one prefix
invalidation per fire re-caches across subsequent turns.

Steps, in order, re-estimating after each. THE ORDER IS WHAT THE LOSS COSTS,
cheapest first, and re-derivable text goes before pixels:

  1. FOLD the oldest spent turns into one script block: the agent's own code
     verbatim plus the fids+census of what those turns found. The only step
     that changes message count, and the only one that TRUNCATES rather than
     rewrites -- it leaves [system, instruction, block, live tail] behind the
     break where every other step leaves a rewritten body to re-prefill.
     Bounded by the caller's shortfall to the low-water mark, so a marginal
     overflow does not collapse the whole history.
  2. Clear stale dumps to fids+census oldest-first, KEEPING a compact per-fid
     color table -- the color-reliability signal lives inside the ``rows``
     render the trim drops, and color must reach the model. Stops at the
     low-water mark (adaptive keep-N; the newest dump is never cleared here).
  3. Still over: drop the color tables from the stubs, oldest-first
     (budget-conditional color -- always-on tables pushed 4/59 monster-dump
     carries back over target).
  4. Still over: degrade remaining tool dumps largest-first, the live dump
     included, to plain fids+census. Keeps every candidate fid.
  5. Still over: strip the findings from THE BLOCK THIS FIRE CREATED, keeping
     the code. The block is inert to every step above it -- it is not a tool
     result, so nothing stubs it -- and without this the ladder ran out of
     levers. Only the newest block: rewriting one an earlier fire already
     served breaks the prefix at the oldest possible point.
  6. Still over: evict stale image crops oldest-first, never the current
     turn's. Crops led this ladder when the budget was 48,128 and sixteen
     1024 px maps were 87% of it; at 221,120, with a delivered crop measuring
     a median 225 tokens, images are ~9% of a large prompt. Text-only
     transcripts skip it and their compaction stays byte-identical.
  7. Still over: thin the CURRENT turn's crops oldest-first, keeping the
     newest -- reachable only when one turn's own images will not fit.
  8. Still over: raise ``CompactionInfeasible``.

APPEND-ONLY ACROSS FIRES. A block is inserted where its span was, and no later
fire rewrites or absorbs an earlier one, so the served prefix extends rather
than moves. That is the whole cache argument, and step 5 is where it is easiest
to break: it edits a block, so it must only ever edit the newest.

Below-budget calls return the input list unchanged (identity, same
``id()``), preserving byte-identical APC across the N parallel reasoner
fanouts. The post-compaction sweep validates ``tool_call_id`` closure AND
coverage; ``CompactionInvariantError`` is raised if either fails. Coverage
became reachable when step 4 started dropping messages -- every step before
it rewrites content bytes only, so message count, order and roles were
invariant and neither violation could arise.

The budget is CALLER-SUPPLIED: ``target_input_tokens`` comes from the
server-discovered ``EndpointSpec.input_budget``
(``maps.llm.endpoint_spec``) — the compactor never guesses the
served window. A prompt that cannot be measured (the tokenizer cannot
load or its chat template cannot render the transcript) raises: sending
it unmeasured against a discovered window is the overflow that produces
a server-side 400.

Environment overrides (read at call time):

  ``SEARCH_COMPACT_ENABLE``              "1" / "0" — global on/off (default on)
  ``SEARCH_COMPACT_TOKENIZER_NAME``      HF tokenizer name (default mapped from
                                         the model alias; falls back to
                                         "Qwen/Qwen3.6-35B-A3B")
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable

from maps.llm.tokenizer import _tokenizer
from maps.llm.vision_content import image_tokens_in, image_tokens_of
from maps.search.analyze_scene.geom_program import join_hints

log = logging.getLogger(__name__)


# Default minimum messages required before compaction is even attempted.
# Below this, identity passthrough is unconditional. Avoids tokenizer load
# on the small grader-6a / stage-1 first-call paths.
_CHEAP_GATE_MSG_COUNT = 4

# Step 2 clears down to (target - this headroom), not merely to target: the
# margin buys turns of quiet growth before the next fire (each fire is one
# prefix-cache invalidation), while stopping oldest-first as early as
# possible keeps the most recent dumps verbatim (adaptive keep-N). Sized at
# roughly half a dense analyze dump; a fire in a dense scene typically
# clears 1-3 dumps.
_CLEAR_HEADROOM_TOKENS = 16384


# ---------------------------------------------------------------------------
# Errors and event record
# ---------------------------------------------------------------------------


class CompactionError(RuntimeError):
    """Base class for autocompact failures."""


class CompactionInfeasible(CompactionError):
    """No strategy can bring the prompt under target_input_tokens."""


class CompactionInvariantError(CompactionError):
    """Post-compaction sweep found a tool_call_id without its counterpart:
    a result with no earlier call, or a call with no result."""


@dataclass(frozen=True)
class CompactionEvent:
    """One compaction event for logging / dashboards.

    Three shapes share this record: a below-budget check (``fired=False,
    infeasible=False`` — the headroom signal), a successful fire
    (``fired=True``), and an infeasible abort (``infeasible=True``, emitted
    just before the ``CompactionInfeasible`` raise so the sink sees the
    token arithmetic that made the transcript uncompactable).

    Step counters: ``folded_turns`` (step 1: spent turns folded into a script
    block, the only step that changes message count), ``cleared_dumps``
    (step 2: stale dumps stubbed), ``color_tables_kept`` (how many stubs kept
    their per-fid color table after the step-3 strip) and ``degraded_dumps``
    (step 4: remaining dumps — the live one included — degraded plain)."""

    site: str
    fired: bool
    input_tokens: int
    output_tokens: int
    latency_s: float
    target_input_tokens: int = -1
    infeasible: bool = False
    reason: str = ""
    degraded_dumps: int = 0
    cleared_dumps: int = 0
    color_tables_kept: int = 0
    folded_turns: int = 0

# Fixed-size stand-in for an OpenAI ``image_url`` content part in the
# transcript-COMPACTION path (token estimate + summarizer serialization). A
# ``look`` tool crop rides as base64 in the live request, but here each image
# collapses to this token so its base64 (a) never inflates the compaction token
# estimate into a false over-budget and (b) never floods the summarizer's
# middle-slice text. Text-only transcripts never hit this path (content is
# str/dict, never a parts list), so the text-only compaction is byte-identical
# (parity).
#
# THE PLACEHOLDER IS NOT THE PRICE. The server's ``--limit-mm image=16`` flag
# caps the image COUNT, and the tokens are billed against the same window as
# the text.
# ``_estimate_input_tokens`` adds ``image_tokens_in`` back after the flatten
# for exactly that reason.
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

    OpenAI / vLLM send ``arguments`` as a JSON string on the wire; the
    Qwen3 chat template's ``tool_call.arguments | items`` filter expects
    a dict and raises ``TypeError: Can only get item pairs from a
    mapping`` on raw strings. vLLM's server-side templating parses the
    string before rendering; the local HF tokenizer does not. We mirror
    that fix-up here for token estimation only — the original messages
    list is unchanged and still flows out to vLLM with arguments as
    strings.

    Multi-part (vision) ``content`` lists — a ``look`` tool crop delivery —
    are flattened to a compact string with each image part replaced by a
    fixed-size placeholder (``_flatten_multimodal_content``), so a look's
    base64 never inflates the compaction token estimate. Text-only
    messages (str / dict content) are untouched.
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


#: What replaces an evicted crop. The pixels stay on disk in the look audit,
#: so only the LIVE context loses them; the text says so rather than leaving a
#: hole the model reads as an image it failed to see.
_EVICTED_CROP_TEXT = "[earlier crop evicted from context]"


def _evict_oldest_crops(
    messages: list[dict], *, shed_tokens: int
) -> tuple[list[dict], int, int]:
    """Drop the OLDEST image crops until ``shed_tokens`` are freed.

    Returns ``(messages, n_evicted, tokens_freed)``; the input list is never
    mutated and is returned unchanged when nothing is evicted.

    Oldest-first because recency is what a crop is for -- the model is
    looking at where it is now -- and the LAST message carrying images is
    never touched: those are the crops the current turn was just handed, and
    evicting them makes the turn meaningless while saving a window the turn
    then has no reason to occupy.

    Exact rather than iterative. Each crop's cost is computed directly from
    its own pixels, so the shed is summed as it goes and the tokenizer runs
    once before and once after instead of once per eviction.
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


#: Marks a span the compactor replaced with the agent's own code. Recognised on
#: a later fire so a block is never itself dropped or re-summarised, and greppable
#: in a transcript so a reader can see where history was folded.
_SCRIPT_BLOCK_HEADER = "[script so far]"

#: Separates the folded code from the census of what those turns found. Named
#: because two places must agree on it: the writer and the step that strips it.
_FINDINGS_SEP = "\nwhat those turns found:\n"

def _protected_head(messages: list[dict]) -> int:
    """How many head messages are never droppable: through the operator
    instruction, the first user message (after the system prompt when one is
    served). The instruction is the ONE user message a bench episode has, so
    losing it loses the task."""
    first_user = next((i for i, m in enumerate(messages) if m.get("role") == "user"), 0)
    return first_user + 1

#: Turn units kept verbatim at the tail. The model is mid-derivation there, and
#: SelfCompact (arXiv 2606.23525) names discarding partial results mid-search as
#: the failure mode of threshold-triggered compaction.
_KEEP_TAIL_UNITS = 2


def _keep_tail_units() -> int:
    """How many recent turn units the fold leaves verbatim.

    An ARM, from ``SEARCH_COMPACT_KEEP_TAIL``. How much recent history survives
    a fold is the one knob that trades context against tokens without changing
    anything else, so it has to be settable per run to be measurable at all --
    and the ``SEARCH_COMPACT_`` prefix is already in ``ARM_ENV_PREFIXES``, so a
    run that sets it stamps it on the row without further wiring.

    Malformed raises rather than falling back: a silently-defaulted arm is an
    arm that measured the control and said otherwise.
    """
    raw = os.environ.get("SEARCH_COMPACT_KEEP_TAIL", "").strip()
    if not raw:
        return _KEEP_TAIL_UNITS
    value = int(raw)   # loud on garbage, deliberately
    if value < 1:
        raise ValueError(
            f"SEARCH_COMPACT_KEEP_TAIL={raw!r} must be >= 1: folding the "
            f"current turn is what step 7 is for, and doing it here would "
            f"take the crops the turn was just handed"
        )
    return value


def _run_code_of(message: dict) -> list[str]:
    """The code an assistant turn ran, verbatim, one entry per tool call.

    The mission surface serves ONE tool, ``run``, whose argument is ``{"code":
    ...}`` -- measured 584 of 584 tool calls across 40 episodes. A call whose
    arguments do not parse, or a surface that is not the code arm, falls back to
    the raw argument string rather than dropping the record: an unreadable
    action still says one happened, and a silent gap would read as the agent
    having done nothing.
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

    A unit is the atom of dropping. ``_validate_invariants`` enforces that a
    tool result has an earlier call; the converse -- a call with no result --
    the server enforces, by rejecting the request. So the two halves can only
    ever move together, and a unit is what makes that structural rather than
    remembered.
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
            # ONLY a unit whose results are all present and CONTIGUOUS is
            # foldable. An assistant whose replies are interleaved, reordered,
            # or missing (a truncated episode) yields a short run here, and
            # folding it would drop the call while its results stayed --
            # orphaning them into a CompactionInvariantError, or worse a
            # server 400 if the sweep missed it. Not foldable is the safe
            # answer; the later steps still reduce those messages.
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

    The code says what the agent asked; this says what came back. Both go in the
    block, because they are different information and the agent references both:
    a fid it saw once is a name it can still act on, and dropping the census to
    keep only the question would lose every candidate the span produced. Reuses
    the same degrade the stubbing steps apply, so a folded turn carries exactly
    what a stubbed one would have.
    """
    out: list[str] = []
    for i in range(span[0], span[1]):
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        stub = _degrade_tool_result_content(content, keep_color=True)
        # None means "nothing left to trim", which an ALREADY-STUBBED dump
        # returns -- and by the second fire most spent turns are exactly that,
        # because step 2 stubbed them on the first. Falling through to `stub or
        # content` is what keeps their fids; dropping them silently emptied the
        # census this block exists to carry, and took step 5's only lever with
        # it (an empty findings list means the block has nothing to strip).
        carried = stub if stub else (content if isinstance(content, str) else None)
        if carried:
            out.append(carried)
    return out


def _script_block(sources: list[str], n_turns: int,
                  findings: list[str] | None = None) -> dict:
    """The artifact: what the agent DID, in the agent's own bytes.

    Verbatim code rather than a generated summary, for four reasons that all
    matter here. It is not a paraphrase, so nothing is lost in the retelling.
    It costs no summariser call on the critical path. Its comments are the
    agent's own reasoning rather than a third party's reading of it. And it is
    DETERMINISTIC -- a summariser sampling at temperature 1.0 emits different
    bytes for the same history, and bytes that differ cannot be cached.

    Sound here because the interpreter namespace is persistent for the whole
    mission (``python_surface``: ``exec(compiled, self.ns)``), so every handle
    and variable the dropped turns bound is still live. The span costs the agent
    its memory of what it tried, never its ability to refer to what it made.
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
    never mutated and is returned unchanged when nothing is folded.
    ``shed_tokens`` is how far to go: folding stops once that many tokens are
    freed. The caller passes the shortfall to the LOW-WATER mark rather than to
    target, so the fold overshoots deliberately.

    THE ONE STEP THAT CHANGES MESSAGE COUNT. Every other step rewrites content
    bytes, which is why the module could promise structural invariance until
    now. Stubbing has a floor -- a stub is still a message, still a tool result,
    still its own overhead -- and past it the ladder used to raise
    CompactionInfeasible and kill the episode: measured 11.7% of episodes at a
    23,552-token budget and 23.3% at 17,408. Folding has no such floor.

    APPEND-ONLY ACROSS FIRES. The block is INSERTED where the folded span was,
    and an existing block is never rewritten or absorbed. Successive fires
    therefore leave earlier blocks byte-identical and add a later one, so the
    served prefix extends instead of moving. Merging into the previous block
    would read more tidily and would break the cache on every fire, which is the
    entire defect this module exists to stop causing.
    """
    units = _turn_units(messages)
    head = _protected_head(messages)
    droppable = [
        (a, b) for (a, b) in units
        if a >= head and not _is_script_block(messages[a])
    ]
    keep_tail = _keep_tail_units()
    if len(droppable) <= keep_tail:
        return messages, 0, 0
    candidates = droppable[:-keep_tail]

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
        # BOUNDED, but generously. The caller passes the shortfall to the LOW
        # WATER mark, not to target, so a fire folds ~16k of headroom deeper
        # than it strictly must and buys quiet turns before the next one --
        # each fire is a prefix break, so deep-and-rare beats shallow-and-often
        # (TokenPilot arXiv:2606.17016 measures the same shape at B=3 turns).
        #
        # It is bounded at all because unbounded is wrong at the fleet window.
        # This folded everything once it fired, which is right at the squeezed
        # 17,408 budget it was measured on and wrong at 221,120, where the
        # first overflow is marginal and collapsing forty turns to clear one
        # token is a trade nobody asked for.
        if freed >= shed_tokens:
            break
    if not folded or freed <= 0:
        return messages, 0, 0
    return out, len(folded), freed


def _evict_current_turn_crops(
    messages: list[dict], *, shed_tokens: int
) -> tuple[list[dict], int, int]:
    """Last resort: thin the CURRENT turn's own crops, oldest-first.

    :func:`_evict_oldest_crops` refuses to touch the newest vision message
    and is right to — those are the crops the turn was just handed. But a
    single turn may carry up to the served per-request image cap, and when
    that one message alone overruns the window every other tier has already
    run and the only remaining choice is between a thinner view and a dead
    episode. Measured on the reasoning-effort ladder: two max-effort
    episodes died 2,500 tokens over with one 314 KB user message holding
    every image in the transcript.

    The NEWEST crop always survives, so the turn keeps the view it most
    recently asked for and never degenerates to a text-only prompt.
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
    so the count matches the production token sequence. Tool-call
    ``arguments`` are pre-parsed to dicts (see
    ``_messages_for_tokenization``) because the Qwen3 chat template
    expects a mapping there, not a JSON string. A tokenizer that cannot
    load or a template that cannot render the transcript raises;
    discovery renders a transcript of every turn kind first, so neither
    happens mid-run.
    """
    tok = _tokenizer(model)
    text = tok.apply_chat_template(
        _messages_for_tokenization(messages),
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(text, str):
        # Older transformers versions can return a list of strings when
        # multiple inputs are queued; we always pass one conversation.
        text = "".join(text)
    ids = tok.encode(text, add_special_tokens=False)
    # The flatten above replaced every crop with a ~3-token placeholder so the
    # base64 never reaches the tokenizer -- correct, and it is why the vision
    # bill has to be added back HERE rather than left to the template. The
    # server charges those pixels against the same window as the text.
    return len(ids) + image_tokens_in(messages)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------




def _validate_invariants(messages: list[dict]) -> None:
    """Sweep ``messages`` for tool-protocol violations.

    Every ``role:"tool"`` must reference a ``tool_call_id`` produced by an
    earlier assistant message.

    CLOSURE (a result has an earlier call) AND COVERAGE (a call has a result).
    Both are checked because step 1 drops messages. Until it existed, every step
    rewrote content bytes and left count, order and roles alone, so neither
    violation could arise and closure alone was a cheap assertion. A
    half-dropped turn can arise now, and coverage is the half the SERVER would
    otherwise catch -- as a 400 mid-episode, with the request already sent.
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _enabled() -> bool:
    raw = os.environ.get("SEARCH_COMPACT_ENABLE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


# Verbose per-row RENDER keys in an analyze-op tool-result (geom_program._c /
# _model_facing_row): the human-facing markdown grid + its units line. They are
# pure surface verbosity — the machine-readable ``fids`` list carries the rank
# order the commit walk reads
# (reasoner_loop._ordered_fids_of_geom_result_value), and
# ``census``/``len``/``named``/``columns`` carry the census + schema. Dropping
# these keys trims a dense-scene dump to fids+census WITHOUT removing any
# candidate fid.
_TRIM_RENDER_KEYS = ("rows", "units", "grid", "legend")


def _iter_op_heads(obj: dict):
    """Yield the per-op head dicts wherever the payload nests them.

    The analyze payload is ``{"results": {"rN": head, ...}}``. Two
    shallower shapes also occur — a bare head, or a top-level
    ``{op_ref: head}`` map.

    The MISSION surface wraps that same dump one level deeper, under
    ``result``: its single ``run`` tool reports the script the model wrote
    beside what the script returned, so the heads sit at
    ``{"said": ..., "result": {"results": {"near_res": head}}}``. Measured
    on one episode: 22 of 33 heads carry render keys worth 61,403 of the
    182,446 tool-result characters.
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
    """``fid=color[@conf]`` entries parsed from one ``rows`` markdown grid.

    The color-reliability signal — the rendered color cell (with its
    trailing ``?`` uncertainty mask) and ``color_conf`` — lives INSIDE the
    ``rows`` render, so a plain render-key trim severs it. This compact table
    lets a stub keep the signal at a fraction of the row cost."""
    entries: list[str] = []
    fid_i = col_i = conf_i = None
    for line in rows.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if "fid" in cells:  # header (repeated every _GRID_HEADER_EVERY rows)
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

    Drops only the verbose render keys from every op head; ``fids`` and every
    scalar/census key survive, so NO candidate leaves the pool. The stub's
    hint names the re-derive affordance (the graph is live; re-running the
    op-program recovers the full rows). ``keep_color`` retains the compact
    per-fid color table parsed from the dropped ``rows``. Returns the new
    JSON string, or ``None`` when there is nothing trimmable / the content is
    not the expected shape (leave the message verbatim)."""
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
        # NB the affordance wording must stay universally true: rN refs from
        # a PRIOR stop round are dead (carry_results is round-scoped), so the
        # stub promises only what always holds — the graph is re-queryable.
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
    keeps its role / tool_call_id / name and a valid JSON content string; only
    the verbose render keys are dropped, so tool-protocol invariants hold and
    NO candidate fid leaves the pool. Best-effort: any per-message anomaly
    leaves that message verbatim. Biggest dumps are trimmed first so the fewest
    trims restore feasibility (as much surface as possible is preserved) —
    except the NEWEST dump, which trims dead-last: it is the round's live
    evidence, and starving it forces the model to decide from memory (the
    choosable-miss forensics found verification queries returning as
    fids+census stubs at exactly the deciding moment). Feasibility is
    unchanged — the newest still degrades when every stale dump together
    cannot pay the budget."""
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

    The SHAPE is the test, never the tool name. Selecting on
    ``name == "analyze"`` read as a shape test and was not one: it scoped
    every text tier to the analyze surface, and a mission declares one tool
    called ``run``.

    Shape also has to be the test for a second reason. Callers protect the
    NEWEST dump as the live view, so the selection must not admit messages
    that are not dumps — the round-era surface declares ``act`` and ``view``
    beside ``analyze``, and one trailing ``act`` result would push the live
    analyze dump into the stale set and starve it.
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
    log_fn: Callable[[CompactionEvent], None] | None = None,
) -> list[dict]:
    """Return a possibly-compacted view of ``messages`` whose
    chat-templated input-token count is <= ``target_input_tokens``.

    Below budget: returns ``messages`` unchanged (identity, same
    ``id()``). Above budget: deterministic clear-to-census (module
    docstring) — every SUPERSEDED analyze dump is rewritten to its
    fids+census stub (color table kept budget-conditionally), then the
    remaining dumps degrade largest-first as the last resort. Message
    count, order, roles and ``tool_call_id`` links never change; only
    stale content bytes do. No LLM call is made.

    ``site`` identifies the call site for log lines (e.g.
    ``"reasoner.step_tool_call"``).

    Raises ``CompactionInfeasible`` when even the fully-degraded
    transcript exceeds the target (assertion-level: unreached on the
    run415_base corpus once the tiers exist). Raises
    ``CompactionInvariantError`` if the post-clear sweep finds a
    tool-protocol violation.
    """
    if not _enabled():
        return messages

    # 1. Cheap gate.
    if len(messages) <= _CHEAP_GATE_MSG_COUNT:
        return messages

    # 2. Token estimate.
    input_tokens = _estimate_input_tokens(messages, model=model, tools=tools)

    # 3. Budget check.
    if input_tokens <= target_input_tokens:
        if log_fn is not None:
            log_fn(
                CompactionEvent(
                    site=site,
                    fired=False,
                    input_tokens=input_tokens,
                    output_tokens=input_tokens,
                    latency_s=0.0,
                    target_input_tokens=target_input_tokens,
                )
            )
        return messages

    def _infeasible(reason: str, *, output_tokens: int = -1,
                    latency_s: float = 0.0) -> CompactionInfeasible:
        """Record the abort in the event sink, then hand back the raise."""
        if log_fn is not None:
            log_fn(
                CompactionEvent(
                    site=site,
                    fired=False,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_s=latency_s,
                    target_input_tokens=target_input_tokens,
                    infeasible=True,
                    reason=reason,
                )
            )
        return CompactionInfeasible(reason)

    t0 = time.time()

    # THE LADDER IS ORDERED BY WHAT THE LOSS COSTS, cheapest first, which
    # puts text before pixels.
    #
    # A delivered crop runs a few hundred tokens against a window of
    # hundreds of thousands, so images are a small share of a large prompt
    # and text is most of it.
    #
    # A dump is also RE-DERIVABLE where a crop is not. The analyze surface
    # rebuilds from the world after every flight, so a stubbed dump costs a
    # re-query; an evicted crop costs the view of a place at a moment, and
    # its pixels survive only in the on-disk look audit, which the model
    # cannot read. Taking crops first would also re-impose, one tier down,
    # an in-context image cap.
    #
    # 4. STEP 1 — fold the spent turns into one script block.
    #
    # FIRST, because the end state is the whole point: every other step rewrites content
    # in place at an old index, which invalidates the served prefix there and
    # still leaves the rewritten body to be re-prefilled behind it. Folding
    # breaks the prefix at the same place and leaves almost nothing behind it.
    # Measured on 38 episodes that compacted at 23,552: 20,635 tokens
    # re-prefilled after the break as the graduated ladder left it, against
    # 4,379 for head+block+tail -- 79% of the recomputation was foldable.
    #
    # Deep also means rare, which is the point TokenPilot (arXiv:2606.17016)
    # measures from the other side: per-turn eviction destabilised the layout
    # where batching every three turns did not. Each fire is one prefix
    # invalidation, so the ladder buys quiet turns rather than re-firing on the
    # next token of growth.
    # Folding first costs the stubbing steps the spans they work on, which
    # killed 12.5% of replayed episodes at 17,408 tokens when nothing could
    # shrink the block afterwards. STEP 5 is what makes it safe: the block has
    # its own degrade, so the ladder still has a lever after the fold.
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

    # 5. STEP 2 — clear stale dumps (never the newest, the live
    # view) to their fids+census stubs oldest-first, color tables kept,
    # stopping at the LOW-WATER mark rather than clearing everything: the
    # most recent dumps stay verbatim whenever the budget allows (adaptive
    # keep-N; over-clearing a capable model's evidence costs re-query turns
    # — arXiv:2606.00408's collapse regime, Anthropic clear_tool_uses
    # keep=3). The headroom below target amortizes the prefix invalidation
    # so a dense scene doesn't re-fire every turn.
    # Never above target (tiny targets), floored at 8000 — the reasoner's
    # minimum viable input budget (EndpointSpec's case-f floor).
    idxs = _dump_result_indices(out)
    stale = idxs[:-1]
    base = out
    out = list(out)
    cleared: list[int] = []
    colored: list[int] = []
    for i in stale:
        # PRE-guard, not post. Checking after the write stubbed one dump on
        # every fire even when the fold had already landed under low water --
        # measured, a 12-turn transcript cleared to 7,161 against a low water
        # of 8,000 and step 2 still took msg[4]. It also biased cleared_dumps
        # +1 on every fire, which is a number readers act on.
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

    # 6. STEP 3 — budget-conditional color: still over target, so strip the
    # per-fid color tables from the stubs oldest-first. (Measured on the
    # run415_base fired corpus: plain stubs clear every carry with margin;
    # always-on color tables pushed 4/59 monster-dump carries back over.)
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

    # 7. STEP 4 — degrade the remaining dumps largest-first to plain
    # fids+census, the LIVE one dead-last (it is this round's evidence;
    # see _degrade_oversized_tool_dumps). Keeps every candidate fid; the
    # agent can re-derive.
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

    # 8. STEP 5 — strip the findings from the script block, keeping the code.
    #
    # The block's own degrade, and the reason folding can lead the ladder at
    # all. A folded block is inert to every step above it -- it is not a tool
    # result, so nothing stubs it -- and without this the ladder ran out of
    # levers and raised CompactionInfeasible on 12.5% of replayed episodes at
    # 17,408 tokens. The census goes and the agent's own code stays, which is
    # the same call step 3 makes between a dump's rows and its fids.
    stripped_findings = False
    if output_tokens > target_input_tokens and folded_turns:
        # ONLY the block this fire just created. Stripping every block in the
        # list rewrites blocks EARLIER fires already served, which breaks the
        # prefix at the oldest one -- the deepest possible break, and precisely
        # the defect this module exists to stop causing. The append-only claim
        # is about bytes already sent, so a later fire may never touch them.
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

    # 9. STEP 6 — evict STALE image crops, oldest-first, never the message
    # carrying the current turn's. Reached only once every re-derivable text
    # dump above has been stubbed, because a crop is the one thing here that
    # cannot be re-derived.
    #
    # Not sufficient on its own, and never was: crops run out. Five max-effort
    # episodes on the effort ladder evicted every stale crop they had and were
    # still 1,822 tokens over. That is why the text steps run first rather
    # than as a fallback.
    #
    # Text-only transcripts skip it entirely (no list content, nothing to
    # evict), so their compaction stays byte-identical.
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

    # 10. STEP 7 — thin the CURRENT turn's crops, oldest-first, keeping the
    # newest. Only reachable when every step above has run and the turn's own
    # images are what will not fit; the alternative at this point is not a
    # richer context, it is a dead episode.
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
        # cleared == degraded == 0 here means nothing was clearable at all
        # (no dumps, or every dump already stubbed): the residue is
        # head/prose mass this design deliberately does not touch. Name the
        # largest messages in the raise so the residue's SOURCE is in the
        # log line, not a post-mortem.
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
        raise _infeasible(
            f"cleared {len(cleared)} stale dump(s), degraded "
            f"{degraded_dumps} dump(s), folded {folded_turns} spent turn(s), "
            f"evicted {n_evicted} stale crop(s) AND thinned {evicted_current} "
            f"current-turn crop(s); tokens={output_tokens} still exceeds "
            f"target={target_input_tokens} (site={site}, "
            f"input_tokens={input_tokens}); largest: {anatomy}",
            output_tokens=output_tokens,
            latency_s=latency,
        )

    # 11. Invariant validation: closure AND coverage.
    _validate_invariants(out)

    if log_fn is not None:
        log_fn(
            CompactionEvent(
                site=site,
                fired=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_s=latency,
                target_input_tokens=target_input_tokens,
                degraded_dumps=degraded_dumps,
                cleared_dumps=len(cleared),
                color_tables_kept=color_tables_kept,
                folded_turns=folded_turns,
            )
        )
    else:
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
    "CompactionEvent",
    "compact_transcript",
]
