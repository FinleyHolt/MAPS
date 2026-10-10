#!/usr/bin/env python
"""Second and later fires, through the WHOLE ladder.

Every other fold test drives ``_drop_spent_turns`` directly, and that is how a
defect that only exists in ``compact_transcript`` stayed invisible: step 5
stripped EVERY script block it found, including ones earlier fires had already
served, so the prefix broke at the oldest block -- the deepest possible break,
and the exact thing this module exists to stop causing. A unit test on the fold
cannot see a step the fold does not call.

So these call ``compact_transcript``, twice, and assert on what the second fire
did to the first fire's bytes.
"""
from __future__ import annotations

import json

import pytest

from maps.search.analyze_scene import compactor as C
from maps.search.analyze_scene.compactor import (
    _FINDINGS_SEP,
    _degrade_tool_result_content,
    _drop_spent_turns,
    _folded_findings,
    _is_script_block,
    _turn_units,
    compact_transcript,
)


def _cs(m: dict) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c, default=str)


@pytest.fixture(autouse=True)
def _fake_estimator(monkeypatch):
    monkeypatch.setattr(
        C, "_estimate_input_tokens",
        lambda messages, *, model, tools=None: sum(len(_cs(m)) // 4 for m in messages),
    )


def _turn(i: int, fill: int = 3000) -> list[dict]:
    return [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": f"t{i}", "type": "function",
                         "function": {"name": "run",
                                      "arguments": json.dumps({"code": f"r{i}=find('x{i}')"})}}]},
        {"role": "tool", "tool_call_id": f"t{i}", "name": "run",
         "content": json.dumps({"result": {"r0": {
             "len": 1, "fids": [f"d{i}"], "units": "m",
             "rows": "| idx | fid |\n| --- | --- |\n" + "z" * fill}}})},
    ]


def _transcript(n: int) -> list[dict]:
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "go"}]
    for i in range(n):
        msgs += _turn(i)
    return msgs


def _blocks(msgs: list[dict]) -> list[int]:
    return [i for i, m in enumerate(msgs) if _is_script_block(m)]


def test_a_later_fire_never_rewrites_an_earlier_block(monkeypatch) -> None:
    """S1. The append-only claim, checked where it can actually fail.

    Forced rather than hoped for: the second fire is given a target only the
    findings-strip can reach, so step 5 definitely runs with TWO blocks in the
    list. Before the fix it stripped both, and the served prefix broke at the
    first one.
    """
    first = compact_transcript(_transcript(8), target_input_tokens=6000,
                               model="m", site="fire1")
    i1 = _blocks(first)[0]
    block1 = first[i1]["content"]
    assert _FINDINGS_SEP in block1, "fixture must leave findings on block 1"

    # Fly on, then squeeze hard enough that step 5 is the only lever left.
    grown = first + _turn(8, fill=200) + _turn(9, fill=200) + _turn(10, fill=200)
    second = compact_transcript(grown, target_input_tokens=400,
                                model="m", site="fire2")

    idx = _blocks(second)
    assert len(idx) >= 2, "the second fire should have appended its own block"
    assert second[idx[0]]["content"] == block1, (
        "an earlier block's bytes moved, so the served prefix broke at it -- "
        "which is the whole defect this module exists to stop causing"
    )


def test_folding_an_already_stubbed_turn_keeps_its_fids() -> None:
    """S2. By the second fire most spent turns have been stubbed by step 2, and
    a stub degrades to None. Dropping those left the census empty exactly where
    the fold is deepest -- and took step 5's only lever with it, since a block
    with no findings has nothing to strip."""
    stub = _degrade_tool_result_content(_turn(4)[1]["content"], keep_color=True)
    assert stub and _degrade_tool_result_content(stub, keep_color=True) is None, (
        "fixture assumes an already-stubbed dump has nothing left to trim"
    )
    already = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "t4", "function": {"name": "run", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t4", "content": stub},
    ]
    found = _folded_findings(already, (0, 2))
    assert found, "a stubbed turn contributed nothing to the block"
    assert "d4" in "".join(found), "its fid was lost"


def test_a_marginal_overflow_does_not_collapse_the_whole_history() -> None:
    """S3. ``shed_tokens`` was accepted and never read, so being one token over
    folded every spent turn. Right at the squeezed budget it was measured on,
    wrong at the fleet's 221,120, where the first overflow is marginal."""
    msgs = _transcript(20)
    total = C._estimate_input_tokens(msgs, model="m")
    out, folded, _ = _drop_spent_turns(msgs, shed_tokens=800, model="m")
    assert folded >= 1
    assert folded < len(_turn_units(msgs)) - 2, (
        f"folded {folded} units to free 800 tokens of a {total}-token "
        f"transcript; the shed target is not bounding the fold"
    )


def test_an_assistant_whose_results_are_not_contiguous_is_not_foldable() -> None:
    """S4. Folding such a turn drops the call and strands its results, which is
    an orphan -- and CompactionInvariantError was a sibling of Infeasible with
    no handler, so it killed the process rather than the episode."""
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "a", "function": {"name": "run", "arguments": "{}"}}]},
        {"role": "assistant", "content": "an interjection before the result"},
        {"role": "tool", "tool_call_id": "a", "content": "late"},
    ]
    assert _turn_units(msgs) == [], (
        "an assistant whose results do not immediately follow must not be a "
        "foldable unit"
    )


def test_code_containing_the_findings_separator_survives_the_strip() -> None:
    """S6. The strip took everything before the FIRST separator. Findings are
    single-line json so it cannot occur inside them, but the model's own code
    can contain anything, and truncating it there loses the artifact."""
    sneaky = f"# a comment mentioning{_FINDINGS_SEP}nothing in particular"
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "go"}]
    msgs += [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "z", "type": "function",
                         "function": {"name": "run",
                                      "arguments": json.dumps({"code": sneaky})}}]},
        {"role": "tool", "tool_call_id": "z", "name": "run",
         "content": json.dumps({"result": {"r0": {"len": 1, "fids": ["dz"],
                                "units": "m", "rows": "| idx |\n" + "z" * 3000}}})},
    ]
    for i in range(4):
        msgs += _turn(i)
    # Through compact_transcript, at a target only the findings-strip can
    # reach -- calling rsplit here ourselves would assert that str.rsplit
    # works and never touch step 5, which is the code under test.
    # Small enough that the ladder MUST reach step 5: after the fold this
    # fixture fits under a looser target and the strip never runs, which is
    # how the previous version of this test passed without testing anything.
    out = compact_transcript(msgs, target_input_tokens=300, model="m",
                             site="sep")
    assert _blocks(out), "the fold must have fired for step 5 to be reachable"
    block = out[_blocks(out)[0]]["content"]
    # The block still CONTAINS a separator -- that is the fixture's whole
    # point, the model wrote one into a comment. What must be gone is the
    # census that followed the LAST one, and what must survive is the code
    # after the model's own.
    assert "dz" not in block, "step 5 did not strip the findings census"
    assert "nothing in particular" in block, (
        "step 5 cut the block at the model's own text instead of at the "
        "findings it was meant to remove -- split() takes the FIRST separator, "
        "and the model's code can contain one"
    )


def test_keep_tail_is_an_env_arm(monkeypatch) -> None:
    """SEARCH_COMPACT_KEEP_TAIL, so how much recent history survives a fold is
    measurable per run rather than a constant nobody can vary. The
    SEARCH_COMPACT_ prefix is already in ARM_ENV_PREFIXES, so setting it stamps
    the arm on the row with no further wiring."""
    from maps.search.analyze_scene.compactor import _keep_tail_units

    monkeypatch.delenv("SEARCH_COMPACT_KEEP_TAIL", raising=False)
    assert _keep_tail_units() == C._KEEP_TAIL_UNITS

    monkeypatch.setenv("SEARCH_COMPACT_KEEP_TAIL", "5")
    assert _keep_tail_units() == 5
    msgs = _transcript(12)
    kept5, folded5, _ = _drop_spent_turns(msgs, shed_tokens=10**6, model="m")
    monkeypatch.setenv("SEARCH_COMPACT_KEEP_TAIL", "2")
    _, folded2, _ = _drop_spent_turns(msgs, shed_tokens=10**6, model="m")
    assert folded5 < folded2, "keeping more tail must fold fewer turns"

    # Loud on garbage: a silently-defaulted arm measured the control.
    monkeypatch.setenv("SEARCH_COMPACT_KEEP_TAIL", "0")
    with pytest.raises(ValueError):
        _keep_tail_units()
    monkeypatch.setenv("SEARCH_COMPACT_KEEP_TAIL", "banana")
    with pytest.raises(ValueError):
        _keep_tail_units()


def test_the_text_steps_still_work_on_a_post_fold_list() -> None:
    """Steps 2-4 had NO coverage on a list the fold had already rewritten.

    Every fixture in test_compactor_arithmetic.py omits the operator
    instruction, so its first turn unit starts at index 1, _PROTECTED_HEAD
    excludes it, and the fold never fires in that whole file. Its
    ``_roles(out) == _roles(msgs)`` assertion passes for that reason -- it is
    the retired pre-fold invariant.

    That left a real regression window: step 3 reads ``base[i]`` because the
    fold shifts indices, and reverting it to ``messages[i]`` broke no test.
    This drives a fold and THEN forces the text steps, asserting every
    surviving tool result still carries its own call's fids -- which is what an
    index skew would corrupt.
    """
    msgs = _transcript(10)
    out = compact_transcript(msgs, target_input_tokens=2600, model="m",
                             site="post-fold")
    assert _blocks(out), "the fold must fire for this to test anything"

    # Every surviving tool result belongs to the assistant call before it, and
    # carries ITS fid -- not a neighbour's, which is how a skew shows up.
    for i, m in enumerate(out):
        if m.get("role") != "tool":
            continue
        tcid = m["tool_call_id"]
        owner = next((out[j] for j in range(i - 1, -1, -1)
                      if out[j].get("role") == "assistant" and out[j].get("tool_calls")), None)
        assert owner is not None, f"tool result at {i} has no owning call"
        assert tcid in {c["id"] for c in owner["tool_calls"]}, (
            f"tool result {tcid} sits under a call that did not make it -- "
            f"the post-fold index mapping is skewed"
        )
        fid = f"d{tcid[1:]}"
        assert fid in _cs(m), (
            f"{tcid}'s result no longer carries its own fid {fid}; a step "
            f"rewrote it from the wrong source index"
        )
