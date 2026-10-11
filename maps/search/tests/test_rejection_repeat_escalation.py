"""A re-served rejection carries its repeat count and the live census.

Corpus fact (episode birmingham_block_12__1_7181): the color-as-filter
rejection was re-served 195x consecutively -- 69 MB served over 205 turns --
because every serve was byte-identical, so nothing in the transcript told the
model it was looping. The repeat machinery exists to break that: the first
serve is the rejection verbatim, and from the second the served text carries
how many times the class has fired plus a census of what the live surface
actually holds (counts and computed column names, never a per-fid inventory,
which is token noise that scales with the surface).

``_loop_breaker_facts`` is reached through the REAL turn loop with a scripted
reasoner, because the counter and the census closure both live inside
``advance_reasoner_one_round`` and neither is importable on its own.

ponytail: the third-consecutive escalation to a compact repetition-state line
(``_repeated_rejection_state``, gated on ``_consec >= 3``) does not fire on
this surface -- the consecutive streak resets between serves, so a spiral gets
the full text plus a growing count forever rather than the compact form. The
count and census still break byte-identity, which is what the 7181 loop
lacked. Tested here is what the surface does; the gate is a separate fix.
"""
from __future__ import annotations

from maps.search.tests.fakes import drive, run, tool_results

# Colour never cuts, so this block is refused every time it is served.
_COLOR_FILTER_BLOCK = ('a = find("car")\n'
                       'b = a.filter(color="gray")\n'
                       'b')
_OK_BLOCK = 'a = find("car")\na'
# An op-SHAPE error: its own echoed line is the fix material, so it gets the
# short count line and never the census.
_SHAPE_BLOCK = 'a = find("car")\nb = a.order(\n'

_CANDS = [
    {"fid": "detected_0", "xy": [0.0, 0.0], "class": "car",
     "color": "gray", "color_conf": 0.8, "search_text": "car"},
    {"fid": "detected_1", "xy": [10.0, 0.0], "class": "car",
     "color": "blue", "color_conf": 0.7, "search_text": "car"},
]


def drive_blocks(blocks: list[str]) -> list[dict]:
    """The served ``run`` results of one mission issuing ``blocks`` in order
    (the replies outlast the no-flight abstain streak)."""
    _, fr, _, _ = drive([[run(b)] for b in blocks] + ["done."] * 3, cands=_CANDS)
    return tool_results(fr)


def test_each_repeat_counts_up_with_the_census_and_no_fid_inventory():
    served = drive_blocks([_COLOR_FILTER_BLOCK] * 5)
    texts = [p["error"] for p in served]
    assert "colour never CUTS" in texts[0]
    assert "occurred" not in texts[0]
    assert len(set(texts)) == len(texts), texts
    for k, err in enumerate(texts[1:], start=2):
        assert "colour never CUTS" in err
        assert f"occurred {k}x." in err, err
        assert "Live surface:" in err and "Columns computed on rows so far" in err
        assert "detected_0" not in err, err
        assert len(err) < 700, len(err)


def test_a_successful_block_does_not_reset_the_class_count():
    served = drive_blocks(
        [_COLOR_FILTER_BLOCK] * 2 + [_OK_BLOCK] + [_COLOR_FILTER_BLOCK])
    assert "error" not in served[2], served[2]
    err4 = served[3]["error"]
    assert "colour never CUTS" in err4
    assert "occurred 3x." in err4


def test_shape_errors_keep_the_short_count_line():
    served = drive_blocks([_SHAPE_BLOCK] * 3)
    assert "syntax error" in served[0]["error"].lower(), served[0]
    for k, payload in enumerate(served[1:], start=2):
        err = payload["error"]
        assert f"occurred {k}x." in err, err
        # a shape error's own echoed line is the fix material; the census is
        # pure token noise beside it
        assert "Live surface:" not in err, err
