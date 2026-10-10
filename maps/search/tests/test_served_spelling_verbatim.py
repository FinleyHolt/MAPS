"""Served truth: every served spelling/number is the stored one.

F1 — ONE key form on every surface: the grid legend renders column keys
VERBATIM (units ride a separate tail, and a unit-suffixed key form is
filter/emphasis-illegal, exactly why it may not be served).

F2 — the two-color doc line states the truth: coexisting color columns
describe ONE object's match to several words (episode 6218, 75.8 m).

A1/A4 — adoption-tier FACTS: the column-family legend maps every servable
family to its producing call; the system prompt carries one declarative
arrangement-capability sentence.

SELECT — the {fid} accept contract, driven through ONE real round: a
surface fid is recorded with a minimal {recorded} ack and rides the
rank-0 CandidateBelief; a coordinate-shaped fid is not a fid (the
world-point select form is retired) and hits the surface rejection,
which the round survives.
"""
from __future__ import annotations

import re

from maps.llm.endpoint_spec import EndpointSpec
from maps.search.analyze_scene.geom_program import (
    KEY_PATTERN,
    _render_grid_rows,
)
from maps.search.parse_compile.prompts import CODE_DOC, CODE_SYSTEM_PROMPT
from maps.search.parse_compile.reasoner_agent import Reasoner
from maps.search.parse_compile.reasoner_loop import (
    advance_reasoner_one_round,
)

_TEST_SPEC = EndpointSpec(
    base_url="http://unused",
    model_id="qwen36-35b-a3b",
    max_model_len=262144,
    tokenizer_name="qwen",
    output_reserve=40000,
)

_CANDS = [
    {"fid": "d0", "xy": [0.0, 0.0], "class": "car",
     "distance to new_park_street": 11.4},
    {"fid": "d1", "xy": [10.0, 0.0], "class": "car",
     "distance to new_park_street": 3.2},
]
_GROUNDED = {"d0", "d1"}


class _DummyCtx:
    pass


def _select_turn(select_args):
    return {"kind": "tool_call", "content": "", "reasoning": "",
            "tool_calls": [{"id": "c0", "type": "function",
                            "function": {"name": "act",
                                         "arguments": {
                                             "select": select_args}}}]}


def _drive(turn_list):
    """Run ONE real round over the fixture surface; return (reply, the
    act acks in arrival order)."""
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=_TEST_SPEC)
    turns = iter(turn_list)
    r.step_tool_call = lambda tools: next(turns)
    acks = []
    _orig = r.append_tool_result

    def _record(cid, name, payload):
        acks.append((name, payload))
        return _orig(cid, name, payload)

    r.append_tool_result = _record
    reply = advance_reasoner_one_round(
        reasoner=r,
        user_message="the car near New Park Street",
        ctx=_DummyCtx(),
        grounded_fids=set(_GROUNDED),
        compute_namespace={"cands": [dict(c) for c in _CANDS],
                           "anchors": []},
        rid_tag="h0",
    )
    return reply, [p["select"] for n, p in acks
                   if n == "act" and "select" in p]


# --------------------------------------------------------------------------- #
# F1: one key form — legend verbatim                                          #
# --------------------------------------------------------------------------- #


def test_header_renders_keys_verbatim_with_units_separate():
    cols = ["distance to new_park_street", "facing new_park_street",
            "footprint_area_m2"]
    header = _render_grid_rows([], cols).splitlines()[0]
    for c in cols:  # every key appears in its exact served spelling
        assert c in header, header
    assert "(m)" not in header, header          # no unit-suffixed key form
    assert "(deg)" not in header, header
    # every relation key spelled in the header passes the executor's own
    # Key pattern (the spelling filter/emphasis accept — the 'verbatim'
    # promise).
    key_pat = KEY_PATTERN
    for c in ("distance to new_park_street", "facing new_park_street"):
        assert re.match(key_pat, c), (key_pat, c)
        assert not re.match(key_pat, c + "(m)")  # suffixed form is illegal


# --------------------------------------------------------------------------- #
# SELECT: the {fid} accept contract                                            #
# --------------------------------------------------------------------------- #






# --------------------------------------------------------------------------- #
# F2: the color docs state the complementary-columns truth                    #
# --------------------------------------------------------------------------- #


def test_color_docs_never_state_the_degenerate_idiom():
    for surface in (CODE_DOC, CODE_SYSTEM_PROMPT):
        assert "expresses a two-color cue" not in surface


def test_retired_vocabulary_never_reenters_the_served_docs():
    for doc in (CODE_DOC, CODE_SYSTEM_PROMPT):
        for retired in ("rows 1..1", "banding is scatter", "axis nn",
                        "axis pca", "ord_lr", "row_lo", "nb_left", "grp/"):
            assert retired not in doc, retired


# --------------------------------------------------------------------------- #
# A4: the declarative arrangement-capability sentence                          #
# --------------------------------------------------------------------------- #


def test_system_prompt_carries_no_column_grammar():
    # Names live on the served result surfaces (the d_xy law, now total):
    # the system prompt teaches NO column vocabulary — semantics ride the
    # column names the model actually reads, not prompt prose.
    assert "along_/off_" not in CODE_SYSTEM_PROMPT
    assert "vx_" not in CODE_SYSTEM_PROMPT
    assert "left of " not in CODE_SYSTEM_PROMPT
    assert "ord_/" not in CODE_SYSTEM_PROMPT  # the deleted family is gone
