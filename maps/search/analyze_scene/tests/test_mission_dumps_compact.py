"""The text tiers fire on the mission surface, not just the analyze one."""

from __future__ import annotations

import json

from maps.search.analyze_scene.compactor import (
    _degrade_tool_result_content,
    _dump_result_indices,
    _iter_op_heads,
)


def _mission_dump(n_rows: int = 40) -> str:
    """One ``run`` tool-result, shaped as the served mission emits it."""
    rows = "\n".join(
        f"| prior_{i} | red | 0.8 | 120.0 |" for i in range(n_rows)
    )
    return json.dumps({
        "said": 'res = find("highway")\nprint(len(res))',
        "result": {
            "note": "no new information since turn 9",
            "results": {
                "res": {
                    "len": n_rows,
                    "fids": [f"prior_{i}" for i in range(n_rows)],
                    "census": {"highway": n_rows},
                    "rows": "| fid | color | color_conf | dist |\n" + rows,
                    "legend": "dist = metres from the standing viewpoint",
                },
            },
        },
        "printed": str(n_rows),
        "script_lines": 2,
    })


def _analyze_dump(n_rows: int = 40) -> str:
    """The analyze surface's shape: the same heads, one level shallower."""
    rows = "\n".join(f"| prior_{i} | blue |" for i in range(n_rows))
    return json.dumps({
        "results": {
            "r1": {
                "len": n_rows,
                "fids": [f"prior_{i}" for i in range(n_rows)],
                "rows": "| fid | color |\n" + rows,
                "legend": "colors are ordering-only",
            },
        },
    })


def _tool_msg(content: str, name: str) -> dict:
    return {"role": "tool", "name": name, "tool_call_id": "c1",
            "content": content}


def test_no_candidate_fid_leaves_the_pool() -> None:
    after = json.loads(_degrade_tool_result_content(_mission_dump()))
    head = after["result"]["results"]["res"]
    assert head["fids"] == [f"prior_{i}" for i in range(40)]
    assert head["len"] == 40
    assert head["census"] == {"highway": 40}
    assert "rows" not in head and "legend" not in head


def test_the_script_the_model_wrote_survives() -> None:
    """``said`` is the model's own turn, not a re-derivable view."""
    after = json.loads(_degrade_tool_result_content(_mission_dump()))
    assert after["said"] == 'res = find("highway")\nprint(len(res))'
    assert after["printed"] == "40"
    assert after["result"]["note"] == "no new information since turn 9"


def test_color_is_kept_when_asked_on_the_mission_shape() -> None:
    """Color rides in the ``rows`` render the trim drops, and it is an ordering signal
    the model has to keep seeing."""
    after = json.loads(
        _degrade_tool_result_content(_mission_dump(), keep_color=True))
    colors = after["result"]["results"]["res"]["colors"]
    assert "prior_0=red" in colors
    assert "@0.8" in colors, "color_conf was severed from the stub"


def test_both_surfaces_are_selected_and_a_stub_is_not_reselected() -> None:
    msgs = [
        {"role": "user", "content": "find it"},
        _tool_msg(_mission_dump(), "run"),
        _tool_msg(_analyze_dump(), "analyze"),
        {"role": "assistant", "content": "thinking"},
    ]
    assert _dump_result_indices(msgs) == [1, 2]

    # A message with nothing left to trim is not a dump: re-selecting a stub
    # would let the tiers re-fire on it every turn for no tokens.
    stub = _degrade_tool_result_content(_mission_dump())
    assert _degrade_tool_result_content(stub) is None
    assert _dump_result_indices([_tool_msg(stub, "run")]) == []


def test_a_trailing_non_dump_does_not_expose_the_live_dump() -> None:
    """Callers protect the NEWEST selected message as the live view, so selection must
    admit dumps ONLY."""
    act_result = json.dumps({"flown": True, "at": [10, 20]})
    msgs = [
        _tool_msg(_analyze_dump(), "analyze"),
        _tool_msg(_analyze_dump(), "analyze"),
        _tool_msg(act_result, "act"),
    ]
    idxs = _dump_result_indices(msgs)
    assert idxs == [0, 1], "a non-dump tool result was selected as a dump"
    assert idxs[-1] == 1, "the live analyze dump lost its protection"


def test_a_tool_result_that_is_not_a_dump_is_left_verbatim() -> None:
    """Most ``run`` results carry no dump at all -- they report a print or an error."""
    for content in (
        json.dumps({"said": 'print("hi")', "printed": "hi"}),
        json.dumps({"error": "name 'foo' is not defined"}),
        "not json at all",
    ):
        assert _degrade_tool_result_content(content) is None


def test_iter_op_heads_does_not_recurse_forever() -> None:
    """``result`` nesting is data, so a pathological payload must terminate."""
    nested: dict = {"result": {}}
    cur = nested
    for _ in range(50):
        cur["result"] = {"result": {}}
        cur = cur["result"]
    assert list(_iter_op_heads(nested)) == []
