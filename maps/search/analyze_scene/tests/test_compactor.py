"""compact_transcript: what each step sheds, in what order, and what it never touches."""
from __future__ import annotations

import base64
import copy
import io
import json

import pytest
from PIL import Image

from maps.search.analyze_scene import compactor as C
from maps.search.analyze_scene.compactor import (
    CompactionInfeasible,
    CompactionInvariantError,
    compact_transcript,
)


class _Tok:
    """Four characters a token over the rendered transcript."""

    def apply_chat_template(self, messages, tools=None, **_kw):
        return json.dumps([messages, tools], default=str)

    def encode(self, text, add_special_tokens=False):
        return range(len(text) // 4)


@pytest.fixture(autouse=True)
def _chars(monkeypatch):
    monkeypatch.setattr(C, "_tokenizer", lambda model: _Tok())


def _est(msgs) -> int:
    return C._estimate_input_tokens(msgs, model="m")


def _compact(msgs, target):
    return compact_transcript(msgs, target_input_tokens=target, model="m")


def _grid(fids, filler=0) -> str:
    lines = ["| fid | class | color |", "| --- | --- | --- |"]
    lines += [f"| {f} | car | red? |" for f in fids]
    return "\n".join(lines) + ("\n" + "z" * filler if filler else "")


def _call(tcids, code="cars = find('car')") -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": t, "type": "function",
         "function": {"name": "run", "arguments": json.dumps({"code": f"{code}  # {t}"})}}
        for t in tcids]}


def _dump(tcid, fids, filler=0) -> dict:
    head = {"len": len(fids), "fids": list(fids), "rows": _grid(fids, filler),
            "legend": "distance to X: metres"}
    return {"role": "tool", "tool_call_id": tcid, "name": "run", "content": json.dumps(
        {"said": f"cars  # {tcid}", "result": {"results": {"cars": head}}})}


def _head() -> list[dict]:
    return [{"role": "system", "content": "SYS"},
            {"role": "user", "content": "Find the red car."}]


def _units(n, filler=2000) -> list[dict]:
    out: list[dict] = []
    for k in range(n):
        out += [_call([f"c{k}"]), _dump(f"c{k}", [f"d{k}a", f"d{k}b"], filler)]
    return out


def _one_unit_three_dumps(filler) -> list[dict]:
    """Three dumps answering ONE call: nothing to fold, so the stub steps act."""
    ids = ["a", "b", "c"]
    return _head() + [_call(ids)] + [
        _dump(t, [f"{t}1", f"{t}2"], filler) for t in ids] + [
        {"role": "user", "content": "go on"}]


def _crop(k=1) -> dict:
    b = io.BytesIO()
    Image.new("RGB", (64 * k, 48 * k)).save(b, "JPEG")
    return {"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()}}


def _look(*crops) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": "crops"}, *crops]}


def _body(m) -> dict:
    return json.loads(m["content"])["result"]["results"]["cars"]


def test_a_transcript_under_budget_comes_back_itself():
    short = _head() + _units(1)
    assert _compact(short, 1) is short          # the cheap gate: <= 4 messages
    msgs = _head() + _units(3)
    assert _compact(msgs, _est(msgs)) is msgs


def test_an_unmeasurable_prompt_raises(monkeypatch):
    class _NoTools(_Tok):
        def apply_chat_template(self, messages, tools=None, **_kw):
            raise TypeError("no tools kwarg")
    monkeypatch.setattr(C, "_tokenizer", lambda model: _NoTools())
    with pytest.raises(TypeError):
        compact_transcript(_head() + _units(3), target_input_tokens=10, model="m",
                           tools=[{"type": "function"}])


def test_the_fold_keeps_the_agents_code_the_head_and_the_live_tail():
    msgs = _head() + _units(6)
    before = copy.deepcopy(msgs)
    out = _compact(msgs, _est(msgs) - 100)
    assert msgs == before                      # the input is never mutated
    assert out[:2] == msgs[:2]
    block = out[2]["content"]
    assert block.startswith(C._SCRIPT_BLOCK_HEADER)
    assert "cars = find('car')  # c0" in block
    assert "d0a" in block.split(C._FINDINGS_SEP)[1]   # what the turn found rides
    assert out[-4:] == msgs[-4:]               # the two newest units, verbatim
    assert "(1 earlier turn folded;" in block   # a marginal overflow folds one


def test_a_later_fire_appends_a_block_and_never_rewrites_the_first():
    first = _compact(_head() + _units(6), _est(_head() + _units(6)) - 100)
    grown = first + _units(4)[4:] + [_call(["n1"]), _dump("n1", ["x"], 2000),
                                     _call(["n2"]), _dump("n2", ["y"], 2000)]
    out = _compact(grown, _est(grown) - 100)
    assert out[2] == first[2]
    assert sum(C._is_script_block(m) for m in out) == 2


def test_a_message_between_folded_turns_keeps_its_place():
    follow_up = {"role": "user", "content": "now find the truck"}
    later = [m for k in (2, 3, 4) for m in (_call([f"c{k}"]), _dump(f"c{k}", ["x"], 2000))]
    msgs = _head() + _units(2) + [follow_up] + later
    out, n, _ = C._drop_spent_turns(msgs, before=_est(msgs), shed_tokens=_est(msgs),
                                    model="m", tools=None)
    assert n == 3                               # every spent turn
    assert out[2]["content"].startswith("[script so far] (2 earlier turns folded")
    assert out[3] == follow_up
    assert out[4]["content"].startswith("[script so far] (1 earlier turn folded;")
    assert "# c2" in out[4]["content"] and "# c2" not in out[2]["content"]
    assert out[5:] == msgs[9:]                  # the two newest units, verbatim


def test_step_5_strips_only_the_blocks_this_fire_made(caplog):
    earlier = C._script_block(["old = 1"], 1, ["OLDFINDING"])
    follow_up = {"role": "user", "content": "now the truck"}

    def big(k):
        return [_call([f"c{k}"]), _dump(f"c{k}", [f"k{k}_{i}" for i in range(300)])]

    msgs = _head() + [earlier] + big(0) + [follow_up] + big(1) + _units(2, 10)
    target = _est(_head() + [earlier, C._script_block(["x"], 1), follow_up,
                             C._script_block(["x"], 1)] + _units(2, 10)) + 60
    with caplog.at_level("INFO", logger=C.log.name):
        out = _compact(msgs, target)
    blocks = [m for m in out if C._is_script_block(m)]
    assert len(blocks) == 3 and out[2] is earlier and out[4] == follow_up
    assert not [b for b in blocks[1:] if C._FINDINGS_SEP in b["content"]]
    assert "folded 2 spent turn(s) into 2 script block(s)" in caplog.text


def test_a_call_whose_results_are_not_contiguous_is_no_unit():
    msgs = _head() + [_call(["a", "b"]), _dump("a", ["x"]),
                      {"role": "user", "content": "between"}, _dump("b", ["y"])]
    assert C._turn_units(msgs) == []
    assert C._turn_units(_head() + _units(2)) == [(2, 4), (4, 6)]


def test_unparseable_arguments_are_carried_raw():
    call = {"role": "assistant", "tool_calls": [
        {"id": "a", "function": {"name": "run", "arguments": "{not json"}}]}
    assert C._run_code_of(call) == ["{not json"]


def test_stale_dumps_clear_to_fids_with_their_colors_and_the_live_dump_stays():
    msgs = _one_unit_three_dumps(4000)
    out = _compact(msgs, _est(msgs) - 1500)
    stale = _body(out[3])
    assert stale["fids"] == ["a1", "a2"] and "rows" not in stale
    assert "legend" not in stale and stale["colors"] == "a1=red?; a2=red?"
    assert "show the handle again" in stale["hint"] and "census" not in stale["hint"]
    assert json.loads(out[3]["content"])["said"] == "cars  # a"
    assert out[5] == msgs[5]                   # the live view


def test_still_over_the_colors_go_then_the_live_dump_last():
    msgs = _one_unit_three_dumps(4000)
    out = _compact(msgs, 900)
    assert all("colors" not in _body(out[i]) for i in (3, 4))
    assert "rows" not in _body(out[5]) and _body(out[5])["fids"] == ["c1", "c2"]


def test_nothing_left_to_shed_raises():
    with pytest.raises(CompactionInfeasible, match="largest: msg"):
        _compact(_one_unit_three_dumps(4000), 50)


def test_the_findings_strip_keeps_code_that_contains_the_separator():
    code = f"x = 1{C._FINDINGS_SEP}y = 2"
    msgs = _head() + [_call(["q"], code=code),
                      _dump("q", [f"k{i}" for i in range(400)])] + _units(2, 10)
    code_only = msgs[:2] + [C._script_block([f"{code}  # q"], 1)] + msgs[4:]
    out = _compact(msgs, _est(code_only) + 5)
    assert out[2] == code_only[2]


def test_stale_crops_go_oldest_first_and_the_current_turn_keeps_its_own():
    msgs = _head() + [_look(_crop(4)), _look(_crop(4)), _look(_crop(2), _crop(2)),
                      {"role": "user", "content": "x"}]
    need = _est(msgs) - C.image_tokens_of(_crop(4)) // 2
    out = _compact(msgs, need)
    assert out[2]["content"][1] == {"type": "text", "text": C._EVICTED_CROP_TEXT}
    assert out[3] == msgs[3] and out[4] == msgs[4]


def test_the_current_turn_is_thinned_last_and_keeps_its_newest_crop():
    msgs = _head() + [_look(_crop(1)), _look(_crop(3), _crop(3), _crop(1)),
                      {"role": "user", "content": "x"}]
    out = _compact(msgs, _est(msgs) - 2 * C.image_tokens_of(_crop(3)))
    assert out[2]["content"][1]["type"] == "text"
    assert [p["type"] for p in out[3]["content"]] == ["text", "text", "text", "image_url"]


def test_the_image_count_bound_takes_the_oldest_across_turns():
    msgs = _head() + [_look(*[_crop()] * 15), _look(*[_crop()] * 15)]
    assert C.bound_image_count(msgs, max_images=30) is msgs
    out = C.bound_image_count(msgs, max_images=25)
    kinds = [p["type"] for m in out[2:] for p in m["content"][1:]]
    assert kinds == ["text"] * 25 + ["image_url"] * 5


def test_the_invariant_sweep_catches_orphans_reorders_and_unanswered_calls():
    with pytest.raises(CompactionInvariantError, match="orphan"):
        C._validate_invariants(_head() + [_dump("a", ["x"])])
    with pytest.raises(CompactionInvariantError, match="orphan"):
        C._validate_invariants(_head() + [_dump("a", ["x"]), _call(["a"])])
    with pytest.raises(CompactionInvariantError, match="unanswered"):
        C._validate_invariants(_head() + [_call(["a"])])


def test_only_a_served_dump_is_selected_and_degraded():
    error = {"role": "tool", "tool_call_id": "e", "content": json.dumps(
        {"said": "bad()", "error": "line 1: no such call"})}
    msgs = _head() + _units(2) + [_call(["e"]), error]
    assert C._dump_result_indices(msgs) == [3, 5]   # the trailing error is no dump
    assert C._degrade_tool_result_content(error["content"]) == (None, False)
    assert C._degrade_tool_result_content("not json") == (None, False)
    assert C._rows_color_entries(_grid(["f1"])) == ["f1=red?"]
