"""Anti-spiral program identity: identical SELF-CONTAINED programs dedup, with ref
LABELS normalized out of the key."""
from __future__ import annotations

from maps.search.parse_compile.carry import _program_dedup_key
from maps.search.tests.fakes import drive, ops_of_block, run, tool_results

_BLOCK = 'r1 = find("car")\nr2 = r1.order(nearest="prior_1")\n'


def _key(code, **kw):
    return _program_dedup_key({"ops": ops_of_block(code, **kw)})


def test_a_reissued_block_shares_its_key_whatever_its_ref_labels():
    assert _key(_BLOCK) is not None
    assert _key(_BLOCK) == _key(_BLOCK, rn=7)
    assert _key(_BLOCK) != _key(_BLOCK.replace("prior_1", "prior_2"))


def test_a_program_reading_an_earlier_handle_is_never_deduped():
    assert _key('r2 = r1.order(nearest="prior_1")\n', bound={"r1": "r7"}) is None
    assert _key("more_of(r1)\n", bound={"r1": "r7"}) is None


def test_wiring_and_emission_order_stay_distinct():
    find = {"op": "find", "args": {"query": "car"}}
    near = {"op": "find", "args": {"query": "gate"}}
    a = [{"ref": "r0", **find}, {"ref": "r1", **near}]
    b = [{"ref": "r0", **near}, {"ref": "r1", **find}]
    assert _program_dedup_key({"ops": a}) != _program_dedup_key({"ops": b})

    def _cut(items):
        return {"ref": "r2", "op": "filter_items",
                "args": {"items": items, "key": "xy", "op": "truthy"}}
    assert _program_dedup_key({"ops": a + [_cut("r0")]}) \
        != _program_dedup_key({"ops": a + [_cut("r1")]})


def test_a_deduped_reissue_answers_under_its_own_labels():
    """The re-run is not re-rendered, and the handle it names still reaches the
    first run's rows."""
    reply, fr, flights, _d = drive([
        [run('a = find("car")\n')],
        [run('b = find("car")\n')],
        [run("go_to(b[0])\n")],
        "done",
    ])
    out = tool_results(fr)
    assert "no new information since turn" in out[1]["result"]["note"], out[1]
    assert flights == [{"target_fid": "car_1"}], flights
