"""Measuring to an empty set is refused for both of its uses: a cut
(dwithin=, near=) and an ordering (nearest=)."""

import pytest

from maps.query.op_io import ProgramError

from maps.query.program import run_program
def test_an_empty_set_ref_refuses_cut_and_order_alike():
    rows = [{"fid": "prior_1", "name": "", "xy": [0.0, 0.0],
             "tags": {"building": "house"}, "search_text": "building house"}]
    ops = [{"ref": "r0", "op": "find", "args": {"query": "zzzz"}},
           {"ref": "r1", "op": "find", "args": {"query": "house"}},
           {"ref": "r2", "op": "compare", "args": {
               "items": "r1", "refs": [{"ref": ["r0"], "as": "lots"}]}}]
    with pytest.raises(ProgramError) as exc:
        run_program([], rows, {"ops": ops})
    assert str(exc.value).startswith(
        "lots is empty, so 'distance to lots' is undefined on every row and "
        "nothing can be cut or ordered by it.")
