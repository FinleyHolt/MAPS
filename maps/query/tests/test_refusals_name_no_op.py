"""The executor's refusals speak the model's surface: they quote what the
model passed and never name the compiled op (compare, ends, crossings,
filter_items, strata, union) the model never wrote."""

import pytest

from maps.query.op_io import ProgramError

from maps.query.program import run_program

_ROWS = [
    {
        "fid": "prior_1",
        "name": "Main St",
        "xy": [0.0, 0.0],
        "centerline": [[-50.0, 0.0], [50.0, 0.0]],
        "tags": {"highway": "primary"},
        "search_text": "main street highway primary",
    },
    {
        "fid": "prior_2",
        "name": "",
        "xy": [10.0, 20.0],
        "tags": {"building": "house"},
        "search_text": "building house",
    },
]
_FIND = {"ref": "r0", "op": "find", "args": {}}


def _refusal(*ops) -> str:
    with pytest.raises(ProgramError) as exc:
        run_program([], _ROWS, {"ops": [_FIND, *ops]})
    return str(exc.value)


def _op(name, **args):
    return {"ref": "r1", "op": name, "args": {"items": "r0", **args}}


@pytest.mark.parametrize(
    "op, start",
    [
        (_op("compare", refs=["nope"]), "'nope' is not a fid on this surface. "),
        (
            _op("compare", refs=[{"ref": ["nope"], "as": "spot"}]),
            "'nope' is not a fid on this surface. ",
        ),
        (
            _op("compare", refs=[5]),
            "5 is not a place: a place is a fid, a set, a row or an [x, y].",
        ),
        (
            {"ref": "r1", "op": "ends", "args": {"refs": ["nope"]}},
            "'nope' names no row on this surface. ",
        ),
        (
            {"ref": "r1", "op": "crossings", "args": {"refs": ["nope", "prior_1"]}},
            "'nope' names no geometry on this surface -- ",
        ),
        (
            _op("filter_items", key="height", op="le", value=3),
            "no item carries a numeric 'height'. ",
        ),
        (_op("filter_items", key="height", op="truthy"), "no item carries 'height'. "),
        (
            _op("stratum", tier="exact"),
            "these rows carry no match tiers: .exact/.tagged/.fuzzy come off a find's own result.",
        ),
    ],
)
def test_refusal_quotes_the_value_not_the_op(op, start):
    assert _refusal(op).startswith(start)
