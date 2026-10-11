from __future__ import annotations

from maps.search.analyze_scene import geom_compute as G
from maps.search.analyze_scene.geom_program import (
    _fmt_float,
    _render_grid_rows,
    run_program,
)


def test_fmt_float_serves_stored_value_verbatim():
    assert _fmt_float(0.04) == "0.04"
    assert _fmt_float(272.4) == "272.4"
    assert _fmt_float(14.75) == "14.75"
    assert _fmt_float(76.0) == "76.0"
    assert _fmt_float(0.0) == "0.0"
    assert _fmt_float(-0.0) == "0.0"       # signed zero is ONE value
    assert _fmt_float(0.0001) == "0.0001"  # stored 4-dp scores stay distinct
    assert _fmt_float(-0.0001) == "-0.0001"


def test_fmt_float_widens_adaptively_past_4dp():
    # An adaptive score stamp can carry more than 4 dp; the render widens
    # exactly when 4 dp would lose the stored value (cap 10 dp) — a stored
    # 1e-06 must never serve as '0.0'.
    assert _fmt_float(1e-06) == "0.000001"
    assert _fmt_float(-1e-06) == "-0.000001"
    assert _fmt_float(1.23e-08) == "0.0000000123"
    assert _fmt_float(0.1235) == "0.1235"  # 4-dp values render as today


def _grid_cells(rows_text: str, cols: list[str]) -> list[dict]:
    """Parse rendered grid rows back into {col: cell} dicts (data rows only)."""
    out = []
    for line in rows_text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != len(cols) + 2 or cells[0] in ("idx", "---"):
            continue
        out.append(dict(zip(["idx", "fid"] + cols, cells)))
    return out


def test_served_grid_never_pairs_d_zero_with_brg_gap():
    # The 6367 shape: a true 4 cm footprint gap. The stored d_ is 0.04 and
    # brg_gap_ attaches (stored > 0); the SERVED row must show 0.04 — the
    # served bytes then satisfy the served law (brg_gap_ absent iff d_=0.0).
    ref = {"fid": "prior_193", "name": "wall", "xy": [10.0, 10.0],
           "tags": {"barrier": "wall"},
           "mask": [[[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0], [0.0, 0.0]]]}
    cands = [
        {"fid": "detected_10", "xy": [22.0, 10.0], "class": "car",
         "mask": [[[20.04, 8.0], [24.0, 8.0], [24.0, 12.0], [20.04, 12.0], [20.04, 8.0]]]},
        {"fid": "detected_11", "xy": [21.0, 10.0], "class": "car",
         "mask": [[[19.0, 8.0], [23.0, 8.0], [23.0, 12.0], [19.0, 12.0], [19.0, 8.0]]]},
    ]
    out = run_program(cands, [ref], {"ops": [
        {"ref": "r0", "op": "find", "args": {}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0", "refs": ["prior_193"]}},
    ]})["results"]
    rows = {r["fid"]: r for r in out["r1"]}
    assert rows["detected_10"]["distance to prior_193"] == 0.04
    assert "facing prior_193" in rows["detected_10"]
    assert rows["detected_11"]["distance to prior_193"] == 0.0   # overlap = contact
    assert "facing prior_193" not in rows["detected_11"]
    # The SERVED bytes: no rendered row pairs a d_=0.0 cell with a brg_gap_.
    cols = ["distance to prior_193", "facing prior_193"]
    for cell_row in _grid_cells(_render_grid_rows(out["r1"], cols), cols):
        if cell_row["distance to prior_193"] == "0.0":
            assert cell_row["facing prior_193"] == "-", cell_row
        if cell_row["fid"] == "detected_10":
            assert cell_row["distance to prior_193"] == "0.04", cell_row


def test_scores_distinct_when_stored_distinct_and_zero_unsigned():
    # The adaptive stamp keeps TRULY-equal raw composites equal, normalized
    # (+0.0): a degenerate (constant) key gives every row the same raw sum,
    # and the zero score serves as '0.0' never '-0.0'.
    scored = G.score_by(
        [{"fid": "a", "distance to x": 10.0}, {"fid": "b", "distance to x": 10.0}],
        ["distance to x"], [-1e-9])
    for r in scored:
        assert str(r["score"]) == "0.0", scored  # -0.0 normalized at stamp
    scored2 = G.score_by(
        [{"fid": "a", "distance to x": 0.0}, {"fid": "b", "distance to x": 10.0}],
        ["distance to x"], [-0.001])
    cells = _grid_cells(
        _render_grid_rows(scored2, ["score"]), ["score"])
    served = [c["score"] for c in cells]
    assert len(set(served)) == len(served), served  # distinct stays distinct


def test_skewed_minmax_scores_stamp_and_render_distinct():
    # The ADAPTIVE stamp widens until adjacent distinct raws stay distinct (cap 10 dp),
    # so the stored scores — and therefore the served cells and the downstream tie
    # annotation — reflect the real ordering.
    items = [{"fid": f"a{i}", "footprint_area_m2": 65.0 + i}
             for i in range(6)]
    items.append({"fid": "outlier", "footprint_area_m2": 6.0e7})
    scored = G.score_by(items, ["footprint_area_m2"], [-1.0])
    stored = [r["score"] for r in scored]
    assert len(set(stored)) == len(stored), stored   # no false stored tie
    cells = _grid_cells(_render_grid_rows(scored, ["score"]), ["score"])
    served = [c["score"] for c in cells]
    assert len(set(served)) == len(served), served   # distinct render distinct
    # rank order is the raw order: smallest footprint first under w:-1
    assert scored[0]["fid"] == "a0" and scored[-1]["fid"] == "outlier", scored
