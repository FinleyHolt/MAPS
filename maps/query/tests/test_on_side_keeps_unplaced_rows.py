"""on_side on a cardinal column no row carries: the compare ran (each row
has its 'distance to'), the ref straddles every row, so every row is kept
unplaced instead of the cut refusing a column it was told to read."""

from maps.query import program as GP


def _sq(x0, y0, x1, y1):
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]


def test_on_side_with_no_placed_row_keeps_every_row():
    school = {"fid": "prior_0", "name": "School", "xy": [0.0, 0.0],
              "mask": _sq(-100, -100, 100, 100), "tags": {"amenity": "school"},
              "search_text": "school amenity school"}
    # Inside the school's north-south extent: no row is north or south of it.
    houses = [{"fid": f"prior_{i}", "name": "", "xy": [300.0 + 10 * i, 10.0 * i],
               "tags": {"building": "house"}, "search_text": "building house"}
              for i in range(1, 4)]
    ops = [{"ref": "r0", "op": "find", "args": {"query": "house"}},
           {"ref": "r1", "op": "compare", "args": {
               "items": "r0", "refs": ["prior_0"],
               "cardinals": [{"dir": "north", "ref": "prior_0"}]}},
           {"ref": "r2", "op": "filter_items", "args": {
               "items": "r1", "key": "north of prior_0", "op": "on_side"}}]
    out = GP.run_program([], [school] + houses, {"ops": ops})["results"]
    assert not any("north of prior_0" in r for r in out["r1"])
    assert len(out["r2"]) == 3
    stats = out["r2"].cut_stats
    assert (stats["on_side"], stats["no_side_kept"]) == (0, 3)
