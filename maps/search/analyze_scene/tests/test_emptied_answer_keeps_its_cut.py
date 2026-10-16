"""find(<thing>, nearest=<place>, dwithin=d) emptied by its distance cut
serves the cut's stats (the nearest row it dropped and the bound that would
admit it), and no line about a compare step the model never wrote."""

from maps.search.analyze_scene import geom_program as GP


def _sq(x0, y0, x1, y1):
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]


# The ops code_surface.compile_find emits for
# find("house", nearest="prior_0", dwithin=5).
_OPS = [
    {"ref": "r0", "op": "find", "args": {"query": "house"}},
    {"ref": "r1", "op": "compare", "args": {"items": "r0", "refs": ["prior_0"]}},
    {"ref": "r2", "op": "filter_items", "args": {
        "items": "r1", "key": "distance to prior_0", "op": "le", "value": 5.0}},
    {"ref": "r3", "op": "compare", "args": {"items": "r2", "refs": ["prior_0"]}},
    {"ref": "r4", "op": "score_by", "args": {
        "items": "r3", "emphasis": [{"key": "distance to prior_0", "w": -1}]}},
]


def test_emptied_nearest_dwithin_serves_the_cut_stats():
    school = {"fid": "prior_0", "name": "School", "xy": [0.0, 0.0],
              "mask": _sq(-10, -10, 10, 10), "tags": {"amenity": "school"},
              "search_text": "school amenity school"}
    houses = [{"fid": f"prior_{i}", "name": "", "xy": [40.0 * i, 0.0],
               "tags": {"building": "house"}, "search_text": "building house"}
              for i in range(1, 4)]
    out = GP.run_program([], [school] + houses, {"ops": _OPS})
    env = GP.compact_results(out["results"], answer_ref="r4")["r4"]
    assert list(env)[:3] == ["len", "items", "cut_stats"]
    assert env["cut_stats"] == {
        "cut": "3->0",
        "nearest_excluded": {"fid": "prior_1", "distance to prior_0": 30.0},
        "admit_at": 30.0, "n_at_admit": 1}
    assert "compare" not in env["hint"]
