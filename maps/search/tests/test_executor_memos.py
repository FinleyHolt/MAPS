"""What the executor memoises per surface serves the same bytes cold and warm."""

from __future__ import annotations

import json

import numpy as np

from maps.search.analyze_scene import geom_program as gp


class _Node:
    def __init__(self, r, c, name=None, provenance="prebaked", cls="building",
                 w=4, h=3):
        self.cells = np.asarray([[r + i, c + j] for i in range(h)
                                 for j in range(w)], dtype=np.int32)
        self.world_xy = None
        self.centroid = (int(r), int(c))
        self.centerline = None
        self.class_name = cls
        self.name = name
        self.provenance = provenance
        self.tags = {"building": "yes"} if provenance == "prebaked" else {}


def _surface(n=40):
    rng = np.random.default_rng(3)
    nodes = {f"prior_{i}": _Node(int(rng.integers(0, 400)),
                                 int(rng.integers(0, 400)), name=f"b{i}")
             for i in range(n)}
    nodes["detected_1"] = _Node(200, 200, provenance="detected", cls="car")
    nodes["detected_2"] = _Node(210, 215, provenance="detected", cls="car")
    return gp.build_surface_from_graph(nodes.items(), (0.0, 0.0), 2.0)


def _served(cands, anchors, program) -> str:
    out = gp.run_program(cands, anchors, program)
    return json.dumps(gp.compact_results(out["results"]), sort_keys=True,
                      default=str)


_COMPARE = {"ops": [
    {"ref": "r0", "op": "find", "args": {"query": "building"}},
    {"ref": "r1", "op": "compare",
     "args": {"items": "r0", "refs": [{"ref": "detected_1", "as": "d"}]}},
    {"ref": "r2", "op": "filter_items",
     "args": {"items": "r1", "key": "distance to d", "op": "le",
              "value": 300}},
    {"ref": "r3", "op": "compare",
     "args": {"items": "r0",
              "refs": [{"ref": ["detected_1", "detected_2"], "as": "cars"}]}},
]}


def test_mask_geometry_memo_serves_the_same_bytes_cold_and_warm():
    cands, anchors = _surface()
    gp._MASK_GEOM.clear()
    cold = _served(cands, anchors, _COMPARE)
    assert len(gp._MASK_GEOM) >= 40, len(gp._MASK_GEOM)    # it was filled
    warm = _served(cands, anchors, _COMPARE)
    assert warm == cold
    # a rebuild hands back the same mask lists for unchanged nodes, so the
    # memo survives it; and its answer is still the cold one
    cands2, anchors2 = _surface()
    assert _served(cands2, anchors2, _COMPARE) == cold
    # vacuity floor: real distances were served
    assert '"distance to d"' in cold and "distance to cars" in cold


_FINDS = {"ops": [
    {"ref": "r0", "op": "find", "args": {"query": "building"}},
    {"ref": "r1", "op": "find", "args": {"query": "b7"}},
    {"ref": "r2", "op": "find", "args": {"query": "zeppelin hangar"}},
    {"ref": "r3", "op": "find", "args": {"query": "car", "color": "red"}},
    {"ref": "r4", "op": "find", "args": {"query": "car"}},
    {"ref": "r5", "op": "find", "args": {"query": "car withcolor"}},
]}


def test_the_find_memo_serves_the_same_bytes_cold_and_warm():
    cands, anchors = _surface()
    gp._FIND_MEMO.update(lists=None, idf={}, scored={}, dead={}, census={})
    cold = _served(cands, anchors, _FINDS)
    assert gp._FIND_MEMO["scored"], "the memo was not filled"
    warm = _served(cands, anchors, _FINDS)
    assert warm == cold
    # the rows a result holds are never the memo's own dicts
    out = gp.run_program(cands, anchors, _FINDS)["results"]
    memo_ids = {id(r) for rows in gp._FIND_MEMO["scored"].values()
                for r in rows}
    for rows in out.values():
        for r in rows:
            assert id(r) not in memo_ids
    # a new surface -- the lists moved -- empties the memo and re-scores
    cands2, anchors2 = _surface(n=41)
    assert _served(cands2, anchors2, _FINDS) != cold
    assert gp._FIND_MEMO["lists"][1] is cands2
    # vacuity floors: a hit, a miss and a dead-token note all rendered
    assert "b7" in cold and "matched nothing" in cold
    assert "withcolor" in cold
