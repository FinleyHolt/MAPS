"""The served surface orders fids NUMERICALLY, never as strings.

Lexicographic collation put every detected_1xx ahead of detected_2, so a
pass that minted ~110 rows buried its earliest stops' finds past every
served head (the 8-crop catch, the 24-row broad head, r1[0:20]) --
140#138's GT lot sat on the surface all episode at string-index 37,
unreachable by position. Numeric order is mint order, which for a pass
is stop order.
"""
from __future__ import annotations

import numpy as np

from maps.search.analyze_scene.geom_program import build_surface_from_graph
from maps.search.tests.fakes import plan_of_block


class _Node:
    def __init__(self, i):
        self.world_xy = [float(10 * i), 0.0]
        self.centroid = (0, int(10 * i))
        self.centerline = None
        self.class_name = "parking lot"
        self.name = None
        self.provenance = "detected"


def test_detected_2_serves_before_detected_100():
    nodes = {f"detected_{i}": _Node(i) for i in range(120)}
    cands, _anchors = build_surface_from_graph(
        nodes.items(), (0.0, 0.0), 2.0)
    fids = [c["fid"] for c in cands]
    assert fids[:8] == [f"detected_{i}" for i in range(8)], fids[:8]
    assert fids.index("detected_2") == 2, fids.index("detected_2")
    # the whole order is the mint order
    assert fids == [f"detected_{i}" for i in range(120)], fids[110:]


def test_a_names_ways_serve_their_shape_and_not_a_tied_table():
    """A street's initial reply is what it IS, not an arbitrary prefix.

    Every way of a name ties at the name tier, so a rendered head is N rows
    in an order the envelope itself calls arbitrary -- and the model reads
    row 0 as the answer (calcity::284#282 took ``street_88[0]`` and swept 1
    of 10 segments of 88th Street).

    The reply is the object: a name, a length, a len. No prose -- the
    caller writes Python and holds the handle, so a sentence naming
    explore() or rN[a:b] is legend, and the tie caveat only ever existed
    to warn about a table this no longer serves.
    """
    from maps.search.analyze_scene.geom_program import (
        _NAME_SET_SHAPE_ROWS,
        build_surface,
        compact_results,
        run_program,
    )

    class _Way:
        def __init__(self, name, cells):
            self.name = name
            self.cells = np.asarray(cells, dtype=int).reshape(-1, 2)
            self.tags = {}
            self.centroid = (float(self.cells[:, 0].mean()),
                             float(self.cells[:, 1].mean()))
            self.world_xy = None
            # A linear feature: the bake gives a road a centerline, and
            # axis_length_m is served only for a node that has one.
            self.centerline = [[int(r), int(c)] for r, c in self.cells]
            self.class_name = ""
            self.world_contour = None
            self.cell_count = len(self.cells)
            self.axis_length_m = 20.0

    n = _NAME_SET_SHAPE_ROWS + 2
    ways = [(f"prior_{i}", _Way("Neuralia Road",
                                [[10, c] for c in range(i * 20, i * 20 + 10)]))
            for i in range(n)]
    _c, anchors = build_surface([], ways, (0.0, 0.0), 2.0)
    out = run_program([], [dict(a) for a in anchors],
                      {"ops": [{"ref": "r1", "op": "find",
                                "args": {"query": "Neuralia Road"}}]})
    head = compact_results(out["results"])["r1"]

    assert head["len"] == n
    # No head to mistake for a ranking, and no fids to hand-copy.
    assert "rows" not in head, head
    assert "fids" not in head, head
    # The shape, as fields.
    assert head["name"] == "Neuralia Road", head
    assert head["length_m"] > 0, head
    # NO PROSE. Not a tie sermon, not a "Best:" endorsement, not an
    # instruction naming the op to call next.
    assert "hint" not in head, head["hint"]
    assert set(head) == {"len", "name", "length_m"}, sorted(head)

    # A SMALL set still renders: the table is cheap and a reader holds it.
    few = ways[:_NAME_SET_SHAPE_ROWS]
    _c, anchors = build_surface([], few, (0.0, 0.0), 2.0)
    out = run_program([], [dict(a) for a in anchors],
                      {"ops": [{"ref": "r1", "op": "find",
                                "args": {"query": "Neuralia Road"}}]})
    small = compact_results(out["results"])["r1"]
    assert "rows" in small, small
    assert len(small["fids"]) == len(few), small


def _crops_of(src: str) -> list:
    """The crop steps one code block compiles to.

    The surface has no parser: a block becomes steps only by EXECUTING,
    so the plan is captured off a recording ``MissionEnv``.
    """

    plan = plan_of_block(src)
    return [st for p in plan.get("plans", [plan])
            for st in p["steps"] if st.get("do") == "crops"]


def test_look_marks_a_street_by_its_handle_not_a_piece_of_it():
    """``marks=`` names what ELSE bears on the choice, and a street is a SET.

    calcity::200#198 explored all of Poppy Boulevard, ranked its cars by
    redness, looked at its own top ten, and committed the reddest -- 116 m
    from Poppy and 11.9 m off Jimson Avenue. Every crop legend read
    ``1 = TARGET car detected_N (center)`` and nothing else: the renderer
    writes a named road's NAME on the picture, but only for a road in the
    mark list, and a bare look supplies none.

    The handle expands to its ways, because ``marks=poppy[0]`` would put
    back the arbitrary-piece index the surface exists to prevent.
    """
    crops = _crops_of('poppy = find("Poppy Boulevard")\n'
                      'cars = find("car")\n'
                      'look(cars[0:10], marks=poppy)\n')
    assert crops, "no crops step compiled"
    # The mark rides the SAME crop request as its subject.
    assert crops[-1].get("mark_fids"), crops[-1]


def test_look_without_marks_asks_for_none():
    """The default is unchanged: a bare look carries no mark list."""
    crops = _crops_of('cars = find("car")\nlook(cars[0:10])\n')
    assert crops, "no crops step compiled"
    assert "mark_fids" not in crops[-1], crops[-1]
