"""A row names ONE PLACE, on this map, and the drone flies to where it says.

Three properties that were each false, measured on one 500-episode calcity
run:

1. A NAME IS NOT A PLACE. Folding same-named ways on the name alone made
   117 of 353 named features one row over several disconnected places.
   "85th Street" was ten streets in a grid city; the row's position was on
   none of them. 313 of 2,271 prior flights (13.8%) landed on such a row and
   the 86 episodes that flew one scored 0.291 against a 0.516 run.

2. THE MAP IS THE WORLD. The bake clips a feature's CELLS to its grid and
   leaves its ``world_contour`` unclipped, so 72 of 353 named features
   carried geometry off the tile. "Neuralia Road" reached y 23,745 on a tile
   6,297 m tall, and because ``representative_point`` resolves several
   chains on the LONGEST one, the road's served position was the arc
   midpoint of a 19 km way at y 14,198 -- north of the map.

3. ONE POSITION PER ROW. The row served ``representative_point``; the
   flight took ``_grid_centroid_to_world_xy(node.centroid)``, the int
   cell-mean of a merged feature's lead way. 17.3% of prior flights landed
   more than 20 m from the ``xy`` their own row had just served, 11.4% more
   than a kilometre, the worst 11,149 m.

The bake itself is faithful and is not what these fix: all 64 "Neuralia
Road" ways were checked against the live OSM API -- 0 vertex-count
mismatches, and all 64 carry ``name=Neuralia Road`` upstream, the 24
``service=alley`` ones included.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from maps.search.analyze_scene import cell_mesh as M
from maps.search.analyze_scene import geom_compute as G
from maps.search.analyze_scene.geom_program import build_surface

# THE bake a Cal City run would use: ``SEARCH_CALCITY_TILES`` wins, else the
# package default -- the rule ``benchmarks.calcity.paths`` applies, read off
# the env directly because ``maps`` never imports ``benchmarks``.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_TILES = Path(os.environ.get("SEARCH_CALCITY_TILES")
              or os.path.join(_REPO, "benchmarks", "calcity", "tiles"))
_HAS_BAKE = (_TILES / "calcity_features.json").is_file()


class _Way:
    """The bit of MapObject the surface builder reads."""

    def __init__(self, name, cells, contour=None):
        self.name = name
        self.cells = np.asarray(cells, dtype=int).reshape(-1, 2)
        self.tags = {}
        self.centroid = (float(self.cells[:, 0].mean()),
                         float(self.cells[:, 1].mean()))
        self.world_xy = None
        self.centerline = None
        self.class_name = ""
        self.world_contour = contour
        self.cell_count = len(self.cells)


def _row(cells_cols, row=10):
    return [[row, c] for c in cells_cols]


# --- 1. a name is not a place ----------------------------------------------


def test_two_places_under_one_name_are_two_rows():
    _c, anchors = build_surface(
        [], [("prior_1", _Way("85th Street", _row(range(0, 10)))),
             ("prior_2", _Way("85th Street", _row(range(400, 410))))],
        (0.0, 0.0), 2.0)
    assert len(anchors) == 2, [a["fid"] for a in anchors]
    assert {a["name"] for a in anchors} == {"85th Street"}


def test_unnamed_features_are_never_folded_together():
    """There is nothing to fold them ON. Folding adjacent unnamed masks
    would put every abutting building in one row."""
    _c, anchors = build_surface(
        [], [("prior_1", _Way(None, [[0, 0]])),
             ("prior_2", _Way("", [[0, 1]]))], (0.0, 0.0), 2.0)
    assert len(anchors) == 2, [a["fid"] for a in anchors]


# --- 2. the map is the world -----------------------------------------------


def test_geometry_is_cut_at_the_edge_of_what_the_tile_covers():
    """A way whose contour runs off the tile keeps only its in-tile half.

    The node's own CELLS are the record of what this tile kept, so they are
    the clip: a feature wholly inside is untouched.
    """
    contour = [[1.0, 21.0], [1.0, 4000.0]]          # runs far north
    way = _Way("Neuralia Road", _row(range(0, 1), row=10), contour=contour)
    way.cells = np.asarray([[r, 0] for r in range(0, 20)], dtype=int)
    _c, anchors = build_surface([], [("prior_1", way)], (0.0, 0.0), 2.0)
    ys = [p[1] for ch in anchors[0]["mask"] for p in ch]
    assert max(ys) <= 44.0, ys        # 20 cells * 2 m + one cell of margin
    assert anchors[0]["xy"][1] <= 44.0, anchors[0]["xy"]


def test_a_feature_wholly_inside_is_not_touched():
    """VACUITY FLOOR: a clip that trims everything would pass the test above
    while deleting the map."""
    contour = [[1.0, 1.0], [1.0, 19.0]]
    way = _Way("Rea Avenue", _row(range(0, 1)), contour=contour)
    way.cells = np.asarray([[r, 0] for r in range(0, 10)], dtype=int)
    _c, anchors = build_surface([], [("prior_1", way)], (0.0, 0.0), 2.0)
    assert anchors[0]["mask"] == [[[1.0, 1.0], [1.0, 19.0]]], anchors[0]["mask"]


def test_clipping_preserves_exact_vertex_identity():
    """The partition rule one module over is exact vertex identity with no
    tolerance. Interpolating an untouched segment's own endpoints -- ``a +
    (b-a)*1.0`` is not bit-identical to ``b`` -- cut 51 rows into "separate
    places" whose pieces touched at a MEDIAN gap of 0.0 m."""
    box = (0.0, 0.0, 100.0, 100.0)
    chains = [[[10.0, 10.0], [30.3, 40.7], [50.0, 20.0]]]
    out = G.clip_chains_to_box(chains, box)
    assert out == chains, out


# --- 3. one position per row ------------------------------------------------


def test_the_served_point_is_on_the_row_it_belongs_to():
    _c, anchors = build_surface(
        [], [("prior_1", _Way("85th Street", _row(range(0, 10)))),
             ("prior_2", _Way("85th Street", _row(range(400, 410))))],
        (0.0, 0.0), 2.0)
    for r in anchors:
        gap, _a, _b = G.gap_witness(G.mask_geom(r["mask"]),
                                    G.mask_geom([[list(r["xy"])]]))
        assert gap <= 0.01, (r["fid"], r["xy"], gap)


# --- the real bake: the vacuity floor under all of it -----------------------


@pytest.mark.skipif(not _HAS_BAKE, reason="calcity bake not present")
def test_the_real_bake_lands_every_row_on_the_map():
    """A rule that fires only on fixtures is a rule that fires on nothing."""
    from maps.search.analyze_scene.geom_program import build_surface_from_graph
    from maps.sws.graph.builder import build_object_graph

    meta = json.loads((_TILES / "calcity_meta.json").read_text())
    b = meta["bounds"]
    cell = float(meta["cell_size_m"])
    graph = build_object_graph(
        json.loads((_TILES / "calcity_features.json").read_text()),
        cell_size_m=cell)
    _c, anchors = build_surface_from_graph(
        graph.nodes.items(), (float(b["x_min"]), float(b["y_min"])), cell)

    assert len(anchors) > 4000, f"implausibly few rows: {len(anchors)}"

    # EVERY SERVED POINT IS INSIDE THE DECLARED BOUNDS, exactly -- no
    # tolerance, because this is the number the model reasons on and the
    # drone flies to. Was 6 rows outside, the worst 7,901 m north of the
    # north edge.
    out = [r for r in anchors
           if not (b["x_min"] <= r["xy"][0] <= b["x_max"]
                   and b["y_min"] <= r["xy"][1] <= b["y_max"])]
    assert not out, [(r["fid"], r.get("name"), r["xy"]) for r in out[:5]]

    # And every VERTEX is within one cell of the tile GRID -- the weaker
    # claim, and the honest one: the clip box is the node's own cells grown
    # by a cell, and the grid itself overhangs the declared bounds by up to
    # a cell. Was 72 named features carrying geometry off the tile entirely,
    # "Neuralia Road" to y 23,745.
    gx = float(meta["grid_shape"][1]) * cell + cell
    gy = float(meta["grid_shape"][0]) * cell + cell
    far = [r for r in anchors
           if any(not (-cell <= p[0] <= gx and -cell <= p[1] <= gy)
                  for ch in (r.get("mask") or []) for p in ch)]
    assert not far, [(r["fid"], r.get("name")) for r in far[:5]]

    by_name: dict = {}
    for r in anchors:
        if r.get("name"):
            by_name.setdefault(r["name"], []).append(r)

    # PINNED BY NAME, because a scan cannot notice its own disappearance.
    # A name is the SET of its ways, and every way is a row of its own.
    assert len(by_name.get("Neuralia Road", [])) == 67, "67 ways, 67 rows"
    assert len(by_name.get("Poppy Boulevard", [])) == 16, "16 ways, 16 rows"
    # 13 ways over the streets that share this name in a grid city; the
    # mapper split some of them, and each split piece is a row of its own.
    assert len(by_name.get("85th Street", [])) == 13, "13 ways, 13 rows"

    # Most names ARE one way, so most are one row: 296 of the bake's named
    # features, against the 212 names that span several ways. Both sides
    # matter -- all-one-row would mean nothing multi-way survived, and
    # none-one-row would mean rows stopped tracking ways.
    single = sum(1 for v in by_name.values() if len(v) == 1)
    assert single > 150, f"only {single} names are one row"
    multi = sum(1 for v in by_name.values() if len(v) > 1)
    assert multi > 100, f"only {multi} names span several rows"


# --- the flight flies the row's own point -----------------------------------


def test_the_flown_pose_is_the_row_s_served_xy():
    """The half nothing pinned. ``test_served_xy_is_on_its_geometry`` puts
    the served point on the geometry; this puts the DRONE on the served
    point. They were separate quantities from separate code paths, and over
    one calcity run 17.3% of prior flights landed more than 20 m from what
    their own row said, 11.4% more than a kilometre, the worst 11,149 m.
    """
    from maps.search.search_loop import SearchLoop

    ways = [("prior_1", _Way("Lime Avenue", _row(range(0, 10)))),
            ("prior_2", _Way("Lime Avenue", _row(range(10, 20)))),
            # A bent way: its cell mean is off its own geometry, which is
            # what made the two rules disagree in the first place.
            ("prior_3", _Way("Bent Road",
                             [[r, 40] for r in range(0, 20)]
                             + [[19, c] for c in range(40, 60)]))]

    class _Graph:
        nodes = dict(ways)

    # ``_cell_size_m`` is a read-only property sourced from the world
    # state's grid_domain -- the single source of truth for the
    # centroid->world conversion -- so the fixture supplies that rather
    # than shadowing the property.
    loop = SearchLoop.__new__(SearchLoop)
    loop._tile_origin_xy = (0.0, 0.0)
    loop._world_state = type("W", (), {
        "grid_domain": type("D", (), {"cell_size_m": 2.0})()})()
    loop._lock = __import__("threading").RLock()
    loop._state = type("S", (), {"runtime_graph": _Graph()})()

    _c, anchors = build_surface([], ways, (0.0, 0.0), 2.0)
    assert anchors, "fixture built no rows"
    for r in anchors:
        flown = loop.world_xy_for_node(r["fid"])
        assert flown is not None, r["fid"]
        assert (round(flown[0], 2), round(flown[1], 2)) == (
            round(r["xy"][0], 2), round(r["xy"][1], 2)), (r["fid"], flown, r["xy"])

    # VACUITY FLOOR: the rule this replaced really did disagree, so the
    # assertion above is not comparing a quantity to itself.
    bent = next(r for r in anchors if r["name"] == "Bent Road")
    node = dict(ways)[bent["fid"]]
    old = loop._grid_centroid_to_world_xy(node.centroid)
    assert old is not None
    off = ((old[0] - bent["xy"][0]) ** 2 + (old[1] - bent["xy"][1]) ** 2) ** 0.5
    assert off > 5.0, (old, bent["xy"], off)
