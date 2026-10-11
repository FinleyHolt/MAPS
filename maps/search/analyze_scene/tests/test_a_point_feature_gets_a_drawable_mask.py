"""A row's ``mask`` is what the picture DRAWS, so it has to be a shape."""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import build_surface

_CELL_M = 2.0
_ORIGIN = (0.0, 0.0)

# THE DRAWABLE FLOOR, and it is the renderers' own: `_proj_poly` returns
# None below three vertices, so a shorter ring is a footprint no picture
# can ever show. Named here rather than spelled 3 in an assert, because
# the number belongs to the render layer.
_DRAWABLE_VERTS = 3


class _PointNode:
    """An OSM POI node: its exact contour IS the point it sits at."""

    def __init__(self, x, y, name=None):
        self.world_contour = [(float(x), float(y))]
        self.name = name
        self.tags = {"amenity": "pharmacy"}
        self.class_name = ""
        self.centerline = None
        self.world_xy = None
        self.cells = [(int(y // _CELL_M), int(x // _CELL_M))]
        self.centroid = (self.cells[0][0], self.cells[0][1])


class _WayNode:
    """A way with a real ring — the case that must NOT change."""

    def __init__(self, ring, name=None):
        self.world_contour = [(float(x), float(y)) for x, y in ring]
        self.name = name
        self.tags = {"building": "yes"}
        self.class_name = ""
        self.centerline = None
        self.world_xy = None
        xs = [p[0] for p in self.world_contour]
        ys = [p[1] for p in self.world_contour]
        self.cells = [(int(yy // _CELL_M), int(xx // _CELL_M))
                      for yy in range(int(min(ys)), int(max(ys)) + 1, 2)
                      for xx in range(int(min(xs)), int(max(xs)) + 1, 2)]
        self.centroid = (self.cells[0][0], self.cells[0][1])


def _row(node, fid="prior_1"):
    _c, anchors = build_surface([], [(fid, node)], _ORIGIN, _CELL_M)
    assert anchors, "the fixture produced no row at all"
    return anchors[0]


def test_a_point_feature_serves_a_mask_a_picture_can_draw():
    row = _row(_PointNode(4659.6, 2965.7, name="Rite Aid"))
    mask = row.get("mask")
    assert mask, "a point feature must still carry a footprint"
    # THE CONTRACT: at least one loop the renderer can actually draw.
    assert max(len(ch) for ch in mask) >= _DRAWABLE_VERTS, mask
    # ...and it is the node's OWN cell, so the drawn square is where the
    # thing is rather than a shape invented for it.
    xs = [p[0] for ch in mask for p in ch]
    ys = [p[1] for ch in mask for p in ch]
    assert min(xs) <= 4659.6 <= max(xs), mask
    assert min(ys) <= 2965.7 <= max(ys), mask
    # ONE CELL, so the served point cannot drift far from the vertex the
    # feature declared: half a cell diagonal is the whole budget.
    assert max(xs) - min(xs) <= _CELL_M + 1e-6, mask
    assert (row["xy"][0] - 4659.6) ** 2 + (row["xy"][1] - 2965.7) ** 2 \
        <= (_CELL_M * 0.75) ** 2, row["xy"]


def test_a_real_ring_still_serves_its_exact_geometry():
    ring = [(100.0, 200.0), (140.0, 200.0), (140.0, 230.0), (100.0, 230.0)]
    row = _row(_WayNode(ring, name="A Building"))
    mask = row.get("mask")
    assert mask, mask
    got = {(round(p[0], 2), round(p[1], 2)) for ch in mask for p in ch}
    # every declared vertex survives verbatim -- this is the exact
    # geometry, not a raster approximation of it
    assert {(x, y) for x, y in ring} <= got, (ring, sorted(got))
