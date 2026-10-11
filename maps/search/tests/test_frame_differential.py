"""Differential battery for the chart/compass laws — shapely as oracle."""
from __future__ import annotations

import math
import random

from shapely.geometry import LineString, Point, Polygon

from maps.search.analyze_scene.geom_compute import (
    COMPASS_UNIT,
    bearing,
    cardinal,
    contour_shape_features,
    frame_polyline,
)

_N = 200


def _az(dx: float, dy: float) -> float:
    """Compass azimuth, written independently: 0 = north (+y), cw."""
    return math.degrees(math.atan2(dx, dy)) % 360.0


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _rand_polyline(r: random.Random) -> list[list[float]]:
    n = r.randint(2, 6)
    x, y = r.uniform(-50, 50), r.uniform(-50, 50)
    pts = [[x, y]]
    heading = r.uniform(0, 2 * math.pi)
    for _ in range(n - 1):
        heading += r.uniform(-0.8, 0.8)
        step = r.uniform(15, 60)
        x, y = x + step * math.cos(heading), y + step * math.sin(heading)
        pts.append([x, y])
    return pts


def test_chart_along_off_match_shapely_and_azimuth_law():
    r = _rng(1234)
    judged = rights = lefts = 0
    for _ in range(_N):
        poly = _rand_polyline(r)
        line = LineString(poly)
        # Sample the point OFF the line: interior arc position + lateral
        # offset, so nearly every case lands in the judged population (the
        # chart is deliberately UN-CLAMPED past the ends, where shapely's
        # clamped project disagrees — stay interior).
        t = r.uniform(0.1, 0.9) * line.length
        f0 = line.interpolate(t)
        ang = r.uniform(0, 2 * math.pi)
        rad = r.uniform(2.0, 40.0)
        pt = [f0.x + rad * math.cos(ang), f0.y + rad * math.sin(ang)]
        proj = line.project(Point(pt))
        if proj < 1.0 or proj > line.length - 1.0:
            continue
        rows, _receipt = frame_polyline([{"fid": "p", "xy": pt}], poly)
        row = rows[0]
        along, off = row["along"], row["off"]
        assert abs(along - proj) < 1.5, (poly, pt, along, proj)
        foot = line.interpolate(proj)
        d = math.hypot(pt[0] - foot.x, pt[1] - foot.y)
        assert abs(abs(off) - d) < 1.5, (pt, off, d)
        if d < 1.0:
            continue  # on-line: sign is noise
        # Independent sign law: right of travel = the half-plane swept
        # CLOCKWISE from the travel azimuth, i.e. (az_pt - az_travel) in
        # (0, 180) mod 360.
        eps = min(1.0, line.length / 100.0)
        ahead = line.interpolate(min(proj + eps, line.length))
        behind = line.interpolate(max(proj - eps, 0.0))
        az_travel = _az(ahead.x - behind.x, ahead.y - behind.y)
        az_pt = _az(pt[0] - foot.x, pt[1] - foot.y)
        expected_right = 0.0 < (az_pt - az_travel) % 360.0 < 180.0
        assert (off > 0) == expected_right, (poly, pt, off, az_travel, az_pt)
        judged += 1
        rights += int(expected_right)
        lefts += int(not expected_right)
    # Vacuity floor: the generator must actually exercise both signs.
    assert judged >= _N // 2, judged
    assert rights >= 10 and lefts >= 10, (rights, lefts)


def test_bearing_and_cardinal_pin_compass_convention():
    r = _rng(99)
    seen = set()
    for _ in range(_N):
        a = [r.uniform(-100, 100), r.uniform(-100, 100)]
        b = [r.uniform(-100, 100), r.uniform(-100, 100)]
        if math.hypot(b[0] - a[0], b[1] - a[1]) < 1e-6:
            continue
        want = _az(b[0] - a[0], b[1] - a[1])
        got = bearing(a, b)
        assert abs((got - want + 180.0) % 360.0 - 180.0) < 1e-9, (a, b)
        rev = bearing(b, a) - (got + 180.0)
        assert abs((rev + 180.0) % 360.0 - 180.0) < 1e-9, (a, b, rev)
        seen.add(cardinal(a, b))
    assert seen == {"north", "northeast", "east", "southeast",
                    "south", "southwest", "west", "northwest"}, seen
    # The four exact convention pins, stated once, in the open:
    assert bearing([0, 0], [0, 1]) == 0.0      # +y IS north
    assert bearing([0, 0], [1, 0]) == 90.0     # +x IS east
    for word, u in COMPASS_UNIT.items():
        assert cardinal([0, 0], list(u)) == word


def _rot(theta: float, p: list[float]) -> list[float]:
    c, s = math.cos(theta), math.sin(theta)
    return [c * p[0] - s * p[1], s * p[0] + c * p[1]]


def test_a_closed_ring_yields_area_and_an_open_way_yields_nothing():
    r = _rng(21)
    judged = 0
    for _ in range(80):
        # An elongated rectangle at a random azimuth.
        theta = r.uniform(0, math.pi)
        L, w = r.uniform(30, 80), r.uniform(2, 8)
        c, s = math.cos(theta), math.sin(theta)
        cx, cy = r.uniform(-50, 50), r.uniform(-50, 50)
        verts = []
        for sx, sy in ((-L, -w), (L, -w), (L, w), (-L, w), (-L, -w)):
            verts.append([cx + c * sx - s * sy, cy + s * sx + c * sy])
        feat = contour_shape_features([verts])
        assert feat is not None
        # The SAME vertices without the closing one are an open way, which
        # encloses nothing and therefore has no footprint to describe.
        assert contour_shape_features([verts[:-1]]) is None
        # No axis heading is produced: orientation is a relation to the
        # feature a thing faces, never a property of its own rectangle.
        assert set(feat) == {"footprint_area_m2"}, feat
        area = Polygon(verts).area
        assert abs(feat["footprint_area_m2"] - area) < max(1.0, 0.01 * area)
        judged += 1
    assert judged >= 60, judged
