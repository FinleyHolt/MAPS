"""Dedup and tag-carry contract for the objects.json build.

The pin that matters: OSM splits a street into many ways, and every one of them
must survive as its own object. A name-only dedup key kept the largest and
dropped the rest -- California City Boulevard baked as a 908 m fragment while
OSM names it across the whole map.
"""

from shapely.geometry import LineString, Polygon

from benchmarks.calcity.data_prep import build_objects_json as boj
from benchmarks.calcity.data_prep.aoi import BLOCKS

#: The AO this bench flies, read rather than restated. The four numbers that
#: stood here were the retired AOI's, which named 6.6 km of ground the bake no
#: longer covers: the builder frames whatever bbox it is handed, so the test
#: passed while exercising a block that does not exist.
BBOX = BLOCKS["calcity"]


def _rec(osm_id, tags, geom):
    return {"layer": "test", "osm_id": osm_id, "tags": tags, "geom": geom}


def _square(cx, cy, half=5.0):
    return Polygon(
        [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ]
    )


def _build(records, monkeypatch):
    monkeypatch.setattr(boj, "load_features_local", lambda _bundle, _lf: records)
    return boj.build_block(None, "calcity", BBOX)


def test_every_way_of_one_street_survives(monkeypatch):
    """Two disjoint ways sharing a name are two objects, not one."""
    street = {"highway": "residential", "name": "Main Street"}
    objects, _lm = _build(
        [
            _rec(1, street, LineString([(100, 100), (900, 100)])),
            _rec(2, street, LineString([(1100, 100), (1900, 100)])),
        ],
        monkeypatch,
    )
    assert len(objects) == 2
    assert [o["name"] for o in objects.values()] == ["Main Street"] * 2
    # ...and they are the two DISTINCT ways, not one way counted twice.
    assert len({o["position"][0] for o in objects.values()}) == 2


def test_colocated_duplicates_still_collapse(monkeypatch):
    """The dedup the name key was doing correctly: one shell tagged twice
    (buildings layer and pois layer) is still one object."""
    objects, _lm = _build(
        [
            _rec(1, {"building": "yes", "name": "City Hall"}, _square(500, 500)),
            _rec(2, {"amenity": "townhall", "name": "City Hall"}, _square(501, 500)),
        ],
        monkeypatch,
    )
    assert len(objects) == 1


def test_unnamed_features_are_unaffected(monkeypatch):
    """Two unnamed footprints far apart stay two; the anon key always had the
    bucket and its behaviour must not shift."""
    objects, _lm = _build(
        [
            _rec(1, {"building": "yes"}, _square(200, 200)),
            _rec(2, {"building": "yes"}, _square(800, 800)),
        ],
        monkeypatch,
    )
    assert len(objects) == 2


def test_every_osm_tag_rides_through(monkeypatch):
    """Tags outside the retired 7-key allowlist reach the object. `ref` and
    `lanes` named a road; `building` said why it classed as a Building."""
    objects, _lm = _build(
        [
            _rec(
                1,
                {
                    "highway": "primary",
                    "name": "California City Boulevard",
                    "ref": "SR-58",
                    "lanes": "4",
                    "surface": "asphalt",
                },
                LineString([(100, 300), (900, 300)]),
            )
        ],
        monkeypatch,
    )
    tags = next(iter(objects.values()))["tags"]
    assert tags["ref"] == "SR-58"
    assert tags["lanes"] == "4"
    assert tags["surface"] == "asphalt"
    assert tags["highway"] == "primary"


def test_landmark_is_the_largest_segment(monkeypatch):
    """A name now maps to many objects, so the one landmark entry is the
    biggest of them -- not whichever sorts first."""
    street = {"highway": "residential", "name": "Main Street"}
    objects, landmarks = _build(
        [
            _rec(1, street, LineString([(100, 100), (300, 100)])),   # short, west
            _rec(2, street, LineString([(1000, 100), (2600, 100)])),  # long, east
        ],
        monkeypatch,
    )
    assert len(objects) == 2
    biggest = max(objects.values(), key=lambda o: o["dimension"][0])
    assert landmarks["Main Street"]["id"] == biggest["id"]
