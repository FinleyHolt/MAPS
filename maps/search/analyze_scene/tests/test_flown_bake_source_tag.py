"""A bake written before the release stamped ``source`` on every prior row;
the as-flown replays read such bakes, so the surface never serves that key."""

from types import SimpleNamespace

from maps.search.analyze_scene.surface import _row_tags


def test_a_flown_bake_s_source_stamp_is_not_served():
    node = SimpleNamespace(tags={"source": "osm", "amenity": "school", "levels": 2})
    assert _row_tags(node) == {"amenity": "school"}
