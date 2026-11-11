"""The Cal City prior bakes the bundle's OSM verbatim.

The contract is the CityNav baker's: every feature with a geometry is kept,
every tag rides, and a geometry kind stays what OSM said it was.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.calcity.data_prep import bake_tiles
from benchmarks.calcity.data_prep.bake_tiles import bake_block
from maps.world_state.prior_graph import build_object_graph


def _bundle(monkeypatch, features):
    """Stand in for the world bundle reader with a literal feature list."""
    monkeypatch.setattr(
        bake_tiles,
        "load_features_local",
        lambda bundle, lf: [
            {
                "layer": f.get("layer", "test"),
                "osm_id": f["osm_id"],
                "tags": f["tags"],
                "geom": f["geom"],
            }
            for f in features
        ],
    )


def _road(osm_id, tags, coords):
    from shapely.geometry import LineString

    return {"osm_id": osm_id, "tags": tags, "geom": LineString(coords)}


def _area(osm_id, tags, x0, y0, x1, y1):
    from shapely.geometry import Polygon

    return {
        "osm_id": osm_id,
        "tags": tags,
        "geom": Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]),
    }


def test_a_road_bakes_as_a_way_not_a_buffered_polygon(tmp_path, monkeypatch):
    """The projection buffered every line into a 4 m ring, so the bake held
    3890 areas and not one way. A LineString must stay a way."""
    _bundle(
        monkeypatch,
        [_road(1, {"highway": "secondary", "name": "South Loop Boulevard"},
               [(100, 100), (900, 100)])],
    )
    bake_block(tmp_path / "bundle", tmp_path / "tiles", "calcity")
    prior = json.loads((tmp_path / "tiles" / "calcity_features.json").read_text())["prior"]
    assert len(prior) == 1 and prior[0]["name"] == "South Loop Boulevard"
    chain = prior[0]["world_contour"]
    assert len(chain) == 2 and chain[0] != chain[-1], "an open chain is a way"


def test_a_feature_with_no_cityrefer_class_is_baked(tmp_path, monkeypatch):
    """30% of raw OSM records carry no CityRefer class -- 76% of every
    LineString. Each bakes."""
    dropped = [
        _road(1, {"highway": "footway"}, [(100, 100), (400, 100)]),
        _road(2, {"highway": "service"}, [(100, 200), (400, 200)]),
        _road(3, {"highway": "cycleway"}, [(100, 300), (400, 300)]),
        _road(4, {"railway": "rail"}, [(100, 400), (400, 400)]),
        _road(5, {"waterway": "stream"}, [(100, 500), (400, 500)]),
    ]
    _bundle(monkeypatch, dropped)
    assert bake_block(tmp_path / "bundle", tmp_path / "tiles", "calcity") == len(dropped)


def test_every_tag_rides_through(tmp_path, monkeypatch):
    """ref / lanes / surface / building ride with every other tag."""
    _bundle(
        monkeypatch,
        [_area(9, {"building": "house", "addr:street": "Hacienda Boulevard",
                   "roof:shape": "gabled"}, 100, 100, 140, 140)],
    )
    bake_block(tmp_path / "bundle", tmp_path / "tiles", "calcity")
    prior = json.loads(
        (tmp_path / "tiles" / "calcity_features.json").read_text()
    )["prior"][0]
    # The baker colon-escapes keys on the way in.
    assert prior["building"] == "house"
    assert prior["addr_street"] == "Hacienda Boulevard"
    assert prior["roof_shape"] == "gabled"


def test_every_way_of_one_street_is_its_own_row(tmp_path, monkeypatch):
    """Name-keyed dedup kept the longest segment and dropped the rest."""
    street = {"highway": "primary", "name": "California City Boulevard"}
    _bundle(
        monkeypatch,
        [
            _road(1, street, [(100, 100), (900, 100)]),
            _road(2, street, [(1100, 100), (1900, 100)]),
        ],
    )
    assert bake_block(tmp_path / "bundle", tmp_path / "tiles", "calcity") == 2


def test_an_empty_bundle_is_an_error_not_an_empty_bake(tmp_path, monkeypatch):
    """A bake that silently produced nothing would read as a valid one."""
    _bundle(monkeypatch, [])
    with pytest.raises(ValueError, match="no OSM features"):
        bake_block(tmp_path / "bundle", tmp_path / "tiles", "calcity")


def test_the_bake_shape_and_provenance_are_unchanged(tmp_path, monkeypatch):
    _bundle(
        monkeypatch,
        [_area(7, {"leisure": "pitch", "name": "Central Park"}, 100, 100, 200, 200)],
    )
    tiles = tmp_path / "tiles"
    assert bake_block(tmp_path / "bundle", tiles, "calcity") == 1
    features = json.loads((tiles / "calcity_features.json").read_text())
    assert list(features) == ["prior"]
    assert sorted(p.name for p in tiles.iterdir()) == ["calcity_features.json", "calcity_meta.json"]
    meta = json.loads((tiles / "calcity_meta.json").read_text())
    graph = build_object_graph(features, cell_size_m=float(meta["cell_size_m"]))
    assert list(graph.nodes) == ["prior_0"]
    assert list(meta) == ["grid_shape", "cell_size_m", "bounds", "coordinate_frame", "georef"]
    assert meta["coordinate_frame"] == {"y_axis": "north"}


def test_ground_truth_keys_can_never_enter_the_prior(tmp_path, monkeypatch):
    """The leak wall: properties ride through, ground-truth keys never do."""
    _bundle(
        monkeypatch,
        [_area(7, {"name": "X", "descriptions": ["leak"], "target_id": 3,
                   "ann_id": 9}, 100, 100, 140, 140)],
    )
    prior = json.loads(
        (
            bake_block(tmp_path / "bundle", tmp_path / "tiles", "calcity")
            and (tmp_path / "tiles" / "calcity_features.json")
        ).read_text()
    )["prior"][0]
    for leaked in ("descriptions", "target_id", "ann_id", "cityref_id"):
        assert leaked not in prior, f"{leaked} reached the prior"


def test_the_bake_reads_no_citynav_raster_and_keeps_the_bounds_rectangle(tmp_path, monkeypatch):
    """Cal City has no CityNav raster: its coverage is the bounds rectangle by
    rule, so a building keeps every cell its footprint covers."""
    from benchmarks.citynav import paths as citynav_paths

    features = [_road(1, {"highway": "residential", "name": "Elm"}, [(100, 100), (400, 120)]),
                _area(2, {"building": "yes"}, 200, 200, 260, 250)]
    _bundle(monkeypatch, features)

    class Untouchable:
        def __fspath__(self):
            raise AssertionError("the Cal City bake reached into CityNav's raster directory")

        __str__ = __truediv__ = __fspath__

    monkeypatch.setattr(citynav_paths, "CITYNAV_NATIVE_RGBD_DIR", Untouchable())
    assert bake_block(tmp_path / "bundle", tmp_path / "tiles", "calcity") == 2
    prior = json.loads((tmp_path / "tiles" / "calcity_features.json").read_text())["prior"]
    building = next(r for r in prior if r.get("building") == "yes")
    assert len(building["cells"]) == 30 * 25


def test_a_bundle_missing_a_layer_is_refused_by_name(tmp_path):
    """A bundle without its roads file would bake a map with no roads."""
    from benchmarks.calcity.data_prep.bundle import OSM_LAYERS, iter_features

    (tmp_path / "osm").mkdir()
    for layer in OSM_LAYERS:
        if layer != "roads":
            (tmp_path / "osm" / f"{layer}.geojson").write_text('{"features": []}')
    with pytest.raises(FileNotFoundError, match="roads.geojson"):
        list(iter_features(tmp_path))
