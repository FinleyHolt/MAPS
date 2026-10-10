"""Provenance travels with the tiles.

Every bake writes ``meta.json`` ``bake.mode`` + ``provenance`` (prior kind,
source identity, fetch/bake times, git sha) so a tile dir always says what
it is and where it came from — and a mode switch can never inherit the
previous bake's claim (``_STALE_BAKE_FLAGS``).
"""

from __future__ import annotations

import json

from benchmarks.citynav.bake.city import bake_osm_blocks


def _seed_meta(tmp_path, block="birmingham_block_0"):
    meta = {
        "map_name": block,
        "bounds": {"x_min": 0.0, "y_min": 0.0, "x_max": 40.0, "y_max": 40.0},
        "grid_shape": [20, 20],
        "cell_size_m": 2.0,
        # A stale claim from a previous cityrefer bake: must NOT survive.
        "provenance": {"prior": "cityrefer"},
        "cityrefer_landmark_priors": True,
    }
    (tmp_path / f"{block}_meta.json").write_text(json.dumps(meta))
    return block


def _records():
    poly = {
        "type": "Polygon",
        "coordinates": [[[2, 2], [10, 2], [10, 10], [2, 10], [2, 2]]],
    }
    return [({"building": "yes", "osm_id": "w1"}, poly)]


def test_osm_bake_stamps_provenance(tmp_path):
    block = _seed_meta(tmp_path)
    bake_osm_blocks(
        "birmingham",
        _records(),
        tiles_dir=tmp_path,
        provenance={
            "prior": "osm",
            "osm_cache": "birmingham_osm_fresh.geojson",
            "osm_cache_sha256": "ab" * 32,
            "fetched_at_utc": "2026-07-19T03:00:00+00:00",
            "records": 1,
        },
    )
    meta = json.loads((tmp_path / f"{block}_meta.json").read_text())
    assert meta["bake"]["mode"] == "osm_blocks"
    prov = meta["provenance"]
    assert prov["prior"] == "osm"
    assert prov["osm_cache"] == "birmingham_osm_fresh.geojson"
    assert prov["osm_cache_sha256"] == "ab" * 32
    assert prov["fetched_at_utc"] == "2026-07-19T03:00:00+00:00"
    # The bake adds WHO/WHEN on top of the source identity.
    assert prov["baked_at_utc"]
    assert "bake_git_sha" in prov
    # The stale cityrefer claims from the seed meta are gone.
    assert "cityrefer_landmark_priors" not in meta


def test_mode_switch_cannot_inherit_stale_provenance(tmp_path):
    block = _seed_meta(tmp_path)
    bake_osm_blocks("birmingham", _records(), tiles_dir=tmp_path,
                    provenance={"prior": "osm"})
    meta = json.loads((tmp_path / f"{block}_meta.json").read_text())
    assert meta["provenance"]["prior"] == "osm", (
        "the seed meta's cityrefer provenance leaked through the rebake"
    )
