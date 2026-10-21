"""The study console's map routes off a download: a tile by its XYZ address out of the
MBTiles file (whose rows count from the south), the paper tone for a tile it lacks, and the AO."""

import sqlite3

from fastapi.testclient import TestClient

from baselines.human import study_server
from benchmarks.calcity.data_prep.aoi import AO_BBOX_WGS84


def test_tiles_by_xyz_and_the_ao(tmp_path) -> None:
    (tmp_path / "console").mkdir()
    (tmp_path / "console" / "index.html").write_text("<!doctype html>")
    db = sqlite3.connect(tmp_path / "calcity_ortho.mbtiles")
    db.execute("CREATE TABLE tiles (zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB)")
    db.execute("INSERT INTO tiles VALUES (14, 2900, ?, ?)", (2**14 - 1 - 6514, b"webp"))
    db.commit()
    db.close()
    c = TestClient(study_server.build_app(None, tmp_path, tmp_path / "console"))
    assert c.get("/study/tiles/14/2900/6514.webp").content == b"webp"
    assert c.get("/study/tiles/14/2900/6515.webp").status_code == 204
    assert c.get("/area").json()["bbox"] == list(AO_BBOX_WGS84)
    assert c.get("/").status_code == 200
