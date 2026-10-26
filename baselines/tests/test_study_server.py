"""The study console's routes off a download: a tile by its XYZ address out of the MBTiles file
(whose rows count from the south), the paper tone for a tile it lacks, and the AO, from any
folder name; a plain answer for every route asked before a session or a label exists; the
OSM kinds in plain words; the command line; a second start while the first still serves."""

import json
import socket
import sqlite3

import pytest
from fastapi.testclient import TestClient

from baselines.human import study_server
from baselines.human.study_server import Visit
from benchmarks.calcity.data_prep.aoi import AO_BBOX_WGS84


def _client(kit):
    (kit / "console").mkdir(parents=True)
    (kit / "console" / "index.html").write_text("<!doctype html>")
    (kit / "plan").mkdir()
    (kit / "plan" / "plan.json").write_text(json.dumps({"participants": {"p01": {"first": "solo"}}}))
    db = sqlite3.connect(kit / "calcity_ortho.mbtiles")
    db.execute("CREATE TABLE tiles (zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB)")
    db.execute("INSERT INTO tiles VALUES (14, 2900, ?, ?)", (2**14 - 1 - 6514, b"webp"))
    db.commit()
    db.close()
    return TestClient(study_server.build_app(kit, kit / "console", Visit(kit)))


@pytest.mark.parametrize("name", ["kit", "odd #1 (1)", "100%25 x"])
def test_tiles_by_xyz_and_the_ao(tmp_path, name) -> None:
    c = _client(tmp_path / name)
    assert c.get("/study/tiles/14/2900/6514.webp").content == b"webp"
    assert c.get("/study/tiles/14/2900/6515.webp").status_code == 204
    assert c.get("/area").json()["bbox"] == list(AO_BBOX_WGS84)


def test_before_a_session_or_a_label(tmp_path) -> None:
    c = _client(tmp_path)
    for r in (c.get("/capacity/state"), c.post("/capacity/act", json={"action": "skip"})):
        assert r.status_code == 409 and r.json()["detail"]
    for path in ("/", "/index.html"):
        assert c.get(path, follow_redirects=False).headers["location"] == "/visit"
    r = c.get("/visit/results.zip")
    assert r.status_code == 404 and "/visit" in r.text


def test_osm_kinds_in_plain_words() -> None:
    kind = study_server._kind
    assert kind({"category": "roads", "highway": "residential"}) == "residential street"
    assert kind({"category": "landuse", "landuse": "residential"}) == "residential area"
    assert kind({"category": "landuse", "leisure": "swimming_pool"}) == "swimming pool"
    assert kind({"category": "pois", "shop": "doityourself"}) == "hardware store"
    assert kind({"category": "buildings"}) == "building"
    assert kind({"category": "pois", "amenity": "clinic"}) == "clinic"


def test_the_command_line_takes_no_participant() -> None:
    with pytest.raises(SystemExit) as e:
        study_server.main(["p01"])
    assert e.value.code == 2


def test_a_second_start_points_at_the_first(tmp_path, capsys) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        assert study_server.main(["--kit", str(tmp_path), "--port", str(s.getsockname()[1]), "--no-browser"]) == 0
    assert "already running" in capsys.readouterr().out
