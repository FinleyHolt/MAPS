"""The study console's routes off a download: a tile by its XYZ address out of the MBTiles file
(whose rows count from the south), the paper tone for a tile it lacks, and the AO, from any
folder name; a plain answer for every route asked before a session or a label exists; a running
session open again on a relaunch; a map click that keeps its report's id; OSM kinds in plain
words; the command line; a second start while the first still serves, and another program on
the port."""

import json
import socket
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from baselines.human import study_server
from baselines.human.study_server import Visit
from benchmarks.calcity.data_prep.aoi import AO_BBOX_WGS84


def _client(kit):
    (kit / "console").mkdir(parents=True)
    (kit / "console" / "index.html").write_text("<!doctype html>")
    (kit / "plan").mkdir()
    (kit / "plan" / "plan.json").write_text(json.dumps({"participants": {"p01": {"first": "solo"}, "p02": {"first": "team"}}}))
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


def test_a_select_keeps_its_report(tmp_path, monkeypatch) -> None:
    import benchmarks.flight.tile_georef as georef

    monkeypatch.setattr(georef, "lonlat_to_local", lambda tiles, name, lon, lat: (lon * 2, lat * 2))
    acts = []

    class Session:
        def state(self) -> dict:
            return {}

        def act(self, req: dict) -> dict:
            acts.append(req)
            return {"ok": True}

    monkeypatch.setattr(study_server, "open_session", lambda kit, participant, condition: Session())
    c = _client(tmp_path)
    c.get("/visit/label?label=p01")
    c.post("/visit/start/practice")
    c.post("/capacity/act", json={"action": "select", "lon": 1.0, "lat": 2.0, "episode_id": "e1"})
    assert acts == [{"action": "select", "episode_id": "e1", "x": 2.0, "y": 4.0}]
    c.get("/visit/label?label=p02")
    assert c.get("/capacity/state").status_code == 409


def test_a_relaunch_reopens_a_running_session(tmp_path, monkeypatch) -> None:
    class Session:
        def state(self) -> dict:
            return {"current": None}

    monkeypatch.setattr(study_server, "open_session", lambda kit, participant, condition: Session())
    for step in ("practice", "solo"):
        (tmp_path / "results" / "p01" / step).mkdir(parents=True)
    (tmp_path / "results" / "p01" / "solo" / "session.json").write_text(json.dumps(
        {"started_epoch": time.time(), "session_s": 2400.0, "episodes_n": 3}))
    c = _client(tmp_path)
    assert c.get("/capacity/state").json() == {"current": None, "practice": False}
    assert "Its clock kept running while you were away." in c.get("/visit").text


def test_osm_kinds_in_plain_words() -> None:
    kind = study_server._kind
    assert kind({"category": "roads", "highway": "residential"}) == "residential street"
    assert kind({"category": "landuse", "landuse": "residential"}) == "residential area"
    assert kind({"category": "landuse", "leisure": "swimming_pool"}) == "swimming pool"
    assert kind({"category": "pois", "shop": "doityourself"}) == "hardware store"
    assert kind({"category": "buildings"}) == "building"
    assert kind({"category": "pois", "amenity": "clinic"}) == "clinic"
    assert kind({"category": "landuse", "landuse": "greenfield"}) == "building land"
    assert kind({"category": "aeroway", "aeroway": "apron"}) == "airport apron"
    assert kind({"category": "buildings", "building": "townhall"}) == "town hall"
    assert kind({"category": "buildings", "building": "roof"}) == "shelter roof"
    assert kind({"category": "pois", "shop": "tobacco"}) == "tobacco shop"
    assert kind({"category": "landuse", "landuse": "religious"}) == "church grounds"
    assert kind({"category": "roads", "highway": "crossing"}) == "crosswalk"
    assert kind({"category": "barrier", "barrier": "fence"}) == "fence or wall"


def test_every_prior_feature_carries_its_layer(monkeypatch) -> None:
    from baselines.human import audit_console
    from baselines.seam import block_landmarks

    square = [[0, 0], [0, 10], [10, 10], [10, 0]]
    features = {"roads": [{"highway": "residential", "name": "Lime Avenue", "world_contour": [[0, 0], [50, 0]]}],
                "buildings": [{"building": "yes", "world_contour": square}],
                "landuse": [{"landuse": "greenfield", "world_contour": square}],
                "pois": [{"amenity": "cafe", "name": "Cafe", "world_contour": square}]}
    monkeypatch.setattr(audit_console, "block_prior", lambda m: (features, block_landmarks(features)))
    doc = study_server.prior_doc("calcity", lambda pts: [list(p) for p in pts])
    assert sorted(f["properties"]["layer"] for f in doc["features"]) == ["building", "landuse", "poi", "road"]


def test_the_command_line_takes_no_participant() -> None:
    with pytest.raises(SystemExit) as e:
        study_server.main(["p01"])
    assert e.value.code == 2


def test_a_second_start_points_at_the_first(tmp_path, capsys) -> None:
    class Study(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<title>The search study</title>")

        def log_message(self, *_args) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Study)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert study_server.main(["--kit", str(tmp_path), "--port", str(server.server_port), "--no-browser"]) == 0
    finally:
        server.shutdown()
    out = capsys.readouterr().out
    assert "already running" in out and "Keep this window open" not in out


def test_another_program_on_the_port(tmp_path, capsys) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        assert study_server.main(["--kit", str(tmp_path), "--port", str(s.getsockname()[1]), "--no-browser"]) == 1
    out = capsys.readouterr().out
    assert "is taken by another program. Restart the computer and start the study again." in out
    assert "already running" not in out
