"""One mission flown from the console's own requests: picked, sent, watched, stopped and reviewed."""

from __future__ import annotations

import math
import time

from fastapi.testclient import TestClient

from benchmarks.flight.tests.test_twin_mission import calcity, serves  # noqa: F401  (fixtures)
from station import serve


def _until(what, timeout_s: float = 120.0):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if (got := what()) is not None:
            return got
        time.sleep(0.1)
    raise AssertionError("the station never got there")


def test_a_mission_is_picked_flown_and_reviewed(calcity, serves, tmp_path):  # noqa: F811
    station = serve.Station(
        bench=calcity.bench,
        config="configs/calcity.yaml",
        out=tmp_path / "out",
        ortho_dir=calcity.rgbd,
        register=None,
    )
    client = TestClient(serve.build_app(station, web=None))

    (row,) = client.get("/missions", params={"q": "church"}).json()["missions"]
    assert (
        len(row["target_lonlat"]) == 4
        and client.get("/missions", params={"q": "mosque"}).json()["missions"] == []
    )
    assert client.post("/mission/tasking", json={"text": "hello"}).status_code == 409
    assert (
        client.get("/reasoner").json()["model"] == "inkling-small"
        and "max" in client.get("/reasoner").json()["efforts"]
    )

    state = client.post(
        "/mission/next", json={"episode_id": row["episode_id"], "instruction": row["instruction"]}
    ).json()
    assert state["kind"] == "twin" and state["path"]
    assert client.post("/mission/next", json={"instruction": "another"}).status_code == 409
    parked = _until(
        lambda: s if (s := station.seat_state())["status"] == "awaiting_tasking" else None
    )
    assert parked["reply_text"] == "The church is under the last flight."
    waypoint = next(r for r in parked["transcript"] if r["kind"] == "waypoint")
    assert waypoint["n"] == 1 and math.isfinite(waypoint["lon"]) and waypoint["coverageFloor"] > 0
    leg = next(r for r in parked["transcript"] if r["kind"] == "leg" and r.get("area"))
    assert leg["coverage"]["fraction"] >= leg["coverageFloor"] and len(
        leg["coverage"]["held"]
    ) == len(leg["area"])
    (flight,) = parked["flights"]
    assert math.isfinite(flight["lon"]) and parked["flights_flown"] == 1
    (shot,) = parked["observations"]
    x0, y0, x1, y1 = shot["world_window"]
    assert x1 > x0 and y1 > y0 and shot["ground_m2"] > 0 and len(shot["shape"]) == 2
    assert (
        client.get(
            "/run/file", params={"path": f"{parked['path']}/observations/{shot['sha']}.jpg"}
        ).content[:2]
        == b"\xff\xd8"
    )
    assert client.get("/run/file", params={"path": "../../etc/passwd"}).status_code == 404
    own = client.get("/mission/ownship").json()
    assert own["live"] and own["trail"] and math.isfinite(own["lat"])

    assert client.post("/mission/clear", json={}).json() == {}
    assert station.seat_state() is None and client.get("/mission/ownship").json() is None
    (run,) = _until(lambda: client.get("/missions/runs").json()["runs"] or None)
    assert (
        run["status"] == "ended"
        and run["path"] == parked["path"]
        and run["episode_id"] == row["episode_id"]
    )
    replay = client.get("/replay", params={"path": run["path"]}).json()
    assert (
        replay["duration_s"] > 0
        and len(replay["track"]) > 10
        and replay["observations"][0]["sha"] == shot["sha"]
    )
    late = client.get(
        "/mission/ownship", params={"path": run["path"], "t": replay["duration_s"]}
    ).json()
    assert not late["live"] and late["t_s"] > 0
    assert client.get("/replay", params={"path": "nowhere/run"}).status_code == 404

    # The basemap: the manifest names the tiles, and the tile over the map's centre is a picture.
    loc = client.get("/location").json()
    w, s, e, n = loc["bounds"]
    assert w < loc["center"]["lon"] < e and s < loc["center"]["lat"] < n
    z = 15
    tx = int((loc["center"]["lon"] + 180) / 360 * 2**z)
    ty = int((1 - math.asinh(math.tan(math.radians(loc["center"]["lat"]))) / math.pi) / 2 * 2**z)
    assert loc["tiles"] == "tiles/calcity/{z}/{x}/{y}.png"
    assert client.get(f"/tiles/calcity/{z}/{tx}/{ty}.png").content[:4] == b"\x89PNG"
    assert client.get(f"/tiles/calcity/{z}/0/0.png").status_code == 404
    about = client.get("/station").json()
    assert (
        about["bench"] == "calcity"
        and about["flies"]
        and about["own_words"]
        and about["map"] == "calcity"
    )
    # A reviewing station with no raster draws its map from a picture of the bake's extent, and flies nothing.
    from PIL import Image

    from benchmarks.calcity.paths import calcity_bench

    Image.new("RGB", (640, 480), (200, 160, 90)).save(tmp_path / "basemap.jpg")
    bare = calcity_bench(
        {**calcity.bench.environ, "SEARCH_CALCITY_RGBD_DIR": str(tmp_path / "no_rasters")}
    )
    reviewer = serve.Station(
        bench=bare,
        config="configs/calcity.yaml",
        out=tmp_path / "out",
        basemap=tmp_path / "basemap.jpg",
    )
    review = TestClient(serve.build_app(reviewer, web=None))
    b = review.get("/location").json()["bounds"]
    assert (
        b == review.get("/area").json()["bbox"]
        and len(review.get("/missions/runs").json()["runs"]) == 1
    )
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    rx = int((cx + 180) / 360 * 2**12)
    ry = int((1 - math.asinh(math.tan(math.radians(cy))) / math.pi) / 2 * 2**12)
    assert review.get(f"/tiles/calcity/12/{rx}/{ry}.png").content[:4] == b"\x89PNG"
    assert review.post("/mission/next", json={"instruction": "fly"}).status_code == 409
    assert not review.get("/station").json()["flies"]
    groups = {g for g, _ in client.get("/mission/prior.json").json()}
    assert "building" in groups and len(client.get("/area").json()["bbox"]) == 4
