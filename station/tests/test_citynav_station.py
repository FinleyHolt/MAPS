"""The station on CityNav: a bench out-root reviewed, and an episode flown from the console's own requests.

The block, its bake, its split and its raster are synthetic; the reasoner is
the twin tests' script (fly to the first prior, then report) and the segmenter
sees nothing. The georeference is the shipped Birmingham OSM cache's own."""

from __future__ import annotations

import json
import math
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from benchmarks.flight.tests.test_twin_mission import serves  # noqa: F401  (fixture)
from station import serve
from station.placement import OSM_CACHE_DIR

BLOCK = "birmingham_block_0"
SIDE_M, PX_M = 200.0, 0.1
CHURCH = [[90.0, 90.0], [110.0, 90.0], [110.0, 110.0], [90.0, 110.0]]
EPISODE = f"{BLOCK}::0#0"


@pytest.fixture
def citynav(tmp_path, monkeypatch):
    """A CityNav data root holding one block: its split row, its catalog
    object, its bake and its 0.1 m raster pair; the bench the environment names."""
    import rasterio
    from PIL import Image
    from rasterio.transform import from_origin

    from benchmarks.citynav.bake.city import bake_block
    from benchmarks.citynav.eval import WorldBounds
    from benchmarks.citynav.paths import citynav_bench
    from benchmarks.flight.tests import world
    from maps.perception.ortho.ground_level import GROUND_LEVEL

    root = tmp_path / "citynav"
    data, tiles = root / "data", root / "tiles"
    for d in ("citynav", "cityrefer", "rgbd_upstream"):
        (data / d).mkdir(parents=True)
    square = {"type": "Polygon", "coordinates": [[*CHURCH, CHURCH[0]]]}
    bake_block(
        BLOCK,
        [({"name": "Elm Church", "building": "church", "osm_id": "w1"}, square)],
        tiles,
        bounds=WorldBounds(x_min=0.0, y_min=0.0, x_max=SIDE_M, y_max=SIDE_M),
    )
    (data / "citynav" / "citynav_val_seen.json").write_text(
        json.dumps(
            [
                {
                    "area": "birmingham",
                    "block": 0,
                    "descriptions": ["The church by the green."],
                    "ann_ids": [0],
                    "object_ids": [7],
                    "target_positions": [[100.0, 100.0, 0.0]],
                    "marker_positions": [],
                    "trajectory": [[100.0, 60.0, GROUND_LEVEL[BLOCK] + 50.0, 0.0, 0.0]],
                }
            ]
        )
    )
    (data / "cityrefer" / "objects.json").write_text(
        json.dumps(
            {
                BLOCK: {
                    "7": {
                        "id": 7,
                        "position": [100.0, 100.0, 0.0],
                        "dimension": [20.0, 20.0, 5.0],
                        "contour": CHURCH,
                    }
                }
            }
        )
    )
    n = int(SIDE_M / PX_M)
    Image.fromarray(world.texture(n, 3000)).save(data / "rgbd_upstream" / f"{BLOCK}.png")
    with rasterio.open(
        data / "rgbd_upstream" / f"{BLOCK}.tif",
        "w",
        driver="GTiff",
        width=n,
        height=n,
        count=1,
        dtype="float32",
        transform=from_origin(0.0, SIDE_M, PX_M, PX_M),
    ) as dst:
        dst.write(np.zeros((1, n, n), np.float32))
    environ = {
        "SEARCH_CITYNAV_DATA": str(root),
        "SEARCH_CITYNAV_TILES": str(tiles),
        "SEARCH_ENDPOINT_OFFLINE": "1",
    }
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    for name in (
        "SEARCH_CITYNAV_SPLITS",
        "SEARCH_CITYNAV_RGBD_DIR",
        "SEARCH_BENCHMARK_NAME",
        "CITYNAV_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    return citynav_bench(environ)


def _until(what, timeout_s: float = 120.0):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if (got := what()) is not None:
            return got
        time.sleep(0.1)
    raise AssertionError("the station never got there")


def _station(bench, out, **kw):
    return serve.Station(bench=bench, config="configs/citynav.yaml", out=out, **kw)


def _cache_lonlat(x: float, y: float) -> tuple[float, float]:
    """Local ``(x, y)`` in WGS84 off the shipped Birmingham cache's own numbers, with no code of the station's."""
    from pyproj import Transformer

    georef = json.loads((OSM_CACHE_DIR / "birmingham_osm_local.geojson").read_text())[
        "citynav_osm_cache"
    ]["georef"]
    a, b, c, d, e, f = georef["citynav_from_osm_affine"]
    east, north = np.linalg.solve([[a, b], [d, e]], [x - c, y - f])
    return Transformer.from_crs(georef["osm_crs"], "EPSG:4326", always_xy=True).transform(
        east, north
    )


def _tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    return (
        int((lon + 180) / 360 * 2**z),
        int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2**z),
    )


@pytest.fixture
def bench_root(citynav, serves, tmp_path, monkeypatch):  # noqa: F811
    """A bench out-root as the driver leaves one: the runner's own row for one
    flown episode, its attempt folder under a worker's ``w3/``."""
    from benchmarks import sandbox
    from benchmarks.citynav.run_benchmark import cli
    from maps.llm import http

    class Client:
        """The bench's requests do not stream: its one POST is answered by the same script."""

        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, headers=None, json=None):
            answer = http.stream_chat_completions(
                url, headers, json, timeout=None, cancel_event=None
            )
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: answer)

    root = tmp_path / "roots" / "val_seen_a"
    with monkeypatch.context() as m:
        m.setattr(http.httpx, "Client", Client)
        m.setattr(
            sys,
            "argv",
            [
                "run_benchmark",
                "--split",
                "val_seen",
                "--config",
                "configs/citynav.yaml",
                "--episode-id",
                EPISODE,
                "--out",
                str(root),
            ],
        )
        m.setattr(sandbox, "confine", lambda *a, **kw: None)
        assert cli.main(citynav) == 0
    (row,) = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]
    (root / "w3").mkdir()
    (root / row["workdir"]).rename(root / "w3" / row["workdir"])
    row = {**row, "workdir": f"w3/{row['workdir']}", "worker_id": 3}
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    return root, row


def test_a_bench_root_reviews_with_its_path_target_commit_score_pictures_and_report(
    citynav, bench_root
):
    root, row = bench_root
    assert row["status"] == "ok" and row["metrics"]["SR"] == 1, row
    station = _station(citynav, root.parent)
    client = TestClient(serve.build_app(station, web=None))

    about = client.get("/station").json()
    # The list and the selector open on one split.
    assert client.get("/missions").json()["split"] == about["split"]
    assert (
        about["bench"] == "citynav"
        and about["split"] == "val_seen"
        and about["splits"] == ["val_seen"]
    )
    assert about["map"] == BLOCK and about["flies"] and not about["own_words"]

    # The catalog names the row's map and its target's footprint, placed through the OSM cache's transform.
    (entry,) = client.get("/missions", params={"split": "val_seen"}).json()["missions"]
    assert entry["episode_id"] == EPISODE and entry["map"] == BLOCK
    assert entry["target_lonlat"][0] == pytest.approx(_cache_lonlat(*CHURCH[0]), abs=1e-7)
    assert client.get("/missions", params={"split": "nowhere"}).status_code == 404

    (run,) = client.get("/missions/runs").json()["runs"]
    assert (
        run["path"] == f"val_seen_a/{row['workdir']}"
        and run["episode_id"] == EPISODE
        and run["status"] == "ended"
    )
    assert (
        run["score"]["SR"] == 1
        and run["score"]["NE"] == row["metrics"]["NE"]
        and run["map_name"] == BLOCK
    )
    assert run["ended_by"] == "reply" and run["turns_used"] == 2 and run["transcript"] == []

    replay = client.get("/replay", params={"path": run["path"]}).json()
    assert (
        replay["reply_text"] == "The church is under the last flight."
        and replay["score"]["SR"] == 1
    )
    assert replay["flights_flown"] == 1
    # No flight_track.jsonl: the path is the row's visited_xy, and there is nothing to scrub.
    assert replay["duration_s"] is None and len(replay["track"]) == len(row["visited_xy"])
    first, last = replay["track"][0], replay["track"][-1]
    assert (first["lon"], first["lat"]) == pytest.approx(_cache_lonlat(*row["start_xy"]), abs=1e-7)
    assert [last["x"], last["y"]] == row["visited_xy"][-1]
    (stop,) = replay["flights"]
    assert stop["target"] == "prior_0" and (stop["lon"], stop["lat"]) == pytest.approx(
        _cache_lonlat(*stop["resolved_xy"]), abs=1e-7
    )
    assert replay["commit_lonlat"] == pytest.approx(_cache_lonlat(*row["commit_xy"]), abs=1e-7)
    assert replay["target_point"] == pytest.approx(_cache_lonlat(*row["target_xy"]), abs=1e-7)
    assert len(replay["target_lonlat"]) == 4
    # The pictures the model was sent ride the transcript; a bench run has no picture book.
    shas = [sha for r in replay["transcript"] for sha in r.get("frames") or []]
    assert shas and replay["observations"] == []
    assert (
        client.get("/run/file", params={"path": f"{run['path']}/frames/{shas[0]}.jpg"}).content[:2]
        == b"\xff\xd8"
    )
    assert client.get("/mission/ownship", params={"path": run["path"], "t": 0}).json() is None

    # One root named outright reviews the same run.
    alone = TestClient(serve.build_app(_station(citynav, root), web=None))
    (same,) = alone.get("/missions/runs").json()["runs"]
    assert same["path"] == f"./{row['workdir']}"
    assert (
        alone.get("/replay", params={"path": same["path"]}).json()["reply_text"]
        == replay["reply_text"]
    )

    # Pictures stripped: the run still reviews as text.
    for jpg in (root / row["workdir"] / "frames").iterdir():
        jpg.unlink()
    stripped = client.get("/replay", params={"path": run["path"]}).json()
    assert stripped["reply_text"] == replay["reply_text"] and len(stripped["transcript"]) == len(
        replay["transcript"]
    )


def test_the_map_of_a_citynav_block_is_placed_by_the_osm_caches_transform(
    citynav, bench_root, tmp_path
):
    root, _row = bench_root
    client = TestClient(serve.build_app(_station(citynav, root.parent), web=None))

    loc = client.get("/location", params={"map": BLOCK}).json()
    w, s = _cache_lonlat(0.0, 0.0)
    e, n = _cache_lonlat(SIDE_M, SIDE_M)
    assert (
        loc["bounds"] == pytest.approx([w, s, e, n], abs=1e-7)
        and -2.1 < w < e < -1.7
        and 52.3 < s < n < 52.7
    )
    assert loc["name"] == BLOCK and BLOCK in loc["tiles"]
    assert client.get("/area", params={"map": BLOCK}).json()["bbox"] == pytest.approx(loc["bounds"])
    x, y = _tile(loc["center"]["lon"], loc["center"]["lat"], 17)
    assert client.get(f"/tiles/{BLOCK}/17/{x}/{y}.png").content[:4] == b"\x89PNG"
    assert client.get(f"/tiles/{BLOCK}/17/0/0.png").status_code == 404
    assert {g for g, _ in client.get("/mission/prior.json", params={"map": BLOCK}).json()} == {
        "building"
    }

    # A map the bench holds no bake of, and a name that is no map's, answer 404 everywhere.
    for name in ("cambridge_block_2", "../citynav", "a/b"):
        for url in ("/location", "/area", "/mission/prior.json"):
            assert client.get(url, params={"map": name}).status_code == 404, (url, name)
    assert client.get("/tiles/..%2Fx/17/0/0.png").status_code == 404

    # A map with no georef anywhere still reviews, with its transcript and no placement.
    bare = TestClient(
        serve.build_app(
            _station(citynav, root.parent, osm_cache_dir=tmp_path / "no_cache"), web=None
        )
    )
    (run,) = bare.get("/missions/runs").json()["runs"]
    replay = bare.get("/replay", params={"path": run["path"]}).json()
    assert (
        replay["transcript"] and "lon" not in replay["track"][0] and replay["target_lonlat"] is None
    )
    assert (
        replay["commit_lonlat"] is None
        and bare.get("/location", params={"map": BLOCK}).status_code == 404
    )


def test_a_citynav_episode_is_flown_from_the_console_and_lands_a_scored_row(
    citynav,
    serves,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    from maps.llm import endpoint_spec

    out = tmp_path / "out"
    station = _station(citynav, out)
    client = TestClient(serve.build_app(station, web=None))

    assert client.post("/mission/next", json={"instruction": "my own words"}).status_code == 400
    assert (
        client.post(
            "/mission/next", json={"episode_id": "nowhere::0#9", "instruction": ""}
        ).status_code
        == 404
    )
    state = client.post("/mission/next", json={"episode_id": EPISODE, "instruction": ""}).json()
    assert (
        state["kind"] == "bench" and state["episode_id"] == EPISODE and state["map_name"] == BLOCK
    )
    assert (
        client.post("/mission/next", json={"episode_id": EPISODE, "instruction": ""}).status_code
        == 409
    )
    parked = _until(
        lambda: (
            s if (s := station.seat_state())["status"] in ("awaiting_tasking", "failed") else None
        )
    )
    assert parked["status"] == "awaiting_tasking", parked["error"]
    assert (
        parked["reply_text"] == "The church is under the last flight."
        and parked["flights_flown"] == 1
    )
    assert any(r.get("frames") for r in parked["transcript"]) and math.isfinite(
        parked["flights"][0]["lon"]
    )
    own = client.get("/mission/ownship").json()
    assert own["live"] and [own["x"], own["y"]] == parked["flights"][0]["resolved_xy"]
    assert serves[0]["messages"][-1] == {"role": "user", "content": "The church by the green."}

    def refused(settings, **kw):
        raise RuntimeError("asked a serve")

    # A seat answers the reasoner menu from the serve it flies, asking no discovery.
    with monkeypatch.context() as m:
        m.setattr(endpoint_spec, "discover_from_settings", refused)
        assert client.get("/reasoner").json()["model"] == "inkling-small"

    # A tasking after the report is one more turn of the same conversation.
    client.post("/mission/tasking", json={"text": "Look again."})
    again = _until(
        lambda: (
            s
            if (s := station.seat_state())["status"] == "awaiting_tasking"
            and sum(r["kind"] == "mission_end" for r in s["transcript"]) == 2
            else None
        )
    )
    assert [r["said"] for r in again["transcript"] if r["kind"] == "retask"] == ["Look again."]
    assert serves[-1]["messages"][-1] == {"role": "user", "content": "Look again."}
    assert serves[-1]["messages"][1] == serves[0]["messages"][1]

    # Ending the mission scores it with the bench's scorer and lands its row.
    assert client.post("/mission/clear", json={}).json() == {}
    (run,) = client.get("/missions/runs").json()["runs"]
    assert run["path"] == parked["path"] and run["score"]["SR"] == 1 and run["status"] == "ended"
    sortie = out / run["path"].split("/")[0]
    (row,) = [json.loads(line) for line in (sortie / "episodes.jsonl").read_text().splitlines()]
    assert (
        row["status"] == "ok" and row["metrics"]["success_criterion"] == "ring" and row["commit_xy"]
    )
    assert row["provenance"]["split"] == "val_seen" and (sortie / "run_manifest.json").is_file()
    assert json.loads((sortie / row["workdir"] / "episode.json").read_text())["metrics"]["SR"] == 1
    assert (
        client.get("/replay", params={"path": run["path"]}).json()["reply_text"]
        == parked["reply_text"]
    )


def test_no_request_reads_outside_the_output_root(citynav, bench_root, tmp_path):
    root, row = bench_root
    client = TestClient(serve.build_app(_station(citynav, root.parent), web=None))
    secret = tmp_path / "secret.txt"
    secret.write_text("no")
    inside = f"val_seen_a/{row['workdir']}/episode.json"
    assert client.get("/run/file", params={"path": inside}).status_code == 200
    for path in (
        "../secret.txt",
        "val_seen_a/../../secret.txt",
        str(secret),
        "val_seen_a/../../../etc/passwd",
        f"val_seen_a/{row['workdir']}/../../../../secret.txt",
    ):
        assert client.get("/run/file", params={"path": path}).status_code == 404, path
        assert client.get("/replay", params={"path": path}).status_code == 404, path
        assert client.get("/mission/ownship", params={"path": path}).status_code == 404, path
    # A link inside the root that points out of it is not followed.
    (root / "leak.txt").symlink_to(secret)
    assert client.get("/run/file", params={"path": "val_seen_a/leak.txt"}).status_code == 404


def test_a_row_on_a_block_without_its_raster_pair_is_refused_before_its_sortie(citynav, tmp_path):
    """The seat checks a row's block as the runners do: without ``<map>.png`` or
    ``<map>_rgb.tif`` every flight would serve a block error and book a miss."""
    (citynav.rgbd_dir / f"{BLOCK}.png").unlink()
    out = tmp_path / "out"
    client = TestClient(serve.build_app(_station(citynav, out), web=None))
    refused = client.post("/mission/next", json={"episode_id": EPISODE, "instruction": ""})
    assert refused.status_code == 409
    assert (
        f"  Missing {citynav.rgbd_dir / f'{BLOCK}_rgb.tif'} (built by "
        "benchmarks/citynav/data_prep/rasterize_citynav_native.sh)"
        in refused.json()["detail"].splitlines()
    )
    assert not out.exists()
