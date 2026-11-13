"""One mission flown from the console's own requests: picked, sent, watched, stopped and reviewed."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from benchmarks.flight.tests.test_twin_mission import calcity, serves  # noqa: F401  (fixtures)
from maps.llm import endpoint_spec
from station import serve


def _until(what, timeout_s: float = 120.0):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if (got := what()) is not None:
            return got
        time.sleep(0.1)
    raise AssertionError("the station never got there")


def test_a_seat_read_as_the_mission_parks_pairs_its_status_with_its_report(tmp_path):
    """The mission thread parks the mission while the console reads the seat: the state
    returned reads the status once, so a parked status always carries its report."""
    m = SimpleNamespace(
        kind="twin", status=serve.STATE_FLYING, map_name="m", flights=[], transcript=[],
        workdir=tmp_path / "r" / "w", split="s", episode_id="e", waiting=None, error=None,
        run_no=1, instruction="i", ended_by=None, turns=lambda: 1, started_at=None,
        git_commit=None, git_dirty=None, score=None,
    )  # fmt: skip

    def parks_now(split, eid):
        m.status = serve.STATE_AWAITING
        return None, None

    seat = SimpleNamespace(
        mission=m, map_name="m", _observations=lambda w, t: [], _reported=lambda w, t: "Here.",
        target=parks_now, lonlat=lambda *a: None, _flights=lambda *a: [],
    )  # fmt: skip
    state = serve.Station.seat_state(seat)
    assert state["status"] == serve.STATE_FLYING and state["reply_text"] is None, state
    assert serve.Station.seat_state(seat)["reply_text"] == "Here."


def test_clearing_the_seat_waits_for_a_twin_mission_to_bank_its_row():
    """Ctrl-C clears the seat as the station exits: a twin mission stopped mid-turn banks
    its row before the process ends, as a bench episode does."""
    import threading

    calls: list = []
    twin = SimpleNamespace(stop=lambda: calls.append("stop"), wait=lambda t: calls.append(t))
    seat = SimpleNamespace(_lock=threading.Lock(), mission=twin, _trail=[1])
    serve.Station.clear(seat)
    assert calls == ["stop", serve.END_WAIT_S] and seat.mission is None


def test_a_twin_station_whose_map_lacks_its_bake_flies_nothing(tmp_path):
    seat = SimpleNamespace(
        bench=SimpleNamespace(name="calcity"), ortho_dir=tmp_path, tiles_dir=tmp_path, map_name="m"
    )
    why = serve.Station.why_not_flying(seat)
    assert why is not None and why.startswith("this station reviews runs only: m carries no georef")


def test_a_mission_is_picked_flown_and_reviewed(calcity, serves, tmp_path, monkeypatch):  # noqa: F811
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
    # A station that flies asks discovery for the reasoner alone; SAM answers a mission, not the console.
    asked, discover = [], endpoint_spec.discover_from_settings
    monkeypatch.setattr(
        endpoint_spec,
        "discover_from_settings",
        lambda settings, **kw: asked.append(kw) or discover(settings, **kw),
    )
    assert (
        client.get("/reasoner").json()["model"] == "inkling-small"
        and "max" in client.get("/reasoner").json()["efforts"]
        and asked == [{"probe_sam": False}] * 2
    )

    # An effort the served card refuses is refused before a run folder opens.
    bogus = client.post(
        "/mission/next",
        json={"episode_id": row["episode_id"], "instruction": "", "reasoning_effort": "bogus"},
    )
    assert bogus.status_code == 400 and "bogus" in bogus.json()["detail"]
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())
    state = client.post(
        "/mission/next", json={"episode_id": row["episode_id"], "instruction": row["instruction"]}
    ).json()
    assert state["kind"] == "twin" and state["path"]
    assert client.post("/mission/next", json={"instruction": "another"}).status_code == 409
    parked = _until(
        lambda: s if (s := station.seat_state())["status"] == "awaiting_tasking" else None
    )
    assert parked["reply_text"] == "The church is under the last flight."

    def refused(settings, **kw):
        raise RuntimeError("asked a serve")

    # A seat answers from the serve it flies, asking no discovery.
    monkeypatch.setattr(endpoint_spec, "discover_from_settings", refused)
    assert client.get("/reasoner").json()["model"] == "inkling-small"
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
    # A reviewing station answers the reasoner menu at once, with no discovery to wait on.
    assert review.get("/reasoner").json() == {
        "model": None,
        "efforts": [],
        "error": reviewer.why_not_flying(),
    }
    groups = {g for g, _ in client.get("/mission/prior.json").json()}
    assert "building" in groups and len(client.get("/area").json()["bbox"]) == 4


@pytest.mark.parametrize("flies", [True, False])
def test_the_command_line_station_confines_itself_before_it_serves(flies, tmp_path, monkeypatch):
    """Missions fly in the station's own threads: the process takes the memory cap and,
    where bubblewrap runs, writes only its runs and, when it flies, the ortho caches, and
    reads the hub cache offline. A station that reviews runs only writes ``--out`` alone."""
    import uvicorn

    from benchmarks import sandbox

    confined: list = []
    monkeypatch.setattr(sandbox, "confine", lambda *a, **kw: confined.append((*a, kw)))
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: confined.append("served"))
    # The station itself is the other tests' subject; this one asks only what it is started under.
    monkeypatch.setattr(
        serve,
        "Station",
        lambda **kw: SimpleNamespace(why_not_flying=lambda: None, clear=lambda: None),
    )
    monkeypatch.setattr(serve, "build_app", lambda station: None)
    monkeypatch.setattr(serve, "why_not_flying", lambda bench, ortho: None if flies else "no")
    monkeypatch.setattr(
        "sys.argv", ["serve", "--out", str(tmp_path / "out"), "--ortho", str(tmp_path / "o")]
    )
    assert serve.main() == 0
    (writable, gib, _log, readable, kw), served = confined
    bench = serve.bench_named("citynav", os.environ)
    assert served == "served" and gib == sandbox.STATION_MEMORY_GIB and kw == {"flies": flies}
    caches = [Path(bench.rgbd_dir) / "ortho", tmp_path / "o" / "ortho"]
    assert writable == [tmp_path / "out", *(caches if flies else [])]
    assert {bench.tiles_dir, bench.splits_dir, tmp_path / "out"} <= set(readable)


def test_a_citynav_station_without_rasters_reviews_runs_only(tmp_path):
    """Annotations alone fly nothing: a mission crops its pictures from the block's
    raster, so a station with no raster under its raster directory reports why."""
    from benchmarks.citynav.paths import citynav_bench

    root = tmp_path / "citynav"
    (root / "data" / "citynav").mkdir(parents=True)
    (root / "data" / "cityrefer").mkdir(parents=True)
    (root / "data" / "cityrefer" / "objects.json").write_text("{}")
    bench = citynav_bench({"SEARCH_CITYNAV_DATA": str(root)})
    why = serve.why_not_flying(bench, None)
    assert why is not None and str(bench.rgbd_dir) in why, why
    (root / "data" / "rgbd_upstream").mkdir()
    (root / "data" / "rgbd_upstream" / "birmingham_block_1.tif").write_bytes(b"")
    assert serve.why_not_flying(bench, None) is None


def test_a_station_opens_on_a_split_its_list_carries(tmp_path):
    """A splits directory without the bench's default split opens the console on a
    split it holds, so the shown split is one the list offers."""
    from types import SimpleNamespace

    from benchmarks.citynav.paths import citynav_bench

    bench = citynav_bench({"SEARCH_CITYNAV_SPLITS": str(tmp_path)})
    (tmp_path / bench.split_files["test_unseen"]).write_text("[]")
    seat = SimpleNamespace(bench=bench, split=bench.default_split, runs=lambda: [])
    assert serve.Station.shown_split(seat) == "test_unseen"
    # A row sent with no split flies from the split the list showed it in.
    import threading

    flown: list = []
    seat.__dict__.update(
        why_not_flying=lambda: None,
        tokenizer_refusal=lambda: None,
        _lock=threading.Lock(),
        mission=None,
        shown_split=lambda: serve.Station.shown_split(seat),
        _bench_mission=lambda *a: flown.append(a[-2]) or SimpleNamespace(start=lambda: None),
        seat_state=lambda: None,
    )
    serve.Station.start(seat, "row", "", None)
    assert flown == ["test_unseen"]


def test_a_cal_city_reviewing_station_draws_its_basemap_from_the_picture(calcity, tmp_path):  # noqa: F811
    """A Cal City station without --ortho flies nothing, so the bench raster at its default
    path stays unread and no ortho cache is built: the map comes from --basemap."""
    from PIL import Image

    Image.new("RGB", (640, 480), (200, 160, 90)).save(tmp_path / "basemap.jpg")
    reviewer = serve.Station(
        bench=calcity.bench,
        config="configs/calcity.yaml",
        out=tmp_path / "out",
        basemap=tmp_path / "basemap.jpg",
    )
    assert reviewer.basemap("calcity")[0].shape == (480, 640, 3)
    assert not (calcity.rgbd / "ortho").exists()


def test_a_raster_that_fails_to_load_falls_back_to_the_picture(calcity, tmp_path, monkeypatch):  # noqa: F811
    """A raster whose ortho cache cannot be built (a read-only directory under bubblewrap)
    leaves the map to --basemap's picture."""
    from PIL import Image

    from benchmarks.flight import map_render

    def read_only(rgbd_dir, map_name):
        raise OSError(30, "Read-only file system")

    Image.new("RGB", (640, 480), (200, 160, 90)).save(tmp_path / "basemap.jpg")
    station = serve.Station(
        bench=calcity.bench,
        config="configs/calcity.yaml",
        out=tmp_path / "out",
        ortho_dir=calcity.rgbd,
        basemap=tmp_path / "basemap.jpg",
    )
    monkeypatch.setattr(map_render, "load_raster", read_only)
    assert station.basemap("calcity")[0].shape == (480, 640, 3)


def test_a_sandboxed_station_without_its_tokenizer_refuses_a_mission_naming_the_remedy(
    calcity,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    """The station's launcher could not cache the reasoner's tokenizer and the hub is
    read offline: a mission is refused before its run folder opens, naming what caches
    the tokenizer; once discovery loads it, missions fly."""
    from benchmarks import sandbox

    monkeypatch.setenv(sandbox.TOKENIZER_ENV, "ConnectError")
    station = serve.Station(
        bench=calcity.bench,
        config="configs/calcity.yaml",
        out=tmp_path / "out",
        ortho_dir=calcity.rgbd,
    )

    def offline(settings, **kw):
        raise RuntimeError("tokenizer 'x' failed to load: check your internet connection")

    monkeypatch.setattr(endpoint_spec, "discover_from_settings", offline)
    client = TestClient(serve.build_app(station, web=None))
    refused = client.post("/mission/next", json={"instruction": "fly"})
    assert refused.status_code == 409, refused.json()
    assert "python -m maps.mission.endpoint_preflight" in refused.json()["detail"]
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())
    monkeypatch.setattr(endpoint_spec, "discover_from_settings", lambda settings, **kw: None)
    assert station.tokenizer_refusal() is None and sandbox.TOKENIZER_ENV not in os.environ


def test_a_sandboxed_station_serves_the_discovery_s_own_failure_and_the_step_it_needs(
    calcity,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    """A reasoner that does not answer yet is served as itself, not as a missing
    tokenizer. A hub cache the sandbox binds shows the host's writes, so the remedy
    asks for the mission again; a cache made after the station started needs a restart."""
    from benchmarks import sandbox

    monkeypatch.setenv(sandbox.TOKENIZER_ENV, "EndpointUnreachable")
    station = serve.Station(
        bench=calcity.bench,
        config="configs/calcity.yaml",
        out=tmp_path / "out",
        ortho_dir=calcity.rgbd,
    )

    def down(settings, **kw):
        raise endpoint_spec.EndpointUnreachable("reasoner at http://r/v1 did not answer")

    monkeypatch.setattr(endpoint_spec, "discover_from_settings", down)
    monkeypatch.setattr(sandbox, "hf_hub_cache", lambda: str(tmp_path))
    why = station.tokenizer_refusal()
    assert why is not None and "reasoner at http://r/v1 did not answer" in why
    assert "not in the hub cache" not in why
    assert why.endswith(f"{sandbox.TOKENIZER_REMEDY}, then send the mission again")
    monkeypatch.setattr(sandbox, "hf_hub_cache", lambda: str(tmp_path / "absent"))
    assert station.tokenizer_refusal().endswith(
        f"{sandbox.TOKENIZER_REMEDY}, then restart the station"
    )


def test_a_sandboxed_station_whose_reasoner_answers_names_the_uncached_tokenizer(
    calcity,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    """The reasoner answers but its tokenizer is absent from the hub cache: the refusal
    names the tokenizer and the remedy on one line, without the loader's own advice."""
    from benchmarks import sandbox
    from benchmarks.flight.tests.test_twin_mission import uncached_tokenizer

    monkeypatch.setenv(sandbox.TOKENIZER_ENV, "EndpointDiscoveryError")
    station = serve.Station(
        bench=calcity.bench,
        config="configs/calcity.yaml",
        out=tmp_path / "out",
        ortho_dir=calcity.rgbd,
    )
    uncached_tokenizer(monkeypatch)
    monkeypatch.setattr(sandbox, "hf_hub_cache", lambda: str(tmp_path))
    why = station.tokenizer_refusal()
    assert len(why.splitlines()) == 1, why
    assert "the reasoner's tokenizer 'org/tok' is not in the hub cache" in why, why
    assert "internet" not in why and "SEARCH_COMPACT_TOKENIZER_NAME" not in why, why
    assert why.endswith(f"{sandbox.TOKENIZER_REMEDY}, then send the mission again")


def test_a_sandboxed_station_whose_cached_tokenizer_fails_names_the_loaders_cause(
    calcity,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    """A tokenizer the hub cache holds that still fails to load (a broken snapshot, a
    gated repo) is refused with the loader's cause, never as absent from the cache."""
    from benchmarks import sandbox
    from benchmarks.flight.tests.test_twin_mission import uncached_tokenizer

    monkeypatch.setenv(sandbox.TOKENIZER_ENV, "EndpointDiscoveryError")
    station = serve.Station(
        bench=calcity.bench,
        config="configs/calcity.yaml",
        out=tmp_path / "out",
        ortho_dir=calcity.rgbd,
    )
    uncached_tokenizer(monkeypatch)
    (tmp_path / "models--org--tok").mkdir()
    monkeypatch.setattr(sandbox, "hf_hub_cache", lambda: str(tmp_path))
    why = station.tokenizer_refusal()
    assert len(why.splitlines()) == 1, why
    assert "not in the hub cache" not in why, why
    assert "failed to load" in why, why
    assert why.endswith(f"{sandbox.TOKENIZER_REMEDY}, then send the mission again")


def test_a_station_whose_ortho_lacks_the_raster_exits_2_with_no_out(calcity, tmp_path, monkeypatch):  # noqa: F811
    """A bad --ortho is refused before the station confines itself, so --out is not made,
    with the twin's exit code."""
    from benchmarks import sandbox

    monkeypatch.setattr(sandbox, "confine", lambda *a, **kw: pytest.fail("confined"))
    (tmp_path / "empty").mkdir()
    monkeypatch.setattr(
        "sys.argv",
        ["serve", "--bench", "calcity", "--out", str(tmp_path / "out"),
         "--ortho", str(tmp_path / "empty")],
    )  # fmt: skip
    assert serve.main() == 2
    assert not (tmp_path / "out").exists()
