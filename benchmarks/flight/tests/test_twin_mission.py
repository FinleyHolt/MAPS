"""One Cal City mission flown end to end through the twin: a synthetic bake and
raster, a scripted reasoner and a segmenter that sees nothing, and the run
records the flown sorties carry, read back by the bench's own reader."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchmarks.flight.tests import world


TARGET = (1000.0, 1000.0)
START = (1000.0, 300.0)


@pytest.fixture
def calcity(tmp_path, monkeypatch):
    """A Cal City data root, bake, split and raster on a 2 km window, the
    bench's paths pointed at them and put back afterwards."""
    from shapely.geometry import Polygon

    import benchmarks.calcity
    from benchmarks.calcity import paths
    from benchmarks.calcity.data_prep import bake_tiles
    from benchmarks.citynav import paths as shared_paths

    x0, y0 = TARGET[0] - 20, TARGET[1] - 20
    church = Polygon([(x0, y0), (x0 + 40, y0), (x0 + 40, y0 + 40), (x0, y0 + 40)])
    shed = Polygon([(x0 - 430, y0), (x0 - 400, y0), (x0 - 400, y0 + 30), (x0 - 430, y0 + 30)])
    monkeypatch.setattr(bake_tiles, "load_features_local", lambda bundle, lf: [
        {"layer": "test", "osm_id": 1, "tags": {"amenity": "place_of_worship", "name": "First Church",
                                                "building": "church"}, "geom": church},
        {"layer": "test", "osm_id": 2, "tags": {"name": "West Shed", "building": "shed"}, "geom": shed}])
    root = tmp_path / "calcity"
    tiles, splits, rgbd = root / "tiles", root / "splits", root / "data" / "rgbd_upstream"
    bake_tiles.bake_block(tmp_path / "bundle", tiles, "calcity")
    splits.mkdir(parents=True)
    (splits / "calcity.json").write_text(json.dumps([{
        "area": "calcity", "descriptions": ["The church at the end of the road."], "ann_ids": [0],
        "object_ids": [], "target_positions": [[*TARGET, 0.0]], "marker_positions": [],
        "trajectory": [[*START, 125.0]], "source": "authored",
        "target_mask": [list(p) for p in church.exterior.coords[:-1]], "success_radius_m": 20.0}]))
    (root / "data" / "cityrefer").mkdir(parents=True)
    (root / "data" / "cityrefer" / "objects.json").write_text(json.dumps({"calcity": {}}))
    rgbd.mkdir(parents=True)
    world.write_raster_pair(rgbd, world.texture(2000, 3000), 1.0)

    objects = root / "data" / "cityrefer" / "objects.json"
    for name, value in (("CALCITY_ROOT", root), ("DATA_DIR", root / "data"), ("SPLITS_DIR", splits),
                        ("OBJECTS_JSON", objects), ("TILES_DIR", tiles), ("CALCITY_NATIVE_RGBD_DIR", rgbd)):
        monkeypatch.setattr(paths, name, value)
    # What benchmarks.calcity.use() gives a fresh process, on the already-imported CityNav paths.
    for name, value in (("CITYNAV_ROOT", root), ("DATA_DIR", root / "data"), ("SPLITS_DIR", splits),
                        ("OBJECTS_JSON", objects), ("TILES_DIR", tiles), ("CITYNAV_NATIVE_RGBD_DIR", rgbd),
                        ("BENCHMARK", "calcity"), ("SUCCESS_CRITERION", "shape"), ("DEFAULT_SPLIT", "calcity"),
                        ("SPLIT_FILES", {"calcity": "calcity.json"})):
        monkeypatch.setattr(shared_paths, name, value)
    monkeypatch.setattr(benchmarks.calcity, "use", lambda: None)
    monkeypatch.setenv("SEARCH_BENCHMARK_NAME", "calcity")
    return SimpleNamespace(root=root, tiles=tiles, rgbd=rgbd)


@pytest.fixture
def serves(monkeypatch):
    """The reasoner scripted (fly to the church, then reply), the segmenter
    seeing nothing; every request body the reasoner was sent."""
    from maps.llm import endpoint_spec, http
    from maps.llm.endpoint_spec import EndpointSpec

    spec = EndpointSpec(base_url="http://unused", model_id="inkling-small", max_model_len=262144, tokenizer_name="qwen",
                        max_output_tokens=40000)
    monkeypatch.setattr(endpoint_spec, "_SPEC", None)
    monkeypatch.setenv("SEARCH_ENDPOINT_OFFLINE", "1")   # no tokenizer to measure the prompt with
    monkeypatch.setattr(endpoint_spec, "discover_from_env", lambda: spec)
    sent: list[dict] = []

    def reply(url, headers, body, *, timeout, cancel_event):
        sent.append(json.loads(json.dumps(body)))
        if len(sent) == 1:
            call = {"id": "c1", "type": "function",
                    "function": {"name": "run", "arguments": json.dumps({"code": "go_to(prior_0)\n"})}}
            message = {"content": "", "reasoning_content": "fly there", "tool_calls": [call]}
        else:
            message = {"content": "The church is under the last flight.", "reasoning_content": "", "tool_calls": []}
        return {"choices": [{"finish_reason": "stop", "message": message}]}

    monkeypatch.setattr(http, "_stream_chat_completions", reply)

    from benchmarks.citynav.run_benchmark import episode_runner

    from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider

    seen = SimpleNamespace(detect=lambda obs: [], reset=lambda: None,
                           concept_provider=ReasonerUnionConceptProvider(), sahi_tiles_failed=0)
    monkeypatch.setattr(episode_runner, "make_perception", lambda: seen)
    return sent


def test_a_twin_mission_flies_its_leg_and_lands_the_records_a_flown_run_carries(calcity, serves, tmp_path):
    from benchmarks.citynav.analysis import run_report
    from benchmarks.flight import twin

    out = tmp_path / "out"
    code = twin.main(["--config", "configs/calcity.yaml", "--episode", "calcity::0#0", "--ortho", str(calcity.rgbd),
                      "--out", str(out)])
    assert code == 0
    sortie = next(out.iterdir())
    rows = [json.loads(line) for line in (sortie / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["status"] == "ok" and rows[0]["ended_by"] == "reply"
    run = sortie / rows[0]["workdir"]
    assert run.name.startswith("maps_") and "_run01_calcity_" in run.name
    assert rows[0]["flights_flown"] == 1
    assert rows[0]["metrics"]["SR"] == 1          # the last flight is the church
    assert rows[0]["metrics"]["committed_provenance"] == "prior"
    conversation = [json.loads(line) for line in (run / "conversation.jsonl").read_text().splitlines()]
    legs = [r["said"] for r in conversation if r.get("kind") == "leg"]
    assert any("arrived" in s for s in legs), legs
    frames = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    assert len(frames) == 1 and (run / "observations" / f"{frames[0]['observation']}.jpg").is_file()
    assert frames[0]["mosaic"]["placed"] >= 1
    assert (run / "flight" / "llm" / "transcript.jsonl").is_file()
    scored, no_metrics = run_report.load_run(sortie)
    assert list(scored) == ["calcity::0#0"] and scored["calcity::0#0"]["SR"] == 1 and no_metrics == 0
    first = serves[0]
    assert first["messages"][-1] == {"role": "user", "content": "The church at the end of the road."}


def test_a_twin_sortie_scores_fully_on_the_closed_loop_measures(calcity, serves, tmp_path):
    """The mission records the ownship track and its times on the vehicle's
    one clock, so the measures read OSR off the track, SPL and time to ID."""
    from benchmarks.flight import measures, twin

    out = tmp_path / "out"
    assert twin.main(["--config", "configs/calcity.yaml", "--episode", "calcity::0#0", "--ortho", str(calcity.rgbd),
                      "--out", str(out)]) == 0
    sortie = next(out.iterdir())
    rows, _radius, _turns = measures.collect([sortie], measures.Truth([calcity.root / "splits" / "calcity.json"]))
    (run,) = rows
    assert run["success"] is True and run["track_source"] == "GNSS track", run
    assert run["track_coverage"] == 1.0 and run["oracle_success"] is True and run["agl_ft"] is not None
    assert 0.0 < run["spl"] <= 1.0 and run["path_ft"] > 0.0
    # Leg 700 m north at the C-12's speed: the time to ID is the twin's flying time, not the host's.
    assert 0.0 < run["time_to_id_s"] < 120.0
    # Each leg handed to the aircraft is a waypoint row at the commanded height: the
    # turn radius and the height fallback read them.
    (window,) = measures.routed_windows(sortie)
    assert window[0] < window[1]
    conversation = [json.loads(line) for line in next(sortie.glob("*/conversation.jsonl")).read_text().splitlines()]
    assert [r["altitudeAglMeters"] for r in conversation if r.get("kind") == "waypoint"] == [pytest.approx(world.AGL_M)]


def test_the_flights_watchlist_reaches_the_segmenter_as_the_benchs_does(calcity, serves, tmp_path, monkeypatch):
    """The flight wires the agent's watchlist channel exactly as the bench
    does: a watchlist change the director makes reaches the segmenter's
    concept provider, not engine state alone."""
    from benchmarks.citynav.run_benchmark import episode_runner
    from benchmarks.flight import mission as M
    from maps.llm import endpoint_spec
    from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider

    monkeypatch.setattr(episode_runner, "make_perception", lambda: SimpleNamespace(
        detect=lambda obs: [], concept_provider=ReasonerUnionConceptProvider()))
    endpoint_spec.install_spec(endpoint_spec.discover_from_env())
    M.activate_calcity()
    h = M.hydrate(episode_id="calcity::0#0", split="calcity", config="configs/calcity.yaml", agl_m=914.4)
    h.search_loop.declare_mission_concepts([{"class": "car", "role": "target"}])
    h.search_loop.state.perception_director.watchlist_apply(["tree"], [])
    assert h.perception.concept_provider.concepts == ["car", "tree"]


def test_a_mission_hydrated_at_a_commanded_height_flies_at_that_height(calcity, serves, monkeypatch):
    from benchmarks.flight import mission as M
    from maps.llm import endpoint_spec

    endpoint_spec.install_spec(endpoint_spec.discover_from_env())
    M.activate_calcity()
    h = M.hydrate(episode_id="calcity::0#0", split="calcity", config="configs/calcity.yaml", agl_m=1828.8)
    assert h.state.altitude_m - h.search_loop._ortho_ground_z_m == pytest.approx(1828.8)


def _fly_failing(tmp_path, monkeypatch, fault):
    """A twin mission whose reasoner raises ``fault`` on its second request,
    after the first flight: the sortie's one ledger row."""
    from benchmarks.flight import twin
    from maps.llm import http

    scripted = http._stream_chat_completions
    calls = []

    def reply(url, headers, body, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise fault
        return scripted(url, headers, body, **kw)

    monkeypatch.setattr(http, "_stream_chat_completions", reply)
    out = tmp_path / "out"
    assert twin.main(["--config", "configs/calcity.yaml", "--episode", "calcity::0#0", "--ortho",
                      str(tmp_path / "calcity" / "data" / "rgbd_upstream"), "--out", str(out)]) == 1
    (row,) = [json.loads(line) for line in (next(out.iterdir()) / "episodes.jsonl").read_text().splitlines()]
    return row


def test_an_engine_fault_mid_flight_is_a_scored_miss(calcity, serves, tmp_path, monkeypatch):
    """The bench's infra rule: a fault that is not the serve's is the method's own, scored."""
    row = _fly_failing(tmp_path, monkeypatch, KeyError("boom"))
    assert row["status"] == "ok" and row["error"] == "KeyError: 'boom'" and row["flights_flown"] == 1


def test_a_segmenter_outage_that_ends_the_mission_is_infra(calcity, serves, tmp_path, monkeypatch):
    from maps.perception.sam.detect_client import SamServerOutage

    row = _fly_failing(tmp_path, monkeypatch, SamServerOutage("no tile answered"))
    assert row["status"] == "infra" and row["error"] == "SamServerOutage: no tile answered"


def test_a_config_that_withholds_a_part_leaves_no_sortie(calcity, serves, tmp_path):
    from benchmarks.flight import twin

    repo = __import__("pathlib").Path(__file__).resolve().parents[3]
    arm = tmp_path / "arm.yaml"
    arm.write_text(f"base: {repo / 'configs' / 'calcity.yaml'}\nwithhold: [images]\n")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="bench ablation"):
        twin.main(["--config", str(arm), "--episode", "calcity::0#0", "--ortho", str(calcity.rgbd),
                   "--out", str(out)])
    assert not out.exists()


def test_a_retasked_mission_is_measured_to_its_last_report(calcity, serves, tmp_path, monkeypatch):
    """A tasking after the report flies a second leg and reports again: the
    measures stop the run at that last report, where its commit and
    ``elapsed_s`` stand."""
    from benchmarks.flight import measures, twin
    from maps.llm import http

    codes = iter(["go_to(prior_0)\n", None, "go_to(prior_1)\n", None])

    def reply(url, headers, body, *, timeout, cancel_event):
        code = next(codes, None)
        if code is None:
            message = {"content": "Reported.", "reasoning_content": "", "tool_calls": []}
        else:
            call = {"id": "c1", "type": "function", "function": {"name": "run", "arguments": json.dumps({"code": code})}}
            message = {"content": "", "reasoning_content": "fly", "tool_calls": [call]}
        return {"choices": [{"finish_reason": "stop", "message": message}]}

    monkeypatch.setattr(http, "_stream_chat_completions", reply)
    out = tmp_path / "out"
    assert twin.main(["--config", "configs/calcity.yaml", "--episode", "calcity::0#0", "--ortho", str(calcity.rgbd),
                      "--then", "Check the west side too.", "--out", str(out)]) == 0
    sortie = next(out.iterdir())
    (row,) = [json.loads(line) for line in (sortie / "episodes.jsonl").read_text().splitlines()]
    (run,), _radius, _turns = measures.collect([sortie], measures.Truth([calcity.root / "splits" / "calcity.json"]))
    assert row["flights_flown"] == 2
    assert run["time_to_id_s"] == round(row["elapsed_s"], 1)


def test_a_reasoner_unreachable_at_discovery_is_infra(calcity, serves, tmp_path, monkeypatch):
    """Discovery's GET of the models list finds nothing listening: the serve's fault, excluded."""
    import socket

    from benchmarks.flight import twin
    from maps.llm import endpoint_spec

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        closed = s.getsockname()[1]
    monkeypatch.setattr(endpoint_spec, "_PROBE_TRIES", 1)
    monkeypatch.setattr(endpoint_spec, "discover_from_env",
                        lambda: endpoint_spec._get_json(f"http://127.0.0.1:{closed}/v1/models"))
    out = tmp_path / "out"
    assert twin.main(["--config", "configs/calcity.yaml", "--episode", "calcity::0#0", "--ortho", str(calcity.rgbd),
                      "--out", str(out)]) == 1
    (row,) = [json.loads(line) for line in (next(out.iterdir()) / "episodes.jsonl").read_text().splitlines()]
    assert row["status"] == "infra" and row["error"].startswith("EndpointUnreachable: endpoint unreachable")


def test_a_serve_fault_after_a_report_scores_the_run_at_that_report(calcity, serves, tmp_path, monkeypatch):
    """The segmenter fails out on the retask after it flew a second leg: the
    report stands (its end, time and commit), the error is recorded, and the
    run is scored, never infra."""
    from benchmarks.flight import measures, twin
    from maps.llm import http
    from maps.perception.sam.detect_client import SamServerOutage

    codes = iter(["go_to(prior_0)\n", None, "go_to(prior_1)\n", "fail"])

    def reply(url, headers, body, *, timeout, cancel_event):
        code = next(codes, None)
        if code == "fail":
            raise SamServerOutage("no tile answered")
        if code is None:
            message = {"content": "Reported.", "reasoning_content": "", "tool_calls": []}
        else:
            call = {"id": "c1", "type": "function", "function": {"name": "run", "arguments": json.dumps({"code": code})}}
            message = {"content": "", "reasoning_content": "fly", "tool_calls": [call]}
        return {"choices": [{"finish_reason": "stop", "message": message}]}

    monkeypatch.setattr(http, "_stream_chat_completions", reply)
    out = tmp_path / "out"
    assert twin.main(["--config", "configs/calcity.yaml", "--episode", "calcity::0#0", "--ortho", str(calcity.rgbd),
                      "--then", "Check the west side too.", "--out", str(out)]) == 1
    sortie = next(out.iterdir())
    (row,) = [json.loads(line) for line in (sortie / "episodes.jsonl").read_text().splitlines()]
    assert row["status"] == "ok" and row["ended_by"] == "reply" and row["error"] == "SamServerOutage: no tile answered"
    assert row["flights_flown"] == 2 and len(row["visited_xy"]) == 2
    (run,), _radius, _turns = measures.collect([sortie], measures.Truth([calcity.root / "splits" / "calcity.json"]))
    assert run["commit_xy"] == pytest.approx(row["visited_xy"][-1], abs=0.01) and run["end"] == "reply" and run["success"] is True
    assert run["time_to_id_s"] == round(row["elapsed_s"], 1)


@pytest.mark.parametrize("end", ["stopped", "turn_cap"])
def test_a_follow_up_that_ends_without_a_new_report_scores_the_run_at_that_report(
        calcity, serves, tmp_path, monkeypatch, end):
    """The retask flies a second leg and ends with no new report, by the
    crew's Stop or at the turn cap: the report stands (its end and commit)
    and both flights are counted."""
    from benchmarks.citynav.run_benchmark import mission_engine
    from benchmarks.flight import measures, twin
    from benchmarks.flight import mission as M
    from maps.llm import http

    missions = []
    start = M.Mission.start
    monkeypatch.setattr(M.Mission, "start", lambda self: (missions.append(self), start(self))[1])

    class Capped(mission_engine.MissionSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mission_spec.turn_cap = 2

    monkeypatch.setattr(mission_engine, "MissionSession", Capped)
    codes = iter(["go_to(prior_0)\n", None, "go_to(prior_1)\n"])

    def reply(url, headers, body, *, timeout, cancel_event):
        code = next(codes, "x = 1\n")
        if code == "x = 1\n" and end == "stopped":
            missions[0].stop()
            raise http.CallCanceled("stopped")
        if code is None:
            message = {"content": "Reported.", "reasoning_content": "", "tool_calls": []}
        else:
            call = {"id": "c1", "type": "function", "function": {"name": "run", "arguments": json.dumps({"code": code})}}
            message = {"content": "", "reasoning_content": "fly", "tool_calls": [call]}
        return {"choices": [{"finish_reason": "stop", "message": message}]}

    monkeypatch.setattr(http, "_stream_chat_completions", reply)
    out = tmp_path / "out"
    assert twin.main(["--config", "configs/calcity.yaml", "--episode", "calcity::0#0", "--ortho", str(calcity.rgbd),
                      "--then", "Check the west side too.", "--out", str(out)]) == 0
    sortie = next(out.iterdir())
    (row,) = [json.loads(line) for line in (sortie / "episodes.jsonl").read_text().splitlines()]
    run_dir = sortie / row["workdir"]
    conversation = [json.loads(line) for line in (run_dir / "conversation.jsonl").read_text().splitlines()]
    ends = [r["ended_by"] for r in conversation if r.get("kind") == "mission_end"]
    assert ends == ["reply"] + (["turn_cap"] if end == "turn_cap" else [])
    assert row["status"] == "ok" and row["ended_by"] == "reply" and row["error"] is None
    assert row["flights_flown"] == 2 and len(row["visited_xy"]) == 2
    second_leg = [r for r in conversation if r.get("kind") == "leg"][-1]
    assert row["elapsed_s"] < second_leg["t_s"]
    (run,), _radius, _turns = measures.collect([sortie], measures.Truth([calcity.root / "splits" / "calcity.json"]))
    assert run["end"] == "reply" and run["success"] is True


def test_an_authored_start_without_the_operators_words_is_refused(calcity, serves, tmp_path):
    from benchmarks.flight import twin

    out = tmp_path / "out"
    with pytest.raises(SystemExit) as refused:
        twin.main(["--config", "configs/calcity.yaml", "--episode", "calcity::authored", "--start-xy", "1000,300",
                   "--ortho", str(calcity.rgbd), "--out", str(out)])
    assert refused.value.code == 2 and not out.exists() and not serves
