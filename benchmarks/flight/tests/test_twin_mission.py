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

    from benchmarks.calcity import paths
    from benchmarks.calcity._citynav_bridge import _replacements, compat_env
    from benchmarks.calcity.data_prep import bake_tiles
    from benchmarks.citynav import paths as shared_paths
    from maps.perception.ortho.ground_level import GROUND_LEVEL

    x0, y0 = TARGET[0] - 20, TARGET[1] - 20
    church = Polygon([(x0, y0), (x0 + 40, y0), (x0 + 40, y0 + 40), (x0, y0 + 40)])
    monkeypatch.setattr(bake_tiles, "load_features_local", lambda bundle, lf: [
        {"layer": "test", "osm_id": 1, "tags": {"amenity": "place_of_worship", "name": "First Church",
                                                "building": "church"}, "geom": church}])
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

    for name, value in (("CALCITY_ROOT", root), ("DATA_DIR", root / "data"), ("SPLITS_DIR", splits),
                        ("OBJECTS_JSON", root / "data" / "cityrefer" / "objects.json"), ("TILES_DIR", tiles),
                        ("CALCITY_NATIVE_RGBD_DIR", rgbd)):
        monkeypatch.setattr(paths, name, value)
    for name, value in _replacements().items():
        monkeypatch.setattr(shared_paths, name, getattr(shared_paths, name))
    for name in compat_env():
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("SEARCH_BENCHMARK_NAME", "calcity")
    monkeypatch.setitem(GROUND_LEVEL, "calcity", 0.0)
    return SimpleNamespace(root=root, tiles=tiles, rgbd=rgbd)


@pytest.fixture
def serves(monkeypatch):
    """The reasoner scripted (fly to the church, then reply), the segmenter
    seeing nothing; every request body the reasoner was sent."""
    from maps.llm import endpoint_spec, http
    from maps.llm.endpoint_spec import EndpointSpec

    spec = EndpointSpec(base_url="http://unused", model_id="inkling-small", max_model_len=262144, tokenizer_name="qwen",
                        output_reserve=40000)
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


def test_a_twin_mission_flies_its_leg_and_lands_the_records_a_flown_sortie_carries(calcity, serves, tmp_path):
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
    h = M.hydrate(episode_id="calcity::0#0", split="calcity", config="configs/calcity.yaml")
    h.search_loop.declare_mission_concepts([{"class": "car", "role": "target"}])
    h.search_loop.state.perception_director.watchlist_apply(["tree"], [])
    assert h.perception.concept_provider.concepts == ["car", "tree"]
