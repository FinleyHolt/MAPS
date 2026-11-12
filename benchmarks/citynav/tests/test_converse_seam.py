"""The seam the station flies a bench episode through: ``converse`` drives the
conversation in place of the bench's one turn, the bench builds, replays and
scores around it, and a process that flies several run contexts stamps each
row with its own."""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.citynav.adapter import parse_episode
from benchmarks.citynav.engine import ENGINE
from benchmarks.citynav.paths import citynav_bench
from benchmarks.citynav.run_benchmark import dispatch
from benchmarks.citynav.run_benchmark import episode_runner as er
from maps.mission.spec import MissionReply
from maps.settings import Settings

_ROW = {
    "area": "b",
    "block": 1,
    "descriptions": ["the church"],
    "ann_ids": [0],
    "target_positions": [[5.0, 5.0, 0.0]],
    "trajectory": [[0.0, 0.0, 100.0]],
}


def _stub_engine(monkeypatch, tmp_path):
    """The episode build, session and replay stubbed as ``test_episode_row`` stubs them; the session flies nothing."""
    h = SimpleNamespace(
        episode=None,
        state=SimpleNamespace(altitude_m=100.0, pose_xy=(0.0, 0.0)),
        ground_z_m=0.0,
        world=SimpleNamespace(set_retro_perceiver=lambda f: None),
        perception=SimpleNamespace(sahi_tiles_failed=0),
    )

    def build(cfg, ep, assets, bench, *, settings, workdir, routing_key):
        h.episode = ep
        return h

    monkeypatch.setattr(er, "load_episode_assets", lambda bench, m: SimpleNamespace())
    monkeypatch.setattr(er, "build_episode", build)
    monkeypatch.setattr(er, "MissionSession", lambda h, **kw: SimpleNamespace(h=h, flights=[]))
    monkeypatch.setattr(
        er, "run_mission_episode", lambda session: 1 / 0
    )  # the bench's one turn is not driven
    monkeypatch.setattr(
        er,
        "synthesize_compliant_trajectory",
        lambda *, flights, **kw: [(0.0, 0.0), tuple(flights[-1]["resolved_xy"])],
    )
    monkeypatch.setattr(er.flight_summary, "summarize", lambda d, tokenizer_name="": {})
    return SimpleNamespace(
        spawn="dataset",
        start_altitude_agl_m=None,
        altitude_cap_m=1000.0,
        success_radius_m=20.0,
        source_path=None,
    )


def _converse(session):
    """Two turns of an operator's conversation: a flight on each, and the last reply."""
    session.flights.append(
        {"target": "prior_1", "resolved_xy": [50.0, 50.0], "provenance": "prior"}
    )
    session.flights.append({"target": "prior_7", "resolved_xy": [5.0, 6.0], "provenance": "prior"})
    return MissionReply(ended_by="reply", turns_used=4)


def test_a_converse_drives_the_conversation_and_the_bench_scores_what_it_flew(
    monkeypatch, tmp_path
) -> None:
    cfg = _stub_engine(monkeypatch, tmp_path)
    row = er.run_episode(
        parse_episode(_ROW),
        {},
        cfg,
        citynav_bench({}),
        settings=Settings(),
        workdir=tmp_path,
        converse=_converse,
    )
    assert row["commit_xy"] == [5.0, 6.0] and row["visited_xy"][-1] == [5.0, 6.0]
    assert row["metrics"]["SR"] == 1 and row["metrics"]["mission"] == {"ended_by": "reply"}


def test_the_engine_flies_an_attempt_as_the_runner_dispatches_it(monkeypatch, tmp_path) -> None:
    cfg = _stub_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(dispatch, "run_env_stamp", lambda bench: {})
    monkeypatch.setattr(
        dispatch, "_cached_run_provenance", lambda flags, **kw: {"split": kw["split"]}
    )
    row = ENGINE.fly_attempt(
        episode=parse_episode(_ROW, row_index=0),
        objects_by_map={},
        cfg=cfg,
        bench=citynav_bench({}),
        settings=Settings(),
        split="val_seen",
        recorder=None,
        episode_dir=tmp_path,
        converse=_converse,
    )
    assert (
        row["status"] == "ok"
        and row["metrics"]["SR"] == 1
        and row["provenance"] == {"split": "val_seen"}
    )
    assert (tmp_path / "episode.json").is_file()


def test_a_process_that_flies_two_run_contexts_stamps_each_with_its_own(monkeypatch) -> None:
    monkeypatch.setattr(dispatch, "_RUN_PROVENANCE", {})
    spec = SimpleNamespace(model_id="m", tokenizer_name="t")
    monkeypatch.setattr(dispatch, "repo_provenance", lambda **kw: dict(kw))
    seen = dispatch._cached_run_provenance(
        {"F": "1"}, spec=spec, config_path=None, split="val_seen"
    )
    unseen = dispatch._cached_run_provenance(
        {"F": "1"}, spec=spec, config_path=None, split="val_unseen"
    )
    assert (seen["split"], unseen["split"]) == ("val_seen", "val_unseen")
    assert (seen["served_model"], seen["served_build"]) == ("m", "t")
    assert (
        dispatch._cached_run_provenance({"F": "1"}, spec=spec, config_path=None, split="val_seen")
        is seen
    )
