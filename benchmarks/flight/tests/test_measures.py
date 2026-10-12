"""The closed-loop measures on a synthetic sortie: SR both ways, OSR, NE, time to identification, the excluded run."""

from __future__ import annotations

import csv
import json

from benchmarks.flight import measures as Q

T0 = 1_800_000_000.0
TARGET = [[90.0, 90.0], [110.0, 90.0], [110.0, 110.0], [90.0, 110.0]]


def _run(root, wd, *, commit, reply=True, t_s=600.0):
    run = root / wd
    (run / "flight" / "llm").mkdir(parents=True)
    (run / "flight" / "tour_cursor.json").write_text(json.dumps({"stops_visited": [{"stop_id": commit[0],
                                                                                     "xy": commit[1]}]}))
    (run / "flight" / "llm" / "transcript.jsonl").write_text(
        json.dumps({"kind": "params", "params": {"model": "m", "reasoning_effort": "0.9"}}) + "\n")
    (run / "conversation.jsonl").write_text(json.dumps({"kind": "leg", "said": "arrived", "t_s": 100.0}) + "\n")
    return {"episode_id": f"calcity::{wd}", "run_no": int(wd[-1]), "workdir": wd, "status": "ok",
            "ended_by": "reply" if reply else "stopped", "started_at_utc": T0, "elapsed_s": t_s,
            "start_xy": [0.0, -2000.0], "visited_xy": [[0.0, -2000.0], commit[1]], "flights_flown": 1, "turns_used": 4}


def test_the_measures_score_the_commit_against_the_truth_and_the_track_against_the_hfov(tmp_path, capsys):
    root, truth, tiles = tmp_path / "20260101T000000Z", tmp_path / "truth", tmp_path / "tiles"
    tiles.mkdir()
    (tiles / "calcity_features.json").write_text(json.dumps({"prior": [{"world_contour": TARGET}]}))
    rows = [_run(root, "run1", commit=("prior_0", [101.0, 99.0])),
            _run(root, "run2", commit=("prior_9", [400.0, 100.0])),
            _run(root, "run3", commit=("prior_0", [101.0, 99.0]), reply=False)]
    (root / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    # North along x = 0 at 100 m/s, 600 m above the ground: the target lies 100 m off the line.
    (root / "flight_track.jsonl").write_text("".join(
        json.dumps({"t_unix": T0 + k, "x": 0.0, "y": -2000.0 + 100.0 * k, "alt_agl_m": 600.0, "track_deg": 0.0,
                    "ground_speed_mps": 100.0}) + "\n" for k in range(700)))
    truth.mkdir()
    (truth / f"{root.name}.json").write_text(json.dumps({
        "calcity::run1": {"target_mask": TARGET, "intel_quality": ["G"]},
        "calcity::run2": {"target_mask": TARGET, "intel_quality": ["M"]},
        "calcity::run3": {"target_mask": TARGET, "excluded": "crew stopped it"}}))
    got, radius, _turns = Q.collect([root], Q.Truth([truth]), tiles_dir=tiles)
    one, two, three = got
    assert (one["success_area"], one["success_radius"], one["oracle_success"]) == (True, True, True)
    assert one["ne_from"] == "committed area centroid" and one["ne_ft"] == 0.0 and one["time_to_id_s"] == 600.0
    assert one["spl"] is not None and 0.0 < one["spl"] <= 1.0
    assert one["agl_ft"] == round(600.0 * Q.FT_PER_M, 1)
    assert (two["success_area"], two["success_radius"], two["spl"]) == (False, False, 0.0)
    assert two["ne_from"] == "commit point" and round(two["ne_ft"] / Q.FT_PER_M) == 300
    assert three["excluded"] == "crew stopped it" and three["end"] == "stopped" and three["time_to_id_s"] is None
    assert list(Q.by(got, "model", "reasoning_effort")) == ["m 0.9"]
    s = Q.summarize(got)
    assert (s["sr_area"]["k"], s["sr_area"]["n"], s["osr"]["n"]) == (1, 2, 2)
    master = tmp_path / "runs_all.csv"
    with master.open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["environment", "sortie", "run_no", "success_area", "success_radius", "oracle_success"])
        w.writeheader()
        for r in got:
            w.writerow({"environment": "real", "sortie": r["sortie"], "run_no": r["run_no"],
                        **{k: "" if r[k] is None else str(r[k]) for k in ("success_area", "success_radius",
                                                                          "oracle_success")}})
    assert Q.check(got, master) == []
    one["success_area"] = False
    assert Q.check(got, master) == [f"('{root.name}', 1) success_area: False here, 'True' in the master table"]
    # run 2 commits prior_9, which this bake does not hold.
    assert Q.main(["--root", str(root), "--truth", str(truth), "--tiles", str(tiles), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["prior_outline_rejected"] == [f"{root.name} run 2"]


def _thin_track_run(tmp_path, keyframe_xy):
    """A run whose recording holds 5 s far from the target, plus one keyframe pose at ``keyframe_xy``."""
    import datetime as dt

    root, truth, tiles = tmp_path / "20260101T000000Z", tmp_path / "truth", tmp_path / "tiles"
    tiles.mkdir()
    (tiles / "calcity_features.json").write_text(json.dumps({"prior": []}))
    row = _run(root, "run1", commit=("prior_9", [400.0, 100.0]))
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    (root / "flight_track.jsonl").write_text("".join(
        json.dumps({"t_unix": T0 + k, "x": 5000.0, "y": 0.0, "alt_agl_m": 600.0, "track_deg": 0.0,
                    "ground_speed_mps": 100.0}) + "\n" for k in range(5)))
    utc = dt.datetime.fromtimestamp(T0 + 300.0, dt.timezone.utc).isoformat()
    (root / "run1" / "frames.jsonl").write_text(json.dumps({"pose_xy": keyframe_xy, "utc": utc,
                                                            "altitude_m": 600.0}) + "\n")
    truth.mkdir()
    (truth / f"{root.name}.json").write_text(json.dumps({"calcity::run1": {"target_mask": TARGET}}))
    (got,), _radius, _turns = Q.collect([root], Q.Truth([truth]), tiles_dir=tiles)
    return got


def test_osr_reads_the_keyframe_poses_beside_a_thin_recording(tmp_path):
    got = _thin_track_run(tmp_path, [120.0, 100.0])
    assert got["track_coverage"] < Q.SPL_MIN_COVERAGE
    assert got["oracle_success"] is True
    assert got["agl_source"] == "keyframes"


def test_a_thin_recording_with_no_pose_near_the_target_cannot_show_a_miss(tmp_path):
    assert _thin_track_run(tmp_path, [3000.0, 3000.0])["oracle_success"] is None


def test_a_run_the_serve_failed_under_is_out_of_every_rate_and_named(tmp_path):
    root, truth, tiles = tmp_path / "20260101T000000Z", tmp_path / "truth", tmp_path / "tiles"
    tiles.mkdir()
    (tiles / "calcity_features.json").write_text(json.dumps({"prior": [{"world_contour": TARGET}]}))
    rows = [_run(root, "run1", commit=("prior_0", [101.0, 99.0])),
            {**_run(root, "run2", commit=("prior_9", [400.0, 100.0])), "ended_by": "transient_failures"},
            {**_run(root, "run3", commit=("prior_9", [400.0, 100.0])), "status": "infra"}]
    (root / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    truth.mkdir()
    (truth / f"{root.name}.json").write_text(json.dumps(
        {f"calcity::run{k}": {"target_mask": TARGET} for k in (1, 2, 3)}))
    got, _radius, _turns = Q.collect([root], Q.Truth([truth]), tiles_dir=tiles)
    assert [r["end"] for r in got] == ["reply", "transient_failures", "no run"]
    s = Q.summarize(got)
    assert (s["sr_20m"]["k"], s["sr_20m"]["n"]) == (1, 1)
    assert s["infra"] == [f"{root.name} run 2", f"{root.name} run 3"]


def test_a_prior_commit_with_no_bake_features_is_refused_not_scored_a_miss(tmp_path):
    import pytest

    root, truth = tmp_path / "20260101T000000Z", tmp_path / "truth"
    (root / "episodes.jsonl").parent.mkdir()
    row = _run(root, "run1", commit=("prior_0", [101.0, 99.0]))
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    truth.mkdir()
    (truth / f"{root.name}.json").write_text(json.dumps({"calcity::run1": {"target_mask": TARGET}}))
    with pytest.raises(SystemExit, match="--tiles"):
        Q.collect([root], Q.Truth([truth]), tiles_dir=tmp_path / "no_bake")


def test_the_tasking_set_scores_a_run_only_for_the_description_it_was_flown_on(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps([{"area": "calcity", "descriptions": ["the Rite Aid"], "ann_ids": [0],
                                  "target_positions": [[100.0, 100.0, 0.0]], "trajectory": [[0.0, 0.0, 125.0]],
                                  "target_mask": TARGET}]))
    truth = Q.Truth([split])
    assert truth.ring({}, {"episode_id": "calcity::0#0", "description": "the Rite Aid"})[1] == "tasking set"
    assert truth.ring({}, {"episode_id": "calcity::0#0", "description": "the school"}) == (None, None)


def test_the_tasking_set_finds_a_run_by_its_description_wherever_the_split_lists_it(tmp_path):
    """A flown episode id is the run's position in that sortie's tasking, not
    in the tasking file, so the description is the join; a description two
    entries give different footprints scores nothing."""
    def entry(desc, x):
        ring = [[x, 0.0], [x + 10, 0.0], [x + 10, 10.0], [x, 10.0]]
        return {"area": "calcity", "descriptions": [desc], "ann_ids": [0], "target_positions": [[x, 5.0, 0.0]],
                "trajectory": [[0.0, 0.0, 125.0]], "target_mask": ring}
    split = tmp_path / "split.json"
    split.write_text(json.dumps([entry("the Rite Aid", 0.0), entry("the school", 500.0),
                                 entry("the courts", 900.0), entry("the courts", 950.0)]))
    truth = Q.Truth([split])
    ring, source = truth.ring({}, {"episode_id": "calcity::0#0", "description": "the school"})
    assert source == "tasking set" and ring[0] == [500.0, 0.0]
    assert truth.ring({}, {"episode_id": "calcity::2#2", "description": "the courts"}) == (None, None)


def test_a_track_with_no_fix_in_the_run_covers_none_of_it():
    track = [{"t_unix": T0 + 100.0, "x": 0.0, "y": 0.0}]
    assert Q.track_coverage(track, T0, T0 + 2.0) == 0.0
    assert Q.track_coverage([], T0, T0 + 2.0) == 0.0


def test_a_success_the_recording_holds_one_fix_of_has_its_spl_named_undefined(tmp_path):
    root, truth, tiles = tmp_path / "20260101T000000Z", tmp_path / "truth", tmp_path / "tiles"
    tiles.mkdir()
    (tiles / "calcity_features.json").write_text(json.dumps({"prior": [{"world_contour": TARGET}]}))
    (root / "episodes.jsonl").parent.mkdir()
    row = _run(root, "run1", commit=("prior_0", [101.0, 99.0]))
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    (root / "flight_track.jsonl").write_text(json.dumps({"t_unix": T0 + 10.0, "x": 0.0, "y": 0.0, "alt_agl_m": 600.0,
                                                         "track_deg": 0.0, "ground_speed_mps": 100.0}) + "\n")
    truth.mkdir()
    (truth / f"{root.name}.json").write_text(json.dumps({"calcity::run1": {"target_mask": TARGET}}))
    (got,), _radius, _turns = Q.collect([root], Q.Truth([truth]), tiles_dir=tiles)
    assert got["success_area"] is True and got["spl"] is None
    assert got["spl_note"] == "undefined: the recording does not cover the run"


def test_a_torn_last_ledger_line_is_skipped(tmp_path):
    """A sortie whose writer was cut mid-line still reads its complete rows."""
    root, truth = tmp_path / "20260101T000000Z", tmp_path / "truth"
    row = _run(root, "run1", commit=("prior_0", [101.0, 99.0]))
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row)[:40])
    truth.mkdir()
    (truth / f"{root.name}.json").write_text(json.dumps({"calcity::run1": {"target_mask": TARGET}}))
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    (tiles / "calcity_features.json").write_text(json.dumps({"prior": [{"world_contour": TARGET}]}))
    got, _radius, _turns = Q.collect([root], Q.Truth([truth]), tiles_dir=tiles)
    assert [r["episode_id"] for r in got] == ["calcity::run1"]


def test_a_mission_that_failed_off_the_serve_reads_failed():
    """A mission that failed with no serve fault is a scored miss and reads
    ``failed``, not ``no run``."""
    assert Q.ended({"status": "ok", "ended_by": None, "error": "KeyError: 'x'"}) == "failed"
    assert Q.ended({"status": "infra", "error": "SamServerOutage"}) == "no run"


def test_a_prior_commit_off_its_outline_is_named_on_a_run_with_no_truth(tmp_path):
    root, tiles = tmp_path / "20260101T000000Z", tmp_path / "tiles"
    tiles.mkdir()
    (tiles / "calcity_features.json").write_text(json.dumps({"prior": [{"world_contour": TARGET}]}))
    row = _run(root, "run1", commit=("prior_0", [400.0, 100.0]))
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    got, _radius, _turns = Q.collect([root], Q.Truth([]), tiles_dir=tiles)
    assert got[0]["truth_xy"] is None
    assert Q.outline_rejected(got, Q.load_prior(tiles)) == [f"{root.name} run 1"]
