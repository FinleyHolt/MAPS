"""The closed-loop measures on a synthetic sortie: SR, OSR, NE, time to identification, the excluded run."""

from __future__ import annotations

import json


from benchmarks.calcity.paths import calcity_bench
from benchmarks.flight import measures as Q

#: The bench whose engine parses the tasking set's rows.
BENCH = calcity_bench({})

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
    return {"episode_id": f"calcity::{wd}", "map_name": "calcity", "run_no": int(wd[-1]), "workdir": wd, "status": "ok",
            "ended_by": "reply" if reply else "stopped", "started_at_utc": T0, "elapsed_s": t_s,
            "start_xy": [0.0, -2000.0], "visited_xy": [[0.0, -2000.0], commit[1]], "flights_flown": 1, "turns_used": 4}


def test_the_measures_score_the_commit_point_against_the_mask_and_the_track_against_the_hfov(tmp_path, capsys):
    root, truth = tmp_path / "20260101T000000Z", tmp_path / "truth"
    rows = [_run(root, "run1", commit=("prior_0", [101.0, 99.0])),
            _run(root, "run2", commit=("prior_9", [400.0, 100.0])),
            _run(root, "run3", commit=("prior_0", [101.0, 99.0]), reply=False),
            # Inside the mask with no outline to recover: a success.
            _run(root, "run4", commit=("detected_3", [95.0, 95.0])),
            # 15.8 m from the centroid but outside the mask: a miss.
            _run(root, "run5", commit=("prior_0", [105.0, 115.0]))]
    (root / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    # North along x = 0 at 100 m/s, 600 m above the ground: the target lies 100 m off the line.
    (root / "flight_track.jsonl").write_text("".join(
        json.dumps({"t_unix": T0 + k, "x": 0.0, "y": -2000.0 + 100.0 * k, "alt_agl_m": 600.0, "track_deg": 0.0,
                    "ground_speed_mps": 100.0}) + "\n" for k in range(700)))
    truth.mkdir()
    (truth / f"{root.name}.json").write_text(json.dumps({
        "calcity::run1": {"target_mask": TARGET, "intel_quality": ["G"]},
        "calcity::run2": {"target_mask": TARGET, "intel_quality": ["M"]},
        "calcity::run3": {"target_mask": TARGET, "excluded": "crew stopped it"},
        "calcity::run4": {"target_mask": TARGET}, "calcity::run5": {"target_mask": TARGET}}))
    got, radius, _turns = Q.collect([root], Q.Truth([truth], BENCH))
    one, two, three, four, five = got
    assert (one["success"], one["oracle_success"]) == (True, True)
    assert one["ne_ft"] == round(2 ** 0.5 * Q.FT_PER_M, 1) and one["time_to_id_s"] == 600.0
    assert one["spl"] is not None and 0.0 < one["spl"] <= 1.0
    assert one["agl_ft"] == round(600.0 * Q.FT_PER_M, 1)
    assert (two["success"], two["spl"]) == (False, 0.0) and round(two["ne_ft"] / Q.FT_PER_M) == 300
    assert three["excluded"] == "crew stopped it" and three["end"] == "stopped" and three["time_to_id_s"] is None
    assert four["success"] is True and five["success"] is False
    assert list(Q.by(got, "model", "reasoning_effort")) == ["m 0.9"]
    s = Q.summarize(got)
    assert (s["sr"]["k"], s["sr"]["n"], s["osr"]["n"]) == (2, 4, 4)
    assert Q.main(["--root", str(root), "--truth", str(truth), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["sr"] == {"k": 2, "n": 4, "pct": 50.0}
    # The track flies straight: no turn is read, and the radius is the fallback's.
    assert Q.main(["--root", str(root), "--truth", str(truth)]) == 0
    assert f"turn radius {Q.TURN_RADIUS_FALLBACK_M} m (fallback: no turn read)" in capsys.readouterr().out


def _keyframe_run(tmp_path, *keyframe_xy, gnss_agl_m=600.0, fixes=5):
    """A run whose recording holds ``fixes`` s far from the target (its fixes' height
    ``gnss_agl_m``, None for none), plus a keyframe pose a second at each ``keyframe_xy``."""
    import datetime as dt

    root, truth = tmp_path / "20260101T000000Z", tmp_path / "truth"
    row = _run(root, "run1", commit=("prior_9", [400.0, 100.0]))
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    (root / "flight_track.jsonl").write_text("".join(
        json.dumps({"t_unix": T0 + k, "x": 5000.0, "y": 0.0, "alt_agl_m": gnss_agl_m, "track_deg": 0.0,
                    "ground_speed_mps": 100.0}) + "\n" for k in range(fixes)))
    (root / "run1" / "frames.jsonl").write_text("".join(
        json.dumps({"pose_xy": xy, "utc": dt.datetime.fromtimestamp(T0 + 300.0 + k, dt.timezone.utc).isoformat(),
                    "altitude_m": 600.0}) + "\n" for k, xy in enumerate(keyframe_xy)))
    truth.mkdir()
    (truth / f"{root.name}.json").write_text(json.dumps({"calcity::run1": {"target_mask": TARGET}}))
    (got,), _radius, _turns = Q.collect([root], Q.Truth([truth], BENCH))
    return got


def test_osr_reads_the_keyframe_poses_beside_a_thin_recording(tmp_path):
    got = _keyframe_run(tmp_path, [120.0, 100.0])
    assert got["track_coverage"] < Q.SPL_MIN_COVERAGE
    assert got["oracle_success"] is True
    assert (got["track_source"], got["agl_source"]) == ("keyframe poses", "keyframes")


def test_osr_reads_the_keyframe_poses_beside_a_full_recording(tmp_path):
    got = _keyframe_run(tmp_path, [120.0, 100.0], fixes=601)
    assert got["track_coverage"] == 1.0
    assert (got["oracle_success"], got["track_source"]) == (True, "keyframe poses")
    assert _keyframe_run(tmp_path / "far", [3000.0, 3000.0], fixes=601)["oracle_success"] is False


def test_a_gnss_fix_with_no_height_reads_the_keyframes_median_height(tmp_path):
    got = _keyframe_run(tmp_path, [9000.0, 9000.0], gnss_agl_m=None)
    assert (got["track_source"], got["agl_source"]) == ("GNSS track", "keyframes (median)")


def test_an_observation_s_height_above_ground_is_its_altitude_over_the_map_s_ground_level(tmp_path, monkeypatch):
    monkeypatch.setitem(Q.GROUND_LEVEL, "calcity", 100.0)
    for keyframe_xy, gnss_agl_m in (([120.0, 100.0], 600.0), ([9000.0, 9000.0], None)):
        got = _keyframe_run(tmp_path / str(gnss_agl_m), keyframe_xy, gnss_agl_m=gnss_agl_m)
        assert got["agl_ft"] == round(500.0 * Q.FT_PER_M, 1), got["agl_source"]


def test_keyframe_poses_are_read_as_points_not_the_line_between_them(tmp_path):
    # Two observations 1,500 m either side of the target: the line joining them crosses it.
    got = _keyframe_run(tmp_path, [-1400.0, 100.0], [1600.0, 100.0], fixes=601)
    assert got["track_source"] == "keyframe poses"
    assert got["closest_approach_ft"] == round(1500.0 * Q.FT_PER_M, 1)
    assert got["oracle_success"] is False


def test_a_thin_recording_with_no_pose_near_the_target_cannot_show_a_miss(tmp_path):
    assert _keyframe_run(tmp_path, [3000.0, 3000.0])["oracle_success"] is None


def test_a_run_the_serve_failed_under_is_out_of_every_rate_and_named(tmp_path):
    root, truth = tmp_path / "20260101T000000Z", tmp_path / "truth"
    rows = [_run(root, "run1", commit=("prior_0", [101.0, 99.0])),
            {**_run(root, "run2", commit=("prior_9", [400.0, 100.0])), "ended_by": "transient_failures"},
            {**_run(root, "run3", commit=("prior_9", [400.0, 100.0])), "status": "infra"}]
    (root / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    truth.mkdir()
    (truth / f"{root.name}.json").write_text(json.dumps(
        {f"calcity::run{k}": {"target_mask": TARGET} for k in (1, 2, 3)}))
    got, _radius, _turns = Q.collect([root], Q.Truth([truth], BENCH))
    assert [r["end"] for r in got] == ["reply", "transient_failures", "no run"]
    s = Q.summarize(got)
    assert (s["sr"]["k"], s["sr"]["n"]) == (1, 1)
    assert s["infra"] == [f"{root.name} run 2", f"{root.name} run 3"]


def test_the_tasking_set_scores_a_run_only_for_the_description_it_was_flown_on(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps([{"area": "calcity", "descriptions": ["the Rite Aid"], "ann_ids": [0],
                                  "target_positions": [[100.0, 100.0, 0.0]], "trajectory": [[0.0, 0.0, 125.0]],
                                  "target_mask": TARGET}]))
    truth = Q.Truth([split], BENCH)
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
    truth = Q.Truth([split], BENCH)
    ring, source = truth.ring({}, {"episode_id": "calcity::0#0", "description": "the school"})
    assert source == "tasking set" and ring[0] == [500.0, 0.0]
    assert truth.ring({}, {"episode_id": "calcity::2#2", "description": "the courts"}) == (None, None)


def test_a_track_with_no_fix_in_the_run_covers_none_of_it():
    track = [{"t_unix": T0 + 100.0, "x": 0.0, "y": 0.0}]
    assert Q.track_coverage(track, T0, T0 + 2.0) == 0.0
    assert Q.track_coverage([], T0, T0 + 2.0) == 0.0


def test_a_success_the_recording_holds_one_fix_of_has_its_spl_named_undefined(tmp_path):
    root, truth = tmp_path / "20260101T000000Z", tmp_path / "truth"
    (root / "episodes.jsonl").parent.mkdir()
    row = _run(root, "run1", commit=("prior_0", [101.0, 99.0]))
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    (root / "flight_track.jsonl").write_text(json.dumps({"t_unix": T0 + 10.0, "x": 0.0, "y": 0.0, "alt_agl_m": 600.0,
                                                         "track_deg": 0.0, "ground_speed_mps": 100.0}) + "\n")
    truth.mkdir()
    (truth / f"{root.name}.json").write_text(json.dumps({"calcity::run1": {"target_mask": TARGET}}))
    (got,), _radius, _turns = Q.collect([root], Q.Truth([truth], BENCH))
    assert got["success"] is True and got["spl"] is None
    assert got["spl_note"] == "undefined: the recording does not cover the run"


def test_a_torn_last_ledger_line_is_skipped(tmp_path):
    """A sortie whose writer was cut mid-line still reads its complete rows."""
    root, truth = tmp_path / "20260101T000000Z", tmp_path / "truth"
    row = _run(root, "run1", commit=("prior_0", [101.0, 99.0]))
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row)[:40])
    truth.mkdir()
    (truth / f"{root.name}.json").write_text(json.dumps({"calcity::run1": {"target_mask": TARGET}}))
    got, _radius, _turns = Q.collect([root], Q.Truth([truth], BENCH))
    assert [r["episode_id"] for r in got] == ["calcity::run1"]


def test_a_mission_that_failed_off_the_serve_reads_failed():
    """A mission that failed with no serve fault is a scored miss and reads
    ``failed``, not ``no run``."""
    assert Q.ended({"status": "ok", "ended_by": None, "error": "KeyError: 'x'"}) == "failed"
    assert Q.ended({"status": "infra", "error": "SamServerOutage"}) == "no run"


def test_a_track_ending_under_a_second_before_the_report_owes_no_way_back(tmp_path):
    root, truth = tmp_path / "20260101T000000Z", tmp_path / "truth"
    row = _run(root, "run1", commit=("prior_0", [100.0, 100.0]), t_s=20.5)
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    # North along x = 100 at 100 m/s, the last fix at t = 20 50 m past the target, the report at 20.5.
    (root / "flight_track.jsonl").write_text("".join(
        json.dumps({"t_unix": T0 + k, "x": 100.0, "y": -1850.0 + 100.0 * k, "alt_agl_m": 600.0, "track_deg": 0.0,
                    "ground_speed_mps": 100.0}) + "\n" for k in range(21)))
    truth.mkdir()
    (truth / f"{root.name}.json").write_text(json.dumps({"calcity::run1": {"target_mask": TARGET}}))
    (got,), _radius, _turns = Q.collect([root], Q.Truth([truth], BENCH))
    assert got["final_leg_ft"] == 0.0


def test_a_run_folder_has_a_sortie_copy_and_a_ground_station_copy(tmp_path):
    root = tmp_path / "20260101T000000Z"
    bases = Q.run_bases(root, {"workdir": "run01"})
    assert bases == [root / "run01", tmp_path / "20260101T000000Z_grnd" / "20260101T000000Z" / "run01"]
