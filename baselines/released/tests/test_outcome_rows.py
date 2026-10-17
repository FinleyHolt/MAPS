"""Converted rows read through the canonical readers, and pair by episode id."""

from __future__ import annotations

import json

import pytest

from baselines.released import outcome_rows as orows
from benchmarks.citynav.analysis import run_report

N = 12  # synthetic split size


def _split_rows():
    return [
        {"area": "birmingham", "block": 1, "object_ids": [i], "ann_ids": [i % 3],
         "descriptions": [f"target {i}"], "trajectory": [[0.0, 0.0, 100.0, 1.0, 0.0, -1.0]],
         "marker_positions": [[100.0 + i, 0.0, 10.0]],
         "target_positions": [[100.0 + i, 0.0, 10.0]],
         "total_score": 1.0, "dist_marker_to_target": 0.0, "split": "test_unseen"}
        for i in range(N)
    ]


def _mgp_native(path, rows, dropped=()):
    """MGP's own result layout: str(tuple) keys, [x, y, z, yaw] poses; even
    episodes land on the target, odd ones 50 m short."""
    logs = {}
    for i, r in enumerate(rows):
        if i in dropped:
            continue
        tx = r["target_positions"][0][0]
        end = tx if i % 2 == 0 else tx - 50.0
        logs[str(("birmingham_block_1", i, i % 3))] = [[0.0, 0.0, 60.0, 0.0], [end, 0.0, 60.0, 0.0]]
    path.write_text(json.dumps({"metrics": {}, "trajectory_logs": logs}))
    return path


def test_citynav_rows_read_through_the_canonical_readers(tmp_path):
    rows = _split_rows()
    trajs, errors = orows.read_native(_mgp_native(tmp_path / "mgp.json", rows, dropped={3}))
    ident = orows.stamp("mgp", "mgp_mturk.pth sha256:abc", "original", "test_unseen")
    out_rows, sets = orows.citynav_rows(trajs, errors, rows, rows, ident)
    orows.write_run(tmp_path / "run", out_rows, sets)

    scored = run_report.load_run(tmp_path / "run")
    assert len(scored) == N - 1 >= 10  # vacuity floor: the read found the rows
    assert set(scored) == set(sets["evaluated"])
    assert sets["dropped"] == {"birmingham_block_1::0#3": "upstream_loader_filter"}
    assert run_report.run_criterion(list(scored.values())) == "ring"
    sr = {eid: m["SR"] for eid, m in scored.items()}
    assert sr["birmingham_block_1::0#0"] == 1 and sr["birmingham_block_1::1#1"] == 0
    assert sum(sr.values()) == 6

    row = json.loads((tmp_path / "run" / "episodes.jsonl").read_text().splitlines()[0])
    for key in ("method", "upstream_sha", "checkpoint", "protocol", "dataset_version"):
        assert row[key] and row["provenance"][key] == row[key]
    assert row["protocol"] == "upstream_released"
    assert row["upstream_sha"] == orows.RELEASES["mgp"].sha
    assert row["provenance"]["split"] == "test_unseen"


def test_rows_pair_against_a_maps_run_on_shared_ids(tmp_path):
    rows = _split_rows()
    trajs, _ = orows.read_native(_mgp_native(tmp_path / "mgp.json", rows))
    ident = orows.stamp("mgp", "ckpt", "original", "test_unseen")
    out_rows, sets = orows.citynav_rows(trajs, [], rows, rows, ident)
    orows.write_run(tmp_path / "baseline", out_rows, sets)
    maps_run = tmp_path / "maps"
    maps_run.mkdir()
    with (maps_run / "episodes.jsonl").open("w") as fh:
        for eid in sets["evaluated"][:5]:
            fh.write(json.dumps({"episode_id": eid, "status": "ok", "metrics": {
                "episode_id": eid, "SR": 1, "success_criterion": "ring"}}) + "\n")
    st = orows.pair(tmp_path / "baseline", maps_run)
    released = run_report.load_run(tmp_path / "baseline")
    want = sum(released[e]["SR"] for e in sets["evaluated"][:5]) / 5
    assert 0 < want < 1  # vacuity floor: the shared set holds a hit and a miss
    assert (st["n"], st["SR_a"], st["SR_b"]) == (5, want, 1.0)
    assert (st["only_a"], st["only_b"]) == (len(sets["evaluated"]) - 5, 0)

    (maps_run / "run_manifest.json").write_text(json.dumps({"selection": {"episodes_n": 6}}))
    with pytest.raises(SystemExit, match="allow-partial"):
        orows.main(["pair", str(tmp_path / "baseline"), str(maps_run)])
    assert orows.main(["pair", str(tmp_path / "baseline"), str(maps_run), "--allow-partial"]) == 0


def test_capture_errors_are_dropped_with_their_reason(tmp_path):
    rows = _split_rows()[:4]
    capture = {"method": "flightgpt", "split": "test_unseen",
               "episodes": [{"key": ["birmingham_block_1", 0, 0], "trajectory": [[0, 0], [100, 0]]}],
               "errors": [["birmingham_block_1", 1, 1]]}
    (tmp_path / "c.json").write_text(json.dumps(capture))
    trajs, errors = orows.read_native(tmp_path / "c.json")
    _, sets = orows.citynav_rows(trajs, errors, rows, rows, orows.stamp("flightgpt", "x", "original", "t"))
    assert sets["dropped"]["birmingham_block_1::1#1"] == "upstream_exception"
    assert sets["dropped"]["birmingham_block_1::2#2"] == "upstream_loader_filter"


def test_a_key_outside_the_id_file_refuses():
    with pytest.raises(ValueError, match="not in the id split file"):
        orows.citynav_rows({("cambridge_block_9", 0, 0): [(0.0, 0.0)]}, [], _split_rows(),
                           [], orows.stamp("hett", "x", "revised", "t"))


def test_calcity_rows_score_on_the_shape(tmp_path):
    square = [[90.0, -10.0], [110.0, -10.0], [110.0, 10.0], [90.0, 10.0]]
    manifest = {
        "blocks": {f"calcity_block_{i}": {"episode_id": f"calcity::0#{i}"} for i in range(3)},
        "excluded": {"calcity::0#3": "start_and_target_exceed_block"},
    }
    shapes = {f"calcity::0#{i}": {"gt_polygon": square, "target_xy": (100.0, 0.0), "description": f"target {i}"}
              for i in range(3)}
    trajs = {
        ("calcity_block_0", 9_999_999, 0): [(0.0, 0.0), (100.0, 0.0)],   # inside the shape
        ("calcity_block_1", 9_999_999, 0): [(0.0, 0.0), (115.0, 0.0)],   # in the 20 m ring, off the shape
    }
    out_rows, sets = orows.calcity_rows(
        trajs, [], manifest, shapes, "shape",
        orows.stamp("hett", "x", "calcity", "calcity"),
    )
    orows.write_run(tmp_path / "run", out_rows, sets)
    scored = run_report.load_run(tmp_path / "run")
    assert run_report.run_criterion(list(scored.values())) == "shape"
    assert scored["calcity::0#0"]["SR"] == 1 and scored["calcity::0#1"]["SR"] == 0
    assert sets["dropped"] == {"calcity::0#2": "upstream_loader_filter"}
    assert sets["excluded_by_export"] == manifest["excluded"]
    assert [r["description"] for r in out_rows] == ["target 0", "target 1"]


def test_a_pair_against_another_disc_radius_is_refused(tmp_path):
    rows = _split_rows()
    trajs, _ = orows.read_native(_mgp_native(tmp_path / "mgp.json", rows))
    ident = orows.stamp("mgp", "ckpt", "original", "test_unseen")
    out_rows, sets = orows.citynav_rows(trajs, [], rows, rows, ident)
    orows.write_run(tmp_path / "baseline", out_rows, sets)
    other = tmp_path / "maps40"
    other.mkdir()
    with (other / "episodes.jsonl").open("w") as fh:
        for eid in sets["evaluated"][:5]:
            fh.write(json.dumps({"episode_id": eid, "status": "ok", "metrics": {
                "episode_id": eid, "SR": 1, "success_criterion": "ring"}}) + "\n")
    (other / "run_manifest.json").write_text(json.dumps({"config_yaml": "success_radius_m: 40\n"}))
    with pytest.raises(SystemExit, match="40 m disc"):
        orows.pair(tmp_path / "baseline", other, allow_partial=True)
    with pytest.raises(SystemExit, match="40 m disc"):
        orows.pair(other, tmp_path / "baseline", allow_partial=True)
