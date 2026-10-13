"""The reviewed allowlist is derived from the shipped corpus and ground truth.

A corpus row is reviewed when its target position matches a position
``ground_truth.json`` lists within 0.2 m. On the shipped
pair that is the 603-episode set the Cal City arms are scored on, pinned
here by count and by the sha256 of the sorted ids.
"""

from __future__ import annotations

import hashlib

from benchmarks.calcity.data_prep import reviewed
from benchmarks.calcity.data_prep.reviewed import reviewed_episode_ids

REVIEWED_N = 603
REVIEWED_SHA256 = "e62c32066d7a06d4e32e73912b1ea8c58d7636827c694b4dded2a01841186bce"


def test_the_derivation_on_the_shipped_corpus_is_the_reviewed_set():
    ids = reviewed_episode_ids()
    assert len(ids) == len(set(ids)) == REVIEWED_N
    assert hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest() == REVIEWED_SHA256


def test_only_the_three_rows_that_name_no_single_object_leave_the_reviewed_set():
    """Of the ten rows whose drawn mask no commit on the named target could
    hit, seven are scored on a reviewed mask and three leave the set."""
    ids = set(reviewed_episode_ids())
    assert ids.isdisjoint({"calcity::31#30", "calcity::614#612", "calcity::636#634"})
    assert ids >= {"calcity::38#37", "calcity::50#49", "calcity::137#135", "calcity::624#622",
                   "calcity::625#623", "calcity::658#656", "calcity::962#956"}


def test_a_row_matches_only_within_the_tolerance(tmp_path, monkeypatch):
    import json

    corpus = tmp_path / "c.json"
    gt = tmp_path / "gt.json"
    row = {"area": "calcity", "descriptions": ["x"], "ann_ids": [0], "object_ids": [],
           "marker_positions": [], "trajectory": [[0.0, 0.0, 125.0]]}
    corpus.write_text(json.dumps([
        {**row, "ann_ids": [0], "target_positions": [[10.15, 20.0, 0.0]]},
        {**row, "ann_ids": [1], "target_positions": [[10.25, 20.0, 0.0]]},
    ]))
    gt.write_text(json.dumps(["calcity::10.0:20.0"]))
    monkeypatch.setattr(reviewed, "split_path", lambda split: corpus)
    monkeypatch.setattr(reviewed, "GROUND_TRUTH", gt)
    assert reviewed_episode_ids() == ["calcity::0#0"]


def test_a_commit_inside_a_self_crossing_reviewed_mask_scores():
    """calcity::177#175's drawn mask crosses itself; a point inside a lobe (even-odd, the flight
    measures' test) is inside the mask as drawn."""
    import json

    from benchmarks.calcity.paths import split_path
    from benchmarks.citynav.adapter import parse_episode
    from benchmarks.citynav.eval import score_arrival

    rows = json.loads(split_path("calcity").read_text())
    ep = next(e for e in (parse_episode(r, row_index=i) for i, r in enumerate(rows))
              if e.episode_id == "calcity::177#175")
    row = score_arrival(criterion="shape", final_pos_xy=(3740.64, 5826.91), target_xy=ep.target_xy,
                        gt_polygon=ep.target_mask)
    assert row["SR"] == 1
