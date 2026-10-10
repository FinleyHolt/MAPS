"""The reviewed allowlist is derived from the shipped corpus and ground truth.

A corpus row is reviewed when its target position matches a reviewed shape
in ``ground_truth.json`` (keyed by position) within 0.2 m. On the shipped
pair that is the 606-episode set the Cal City arms were scored on, pinned
here by count and by the sha256 of the sorted ids.
"""

from __future__ import annotations

import hashlib

from benchmarks.calcity.data_prep.reviewed import reviewed_episode_ids

REVIEWED_N = 606
REVIEWED_SHA256 = "b547f241609d576aa6033ba38504b7047a9811d580e5adf221d1583b4228ad4a"


def test_the_derivation_on_the_shipped_corpus_is_the_reviewed_set():
    ids = reviewed_episode_ids()
    assert len(ids) == len(set(ids)) == REVIEWED_N
    assert hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest() == REVIEWED_SHA256


def test_a_row_matches_only_within_the_tolerance(tmp_path):
    import json

    corpus = tmp_path / "c.json"
    gt = tmp_path / "gt.json"
    row = {"area": "calcity", "descriptions": ["x"], "ann_ids": [0], "object_ids": [],
           "marker_positions": [], "trajectory": [[0.0, 0.0, 125.0]], "source": "authored"}
    corpus.write_text(json.dumps([
        {**row, "ann_ids": [0], "target_positions": [[10.15, 20.0, 0.0]]},
        {**row, "ann_ids": [1], "target_positions": [[10.25, 20.0, 0.0]]},
    ]))
    gt.write_text(json.dumps({"calcity::10.0:20.0": {"object_id": None, "target_mask": []}}))
    assert reviewed_episode_ids(corpus, gt) == ["calcity::0#0"]
