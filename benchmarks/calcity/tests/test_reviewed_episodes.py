"""The reviewed allowlist is derived from the shipped corpus and ground truth.

A corpus row is reviewed when its target position matches a position
``ground_truth.json`` lists within 0.2 m. On the shipped
pair that is the 1,100-episode set the Cal City arms are scored on, pinned
here by count and by the sha256 of the sorted ids.
"""

from __future__ import annotations

import hashlib
import os

from benchmarks.calcity.data_prep import reviewed
from benchmarks.calcity.data_prep.reviewed import reviewed_episode_ids
from benchmarks.calcity.paths import calcity_bench

# The corpus as this process's environment places it.
_SPLIT = calcity_bench(os.environ).split_path("calcity")

REVIEWED_N = 1100
REVIEWED_SHA256 = "d23422a381db7708956fae175ce5765f30793edd6a2bb8ecbd6149182707b291"


def test_the_derivation_on_the_shipped_corpus_is_the_reviewed_set():
    ids = reviewed_episode_ids(_SPLIT)
    assert len(ids) == len(set(ids)) == REVIEWED_N
    assert hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest() == REVIEWED_SHA256


def test_the_rows_no_mask_can_score_leave_the_reviewed_set():
    """The two rows that name no single object and the one with neither a mask nor a linked
    object leave the set; the seven whose masks were corrected in review stay in it."""
    ids = set(reviewed_episode_ids(_SPLIT))
    assert ids.isdisjoint({"calcity::610#609", "calcity::633#632", "calcity::99#98"})
    assert ids >= {"calcity::37#36", "calcity::49#48", "calcity::135#134", "calcity::621#620",
                   "calcity::622#621", "calcity::770#769", "calcity::2546#2545"}


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
    monkeypatch.setattr(reviewed, "GROUND_TRUTH", gt)
    assert reviewed_episode_ids(corpus) == ["calcity::0#0"]


def test_a_commit_inside_a_self_crossing_reviewed_mask_scores():
    """calcity::175#174's drawn mask crosses itself; a point inside a lobe (even-odd, the flight
    measures' test) is inside the mask as drawn."""
    import json

    from benchmarks.citynav.adapter import parse_episode
    from benchmarks.citynav.eval import score_arrival

    rows = json.loads(_SPLIT.read_text())
    ep = next(e for e in (parse_episode(r, row_index=i) for i, r in enumerate(rows))
              if e.episode_id == "calcity::175#174")
    row = score_arrival(criterion="shape", final_pos_xy=(3740.64, 5826.91), target_xy=ep.target_xy,
                        gt_polygon=ep.target_mask)
    assert row["SR"] == 1


def test_every_reviewed_row_names_its_intel_quality_or_none():
    """A row's ``intel_quality`` is its graded candidate count's class: G one, M two to five,
    P six or more; the hand-written rows no count was graded for carry none."""
    import json
    from collections import Counter

    from benchmarks.citynav.adapter import parse_episode

    rows = json.loads(_SPLIT.read_text())
    by_id = {e.episode_id: e for e in (parse_episode(r, row_index=i) for i, r in enumerate(rows))}
    assert Counter(by_id[e].intel_quality for e in reviewed_episode_ids(_SPLIT)) == {
        "G": 593, "M": 89, "P": 7, None: 411}
    assert {r.get("intel_quality") for r in rows} == {"G", "M", "P", None}
