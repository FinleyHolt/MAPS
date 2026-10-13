"""The shipped corpus has episodes that start ON their own target.

The zero-flight branch writes an honest zero: the drone never took off and
nothing was committed, so no success flag may be set, ``SR``/``OSR``
included. No episode reaches a ring verdict from its start pose.

Under the SHAPE criterion the distance that matters is to the footprint's
boundary, not its centroid, and rows of the shipped corpus start inside their
own target. On those, a never-flown episode scored SR = 1.

Derived from the corpus rather than hardcoded, so a rebuild that loses the
case says so instead of leaving a test that passes on nothing.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.calcity import paths as calcity_paths
from benchmarks.citynav import eval as cn_eval


def _episodes_that_start_on_their_target() -> list[int]:
    rows = json.loads(calcity_paths.split_path("calcity").read_text())
    out = []
    for i, row in enumerate(rows):
        mask = row.get("target_mask") or []
        traj = row.get("trajectory") or []
        target = (row.get("target_positions") or [None])[0]
        if len(mask) < 3 or not traj or not target:
            continue
        start = (float(traj[0][0]), float(traj[0][1]))
        scored = cn_eval.score_precise(
            final_xy=start,
            target_xy=(float(target[0]), float(target[1])),
            gt_polygon=mask,
            radius_m=float(row.get("success_radius_m") or 20.0),
        )
        if scored["SR"]:
            out.append(i)
    return out


@pytest.fixture(scope="module")
def start_on_target() -> list[int]:
    found = _episodes_that_start_on_their_target()
    # VACUITY FLOOR. Without the case this whole file passes on an empty
    # loop, and the bug it guards would be reachable again unnoticed.
    assert found, (
        "no Cal City episode starts on its own target any more -- the corpus "
        "changed. Confirm the honest zero is still guarded before deleting "
        "this file: the bug it caught was a never-flown episode scoring SR=1."
    )
    return found


def test_the_corpus_still_holds_the_case(start_on_target):
    assert len(start_on_target) >= 1


def test_those_episodes_score_zero_when_nothing_flew(start_on_target):
    """What the zero-flight branch produces for them, through the one
    scorer it calls."""
    rows = json.loads(calcity_paths.split_path("calcity").read_text())
    for idx in start_on_target:
        row = rows[idx]
        start = (float(row["trajectory"][0][0]), float(row["trajectory"][0][1]))
        target = (
            float(row["target_positions"][0][0]),
            float(row["target_positions"][0][1]),
        )
        mask = {
            "gt_polygon": row["target_mask"],
            "radius_m": float(row.get("success_radius_m") or 20.0),
        }
        flown = cn_eval.score_arrival(
            criterion="shape", final_pos_xy=start,
            target_xy=target, pos_trajectory=[start, start], **mask)
        assert flown["SR"] == 1, (
            f"row {idx}: the fixture must win when it flies, or the zero "
            "below proves nothing"
        )
        grounded = cn_eval.score_arrival(
            criterion="shape", final_pos_xy=start,
            target_xy=target, pos_trajectory=[start, start], flew=False, **mask)
        assert grounded["SR"] == 0, f"row {idx}: never flew, must not score"
