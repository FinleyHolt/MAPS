"""The flight report serves what the flight just found (acquisition lever 2).

105/135 observed-and-zero-row episodes never analyzed after acquisition
: the detection existed, the report said only a
count, and no row ever reached the model. The digest puts the handles in
the report itself. These pin the two seams: the digest's cap arithmetic
(a silent cap would read as "served everything") and the state stash's
replace-whole contract (a stale list would re-serve the LAST flight's
finds on a flight that found nothing).
"""

from __future__ import annotations

from benchmarks.citynav.run_benchmark.mission_engine import (
    _DETECTED_HEAD,
    detected_digest,
)


def _det(i: int) -> dict:
    return {"fid": f"detected_{i}", "class": "house", "xy": [i * 10.0, 5.0]}


def test_digest_serves_fid_class_xy_in_fold_order():
    d = detected_digest([_det(1), _det(2)])
    assert d == {
        "detected": [
            {"fid": "detected_1", "class": "house", "xy": [10.0, 5.0]},
            {"fid": "detected_2", "class": "house", "xy": [20.0, 5.0]},
        ]
    }
    assert "detected_more" not in d


def test_digest_caps_loud_never_silent():
    dets = [_det(i) for i in range(_DETECTED_HEAD + 5)]
    d = detected_digest(dets)
    assert len(d["detected"]) == _DETECTED_HEAD
    assert d["detected_more"] == 5


def test_digest_empty_flight_serves_nothing():
    assert detected_digest([]) == {}


def test_state_stash_replaces_whole_each_step():
    """A step folding nothing must leave an empty list, never the previous
    flight's finds. Exercised at the state seam the loop writes."""
    from maps.mission.tests.fakes import episode_world

    loop = episode_world()
    loop.state.last_flight_detections = [_det(1)]
    got = loop.last_flight_detections()
    assert got == [_det(1)]
    got[0]["fid"] = "mutated"  # accessor returns copies
    assert loop.state.last_flight_detections[0]["fid"] == "detected_1"
    loop.state.last_flight_detections = []
    assert loop.last_flight_detections() == []


def test_every_armed_class_reaches_the_head_however_the_fold_ordered_it():
    """One class must not spend the whole head: a confident car folded
    among many houses still reaches the report."""
    dets = [
        {"fid": f"h{i}", "class": "house", "xy": [float(i), 0.0], "conf": 0.90} for i in range(63)
    ]
    dets.append({"fid": "car_gt", "class": "car", "xy": [3311.5, 3770.6], "conf": 0.88})

    head = [d["fid"] for d in detected_digest(dets)["detected"]]
    assert "car_gt" in head, head
    # VACUITY FLOOR: fold order really would have buried it.
    assert "car_gt" not in [d["fid"] for d in dets[:12]]
    # The count is still the whole truth about what was left out.
    assert detected_digest(dets)["detected_more"] == len(dets) - len(head)


def test_within_a_class_the_head_takes_the_most_confident():
    """The head is the best of each class, not the first folded."""
    dets = [{"fid": "weak", "class": "car", "xy": [0.0, 0.0], "conf": 0.31}]
    dets += [
        {"fid": f"strong{i}", "class": "car", "xy": [1.0, float(i)], "conf": 0.94}
        for i in range(12)
    ]
    head = [d["fid"] for d in detected_digest(dets)["detected"]]
    assert head == [f"strong{i}" for i in range(12)], head


def test_no_confidences_leaves_the_fold_order_intact():
    """A run whose detections carry no score keeps the fold order."""
    dets = [{"fid": f"d{i}", "class": "house", "xy": [float(i), 0.0]} for i in range(20)]
    head = [d["fid"] for d in detected_digest(dets)["detected"]]
    assert head == [f"d{i}" for i in range(12)], head
