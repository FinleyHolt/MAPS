"""Reader/writer schema lock for ``tour.json``.

``tour_synth.write_tour_json`` (the writer) and
``tour_state.FeatureTour.from_tour_json`` (the flight-side reader) parse
the same stop-dict schema; these tests pin the round trip so a wire
change in one without the other fails here instead of mid-flight.
"""

from __future__ import annotations

import json

import pytest

from maps.search.parse_compile.tour_state import FeatureTour
from maps.search.parse_compile.tour_synth import TourStop, write_tour_json


def _write_workdir_tour(tmp_path, stops):
    tour_path = tmp_path / "compile" / "turn_0000" / "tour.json"
    write_tour_json(stops, tour_path, header={"turn": 0})
    return tour_path


def test_write_read_round_trip(tmp_path):
    stops = [
        TourStop(
            stop_id="stop_000",
            pose=(101.5, -42.25, 48.0, 90.0),
            observe_radius_m=15.0,
            aim_targets=[
                {
                    "fid": "building_3",
                    "x": 103.0,
                    "y": -40.0,
                    "z": 12.0,
                    "priority": 0.8,
                }
            ],
        ),
        TourStop(
            stop_id="stop_001",
            pose=(220.0, 15.0, 52.0, 270.0),
            observe_radius_m=15.0,
        ),
    ]
    _write_workdir_tour(tmp_path, stops)

    tour = FeatureTour.from_tour_json(tmp_path)

    assert [vp.fid for vp in tour.viewpoints] == ["stop_000", "stop_001"]
    vp0 = tour.viewpoints[0]
    assert vp0.xy == (101.5, -42.25)
    assert vp0.altitude_m == 48.0
    assert vp0.heading_deg == 90.0
    # An aim target is a place, not a camera pose: the observation is an
    # orthographic nadir crop, so there is nothing to aim.
    assert vp0.aim_targets == (
        {
            "fid": "building_3",
            "x": 103.0,
            "y": -40.0,
            "z": 12.0,
            "priority": 0.8,
        },
    )
    assert tour.viewpoints[1].aim_targets == ()
    assert tour.cursor == 0 and tour.current() is vp0


def test_legacy_anchors_key_hard_fails(tmp_path):
    tour_path = tmp_path / "compile" / "turn_0000" / "tour.json"
    tour_path.parent.mkdir(parents=True)
    tour_path.write_text(
        json.dumps(
            {
                "stops": [
                    {
                        "stop_id": "stop_000",
                        "pose": {"x": 1.0, "y": 2.0},
                        "anchors": [],
                    }
                ]
            }
        )
    )
    with pytest.raises(RuntimeError, match="anchors"):
        FeatureTour.from_tour_json(tmp_path)


def test_malformed_stop_is_skipped_not_fatal(tmp_path):
    # The flight-side reader is deliberately lenient (unlike the strict
    # resume parser in tour_synth): a stop with an unreadable pose is
    # dropped and the rest of the tour still flies.
    tour_path = tmp_path / "compile" / "turn_0000" / "tour.json"
    tour_path.parent.mkdir(parents=True)
    tour_path.write_text(
        json.dumps(
            {
                "stops": [
                    {"stop_id": "stop_000", "pose": {"x": "nan?", "y": None}},
                    {"stop_id": "stop_001", "pose": {"x": 5.0, "y": 6.0}},
                ]
            }
        )
    )
    tour = FeatureTour.from_tour_json(tmp_path)
    assert [vp.fid for vp in tour.viewpoints] == ["stop_001"]
