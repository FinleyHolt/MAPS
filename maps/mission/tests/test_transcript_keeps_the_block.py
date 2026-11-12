"""The transcript's run row keeps the whole block and the stops it flew.

E2's analysis read flights from ``said``, then cut at 400 characters: a go_to
past the cut, or one before a later line errored, left no trace of the flight.
"""

from __future__ import annotations

from maps.mission.loop import _Mission
from maps.mission.spec import MissionSpec
from maps.mission.tests.fakes import FakeReasoner, bare_ctx, unarmed_boost


def test_a_long_block_that_flies_then_errors_keeps_its_code_and_flight():
    code = "# " + "x" * 600 + '\ngo_to("prior_0")\nundefined_name\n'
    tr: list[dict] = []
    _Mission(
        boost_queue=unarmed_boost,
        reasoner=FakeReasoner([[("run", {"code": code})], "over"]),
        ctx=bare_ctx(),
        rid_tag="m0",
        transcript=tr,
        compute_namespace={
            "cands": [],
            "anchors": [
                {"fid": "prior_0", "xy": [1.0, 2.0], "name": "shop", "search_text": "shop"}
            ],
            "cell_m": 1.0,
        },
        diag={},
        look_renderer=lambda reqs: [],
        mission_spec=MissionSpec(
            fly=lambda item: {
                "stop_id": "stop_000",
                "target": item["target_fid"],
                "pose_xy": [0.0, 0.0],
                "new_detections": 0,
                "cue": "cue",
            },
            declare_concepts=lambda items: None,
        ),
    ).converse("i")
    runs = [t for t in tr if t.get("kind") == "run"]
    assert len(runs) == 1, tr
    assert runs[0]["said"] == code
    assert runs[0]["errored"] is True, runs[0]
    assert runs[0]["flights"] == 1, runs[0]
