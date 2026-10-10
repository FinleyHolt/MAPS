""""look at them." delivers ONE CROP PER ROW over the set in scope —
end to end, real pixels, through the mission loop.

It used to deliver one wide frame with the rows badged in it, and that
frame could not resolve what these cues turn on: median 300 m window, 70%
of them 300 m or wider, the widest 37,243 m at 36.37 m/px, against 78.6 m
at 0.15 m/px for a single-row look. So the only form that named more than
one row could not tell them apart, and the only form that could resolve a
difference named exactly the row already picked.

Compile-only coverage does not hold this: the sentence has to reach
``_build_crop_requests`` with a populated ``mark_fids``, and that has to
reach ``render_candidate_crops``, before any of the renderer's badge
machinery runs. So the loop here is driven with the REAL renderer over a
synthetic ortho, and the assertions are on the delivered record and the
bytes the model would read.

The other half of the contract is that nothing was CHANGED: a one-id look
still hands the renderer exactly the request it handed before the set look
existed, and the renderer is a pure function of its request.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from maps.search.tests.ortho_fixture import write_ortho

from maps.search.parse_compile.reasoner_loop import (
    MissionSpec,
    advance_reasoner_one_round,
)
from maps.search.render.look_render import (
    LOOK_MAX_BADGES,
    render_candidate_crops,
    look_window_bounds_m,
)
from maps.search.tests.test_mission_loop import FakeReasoner


def _run(code: str):
    return ("run", {"code": code})

# The bounds are pixel counts; the fixture ortho below is 0.1 m/px.
_MIN_M, _BUDGET_M = look_window_bounds_m(0.1)

MAP = "unit_test_set_frame_block"
GROUND_Z = 10.0


@pytest.fixture(scope="module")
def _ortho_root(tmp_path_factory):
    """A 200 m x 200 m synthetic ortho at the native 0.1 m/px."""
    return write_ortho(tmp_path_factory.mktemp("ortho"), MAP, n=2000, ground_z=GROUND_Z, ramp=True)


@pytest.fixture
def ortho_dir(_ortho_root, monkeypatch):
    from maps.perception.ortho import camera_model
    from maps.perception.ortho.ground_level import GROUND_LEVEL

    monkeypatch.setitem(GROUND_LEVEL, MAP, GROUND_Z)
    camera_model._ORTHO_RASTER_CACHE.clear()
    root = _ortho_root
    return root


def _cars(n: int, step_m: float = 6.0) -> list[dict]:
    """``n`` cars on a square lattice centred near (100, 100)."""
    side = int(n ** 0.5) + 1
    out = []
    for i in range(n):
        out.append({"fid": f"detected_{i}",
                    "xy": [100.0 + step_m * (i % side - side / 2.0),
                           100.0 + step_m * (i // side - side / 2.0)],
                    "class": "car", "search_text": "car"})
    return out


def _fly(script, cands, ortho_dir):
    """Drive one mission with the REAL look renderer; return
    (tool results, the requests the renderer saw, the rendered records,
    the vision messages the model would read)."""
    seen_reqs: list[list[dict]] = []
    seen_recs: list[list[dict]] = []

    def renderer(reqs):
        seen_reqs.append(reqs)
        recs = render_candidate_crops(reqs, map_name=MAP, rgbd_dir=ortho_dir)
        seen_recs.append(recs)
        return recs

    fr = FakeReasoner(script)
    advance_reasoner_one_round(
        reasoner=fr,
        user_message="find the car",
        ctx=SimpleNamespace(graph=None, cell_size_m=1.0,
                            tile_origin_xy=(0.0, 0.0), grid_shape=(10, 10),
                            robot_pose=None),
        grounded_fids=set(),
        rid_tag="m0",
        transcript=[],
        compute_namespace={"cands": list(cands), "anchors": [],
                           "cell_m": 1.0},
        diag={},
        look_renderer=renderer,
        mission_spec=MissionSpec(fly=lambda i: {"error": "unused"},
                                 declare_concepts=lambda i: {"declared": []}),
    )
    results = [json.loads(m["content"]) for m in fr.messages
               if m.get("role") == "tool"]
    vision = [str(m.get("content")) for m in fr.messages
              if m.get("role") == "user" and "badge legend" in
              str(m.get("content"))]
    return results, seen_reqs, seen_recs, vision


def test_the_set_look_delivers_one_crop_per_row(ortho_dir):
    cands = _cars(6)
    results, reqs, recs, vision = _fly(
        [[_run('a = find("car")')], [_run("look(a)")], "over"],
        cands, ortho_dir)

    # ONE frame, not one crop per row: the set look is a relational read.
    assert len(reqs) == 1 and len(reqs[0]) >= 1, reqs
    req = reqs[0][0]
    assert req["center_fid"] == "detected_0", req      # rank 0 owns the frame
    # each request is its OWN subject: it marks itself and nothing else,
    # which is what makes badge->fid unambiguous by construction.
    assert [m["fid"] for m in req.get("marks") or []] == ["detected_0"], req
    assert [r["center_fid"] for r in reqs[0]] == [c["fid"] for c in cands]
    # one record PER ROW now; the first is rank 0's own crop
    rec = recs[0][0]

    # ONE badge per crop, and it is that crop's own subject
    legend = rec["legend"]
    # one entry per crop, and every entry names its own fid
    assert len(legend) == 1, legend
    assert legend[0]["fid"] in legend[0]["label"], legend
    assert legend[0]["role"] == "center" and legend[0]["fid"] == "detected_0"
    assert [e["digit"] for e in legend] == [1], legend
    # ...and the SET is covered across crops, one row each, in the handle's
    # own order -- the correspondence the frame form served in a legend is
    # now served by there being one picture per row.
    assert [r["legend"][0]["fid"] for r in recs[0]] == [
        c["fid"] for c in cands]
    assert rec["marks_dropped"] == 0 and rec["marks_out_of_view"] == [], rec

    # the WINDOW fits their hull, so it is wider than a lone appearance
    # crop and every row is in it
    # SINGLE-LOOK SCALE, not the set hull. This is the change: the set
    # frame was a median 300 m window (70% of them 300 m or wider,
    # the widest 37,243 m at 36.37 m/px) and could not resolve a
    # round pool from a rectangular one.
    assert rec["context_m"] == _MIN_M, rec["context_m"]
    # Each row sits inside ITS OWN window. The frame form needed every row
    # inside ONE window and clipped whatever did not fit; that constraint,
    # and the scale it forced, is the thing being removed.
    for c, r in zip(cands, recs[0]):
        x0, y0, x1, y1 = r["world_window"]
        assert x0 <= c["xy"][0] <= x1 and y0 <= c["xy"][1] <= y1, (c, r)

    # the bytes the model reads bind each badge to a fid, in every crop
    assert len(vision) == 1, vision
    for c in cands:
        assert c["fid"] in vision[0], (c["fid"], vision[0])
    # ... and say, ON the message the pixels are in, that these badges ARE
    # the rows of a handle it still holds: the frame and the result table
    # are one object, so the frame's contents stay nameable afterwards.
    # the handle rides the ref_notes / preamble now, not a frame count note
    handle = results[1]["crop"].get("handle") or "r0"
    assert handle.startswith("r"), results[1]
    # No spelling is invited: the old preamble said "of <ref>, ...",
    # which is sentence phrasing naming the EXECUTOR's ref, so a
    # code-surface model was told to say a handle it never bound.
    # What the caption owes is which row each picture shows.
    assert "of r" not in vision[0].split("Overhead")[0], vision[0]
    for c in cands:
        assert c["fid"] in vision[0], (c["fid"], vision[0])
    assert "in order" in vision[0], vision[0]
    # one crop PER ROW: the set is covered, not summarised
    assert results[1]["crop"]["delivered"] == len(cands), results[1]
    assert "error" not in results[1], results[1]


def test_over_budget_is_a_served_fact_not_a_silent_truncation(ortho_dir):
    # More rows than badges: the frame is the first LOOK_MAX_BADGES - 1 in
    # the standing order, the count is stated, and nothing renders as if
    # the rest were absent from the set.
    n = LOOK_MAX_BADGES + 11
    results, reqs, recs, vision = _fly(
        [[_run('a = find("car")')], [_run("look(a)")], "over"],
        _cars(n), ortho_dir)
    # the cap that bites is the per-call CROP budget, not the badge cap
    assert len(reqs[0]) == 16, len(reqs[0])
    # one record PER ROW now; the first is rank 0's own crop
    rec = recs[0][0]
    # center + (cap - 2) neighbours saturates the renderer's own ceiling
    # exactly, so the cap is reached and nothing is dropped inside it
    # ONE legend entry per crop: badge -> fid is unambiguous by
    # construction rather than by a legend that got it wrong on
    # 2,115 of 6,256 served entries.
    assert len(rec["legend"]) == 1, rec["legend"]
    assert rec["marks_dropped"] == 0, rec
    note = " ".join(results[1]["crop"].get("ref_notes") or [])
    assert f"{n} rows" in note, note
    assert "slice the handle" in note, note


def test_a_spread_set_gets_a_crop_each_at_single_look_scale(
        ortho_dir):
    """A set spread wider than one frame loses nothing now.

    The frame form clipped it: rows outside rank 0's neighbourhood got no
    badge, and the ack had to count what the renderer DREW and name what it
    could not. Per-row crops have no frame to fall outside of -- every row
    is its own subject, at its own single-look scale.
    """
    cands = [{"fid": f"detected_{i}", "xy": [12.5 + 400.0 * i, 12.5],
              "class": "car", "search_text": "car"} for i in range(4)]
    results, reqs, recs, _vis = _fly(
        [[_run('a = find("car")')], [_run("look(a)")], "over"],
        cands, ortho_dir)
    assert len(reqs[0]) == len(cands), reqs
    for rec in recs[0]:
        assert len(rec["legend"]) == 1, rec["legend"]
        assert rec["context_m"] == _MIN_M, rec["context_m"]
        assert not rec.get("marks_out_of_view"), rec
    crop = results[1]["crop"]
    assert crop["delivered"] == len(cands), crop

def test_the_one_id_look_is_the_request_it_always_was(ortho_dir):
    # The capability is ADDED, not swapped: a one-id look still emits the
    # single-item request with its own fid as the target and no mark set,
    # and render_candidate_crops is a pure function of that request — so
    # the delivered bytes are the ones it delivered before.
    results, reqs, recs, vision = _fly(
        [[_run('look("detected_1")')], "over"], _cars(6), ortho_dir)
    (req,) = reqs[0]
    assert req["center_fid"] == "detected_1"
    assert req["annotate"] is True
    # the target IS the center, so the ONE mark resolves onto the subject
    # and the frame keeps the tight appearance geometry
    assert [m["fid"] for m in req["marks"]] == ["detected_1"], req
    # one record PER ROW now; the first is rank 0's own crop
    rec = recs[0][0]
    assert rec["context_m"] == _MIN_M, rec["context_m"]
    assert rec["legend"] == [{"digit": 1, "fid": "detected_1",
                              "label": "TARGET car detected_1",
                              "role": "center"}], rec["legend"]
    assert results[0]["crop"]["delivered"] == 1, results[0]


def test_a_set_look_with_no_set_says_so_in_register(ortho_dir):
    """Looking at a name nothing bound renders nothing and says which name.

    The say channel answered this with a grammar rule id; Python answers it
    with the name itself, which is strictly more actionable -- the model can
    see WHICH of its names it never bound.
    """
    results, reqs, _recs, _vision = _fly(
        [[_run("look(a)")], "over"], _cars(3), ortho_dir)
    assert reqs == [], reqs           # nothing rendered
    err = str(results[0].get("error"))
    assert "NameError" in err and "'a'" in err, results[0]
