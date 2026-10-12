""""look at them." delivers ONE CROP PER ROW over the set in scope — end to end, real
pixels, through the mission loop."""

from __future__ import annotations

from maps.search.render.look_render import (
    LOOK_MAX_BADGES,
    look_window_bounds_m,
)
from maps.search.tests.fakes import cars, look_drive, run

# The bounds are pixel counts; the fixture ortho below is 0.1 m/px.
_MIN_M, _BUDGET_M = look_window_bounds_m(0.1)

def test_the_set_look_delivers_one_crop_per_row(ortho_dir):
    cands = cars(6)
    results, reqs, recs, vision = look_drive(
        [[run('a = find("car")')], [run("look(a)")], "over"],
        cands, ortho_dir)

    # ONE frame, not one crop per row: the set look is a relational read.
    assert len(reqs) == 1 and len(reqs[0]) >= 1, reqs
    req = reqs[0][0]
    assert req["center_fid"] == "detected_0", req      # rank 0 owns the frame
    # each request is its OWN subject: it marks itself and nothing else,
    # which is what makes badge->fid unambiguous by construction.
    assert [m["fid"] for m in req.get("marks") or []] == ["detected_0"], req
    assert [r["center_fid"] for r in reqs[0]] == [c["fid"] for c in cands]
    rec = recs[0][0]

    # ONE badge per crop, and it is that crop's own subject
    legend = rec["legend"]
    # one entry per crop, and every entry names its own fid
    assert len(legend) == 1, legend
    assert legend[0]["fid"] in legend[0]["label"], legend
    assert legend[0]["role"] == "center" and legend[0]["fid"] == "detected_0"
    assert [e["digit"] for e in legend] == [1], legend
    assert [r["legend"][0]["fid"] for r in recs[0]] == [
        c["fid"] for c in cands]
    assert rec["marks_dropped"] == 0 and rec["marks_out_of_view"] == [], rec

    # Each crop is at single-look scale, not the set's hull.
    assert rec["context_m"] == _MIN_M, rec["context_m"]
    # Each row sits inside ITS OWN window.
    for c, r in zip(cands, recs[0]):
        x0, y0, x1, y1 = r["world_window"]
        assert x0 <= c["xy"][0] <= x1 and y0 <= c["xy"][1] <= y1, (c, r)

    # the bytes the model reads bind each badge to a fid, in every crop
    assert len(vision) == 1, vision
    for c in cands:
        assert c["fid"] in vision[0], (c["fid"], vision[0])
    handle = results[1]["crop"].get("handle") or "r0"
    assert handle.startswith("r"), results[1]
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
    results, reqs, recs, vision = look_drive(
        [[run('a = find("car")')], [run("look(a)")], "over"],
        cars(n), ortho_dir)
    # the cap that bites is the per-call CROP budget, not the badge cap
    assert len(reqs[0]) == 16, len(reqs[0])
    rec = recs[0][0]
    assert len(rec["legend"]) == 1, rec["legend"]
    assert rec["marks_dropped"] == 0, rec
    note = " ".join(results[1]["crop"].get("ref_notes") or [])
    assert f"only the first 16 crops of this call rendered ({n - 16} item(s) dropped)" in note, note
    assert note.count("16") == 2, note       # the cap is stated once


def test_a_spread_set_gets_a_crop_each_at_single_look_scale(
        ortho_dir):
    cands = [{"fid": f"detected_{i}", "xy": [12.5 + 400.0 * i, 12.5],
              "class": "car", "search_text": "car"} for i in range(4)]
    results, reqs, recs, _vis = look_drive(
        [[run('a = find("car")')], [run("look(a)")], "over"],
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
    results, reqs, recs, vision = look_drive(
        [[run('look("detected_1")')], "over"], cars(6), ortho_dir)
    (req,) = reqs[0]
    assert req["center_fid"] == "detected_1"
    assert req["annotate"] is True
    # the target IS the center, so the ONE mark resolves onto the subject
    # and the frame keeps the tight appearance geometry
    assert [m["fid"] for m in req["marks"]] == ["detected_1"], req
    rec = recs[0][0]
    assert rec["context_m"] == _MIN_M, rec["context_m"]
    assert rec["legend"] == [{"digit": 1, "fid": "detected_1",
                              "label": "TARGET car detected_1",
                              "role": "center"}], rec["legend"]
    assert results[0]["crop"]["delivered"] == 1, results[0]


def test_a_set_look_with_no_set_says_so_in_register(ortho_dir):
    """Looking at a name nothing bound renders nothing and says which name."""
    results, reqs, _recs, _vision = look_drive(
        [[run("look(a)")], "over"], cars(3), ortho_dir)
    assert reqs == [], reqs           # nothing rendered
    err = str(results[0].get("error"))
    assert "NameError" in err and "'a'" in err, results[0]
