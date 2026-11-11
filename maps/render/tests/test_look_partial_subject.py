"""A subject bigger than its frame says so, in metres."""

from __future__ import annotations

from maps.render.look_render import render_candidate_crops

MAP = "unit_test_partial_block"
GROUND_Z = 10.0


ORTHO_N = 2400
ORTHO_RAMP = False


def _crop(ortho_dir, boundary):
    xs = [p[0] for p in boundary]
    ys = [p[1] for p in boundary]
    req = {"center_fid": "f0", "center_xy": [(min(xs) + max(xs)) / 2,
                                             (min(ys) + max(ys)) / 2],
           "center_label": "f0", "center_boundary": boundary,
           "marks": []}
    recs = render_candidate_crops([req], map_name=MAP, rgbd_dir=ortho_dir)
    assert len(recs) == 1, recs
    return recs[0]


def test_a_subject_that_overruns_its_frame_reports_its_true_size(ortho_dir):
    # 900 x 1200 m against a 100 m budget ceiling on this 0.1 m/px ortho.
    big = [[-400.0, -550.0], [500.0, -550.0], [500.0, 650.0], [-400.0, 650.0]]
    rec = _crop(ortho_dir, big)
    assert rec["subject_outline_cut"] is True, rec
    assert rec["subject_extent_m"] == [900.0, 1200.0], rec["subject_extent_m"]
    # ...and it is still IN the window by the other two window facts.
    assert rec["subject_in_window"] is True, rec
    assert rec["subject_centroid_in_window"] is True, rec


def test_a_subject_inside_its_frame_is_not_flagged(ortho_dir):
    # VACUITY FLOOR: without this, a renderer that flags everything passes.
    small = [[195.0, 195.0], [205.0, 195.0], [205.0, 205.0], [195.0, 205.0]]
    rec = _crop(ortho_dir, small)
    assert rec["subject_outline_cut"] is False, rec
    assert rec["subject_extent_m"] == [10.0, 10.0], rec["subject_extent_m"]


def test_the_caption_states_the_extent_and_calls_the_ink_a_fragment():
    from maps.mission.picture_requests import look_vision_text

    text = look_vision_text([{
        "center_fid": "f0", "center_label": "f0",
        "context_m": 100.0, "side_px": 1000,
        "world_window": [0.0, 0.0, 100.0, 100.0],
        "subject_in_window": True, "subject_centroid_in_window": True,
        "subject_on_data": True, "window_valid_frac": 1.0,
        "subject_extent_m": [900.0, 1200.0], "subject_outline_cut": True,
        "legend": [{"digit": 1, "label": "f0", "role": "center"}],
    }])
    assert "900 x 1200 m" in text, text
    assert "fragment" in text, text
