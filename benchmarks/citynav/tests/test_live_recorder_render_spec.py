"""A step records a render spec, not pixels, and a detection is its own row.

The audit rebuilds every frame from ``pose_xy`` + ``altitude_m`` +
``heading_deg`` + ``frame_px`` over the one hole-free ortho, and places the
detection masks in that ``frame_px`` pixel space. These tests pin that the
recorder writes the whole spec as one ``steps.jsonl`` line, every detection as
one ``detections.jsonl`` line stamped with its step, and no images: a step that
drops ``frame_px`` renders nothing, and a step that writes a PNG has re-grown
the artifact this layout exists to delete.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks.citynav.live_recorder import LiveRecorder
from maps.util.mask_codec import encode_mask_record


def _bootstrap(tmp_path: Path) -> LiveRecorder:
    return LiveRecorder(tmp_path, episode_id="ep0")


def _kwargs(step: int, **over) -> dict:
    base = dict(
        step=step,
        pose_xy=(12.5, -3.25),
        altitude_m=122.0,
        heading_deg=90.0,
        frame_px=2016,
        visited_xy=[],
        fov_corners_xy=None,
        detections=[],
        wall_t_monotonic=0.0,
    )
    base.update(over)
    return base


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_step_carries_the_whole_render_spec_and_no_detections(tmp_path: Path) -> None:
    rec = _bootstrap(tmp_path)
    rec.write_step(**_kwargs(0))

    (payload,) = _rows(tmp_path / "steps.jsonl")
    assert payload["pose_xy"] == [12.5, -3.25]
    assert payload["altitude_m"] == 122.0
    assert payload["heading_deg"] == 90.0
    assert payload["frame_px"] == 2016
    assert payload["fov_corners_xy"] is None
    assert "detections" not in payload


def test_both_jsonl_files_exist_before_the_first_step(tmp_path: Path) -> None:
    """A mission that flies nothing has zero steps; an empty file says so
    where an absent one reads as a recorder that never ran."""
    _bootstrap(tmp_path)
    assert (tmp_path / "steps.jsonl").read_bytes() == b""
    assert (tmp_path / "detections.jsonl").read_bytes() == b""


def test_recorder_writes_no_images(tmp_path: Path) -> None:
    """The whole point: a step is text. A stray PNG or JPG means a per-step
    image bundle has come back; anything beyond the two JSONLs means a
    per-episode artifact has re-grown."""
    rec = _bootstrap(tmp_path)
    rec.write_step(**_kwargs(0))
    rec.write_step(**_kwargs(1, detections=[{"label": "car", "confidence": 0.5,
                                             "world_xy": [1.0, 2.0]}]))

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "detections.jsonl", "steps.jsonl",
    ]
    assert not list(tmp_path.rglob("*.png"))
    assert not list(tmp_path.rglob("*.jpg"))
    assert [s["step"] for s in _rows(tmp_path / "steps.jsonl")] == [0, 1]


def test_masks_ride_in_the_detection_row_in_frame_px_space(tmp_path: Path) -> None:
    """A mask's bbox is measured in the ``frame_px`` frame, so the detection
    row carries its step and the step row carries ``frame_px``, or the audit
    cannot place it."""
    frame_px = 64
    mask = np.zeros((frame_px, frame_px), dtype=bool)
    mask[10:20, 30:36] = True
    det = {"label": "car", "confidence": 0.9, "world_xy": [3.0, 4.0],
           **encode_mask_record(mask)}

    rec = _bootstrap(tmp_path)
    rec.write_step(**_kwargs(0, frame_px=frame_px, detections=[det]))

    (payload,) = _rows(tmp_path / "steps.jsonl")
    assert payload["frame_px"] == frame_px
    (recorded,) = _rows(tmp_path / "detections.jsonl")
    assert recorded["step"] == 0
    assert recorded["label"] == "car"
    assert recorded["confidence"] == 0.9
    assert recorded["world_xy"] == [3.0, 4.0]
    assert recorded["pixel_bbox_rcrc"] == [10, 30, 19, 35]
    assert recorded["mask_shape"] == [10, 6]
    assert isinstance(recorded["mask_zpacked_b64"], str)
    r0, c0, r1, c1 = recorded["pixel_bbox_rcrc"]
    assert 0 <= r0 <= r1 < frame_px
    assert 0 <= c0 <= c1 < frame_px


def test_a_step_without_frame_px_records_none(tmp_path: Path) -> None:
    """An observation-less step still writes; the audit reads the missing
    ``frame_px`` as "no renderable frame" rather than guessing one."""
    rec = _bootstrap(tmp_path)
    rec.write_step(**_kwargs(0, frame_px=None))
    (payload,) = _rows(tmp_path / "steps.jsonl")
    assert payload["frame_px"] is None
