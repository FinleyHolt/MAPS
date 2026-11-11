"""A detection is its own row, stamped with its step, and no images are written.

A step that writes a PNG has re-grown a per-step image bundle; anything
beyond ``detections.jsonl`` is a per-episode artifact nothing reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks.citynav.episode_workdir import LiveRecorder
from maps.perception.mask_codec import encode_mask_record


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_detections_file_exists_before_the_first_step(tmp_path: Path) -> None:
    """A mission that detects nothing has zero rows; an empty file says so
    where an absent one reads as a recorder that never ran."""
    LiveRecorder(tmp_path)
    assert (tmp_path / "detections.jsonl").read_bytes() == b""


def test_recorder_writes_no_images(tmp_path: Path) -> None:
    rec = LiveRecorder(tmp_path)
    rec.write_step(step=0, detections=[])
    rec.write_step(step=1, detections=[{"label": "car", "confidence": 0.5,
                                        "world_xy": [1.0, 2.0]}])

    assert sorted(p.name for p in tmp_path.iterdir()) == ["detections.jsonl"]
    assert [d["step"] for d in _rows(tmp_path / "detections.jsonl")] == [1]


def test_masks_ride_in_the_detection_row(tmp_path: Path) -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 30:36] = True
    det = {"label": "car", "confidence": 0.9, "world_xy": [3.0, 4.0],
           **encode_mask_record(mask)}

    LiveRecorder(tmp_path).write_step(step=0, detections=[det])

    (recorded,) = _rows(tmp_path / "detections.jsonl")
    assert recorded["step"] == 0
    assert recorded["label"] == "car"
    assert recorded["world_xy"] == [3.0, 4.0]
    assert recorded["pixel_bbox_rcrc"] == [10, 30, 19, 35]
    assert recorded["mask_shape"] == [10, 6]
    assert isinstance(recorded["mask_zpacked_b64"], str)
