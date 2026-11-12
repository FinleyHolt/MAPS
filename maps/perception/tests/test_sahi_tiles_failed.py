"""A SAHI tile the segmenter could not answer is counted on the client.

A frame with some failed tiles keeps the detections of the tiles that did
answer and is partial-blind, not empty; ``sahi_tiles_failed`` accumulates
over the client's life (one episode), and the runner stamps it on the row as
``metrics.sam_tiles_failed``.
"""

from __future__ import annotations

from maps.perception.sam.detect_client import SamDetectClient
from maps.perception.tests.test_sahi_nms import _build_tile_case


def test_a_partially_failed_frame_counts_its_failed_tiles(monkeypatch):
    client, obs, payload_by_b64, _ref = _build_tile_case()
    answered = list(payload_by_b64)[:1]  # the second tile gets no usable response

    monkeypatch.setattr(
        "maps.perception.sam.detect_client._post_segment",
        lambda json_body, endpoint=None: (
            payload_by_b64.get(json_body["image_b64"])
            if json_body["image_b64"] in answered
            else None
        ),
    )
    captured = []
    monkeypatch.setattr(
        SamDetectClient,
        "detections",
        lambda self, obs, masks: captured.append(masks) or [],
    )
    assert client.sahi_tiles_failed == 0
    client.segment(obs, ["car"])
    assert client.sahi_tiles_failed == 1
    assert captured and captured[0], "the answering tile's detections are kept"
    client.segment(obs, ["car"])
    assert client.sahi_tiles_failed == 2
