"""``SamDetectClient`` over a loopback ``/segment`` serve: one stateless POST per tile, wire detections
mapped into ``DetectedObject`` metadata (``sam3_1_object_id``, the mask's index after the cross-tile
dedup, ``sam3_1_score``, position-derived ``track_id``)."""

from __future__ import annotations

import numpy as np
import pytest

from maps.perception.mask_codec import encode_mask_record
from maps.perception.observation import Observation
from maps.perception.sam.detect_client import _SAM_INPUT_SIZE, SamDetectClient
from tests.serves import SegmentServe


def _build_segment_payload(
    *, masks: list[np.ndarray], scores: list[float], concepts: list[str]
) -> dict:
    """A ``POST /segment`` answer built with the server's own mask encoder."""
    return {
        "detections": [
            {"label": label, "score": float(score), **encode_mask_record(mask)}
            for mask, score, label in zip(masks, scores, concepts)
        ]
    }


def _build_observation(rgb: np.ndarray) -> Observation:
    # constant depth: every mask pixel projects
    depth = np.full(rgb.shape[:2], 50.0, dtype=np.float32)
    return Observation(
        pose_xy=(0.0, 0.0),
        altitude_m=50.0,
        world_window=(-60.0, -60.0, 60.0, 60.0),
        rgb=rgb,
        depth=depth,
        ortho_view_corners_xy=((-60.0, 60.0), (60.0, 60.0), (60.0, -60.0), (-60.0, -60.0)),
    )


def _square_frame(fill: int = 200) -> np.ndarray:
    """A ``_SAM_INPUT_SIZE`` square frame: one tile, so masks are authored in wire pixels."""
    return np.full((_SAM_INPUT_SIZE, _SAM_INPUT_SIZE, 3), fill, dtype=np.uint8)


def _block_mask(r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    mask = np.zeros((_SAM_INPUT_SIZE, _SAM_INPUT_SIZE), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


@pytest.fixture
def serve():
    with SegmentServe() as fake:
        yield fake


@pytest.fixture
def adapter(monkeypatch, serve):
    """A client whose run names ``serve`` as its one segmenter replica (SAM_SERVER_URL)."""
    from maps.perception.sam import detect_client as dc
    from maps.settings import from_environ

    monkeypatch.setattr(dc, "MIN_AREA_PX", 1)
    return SamDetectClient(
        concepts=["building"], endpoint=from_environ({"SAM_SERVER_URL": serve.url}).endpoint
    )


def test_detect_posts_segment_and_maps_wire_to_metadata(adapter, serve):
    """Each ``detect`` is one POST with the full concept list and no session; the track id is
    position-derived, so it is stable across calls."""
    obs = _build_observation(_square_frame())
    mask = _block_mask(404, 604, 404, 604)
    serve.answers = [
        (200, _build_segment_payload(masks=[mask], scores=[score], concepts=["building"]), {})
        for score in (0.9, 0.88)
    ]
    dets_first = adapter.detect(obs)
    dets_second = adapter.detect(obs)

    assert len(dets_first) == 1
    assert len(dets_second) == 1
    meta = dets_first[0].metadata
    assert dets_first[0].label == "building"
    assert meta["sam3_1_object_id"] == 0
    assert meta["sam3_1_score"] == pytest.approx(0.9)
    assert meta["track_id"] == "sam3_1_img_0"
    assert set(meta) == {"sam3_1_object_id", "sam3_1_score", "track_id"}
    assert dets_second[0].metadata["track_id"] == "sam3_1_img_0"
    assert len(serve.bodies) == 2
    for body in serve.bodies:
        assert body["concepts"] == ["building"]
        assert set(body) == {"concepts", "min_area_px", "max_per_image", "image_b64"}


def test_each_kept_mask_gets_its_own_ids(adapter, serve):
    obs = _build_observation(_square_frame())
    payload = _build_segment_payload(
        masks=[_block_mask(100, 250, 100, 250), _block_mask(700, 850, 700, 850)],
        scores=[0.6, 0.6],
        concepts=["building", "building"],
    )
    serve.answers = [(200, payload, {})]
    dets = adapter.detect(obs)

    assert len(dets) == 2
    assert sorted(d.metadata["track_id"] for d in dets) == ["sam3_1_img_0", "sam3_1_img_1"]
    assert sorted(d.metadata["sam3_1_object_id"] for d in dets) == [0, 1]


def test_a_frame_under_a_tile_is_padded_not_resampled(adapter, serve):
    """A frame smaller than the encoder input is posted 1:1 at the tile's top-left over zeros, and a
    mask reaching into the padding is clipped to the frame."""
    import base64
    import io

    from PIL import Image

    rgb = np.random.default_rng(0).integers(0, 256, (600, 800, 3), dtype=np.uint8)
    serve.answers = [
        (
            200,
            _build_segment_payload(
                masks=[_block_mask(500, 700, 700, 900)], scores=[0.9], concepts=["building"]
            ),
            {},
        )
    ]
    dets = adapter.detect(_build_observation(rgb))

    posted = np.asarray(
        Image.open(io.BytesIO(base64.b64decode(serve.bodies[0]["image_b64"]))).convert("RGB")
    )
    assert posted.shape == (_SAM_INPUT_SIZE, _SAM_INPUT_SIZE, 3)
    assert np.array_equal(posted[:600, :800], rgb)
    assert not posted[600:].any() and not posted[:, 800:].any()
    assert [d.mask.bbox for d in dets] == [(500, 700, 599, 799)]
    assert dets[0].mask.frame_shape == (600, 800)


def test_returns_empty_when_rgb_missing(adapter, serve):
    obs = _build_observation(np.full((16, 16, 3), 128, dtype=np.uint8))
    obs.rgb = None  # type: ignore[assignment]
    assert adapter.detect(obs) == []
    assert serve.bodies == []


def test_requires_ortho_corners(adapter):
    obs = _build_observation(np.full((16, 16, 3), 128, dtype=np.uint8))
    obs.ortho_view_corners_xy = None
    with pytest.raises(ValueError):
        adapter.detect(obs)


def test_pick_url_is_round_robin():
    """Nine tiles over three replicas land 3/3/3, never twice in a row on one replica."""
    from maps.perception.sam import detect_client as dc

    eps = ["http://a:1", "http://b:2", "http://c:3"]
    picks = [dc._pick_url(eps) for _ in range(9)]

    assert sorted(picks) == sorted(eps * 3)
    # consecutive picks never repeat a replica -> a burst spreads immediately
    assert all(a != b for a, b in zip(picks, picks[1:]))


@pytest.mark.parametrize("n_masks,cap_hit", [(2, False), (3, True)])
def test_cap_hit_says_whether_detections_were_dropped(
    adapter, monkeypatch, capsys, n_masks, cap_hit
):
    """A frame filling the cap exactly drops nothing, so ``cap_hit`` is False; one past it is True."""
    from maps.perception.mask_region import MaskRegion
    from maps.perception.sam import detect_client as dc

    monkeypatch.setattr(dc, "MAX_PER_STEP", 2)
    obs = _build_observation(_square_frame())
    blocks = [
        MaskRegion(np.ones((10, 10), dtype=bool), (100 * i, 0, 100 * i + 9, 9), obs.rgb.shape[:2])
        for i in range(n_masks)
    ]
    dets = adapter.detections(obs, [(b, "building", i, 0.9) for i, b in enumerate(blocks)])
    assert len(dets) == 2
    assert ("cap_hit=True" in capsys.readouterr().out) is cap_hit
