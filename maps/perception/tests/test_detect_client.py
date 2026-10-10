"""HTTP-contract tests for ``SamDetectClient``.

The adapter is **stateless image mode**: ``detect`` letterboxes the
frame to ``_SAM_INPUT_SIZE`` and POSTs it to the unified SAM 3.1 server's
``POST /segment`` endpoint via ``requests`` (the pool-avoiding
``_post_segment_avoid_stale_pool``). There is NO server-side session, NO
``/sessions`` route, and NO ``/detect`` route — cross-frame object
identity comes from the world-space tracker, not a SAM masklet session.

The server's ``/segment`` response is ``{"detections": [...],
"image_shape_hw": [H, W]}`` where each detection is::

    {"label", "score", "object_id",
     "pixel_bbox_rcrc": [r0, c0, r1, c1],
     "mask_shape": [rows, cols],
     "mask_zpacked_b64": zlib(np.packbits(crop.ravel()))}

The adapter unpacks the mask, projects it through ``obs.depth``, and packs
a ``DetectedObject``. Field mapping into ``metadata``:

* ``sam3_1_object_id`` <- the wire ``object_id``;
* ``sam3_1_score`` <- the wire ``score``;
* ``track_id`` <- ``sam3_1_img_<det_idx>`` (position-derived,
  deterministic).

These tests mock ``requests.Session.post`` so the real wire path runs with
no live server, no GPU and no torch weights.

"""

from __future__ import annotations

import base64
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider
from maps.perception.observation import Observation
from maps.perception.sam.detect_client import _SAM_INPUT_SIZE, SamDetectClient
from maps.util.mask_codec import encode_mask_record


# ── Source guards: the server + engine run the real SAM 3.1 runtime ──


def test_server_and_image_engine_use_sam31_runtime_not_name_only() -> None:
    """Discriminate a real SAM 3.1 runtime from a name-only stub.

    The server is a THIN wrapper that delegates every forward to the
    engine (``serving.sam_engine``) — it holds no model of
    its own and must NOT smuggle in a ``transformers`` HF stub or a phantom
    ``Sam3Video*`` class. The engine loads the real FAIR SAM 3.1 checkpoint
    via ``sam3.model_builder`` — the strings below fail loudly if either the
    delegation or the real weight loader is swapped for a stub.
    """
    root = Path(__file__).resolve().parents[3]
    server_source = (root / "serving/sam_server.py").read_text()
    engine_source = (root / "serving/sam_engine.py").read_text()

    # Server delegates to the real engine; no transformers/name-only stub.
    assert "from serving.sam_engine import segment" in server_source
    assert "model(batch)" in engine_source
    assert "from transformers import" not in server_source
    assert "import transformers" not in server_source
    assert "Sam3VideoModel" not in server_source
    assert "Sam3VideoProcessor" not in server_source

    # Engine loads the real FAIR SAM 3.1 checkpoint (not a name-only alias).
    assert "from sam3.model_builder import" in engine_source
    assert "build_sam3_image_model" in engine_source
    assert 'download_ckpt_from_hf(version="sam3.1")' in engine_source


# ── Fixtures / helpers for the adapter HTTP path ──


def _build_segment_payload(
    *,
    object_ids: list[int],
    masks: list[np.ndarray],
    scores: list[float],
    concepts: list[str],
    image_shape: tuple[int, int],
) -> dict:
    """Build a faithful ``POST /segment`` response payload.

    Mirrors the server's ``_detections_to_wire`` output: each mask is
    bbox-cropped + bit-packed, coords absolute in the wire frame. The
    ``/segment`` response carries no ``session_id`` and no ``frame_index``.
    """
    detections = []
    for oid, mask, score, label in zip(object_ids, masks, scores, concepts):
        # Build the payload with the real encoder so the fixture cannot drift
        # from the server.
        detections.append(
            {
                "object_id": int(oid),
                "label": label,
                "score": float(score),
                **encode_mask_record(mask),
            }
        )
    return {
        "image_shape_hw": list(image_shape),
        "detections": detections,
    }


def _build_observation(rgb: np.ndarray) -> Observation:
    # The adapter drops detections without depth. Constant 50 m gives every
    # mask pixel a valid projection.
    depth = np.full(rgb.shape[:2], 50.0, dtype=np.float32)
    return Observation(
        pose_xy=(0.0, 0.0),
        altitude_m=50.0,
        yaw_deg=0.0,
        semantic=np.zeros(rgb.shape[:2], dtype=np.int32),
        height=None,
        fov_m=120.0,
        world_window=(-60.0, -60.0, 60.0, 60.0),
        rgb=rgb,
        depth=depth,
        ortho_view_corners_xy=((-60.0, 60.0), (60.0, 60.0), (60.0, -60.0), (-60.0, -60.0)),
    )


def _square_frame(fill: int = 200) -> np.ndarray:
    """A ``_SAM_INPUT_SIZE`` square RGB frame.

    The adapter letterboxes to ``_SAM_INPUT_SIZE`` before POSTing; a native
    ``_SAM_INPUT_SIZE`` square is the identity-letterbox special case (scale
    1.0, zero pad), so the mocked server's wire frame == the frame the
    adapter projects, and masks can be authored directly in wire coords.
    """
    return np.full((_SAM_INPUT_SIZE, _SAM_INPUT_SIZE, 3), fill, dtype=np.uint8)


def _block_mask(r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    mask = np.zeros((_SAM_INPUT_SIZE, _SAM_INPUT_SIZE), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


class _FakeResp:
    """Minimal stand-in for a ``requests`` Response on the ``/segment`` path."""

    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


@contextmanager
def _mock_requests_post(monkeypatch, response_queue: list[dict]):
    """Patch ``requests.Session.post`` so each call pops one queued payload.

    Records ``{"url", "body"}`` per call. The pool-avoiding session setup
    (mount / HTTPAdapter / Retry) still runs but never opens a socket, so
    the real wire path is exercised hermetically. An exhausted queue
    returns a clean empty-detection 200 (not ``None``), matching a genuinely
    empty frame rather than an outage.
    """
    import requests

    calls: list[dict] = []

    def fake_post(self, url, *args, **kwargs):
        body = kwargs.get("json")
        calls.append({"url": url, "body": body})
        if response_queue:
            return _FakeResp(response_queue.pop(0))
        return _FakeResp(
            {"detections": [], "image_shape_hw": [_SAM_INPUT_SIZE, _SAM_INPUT_SIZE]}
        )

    monkeypatch.setattr(requests.Session, "post", fake_post, raising=True)
    yield calls


@contextmanager
def _mock_urlopen(monkeypatch, response_queue: list[dict]):
    """Patch urllib.request.urlopen so each call pops one queued payload."""
    calls: list[dict] = []

    class _FakeUrlResp:
        def __init__(self, payload: dict):
            self._payload = payload

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        del timeout
        body_bytes = req.data or b"{}"
        body = json.loads(body_bytes.decode("utf-8"))
        calls.append({"url": req.full_url, "body": body})
        if not response_queue:
            return _FakeUrlResp({"detections": [], "image_shape_hw": [0, 0]})
        return _FakeUrlResp(response_queue.pop(0))

    import urllib.request as _ur

    monkeypatch.setattr(_ur, "urlopen", fake_urlopen)
    yield calls


@pytest.fixture
def adapter():
    provider = ReasonerUnionConceptProvider()
    provider.set_concepts(["building"])
    return SamDetectClient(
        concept_provider=provider,
        score_threshold=0.1,
        min_area_px=1,
    )


# ── Single-camera image path (detect) ──


def test_the_clients_read_outwaits_the_servers_in_flight_hold() -> None:
    """The server holds a /segment request behind its in-flight semaphore
    for ``_INFLIGHT_WAIT_S`` before it answers 503; a client that gives up
    first abandons a request the server is still going to serve and posts
    the tile again on top of it. The server's number is read off its
    source, since importing it loads torch."""
    import re

    from maps.perception.sam import detect_client as dc

    root = Path(__file__).resolve().parents[3]
    server_source = (root / "serving/sam_server.py").read_text()
    held = re.search(r'os\.environ\.get\("SAM_INFLIGHT_WAIT_S", "([0-9.]+)"\)', server_source)
    assert held is not None, "sam_server.py no longer declares SAM_INFLIGHT_WAIT_S with a default"
    assert dc.SEGMENT_READ_TIMEOUT_S > float(held.group(1)) >= 600.0
    # The connect budget outlasts a radio dropout.
    assert dc.SEGMENT_CONNECT_TIMEOUT_S > 20.0


def test_detect_posts_segment_and_maps_wire_to_metadata(adapter, monkeypatch):
    """``detect`` POSTs exactly one ``/segment`` (no session), ships
    the full concept list, and maps the wire detection into metadata:
    ``sam3_1_object_id`` <- wire object_id, ``sam3_1_score`` <- wire score,
    ``track_id`` = deterministic ``sam3_1_img_<det_idx>``.

    Called twice on the same frame the id is stable (position-derived) and
    both calls hit ``/segment`` — the adapter keeps NO session, so identity
    across frames is the world tracker's job, not the wire.
    """
    obs = _build_observation(_square_frame())
    mask = _block_mask(404, 604, 404, 604)  # centered 200x200 block

    payloads = [
        _build_segment_payload(
            object_ids=[7],
            masks=[mask],
            scores=[0.9],
            concepts=["building"],
            image_shape=(_SAM_INPUT_SIZE, _SAM_INPUT_SIZE),
        ),
        _build_segment_payload(
            object_ids=[7],
            masks=[mask],
            scores=[0.88],
            concepts=["building"],
            image_shape=(_SAM_INPUT_SIZE, _SAM_INPUT_SIZE),
        ),
    ]
    with _mock_requests_post(monkeypatch, payloads) as calls:
        dets_first = adapter.detect(obs, (0.0, 0.0))
        dets_second = adapter.detect(obs, (0.0, 0.0))

    assert len(dets_first) == 1
    assert len(dets_second) == 1

    meta = dets_first[0].metadata
    assert dets_first[0].label == "building"
    assert meta["sam3_1_object_id"] == 7
    assert meta["sam3_1_score"] == pytest.approx(0.9)
    assert meta["track_id"] == "sam3_1_img_0"
    # Stable across frames (same index), and it is the IMAGE namespace — never
    # the raw ``sam3_1_<object_id>`` a session protocol would have emitted.
    assert dets_second[0].metadata["track_id"] == "sam3_1_img_0"

    # Exactly one POST per detect call, all to /segment, full concept list,
    # a base64 frame, and NO session_id anywhere.
    assert len(calls) == 2
    for call in calls:
        assert call["url"].endswith("/segment")
        assert call["body"]["concepts"] == ["building"]
        assert "image_b64" in call["body"]
        assert "session_id" not in call["body"]


def test_distinct_object_ids_yield_distinct_metadata(adapter, monkeypatch):
    """Two detections in one frame get distinct position-derived track_ids and
    carry their distinct wire ``object_id`` in ``sam3_1_object_id``."""
    obs = _build_observation(_square_frame())
    mask_a = _block_mask(100, 250, 100, 250)
    mask_b = _block_mask(700, 850, 700, 850)

    payload = _build_segment_payload(
        object_ids=[1, 42],
        masks=[mask_a, mask_b],
        scores=[0.6, 0.6],
        concepts=["building", "building"],
        image_shape=(_SAM_INPUT_SIZE, _SAM_INPUT_SIZE),
    )
    with _mock_requests_post(monkeypatch, [payload]):
        dets = adapter.detect(obs, (0.0, 0.0))

    assert len(dets) == 2
    track_ids = sorted(d.metadata["track_id"] for d in dets)
    assert track_ids == ["sam3_1_img_0", "sam3_1_img_1"]
    object_ids = sorted(d.metadata["sam3_1_object_id"] for d in dets)
    assert object_ids == [1, 42]


def test_returns_empty_when_rgb_missing(adapter, monkeypatch):
    obs = _build_observation(np.full((16, 16, 3), 128, dtype=np.uint8))
    obs.rgb = None  # type: ignore[assignment]
    with _mock_urlopen(monkeypatch, []) as calls:
        assert adapter.detect(obs, obs.pose_xy) == []
    assert calls == []  # no HTTP traffic when there's no frame to send


def test_requires_ortho_corners(adapter):
    obs = _build_observation(np.full((16, 16, 3), 128, dtype=np.uint8))
    obs.ortho_view_corners_xy = None
    with pytest.raises(ValueError):
        adapter.detect(obs, obs.pose_xy)


# --- replica placement + the profile gate -------------------------------


def test_pick_url_is_round_robin(monkeypatch):
    """Even placement, not statistical placement. Nine concurrent tiles over
    three replicas must land 3/3/3: the wave arithmetic behind the replica
    count assumes it, and ``random.choice`` leaves an expected max load above
    ``tiles/replicas`` while other GPUs idle."""
    from maps.perception.sam import detect_client as dc

    eps = ["http://a:1", "http://b:2", "http://c:3"]
    monkeypatch.setenv("SAM_VIDEO_SERVER_URL", ",".join(eps))
    picks = [dc._pick_url() for _ in range(9)]

    assert sorted(picks) == sorted(eps * 3)
    # consecutive picks never repeat a replica -> a burst spreads immediately
    assert all(a != b for a, b in zip(picks, picks[1:]))


def test_pick_url_survives_a_shrinking_endpoint_list(monkeypatch):
    """The counter is module-level and the list is read per call (preflight
    prunes dead replicas into the env), so the modulo must not index past a
    list that just got shorter."""
    from maps.perception.sam import detect_client as dc

    monkeypatch.setenv("SAM_VIDEO_SERVER_URL", "http://a:1,http://b:2,http://c:3")
    for _ in range(5):
        dc._pick_url()
    monkeypatch.setenv("SAM_VIDEO_SERVER_URL", "http://a:1")
    assert {dc._pick_url() for _ in range(4)} == {"http://a:1"}


def test_tile_workers_track_the_replica_count(monkeypatch):
    """Default width follows ``SAM_SERVER_URL``; an explicit value still wins
    (1 is the serial triage knob). A hardcoded width goes stale the next time
    the plane changes replica count and leaves the new GPUs idle."""
    from maps.perception.sam import detect_client as dc

    monkeypatch.delenv("SAM_VIDEO_SERVER_URL", raising=False)
    monkeypatch.setenv("SAM_SERVER_URL", ",".join(f"http://r{i}:8772" for i in range(7)))

    auto = SamDetectClient(concept_provider=ReasonerUnionConceptProvider())
    assert auto.sahi_tile_workers is None
    # AUTO COVERS THE TILES. Sizing off the replica list alone left a tail
    # wave: 9 tiles at a width of 8 pays two round trips to do one tile's
    # work, which is what a Cal City stop was doing on every flight.
    assert dc.tile_pool_width(9, auto.sahi_tile_workers) == 9
    # ...and with fewer tiles than replicas the pool is still the tile count,
    # never idle threads.
    assert dc.tile_pool_width(4, auto.sahi_tile_workers) == 4

    pinned = SamDetectClient(
        concept_provider=ReasonerUnionConceptProvider(), sahi_tile_workers=1
    )
    assert dc.tile_pool_width(9, pinned.sahi_tile_workers) == 1
    # a pin above the tile count still cannot spawn idle threads
    assert dc.tile_pool_width(3, 32) == 3
    # and a degenerate tile count never asks for a zero-wide pool
    assert dc.tile_pool_width(0, None) == 1


