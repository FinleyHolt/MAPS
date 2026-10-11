"""SAM 3.1 image segmentation server.

A thin HTTP wrapper around ``serving.sam_engine``. It runs in its own env
(``envs/sam3_1.yml``: ``sam3`` needs numpy<2) from the repo root, and the
bench reaches it over ``SAM_SERVER_URL``::

    python -m serving.sam_server --port 8772

Endpoints
---------
``GET /health``
    ``{"ok", "service": "sam", "version", "device", "dtype", "model_id",
    "vram"}``. Endpoint discovery (``maps.llm.endpoint_spec``) checks
    ``service`` so a foreign server answering on the SAM port aborts the
    run at startup. A server whose free device memory falls under
    ``SAM_HEALTH_MIN_FREE_GB`` answers 503, so a wedged server reads as an
    outage instead of a slow one.

``POST /segment`` (stateless, single image)
    body = {image_b64, concepts, score_threshold, min_area_px,
            max_per_image, resolution}
    response = {detections: [...], image_shape_hw}

Each detection is ``{label, score, object_id}`` plus the bbox-cropped mask
record of :mod:`maps.util.mask_codec`.

A ``threading.Lock`` serializes GPU access: one forward at a time. A
bounded admission semaphore (``SAM_MAX_INFLIGHT``, default 2) is acquired
BEFORE a handler decodes its payload, so at most that many frames are
decoded and resident at once; a request that waits past
``SAM_INFLIGHT_WAIT_S`` gets a 503.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import threading

# Set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from fastapi import Depends, FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from maps.util.mask_codec import encode_mask_record

log = logging.getLogger("sam_server")
logging.basicConfig(level=logging.INFO)

_MODEL_ID = os.environ.get("SAM_MODEL_ID", "facebook/sam3.1")
_DEVICE = os.environ.get("SAM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
_DTYPE_STR = os.environ.get("SAM_DTYPE", "bfloat16")

_state_lock = threading.Lock()
_MAX_INFLIGHT = int(os.environ.get("SAM_MAX_INFLIGHT", "2"))
_INFLIGHT_WAIT_S = float(os.environ.get("SAM_INFLIGHT_WAIT_S", "600"))
_inflight_sem = threading.BoundedSemaphore(max(1, _MAX_INFLIGHT))


def _admission():
    """FastAPI dependency: bound concurrent decoded-payload residency."""
    if not _inflight_sem.acquire(timeout=_INFLIGHT_WAIT_S):
        raise HTTPException(
            503,
            detail={
                "error": "sam_admission_timeout",
                "waited_s": _INFLIGHT_WAIT_S,
                "max_inflight": _MAX_INFLIGHT,
            },
        )
    try:
        yield
    finally:
        _inflight_sem.release()


def _detections_to_wire(dets: list) -> list[dict]:
    """Pack engine detections into the wire format. A mask with no pixels
    carries no detection, so it is dropped."""
    out: list[dict] = []
    for d in dets:
        record = encode_mask_record(d.pixel_mask, bbox_rcrc=d.pixel_bbox_rcrc)
        if record is None:
            continue
        out.append(
            {
                "label": str(d.label),
                "score": float(d.score),
                "object_id": int(d.object_id),
                **record,
            }
        )
    return out


class SegmentRequest(BaseModel):
    image_b64: str
    concepts: list[str]
    score_threshold: float = 0.3
    min_area_px: int = 32
    max_per_image: int = 1024
    resolution: int = 1008


app = FastAPI(title="sam3.1-image", version="0.1")


@app.on_event("shutdown")
def _release() -> None:
    """Hand the resident model's device memory back before the process ends."""
    from serving.sam_engine import release

    with _state_lock:
        release()


@app.on_event("startup")
def _prewarm() -> None:
    """One forward on a black frame so the first ``/segment`` does not pay
    the checkpoint load and kernel warmup. ``SAM_PREWARM=0`` skips it; a
    failure is logged and does not block readiness."""
    if os.environ.get("SAM_PREWARM", "1") == "0" or not torch.cuda.is_available():
        return
    try:
        from serving.sam_engine import segment

        rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        with _state_lock:
            segment(rgb, ["car", "building", "road"], device=_DEVICE,
                    score_threshold=0.3, min_area_px=32, max_per_image=4)
        log.info("prewarm: SAM 3.1 warm")
    except Exception as exc:  # noqa: BLE001
        log.warning("prewarm failed (%s); continuing without warmup", exc)


@app.get("/health")
def health() -> dict:
    vram: dict[str, float] = {}
    if torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            vram = {
                "used_gb": round((total_b - free_b) / (1024**3), 3),
                "free_gb": round(free_b / (1024**3), 3),
                "total_gb": round(total_b / (1024**3), 3),
            }
        except (RuntimeError, AssertionError):
            vram = {}
    payload = {
        "ok": True,
        "service": "sam",
        "version": "sam3.1-image",
        "device": str(_DEVICE),
        "dtype": _DTYPE_STR,
        "model_id": _MODEL_ID,
        "vram": vram,
    }
    floor = float(os.environ.get("SAM_HEALTH_MIN_FREE_GB", "1.0"))
    if vram and vram["free_gb"] < floor:
        payload["ok"] = False
        payload["wedged"] = f"free_gb {vram['free_gb']} < floor {floor}"
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.post("/segment")
def segment(req: SegmentRequest, _gate: None = Depends(_admission)) -> dict:
    """One SAM 3.1 forward over one image and every requested concept."""
    from serving.sam_engine import segment as engine_segment

    rgb = np.array(
        Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB"),
        dtype=np.uint8,
    )
    with _state_lock:
        dets = engine_segment(
            rgb,
            list(req.concepts),
            device=_DEVICE,
            score_threshold=req.score_threshold,
            min_area_px=req.min_area_px,
            max_per_image=req.max_per_image,
            resolution=req.resolution,
        )
    H, W = rgb.shape[:2]
    return {"detections": _detections_to_wire(dets), "image_shape_hw": [int(H), int(W)]}


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(prog="python -m serving.sam_server", description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("SAM_PORT", "8772")))
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
