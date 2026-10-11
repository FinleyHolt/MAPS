"""Bbox-cropped, bit-packed, deflated pixel-mask records.

One encoding for every pixel mask that crosses a process boundary: the SAM
server's detection wire and the per-step detection record. A mask is stored as its bounding-box crop, bit-packed one bit per pixel,
then deflated::

    {
      "pixel_bbox_rcrc":  [r0, c0, r1, c1],   # absolute, INCLUSIVE
      "mask_shape":       [rows, cols],       # = [r1-r0+1, c1-c0+1]
      "mask_zpacked_b64": "<b64 of zlib(packbits(crop.ravel()))>",
    }

Bbox coordinates are absolute in the source image's pixel space, so a decoder
can place the crop back on a full frame without any other context.

Measured on one real CityNav step (38 detections, 2016x2016 frame): a
full-frame PNG per mask costs 215 KiB, the bare bit-packed crop 67 KiB, and
this record 10 KiB — 21x smaller than the PNG for about 1 ms of encode per
frame, which is noise beside SAM inference.

``mask_zpacked_b64`` is deliberately not named ``mask_packed_b64``: the payload
is deflated, so a reader expecting raw ``packbits`` bytes must fail on the
missing key rather than silently unpack compressed data into a plausible-looking
wrong mask.
"""

from __future__ import annotations

import base64
import zlib

import numpy as np

# Deflate level. 6 buys ~21x over the raw packed crop for ~1 ms per frame;
# 9 buys a further 3% for 4x the CPU.
_DEFLATE_LEVEL = 6


def _extent(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive ``(r0, c0, r1, c1)`` of a 2-D bool mask's set pixels; ``None`` when empty."""
    rows = mask.any(axis=1)
    if not rows.any():
        return None
    cols = mask.any(axis=0)
    return (
        int(rows.argmax()),
        int(cols.argmax()),
        int(len(rows) - 1 - rows[::-1].argmax()),
        int(len(cols) - 1 - cols[::-1].argmax()),
    )


def encode_mask_record(
    mask,
    *,
    bbox_rcrc: tuple[int, int, int, int] | None = None,
) -> dict | None:
    """Encode a boolean mask as a bbox-cropped, packed, deflated record.

    ``mask`` is either a full-frame ``(H, W)`` bool array or an already-cropped
    one; pass ``bbox_rcrc`` when it is already cropped, otherwise the bbox is
    derived from the mask's own extent. Returns ``None`` for an empty mask —
    a detection with no pixels carries no mask, and callers drop it rather than
    recording a zero-area record.
    """
    arr = np.asarray(mask).astype(bool, copy=False)

    if bbox_rcrc is None:
        ext = _extent(arr)
        if ext is None:
            return None
        r0, c0, r1, c1 = ext
        crop = arr[r0 : r1 + 1, c0 : c1 + 1]
    else:
        r0, c0, r1, c1 = (int(v) for v in bbox_rcrc)
        crop = arr
        if crop.shape != (r1 - r0 + 1, c1 - c0 + 1):
            # A full-frame mask arrived with its bbox alongside; reduce it.
            crop = arr[r0 : r1 + 1, c0 : c1 + 1]
        if not crop.any():
            return None

    packed = np.packbits(crop.ravel()).tobytes()
    return {
        "pixel_bbox_rcrc": [r0, c0, r1, c1],
        "mask_shape": [int(crop.shape[0]), int(crop.shape[1])],
        "mask_zpacked_b64": base64.b64encode(
            zlib.compress(packed, _DEFLATE_LEVEL)
        ).decode("ascii"),
    }


def decode_mask_crop(record: dict) -> np.ndarray:
    """Decode a record back to its ``(rows, cols)`` bool crop."""
    rows, cols = (int(v) for v in record["mask_shape"])
    packed = zlib.decompress(base64.b64decode(record["mask_zpacked_b64"]))
    return (
        np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=rows * cols)
        .reshape((rows, cols))
        .astype(bool)
    )


__all__ = ["decode_mask_crop", "encode_mask_record"]
