"""The bytes SAM decodes off the wire are bit-identical to the frame handed in."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from maps.perception.sam import detect_client as dc


def test_wire_png_is_lossless() -> None:
    rng = np.random.default_rng(0)
    img = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))[..., None].repeat(3, axis=2)
    img[64:128, 64:192] = rng.integers(0, 255, (64, 128, 3), dtype=np.uint8)
    back = np.array(Image.open(io.BytesIO(dc._png_wire_bytes(img))).convert("RGB"))
    assert np.array_equal(back, img)
