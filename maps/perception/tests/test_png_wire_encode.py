"""The SAM wire PNG must stay lossless, and must stay the only encoder.

The speed came from PNG's zlib LEVEL, which does not touch pixels. The failure
mode this guards is someone reaching for JPEG on the same wire for another 20x:
measured on real tiles against a bit-deterministic SAM, q95 changed 12% of
detections and 7.4% of cars. That is an SR question, not a latency one.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from maps.perception.sam import detect_client as dc


def _tile(seed: int = 0) -> np.ndarray:
    """Structured, compressible pixels — noise would hide any level effect."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, 256, dtype=np.uint8)
    img = np.tile(base, (256, 1))[..., None].repeat(3, axis=2)
    img[64:128, 64:192] = rng.integers(0, 255, (64, 128, 3), dtype=np.uint8)
    return img


def test_wire_png_is_lossless() -> None:
    """The bytes SAM decodes are bit-identical to the array handed in."""
    from PIL import Image

    img = _tile()
    back = np.array(Image.open(io.BytesIO(dc._png_wire_bytes(img))).convert("RGB"))
    assert np.array_equal(back, img)


def test_wire_png_is_actually_png() -> None:
    """A lossy format on this wire would still decode; the magic number is the
    thing that changes. This is the test that fails if someone swaps in JPEG."""
    assert dc._png_wire_bytes(_tile())[:8] == b"\x89PNG\r\n\x1a\n"


def test_wire_level_is_the_fast_one() -> None:
    """Pin the level itself. PIL's default is 6 and that default cost 2.4 s of
    client CPU per SAHI stop; a silent revert to it would not fail any other
    assertion here, because every level is lossless."""
    assert dc._PNG_WIRE_LEVEL == 1


def test_faster_level_is_not_much_larger() -> None:
    """Level 1 trades bytes for time. Measured on a real 1008 native tile the
    trade was +0.4%; this bounds it loosely so a future level change that
    doubled the payload would fail rather than quietly cost link budget."""
    from PIL import Image

    img = _tile()
    ref = io.BytesIO()
    Image.fromarray(img).save(ref, format="PNG", compress_level=9)
    assert len(dc._png_wire_bytes(img)) < 2.0 * len(ref.getvalue())


def test_both_segment_paths_use_the_one_encoder(monkeypatch) -> None:
    """Vacuity floor: the SAHI tile path and the single-frame path must BOTH
    route through ``_png_wire_bytes``. Two call sites encoded inline before
    this change, and a revert of either is invisible to every test above.
    """
    import inspect

    src = inspect.getsource(dc)
    # No inline encoder survives outside the helper. Counted across BOTH
    # spellings so the check keeps working across an encoder swap: the wire
    # moved from PIL to cv2 and a grep for one of them would have gone quietly
    # vacuous, which is the failure mode this floor exists to prevent.
    encodes = src.count('format="PNG"') + src.count('cv2.imencode(')
    assert encodes == 1, f"expected one PNG encode site, found {encodes}"
    assert "cv2.imencode(" in inspect.getsource(dc._png_wire_bytes)
    for fn in (dc.SamDetectClient._sahi_segment,
               dc.SamDetectClient._segment_image_via_server):
        body = inspect.getsource(fn)
        assert "_png_wire_bytes(" in body, f"{fn.__name__} no longer uses the encoder"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
