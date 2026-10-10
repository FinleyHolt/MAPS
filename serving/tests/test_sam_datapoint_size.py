"""The SAM datapoint must declare its image size as (H, W).

``PostProcessImage`` unbinds ``original_size`` as ``img_h, img_w =
target_sizes.unbind(1)`` for boxes and feeds it straight to ``interpolate`` for
masks, so the order is load-bearing. Reversed, every detection on a non-square
frame comes back transposed: right shapes, plausible scores, wrong pixels.

Nothing caught it because every frame the runtime feeds SAM is square --
``rgb_upscale_px`` renders 1008 or 2016, and SAHI tiles are 1008 square -- and
``w == h`` makes the two orders identical. A caller that hands SAM a real
camera frame (the 4:3 Oryx, 2048x1536) is the first to see it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sam3", reason="the FAIR sam3 stack is not in every env")

import numpy as np  # noqa: E402

from serving.sam_engine import _build_datapoint  # noqa: E402


@pytest.mark.parametrize("h,w", [(1536, 2048), (2048, 1536), (1008, 1008), (720, 1280)])
def test_original_size_is_height_then_width(h: int, w: int) -> None:
    dp, qids = _build_datapoint(np.zeros((h, w, 3), np.uint8), ["car", "pool"], [0])

    assert len(qids) == 2, "one query per concept"
    assert dp.images[0].size == [h, w]
    for q in dp.find_queries:
        assert list(q.inference_metadata.original_size) == [h, w], (
            "postprocessors.py unbinds this as (img_h, img_w); reversed, a "
            "non-square frame's boxes and masks come back transposed"
        )


def test_a_non_square_frame_would_expose_a_reversed_order() -> None:
    """The vacuity floor: the parametrized case above only means something if
    at least one shape actually distinguishes the two orders."""
    dp, _ = _build_datapoint(np.zeros((1536, 2048, 3), np.uint8), ["car"], [0])
    size = list(dp.find_queries[0].inference_metadata.original_size)
    assert size != list(reversed(size)), "a square frame cannot pin the order"
