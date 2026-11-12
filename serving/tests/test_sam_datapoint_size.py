"""The SAM datapoint declares its image size as (H, W): ``PostProcessImage`` unbinds it as
``img_h, img_w``, so a reversed order transposes every detection on a non-square frame."""

from __future__ import annotations

import pytest

pytest.importorskip("sam3", reason="the FAIR sam3 stack is not in every env")

import numpy as np

from serving.sam_engine import _build_datapoint


@pytest.mark.parametrize("h,w", [(1536, 2048), (2048, 1536)])
def test_original_size_is_height_then_width(h: int, w: int) -> None:
    dp, qids = _build_datapoint(np.zeros((h, w, 3), np.uint8), ["car", "pool"])

    assert len(qids) == 2, "one query per concept"
    assert dp.images[0].size == [h, w]
    for q in dp.find_queries:
        assert list(q.inference_metadata.original_size) == [h, w], (
            "postprocessors.py unbinds this as (img_h, img_w); reversed, a "
            "non-square frame's boxes and masks come back transposed"
        )
