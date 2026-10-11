"""Prior-enrich build/apply seam: the per-frame BUILD returns exactly the
items to apply (empty-fid / missing-mask / empty-mix filtered), and a
failure is CARRIED (not raised) with the items done before it, so the
flight's apply warns and counts it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.citynav.run_benchmark import osm_prior
from maps.util.mask_region import MaskRegion


def _mask(n: int = 4):
    """What the prior builders now hand back: a region, not a frame."""
    m = np.zeros((8, 8), dtype=bool)
    m[:n, 0] = True
    return MaskRegion.from_full(m)


def _obs():
    return SimpleNamespace(rgb=np.zeros((8, 8, 3), dtype=np.uint8))


_ASSETS = SimpleNamespace(bounds=SimpleNamespace(x_min=0.0, y_min=0.0), cell_size_m=1.0,
                          grid_shape=(8, 8), base_features={})


def _build(monkeypatch, polys):
    monkeypatch.setattr(osm_prior, "_prior_fids_in_view", lambda *a, **k: ["f1"])
    monkeypatch.setattr(osm_prior, "build_prior_polys_ortho", lambda **k: (polys, []))
    return osm_prior._build_frame_enrich(
        obs=_obs(), step_idx=0, view_corners=((0, 1), (1, 1), (1, 0), (0, 0)), assets=_ASSETS,
        build_prior=True)


def test_build_items_filters(monkeypatch):
    m_ok, m_empty_mix = _mask(4), _mask(6)
    monkeypatch.setattr(
        osm_prior,
        "color_from_mask_m6",
        lambda rgb, region: (
            {}
            if region is m_empty_mix
            else {
                "color": "red",
                "color_conf": 0.5,
                "color_shade": "dark",
                "color_flag": "ok",
            }
        ),
    )
    out = _build(monkeypatch, [
        {"fid": "building_3", "mask": m_ok},  # kept
        {"fid": "", "mask": m_ok},  # empty fid -> dropped
        {"fid": "f2", "mask": None},  # no mask -> dropped
        {"fid": "f3", "mask": m_empty_mix},  # empty mix -> dropped
    ])

    # vacuity floor: the filters really dropped 3 of 4 prior polys.
    assert out["prior"] == [
        (
            "building_3",
            {
                "color": "red",
                "color_conf": "0.500",
                "color_shade": "dark",
                "color_flag": "ok",
            },
            _mask(4).area,
        )
    ]
    assert out["prior_exc"] is None


def test_build_failure_is_carried_with_the_items_done_before_it(monkeypatch):
    seen: list = []

    def _colour(rgb, region):
        if seen:
            raise RuntimeError("colour boom")
        seen.append(region)
        return {"color": "red", "color_conf": 0.5, "color_shade": "dark", "color_flag": "ok"}

    monkeypatch.setattr(osm_prior, "color_from_mask_m6", _colour)
    out = _build(monkeypatch, [{"fid": "prior_1", "mask": _mask(4)}, {"fid": "prior_2", "mask": _mask(5)}])
    assert isinstance(out["prior_exc"], RuntimeError)
    assert [fid for fid, _a, _px in out["prior"]] == ["prior_1"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
