"""Prior-enrich build/apply seam: the per-frame BUILD returns exactly the
items to apply (a mask M6 reads nothing on is dropped), and a failure is
CARRIED with the items done before it, so the flight's apply warns and
counts it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.citynav.run_benchmark import osm_prior
from maps.util.mask_region import MaskRegion


def _mask(n: int = 4):
    m = np.zeros((8, 8), dtype=bool)
    m[:n, 0] = True
    return MaskRegion.from_full(m)


def _obs():
    return SimpleNamespace(rgb=np.zeros((8, 8, 3), dtype=np.uint8))


_ASSETS = SimpleNamespace(bounds=SimpleNamespace(x_min=0.0, y_min=0.0), cell_size_m=1.0,
                          grid_shape=(8, 8), base_features={})


def _build(monkeypatch, polys):
    monkeypatch.setattr(osm_prior, "_prior_fids_in_view", lambda *a, **k: ["f1"])
    monkeypatch.setattr(osm_prior, "build_prior_polys_ortho", lambda **k: polys)
    return osm_prior._build_frame_enrich(
        obs=_obs(), step_idx=0, view_corners=((0, 1), (1, 1), (1, 0), (0, 0)), assets=_ASSETS,
        build_prior=True)


def test_build_items_filters(monkeypatch):
    m_ok, m_empty_mix = _mask(4), _mask(6)
    attrs = {"color": "red", "color_conf": "0.500", "color_shade": "dark", "color_flag": "ok"}
    monkeypatch.setattr(osm_prior, "m6_attributes",
                        lambda rgb, region: None if region is m_empty_mix else attrs)
    out = _build(monkeypatch, [("building_3", m_ok), ("f3", m_empty_mix)])
    assert out["prior"] == [
        (
            "building_3",
            attrs,
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
        return {"color": "red", "color_conf": "0.500", "color_shade": "dark", "color_flag": "ok"}

    monkeypatch.setattr(osm_prior, "m6_attributes", _colour)
    out = _build(monkeypatch, [("prior_1", _mask(4)), ("prior_2", _mask(5))])
    assert isinstance(out["prior_exc"], RuntimeError)
    assert [fid for fid, _a, _px in out["prior"]] == ["prior_1"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
