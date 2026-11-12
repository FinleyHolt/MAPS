"""Per-frame prior enrichment yields exactly the items to apply (a mask M6
reads nothing on is dropped), and an M6 failure raises only after the items
done before it were yielded, so the flight applies them and counts the
failure.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.citynav.run_benchmark import osm_prior
from maps.perception.tests.regions import region_of


def _mask(n: int = 4):
    m = np.zeros((8, 8), dtype=bool)
    m[:n, 0] = True
    return region_of(m)


def _obs():
    return SimpleNamespace(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        ortho_view_corners_xy=((0, 1), (1, 1), (1, 0), (0, 0)),
    )


_ASSETS = SimpleNamespace(
    bounds=SimpleNamespace(x_min=0.0, y_min=0.0),
    cell_size_m=1.0,
    grid_shape=(8, 8),
    base_features={},
)


def _build(monkeypatch, polys):
    monkeypatch.setattr(osm_prior, "_prior_fids_in_view", lambda *a, **k: ["f1"])
    monkeypatch.setattr(osm_prior, "build_prior_polys_ortho", lambda **k: polys)
    return osm_prior.prior_percepts(_obs(), SimpleNamespace(assets=_ASSETS, tile_frame=None), 0)


def test_build_items_filters(monkeypatch):
    m_ok, m_empty_mix = _mask(4), _mask(6)
    attrs = {"color": "red", "color_conf": "0.500", "color_shade": "dark", "color_flag": "ok"}
    monkeypatch.setattr(
        osm_prior, "m6_attributes", lambda rgb, region: None if region is m_empty_mix else attrs
    )
    out = _build(monkeypatch, [("building_3", m_ok), ("f3", m_empty_mix)])
    assert list(out) == [("building_3", attrs, _mask(4).area)]


def test_a_failure_raises_after_the_items_done_before_it(monkeypatch):
    seen: list = []

    def _colour(rgb, region):
        if seen:
            raise RuntimeError("colour boom")
        seen.append(region)
        return {"color": "red", "color_conf": "0.500", "color_shade": "dark", "color_flag": "ok"}

    monkeypatch.setattr(osm_prior, "m6_attributes", _colour)
    got: list = []
    with pytest.raises(RuntimeError, match="colour boom"):
        for item in _build(monkeypatch, [("prior_1", _mask(4)), ("prior_2", _mask(5))]):
            got.append(item)
    assert [fid for fid, _a, _px in got] == ["prior_1"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
