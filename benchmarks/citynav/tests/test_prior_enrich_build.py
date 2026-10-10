"""OSM-enrich build/apply seam: the pure per-frame BUILD returns exactly
the items the serial inline passes would have applied — remap through
graph_fid_by_fid, empty-fid / missing-mask / empty-mix filters — and a
build failure is CARRIED (not raised) with the other pass unaffected, so
the flight body's apply can replay the serial partial-apply + rate-limited
warn semantics.
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


def _ctx():
    return SimpleNamespace(
        graph_fid_by_fid={"cityref_9": "building_3"},
        meta_by_fid={},
        percept_by_fid={},
    )


def test_build_items_remap_and_filters(monkeypatch):
    m_ok, m_empty_mix = _mask(4), _mask(6)
    monkeypatch.setattr(osm_prior, "_prior_fids_in_view", lambda *a, **k: ["f1"])
    monkeypatch.setattr(
        osm_prior,
        "_maybe_build_prior_polys",
        lambda **k: (
            [
                {"fid": "cityref_9", "mask": m_ok},  # remapped + kept
                {"fid": "", "mask": m_ok},  # empty fid -> dropped
                {"fid": "f2", "mask": None},  # no mask -> dropped
                {"fid": "f3", "mask": m_empty_mix},  # empty mix -> dropped
            ],
            None,
        ),
    )
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
    monkeypatch.setattr(osm_prior, "_all_osm_fids_in_view", lambda *a, **k: ["o1"])
    monkeypatch.setattr(
        osm_prior,
        "_build_all_osm_polys",
        lambda **k: ([{"fid": "o1", "mask": m_ok}, {"fid": "", "mask": m_ok}], None),
    )
    monkeypatch.setattr(
        "maps.perception.sam.detect_client._set_color_attr",
        lambda attrs, rgb, mask: attrs.__setitem__("color", "blue"),
    )

    out = osm_prior._build_frame_enrich(
        obs=_obs(),
        step_idx=0,
        view_corners=((0, 1), (1, 1), (1, 0), (0, 0)),
        assets=SimpleNamespace(),
        osm_ctx=_ctx(),
        build_prior=True,
    )

    # vacuity floors: the filters really dropped 3 of 4 prior polys and
    # 1 of 2 osm polys, and the remap really rewrote the fid.
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
    assert out["osm"] == [("o1", {"color": "blue"})]
    assert out["prior_exc"] is None and out["osm_exc"] is None


def test_build_failure_is_carried_and_passes_stay_independent(monkeypatch):
    monkeypatch.setattr(osm_prior, "_prior_fids_in_view", lambda *a, **k: ["f1"])

    def _boom(**k):
        raise RuntimeError("poly build boom")

    monkeypatch.setattr(osm_prior, "_maybe_build_prior_polys", _boom)
    monkeypatch.setattr(osm_prior, "_all_osm_fids_in_view", lambda *a, **k: ["o1"])
    monkeypatch.setattr(
        osm_prior, "_build_all_osm_polys", lambda **k: ([{"fid": "o1", "mask": _mask()}], None)
    )
    monkeypatch.setattr(
        "maps.perception.sam.detect_client._set_color_attr",
        lambda attrs, rgb, mask: None,
    )

    out = osm_prior._build_frame_enrich(
        obs=_obs(),
        step_idx=3,
        view_corners=None,
        assets=SimpleNamespace(),
        osm_ctx=_ctx(),
        build_prior=True,
    )

    assert isinstance(out["prior_exc"], RuntimeError)
    assert out["prior"] == []
    # pass 2 is unaffected by pass 1's failure
    assert out["osm"] == [("o1", {})]
    assert out["osm_exc"] is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
