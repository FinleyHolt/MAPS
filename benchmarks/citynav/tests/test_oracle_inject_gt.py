"""Tier-(c) oracle GT-injection diagnostic.

Pins the three benchmarks-side contracts: the GT-nearest prior construct
(same prior set as the osm_percepts sidecar), the C1 ack gate
(``SEARCH_ORACLE_INJECT_GT`` demands ``SEARCH_ALLOW_GT_DIAGNOSTIC``,
tier-independent), and the off-path default on ``CompilerInput``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.citynav.run_benchmark.config import (
    _assert_realstack_gt_isolation,
)
from benchmarks.citynav.run_benchmark.osm_prior import _gt_nearest_prior_fid
from maps.search.protocol import CompilerInput


def _square(cx: float, cy: float, r: float = 1.0) -> list[tuple[float, float]]:
    return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]


def _graph(nodes: dict) -> SimpleNamespace:
    return SimpleNamespace(nodes=nodes)


def test_nearest_contour_prior_wins_and_detected_never_does():
    g = _graph(
        {
            "prior_far": SimpleNamespace(
                provenance="osm", world_contour=_square(100.0, 0.0)
            ),
            "prior_near": SimpleNamespace(
                provenance="osm", world_contour=_square(10.0, 0.0)
            ),
            # Nearer than every prior — must still lose (not a prior).
            "detected_0": SimpleNamespace(
                provenance="detected", world_contour=_square(1.0, 0.0)
            ),
        }
    )
    fid, dist = _gt_nearest_prior_fid(g, (0.0, 0.0))
    assert fid == "prior_near"
    assert dist == pytest.approx(10.0)


def test_no_contour_bearing_prior_returns_none():
    for nodes in (
        {},  # empty graph
        {"prior_0": SimpleNamespace(provenance="osm", world_contour=None)},
        {  # < 3 points is not a contour
            "prior_1": SimpleNamespace(
                provenance="osm", world_contour=[(0.0, 0.0), (1.0, 1.0)]
            )
        },
        {  # detections only
            "detected_0": SimpleNamespace(
                provenance="detected", world_contour=_square(0.0, 0.0)
            )
        },
    ):
        assert _gt_nearest_prior_fid(_graph(nodes), (0.0, 0.0)) == (None, None)


def test_ack_gate_refuses_unacknowledged_injection(monkeypatch):
    """SEARCH_ORACLE_INJECT_GT without the C1 ack fails loud at config load,
    on EVERY perception tier (checked before the sam-tier early return)."""
    monkeypatch.setenv("SEARCH_ORACLE_INJECT_GT", "1")
    monkeypatch.delenv("SEARCH_ALLOW_GT_DIAGNOSTIC", raising=False)
    cfg = SimpleNamespace(perception="gt_perfect")  # non-sam tier
    with pytest.raises(RuntimeError, match="SEARCH_ALLOW_GT_DIAGNOSTIC"):
        _assert_realstack_gt_isolation(cfg)
    # The documented ack clears it (non-sam tier then returns early).
    monkeypatch.setenv("SEARCH_ALLOW_GT_DIAGNOSTIC", "1")
    _assert_realstack_gt_isolation(cfg)


def test_compiler_input_field_defaults_off():
    inp = CompilerInput(mission={}, grid_shape=(1, 1))
    assert inp.oracle_inject_fid is None
