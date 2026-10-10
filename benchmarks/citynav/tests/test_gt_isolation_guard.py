"""Regression guard for the C1 GT-isolation runtime gate.

``_assert_realstack_gt_isolation`` lives in ``run_benchmark.config`` and, under
the real-SAM ``sam`` tier, asserts a resolved perception OBJECT is a
``SamDetectClient`` via ``isinstance``. That branch only evaluates
when a non-None ``perception`` is passed (the post-construction call in
``_run_episode``) — so a missing ``SamDetectClient`` import in
``config.py`` slips past config-load and import smoke, surfacing only mid-episode
as a ``NameError``. This exercises the branch so the import can never regress.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.citynav.run_benchmark import load_config
from benchmarks.citynav.run_benchmark.config import _assert_realstack_gt_isolation

_CITYNAV_YAML = Path(__file__).resolve().parents[3] / "configs" / "citynav.yaml"


def test_realstack_gt_isolation_rejects_non_sam_perception():
    """A non-SAM perception object under the sam tier must raise the C1
    RuntimeError (executing the ``isinstance(..., SamDetectClient)``
    guard) — never a NameError from a missing import."""
    cfg = load_config(str(_CITYNAV_YAML))  # canonical config -> perception == sam
    with pytest.raises(RuntimeError, match="GT-ISOLATION"):
        _assert_realstack_gt_isolation(cfg, perception=object())
