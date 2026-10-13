"""The as-released GeoNav arm: ``baselines.geonav`` with ``released=True``.

The seam sends ``reasoning_effort: none`` for this policy name and the
episode runs under the released 800 / 2,048-token caps; prompts, loop, map,
graph and query ladder are ``baselines/geonav``'s.
"""

from __future__ import annotations

from baselines.geonav.policy import run_episode as _run_geonav
from baselines.seam import BaselineHarness, BaselineResult


def run_episode(h: BaselineHarness) -> BaselineResult:
    return _run_geonav(h, released=True)
