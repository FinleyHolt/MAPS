"""``TileBounds``: a prebake tile's world-grid layout.

Robot-agnostic: this module lives in ``maps/perception/`` and does
not import from ``maps.search`` or ``benchmark`` per the
robot-agnostic rule in ``maps/perception/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TileBounds:
    """Minimal description of a prebake tile's world-grid layout.

    Mirrors the shape of ``benchmarks.citynav.BlockAssets`` (``x_min``,
    ``y_min``, ``cell_size_m``, ``grid_shape``) without importing it, so this
    module stays robot-agnostic per ``maps/perception/README.md``.
    """

    x_min: float
    y_min: float
    cell_size_m: float
    grid_shape: tuple[int, int]
