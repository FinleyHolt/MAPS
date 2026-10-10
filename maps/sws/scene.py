"""The LabelMask primitive: one labeled boolean region on the tile grid.

A scene is an unordered ``list[LabelMask]``. Each ``LabelMask`` carries an
open-vocabulary ``label``, a boolean ``mask`` on the grid, a ``source`` tag
naming the ingest path or detector that produced it, and a free-form
``metadata`` dict. Multiple masks with the same label coexist — one per
observed instance — and masks are *expected* to overlap (a building inside a
park, a car on a road). Overlap is the world model, and the mask stack here
is where it is preserved: the ``ObjectGraph`` stores no edges at all, so
every spatial relation is computed geometrically from node xy/cells by the
``geom_compute`` ops at query time. The scene is canonical everywhere; it is
never collapsed to one-label-per-cell for storage or grounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, eq=False)
class LabelMask:
    """A single labeled boolean region on the tile grid."""

    label: str
    mask: np.ndarray
    source: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mask.dtype != np.bool_:
            raise TypeError(f"LabelMask.mask must be bool, got {self.mask.dtype}")
        if self.mask.ndim != 2:
            raise ValueError(f"LabelMask.mask must be 2-D, got shape {self.mask.shape}")

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.mask.shape  # type: ignore[return-value]

    @property
    def cell_count(self) -> int:
        return int(self.mask.sum())
