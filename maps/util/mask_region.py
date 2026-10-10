"""A detection mask as what it is: a crop, and where it sits in the frame.

The SAM wire carries a bbox crop. The decode path used to paste that crop onto
a full frame-sized canvas, at which point the mask's location was no longer
carried anywhere — so every downstream stage rediscovered it by scanning the
whole frame. Projection, colour, component labelling and the area count each
paid for the same rediscovery, per detection, and the cost scaled with the
SENSOR's resolution rather than with the object.

Passing ``(crop, bbox)`` as two arguments fixes the cost and introduces a
worse hazard: nothing checks that the box actually bounds the mask. A stale or
mismatched box does not raise. It silently narrows a scan, so the detection
keeps a plausible ``world_xy`` a few metres off with every count still
matching — the failure this codebase has repeatedly paid for.

``MaskRegion`` makes that state unrepresentable. The crop and its box are
constructed together and validated against each other, so a function taking a
region cannot be handed a box belonging to a different mask.

``frame_shape`` is carried because it is genuinely needed and not derivable
from a crop: the ortho projectors normalise pixel coordinates by the FRAME
extent, so a region that forgot its frame would project to the wrong world
position while looking entirely well-formed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskRegion:
    """A boolean mask crop plus its inclusive ``(r0, c0, r1, c1)`` in-frame box.

    Invariants, enforced at construction:

    * ``crop.shape == (r1 - r0 + 1, c1 - c0 + 1)``
    * the box lies inside ``frame_shape``

    The crop is NOT required to be tight against its box — a component carved
    out of a larger mask keeps its parent's box unless it is re-tightened —
    but the box always bounds the crop, which is what every consumer relies on.
    """

    crop: np.ndarray
    bbox: tuple[int, int, int, int]
    frame_shape: tuple[int, int]

    def __post_init__(self) -> None:
        r0, c0, r1, c1 = self.bbox
        H, W = self.frame_shape
        if not (0 <= r0 <= r1 < H and 0 <= c0 <= c1 < W):
            raise ValueError(f"bbox {self.bbox} outside frame {self.frame_shape}")
        want = (r1 - r0 + 1, c1 - c0 + 1)
        if tuple(self.crop.shape) != want:
            raise ValueError(
                f"crop shape {tuple(self.crop.shape)} does not match bbox "
                f"{self.bbox} (expected {want})"
            )

    # --- construction ---------------------------------------------------

    @classmethod
    def from_full(cls, mask: np.ndarray, *, bbox=None) -> "MaskRegion | None":
        """Crop a frame-sized mask down to its own extent.

        ``None`` when nothing is set — an empty mask is not a region, and
        callers drop it rather than carry a degenerate one. Pass ``bbox`` when
        the caller already knows it; it is validated like any other.
        """
        m = np.asarray(mask, dtype=bool)
        H, W = int(m.shape[0]), int(m.shape[1])
        if bbox is None:
            rows = m.any(axis=1)
            if not rows.any():
                return None
            cols = m.any(axis=0)
            r0 = int(np.argmax(rows))
            r1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
            c0 = int(np.argmax(cols))
            c1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
        else:
            r0, c0, r1, c1 = (int(v) for v in bbox)
        return cls(m[r0 : r1 + 1, c0 : c1 + 1], (r0, c0, r1, c1), (H, W))

    def with_crop(self, crop: np.ndarray) -> "MaskRegion":
        """A region over the same box carrying a different selection.

        For per-component masks, which live in the parent's box.
        """
        return MaskRegion(crop, self.bbox, self.frame_shape)

    def tightened(self) -> "MaskRegion | None":
        """The same pixels, re-boxed to their own extent."""
        rows = self.crop.any(axis=1)
        if not rows.any():
            return None
        cols = self.crop.any(axis=0)
        r0, c0, _r1, _c1 = self.bbox
        dr0 = int(np.argmax(rows))
        dr1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
        dc0 = int(np.argmax(cols))
        dc1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
        return MaskRegion(
            self.crop[dr0 : dr1 + 1, dc0 : dc1 + 1],
            (r0 + dr0, c0 + dc0, r0 + dr1, c0 + dc1),
            self.frame_shape,
        )

    # --- reading --------------------------------------------------------

    @property
    def area(self) -> int:
        return int(self.crop.sum())

    def window(self, arr: np.ndarray) -> np.ndarray:
        """The view of a frame-sized array under this region's box.

        Read or write through it: ``region.window(coverage)[region.crop] += 1``
        touches only the region, where a frame-sized boolean index would walk
        every pixel.
        """
        r0, c0, r1, c1 = self.bbox
        return arr[r0 : r1 + 1, c0 : c1 + 1]

    def rows_cols(self) -> tuple[np.ndarray, np.ndarray]:
        """Absolute ``(rows, cols)`` of the set pixels, row-major.

        The same sequence ``np.where`` over the frame-sized mask returns, so
        aggregates taken over it are unchanged.
        """
        rr, cc = np.where(self.crop)
        r0, c0, _r1, _c1 = self.bbox
        return rr + r0, cc + c0


__all__ = ["MaskRegion"]
