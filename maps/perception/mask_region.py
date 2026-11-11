"""A detection mask as a crop plus its box in the frame, built and validated
together so the two cannot be mispaired. Work scales with the object, not the
sensor. ``frame_shape`` rides along because the projectors normalise pixel
coordinates by the frame's extent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maps.perception.mask_codec import extent


@dataclass(frozen=True)
class MaskRegion:
    """A boolean mask crop plus its inclusive ``(r0, c0, r1, c1)`` box, which
    lies inside ``frame_shape`` and matches the crop's shape (checked at
    construction). The crop need not be tight: a component keeps its parent's
    box until re-tightened."""

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

    def with_crop(self, crop: np.ndarray) -> "MaskRegion":
        """A region over the same box carrying another selection (a component)."""
        return MaskRegion(crop, self.bbox, self.frame_shape)

    def tightened(self) -> "MaskRegion | None":
        """The same pixels, re-boxed to their own extent."""
        ext = extent(self.crop)
        if ext is None:
            return None
        dr0, dc0, dr1, dc1 = ext
        r0, c0, _r1, _c1 = self.bbox
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
        """The view of a frame-sized array under this region's box, to read or
        write through (``region.window(a)[region.crop] += 1``)."""
        r0, c0, r1, c1 = self.bbox
        return arr[r0 : r1 + 1, c0 : c1 + 1]

    def rows_cols(self) -> tuple[np.ndarray, np.ndarray]:
        """Absolute ``(rows, cols)`` of the set pixels, row-major, as
        ``np.where`` over the frame-sized mask returns them."""
        rr, cc = np.where(self.crop)
        r0, c0, _r1, _c1 = self.bbox
        return rr + r0, cc + c0
