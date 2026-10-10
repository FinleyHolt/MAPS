"""What this run has captured, as the ortho a look crops.

THE PRIOR IS OSM; THE IMAGERY IS THE RUN'S. Every picture the model is given
in flight is this run's own capture: a waypoint's observation is the pass
fused aboard, and ``look`` crops the imagery those passes laid down, never
the bake's raster, which is the bench's camera. Ground the run has not
imaged is black and captioned unobserved.

Each waypoint's observation (``LiveSource.observe_area``, one north-up
product on the map grid at the frames' own ground sample) is added as it is
perceived; a look asks for the window it wants and gets the products that
reach it pasted onto one canvas, the latest on top, in the shape
``maps.perception.observation.crop_ortho_observation`` cuts from (``rgb``,
``alpha``, ``height``, ``x_min``, ``y_min``, ``px_per_m``).
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

#: The grid a look is cut at when nothing has been captured yet.
DEFAULT_GSD_M = 0.10


class RunImagery:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        #: ``(rgb, alpha, (x0, y0, x1, y1), gsd_m)`` per product, in the order perceived.
        self._products: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]] = []

    @property
    def gsd_m(self) -> float:
        """The finest ground sample any product carries: the grid a look is cut at."""
        with self._lock:
            return min((g for *_r, g in self._products), default=DEFAULT_GSD_M)

    def add(self, obs: Any) -> bool:
        """One perceived observation, kept when it is a north-up product on
        the map grid (``ortho_view_corners_xy`` set, a ``world_window``);
        a perspective frame is not, and is left to its own path."""
        rgb = getattr(obs, "rgb", None)
        window = getattr(obs, "world_window", None)
        if rgb is None or window is None or getattr(obs, "ortho_view_corners_xy", None) is None:
            return False
        x0, y0, x1, y1 = (float(v) for v in window)
        h, w = rgb.shape[:2]
        if w < 1 or h < 1 or x1 <= x0 or y1 <= y0:
            return False
        alpha = getattr(obs, "alpha", None)
        alpha = np.full((h, w), 255, np.uint8) if alpha is None else np.asarray(alpha, np.uint8)
        with self._lock:
            self._products.append((np.ascontiguousarray(rgb[..., :3], np.uint8), alpha, (x0, y0, x1, y1),
                                   (x1 - x0) / w))
        return True

    def ortho(self, window: tuple[float, float, float, float]) -> dict[str, Any]:
        """The run's imagery over ``window`` as one north-up canvas at the
        finest product's ground sample: every product that
        reaches the window pasted where its alpha holds, the latest on top.
        Where nothing reached, alpha 0 and black -- ground this run has not
        imaged. Row 0 is the window's north edge, the layout every ortho
        crop assumes."""
        import cv2  # noqa: PLC0415

        x0, y0, x1, y1 = (float(v) for v in window)
        gsd = float(self.gsd_m)
        w = max(1, int(round((x1 - x0) / gsd)))
        h = max(1, int(round((y1 - y0) / gsd)))
        rgb = np.zeros((h, w, 3), np.uint8)
        alpha = np.zeros((h, w), np.uint8)
        with self._lock:
            products = list(self._products)
        for prgb, palpha, (px0, py0, px1, py1), pgsd in products:
            ox0, oy0, ox1, oy1 = max(x0, px0), max(y0, py0), min(x1, px1), min(y1, py1)
            if ox1 <= ox0 or oy1 <= oy0:
                continue
            # The overlap on the canvas (row 0 = y1) and on the product (row 0 = py1).
            c0, c1 = int((ox0 - x0) / gsd), int(np.ceil((ox1 - x0) / gsd))
            r0, r1 = int((y1 - oy1) / gsd), int(np.ceil((y1 - oy0) / gsd))
            c0, r0 = max(0, c0), max(0, r0)
            c1, r1 = min(w, c1), min(h, r1)
            if c1 <= c0 or r1 <= r0:
                continue
            pc0, pc1 = int((ox0 - px0) / pgsd), int(np.ceil((ox1 - px0) / pgsd))
            pr0, pr1 = int((py1 - oy1) / pgsd), int(np.ceil((py1 - oy0) / pgsd))
            pc0, pr0 = max(0, pc0), max(0, pr0)
            pc1, pr1 = min(prgb.shape[1], pc1), min(prgb.shape[0], pr1)
            if pc1 <= pc0 or pr1 <= pr0:
                continue
            size = (c1 - c0, r1 - r0)
            piece = cv2.resize(np.ascontiguousarray(prgb[pr0:pr1, pc0:pc1]), size, interpolation=cv2.INTER_LINEAR)
            held = cv2.resize(np.ascontiguousarray(palpha[pr0:pr1, pc0:pc1]), size, interpolation=cv2.INTER_NEAREST)
            on = held > 0
            rgb[r0:r1, c0:c1][on] = piece[on]
            alpha[r0:r1, c0:c1][on] = 255
        return {"rgb": rgb, "alpha": alpha, "height": np.zeros((h, w), np.float32),
                "x_min": x0, "y_min": y0, "px_per_m": 1.0 / gsd, "m_per_px": gsd}
