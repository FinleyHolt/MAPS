"""GeoNav's schematic cognitive map, rendered as released.

``gsamllavanav/maps/landmark_nav_map.py`` (+ ``landmark_map.py``,
``tracking_map.py``, the colour half of ``gsam_map.py``) of
github.com/Xhtshr/geonav-official: a north-up raster over the block at
480/410 px per metre carrying the landmark contours (gray ramp), the explored
view squares, the target and surroundings colour layers, and a matplotlib
10x10 in figure that stacks them with name labels, the dashed trajectory,
the heading arrow and a legend. The 1000x1000 canvas is what the MLLM sees.
"""

from __future__ import annotations

import math

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import rgb_to_hsv  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from baselines.seam import Landmark  # noqa: E402

# gsamllavanav/parser.py: map_size 480 over map_meters 410.
PX_PER_M = 480.0 / 410.0

_TARGET_COLORS = [(255, 0, 0), (0, 100, 0), (210, 105, 30), (32, 32, 32)]
_SURROUNDINGS_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
]


def rgb_entropy(rgb_img: np.ndarray) -> float:
    """``agent.rgb_entropy``: the sum over H, S, V of the 256-bin histogram
    entropy. substitution: matplotlib's rgb_to_hsv for scikit-image's."""
    arr = np.asarray(rgb_img)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(np.clip(arr, 0.0, 1.0))
    entropy = 0.0
    for i in range(3):
        channel = (hsv[..., i] * 255).astype(np.uint8)
        hist = np.bincount(channel.ravel(), minlength=256).astype(np.float64)
        hist = hist / hist.sum() + 1e-10
        entropy += float(-np.sum(hist * np.log2(hist)))
    return entropy


class ColorLayer:
    """One ``GSamMap`` colour layer: phrase -> RGBA, painted by maximum."""

    def __init__(self, shape: tuple[int, int], palette: list[tuple[int, int, int]]) -> None:
        self.palette = palette
        self.semantic_colors: dict[str, tuple[int, int, int, int]] = {}
        self.color_array = np.zeros((*shape, 4), dtype=np.uint8)

    def color_for(self, phrase: str) -> tuple[int, int, int, int]:
        phrase = phrase.lower()
        if phrase not in self.semantic_colors:
            used = set(self.semantic_colors.values())
            for c in self.palette:
                if (*c, 255) not in used:
                    self.semantic_colors[phrase] = (*c, 255)
                    break
            else:
                golden = (5 ** 0.5 - 1) / 2
                hue = (len(self.semantic_colors) * golden) % 1.0
                hsv = np.array([[[hue * 180, 0.8 * 255, 0.9 * 255]]], dtype=np.float32)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0, ::-1]
                self.semantic_colors[phrase] = (*[int(c) for c in rgb], 255)
        return self.semantic_colors[phrase]

    def paint(self, phrase: str, pts_rc: np.ndarray) -> None:
        """Fill a polygon of (row, col) points with the phrase's colour;
        the layer keeps the per-channel maximum, as released."""
        if not phrase:
            return
        temp = np.zeros_like(self.color_array)
        cv2.fillPoly(temp, [pts_rc[:, ::-1].astype(np.int32)], color=self.color_for(phrase))
        self.color_array = np.maximum(self.color_array, temp)


class CognitiveMap:
    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        landmarks: list[Landmark],
        target_name: str,
    ) -> None:
        self.x_min, self.y_min, self.x_max, self.y_max = (float(b) for b in bounds)
        self.ppm = PX_PER_M
        self.shape = (
            max(2, int(round((self.y_max - self.y_min) * self.ppm))),
            max(2, int(round((self.x_max - self.x_min) * self.ppm))),
        )
        self.landmarks = list(landmarks)
        self.gray_colors = {lm.name: (200 - i * 30,) * 3 for i, lm in enumerate(self.landmarks)}
        self.landmark_color = np.zeros((*self.shape, 4), dtype=np.uint8)
        for lm, color in zip(self.landmarks, self.gray_colors.values()):
            rows, cols = self.to_rows_cols(lm.contour)
            pts = np.stack([cols, rows]).T.astype(np.int32)
            cv2.polylines(self.landmark_color, [pts], isClosed=True, color=(*color, 255), thickness=2)
        self.explored_area = np.zeros(self.shape, dtype=np.uint8)
        self.target = ColorLayer(self.shape, _TARGET_COLORS)
        self.surroundings = ColorLayer(self.shape, _SURROUNDINGS_COLORS)
        self.target_name = target_name.lower()
        self.trajectory: list[tuple[float, float]] = []

    # --- geometry (gsamllavanav/maps/map.py) ---------------------------------

    def to_row_col(self, xy) -> tuple[int, int]:
        return (int(round((self.y_max - xy[1]) * self.ppm)), int(round((xy[0] - self.x_min) * self.ppm)))

    def to_rows_cols(self, xys) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(xys, dtype=np.float64).reshape(-1, 2)
        cols = np.round((arr[:, 0] - self.x_min) * self.ppm).astype(int)
        rows = np.round((self.y_max - arr[:, 1]) * self.ppm).astype(int)
        return rows, cols

    # --- updates ---------------------------------------------------------------

    def add_pose(self, xy, half_m: float) -> None:
        """One perceive: the trajectory point and the view square
        (``TrackingMap.mark_current_view_area``, whose half side z - ground
        is upstream's 90-degree square; ``half_m`` is the sensor's own half
        footprint)."""
        self.trajectory.append((float(xy[0]), float(xy[1])))
        half = float(half_m)
        corners = [(xy[0] - half, xy[1] + half), (xy[0] + half, xy[1] + half),
                   (xy[0] + half, xy[1] - half), (xy[0] - half, xy[1] - half)]
        rows, cols = self.to_rows_cols(corners)
        view = cv2.fillConvexPoly(np.zeros(self.shape, dtype=np.uint8), np.stack([cols, rows]).T.astype(np.int32), color=1)
        self.explored_area = np.maximum(self.explored_area, view)

    def paint_detection(self, phrase: str, world_polygon) -> None:
        """A detection onto its layer by phrase. substitution: the released
        map paints every mask pixel's ground point; here the detection's
        frame-bbox corners, unprojected, are filled as one polygon."""
        rows, cols = self.to_rows_cols(world_polygon)
        pts = np.stack([rows, cols]).T
        layer = self.target if phrase.lower() == self.target_name else self.surroundings
        layer.paint(phrase, pts)

    def semantic_map_array(self) -> np.ndarray:
        """``get_semantic_map``: the max-blend of the two colour layers."""
        return np.maximum(self.surroundings.color_array[..., :3], self.target.color_array[..., :3]).astype(np.uint8)

    # --- the render (landmark_nav_map.plot) -----------------------------------

    def render(self, map_type: str, object_nodes: list[tuple[str, tuple[float, float], bool]] | None = None) -> np.ndarray:
        """``map_type`` is ``landmark`` (Navigate) or ``semantic`` (Search).
        ``object_nodes`` are ``(id, xy, is_target)`` within 500 m, drawn on
        the semantic map. Returns the 1000x1000 RGB canvas."""
        fig = plt.figure(figsize=(10, 10), facecolor="white")
        ax = fig.add_subplot(111)
        if map_type != "landmark":
            # released: the layer drawn under the 'current view area' key is
            # to_array()[1], the accumulated EXPLORED area.
            rgba = (0.5, 0, 0.5, 0.3)
            color_map = np.zeros((*self.shape, 4), dtype=np.float32)
            color_map[..., :3] = rgba[:3]
            color_map[..., 3] = self.explored_area * rgba[3]
            ax.imshow(color_map, alpha=0.5)
        layers = [(self.landmark_color, 1.0)]
        if map_type != "landmark":
            layers += [(self.surroundings.color_array, 0.75), (self.target.color_array, 0.75)]
        for arr, alpha in layers:
            ax.imshow(arr, alpha=alpha)
        for lm in self.landmarks:
            r, c = self.to_row_col(lm.centroid)
            ax.text(c, r, lm.name, fontsize=10, color="black", ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"))
        arrow_length = 15
        if len(self.trajectory) == 1:
            r, c = self.to_row_col(self.trajectory[0])
            # released: the first pose's arrow points along a fixed yaw of pi/2.
            yaw = math.pi / 2
            dx = arrow_length * np.cos(yaw)
            dy = -arrow_length * np.sin(yaw)
            # released: dx and dy are swapped into the arrow call.
            ax.arrow(x=c, y=r, dx=dy, dy=dx, width=3, head_width=8, head_length=10, fc="red", ec="yellow")
        elif len(self.trajectory) > 1:
            pts = [self.to_row_col((x, y))[::-1] for x, y in self.trajectory]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="black", linestyle="--", linewidth=1.5, alpha=0.3)
            (px, py), (cx, cy) = pts[-2], pts[-1]
            theta = np.arctan2(cy - py, cx - px)
            ax.arrow(x=cx, y=cy, dx=arrow_length * np.cos(theta), dy=arrow_length * np.sin(theta),
                     width=3, head_width=8, head_length=10, fc="red", ec="yellow")
        if map_type != "landmark":
            for node_id, xy, is_target in object_nodes or []:
                r, c = self.to_row_col(xy)
                ax.text(c, r, f"{node_id}", fontsize=10, color="red" if is_target else "black",
                        ha="center", va="center", bbox=dict(facecolor="white", alpha=0.5, boxstyle="round"))
            elements = [
                Patch(facecolor=np.clip(np.array(color) / 255.0, 0, 1), edgecolor="gray", label=name,
                      linestyle="-", linewidth=2)
                for name, color in self.gray_colors.items()
            ]
            for phrase, color in {**self.target.semantic_colors, **self.surroundings.semantic_colors}.items():
                elements.append(Patch(facecolor=np.clip(np.array(color[:3]) / 255.0, 0, 1), label=phrase.capitalize()))
            if elements:
                ax.legend(handles=elements, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return rgb
