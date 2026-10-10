"""The edge-structure matcher the frames are registered with.

A frame's edge structure (:func:`edge_maps`: the gradient normalised against
the frame's own noise floor and squared into its structure tensor) forgets
brightness, contrast and the sign of every edge, so a camera frame and the
map's render can be correlated without either one's intensities meeting the
other's. :func:`locate_edges` finds where a template of one structure sits in
the other, and :func:`_peak` says how unique the peak is.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

#: The template and search window, in work pixels: the matcher searches
#: ``SEARCH_HALF - TEMPLATE_HALF`` of displacement about the prediction, and a
#: prediction nearer the border than ``SEARCH_HALF`` is out.
TEMPLATE_HALF = 20
SEARCH_HALF = 80
#: The edge template is wider, a junction's structure at altitude being a few
#: road widths; the maps are padded by the difference.
EDGE_TEMPLATE_HALF = 96
#: The maps' smoothing before and after the gradient, px, and the gradient
#: percentile of the frame's own that is its noise floor.
EDGE_SIGMA = 1.0
EDGE_CLIP_PCT = 50
#: A peak is unique when nothing this far from it scores as well: a road with
#: no cross street scores the same all along its own ridge.
UNIQUE_RADIUS = 16
#: A tie's gate: the peak's margin over the best score ``UNIQUE_RADIUS`` away,
#: set so that at most one percent of a null rung (each frame against the
#: render of the frame three on) passes.
MIN_EDGE_MARGIN = 0.082
#: The ground a pixel covers where the windows above were fitted, m/px. A frame
#: finer than it by more than ``BIN_ABOVE`` is binned to it before it is
#: searched (:func:`work_sensor`), so every window means the same ground.
WORK_GSD_M = 0.261
BIN_ABOVE = 1.25
#: Correlations run this many at a time: the operators release the interpreter.
LOCATE_THREADS = 4
_pool = ThreadPoolExecutor(max_workers=LOCATE_THREADS, thread_name_prefix="locate")


def frame_gsd_m(sensor, agl_m: float) -> float:
    """The ground a frame's pixel covers at this height, m/px, at the optical axis."""
    return float(agl_m) / float(sensor.intrinsics.fx)


def sized(sensor, width: int, height: int):
    """``sensor`` at ``width`` x ``height`` pixels, the same fields of view."""
    from maps.perception.ortho.camera_model import CameraIntrinsics  # noqa: PLC0415

    intr = sensor.intrinsics
    if (intr.image_width, intr.image_height) == (width, height):
        return sensor
    return replace(sensor, intrinsics=CameraIntrinsics(hfov_deg=intr.hfov_deg, vfov_deg=intr.vfov_deg,
                                                       image_width=width, image_height=height))


def work_sensor(sensor, agl_m: float):
    """``(sensor, scale)``: the sensor binned to :data:`WORK_GSD_M`, and the
    frame's pixels per binned pixel; the sensor itself and 1.0 for a frame no
    finer than that."""
    gsd = frame_gsd_m(sensor, agl_m)
    if gsd <= 0.0 or gsd * BIN_ABOVE >= WORK_GSD_M:
        return sensor, 1.0
    intr = sensor.intrinsics
    w = max(2 * SEARCH_HALF + 1, int(round(intr.image_width * gsd / WORK_GSD_M)))
    h = max(2 * SEARCH_HALF + 1, int(round(intr.image_height * gsd / WORK_GSD_M)))
    return sized(sensor, w, h), w / intr.image_width


def gray(rgb: np.ndarray) -> np.ndarray:
    """Luma of an RGB frame; a frame already gray comes back as it is."""
    import cv2  # noqa: PLC0415

    a = np.asarray(rgb)
    if a.ndim == 2:
        return a
    return cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_RGB2GRAY)


def _peak(score: np.ndarray, radius: int = UNIQUE_RADIUS) -> tuple[tuple[int, int], float, float]:
    """``(at, peak, margin)``: the best score and by how much it beats everything ``radius`` away."""
    import cv2  # noqa: PLC0415

    _min, peak, _at_min, at = cv2.minMaxLoc(score)
    side = np.ones(score.shape, dtype=np.uint8)
    cv2.circle(side, at, radius, 0, -1)
    rest = score[side > 0]
    if rest.size < 2:
        return (int(at[0]), int(at[1])), float(peak), float("inf")
    return (int(at[0]), int(at[1])), float(peak), float(peak - rest.max())


def edge_maps(gray: np.ndarray, *, template_half: int = EDGE_TEMPLATE_HALF) -> np.ndarray:
    """The frame's structure tensor as three maps, smoothed and padded for the
    template, float32 ``(3, H + 2 pad, W + 2 pad)``."""
    import cv2  # noqa: PLC0415

    g = cv2.GaussianBlur(np.asarray(gray, dtype=np.float32), (0, 0), EDGE_SIGMA)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag2 = gx * gx + gy * gy
    scale = max(float(np.sqrt(np.percentile(mag2, EDGE_CLIP_PCT))), 1e-6)
    norm = np.sqrt(mag2 + scale * scale)
    nx, ny = gx / norm, gy / norm
    channels = [nx * nx, ny * ny, np.sqrt(2.0) * nx * ny]
    pad = template_half - TEMPLATE_HALF
    out = np.empty((len(channels), gray.shape[0] + 2 * pad, gray.shape[1] + 2 * pad), dtype=np.float32)
    for c, dst in zip(channels, out):
        cv2.copyMakeBorder(cv2.GaussianBlur(c.astype(np.float32), (0, 0), EDGE_SIGMA), pad, pad, pad, pad,
                           cv2.BORDER_CONSTANT, dst=dst, value=0.0)
    return out


def _joint_ncc(window: np.ndarray, template: np.ndarray) -> np.ndarray | None:
    """Normalised cross-correlation of a stack of maps, the mean and the norm
    taken over the whole stack: the cosine between the two stacks at every
    window position."""
    import cv2  # noqa: PLC0415

    t = template.astype(np.float32) - float(template.mean())
    t_norm = float(np.sqrt((t * t).sum()))
    if t_norm < 1e-6:
        return None
    k = template.shape[1:]
    n = float(template.size)
    num = sum(cv2.matchTemplate(np.ascontiguousarray(w), np.ascontiguousarray(tc), cv2.TM_CCORR)
              for w, tc in zip(window, t))
    box = lambda a: cv2.boxFilter(np.ascontiguousarray(a), -1, (k[1], k[0]), normalize=False,  # noqa: E731
                                  anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)[:window.shape[1] - k[0] + 1,
                                                                                    :window.shape[2] - k[1] + 1]
    w_sum = sum(box(w) for w in window)
    w_sq = sum(box(w * w) for w in window)
    w_norm = np.sqrt(np.maximum(w_sq - w_sum * w_sum / n, 0.0))
    return (num / np.maximum(t_norm * w_norm, 1e-6)).astype(np.float32)


def locate_edges(edges_actual: np.ndarray, edges_expected: np.ndarray, predicted: tuple[float, float]
                 ) -> tuple[tuple[float, float], float, float] | None:
    """Where the edge structure at ``predicted`` in the expected frame sits in
    the actual one: each map's normalised correlation averaged, so an edge
    counts where it is and which way it runs. ``((u, v), peak, margin)``, or
    None off the frame or over no structure."""
    import cv2  # noqa: PLC0415

    pad = EDGE_TEMPLATE_HALF - TEMPLATE_HALF
    u, v = int(round(predicted[0])), int(round(predicted[1]))
    h, w = edges_actual.shape[1] - 2 * pad, edges_actual.shape[2] - 2 * pad
    if not (SEARCH_HALF <= u < w - SEARCH_HALF and SEARCH_HALF <= v < h - SEARCH_HALF):
        return None
    u, v = u + pad, v + pad
    reach = SEARCH_HALF - TEMPLATE_HALF
    win = EDGE_TEMPLATE_HALF + reach
    template = edges_expected[:, v - EDGE_TEMPLATE_HALF:v + EDGE_TEMPLATE_HALF + 1,
                              u - EDGE_TEMPLATE_HALF:u + EDGE_TEMPLATE_HALF + 1]
    window = edges_actual[:, v - win:v + win + 1, u - win:u + win + 1]
    score = None
    n = 0
    for actual, expected in zip(window, template):
        if expected.std() < 1e-6:
            continue
        s = cv2.matchTemplate(np.ascontiguousarray(actual), np.ascontiguousarray(expected), cv2.TM_CCOEFF_NORMED)
        score = s if score is None else score + s
        n += 1
    if score is None:
        return None
    score /= n
    at, peak, margin = _peak(score)
    return (predicted[0] + at[0] - reach, predicted[1] + at[1] - reach), peak, margin
