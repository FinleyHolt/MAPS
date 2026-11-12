"""Mark objects on a picture without covering them, for the look and map
renderers, whose tool text carries the digit -> fid legend."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

#: The JPEG quality of every image the model is sent.
JPEG_QUALITY = 85

_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX

_LOOK_BAND_BGR: tuple[int, int, int] = (28, 28, 28)  # dark margin strip
# The ring a maskless mark gets, in fixed delivered pixels: a position, never
# an extent.
_POINT_RING_PX: int = 6


def _draw_badge(canvas, placed: list, cx, cy, half_w, half_h, digit) -> None:
    """A ~24 px delivered glyph at the least crowded diagonal outside the
    mark centred at ``(cx, cy)`` with half extents ``(half_w, half_h)``, with
    a leader when it sits more than ~12 px away; its box joins ``placed``."""
    h, w = canvas.shape[:2]
    txt = str(digit)
    fs, th = 0.9, 2
    (tw, tht), _ = cv2.getTextSize(txt, _LABEL_FONT, fs, th)
    pw, ph = tw + 12, tht + 10
    cands = []
    for sx, sy in ((1, -1), (1, 1), (-1, -1), (-1, 1)):
        px = cx + sx * (half_w + 5) + (0 if sx > 0 else -pw)
        py = cy + sy * (half_h + 5) + (0 if sy > 0 else -ph)
        px = min(max(px, 1.0), w - pw - 1.0)
        py = min(max(py, 1.0), h - ph - 1.0)
        bb = (px, py, px + pw, py + ph)
        crowd = 0.0
        for ob in placed:
            ix = max(0.0, min(bb[2], ob[2]) - max(bb[0], ob[0]))
            iy = max(0.0, min(bb[3], ob[3]) - max(bb[1], ob[1]))
            crowd += ix * iy
        cands.append((crowd, bb))
    _, bb = min(cands, key=lambda c: c[0])
    x0, y0, x1, y1 = (int(round(v)) for v in bb)
    badge_cx, badge_cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    dx, dy = cx - badge_cx, cy - badge_cy
    dn = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    edge_x = cx - dx / dn * half_w
    edge_y = cy - dy / dn * half_h
    gap = ((edge_x - badge_cx) ** 2 + (edge_y - badge_cy) ** 2) ** 0.5
    if gap > 12.0:
        ends = (int(badge_cx), int(badge_cy)), (int(edge_x), int(edge_y))
        cv2.line(canvas, *ends, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(canvas, *ends, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, txt, (x0 + 6, y1 - 6), _LABEL_FONT, fs, (255, 255, 255), th, cv2.LINE_AA)
    placed.append(bb)


def _has_footprint(arcs) -> bool:
    """Whether a mark carries a footprint to outline. A footprint arrives as
    OPEN arcs (``look_render.frame_arcs``): a shape wholly in frame is one arc
    whose ends meet, and a clipped one draws no edge along the frame."""
    if not arcs:
        return False
    # a flat vertex list would draw as scatter: refuse it
    if any(
        not isinstance(a, (list, tuple)) or len(a) < 2 or not isinstance(a[0], (list, tuple))
        for a in arcs
    ):
        raise TypeError(
            "poly must be a list of OPEN arcs [[[x, y], ...], ...] "
            "(maps.query.polyline.clip_ring_to_box output), not a flat "
            f"vertex list; got {arcs[0]!r}"
        )
    return True


def _draw_outline(canvas, placed: list, arcs, thick) -> tuple[float, float, float, float]:
    """Trace ``arcs`` white over a dark halo and return the drawn extent as
    ``(centre x, centre y, half width, half height)``, the badge's anchor;
    its box joins ``placed``."""
    runs = [
        np.asarray([[int(round(float(x))), int(round(float(y)))] for x, y in arc], dtype=np.int32)
        for arc in arcs
        if len(arc) >= 2
    ]
    # A linear mark (area small against extent) is traced at 4 px at
    # least, or a 1 px neighbour stroke vanishes into the photograph.
    for run in runs:
        ext = max(
            float(run[:, 0].max() - run[:, 0].min()), float(run[:, 1].max() - run[:, 1].min())
        )
        if ext >= 8.0 and cv2.contourArea(run) / (ext * ext) < 0.06:
            thick = max(thick, 4)
            break
    cv2.polylines(canvas, runs, False, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.polylines(canvas, runs, False, (255, 255, 255), thick, cv2.LINE_AA)
    xs = np.concatenate([r[:, 0] for r in runs])
    ys = np.concatenate([r[:, 1] for r in runs])
    placed.append((float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())))
    # the badge anchors to the drawn (visible) extent
    return (
        float((xs.min() + xs.max()) / 2.0),
        float((ys.min() + ys.max()) / 2.0),
        float(xs.max() - xs.min()) / 2.0,
        float(ys.max() - ys.min()) / 2.0,
    )


def _draw_ring(canvas, placed: list, x: float, y: float) -> tuple[float, float, float, float]:
    """The ring of a mark with no footprint, and its badge anchor; its box
    joins ``placed``."""
    centre = int(round(x)), int(round(y))
    cv2.circle(canvas, centre, _POINT_RING_PX + 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(canvas, centre, _POINT_RING_PX, (255, 255, 255), 1, cv2.LINE_AA)
    placed.append((x - _POINT_RING_PX, y - _POINT_RING_PX, x + _POINT_RING_PX, y + _POINT_RING_PX))
    return (x, y, float(_POINT_RING_PX), float(_POINT_RING_PX))


def _with_band(canvas, band_text: str):
    """``canvas`` under the dark strip that carries only the orientation
    ("N up" or the rotated view's up)."""
    band = np.full((20, canvas.shape[1], 3), _LOOK_BAND_BGR, np.uint8)
    text = str(band_text or "N up")
    (text_w, _), _ = cv2.getTextSize(text, _LABEL_FONT, 0.45, 1)
    origin = (canvas.shape[1] - text_w - 6, 14)
    cv2.putText(band, text, origin, _LABEL_FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(band, text, origin, _LABEL_FONT, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([band, canvas])


def render_look_overlay(
    rgb: np.ndarray,
    *,
    center_px: tuple[float, float],
    center_poly_px: Sequence[tuple[float, float]] | None = None,
    target_px: tuple[float, float] | None = None,
    target_poly_px: Sequence[tuple[float, float]] | None = None,
    neighbor_px: Sequence[tuple[float, float]] | None = None,
    neighbor_polys_px: Sequence[Sequence[tuple[float, float]] | None] | None = None,
    band_text: str = "N up",
    draw_digits: bool = True,
    outline_marks: bool = True,
) -> bytes:
    """Mark the centre (and a target and neighbours) on ``rgb`` (H, W, 3)
    and return JPEG bytes with the orientation band on top.

    Each mark is its footprint's in-frame arcs as an achromatic outline (white
    over a dark halo, no fill; centre 3 px, target 2, neighbours 1), or a
    small ring when it has no footprint, plus a numbered badge (1 centre,
    2 target, 3.. neighbours) at the least crowded diagonal outside it.
    ``draw_digits`` False draws no badges; ``outline_marks`` False outlines
    only the centre (a schematic's fill already shows the rest).
    """
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be a (H, W, 3) ndarray")
    if rgb.dtype != np.uint8:
        rgb_u8 = (
            np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            if np.issubdtype(rgb.dtype, np.floating)
            else rgb.astype(np.uint8)
        )
    else:
        rgb_u8 = rgb
    canvas = cv2.cvtColor(np.ascontiguousarray(rgb_u8), cv2.COLOR_RGB2BGR)

    # The marks in badge order: ``(x, y, footprint arcs, outline px, digit)``.
    # Digit 2 is the target's whether or not there is one.
    marks: list[tuple] = [(float(center_px[0]), float(center_px[1]), center_poly_px, 3, 1)]
    if target_px is not None:
        marks.append((float(target_px[0]), float(target_px[1]), target_poly_px, 2, 2))
    neighbor_polys = list(neighbor_polys_px or [])
    digit = 3
    for i, nb in enumerate(neighbor_px or []):
        if isinstance(nb, (tuple, list)) and len(nb) == 2:
            poly = neighbor_polys[i] if i < len(neighbor_polys) else None
            marks.append((float(nb[0]), float(nb[1]), poly, 1, digit))
            digit += 1

    # Lowest priority first so the centre lands on top; badges after every
    # outline so their placement sees all of them.
    placed: list[tuple[float, float, float, float]] = []  # occupied px boxes
    badges: list[tuple] = []
    for x, y, poly, thick, digit in reversed(marks):
        outlined = outline_marks or digit == 1
        if _has_footprint(poly) and outlined:
            anchor = _draw_outline(canvas, placed, poly, thick)
        elif not outlined:
            anchor = (x, y, 0.0, 0.0)
        else:
            anchor = _draw_ring(canvas, placed, x, y)
        badges.append((anchor, digit))
    if draw_digits:
        for anchor, digit in badges:
            _draw_badge(canvas, placed, *anchor, digit)

    ok, buf = cv2.imencode(
        ".jpg", _with_band(canvas, band_text), [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed on look overlay")
    return bytes(buf.tobytes())
