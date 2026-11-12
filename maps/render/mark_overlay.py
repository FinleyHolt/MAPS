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
    h, w = canvas.shape[:2]

    placed: list[tuple[float, float, float, float]] = []  # occupied px boxes

    def _badge(cx_, cy_, ax_, ay_, digit):
        # A ~24 px delivered glyph at the least crowded diagonal outside
        # the mark, with a leader when it sits more than ~12 px away.
        txt = str(digit)
        fs, th = 0.9, 2
        (tw, tht), _ = cv2.getTextSize(txt, _LABEL_FONT, fs, th)
        pw, ph = tw + 12, tht + 10
        cands = []
        for sx_, sy_ in ((1, -1), (1, 1), (-1, -1), (-1, 1)):
            px_ = cx_ + sx_ * (ax_ + 5) + (0 if sx_ > 0 else -pw)
            py_ = cy_ + sy_ * (ay_ + 5) + (0 if sy_ > 0 else -ph)
            px_ = min(max(px_, 1.0), w - pw - 1.0)
            py_ = min(max(py_, 1.0), h - ph - 1.0)
            bb = (px_, py_, px_ + pw, py_ + ph)
            crowd = 0.0
            for ob in placed:
                ix = max(0.0, min(bb[2], ob[2]) - max(bb[0], ob[0]))
                iy = max(0.0, min(bb[3], ob[3]) - max(bb[1], ob[1]))
                crowd += ix * iy
            cands.append((crowd, bb))
        _, bb = min(cands, key=lambda c: c[0])
        x0_, y0_, x1_, y1_ = (int(round(v)) for v in bb)
        bcx_, bcy_ = (x0_ + x1_) / 2.0, (y0_ + y1_) / 2.0
        dx_, dy_ = cx_ - bcx_, cy_ - bcy_
        dn_ = max((dx_ * dx_ + dy_ * dy_) ** 0.5, 1e-6)
        ex_ = cx_ - dx_ / dn_ * ax_
        ey_ = cy_ - dy_ / dn_ * ay_
        gap_ = ((ex_ - bcx_) ** 2 + (ey_ - bcy_) ** 2) ** 0.5
        if gap_ > 12.0:
            cv2.line(
                canvas, (int(bcx_), int(bcy_)), (int(ex_), int(ey_)), (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.line(
                canvas,
                (int(bcx_), int(bcy_)),
                (int(ex_), int(ey_)),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.rectangle(canvas, (x0_, y0_), (x1_, y1_), (0, 0, 0), -1)
        cv2.rectangle(canvas, (x0_, y0_), (x1_, y1_), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            canvas, txt, (x0_ + 6, y1_ - 6), _LABEL_FONT, fs, (255, 255, 255), th, cv2.LINE_AA
        )
        placed.append(bb)

    # A footprint arrives as OPEN arcs (``look_render.frame_arcs``): a shape
    # wholly in frame is one arc whose ends meet, and a clipped one draws no
    # edge along the frame.
    def _has_poly(arcs):
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

    def _poly_outline(arcs, thick):
        runs = [
            np.asarray(
                [[int(round(float(a_))), int(round(float(b_)))] for a_, b_ in arc], dtype=np.int32
            )
            for arc in arcs
            if len(arc) >= 2
        ]
        # A linear mark (area small against extent) is traced at 4 px at
        # least, or a 1 px neighbour stroke vanishes into the photograph.
        _lin = False
        for r_ in runs:
            xs_, ys_ = r_[:, 0], r_[:, 1]
            ext = max(float(xs_.max() - xs_.min()), float(ys_.max() - ys_.min()))
            if ext < 8.0:
                continue
            area = cv2.contourArea(r_)
            if area / (ext * ext) < 0.06:
                _lin = True
                break
        if _lin:
            thick = max(thick, 4)
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

    cx, cy = float(center_px[0]), float(center_px[1])
    specs: list[dict] = [
        {"x": cx, "y": cy, "poly": center_poly_px, "thick": 3, "digit": 1 if draw_digits else None}
    ]
    if target_px is not None:
        specs.append(
            {
                "x": float(target_px[0]),
                "y": float(target_px[1]),
                "poly": target_poly_px,
                "thick": 2,
                "digit": 2 if draw_digits else None,
            }
        )
    _nb_polys = list(neighbor_polys_px or [])
    _digit = 3
    for _i, nb in enumerate(neighbor_px or []):
        if isinstance(nb, (tuple, list)) and len(nb) == 2:
            specs.append(
                {
                    "x": float(nb[0]),
                    "y": float(nb[1]),
                    "poly": _nb_polys[_i] if _i < len(_nb_polys) else None,
                    "thick": 1,
                    "digit": _digit if draw_digits else None,
                }
            )
            _digit += 1
    # lowest priority first so the centre lands on top; badges after every
    # outline so their placement sees all of them
    for sp in reversed(specs):
        _skip_outline = (not outline_marks) and sp is not specs[0]
        if _has_poly(sp["poly"]) and not _skip_outline:
            sp["anchor"] = _poly_outline(sp["poly"], sp["thick"])
        elif _skip_outline:
            sp["anchor"] = (sp["x"], sp["y"], 0.0, 0.0)
        else:
            _rx, _ry = int(round(sp["x"])), int(round(sp["y"]))
            cv2.circle(canvas, (_rx, _ry), _POINT_RING_PX + 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, (_rx, _ry), _POINT_RING_PX, (255, 255, 255), 1, cv2.LINE_AA)
            placed.append(
                (
                    sp["x"] - _POINT_RING_PX,
                    sp["y"] - _POINT_RING_PX,
                    sp["x"] + _POINT_RING_PX,
                    sp["y"] + _POINT_RING_PX,
                )
            )
            sp["anchor"] = (sp["x"], sp["y"], float(_POINT_RING_PX), float(_POINT_RING_PX))
    for sp in reversed(specs):
        if sp["digit"] is None:
            continue
        ax_, ay_ = sp["anchor"][2], sp["anchor"][3]
        _badge(sp["anchor"][0], sp["anchor"][1], ax_, ay_, sp["digit"])

    # the band carries only the orientation ("N up" or the rotated view's up)
    band_h = 20
    band = np.full((band_h, w, 3), _LOOK_BAND_BGR, np.uint8)
    _bt = str(band_text or "N up")
    (nw, _), _ = cv2.getTextSize(_bt, _LABEL_FONT, 0.45, 1)
    cv2.putText(band, _bt, (w - nw - 6, 14), _LABEL_FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(band, _bt, (w - nw - 6, 14), _LABEL_FONT, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    out = np.vstack([band, canvas])

    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed on look overlay")
    return bytes(buf.tobytes())
