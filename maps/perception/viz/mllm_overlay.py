"""Look-crop overlay: mark objects on a photograph without covering them.

``render_look_overlay`` draws each marked object's recorded footprint as an
achromatic outline (white stroke over a dark halo, no fill, so the colour the
reasoner reads stays untouched), binds a numbered badge to every mark, adds
optional named-place labels, and returns JPEG bytes. The look and map
renderers (``maps/search/render``) are its callers; the digit -> fid legend
lives in their tool text.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

__all__ = [
    "render_look_overlay",
]


_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX

_LOOK_BAND_BGR: tuple[int, int, int] = (28, 28, 28)  # dark margin strip
# The ring a MASKLESS mark gets, so its badge points at something instead of
# at bare ground. Frozen in delivered pixels like the badge glyph: a mark
# with no footprint is a position, and the ring must never read as extent.
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
    place_labels: "Sequence[tuple[float, float, str]] | None" = None,
    neighbor_digits: "Sequence[int | None] | None" = None,
) -> bytes:
    """Identify the center fid car (and optionally a target / neighbors) on a
    look crop WITHOUT occluding any car's pixels, and return JPEG bytes.

    The agent is shown the LASSO around the object's actual segmentation
    mask, never a proxy shape (badge + legend grounding: OmniParser
    2408.00203 digit-badge + text legend +23.3pp; STVG-R1 2602.11730
    digits >> mixed ids; 2604.12371 Qwen glyph floor ~10-12 px):
    - TRUE MASK OUTLINE, achromatic (white stroke + dark halo), NO fill —
      the mark IS the object's recorded footprint polygon (the fine
      SAM-derived world_contour where one exists, else the mask's boundary
      loop as ingested), clipped to the frame and drawn as open arcs so a
      shape larger than its window shows the part you can see and nothing
      along the cut. Center = 3 px, target = 2 px, neighbors = 1 px.
    - A mark whose object carries NO mask polygon gets its numbered badge,
      a leader line, and a small fixed-radius RING at the point — no
      outline. The absence of a footprint stays visibly absent (the ring
      is frozen in delivered px and scales with nothing, so it states a
      position and never an extent), but the badge points at something
      rather than at bare ground.
    - Every marked object gets a NUMBERED badge (white numeral on a black
      pill, glyph ~14 px, screen-upright; the pill sizes from getTextSize,
      so two-digit numerals widen it — drawn to 25 badges in the SoM probe)
      placed at the least-crowded of 4 diagonal offsets OUTSIDE its mark —
      numerals are 1 = center, 2 = target, 3.. = neighbors in the
      caller's order. The numeral -> full fid binding lives in the TOOL
      TEXT legend (the caller mirrors the same ordering), not on the image.
    - The margin band carries ONLY the tiny "N up" strip; all naming lives
      in the tool-text legend.
    ``rgb`` is (H, W, 3) uint8 RGB; the returned JPEG is BGR-encoded.
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

    # Marks: mask outlines + digit badges (see docstring). Draw neighbors
    # first, then target, then center on top (most salient); badges placed
    # with collision avoidance against everything already drawn.
    placed: list[tuple[float, float, float, float]] = []  # occupied px boxes

    def _badge(cx_, cy_, ax_, ay_, digit):
        # Single-digit pill at the least-crowded of 4 diagonal offsets
        # OUTSIDE the mark, screen-upright. Glyph ~24 px DELIVERED
        # (~one Qwen 28-32 px grid cell — the measured character floor;
        # frozen absolute, independent of crop upscale). A badge whose
        # gap to its mark exceeds ~12 px gets a LEADER LINE so the
        # binding is geometric, not proximal (VP-Bench arrows 82.6%).
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
        # leader line first (under the pill): badge center -> mark edge
        bcx_, bcy_ = (x0_ + x1_) / 2.0, (y0_ + y1_) / 2.0
        dx_, dy_ = cx_ - bcx_, cy_ - bcy_
        dn_ = max((dx_ * dx_ + dy_ * dy_) ** 0.5, 1e-6)
        # point on the mark's extent boundary toward the badge
        ex_ = cx_ - dx_ / dn_ * ax_
        ey_ = cy_ - dy_ / dn_ * ay_
        gap_ = ((ex_ - bcx_) ** 2 + (ey_ - bcy_) ** 2) ** 0.5
        if gap_ > 12.0:
            cv2.line(canvas, (int(bcx_), int(bcy_)), (int(ex_), int(ey_)),
                     (0, 0, 0), 3, cv2.LINE_AA)
            cv2.line(canvas, (int(bcx_), int(bcy_)), (int(ex_), int(ey_)),
                     (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0_, y0_), (x1_, y1_), (0, 0, 0), -1)
        cv2.rectangle(canvas, (x0_, y0_), (x1_, y1_), (255, 255, 255), 1,
                      cv2.LINE_AA)
        cv2.putText(canvas, txt, (x0_ + 6, y1_ - 6), _LABEL_FONT, fs,
                    (255, 255, 255), th, cv2.LINE_AA)
        placed.append(bb)

    # MARK GEOMETRY: the mark IS the object's
    # recorded mask polygon — ALWAYS, however coarse. A stair-stepped 2 m
    # grid loop is the honest realisation of what the system knows about
    # that footprint; a proxy shape (circle/ellipse/right-sized square)
    # asserts geometry the mask does not carry, so none is drawn. A mark with
    # NO polygon draws no outline: its numbered badge + leader line binds
    # the digit to the point, and the missing footprint stays visibly
    # missing. Salience: center 3 px > target 2 > nbr 1.
    # A mark's footprint arrives as a list of OPEN arcs — the parts of its
    # ring that lie in frame (``look_render._proj_poly`` clips it). A shape
    # wholly inside is one arc whose ends coincide, so drawing every arc
    # open reproduces a closed ring with no special case. Open is the whole
    # point: joining two arcs across a gap would draw an edge along the
    # frame edge, which reads as "the shape ends here" and is a fact about
    # the crop, not about the world.
    def _has_poly(arcs):
        if not arcs:
            return False
        # A FLAT vertex list fails open: each (x, y) has len 2, so it reads
        # as a two-point arc and draws as scatter. Both renderers clip to
        # arcs; anything else is a caller bug, and it says so.
        if any(not isinstance(a, (list, tuple)) or len(a) < 2
               or not isinstance(a[0], (list, tuple)) for a in arcs):
            raise TypeError(
                "poly must be a list of OPEN arcs [[[x, y], ...], ...] "
                "(maps.util.polyline.clip_ring_to_box output), not a flat "
                f"vertex list; got {arcs[0]!r}")
        return True

    def _poly_outline(arcs, thick):
        runs = [np.asarray([[int(round(float(a_))), int(round(float(b_)))]
                            for a_, b_ in arc], dtype=np.int32)
                for arc in arcs if len(arc) >= 2]
        # A LINEAR mark is not lassoed, it is TRACED. A road arrives as a
        # centreline -- a polyline with no enclosed area -- so there is no
        # boundary to draw around it and the stroke itself is the mark.
        # At a neighbour's weight that stroke is a 1 px white core in a
        # 3 px halo, which disappears into photography; a mark nobody can
        # see identifies nothing. Detected from the geometry (area small
        # against extent) so any linear feature gets it without a caller
        # having to declare its kind.
        _lin = False
        for r_ in runs:
            xs_, ys_ = r_[:, 0], r_[:, 1]
            ext = max(float(xs_.max() - xs_.min()),
                      float(ys_.max() - ys_.min()))
            if ext < 8.0:
                continue
            area = cv2.contourArea(r_)
            if area / (ext * ext) < 0.06:
                _lin = True
                break
        if _lin:
            thick = max(thick, 4)
        cv2.polylines(canvas, runs, False, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.polylines(canvas, runs, False, (255, 255, 255), thick,
                      cv2.LINE_AA)
        xs = np.concatenate([r[:, 0] for r in runs])
        ys = np.concatenate([r[:, 1] for r in runs])
        placed.append((float(xs.min()), float(ys.min()),
                       float(xs.max()), float(ys.max())))
        # badge anchor: the DRAWN extent's bbox half-extents around its
        # center — a clipped shape anchors its badge to the visible part,
        # which is the only part the reader can bind the digit to.
        return (float((xs.min() + xs.max()) / 2.0),
                float((ys.min() + ys.max()) / 2.0),
                float(xs.max() - xs.min()) / 2.0,
                float(ys.max() - ys.min()) / 2.0)

    cx, cy = float(center_px[0]), float(center_px[1])
    specs: list[dict] = [{"x": cx, "y": cy, "poly": center_poly_px,
                          "thick": 3, "digit": 1 if draw_digits else None}]
    if target_px is not None:
        specs.append({"x": float(target_px[0]), "y": float(target_px[1]),
                      "poly": target_poly_px, "thick": 2,
                      "digit": 2 if draw_digits else None})
    _nb_polys = list(neighbor_polys_px or [])
    # Neighbors count from 3 (1 = center, 2 = target); the badge numerals
    # and the caller's text legend count the same way.
    _digit = 3
    for _i, nb in enumerate(neighbor_px or []):
        if isinstance(nb, (tuple, list)) and len(nb) == 2:
            # An explicit None means this mark is identified some OTHER
            # way -- a named road carries its name -- so it takes no
            # ordinal. Two identity glyphs on one object compete: the
            # centre badge beat the fid chip on 17 of 18 probe items.
            _d = (neighbor_digits[_i]
                  if neighbor_digits is not None and _i < len(neighbor_digits)
                  else _digit)
            specs.append({"x": float(nb[0]), "y": float(nb[1]),
                          "poly": _nb_polys[_i] if _i < len(_nb_polys)
                          else None,
                          "thick": 1,
                          "digit": _d if draw_digits else None})
            if _d is not None:
                _digit = int(_d) + 1
    # draw lowest-priority first so the center lands on top; badges after
    # all outlines so leader/crowding sees every mark.
    for sp in reversed(specs):
        # outline_marks False: only the CENTRE is lassoed. A lasso earns
        # its pixels on a PHOTOGRAPH, where it separates one object from
        # photographic clutter. On a filled schematic the fill already IS
        # the mask, so outlining every mark redraws the same boundary in
        # high-contrast black-on-white and adds a jagged edge carrying no
        # information the fill did not. The centre keeps its lasso: it is
        # the only way to say THIS one, and nothing else designates it.
        _skip_outline = (not outline_marks) and sp is not specs[0]
        if _has_poly(sp["poly"]) and not _skip_outline:
            sp["anchor"] = _poly_outline(sp["poly"], sp["thick"])
        elif _skip_outline:
            sp["anchor"] = (sp["x"], sp["y"], 0.0, 0.0)
        else:
            # No mask -> no outline, but the badge still has to point at
            # SOMETHING: a leader line ending on bare ground reads as a
            # badge tagging nothing. The flight view's pose mark is exactly
            # that case — a position, with no footprint anywhere — so a
            # point mark gets a small ring AT the point.
            # This ring is no footprint proxy: the
            # radius is frozen in DELIVERED pixels and scales with nothing,
            # so it asserts a location and never an extent, which is the
            # thing a circle-standing-in-for-a-footprint got wrong.
            _rx, _ry = int(round(sp["x"])), int(round(sp["y"]))
            cv2.circle(canvas, (_rx, _ry), _POINT_RING_PX + 1, (0, 0, 0),
                       2, cv2.LINE_AA)
            cv2.circle(canvas, (_rx, _ry), _POINT_RING_PX, (255, 255, 255),
                       1, cv2.LINE_AA)
            placed.append((sp["x"] - _POINT_RING_PX, sp["y"] - _POINT_RING_PX,
                           sp["x"] + _POINT_RING_PX, sp["y"] + _POINT_RING_PX))
            sp["anchor"] = (sp["x"], sp["y"],
                            float(_POINT_RING_PX), float(_POINT_RING_PX))
    for sp in reversed(specs):
        # digit None: OUTLINE ONLY. Numerals are withheld wherever the
        # caller carries its own identity glyphs -- the schematic names
        # every object by its fid chip, so an ordinal would be a second
        # identity for the same object. Measured twice: the centre badge
        # "1" was answered on 17 of 18 id_binding items, and once marks
        # were chipped their ordinal badges drew straight over the chips.
        # The mark still draws; only the numeral is withheld.
        if sp["digit"] is None:
            continue
        ax_, ay_ = sp["anchor"][2], sp["anchor"][3]
        _badge(sp["anchor"][0], sp["anchor"][1], ax_, ay_, sp["digit"])

    # NAMED PLACES, drawn on the photograph. The cue is written in street
    # names -- "the house on Quezon Avenue, east of 85th Street" -- and a
    # crop that carries only ordinal badges makes the model leave the
    # image for the row table to find out WHICH road it is looking at. A
    # road label occludes nothing the lasso exists to protect: the
    # evidence being examined is the building, not the asphalt.
    #
    # Placed after the marks so a name never covers an id, and skipped on
    # collision rather than stacked -- an unnamed road is still findable
    # by position, an unreadable badge names nothing.
    for _lx, _ly, _ltxt in (place_labels or []):
        _t = str(_ltxt)[:22]
        (_tw, _tht), _ = cv2.getTextSize(_t, _LABEL_FONT, 0.6, 1)
        _pw, _ph = _tw + 8, _tht + 8
        # A road's own BADGE already sits on this point, so the label has
        # to step off it -- centred, it collided every time and no name
        # ever drew. Same least-crowded search the badge uses, along the
        # road's own directions first.
        _best = None
        for _ox, _oy in ((0, 0), (0, 34), (0, -34), (70, 0), (-70, 0),
                         (0, 60), (0, -60), (120, 0), (-120, 0)):
            _cx0 = int(min(max(_lx + _ox - _pw / 2.0, 1), w - _pw - 1))
            _cy0 = int(min(max(_ly + _oy - _ph / 2.0, 1), h - _ph - 1))
            _cbb = (_cx0, _cy0, _cx0 + _pw, _cy0 + _ph)
            _crowd = 0.0
            for ob in placed:
                _ix = max(0, min(_cbb[2], ob[2]) - max(_cbb[0], ob[0]))
                _iy = max(0, min(_cbb[3], ob[3]) - max(_cbb[1], ob[1]))
                _crowd += _ix * _iy
            if _best is None or _crowd < _best[0]:
                _best = (_crowd, _cbb)
            if _crowd == 0:
                break
        if _best is None or _best[0] > 0:
            continue                      # genuinely no room: skip, never stack
        _bb = _best[1]
        _x0, _y0 = _bb[0], _bb[1]
        placed.append(_bb)
        cv2.rectangle(canvas, (_x0, _y0), (_x0 + _pw, _y0 + _ph),
                      (250, 250, 250), -1)
        cv2.rectangle(canvas, (_x0, _y0), (_x0 + _pw, _y0 + _ph),
                      (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, _t, (_x0 + 4, _y0 + _tht + 4), _LABEL_FONT,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # Minimal band: ONLY the tiny orientation strip ("N up", or the
    # caller's true frame-up when the image is rotated) — all naming lives
    # in the tool-text legend (OmniParser 2408.00203: the digit->description
    # text binding is the +23.3pp lever; on-image long labels are not).
    band_h = 20
    band = np.full((band_h, w, 3), _LOOK_BAND_BGR, np.uint8)
    _bt = str(band_text or "N up")
    (nw, _), _ = cv2.getTextSize(_bt, _LABEL_FONT, 0.45, 1)
    cv2.putText(band, _bt, (w - nw - 6, 14), _LABEL_FONT, 0.45,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(band, _bt, (w - nw - 6, 14), _LABEL_FONT, 0.45,
                (230, 230, 230), 1, cv2.LINE_AA)
    out = np.vstack([band, canvas])

    ok, buf = cv2.imencode(
        ".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed on look overlay")
    return bytes(buf.tobytes())
