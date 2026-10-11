"""color.py — the deterministic Lab-chroma-blended van de Weijer (w2c) color extractor.

Owns color for the runtime detection path, read from pixels only (color is
ORDERING-only, never a filter or gate). The w2c color lattice lives in
``data/w2c.txt``.

``color_from_mask_m6(rgb_crop, region)`` returns the runtime dict
  {"color": "<word> <frac> / ...", "color_conf": float, "color_shade": "dk"|"mid"|"lt",
   "color_flag": "" | "look"}
where color_shade drives the dk/lt prefix on the compact surface render and
color_flag=='look' is the uncertain '?' that routes to a viewpoint.
Returns ``None`` on an empty/degenerate mask (never raises on the detect path).
``m6_attributes(rgb_crop, region)`` is the same reading as the string
attributes a detection or a prior node carries (``color_conf`` to 3 dp).

Body pixels are selected by an erode plus a luminance drop, so the reported
color comes from the object's interior rather than its edges or its shadow.
"""
from __future__ import annotations
import logging
import math
import os

import numpy as np

from maps.util.mask_region import MaskRegion

_log = logging.getLogger(__name__)
_M6_FAIL_N = 0  # systematic-failure counter for the rate-limited warn below

_W2C_PATH = os.path.join(os.path.dirname(__file__), "data", "w2c.txt")

# van de Weijer 11 basic color terms (canonical order) + map onto our scorer vocabulary
W2C_NAMES = ["black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"]
_VDW_TO_VOCAB = {"black": "black", "blue": "blue", "brown": "brown", "gray": "gray", "green": "green",
                 "orange": "orange", "pink": "red", "purple": "purple", "red": "red",
                 "white": "white", "yellow": "yellow"}

# erode / luminance-drop: the body-pixel selection
_MIX_ERODE_PX_SMALL = 2
_MIX_ERODE_PX_LARGE = 3
_MIX_ERODE_LARGE_AREA_PX = 900
_MIX_DARK_PCTL = 30.0
_MIX_BRIGHT_PCTL = 95.0
_MIX_DARK_ABS_LUM = 50.0
# M6 blend parameters -- the source of truth for the runtime and offline paths
_CMID = 15.0
_CSCALE = 4.0
_CLO = 10.0
_CHI = 20.0
_ACHR_CENTERS = {"white": 88.0, "gray": 64.0, "black": 40.0}
_ACHR_T = 16.0
_MAX_COMP = 3
_MIN_COMP_FRAC = 0.12

_TABLE = None


def _table() -> np.ndarray:
    global _TABLE
    if _TABLE is None:
        rows = np.loadtxt(_W2C_PATH, dtype=np.float64)
        probs = rows[:, 3:14] if rows.shape[1] == 14 else rows
        s = probs.sum(axis=1, keepdims=True)
        _TABLE = np.divide(probs, s, out=np.zeros_like(probs), where=s > 0).astype(np.float32)
    return _TABLE


def _posterior(px: np.ndarray) -> np.ndarray:
    p = np.clip(px, 0, 255).astype(np.int64)
    idx = p[:, 0] // 8 + 32 * (p[:, 1] // 8) + 1024 * (p[:, 2] // 8)
    return _table()[idx]


def _survivors(rgb, region):
    """Erode + luminance-drop to the object's body pixels.

    Reads the region's crop, so every reduction here is sized by the object
    rather than by the frame. The region's box is guaranteed to bound its own
    crop, which is what makes that substitution safe without a check.
    """
    import cv2

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3 or tuple(rgb.shape[:2]) != tuple(region.frame_shape):
        return None
    m_box = region.crop
    if not m_box.any():
        return None
    n_orig = int(m_box.sum())
    sub = np.pad(m_box, 1).astype(np.uint8)
    radius = _MIX_ERODE_PX_LARGE if n_orig >= _MIX_ERODE_LARGE_AREA_PX else _MIX_ERODE_PX_SMALL
    eroded_sub = None
    for r in range(radius, 0, -1):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
        cand = cv2.erode(sub, k)
        if cand.any():
            eroded_sub = cand
            break
    if eroded_sub is None:
        return None
    # Index the box, not the frame: boolean indexing walks row-major either
    # way and every eroded pixel lies inside the box, so the survivor sequence
    # is the same one a full-frame `rgb[eroded]` would produce.
    eroded_box = eroded_sub[1:-1, 1:-1].astype(bool)
    px = region.window(rgb)[eroded_box][:, :3].astype(np.float32)
    lum = px @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    keep = ((lum >= np.percentile(lum, _MIX_DARK_PCTL))
            & (lum <= np.percentile(lum, _MIX_BRIGHT_PCTL))
            & (lum >= _MIX_DARK_ABS_LUM))
    surv = px[keep]
    return surv, n_orig, int(surv.shape[0])


def _median_lab(surv):
    import cv2

    lab = cv2.cvtColor(surv.astype(np.uint8).reshape(-1, 1, 3),
                       cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    L = float(np.median(lab[:, 0])) * 100.0 / 255.0
    a = float(np.median(lab[:, 1])) - 128.0
    b = float(np.median(lab[:, 2])) - 128.0
    return L, (a * a + b * b) ** 0.5


def color_from_mask_m6(rgb_crop, region) -> "dict | None":
    """M6 runtime color extraction. See module docstring for the returned dict. Robust: None on
    empty/degenerate input, never raises on the detection path.

    ``region`` is a ``MaskRegion``: the crop and where it sits, so the read
    never touches the frame.

    A wrong TYPE raises rather than joining the degenerate-input path below.
    The except is deliberately broad because a genuine failure here must not
    kill a flight, but that same breadth turned an un-migrated caller passing a
    bare array into colour silently returning None fleet-wide — visible only as
    "no colour anywhere". A signature mismatch is a programming error and has
    to fail at the call site."""
    if region is not None and not isinstance(region, MaskRegion):
        raise TypeError(
            f"color_from_mask_m6 takes a MaskRegion, got {type(region).__name__}"
        )
    try:
        sp = _survivors(rgb_crop, region)
        if sp is None or sp[2] == 0:
            return None
        surv, n_orig, n_surv = sp
        L, C = _median_lab(surv)
        post = _posterior(surv).mean(axis=0)
        vdist = {}
        for i in range(11):
            w = _VDW_TO_VOCAB[W2C_NAMES[i]]
            vdist[w] = vdist.get(w, 0.0) + float(post[i])
        # soft blend: value-axis achromatic dist (by L*) mixed with the hue posterior by p_achr
        pa = 1.0 / (1.0 + math.exp((C - _CMID) / _CSCALE))
        ad = {k: math.exp(-abs(L - c) / _ACHR_T) for k, c in _ACHR_CENTERS.items()}
        sa = sum(ad.values()) or 1.0
        ad = {k: v / sa for k, v in ad.items()}
        final = {}
        for k in set(list(vdist) + list(ad)):
            final[k] = pa * ad.get(k, 0.0) + (1.0 - pa) * vdist.get(k, 0.0)
        s = sum(final.values()) or 1.0
        final = {k: v / s for k, v in final.items()}
        ranked = sorted(final.items(), key=lambda kv: -kv[1])
        p1 = ranked[0][1]
        p2 = ranked[1][1] if len(ranked) > 1 else 0.0
        comps = [(k, v) for k, v in ranked[:_MAX_COMP] if v >= _MIN_COMP_FRAC] or [ranked[0]]
        gate_conf = min(1.0, abs(C - _CMID) / (_CMID - _CLO))
        conf = round(max(0.05, min(1.0, 0.5 * (p1 - p2) + 0.5 * gate_conf)), 3)
        shade = "dk" if L < 50.0 else ("lt" if L > 75.0 else "mid")
        flag = "look" if _CLO <= C <= _CHI else ""
        color = " / ".join(f"{w} {frac:.2f}" for w, frac in comps)
        return {"color": color, "color_conf": conf, "color_shade": shade, "color_flag": flag}
    except Exception:
        # A degenerate/empty mask already returned None above (``sp is None``); a
        # failure THIS deep is SYSTEMATIC (missing data/w2c.txt, a cv2/numpy break)
        # and — now that BOTH detections and priors route color through M6 — would
        # silently kill color fleet-wide, surfacing only as "no color anywhere".
        # Log rate-limited so it is VISIBLE, but never raise (color must not crash a
        # flight; the caller treats None as "no color this frame").
        global _M6_FAIL_N
        _M6_FAIL_N += 1
        if _M6_FAIL_N == 1 or _M6_FAIL_N % 100 == 0:
            _log.warning(
                "color_from_mask_m6 failed (n=%d) — color extraction may be "
                "systematically broken (check perception/viz/data/w2c.txt, cv2)",
                _M6_FAIL_N, exc_info=True,
            )
        return None


def m6_attributes(rgb_crop, region) -> "dict[str, str] | None":
    """The M6 mixture as node attributes: ``color``, ``color_conf`` (3 dp),
    ``color_shade``, ``color_flag``; ``None`` where M6 reads nothing."""
    mix = color_from_mask_m6(rgb_crop, region)
    if not mix:
        return None
    return {
        "color": mix["color"],
        "color_conf": f"{mix['color_conf']:.3f}",
        "color_shade": mix["color_shade"],
        "color_flag": mix["color_flag"],
    }
