"""Set-of-Mark: numbered discs on the frame, and the mark -> world mapping.

``mark_frame`` is ViSA's Visual Prompt Generator output ``V_som = (I_som,
M_som)``: the frame with a numbered disc at each candidate's mask centroid
and a thin outline on the mask, and the mapping from each number to what it
marks. The candidates are the run's SAM 3.1 detections, so a mark's world
position is the mask's own ortho unprojection: ``mark_world_xy`` is the
Executor's stop point.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTLINE_RGB = (255, 230, 0)
DISC_RGB = (255, 255, 255)
TEXT_RGB = (0, 0, 0)


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Set pixels with at least one unset 4-neighbour."""
    inner = mask.copy()
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return mask & ~inner


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # an older Pillow: the bitmap default
        return ImageFont.load_default()


def mark_frame(rgb: np.ndarray, detections: list) -> tuple[np.ndarray, list[dict]]:
    """Draw one numbered disc per detection; return the frame and ``M_som``.

    Ids run 1..N by confidence, descending, so mark 1 is the strongest
    candidate on every frame. A detection without a mask has no place in the
    frame and is skipped.
    """
    frame = np.array(rgb, dtype=np.uint8, copy=True)
    h, w = frame.shape[:2]
    radius = max(10, w // 80)
    ordered = sorted(
        (d for d in detections if getattr(d, "mask", None) is not None),
        key=lambda d: -float(getattr(d, "confidence", 0.0)),
    )
    marks: list[dict] = []
    for d in ordered:
        rr, cc = d.mask.rows_cols()
        if rr.size == 0:
            continue
        full = np.zeros(d.mask.frame_shape, dtype=bool)
        d.mask.window(full)[d.mask.crop] = True
        frame[_boundary(full)] = OUTLINE_RGB
        marks.append(
            {
                "id": len(marks) + 1,
                "label": str(d.label),
                "color": str((getattr(d, "attributes", None) or {}).get("color") or ""),
                "confidence": float(getattr(d, "confidence", 0.0)),
                "world_xy": [float(d.world_xy[0]), float(d.world_xy[1])],
                "frame_rc": [float(rr.mean()), float(cc.mean())],
                "bbox": [int(v) for v in d.mask.bbox],
            }
        )
    im = Image.fromarray(frame)
    draw = ImageDraw.Draw(im)
    font = _font(int(radius * 1.3))
    for m in marks:
        r, c = m["frame_rc"]
        draw.ellipse(
            [c - radius, r - radius, c + radius, r + radius],
            fill=DISC_RGB,
            outline=TEXT_RGB,
            width=max(1, radius // 8),
        )
        draw.text((c, r), str(m["id"]), fill=TEXT_RGB, font=font, anchor="mm")
    return np.asarray(im), marks


def mark_world_xy(mark: dict) -> tuple[float, float]:
    """The commit point for a mark: the detection's ortho unprojection."""
    return (float(mark["world_xy"][0]), float(mark["world_xy"][1]))


def legend(marks: list[dict]) -> str:
    """The SoM legend as prose, one line per mark."""
    if not marks:
        return "(no candidates marked in this frame)"
    lines = []
    for m in marks:
        colour = f", {m['color']}" if m["color"] else ""
        lines.append(f"[{m['id']}] {m['label']}{colour}")
    return "\n".join(lines)
