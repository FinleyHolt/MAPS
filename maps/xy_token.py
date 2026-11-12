"""The ``[x, y]`` world-point token, the one grammar for every slot that
takes a fid or a point (``"[1530, 1378]"``; a bare ``"1530, 1378"`` too)."""

from __future__ import annotations

import re

_XY_TOKEN_RE = re.compile(r"^\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?\s*$")


def parse_xy_token(token) -> tuple[float, float] | None:
    """``(x, y)`` when ``token`` is an ``[x, y]`` world-point literal,
    else ``None`` (non-strings included)."""
    if not isinstance(token, str):
        return None
    m = _XY_TOKEN_RE.match(token)
    if not m:
        return None
    return (float(m.group(1)), float(m.group(2)))


def format_xy_token(x: float, y: float) -> str:
    """The canonical ``"[x, y]"`` string a fid-or-point slot stores;
    round-trips through :func:`parse_xy_token`."""
    return f"[{x:g}, {y:g}]"


def is_xy(v) -> bool:
    """True for a bare ``[x, y]`` point value: a list or tuple of exactly two
    numbers (a bool is not a number here)."""
    return (
        isinstance(v, (list, tuple))
        and len(v) == 2
        and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in v)
    )
