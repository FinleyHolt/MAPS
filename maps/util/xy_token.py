"""The ``[x, y]`` world-point token — ONE parse for every fid-or-point slot.

Uniform addressing: wherever a slot takes a fid (``act.select``,
``viewpoints.add[].target_fid``, crop centers/marks), an ``[x, y]``
world-coordinate literal is accepted too — the agent points. The token
arrives as its string form (``"[1530, 1378]"``; a bare ``"1530, 1378"``
matches too). This module is the single grammar so the accept sites and
the bind sites can never drift.
"""
from __future__ import annotations

import re

XY_TOKEN_RE = re.compile(
    r"^\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?\s*$")


def parse_xy_token(token) -> tuple[float, float] | None:
    """``(x, y)`` when ``token`` is an ``[x, y]`` world-point literal,
    else ``None`` (non-strings included)."""
    if not isinstance(token, str):
        return None
    m = XY_TOKEN_RE.match(token)
    if not m:
        return None
    return (float(m.group(1)), float(m.group(2)))


def format_xy_token(x: float, y: float) -> str:
    """The canonical ``"[x, y]"`` string — the ONE wire form every fid-or-
    point slot stores. Round-trips through :func:`parse_xy_token`; the
    mission contract's ``target_xy`` arrays normalize through here at the
    accept site so every downstream consumer keeps reading one grammar."""
    return f"[{x:g}, {y:g}]"
