"""The cells a row occupies: the bake's 2 m ``(row, col)`` grid record every
graph node persists."""

from __future__ import annotations

import numpy as np


def cells_of(row) -> np.ndarray:
    """A surface row's or graph node's ``cells`` as an ``(n, 2)`` int array,
    empty when it carries none."""
    c = row.get("cells") if isinstance(row, dict) else getattr(row, "cells", None)
    return np.zeros((0, 2), dtype=int) if c is None else np.asarray(c).reshape(-1, 2).astype(int, copy=False)
