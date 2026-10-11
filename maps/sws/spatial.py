"""Grid geometry: the PCA orientation of a cell set. numpy only."""

from __future__ import annotations

import math

import numpy as np


def compute_orientation(rows: np.ndarray, cols: np.ndarray) -> tuple[float, float]:
    """PCA-based orientation angle and confidence for a feature.

    Parameters
    ----------
    rows, cols : ndarray
        Cell coordinates (integer grid positions).

    Returns
    -------
    (angle_deg, confidence)
        angle_deg : float in [0, 180). 0 = north-south, 90 = east-west.
        confidence : float in [0, 1]. 0 = round/ambiguous, 1 = perfectly linear.
    """
    if len(rows) <= 1:
        return 0.0, 0.0

    coords = np.column_stack(
        [rows.astype(float) - rows.mean(), cols.astype(float) - cols.mean()]
    )
    cov = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order: eigenvalues[0]=minor, eigenvalues[1]=major

    major = max(float(eigenvalues[1]), 1e-9)
    minor = max(float(eigenvalues[0]), 0.0)
    confidence = round(1.0 - (minor / major), 3)

    # First principal component (largest eigenvalue) = eigenvectors[:, 1]
    pc = eigenvectors[:, 1]  # [row_component, col_component]
    # atan2(dc, dr): compass bearing (0=north). Grid rows run south-up
    # (row 0 = south edge, +row = +north; col = +east), so (d_row, d_col)
    # is (d_north, d_east) directly.
    angle_rad = math.atan2(float(pc[1]), float(pc[0]))
    angle_deg = math.degrees(angle_rad) % 360.0
    # Collapse to [0, 180) — features are undirected
    if angle_deg >= 180.0:
        angle_deg -= 180.0

    return round(angle_deg, 1), confidence
