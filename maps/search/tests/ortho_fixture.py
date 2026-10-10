"""One synthetic ortho on disk for the look-render tests, written once per module."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ortho(root: Path, map_name: str, *, n: int, ground_z: float, ramp: bool = True) -> Path:
    """An ``n`` x ``n`` ortho at 0.1 m/px under ``root/ortho``, compressed: ``ramp``
    pixels (row and column ramps) so an off-by-one window shift changes bytes,
    else flat grey. The loader bakes its memory map beside it on first read."""
    if ramp:
        rgb = np.zeros((n, n, 3), dtype=np.uint8)
        rgb[..., 0] = (np.arange(n, dtype=np.uint32)[:, None] % 256).astype(np.uint8)
        rgb[..., 1] = (np.arange(n, dtype=np.uint32)[None, :] % 256).astype(np.uint8)
        rgb[..., 2] = 96
    else:
        rgb = np.full((n, n, 3), 120, dtype=np.uint8)
    out = Path(root) / "ortho"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{map_name}_ortho_m0.1.npz",
        rgb=rgb,
        height=np.full((n, n), ground_z + 5.0, dtype=np.float32),
        alpha=np.full((n, n), 255, dtype=np.uint8),
        x_min=np.float64(0.0), y_min=np.float64(0.0),
        px_per_m=np.float64(10.0),
    )
    return Path(root)
