"""World-space tracker (UCMCTrack core, arXiv:2312.08952, static-scene
variant): stamps a stable ``metadata["track_id"]`` on each detection by
Mahalanobis-gated Hungarian association on ``world_xy`` within one label,
under a constant-position Kalman filter. Confirmed tracks are never deleted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from maps.perception.detection import DetectedObject

# χ²(2 dof, p=0.99) ≈ 9.21 — Mahalanobis gate for a 2-D position residual.
_CHI2_2DOF_P99 = 9.21
_BIG = 1.0e9  # finite sentinel for gated-out assignment-matrix cells


# Every term in metres² / dimensionless. ``_R_POS_VAR`` is the
# world-projection 1σ position error (depth-error dominated), squared.
_Q_POS_VAR = 0.2**2  # per-frame position random-walk variance
_R_POS_VAR = 1.0**2  # measurement (world-projection) variance
_P0_POS_VAR = 1.0**2  # initial track variance
_P_MAX_POS_VAR = 20.0**2  # cap on a dormant track's variance
_N_INIT = 3  # consecutive hits before a track is "confirmed"


@dataclass
class _Track:
    track_id: int
    label: str
    x: np.ndarray  # (2,) world (east, north) meters
    # Per-axis variance. Isotropic: every step adds or scales the same
    # amount on both axes, so the 2x2 covariance is ``p * I``.
    p: float
    hits: int = 1
    misses: int = 0
    confirmed: bool = False


@dataclass
class WorldTracker:
    """UCMCTrack-core, constant-position, persistent-ID world tracker."""

    _tracks: list[_Track] = field(default_factory=list, init=False)
    _next_id: int = field(default=0, init=False)

    def update(self, detections: list[DetectedObject]) -> None:
        """Predict → Mahalanobis-assign → KF-update → lifecycle, then
        stamp a stable ``metadata['track_id']`` ('wt_<int>') in place on
        every detection with a finite ``world_xy``: it binds to a track or
        spawns one. Admission is the detector's job (its score floor);
        provisional births + ``_N_INIT`` pruning are the tracker's own
        flicker valve, never a second admission gate."""
        # --- predict (constant position; variance inflates, capped) ---
        for t in self._tracks:
            t.p = min(t.p + _Q_POS_VAR, _P_MAX_POS_VAR)

        # --- gather valid detections (finite world_xy) ----------------
        dets: list[tuple[int, np.ndarray, str]] = []
        for i, d in enumerate(detections):
            z = np.asarray(d.world_xy, dtype=np.float64).reshape(-1)
            if z.shape[0] != 2 or not np.all(np.isfinite(z)):
                continue
            dets.append((i, z, str(d.label or "")))

        # --- associate (Mahalanobis-gated Hungarian) ------------------
        # ``matched_track_idx`` indexes the pre-birth ``self._tracks``;
        # births only append, so those indices stay valid and the new
        # tail indices go in ``born_idx``. The lifecycle pass below then
        # treats a track as a true miss only when it is NEITHER matched
        # NOR born this frame (a just-born track is new, not a miss).
        matched_track_idx: set[int] = set()
        if self._tracks and dets:
            cost, gated = self._cost_matrix(dets)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if gated[r, c]:
                    continue  # assignment fell outside the χ² gate
                trk = self._tracks[r]
                det_i, z, _ = dets[c]
                self._kf_update(trk, z)
                trk.hits += 1
                trk.misses = 0
                if not trk.confirmed and trk.hits >= _N_INIT:
                    trk.confirmed = True
                matched_track_idx.add(r)
                detections[det_i].metadata["track_id"] = f"wt_{trk.track_id}"
                dets[c] = (-1, z, "")  # consume

        # --- unmatched detections -> births ---------------------------
        born_idx: set[int] = set()
        for det_i, z, label in dets:
            if det_i < 0:
                continue
            trk = _Track(track_id=self._next_id, label=label, x=z.copy(), p=_P0_POS_VAR)
            self._next_id += 1
            self._tracks.append(trk)
            born_idx.add(len(self._tracks) - 1)
            detections[det_i].metadata["track_id"] = f"wt_{trk.track_id}"

        # --- true misses -> dormant (confirmed tracks never deleted) --
        survivors: list[_Track] = []
        for idx, trk in enumerate(self._tracks):
            if idx in matched_track_idx or idx in born_idx:
                survivors.append(trk)
                continue
            trk.misses += 1
            # Confirmed tracks live forever (whole-flight re-ID; their
            # variance keeps inflating, capped, so the gate widens for
            # graceful re-acquisition on revisit). Never-confirmed
            # provisional tracks are pruned after _N_INIT consecutive
            # misses (kills SAM single-frame false-positive flicker).
            if trk.confirmed or trk.misses < _N_INIT:
                survivors.append(trk)
        self._tracks = survivors

    # ------------------------------------------------------------------
    def _cost_matrix(
        self, dets: list[tuple[int, np.ndarray, str]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mapped Mahalanobis Distance ``εᵀS⁻¹ε + ln|S|`` (UCMCTrack),
        χ²-gated, with a hard class gate. Vectorised over T×D.

        ``S = s * I`` with ``s = p + r``, so ``|S| = s²`` and ``S⁻¹ = I * s/s²``,
        each written the way the 2x2 closed form evaluates it."""
        tx = np.stack([t.x for t in self._tracks])  # (T,2)
        s = np.array([t.p for t in self._tracks]) + _R_POS_VAR  # (T,)
        det_s = s * s
        sinv = s / det_s
        dz = np.stack([z for _, z, _ in dets])  # (D,2)
        eps = dz[None, :, :] - tx[:, None, :]  # (T,D,2)
        se = eps * sinv[:, None, None]
        maha = eps[..., 0] * se[..., 0] + eps[..., 1] * se[..., 1]  # (T,D)
        cost = maha + np.log(det_s)[:, None]

        gated = maha > _CHI2_2DOF_P99
        # Class gate: forbid cross-label association so a detection only ever
        # re-identifies a track of the SAME class (a "gray car" never folds
        # onto a "car" track). Unconditional — the two-layer perception
        # contract depends on it.
        tl = np.array([t.label for t in self._tracks], dtype=object)
        dl = np.array([lab for _, _, lab in dets], dtype=object)
        gated = gated | (tl[:, None] != dl[None, :])
        return np.where(gated, _BIG, cost), gated

    @staticmethod
    def _kf_update(trk: _Track, z: np.ndarray) -> None:
        """Linear KF correction, H = I (position observed directly)."""
        s = trk.p + _R_POS_VAR
        k = trk.p * (s / (s * s))  # Kalman gain, the closed-form 2x2 inverse's arithmetic
        trk.x = trk.x + k * (z - trk.x)
        trk.p = (1.0 - k) * trk.p
