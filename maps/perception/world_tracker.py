"""World-space multi-object tracker (UCMCTrack core, static-scene variant).

Assigns a stable cross-frame ``track_id`` to per-frame
``DetectedObject``s so the belief engine's re-identification
(``maps.sws.graph.runtime._find_track`` primary exact-id match)
becomes trivial instead of relying on world-bbox-IoU churn.

Why this algorithm (literature-grounded, 2024-2026 SOTA review):
UCMCTrack (AAAI 2024, arXiv:2312.08952) is the only recent motion-only,
appearance-free tracker that *decisively* beats ByteTrack
(+2.7 HOTA MOT17, +16.3 HOTA DanceTrack) by associating on the
ground/world plane with a Kalman filter and a Mapped Mahalanobis
Distance cost rather than image-plane IoU. Our regime is an even
stronger fit than UCMCTrack's:

* World coordinates are *given* (depth projection from a known sim pose,
  or DSM ray-cast depth + the flight controller's EKF pose on hardware), so UCMCTrack's
  homography + camera-motion
  compensation machinery is unnecessary and is dropped entirely — its
  own ablation showed per-frame CMC hurts when the plane is already
  correct; the value was the world-plane Mahalanobis association.
* Targets are a STATIONARY worldstate (buildings, parked cars, roads,
  trees) observed by a moving drone. So the constant-velocity Kalman is
  replaced by **constant-position** (``F = I``): the prior is "this
  object is where it was, ± projection/depth noise". A CV model would
  let static-object tracks drift on noisy detections — strictly worse.
* IDs must persist for the whole flight (re-acquire on revisit). Tracks
  are **never deleted**: unseen tracks go dormant, their covariance
  inflates (capped) so the gate widens, and the same Mahalanobis cost
  re-binds the old ID when the drone flies back over the object. Only
  never-confirmed provisional tracks are pruned.

The one ByteTrack idea kept: confirm a track only after ``n_init``
consecutive hits, to reject SAM single-frame false positives.

Robot-agnostic: imports only numpy / scipy / ``perception.base``. No
ROS, world-graph, maps.search, or benchmark imports. Consumes
``DetectedObject.world_xy`` (the depth-projected world centroid — the
world-plane position UCMCTrack associates on) and ``.label`` (a hard
class gate, a large win at hundreds of objects/frame). Masks/boxes are
not needed: centroid-only Mahalanobis is UCMCTrack's design and is fast.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from maps.perception.base import DetectedObject

log = logging.getLogger(__name__)

# χ²(2 dof, p=0.99) ≈ 9.21 — Mahalanobis gate for a 2-D position residual.
_CHI2_2DOF_P99 = 9.21
_BIG = 1.0e9  # finite sentinel for gated-out assignment-matrix cells


@dataclass
class WorldTrackerConfig:
    """All knobs in meters² / dimensionless. Tune ``r_pos_var`` to the
    actual world-projection 1σ position error (depth-error dominated)."""

    q_pos_var: float = 0.2**2  # per-frame position random-walk variance
    r_pos_var: float = 1.0**2  # measurement (world-projection) variance
    p0_pos_var: float = 1.0**2  # initial track covariance
    p_max_pos_var: float = 20.0**2  # cap on dormant-track covariance
    gate_chi2: float = _CHI2_2DOF_P99
    n_init: int = 3  # consecutive hits before a track is "confirmed"
    # EVERY ADMITTED DETECTION DESERVES AN IDENTITY. Admission is the
    # detector's job (score_threshold); a birth floor ABOVE it opened a
    # dead band (0.3-0.5) whose detections got no wt_ id, fell back to a
    # per-frame-churning key downstream, and re-homed each other's graph
    # nodes across a pass -- 140#138's GT lot at conf 0.41 was minted and
    # destroyed by every later stop, four runs in a row. Provisional
    # births + n_init pruning already kill single-frame flicker; that is
    # the tracker's own noise valve, not a second admission gate.
    birth_score: float = 0.0  # spawn a track for every admitted detection

    @classmethod
    def from_env(cls) -> "WorldTrackerConfig":
        def _f(key: str, default: float) -> float:
            v = os.environ.get(key)
            return float(v) if v not in (None, "") else default

        return cls(
            q_pos_var=_f("WORLD_TRACK_Q_M", 0.2) ** 2,
            r_pos_var=_f("WORLD_TRACK_R_M", 1.0) ** 2,
            p0_pos_var=_f("WORLD_TRACK_P0_M", 1.0) ** 2,
            p_max_pos_var=_f("WORLD_TRACK_PMAX_M", 20.0) ** 2,
            gate_chi2=_f("WORLD_TRACK_GATE_CHI2", _CHI2_2DOF_P99),
            n_init=int(_f("WORLD_TRACK_N_INIT", 3)),
            birth_score=_f("WORLD_TRACK_BIRTH_SCORE", 0.0),
        )


@dataclass
class _Track:
    track_id: int
    label: str
    x: np.ndarray  # (2,) world (east, north) meters
    p: np.ndarray  # (2, 2) covariance
    hits: int = 1
    misses: int = 0
    confirmed: bool = False


@dataclass
class WorldTracker:
    """UCMCTrack-core, constant-position, persistent-ID world tracker."""

    cfg: WorldTrackerConfig = field(default_factory=WorldTrackerConfig)
    _tracks: list[_Track] = field(default_factory=list, init=False)
    _next_id: int = field(default=0, init=False)
    _frame: int = field(default=0, init=False)

    def reset(self) -> None:
        """Drop all tracks (call once per episode)."""
        self._tracks = []
        self._next_id = 0
        self._frame = 0

    def update(self, detections: list[DetectedObject]) -> None:
        """Predict → Mahalanobis-assign → KF-update → lifecycle, then
        stamp a stable ``metadata['track_id']`` ('wt_<int>') in place on
        every detection that bound to (or spawned) a track. Detections
        below ``birth_score`` that match nothing are left untouched."""
        self._frame += 1
        cfg = self.cfg

        # --- predict (constant position; covariance inflates, capped) --
        q = np.array([[cfg.q_pos_var, 0.0], [0.0, cfg.q_pos_var]])
        for t in self._tracks:
            t.p = t.p + q
            np.fill_diagonal(t.p, np.minimum(np.diag(t.p), cfg.p_max_pos_var))

        # --- gather valid detections (finite world_xy) ----------------
        dets: list[tuple[int, np.ndarray, str]] = []
        for i, d in enumerate(detections):
            wxy = getattr(d, "world_xy", None)
            if wxy is None:
                continue
            z = np.asarray(wxy, dtype=np.float64).reshape(-1)
            if z.shape[0] != 2 or not np.all(np.isfinite(z)):
                continue
            dets.append((i, z, str(getattr(d, "label", "") or "")))

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
                if not trk.confirmed and trk.hits >= cfg.n_init:
                    trk.confirmed = True
                matched_track_idx.add(r)
                detections[det_i].metadata["track_id"] = f"wt_{trk.track_id}"
                _stamp(detections[det_i], trk)  # after confirm flip
                dets[c] = (-1, z, "")  # consume

        # --- unmatched detections -> births ---------------------------
        born_idx: set[int] = set()
        for det_i, z, label in dets:
            if det_i < 0:
                continue
            score = float(getattr(detections[det_i], "confidence", 0.0))
            if score < cfg.birth_score:
                continue
            trk = _Track(
                track_id=self._next_id,
                label=label,
                x=z.copy(),
                p=np.array([[cfg.p0_pos_var, 0.0], [0.0, cfg.p0_pos_var]]),
                confirmed=cfg.n_init <= 1,
            )
            self._next_id += 1
            self._tracks.append(trk)
            born_idx.add(len(self._tracks) - 1)
            detections[det_i].metadata["track_id"] = f"wt_{trk.track_id}"
            _stamp(detections[det_i], trk)

        # --- true misses -> dormant (confirmed tracks never deleted) --
        survivors: list[_Track] = []
        for idx, trk in enumerate(self._tracks):
            if idx in matched_track_idx or idx in born_idx:
                survivors.append(trk)
                continue
            trk.misses += 1
            # Confirmed tracks live forever (whole-flight re-ID; their
            # covariance keeps inflating, capped, so the gate widens for
            # graceful re-acquisition on revisit). Never-confirmed
            # provisional tracks are pruned after n_init consecutive
            # misses (kills SAM single-frame false-positive flicker).
            if trk.confirmed or trk.misses < cfg.n_init:
                survivors.append(trk)
        self._tracks = survivors

    # ------------------------------------------------------------------
    def _cost_matrix(
        self, dets: list[tuple[int, np.ndarray, str]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mapped Mahalanobis Distance ``εᵀS⁻¹ε + ln|S|`` (UCMCTrack),
        χ²-gated, with a hard class gate. Vectorised over T×D."""
        cfg = self.cfg
        tx = np.stack([t.x for t in self._tracks])  # (T,2)
        tp = np.stack([t.p for t in self._tracks])  # (T,2,2)
        dz = np.stack([z for _, z, _ in dets])  # (D,2)

        r = np.array([[cfg.r_pos_var, 0.0], [0.0, cfg.r_pos_var]])
        s = tp + r  # (T,2,2) innovation covariance
        a, b = s[:, 0, 0], s[:, 0, 1]
        c_, d_ = s[:, 1, 0], s[:, 1, 1]
        det_s = a * d_ - b * c_  # (T,)
        det_s = np.where(det_s > 1e-12, det_s, 1e-12)
        # closed-form 2×2 inverse
        sinv = np.empty_like(s)
        sinv[:, 0, 0] = d_ / det_s
        sinv[:, 0, 1] = -b / det_s
        sinv[:, 1, 0] = -c_ / det_s
        sinv[:, 1, 1] = a / det_s

        eps = dz[None, :, :] - tx[:, None, :]  # (T,D,2)
        # mahalanobis m[t,d] = εᵀ Sinv ε
        se = np.einsum("tij,tdj->tdi", sinv, eps)  # (T,D,2)
        maha = np.einsum("tdi,tdi->td", eps, se)  # (T,D)
        logdet = np.log(det_s)[:, None]  # (T,1)
        cost = maha + logdet

        gated = maha > cfg.gate_chi2
        # Class gate: forbid cross-label association so a detection only ever
        # re-identifies a track of the SAME class (a "gray car" never folds
        # onto a "car" track). Unconditional — the two-layer perception
        # contract depends on it.
        tl = np.array([t.label for t in self._tracks], dtype=object)
        dl = np.array([lab for _, _, lab in dets], dtype=object)
        label_mismatch = tl[:, None] != dl[None, :]
        gated = gated | label_mismatch
        cost = np.where(gated, _BIG, cost)
        return cost.astype(np.float64), gated

    def _kf_update(self, trk: _Track, z: np.ndarray) -> None:
        """Linear KF correction, H = I (we observe position directly)."""
        cfg = self.cfg
        r = np.array([[cfg.r_pos_var, 0.0], [0.0, cfg.r_pos_var]])
        s = trk.p + r
        det_s = s[0, 0] * s[1, 1] - s[0, 1] * s[1, 0]
        det_s = det_s if abs(det_s) > 1e-12 else math.copysign(1e-12, det_s)
        sinv = np.array([[s[1, 1], -s[0, 1]], [-s[1, 0], s[0, 0]]]) / det_s
        k = trk.p @ sinv  # Kalman gain (2,2)
        trk.x = trk.x + k @ (z - trk.x)
        trk.p = (np.eye(2) - k) @ trk.p


def _stamp(det: DetectedObject, trk: _Track) -> None:
    """Record tracker provenance for downstream/debug (inert metadata)."""
    md = det.metadata
    md["world_track_confirmed"] = bool(trk.confirmed)
    md["world_track_hits"] = int(trk.hits)


__all__ = ["WorldTracker", "WorldTrackerConfig"]
