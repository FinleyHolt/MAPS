"""The corrector's pose graph: the last seconds of frames placed together, each by its own registration and its ties to the frame before it.

Every frame is a node in a window of the last :data:`WINDOW_S` seconds: its
registration (``live_source.register_frame``) an anchor weighted by its
response, its ties to the frame before it (:mod:`frame_ties`) holding the two
together. Each new frame re-solves the window (:func:`tie_adjust.solve`), so a
refused frame is placed by its neighbours and a false peak is outvoted; the
window carries one yaw and one scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from benchmarks.flight import frame_ties, tie_adjust

#: The frames solved together: those within this many seconds of the newest.
#: One yaw and one scale hold ten seconds of a leg and not forty.
WINDOW_S = 10.0
#: A solved node is kept this long after it leaves the window, for a ring frame that old.
NODE_AGE_S = 60.0
#: Each new frame is tied to this many frames before it.
TIE_NEIGHBOURS = 1
#: Anchors the window needs before any frame is moved.
MIN_ANCHORS = 2
#: A measurement further than this from the graph's estimate is refused: sized
#: to how far the chain's own residual moves along a leg, tens of metres on
#: the C-12, so a lock one block over stays refused.
INNOVATION_CAP_M = 40.0


@dataclass
class _Frame:
    seq: int
    t: float
    pose: frame_ties.Pose
    gray: np.ndarray
    geometry: frame_ties.FrameGeometry


class PoseGraph:
    """The window of frames, their anchors and ties, and the solution over them.

    ``sensor`` is the camera at the frames' size and ``ground`` the height
    under a nadir as a callable of ``(x, y)``.
    """

    def __init__(self, sensor, ground) -> None:
        self.sensor, self.ground = sensor, ground
        self.matcher = frame_ties.TieMatcher(sensor, ground)
        self.frames: dict[int, _Frame] = {}
        self.measurements: dict[int, tuple[float, float, float]] = {}
        self.ties: list[tuple[int, int, tuple[float, float], tuple[float, float]]] = []
        #: Every solved node: ``seq -> {e_m, n_m, yaw_deg, alt_m, sigma_m, ties, t}``.
        self.nodes: dict[int, dict[str, Any]] = {}

    def estimate(self, t: float) -> dict[str, Any] | None:
        """The solved node nearest ``t`` in time within :data:`NODE_AGE_S`, or None."""
        near = [n for n in self.nodes.values() if abs(t - n["t"]) <= NODE_AGE_S]
        return min(near, key=lambda n: abs(t - n["t"])) if near else None

    def _placed(self, f: _Frame) -> frame_ties.Pose:
        """The frame's pose as a search starts from it: the chain's, moved by the graph's estimate."""
        est = self.estimate(f.t)
        if est is None:
            return f.pose
        return replace(f.pose, pose_xy=(f.pose.pose_xy[0] + est["e_m"], f.pose.pose_xy[1] + est["n_m"]))

    def _forget(self, now: float) -> None:
        old = [seq for seq, f in self.frames.items() if now - f.t > WINDOW_S]
        for seq in old:
            del self.frames[seq]
            self.measurements.pop(seq, None)
        if old:
            gone = set(old)
            self.ties = [t for t in self.ties if t[0] not in gone and t[1] not in gone]
        for seq in [seq for seq, n in self.nodes.items() if now - n["t"] > NODE_AGE_S]:
            del self.nodes[seq]

    def offer(self, seq: int, t: float, pose: frame_ties.Pose, gray: np.ndarray,
              measurement: tuple[float, float, float] | None) -> dict[str, Any]:
        """Add a frame (its chain pose, its luma, its own registration
        ``(east, north, response)`` or None), re-solve the window, and return
        the frame's record: ``source`` is ``measured``, ``tied``, ``nearest``
        or None, with the shift, ``ties``, ``sigma_m`` and ``why``."""
        self._forget(t)
        frame = _Frame(seq, t, pose, gray, frame_ties.FrameGeometry(pose, self.sensor, self.ground(tuple(pose.pose_xy))))
        record: dict[str, Any] = {"applied": False, "source": None, "shift_yaw_deg": 0.0, "shift_alt_m": 0.0, "ties": 0}
        before = sorted((f for f in self.frames.values() if f.t <= t), key=lambda f: f.t)[-TIE_NEIGHBOURS:]
        for prev in before:
            for tie in self.matcher.ties(prev.seq, self._placed(prev), prev.gray, self._placed(frame), gray):
                self.ties.append((prev.seq, seq, tie.uv_i, tie.uv_j))
        self.frames[seq] = frame
        if measurement is not None:
            e, n, response = measurement
            est = self.estimate(t)
            if est is not None and math.hypot(e - est["e_m"], n - est["n_m"]) > INNOVATION_CAP_M:
                record["why"] = (f"disagrees with the graph's estimate by {math.hypot(e - est['e_m'], n - est['n_m']):.1f} m, "
                                 f"past the {INNOVATION_CAP_M:.0f} m cap")
            else:
                self.measurements[seq] = (float(e), float(n), float(response))
        if len(self.measurements) < MIN_ANCHORS:
            record.setdefault("why", f"{len(self.measurements)} anchor in the window, {MIN_ANCHORS} needed")
        else:
            self._solve(seq, t, record)
        # This frame's work grid and edge maps, made now for the pair its
        # successor opens, at the placement the solve just gave it.
        self.matcher.warm(seq, self._placed(frame), gray)
        return record

    def _solve(self, seq: int, t: float, record: dict[str, Any]) -> None:
        """Re-solve the window and write the frame's placement into ``record``."""
        geometry = {s: f.geometry for s, f in self.frames.items()}
        nodes, anchors, tie_rows = tie_adjust.rows_from(geometry, self.measurements, self.ties)
        got = tie_adjust.solve(nodes, anchors, tie_rows)
        for s, (e, n) in got.correction.items():
            agl = self.frames[s].pose.alt_msl_m - self.ground(tuple(self.frames[s].pose.pose_xy))
            self.nodes[s] = {"e_m": e, "n_m": n, "yaw_deg": got.yaw_deg.get(s, 0.0), "alt_m": got.scale.get(s, 0.0) * agl,
                             "sigma_m": float(math.hypot(*got.sigma[s])), "ties": got.ties_per_frame.get(s, 0),
                             "t": self.frames[s].t}
        if seq in got.correction:
            node = self.nodes[seq]
            record.update(applied=True, source="measured" if seq in self.measurements else "tied",
                          shift_e_m=node["e_m"], shift_n_m=node["n_m"], shift_yaw_deg=node["yaw_deg"],
                          shift_alt_m=node["alt_m"], ties=node["ties"], sigma_m=node["sigma_m"])
        else:
            est = self.estimate(t)
            if est is not None:
                record.update(applied=True, source="nearest", shift_e_m=est["e_m"], shift_n_m=est["n_m"],
                              ties=0, sigma_m=est["sigma_m"], nearest_s=round(abs(t - est["t"]), 2))
            else:
                record.setdefault("why", "no tie reaches an anchored frame and no node is near in time")
