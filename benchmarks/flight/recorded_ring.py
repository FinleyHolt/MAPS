"""A camera recording read back as the aircraft's ring: a :class:`~benchmarks.flight.vehicle.Vehicle` over ``sensors.mcap``.

The recorder wrote every camera frame and every fix and attitude to one ROS 2
MCAP on the receiver's UTC (``log_time``, nanoseconds). This module reads it
back the way the aircraft's ring held it, over one time window, streamed and
never loaded whole:

* **Poses** are the recording's own, built as the aircraft built them: every
  ``/gnss/pvt`` fix and ``/nav/attitude`` document in the window into a
  :class:`NavTrack`, and each frame tagged with the track's pose at its stamp
  (``nav_track.pose_tag``), the antenna's, so the live source applies the
  lever arm and the boresight once, as in flight.
* **The index** is the frames the ring kept: one each
  ``1 / RING_HZ`` s of log time, the rate the recorder spilled its ring at.
  ponytail: the newest quarter-gigabyte the ring held at the camera's full
  rate is read at the spill rate too; seconds of extra frames per pass.
* **The clock** starts at the window's start and moves only while the
  harness waits, and the index holds the frames stamped up to it: a leg
  posted to the recording arrives when the recording reaches its ground.

A fix that cannot pose a frame (missing a field, further than
``MAX_GAP_S`` from any fix, worse than ``MAX_H_ACC_M``) leaves the frame
out of the index, as the aircraft's ring carried no tag for it.
"""

from __future__ import annotations

import bisect
import json
import math
import struct
import threading
from pathlib import Path
from typing import Any, Iterator

from benchmarks.flight.airframe import M_PER_DEG_LAT
from benchmarks.flight.nav_track import Pose, pose_tag

TOPIC_CAMERA = "/camera/image_raw/compressed"
TOPIC_PVT = "/gnss/pvt"
TOPIC_ATT = "/nav/attitude"
#: The rate the recorder kept its ring at, frames a second.
RING_HZ = 2.0
#: Further than this from any fix, a frame has no pose.
MAX_GAP_S = 1.0
#: A fix worse than this horizontally poses no frame.
MAX_H_ACC_M = 10.0
#: The track's turn rate is read over the fixes this far either side of a frame, and needs this span of them.
TURN_WINDOW_S = 2.0
TURN_MIN_SPAN_S = 1.5
G_MPS2 = 9.80665
#: Fixes and attitude are read this far either side of the window, so its edges have a track.
NAV_MARGIN_NS = 60 * 10**9
#: The frames before a run's start the ring already held, and after its end the window still reads.
RUN_LEAD_S, RUN_TAIL_S = 300.0, 120.0

_LINEAR = ("lat", "lon", "altitudeMslMeters", "altitudeHaeMeters")
_NEEDED = _LINEAR + ("speedMps", "trackDeg", "accuracyMeters", "climbMps")


def messages(path: Path, topics: list[str], start_ns: int | None = None, end_ns: int | None = None) -> Iterator:
    """``(channel, message)`` on ``topics`` in ``[start_ns, end_ns)``, through
    the file's index when it has one (a window is a seek), else streamed from
    the top to the first torn record."""
    from mcap.exceptions import McapError  # noqa: PLC0415
    from mcap.reader import NonSeekingReader, make_reader  # noqa: PLC0415

    with Path(path).open("rb") as fh:
        try:
            for _schema, channel, message in make_reader(fh).iter_messages(topics=topics, start_time=start_ns,
                                                                             end_time=end_ns):
                yield channel, message
            return
        except (McapError, ValueError, EOFError, struct.error):
            pass
        fh.seek(0)
        try:
            for _schema, channel, message in NonSeekingReader(fh).iter_messages(
                    topics=topics, start_time=start_ns, end_time=end_ns, log_time_order=False):
                yield channel, message
        except (McapError, ValueError, EOFError, struct.error):
            return


def json_docs(path: Path, topic: str, start_ns=None, end_ns=None) -> Iterator[tuple[int, dict]]:
    """``(log_time, doc)`` for every JSON document on ``topic`` in the window."""
    for _channel, message in messages(path, [topic], start_ns, end_ns):
        try:
            doc = json.loads(message.data)
        except ValueError:
            continue
        if isinstance(doc, dict):
            yield message.log_time, doc


def compressed_jpeg(data: bytes) -> bytes:
    """The image bytes of a CDR ``sensor_msgs/CompressedImage``: past the
    encapsulation, the header's stamp and frame id, and the format string."""
    def string(off: int) -> int:
        off += -off % 4
        (n,) = struct.unpack_from("<I", data, off)
        return off + 4 + n
    off = string(4 + 8)          # encapsulation, stamp; then frame_id
    off = string(off)            # format
    off += -off % 4
    (n,) = struct.unpack_from("<I", data, off)
    return bytes(data[off + 4:off + 4 + n])


# -- the track -------------------------------------------------------------------


def _usable(fix: dict) -> bool:
    return all(isinstance(fix.get(k), (int, float)) and math.isfinite(fix[k]) for k in _NEEDED)


def _velocity_en(fix: dict) -> tuple[float, float]:
    psi = math.radians(float(fix["trackDeg"]))
    return float(fix["speedMps"]) * math.sin(psi), float(fix["speedMps"]) * math.cos(psi)


def _interpolate(lo: dict, hi: dict, w: float) -> dict:
    """Position and altitudes linearly, the velocity as a vector (the track crosses north cleanly), the worse accuracy."""
    out = dict(hi if w >= 0.5 else lo)
    for key in _LINEAR:
        out[key] = float(lo[key]) + (float(hi[key]) - float(lo[key])) * w
    (e0, n0), (e1, n1) = _velocity_en(lo), _velocity_en(hi)
    ve, vn = e0 + (e1 - e0) * w, n0 + (n1 - n0) * w
    out["speedMps"], out["trackDeg"] = math.hypot(ve, vn), math.degrees(math.atan2(ve, vn)) % 360.0
    out["accuracyMeters"] = max(float(lo["accuracyMeters"]), float(hi["accuracyMeters"]))
    return out


def _extrapolate(fix: dict, dt_s: float) -> dict:
    out = dict(fix)
    ve, vn = _velocity_en(fix)
    lat = float(fix["lat"])
    out["lat"] = lat + vn * dt_s / M_PER_DEG_LAT
    out["lon"] = float(fix["lon"]) + ve * dt_s / (M_PER_DEG_LAT * max(math.cos(math.radians(lat)), 1e-6))
    climb = float(fix["climbMps"]) * dt_s
    out["altitudeMslMeters"] = float(fix["altitudeMslMeters"]) + climb
    out["altitudeHaeMeters"] = float(fix["altitudeHaeMeters"]) + climb
    return out


class NavTrack:
    """Fixes and attitudes on one time axis (seconds of log time), answering ``pose_at``.

    Between two fixes the pose is interpolated; past the nearer within
    ``MAX_GAP_S`` it is carried along that fix's velocity. Heading, roll and
    pitch come off the fused attitude documents about ``t``; without one the
    heading is the track, and the roll the bank of a coordinated turn read
    off the track's own rate (``atan(V * omega / g)``, ``estimated``), or
    zero (``track``) when the fixes about ``t`` read no rate.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fix_t: list[float] = []
        self._fix: list[dict] = []
        self._att_t: list[float] = []
        self._att: list[dict] = []

    def add_fix(self, t: float, fix: dict) -> None:
        if _usable(fix):
            with self._lock:
                i = bisect.bisect_right(self._fix_t, t)
                self._fix_t.insert(i, float(t))
                self._fix.insert(i, dict(fix))

    def add_attitude(self, t: float, doc: dict) -> None:
        """A fused attitude view poses frames; a receiver's unfused view does not."""
        if doc.get("fused") or doc.get("attitude") in ("fusion", "estimated"):
            with self._lock:
                i = bisect.bisect_right(self._att_t, t)
                self._att_t.insert(i, float(t))
                self._att.insert(i, dict(doc))

    def pose_at(self, t: float) -> Pose | None:
        with self._lock:
            return self._pose_at(float(t))

    def _pose_at(self, t: float) -> Pose | None:
        if not self._fix:
            return None
        i = bisect.bisect_left(self._fix_t, t)
        lo, hi = i - 1, i
        if lo >= 0 and hi < len(self._fix) and self._fix_t[hi] - self._fix_t[lo] <= MAX_GAP_S:
            span = self._fix_t[hi] - self._fix_t[lo]
            fix = _interpolate(self._fix[lo], self._fix[hi], (t - self._fix_t[lo]) / span if span > 0 else 1.0)
            gap = min(t - self._fix_t[lo], self._fix_t[hi] - t)
        else:
            j = min((k for k in (lo, hi) if 0 <= k < len(self._fix)), key=lambda k: abs(self._fix_t[k] - t))
            gap = abs(self._fix_t[j] - t)
            if gap > MAX_GAP_S:
                return None
            fix = _extrapolate(self._fix[j], t - self._fix_t[j])
        if float(fix["accuracyMeters"]) > MAX_H_ACC_M:
            return None
        heading, roll, pitch, attitude = float(fix["trackDeg"]), 0.0, 0.0, "track"
        fused = self._attitude_at(t)
        if fused is not None:
            heading, roll, pitch, attitude = fused
        else:
            rate = self._turn_rate(t)
            if rate is not None:
                roll = math.degrees(math.atan(float(fix["speedMps"]) * math.radians(rate) / G_MPS2))
                attitude = "estimated"
        return Pose(t=t, lat=float(fix["lat"]), lon=float(fix["lon"]), alt_msl_m=float(fix["altitudeMslMeters"]),
                    alt_hae_m=float(fix["altitudeHaeMeters"]), track_deg=float(fix["trackDeg"]) % 360.0,
                    speed_mps=float(fix["speedMps"]), h_acc_m=float(fix["accuracyMeters"]), gap_s=gap,
                    utc=str(fix.get("timestamp", "")), heading_deg=heading % 360.0, roll_deg=roll, pitch_deg=pitch,
                    attitude=attitude, climb_mps=float(fix.get("climbMps") or 0.0))

    def _turn_rate(self, t: float) -> float | None:
        """The track's rate about ``t``, degrees a second clockwise: the least-squares slope over the window."""
        lo = bisect.bisect_left(self._fix_t, t - TURN_WINDOW_S)
        hi = bisect.bisect_right(self._fix_t, t + TURN_WINDOW_S)
        times = self._fix_t[lo:hi]
        if len(times) < 3 or times[-1] - times[0] < TURN_MIN_SPAN_S:
            return None
        tracks = [float(self._fix[lo]["trackDeg"])]
        for fix in self._fix[lo + 1:hi]:  # unwrapped, so a turn through north is one slope
            tracks.append(tracks[-1] + (float(fix["trackDeg"]) - tracks[-1] + 180.0) % 360.0 - 180.0)
        tm, km = sum(times) / len(times), sum(tracks) / len(tracks)
        return sum((a - tm) * (b - km) for a, b in zip(times, tracks)) / sum((a - tm) ** 2 for a in times)

    def _attitude_at(self, t: float) -> tuple[float, float, float, str] | None:
        if not self._att:
            return None
        i = bisect.bisect_left(self._att_t, t)
        lo, hi = i - 1, i
        if lo >= 0 and hi < len(self._att) and self._att_t[hi] - self._att_t[lo] <= MAX_GAP_S:
            span = self._att_t[hi] - self._att_t[lo]
            w = (t - self._att_t[lo]) / span if span > 0 else 1.0
            a, b = self._att[lo], self._att[hi]
            ha, hb = math.radians(float(a["heading_deg"])), math.radians(float(b["heading_deg"]))
            e = math.sin(ha) + (math.sin(hb) - math.sin(ha)) * w
            n = math.cos(ha) + (math.cos(hb) - math.cos(ha)) * w
            return (math.degrees(math.atan2(e, n)) % 360.0,
                    float(a["roll_deg"]) + (float(b["roll_deg"]) - float(a["roll_deg"])) * w,
                    float(a["pitch_deg"]) + (float(b["pitch_deg"]) - float(a["pitch_deg"])) * w,
                    str((b if w >= 0.5 else a).get("attitude") or "fusion"))
        j = min((k for k in (lo, hi) if 0 <= k < len(self._att)), key=lambda k: abs(self._att_t[k] - t))
        if abs(self._att_t[j] - t) > MAX_GAP_S:
            return None
        a = self._att[j]
        return float(a["heading_deg"]) % 360.0, float(a["roll_deg"]), float(a["pitch_deg"]), str(a.get("attitude") or "fusion")


# -- the ring --------------------------------------------------------------------


def run_window(row: dict[str, Any], conversation: list[dict[str, Any]]) -> tuple[int, int]:
    """The log-time window a flown run is read over: from ``RUN_LEAD_S``
    before its start (``started_at_utc``) to ``RUN_TAIL_S`` past its end."""
    start = float(row["started_at_utc"])
    end = max([float(row.get("elapsed_s") or 0.0)] + [float(r.get("t_s") or 0.0) for r in conversation])
    return int((start - RUN_LEAD_S) * 1e9), int((start + end + RUN_TAIL_S) * 1e9)


class RecordedRing:
    """The camera recording ``frames`` as the vehicle's ring over ``[start_ns,
    end_ns)``, posed off ``nav`` (the recordings holding the fixes and
    attitude; the camera recording itself by default). The clock starts at
    ``start_ns``; ``clock_ns`` starts it elsewhere."""

    def __init__(self, frames: Path, *, nav: list[Path] | None = None, start_ns: int, end_ns: int,
                 clock_ns: int | None = None) -> None:
        self.path = Path(frames)
        self.last_error = ""
        self.legs: list[Any] = []
        self._lock = threading.Lock()
        self.track = NavTrack()
        for mcap in nav or [self.path]:
            for t_ns, fix in json_docs(mcap, TOPIC_PVT, start_ns - NAV_MARGIN_NS, end_ns + NAV_MARGIN_NS):
                self.track.add_fix(t_ns / 1e9, fix)
            for t_ns, doc in json_docs(mcap, TOPIC_ATT, start_ns - NAV_MARGIN_NS, end_ns + NAV_MARGIN_NS):
                self.track.add_attitude(t_ns / 1e9, doc)
        #: ``{"seq", "log_time", "t_host", "pose"}`` per frame the ring kept, oldest first.
        self.rows: list[dict[str, Any]] = []
        last = None
        for _channel, message in messages(self.path, [TOPIC_CAMERA], start_ns, end_ns):
            lt = int(message.log_time)
            if last is not None and lt - last < 1e9 / RING_HZ:
                continue
            pose = self.track.pose_at(lt / 1e9)
            if pose is None:
                continue
            last = lt
            self.rows.append({"seq": int(message.sequence), "log_time": lt, "t_host": lt / 1e9,
                              "pose": pose_tag(pose)})
        self._by_seq = {r["seq"]: r for r in self.rows}
        self._stamps = [r["log_time"] for r in self.rows]
        #: The clock, nanoseconds of log time: a float second this far from 1970 cannot hold a nanosecond.
        self.t_ns = int(start_ns if clock_ns is None else clock_ns)

    def post(self, leg) -> None:
        self.legs.append(leg)

    def now(self) -> float:
        return self.t_ns / 1e9

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.t_ns += max(0, int(round(float(seconds) * 1e9)))

    def pose(self, t: float | None = None) -> Pose | None:
        return self.track.pose_at(self.now() if t is None else t)

    def index(self, since: int | None = None) -> list[dict[str, Any]]:
        upto = bisect.bisect_right(self._stamps, self.t_ns)
        lo = 0 if since is None else bisect.bisect_left(self._stamps, int(since))
        return [dict(r) for r in self.rows[lo:upto]]

    def frame(self, seq: int):
        """``(jpeg, t_host, log_time, seq, tag)`` read off the recording at the frame's stamp."""
        row = self._by_seq.get(int(seq))
        if row is None:
            self.last_error = f"no frame {seq} in the ring"
            return None
        for _channel, message in messages(self.path, [TOPIC_CAMERA], row["log_time"], row["log_time"] + 1):
            if int(message.log_time) == row["log_time"]:
                self.last_error = ""
                return compressed_jpeg(message.data), row["t_host"], row["log_time"], row["seq"], row["pose"]
        self.last_error = f"frame {seq} is not in {self.path.name} at its stamp"
        return None
