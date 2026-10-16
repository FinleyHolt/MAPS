"""The closed-loop measures of each run, from its records: SR, OSR, NE, time to identification, SPL.

    python -m benchmarks.flight.measures --root <sortie>... --truth <adjudications dir> [<split.json>]
        [--json]

The flight README says what each measure is; this is how each is read off
the records. A root is a flown or twin sortie (``episodes.jsonl`` beside
``flight_track.jsonl``) or a replay's (its sortie folder named for the
recorded one, so the same truth scores both; it records no start time and
no track, so it reads SR and NE only).

* **The commit**: the last of the row's ``visited_xy``.
* **The truth**: the analyst's entry (``<sortie>.json`` in a truth
  directory, ``{episode_id: {"target_mask", "excluded", "intel_quality"}}``,
  keyed by episode id, so it speaks for every run of its sortie flown under
  that id; a run reads its first tasking's quality, ``intel_quality[0]``):
  its hand-drawn mask, else the tasking set's footprint for the run's
  description (a split JSON given as a truth path). The mask's area
  centroid is the target.
* **The path**: the GNSS track (``flight_track.jsonl``, 1 Hz) and the fused
  observations' centres (``frames.jsonl`` ``pose_xy``) beside it;
  ``track_source`` names the one the closest approach came off.

OSR's width is one camera HFOV ground width at the height ``agl_ft``: the
fix nearest the target's, else the observations' median camera height
(``keyframes (median)``), else the median commanded waypoint height
(``agl_source`` names which); a miss on a recording holding under 95% of
the run reads unknown. Time to identification runs from the tasking to the
end the row stands at. SPL is ``S * l / max(p, l)``: ``p`` the GNSS path
from the crew's last pass of the loiter waypoint before the first ``leg``
row (the crew side's first word on a flight: a note said as its leg is
posted or re-asked, its outcome, or that no leg is flown) to that end,
plus the way back to the commit a crew flies (one turn and a straight, or
a teardrop, at the median radius the crews turned on routed legs), ``l``
the straight from the loiter waypoint to the target; undefined for a
success the recording holds under 95% of. A run with an ``excluded`` reason
is out of every rate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Any

from benchmarks.citynav.action_snap import bearing_deg
from benchmarks.citynav.analysis.rows import is_infra_abort, iter_json_rows
from benchmarks.citynav.episode_workdir import CONVERSATION_JSONL, FRAMES_JSONL, TRACK_JSONL
from benchmarks.citynav.episodes import EPISODES_JSONL
from benchmarks.flight.airframe import C12_SENSOR
from maps.llm.http import TRANSCRIPT_JSONL
from maps.perception.ortho.ground_level import GROUND_LEVEL
from maps.perception.geometry.sensor_spec import resolve_sensor
from maps.perception.ortho.sensor_frame import footprint_m
from maps.search.analyze_scene.geom_compute import point_in_polygon

FT_PER_M = 1.0 / 0.3048
#: A gap between fixes longer than this is a hole in the recording.
TRACK_GAP_S = 3.0
#: A success's SPL needs the recording to hold this share of its run.
SPL_MIN_COVERAGE = 0.95
#: A pass of the loiter waypoint counts within this distance of it.
LOITER_PASS_M = 1000.0
#: A sustained turn on a routed leg: the ground track turning faster than this
#: for at least ``TURN_MIN_S`` one way, at flying speed; one sweeping
#: ``ORBIT_DEG`` or more is an orbit and is left out.
TURN_RATE_DEG_S = 1.5
TURN_MIN_S = 10
FLYING_MPS = 50.0
ORBIT_DEG = 300.0
#: The radius the way back is turned at when no turn was read.
TURN_RADIUS_FALLBACK_M = 1500.0


def _utc(t: float | None) -> str:
    return "" if t is None else dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4] + "Z"


def centroid(ring: list[list[float]]) -> tuple[float, float]:
    """Area centroid of a polygon ring (shoelace); the vertex mean when degenerate."""
    a = cx = cy = 0.0
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i][0], ring[i][1]
        x1, y1 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a) < 1e-9:
        return sum(p[0] for p in ring) / n, sum(p[1] for p in ring) / n
    return cx / (3.0 * a), cy / (3.0 * a)


# -- the truth -------------------------------------------------------------------


class Truth:
    """The analyst's entries by sortie name (a directory of ``<sortie>.json``)
    and the tasking set's footprints by description (split JSONs). A flown
    episode id is the run's place in its sortie's tasking, not in the file, so
    the description is the join; a description two entries give different
    footprints joins nothing."""

    def __init__(self, paths: list[Path]) -> None:
        self.dirs = [p for p in paths if p.is_dir()]
        self.masks: dict[str, list[list[float]] | None] = {}
        for p in paths:
            if p.is_file():
                from benchmarks.citynav.adapter import parse_episode  # noqa: PLC0415

                for i, r in enumerate(json.loads(p.read_text())):
                    ep = parse_episode(r, row_index=i)
                    if ep.target_mask:
                        ring = [[float(x), float(y)] for x, y in ep.target_mask]
                        seen = self.masks.setdefault(ep.description, ring)
                        if seen is not None and seen != ring:
                            self.masks[ep.description] = None

    def entries(self, sortie: str) -> dict[str, dict[str, Any]]:
        for d in self.dirs:
            if (d / f"{sortie}.json").is_file():
                return json.loads((d / f"{sortie}.json").read_text())
        return {}

    def ring(self, entry: dict[str, Any], row: dict[str, Any]) -> tuple[list[list[float]] | None, str | None]:
        ring = entry.get("target_mask") or []
        if len(ring) >= 3:
            return [[float(p[0]), float(p[1])] for p in ring], "hand-drawn"
        mask = self.masks.get(str(row.get("description") or ""))
        return (mask, "tasking set") if mask else (None, None)


# -- the commit ------------------------------------------------------------------


def judged_xy(row: dict[str, Any]) -> tuple[float, float] | None:
    """The commit: the last visited point; None when nothing flew (the start alone)."""
    visited = row.get("visited_xy") or []
    return (float(visited[-1][0]), float(visited[-1][1])) if len(visited) >= 2 else None


# -- the way back: Dubins paths at one turn radius -------------------------------
# Tile metres, heading 0 = north, clockwise; sense +1 a right turn, -1 a left.

#: An arc shorter than this is no turn: the aircraft is on heading.
ARC_EPS_DEG = 1.0
#: Each Dubins word's three senses, in flying order.
SENSES = {"LSL": (-1, 0, -1), "RSR": (1, 0, 1), "LSR": (-1, 0, 1), "RSL": (1, 0, -1),
          "LRL": (-1, 1, -1), "RLR": (1, -1, 1)}


def turn_centre(p, heading_deg: float, sense: int, r: float) -> tuple[float, float]:
    psi = math.radians(heading_deg)
    return p[0] + sense * r * math.cos(psi), p[1] - sense * r * math.sin(psi)


def _arc(x: float) -> float:
    a = x % (2.0 * math.pi)
    return 0.0 if a > 2.0 * math.pi - 1e-9 else a


def _csc(a0, a1, s0, s1, cx0, cy0, cx1, cy1, r):
    dx, dy = cx1 - cx0, cy1 - cy0
    gap = math.hypot(dx, dy)
    if s0 == s1:
        theta, straight = (math.atan2(dx, dy) if gap > 1e-9 else a0), gap
    else:
        if gap < 2.0 * r or gap < 1e-9:
            return None
        theta = math.atan2(dx, dy) + s0 * math.asin(2.0 * r / gap)
        straight = math.sqrt(gap * gap - 4.0 * r * r)
    arc0, arc1 = _arc(s0 * (theta - a0)), _arc(s1 * (a1 - theta))
    return (arc0 + arc1) * r + straight, arc0 * r, straight, arc1 * r


def _ccc(a0, a1, s, cx0, cy0, cx1, cy1, r):
    dx, dy = cx1 - cx0, cy1 - cy0
    gap = math.hypot(dx, dy)
    if gap > 4.0 * r or gap < 1e-9:
        return None
    h = math.sqrt(4.0 * r * r - 0.25 * gap * gap) / gap
    mx, my = cx0 + 0.5 * dx + s * h * dy, cy0 + 0.5 * dy - s * h * dx
    t0 = math.atan2(mx - cx0, my - cy0) + s * math.pi / 2.0
    t1 = math.atan2(mx - cx1, my - cy1) + s * math.pi / 2.0
    arc0, arcm, arc1 = _arc(s * (t0 - a0)), _arc(-s * (t1 - t0)), _arc(s * (a1 - t1))
    return (arc0 + arcm + arc1) * r, arc0 * r, arcm * r, arc1 * r


def solve(p0, h0: float, p1, h1: float, r: float) -> tuple[str, tuple[float, float, float, float]]:
    """The shortest Dubins word from pose ``(p0, h0)`` to ``(p1, h1)`` and its ``(length, first_m, middle_m, last_m)``."""
    a0, a1 = math.radians(h0), math.radians(h1)
    rc0, lc0 = turn_centre(p0, h0, 1, r), turn_centre(p0, h0, -1, r)
    rc1, lc1 = turn_centre(p1, h1, 1, r), turn_centre(p1, h1, -1, r)
    words = {"RSR": _csc(a0, a1, 1, 1, *rc0, *rc1, r), "LSL": _csc(a0, a1, -1, -1, *lc0, *lc1, r),
             "RSL": _csc(a0, a1, 1, -1, *rc0, *lc1, r), "LSR": _csc(a0, a1, -1, 1, *lc0, *rc1, r),
             "RLR": _ccc(a0, a1, 1, *rc0, *rc1, r), "LRL": _ccc(a0, a1, -1, *lc0, *lc1, r)}
    kind = min((k for k, v in words.items() if v is not None), key=lambda k: words[k][0])
    return kind, words[kind]


# -- the path --------------------------------------------------------------------


def _window(track, t0: float, t1: float):
    ts = [r["t_unix"] for r in track]
    return track[bisect_left(ts, t0):bisect_left(ts, t1 + 1e-6)]


def _at(track, t: float) -> tuple[float, float] | None:
    """The track's position at ``t``, interpolated between fixes; the last fix for
    ``TRACK_GAP_S`` past it (a gap the coverage counts as held); None outside."""
    if not track or t < track[0]["t_unix"] or t > track[-1]["t_unix"] + TRACK_GAP_S:
        return None
    i = bisect_left([r["t_unix"] for r in track], t)
    if i == len(track):
        return track[-1]["x"], track[-1]["y"]
    b = track[i]
    if i == 0 or b["t_unix"] == t:
        return b["x"], b["y"]
    a = track[i - 1]
    w = (t - a["t_unix"]) / (b["t_unix"] - a["t_unix"])
    return a["x"] + w * (b["x"] - a["x"]), a["y"] + w * (b["y"] - a["y"])


def _seg_dist(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = dx * dx + dy * dy
    t = 0.0 if n == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / n))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def closest_approach(path, p) -> tuple[float, dict[str, Any]]:
    """Distance from ``p`` to the path between its fixes, and the fix nearest it."""
    pts = [(f["x"], f["y"]) for f in path]
    if len(pts) == 1:
        return math.dist(pts[0], p), path[0]
    i = min(range(len(pts) - 1), key=lambda k: _seg_dist(p, pts[k], pts[k + 1]))
    return _seg_dist(p, pts[i], pts[i + 1]), path[i] if math.dist(pts[i], p) <= math.dist(pts[i + 1], p) else path[i + 1]


def nearest_fix(fixes, p) -> tuple[float, dict[str, Any]]:
    """Distance from ``p`` to the nearest of ``fixes`` read as points, and that fix."""
    f = min(fixes, key=lambda f: math.dist((f["x"], f["y"]), p))
    return math.dist((f["x"], f["y"]), p), f


def track_coverage(track, t0: float, t1: float) -> float:
    """The share of ``[t0, t1]`` the recording holds."""
    fixes = _window(track, t0, t1) if t1 > t0 else []
    if not fixes:
        return 0.0
    ts = [t0] + [f["t_unix"] for f in fixes] + [t1]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    return round(sum(g for g in gaps if g <= TRACK_GAP_S) / (t1 - t0), 3)


def last_pass(track, xy, t0: float, t1: float, within_m: float) -> tuple[float, float] | None:
    """The last local minimum of distance to ``xy`` in ``[t0, t1]`` within ``within_m``: ``(t, distance)``."""
    w = _window(track, t0, t1)
    d = [math.dist((f["x"], f["y"]), xy) for f in w]
    best = None
    for i, di in enumerate(d):
        if (i == 0 or di < d[i - 1]) and (i == len(d) - 1 or di <= d[i + 1]) and di <= within_m:
            best = (w[i]["t_unix"], di)
    return best


def keyframe_poses(frames: list[dict[str, Any]], t0: float, t1: float, map_name: str) -> list[dict[str, Any]]:
    """The centre of each observation the run fused (``frames.jsonl`` ``pose_xy``) and its camera height
    above ``map_name``'s ground (``altitude_m`` is on the tile frame), as track fixes."""
    out = []
    for f in frames:
        xy, utc = f.get("pose_xy"), f.get("utc")
        if xy and utc:
            t = dt.datetime.fromisoformat(str(utc).replace("Z", "+00:00")).timestamp()
            if t0 <= t <= t1:
                out.append({"t_unix": t, "x": float(xy[0]), "y": float(xy[1]),
                            "alt_agl_m": frame_agl(f, map_name), "src": "keyframes"})
    return sorted(out, key=lambda f: f["t_unix"])


def frame_agl(frame: dict[str, Any], map_name: str) -> float | None:
    """An observation's camera height above the map's ground; None when it records none."""
    alt = frame.get("altitude_m")
    return None if alt is None else float(alt) - GROUND_LEVEL[map_name]


def _agl(fix, frames: list[dict[str, Any]], events, map_name: str) -> tuple[float | None, str | None]:
    """Height above ground at ``fix``: its own (a GNSS fix or an observation's), else the observations' median, else the waypoints' commanded."""
    if fix.get("alt_agl_m") is not None:
        return float(fix["alt_agl_m"]), fix.get("src", "GNSS track")
    alts = [frame_agl(f, map_name) for f in frames if f.get("altitude_m") is not None]
    if alts:
        return statistics.median(alts), "keyframes (median)"
    asked = [float(e["altitudeAglMeters"]) for e in events if e.get("kind") == "waypoint" and e.get("altitudeAglMeters")]
    return (statistics.median(asked), "waypoint altitude") if asked else (None, None)


def first_leg_m(p, h: float, q, r: float) -> float | None:
    """The length of one turn at ``r`` and the straight to ``q`` from pose ``(p, h)``, the shorter side; None inside both circles."""
    best = None
    for sense in (1, -1):
        c = turn_centre(p, h, sense, r)
        d = math.dist(c, q)
        if d <= r:
            continue
        phi = bearing_deg(c, q) - sense * math.degrees(math.acos(r / d))
        arc = (sense * (phi - (h - sense * 90.0))) % 360.0
        if arc > 360.0 - ARC_EPS_DEG:
            arc = 0.0
        length = math.radians(arc) * r + math.sqrt(d * d - r * r)
        best = length if best is None else min(best, length)
    return best


def return_leg_m(track, t: float, q, r: float) -> float:
    """The way from the aircraft at ``t`` to ``q`` as a crew flies it at ``r``:
    the shorter of one turn and the straight, and a teardrop (two turns the
    opposite ways, a straight or none between), arrival heading free."""
    i = min(range(len(track)), key=lambda k: abs(track[k]["t_unix"] - t))
    xy = _at(track, t) or (track[i]["x"], track[i]["y"])
    heading = float(track[i].get("track_deg") or 0.0)
    best = first_leg_m(xy, heading, q, r)
    eps = math.radians(ARC_EPS_DEG) * r
    for h in range(0, 360):
        kind, (length, d0, dm, d1) = solve(xy, heading, q, float(h), r)
        s0, sm, s1 = SENSES[kind]
        turns = [s for s, d in ((s0, d0), (sm, dm), (s1, d1)) if s != 0 and d > eps]
        if len(turns) == 2 and turns[0] != turns[1] and (best is None or length < best):
            best = length
    if best is None:
        h = min(range(0, 360, 5), key=lambda a: solve(xy, heading, q, float(a), r)[1][0])
        best = solve(xy, heading, q, float(h), r)[1][0]
    return best


def routed_windows(root: Path) -> list[tuple[float, float]]:
    """When the crews flew to a waypoint: from each waypoint handed to the next leg line, per run."""
    out = []
    for row in iter_json_rows(root / EPISODES_JSONL):
        t0 = row.get("started_at_utc")
        if not t0 or not row.get("workdir"):
            continue
        handed = None
        for e in _conversation(run_bases(root, row)):
            t = float(t0) + float(e.get("t_s") or 0.0)
            if e.get("kind") == "waypoint" and handed is None:
                handed = t
            elif e.get("kind") == "leg" and handed is not None:
                out.append((handed, t))
                handed = None
    return out


def flown_turn_radius_m(tracks, windows) -> tuple[float | None, int]:
    """The median radius the crews turned at on a routed leg (groundspeed over
    ground-track rate, per sustained turn, orbits out), and how many turns it read."""
    radii = []
    for track, wins in zip(tracks, windows):
        cur: list[tuple[float, float]] = []

        def flush() -> None:
            if len(cur) >= TURN_MIN_S and sum(abs(w) for _, w in cur) < ORBIT_DEG:
                radii.append(statistics.median(r for r, _ in cur))
            cur.clear()

        for a, b, c in zip(track, track[1:], track[2:]):
            dt_s = c["t_unix"] - a["t_unix"]
            v = float(b.get("ground_speed_mps") or 0.0)
            if (not any(w0 <= b["t_unix"] <= w1 for w0, w1 in wins) or dt_s > 2.5 or v < FLYING_MPS
                    or a.get("track_deg") is None or c.get("track_deg") is None):
                flush()
                continue
            rate = ((c["track_deg"] - a["track_deg"] + 540.0) % 360.0 - 180.0) / dt_s
            if abs(rate) < TURN_RATE_DEG_S or (cur and (rate > 0) != (cur[-1][1] > 0)):
                flush()
                if abs(rate) < TURN_RATE_DEG_S:
                    continue
            cur.append((v / math.radians(abs(rate)), rate * dt_s / 2.0))
        flush()
    return (round(statistics.median(radii), 1) if radii else None), len(radii)


# -- one run -------------------------------------------------------------------


def run_bases(root: Path, row: dict[str, Any]) -> list[Path]:
    """A ledger row's run folder: the sortie's copy, then the ground station's (``<sortie>_grnd``)."""
    wd = str(row.get("workdir") or "")
    return [root / wd, root.parent / f"{root.name}_grnd" / root.name / wd] if wd else []


def sortie_of(run_dir: Path) -> Path:
    """The sortie a run folder belongs to, either copy :func:`run_bases` names."""
    sortie = run_dir.parent
    return sortie.parent.parent / sortie.name if sortie.parent.name.endswith("_grnd") else sortie


def _conversation(bases: list[Path]) -> list[dict[str, Any]]:
    return next((list(iter_json_rows(b / CONVERSATION_JSONL)) for b in bases if (b / CONVERSATION_JSONL).is_file()), [])


def _frames(bases: list[Path]) -> list[dict[str, Any]]:
    """The run's ``frames.jsonl`` off its first copy that holds one (:func:`run_bases`)."""
    return next((list(iter_json_rows(b / FRAMES_JSONL)) for b in bases if (b / FRAMES_JSONL).is_file()), [])


def ended(row: dict[str, Any]) -> str:
    """``no run`` (status not ``ok``, or no recorded end), ``stopped`` (a
    crew end), ``failed`` (a failure not the serve's: a scored miss), else
    the row's ``ended_by``: ``reply`` (the model's report) is the one end
    that scores."""
    if row.get("status") != "ok":
        return "no run"
    if row.get("ended_by") == "stopped":
        return "stopped"
    if not row.get("ended_by") and row.get("error"):
        return "failed"
    return str(row.get("ended_by") or "no run")


def tokens(bases: list[Path]) -> dict[str, Any]:
    """The run's reasoner use off its transcript (``maps.llm.flight_summary``)."""
    from maps.llm.flight_summary import summarize  # noqa: PLC0415

    flight = next((b / "flight" for b in bases if (b / "flight" / "llm" / TRANSCRIPT_JSONL).is_file()), None)
    llm = summarize(flight) if flight else {}
    keys = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")
    if not llm.get("requests"):
        return {k: None for k in (*keys, "reasoner_calls")}
    return {**{k: int(llm.get(k) or 0) for k in keys},
            "reasoner_calls": int(llm.get("requests") or 0) - int(llm.get("unanswered") or 0)}


def run_measures(root: Path, row: dict[str, Any], *, entry: dict[str, Any], truth: Truth, track,
                 turn_radius_m: float) -> dict[str, Any]:
    bases = run_bases(root, row)
    events = _conversation(bases)
    end = ended(row)
    xy = judged_xy(row)
    determined = end == "reply" and xy is not None
    reason = entry.get("excluded")
    params = next((r.get("params") or {} for b in bases for r in iter_json_rows(b / "flight" / "llm" / TRANSCRIPT_JSONL)
                   if r.get("kind") == "params"), {})
    ring, truth_source = truth.ring(entry, row)
    target = centroid(ring) if ring else None
    # The end the row stands at (``Mission._record_reply``): the first, or a later plain reply.
    ends = [e for e in events if e.get("kind") == "mission_end"]
    stands = ends[:1] + [e for e in ends[1:] if e.get("ended_by") == "reply"]
    t_end = float(stands[-1]["t_s"]) if stands else None
    if t_end is None and row.get("elapsed_s"):
        t_end = float(row["elapsed_s"])
    t_start = float(row["started_at_utc"]) if row.get("started_at_utc") else None
    t_stop = None if t_start is None or t_end is None else t_start + t_end
    quality = list(entry.get("intel_quality") or [None])
    out: dict[str, Any] = {
        "sortie": root.name, "run_no": row.get("run_no"), "episode_id": row.get("episode_id"),
        "replay_of": row.get("replay_of"), "model": params.get("model"),
        "reasoning_effort": params.get("reasoning_effort"), "intel_quality": quality[0],
        "end": end, "excluded": " ".join(str(reason or "").split()) or None,
        "infra": is_infra_abort(row),
        "determination": determined,
        "commit_xy": None if xy is None else [round(xy[0], 2), round(xy[1], 2)],
        "truth_xy": None if target is None else [round(target[0], 2), round(target[1], 2)],
        "truth_source": truth_source, "flights": row.get("flights_flown"), "turns": row.get("turns_used"),
        "success": None, "ne_ft": None, "oracle_success": None,
        "closest_approach_ft": None, "hfov_width_ft": None, "agl_ft": None, "agl_source": None,
        "track_source": None,
        "track_coverage": None, "time_to_id_s": round(t_end, 1) if determined and t_stop is not None else None,
        "spl": None, "spl_note": None, "path_ft": None, "final_leg_ft": None, "shortest_ft": None,
        "start_utc": _utc(t_start), "stop_utc": _utc(t_stop), **tokens(bases),
    }
    if target is None:
        return out
    out["success"] = determined and point_in_polygon(xy, ring)
    if determined:
        out["ne_ft"] = round(math.dist(xy, target) * FT_PER_M, 1)
    if t_start is None:
        return out
    t_end_run = t_stop if t_stop is not None else t_start
    success = out["success"]
    if not success:
        out["spl"] = 0.0
    out["track_coverage"] = track_coverage(track, t_start, t_end_run)
    flown = _window(track, t_start, t_end_run) if track else []
    # The observations are centred on ground the camera held: they can show the
    # target came within the width where the track does not.
    thin = out["track_coverage"] < SPL_MIN_COVERAGE
    frames = _frames(bases)
    poses = keyframe_poses(frames, t_start, t_end_run, row.get("map_name"))
    if success and len(flown) < 2:
        out["spl_note"] = "undefined: the recording does not cover the run"
    # The GNSS track is joined between its fixes; each observation stands alone.
    cands = ([(closest_approach(flown, target), "GNSS track")] if flown else []) + \
        ([(nearest_fix(poses, target), "keyframe poses")] if poses else [])
    if not cands:
        out["track_source"] = "none"
        return out
    # The path the closest approach came off names the track.
    (d, nearest), out["track_source"] = min(cands, key=lambda c: c[0][0])
    out["closest_approach_ft"] = round(d * FT_PER_M, 1)
    agl, out["agl_source"] = _agl(nearest, frames, events, row.get("map_name"))
    width = 0.0
    if agl is not None:
        out["agl_ft"] = round(agl * FT_PER_M, 1)
        width = footprint_m(resolve_sensor(C12_SENSOR), agl)[0]
        out["hfov_width_ft"] = round(width * FT_PER_M, 1)
        out["oracle_success"] = d <= width
    if out["oracle_success"] is False and thin:
        out["oracle_success"] = None  # a thin recording cannot show the aircraft never came near
    if len(flown) < 2:
        return out
    loiter = row.get("start_xy")
    first = next((t_start + float(e["t_s"]) for e in events if e.get("kind") == "leg"), t_end_run)
    passed = last_pass(track, (float(loiter[0]), float(loiter[1])), t_start, first, LOITER_PASS_M) if loiter else None
    p0 = passed[0] if passed else t_start
    origin = tuple(map(float, loiter)) if loiter else _at(track, p0)
    leg = 0.0
    at = _at(track, t_end_run)
    if xy is not None and (at is None or math.dist(at, xy) > width / 2.0):
        leg = return_leg_m(track, t_end_run, xy, turn_radius_m)
    w = _window(track, p0, t_end_run)
    p = sum(math.dist((a["x"], a["y"]), (b["x"], b["y"])) for a, b in zip(w, w[1:]))
    shortest = math.dist(origin, target)
    out["path_ft"], out["final_leg_ft"] = round(p * FT_PER_M, 1), round(leg * FT_PER_M, 1)
    out["shortest_ft"] = round(shortest * FT_PER_M, 1)
    if success:
        if thin:
            out["spl_note"] = f"undefined: the recording holds {out['track_coverage']:.0%} of the run"
        else:
            out["spl"] = round(shortest / max(p + leg, shortest), 4) if max(p + leg, shortest) > 0 else None
    return out


def collect(roots: list[Path], truth: Truth) -> tuple[list[dict[str, Any]], float, int]:
    """Every run under ``roots``, with the crews' turn radius over them all."""
    tracks = [sorted(iter_json_rows(r / TRACK_JSONL), key=lambda f: f["t_unix"]) for r in roots]
    radius, turns = flown_turn_radius_m(tracks, [routed_windows(r) for r in roots])
    radius = radius or TURN_RADIUS_FALLBACK_M
    out = []
    for root, track in zip(roots, tracks):
        entries = truth.entries(root.name)
        for row in iter_json_rows(root / EPISODES_JSONL):
            out.append(run_measures(root, row, entry=entries.get(str(row.get("episode_id") or "")) or {},
                                    truth=truth, track=track, turn_radius_m=radius))
    out.sort(key=lambda r: (r["sortie"], int(r["run_no"] or 0)))
    return out, radius, turns


def _rate(rows, key: str) -> dict[str, Any]:
    vals = [r[key] for r in rows if r[key] is not None]
    k = sum(1 for v in vals if v)
    return {"k": k, "n": len(vals), "pct": round(100.0 * k / len(vals), 1) if vals else None}


def _median(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 1) if vals else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The rates over the runs kept that carry a truth; times and tokens over the
    runs kept. A run the serve failed under (``rows.is_infra_abort``, the
    bench's rule) is out of every rate and named in ``infra``."""
    infra = [f"{r['sortie']} run {r['run_no']}" for r in rows if r["infra"] and not r["excluded"]]
    kept = [r for r in rows if not r["excluded"] and not r["infra"]]
    scored = [r for r in kept if r["truth_xy"] is not None]
    spl = [r["spl"] for r in scored if r["spl"] is not None]
    return {"runs": len(rows), "kept": len(kept), "scored": len(scored), "infra": infra,
            "sr": _rate(scored, "success"),
            "osr": _rate(scored, "oracle_success"),
            "ne_median_ft": _median(r["ne_ft"] for r in scored),
            "time_to_id_median_s": _median(r["time_to_id_s"] for r in kept),
            "spl_pct": round(100.0 * statistics.fmean(spl), 1) if spl else None, "spl_n": len(spl),
            "total_tokens_median": _median(r["total_tokens"] for r in kept)}


def by(rows, *keys: str) -> dict[str, dict[str, Any]]:
    """The summary per value of ``keys`` (joined by a space)."""
    def name(r) -> str:
        return " ".join(str(r[k]) for k in keys)
    return {v: summarize([r for r in rows if name(r) == v]) for v in sorted({name(r) for r in rows})}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, nargs="+", required=True, help="flown or replay sortie roots")
    ap.add_argument("--truth", type=Path, nargs="+", required=True,
                    help="a directory of <sortie>.json analyst entries, and/or the tasking set's split JSON")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    roots = sorted(r for r in args.root if (r / EPISODES_JSONL).is_file())
    rows, radius, turns = collect(roots, Truth(args.truth))
    doc = {"turn_radius_m": radius, "turns_read": turns, "summary": summarize(rows),
           "by_intel_quality": by(rows, "intel_quality"), "by_effort": by(rows, "model", "reasoning_effort"),
           "runs": rows}
    if args.json:
        json.dump(doc, sys.stdout, indent=1)
        print()
    else:
        def line(name: str, s: dict[str, Any]) -> str:
            return (f"{name:14s} runs {s['runs']:3d} scored {s['scored']:3d}  SR {s['sr']['k']}/{s['sr']['n']}"
                    f"  OSR {s['osr']['k']}/{s['osr']['n']}"
                    f"  NE med {s['ne_median_ft']} ft  time to ID med {s['time_to_id_median_s']} s"
                    f"  SPL {s['spl_pct']}% (n {s['spl_n']})  tokens med {s['total_tokens_median']}")
        print(line("all", doc["summary"]))
        if doc["summary"]["infra"]:
            print(f"out of every rate (the serve failed under them): {', '.join(doc['summary']['infra'])}")
        for k, v in doc["by_intel_quality"].items():
            print(line(f"intel {k}", v))
        for k, v in doc["by_effort"].items():
            print(line(f"effort {k}", v))
        print(f"turn radius flown (median) {radius} m over {turns} turns" if turns
              else f"turn radius {radius} m (fallback: no turn read)")
        for r in rows:
            print(f"  {r['sortie']} run {r['run_no']}: {r['end']}, SR {r['success']}, OSR {r['oracle_success']}, NE {r['ne_ft']} ft, "
                  f"time to ID {r['time_to_id_s']} s, {r['flights']} flights / {r['turns']} turns, SPL {r['spl']}"
                  + (f", excluded: {r['excluded']}" if r["excluded"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
