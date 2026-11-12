"""A flight on the aircraft: the leg posted for an area, its arrival, and what
the pass held.

A flight is a leg flown for an AREA (:meth:`Legs.fly`). It arrives when the
recording holds the area (:meth:`Legs.wait_arrival`), a pass that flew the
waypoint and missed a side of the ground is asked again, and the flight's
observation is the frames that held the area, fused. A :class:`Vehicle`
(``vehicle.py``) is handed each :class:`Leg` through ``post``.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from typing import Any

from benchmarks.flight.live_source import LiveSource
from benchmarks.flight.vehicle import Leg
from maps.perception.observation import ground_sides_m

#: The engine's hydrated search stack (``EpisodeEngine.build_episode``), reached through the bench.
Hydrated = Any

# A leg's lines are the mission's: one logger for the flight.
log = logging.getLogger("flight.mission")

#: How long one wait on a leg's arrival runs before the leg is given up,
#: counted from the aircraft's last moment on the ground: the cap ends a leg
#: the crew never fly. Each short-pass re-fly starts a fresh wait.
ARRIVAL_TIMEOUT_S = 1800.0
#: A leg has arrived when the recording holds this share of the area it was
#: flown for. ponytail: a share rather than all of it, because a footprint's
#: rim cells sit under a frame's edge at the pass's longest lever and nine
#: tenths of a house is the house; the leg row states the points held.
COVERED_FRACTION = 0.9
#: How many of a footprint's cells sample its area for the arrival rule.
AREA_POINTS = 48
#: How many times the recording is asked whether it holds the area before a
#: leg is posted anyway, and the pause between asks: a read that never
#: answered is not ground the pass never held. Bounded, because the aircraft
#: can always fly the leg.
COVERAGE_TRIES = 4
COVERAGE_RETRY_S = 3.0
#: A pass is SHORT when the aircraft came within the leg's reach of its point,
#: has since drawn this far past its closest approach and stayed there this
#: long, and the recording holds part of the area but less than
#: ``COVERED_FRACTION``: the crew flew the waypoint and the frame missed a side
#: of the ground. It is said in the transcript and asked again with the
#: waypoint moved toward the missed ground, ``SHORT_PASS_REQUESTS`` times
#: before the leg falls to the arrival timeout.
SHORT_PASS_LEAVE_M = 150.0
SHORT_PASS_LEAVE_S = 5.0
SHORT_PASS_REQUESTS = 2
#: Below this above the map's floor the aircraft is on the ground, and an
#: arrival wait does not spend its timeout.
AIRBORNE_AGL_M = 10.0
#: The most a leg waits for its point to leave the frame after arriving.
FRAME_WAIT_S = 30.0

#: What a flight asked for after Stop is answered with.
STOPPED = {"error": "the mission is stopped"}


def _holds(cov: dict[str, Any] | None) -> bool:
    """Whether a coverage answer holds its area: the arrival rule, ``COVERED_FRACTION`` of it."""
    return bool(cov) and float(cov.get("fraction") or 0.0) >= COVERED_FRACTION


def _held_points(cov: dict[str, Any] | None) -> dict[str, Any] | None:
    """A coverage answer as a run record states it: the share held, one flag per area point, the frames asked."""
    if not cov:
        return None
    return {
        "fraction": cov["fraction"],
        "held": [b is not None for b in cov["best"]],
        "frames": cov["frames"],
    }


def leg_span_m(h: Hydrated) -> float:
    """The run's observation span in metres: what one frame holds along its
    long side at the mission's height."""
    return float(max(ground_sides_m(float(h.state.altitude_m), h.ground_z_m, h.sensor)))


def leg_area(h: Hydrated, token: str, xy) -> list[tuple[float, float]]:
    """The ground a leg is flown for, in tile metres: the target's footprint
    cells (at most ``AREA_POINTS``), else a disc of half the observation span
    about the point (a centre, six points at a quarter span, twelve at half)."""
    cells = h.world.node_cells_xy(token) if token else None
    # Three cells at least: fewer is a point or a line, which no coverage
    # fraction answers. A row that thin is flown as a cover stop.
    if cells and len(cells) >= 3:
        step = max(1, -(-len(cells) // AREA_POINTS))
        return [(float(c[0]), float(c[1])) for c in cells[::step]][:AREA_POINTS]
    r = leg_span_m(h) / 2.0
    x, y = float(xy[0]), float(xy[1])
    points = [(x, y)]
    for n, radius in ((6, r / 2.0), (12, r)):
        points += [
            (
                x + radius * math.cos(2.0 * math.pi * k / n),
                y + radius * math.sin(2.0 * math.pi * k / n),
            )
            for k in range(n)
        ]
    return points


class Legs:
    """The legs of one mission: each flight's leg posted to the vehicle and
    waited on, and the pass the flight in progress held."""

    def __init__(
        self,
        *,
        live: LiveSource,
        transcript: list,
        cancel: threading.Event,
        agl_m: float,
        to_lonlat: Callable[[list], list],
        keep_observation: Callable[[Any], str | None],
        flights_flown: Callable[[], int],
    ) -> None:
        self.live = live
        self._vehicle = live.vehicle
        #: The mission's conversation log: a leg writes its ``waypoint`` and ``leg`` rows.
        self.transcript = transcript
        #: Set by the crew's Stop: nothing more is posted, fetched or waited on.
        self._cancel = cancel
        #: The height above ground the crew is commanded to fly.
        self.agl_m = agl_m
        #: An area in tile metres as WGS84, the form the recording is asked in.
        self._to_lonlat = to_lonlat
        #: Writes a flight's observation to the run's record and returns its sha.
        self._keep_observation = keep_observation
        #: How many flights the mission has flown: the step of the next leg.
        self._flights_flown = flights_flown
        #: The fused pass the flight in progress fetched when it held its area.
        self.held: tuple[Any, dict[str, Any]] | None = None
        #: The nearest the pose came to the last leg's point since it was posted, re-flies included.
        self.closest_m = float("inf")
        #: The leg being waited on (its area and the share the recording holds so far), None between legs.
        self.waiting: dict[str, Any] | None = None

    def fly(self, session, item: dict[str, Any]) -> dict[str, Any]:
        """A flight on the aircraft. An area the ring already holds is served
        without a leg; otherwise the leg is posted and waited on until the
        recording holds the area and the point leaves the frame, or until a
        wait runs ``ARRIVAL_TIMEOUT_S`` airborne (each re-fly waits afresh),
        when what the pass held is served. The ``leg``
        row says which. A stopped mission posts, fetches and perceives nothing
        more: the flight is refused."""
        if self._cancel.is_set():
            return dict(STOPPED)
        token = str(item.get("target_fid") or "")
        resolved = session.resolve(token)
        if "error" not in resolved:
            xy = [float(resolved["xy"][0]), float(resolved["xy"][1])]
            area = leg_area(session.h, token, xy)
            area_ll = self._to_lonlat(area)
            cov, unread = self.coverage_before_leg(area_ll)
            held = self._held_frame(area_ll, xy) if _holds(cov) else None
            at = f"{token} at ({xy[0]:.0f}, {xy[1]:.0f})"
            if held is not None:
                self._said(
                    f"{at} is held by the pass already: {self._held_words(held[1])}; no leg is flown"
                )
            elif not self._cancel.is_set():
                # Frames recorded after this vehicle time are the leg's own; None counts the whole ring.
                after = None
                if unread is not None:
                    log.warning(
                        "the leg to %s is posted unread: the recording did not answer in %d reads: %s",
                        at,
                        COVERAGE_TRIES,
                        unread,
                    )
                    self._said(
                        f"{at} may be held by the pass already, but the recording could not be asked "
                        f"in {COVERAGE_TRIES} reads ({unread}); the leg is flown"
                    )
                elif cov is not None and not _holds(cov):
                    log.info(
                        "the pass holds %d/%d points of %s in %d frames, short of %.0f%%; the leg is flown",
                        cov["held"],
                        cov["points"],
                        at,
                        len({b[1] for b in cov["best"] if b}),
                        100.0 * COVERED_FRACTION,
                    )
                elif cov is not None:
                    self._said(
                        f"{at} is held by the pass ({cov['held']}/{cov['points']} points), but no frame of "
                        f"it could be served ({self.live.last_refusal or 'no reason recorded'}); "
                        f"the leg is flown"
                    )
                    # The ring already holds the area, so only a pass after the post can arrive.
                    after = self._vehicle.now()
                ahead = [(float(x), float(y)) for x, y in (item.get("pass_ahead") or [])]
                held = self.post_and_wait(token, at, xy, area, area_ll, ahead, after)
            self.held = held
        if self._cancel.is_set():
            return dict(STOPPED)
        try:
            return session.fly(item)
        finally:
            # The fused pass is spent with the flight.
            self.held = None

    def post_and_wait(
        self,
        token: str,
        at: str,
        xy: list[float],
        area: list[tuple[float, float]],
        area_ll: list[tuple[float, float]],
        ahead: list[tuple[float, float]],
        after: float | None,
    ) -> tuple[Any, dict[str, Any]] | None:
        """The leg posted for ``area`` and waited on (arrival counting the
        frames after ``after`` when given), a short pass asked again up to
        ``SHORT_PASS_REQUESTS`` times; then what the pass held,
        ``(observation, record)``, or None. A stopped mission fetches nothing more."""
        step = self._flights_flown()
        self._post(Leg(target=token, point=(xy[0], xy[1]), step=step, area=area_ll, ahead=ahead))
        # The leg's reach: how far from the point the aircraft is still over
        # the area. A pass that came inside it and left with the area part
        # held is short.
        reach = max(math.dist(xy, p) for p in area)
        shorts = 0
        self.closest_m = float("inf")
        self.waiting = {
            "n": step + 1,
            "flight": step,
            "target": token,
            "area": area_ll,
            "coverageFloor": COVERED_FRACTION,
            "coverage": None,
            "posted_at": self._vehicle.now(),
        }
        while True:
            outcome, last_cov = self.wait_arrival(
                xy,
                area=area_ll,
                timeout_s=ARRIVAL_TIMEOUT_S,
                after=after,
                reach_m=reach if shorts < SHORT_PASS_REQUESTS else None,
            )
            if outcome != "short" or self._cancel.is_set():
                break
            shorts += 1
            moved, note = self._short_pass(token, xy, area, last_cov, shorts)
            self._post(
                Leg(
                    target=token,
                    point=(moved[0], moved[1]),
                    step=step,
                    area=area_ll,
                    ahead=ahead,
                    revision=shorts,
                    note=note,
                )
            )
        waited = self._wait_departure(xy) if outcome == "arrived" else 0.0
        self.waiting = None
        if self._cancel.is_set():
            return None
        held = self._held_frame(area_ll, xy, after)
        self.transcript.append(
            {
                "kind": "leg",
                "said": self.outcome_words(outcome, at, held, waited),
                "area": area_ll,
                "coverageFloor": COVERED_FRACTION,
                "coverage": _held_points(last_cov),
            }
        )
        return held

    def _post(self, leg: Leg) -> None:
        """Hand ``leg`` to the aircraft; a ``waypoint`` row says when and where, at the
        commanded height (what the measures' turn radius and height fallback read)."""
        self._vehicle.post(leg)
        row = {
            "kind": "waypoint",
            "altitudeAglMeters": self.agl_m,
            "flight": leg.step,
            "n": leg.step + 1,
            "label": leg.target,
            "coverageFloor": COVERED_FRACTION,
        }
        if (lonlat := self._to_lonlat([leg.point])[0]) is not None:
            row["lon"], row["lat"] = lonlat
        self.transcript.append(row)

    def outcome_words(
        self, outcome: str, at: str, held: tuple[Any, dict[str, Any]] | None, waited: float
    ) -> str:
        """The ``leg`` row of a leg that arrived or was lost: what the pass
        held, and for an arrival whether the map registered it."""
        if outcome == "arrived":
            if held is None:
                return (
                    f"the leg to {at} arrived, but no frame of the pass could be served "
                    f"({self.live.last_refusal or 'no reason recorded'}); the flight has no observation"
                )
            reg = held[1].get("registration") or {}
            by = ", ".join(f"{n} {k}" for k, n in (reg.get("sources") or {}).items())
            posed = (
                f"the map registered it{f' ({by})' if by else ''}"
                if reg.get("applied")
                else f"the map did not register it: {reg.get('why') or 'the corrector is not running'}"
            )
            return f"the leg to {at} arrived; {waited:.1f} s later {self._held_words(held[1])}; {posed}"
        lost = f"the leg to {at} never arrived in {ARRIVAL_TIMEOUT_S:.0f} s ({self._closest_words()}); "
        if held is None:
            return lost + "no frame held any of it, and the flight has no observation"
        return lost + f"{self._held_words(held[1])}, and those frames are the observation"

    def _closest_words(self) -> str:
        """The nearest the pose came to the leg's point since the leg was posted, or that no pose was read."""
        return (
            f"closest {self.closest_m:.0f} m" if math.isfinite(self.closest_m) else "no pose read"
        )

    def _said(self, text: str) -> None:
        """A ``leg`` row of the transcript: what the crew's side of a flight did."""
        self.transcript.append({"kind": "leg", "said": text})

    def _wait_departure(self, xy) -> float:
        """Hold until the point leaves the frame, so the whole pass is in the
        ring, at most the frame's transit and :data:`FRAME_WAIT_S`; the
        seconds it took."""
        bound = self.live.frame_transit_s()
        if bound is None:
            return 0.0
        bound = min(float(bound), FRAME_WAIT_S)
        t0 = self._vehicle.now()
        while not self._cancel.is_set() and self._vehicle.now() - t0 < bound:
            if not self.live.in_frame(float(xy[0]), float(xy[1])):
                break
            self._vehicle.sleep(0.25)
        return self._vehicle.now() - t0

    def coverage_before_leg(self, area_lonlat) -> tuple[dict[str, Any] | None, str | None]:
        """How much of the area the recording holds, asked up to
        ``COVERAGE_TRIES`` times, since only a read that answers decides
        whether a leg is flown: ``(coverage, refusal)``."""
        refusal = None
        for k in range(1, COVERAGE_TRIES + 1):
            cov = self.live.coverage(area_lonlat)
            if cov is not None:
                return cov, None
            refusal = self.live.last_refusal or "no reason recorded"
            log.warning(
                "read %d of %d: the recording could not be asked whether it holds the area: %s",
                k,
                COVERAGE_TRIES,
                refusal,
            )
            if k < COVERAGE_TRIES:
                self._vehicle.sleep(COVERAGE_RETRY_S)
                if self._cancel.is_set():
                    break
        return None, refusal

    def held_now(self, h: Hydrated, stops: list) -> list[bool]:
        """Which of a pass's stops the recording holds already, off one index read (``MissionSpec.held_now``)."""
        asked = {
            i: self._to_lonlat(leg_area(h, str(token), [float(xy[0]), float(xy[1])]))
            for i, (token, xy) in enumerate(stops)
            if xy is not None
        }
        covs = self.live.coverage_of_each(list(asked.values())) if asked else None
        held = dict(zip(asked, covs or []))
        return [_holds(held.get(i)) for i in range(len(stops))]

    def _short_pass(
        self, token: str, xy, area: list[tuple[float, float]], cov: dict[str, Any], n: int
    ) -> tuple[list[float], str]:
        """Where a short pass is asked again, and the note for why: the area's
        centroid moved toward the unheld points' centroid (off ``cov``, the
        recording's answer that read the pass short) by the missing ground's
        extent."""
        best = cov.get("best") or []
        unheld = [p for p, b in zip(area, best) if b is None] or list(area)
        cx, cy = (sum(p[0] for p in area) / len(area), sum(p[1] for p in area) / len(area))
        ux, uy = (sum(p[0] for p in unheld) / len(unheld), sum(p[1] for p in unheld) / len(unheld))
        dx, dy = ux - cx, uy - cy
        span = math.hypot(dx, dy)
        if span < 1.0:
            shift, moved = 0.0, [cx, cy]
        else:
            # Both lie inside the area, so the shift never passes its reach from the centroid.
            extent = max(((p[0] - cx) * dx + (p[1] - cy) * dy) / span for p in unheld)
            shift = max(extent, span)
            moved = [cx + dx / span * shift, cy + dy / span * shift]
        side = (
            ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][
                round(math.degrees(math.atan2(dx, dy)) / 45.0) % 8
            ]
            if span >= 1.0
            else "-"
        )
        frames = len({b[1] for b in best if b is not None})
        held = sum(1 for b in best if b is not None)
        said = (
            f"the pass over {token} at ({xy[0]:.0f}, {xy[1]:.0f}) held {held}/{len(area)} points of the area in "
            f"{frames} frame(s), the {side} side missing; "
            f"the leg is asked again {shift:.0f} m {side} of the point (re-fly {n} of {SHORT_PASS_REQUESTS})"
        )
        log.info(said)
        self._said(said)
        return (
            moved,
            f"Re-fly {token}: the pass held {held}/{len(area)} points, the {side} side missing",
        )

    @staticmethod
    def _held_words(last: dict[str, Any]) -> str:
        """The leg row's account of what held the area: the points held, the
        frames the mosaic asked for and how many were not fetched (no count
        where the observation records no mosaic), the nearest to nadir."""
        cov, mosaic = last["coverage"], last.get("mosaic")
        frames = ""
        if mosaic:
            frames = f" in {mosaic['asked']} frame(s)"
            if mosaic.get("fetch_failed"):
                frames += f", {mosaic['fetch_failed']} of them not fetched"
        return (
            f"the pass held {cov['held']}/{cov['points']} points of it{frames}, "
            f"the nearest {last.get('seq')} at {last.get('off_nadir_deg')} deg off nadir"
        )

    def _held_frame(
        self, area_ll: list[tuple[float, float]], xy, after: float | None = None
    ) -> tuple[Any, dict[str, Any]] | None:
        """``(observation, record)`` of the ground the pass held over ``area_ll``
        (lon/lat; the frames after ``after`` when given); None when no frame held any."""
        obs = self.live.observe_area(area_ll, after)
        last, refusal = dict(self.live.last or {}), self.live.last_refusal
        if obs is None:
            log.info(
                "no frame in the recording held the area about (%.0f, %.0f): %s",
                float(xy[0]),
                float(xy[1]),
                refusal,
            )
            return None
        last["observation"] = self._keep_observation(obs)
        return obs, last

    def wait_arrival(
        self,
        xy: Any,
        *,
        area: list[tuple[float, float]],
        timeout_s: float,
        reach_m: float | None = None,
        after: float | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        """``(outcome, the recording's last answer)``: ``arrived`` when the
        recording (the frames after ``after`` when given) holds ``area``;
        ``short`` when the pass came inside ``reach_m`` and left with the area
        part held; ``lost`` when ``timeout_s`` ran out airborne; ``cancel``
        when the mission was stopped."""
        best = math.inf
        receding_since: float | None = None
        # The clock runs on the FLIGHT: it does not spend its timeout while the
        # aircraft is still on the ground.
        t0 = self._vehicle.now()
        cov = None
        while not self._cancel.is_set():
            # The recording decides arrival; the aircraft's own pose only the
            # short pass and the timeout's start.
            cov = self.live.coverage(area, after)
            if self.waiting is not None:
                self.waiting = {
                    **self.waiting,
                    "coverage": _held_points(cov),
                    "closest_m": self.closest_m,
                }
            if _holds(cov):
                return "arrived", cov
            fraction = float((cov or {}).get("fraction") or 0.0)
            pose = self.live.ownship()
            if pose is not None:
                d = math.hypot(pose["x"] - float(xy[0]), pose["y"] - float(xy[1]))
                best = min(best, d)
                self.closest_m = min(self.closest_m, d)
                # Over the area and gone, with part of it held: a short pass.
                if (
                    reach_m is not None
                    and fraction > 0.0
                    and best <= reach_m
                    and d > best + SHORT_PASS_LEAVE_M
                ):
                    if receding_since is None:
                        receding_since = self._vehicle.now()
                    if self._vehicle.now() - receding_since >= SHORT_PASS_LEAVE_S:
                        log.info(
                            "short pass at (%.0f, %.0f): closest %.0f m, now %.0f m off, %.0f%% held",
                            float(xy[0]),
                            float(xy[1]),
                            best,
                            d,
                            100.0 * fraction,
                        )
                        return "short", cov
                else:
                    receding_since = None
                if pose["agl_m"] < AIRBORNE_AGL_M:
                    t0 = self._vehicle.now()
            if self._vehicle.now() - t0 > timeout_s:
                log.warning(
                    "no arrival at (%.0f, %.0f) in %.0f s, %s: the leg is lost",
                    float(xy[0]),
                    float(xy[1]),
                    timeout_s,
                    self._closest_words(),
                )
                return "lost", cov
            self._vehicle.sleep(1.0)
        return "cancel", cov
