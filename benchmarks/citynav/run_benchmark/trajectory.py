"""Deterministic replay of the flown stops + committed descent, in whatever
action space the config declares (honest truncation; final pose is what
scores).
"""

from __future__ import annotations

from benchmarks.citynav.adapter import CityNavEpisode
from benchmarks.citynav.episode import EpisodeState
from maps.search.parse_compile.tour_state import FeatureTour
from benchmarks.citynav.flight_model import resolve_flight_model
from benchmarks.citynav.run_benchmark.config import RunConfig
from pathlib import Path
import json
import math
import os
import logging

log = logging.getLogger("run_benchmark")


def synthesize_compliant_trajectory(
    *,
    cfg: RunConfig,
    state: "EpisodeState",
    episode: "CityNavEpisode",
    feature_tour: "FeatureTour",
    start_pose_xyh: tuple[float, float, float],
    start_alt: float,
) -> int:
    """Deterministic pure-kinematics replay of the mission's flown stops.

    The mission produces the flown viewpoint order (``feature_tour``) and
    the commit (``state.belief_snapshot.confirmed_target_xy`` /
    ``locked_detection_id``); perception already happened, at those stops,
    in the order they flew. Replaying that order + the committed descent
    through the declared action-space model (no perception, no LLM)
    yields the genuine submittable
    ``state.visited_xy`` / final drone pose the serial-compliant path
    would have produced.

    Flies the config-declared action-space model (``flight_model.kind``,
    resolved by ``resolve_flight_model``) and honors the ``cfg.max_steps``
    budget identically; no special-casing to force success on overrun.

    Final-leg altitude policy. The leg from the last tour stop to the
    committed centroid (``descent_xy``; name retained for historical
    reasons) flies AT cruise altitude by default -- ``cfg.descend_to_land``
    defaults to ``False`` because the final-leg target is 2D (xy only),
    so dropping altitude never changes the scored xy position
    and only burns budget for zero scoring benefit (22-60 of 100 actions
    under ``discrete6``). The drone's final ``z``
    is whatever the cruise leg left it at. Set ``descend_to_land: true``
    only for a benchmark variant that needs the drone to physically
    descend; do NOT enable it for paper-comparable runs.

    Returns the tick count.
    """
    tour_wps: list[tuple[float, float]] = []
    tour_alts: list[float] = []
    tour_yaws: list[float | None] = []
    tour_fids: list[str] = []
    for vp in feature_tour.viewpoints:
        tour_wps.append((float(vp.xy[0]), float(vp.xy[1])))
        tour_alts.append(
            float(vp.altitude_m) if vp.altitude_m is not None else float(cfg.altitude_m)
        )
        tour_yaws.append(float(vp.heading_deg) if vp.heading_deg is not None else None)
        tour_fids.append(str(vp.fid))

    snap = state.belief_snapshot
    descent_xy = None
    if snap is not None:
        descent_xy = getattr(snap, "confirmed_target_xy", None)
    if descent_xy is not None:
        descent_xy = (float(descent_xy[0]), float(descent_xy[1]))

    # The action-space model the config declares. ``max_steps`` counts its
    # ticks: discrete actions under ``discrete6``, camera frames under a
    # continuous model.
    synth_sim = resolve_flight_model(
        cfg.flight_model,
        start_pose=start_pose_xyh,
        start_altitude_m=float(start_alt),
    )
    max_frames = int(cfg.max_steps)

    state.visited_xy = []
    steps = 0

    def _drain(sim, limit: int) -> None:
        nonlocal steps
        while steps < limit and sim.has_pending_target():
            fp = next(sim)
            if fp.frame_idx >= limit:
                break
            state.pose_xy = (float(fp.x), float(fp.y))
            state.heading_deg = float(fp.heading_deg)
            if fp.altitude_m is not None:
                state.altitude_m = float(fp.altitude_m)
            state.record_visit()
            steps = fp.frame_idx + 1

    # Re-aim from the current pose between legs so each leg consumes only as
    # many ticks as it needs, bounded by the per-leg ``budget`` (the TOUR legs
    # pass ``tour_budget`` so a single over-long leg can never eat the descent
    # reserve mid-leg — the round-2 #6450 truncation path; the descent leg
    # passes ``max_frames``).
    #
    # Arrival is judged against the FLYING model's own tolerance: a discrete
    # walk stops within half a 5 m step of what it aimed at, a continuous one
    # within a tick of travel. A literal here would apply the quadcopter's
    # quantization to every airframe.
    snap_xy_tol_m = float(synth_sim.arrival_tol_m)
    # How far one tick carries the drone, for sizing the commit reserve below
    # in the model's own ticks rather than in 5 m discrete steps.
    _advance_m = float(synth_sim.advance_per_tick_m)

    def _drive_to(
        target_xy: tuple[float, float],
        *,
        altitude_m: float | None,
        budget: int | None = None,
    ) -> bool:
        """Fly one leg. Returns True when the leg was TRUNCATED by the
        action budget short of its target (False on arrival or on an
        honest no-progress stop below the budget)."""
        nonlocal steps
        limit = max_frames if budget is None else min(int(budget), max_frames)
        while steps < limit:
            if altitude_m is None:
                synth_sim.set_next_target(target_xy)
            else:
                synth_sim.set_next_target(target_xy, altitude_m=altitude_m)
            if not synth_sim.has_pending_target():
                break  # snap returned no actions; already at target
            before = steps
            _drain(synth_sim, limit)
            if steps == before:
                break  # snap couldn't make progress; honest stop
            dx = float(state.pose_xy[0]) - float(target_xy[0])
            dy = float(state.pose_xy[1]) - float(target_xy[1])
            if math.hypot(dx, dy) <= snap_xy_tol_m:
                break
        dx = float(state.pose_xy[0]) - float(target_xy[0])
        dy = float(state.pose_xy[1]) - float(target_xy[1])
        return steps >= limit and math.hypot(dx, dy) > snap_xy_tol_m

    # Budget reserve: replaying the flown stops must NOT consume the
    # discrete-action budget needed for the final leg to the committed target —
    # that leg sets ``pred_xy`` (the scored position). Reserve enough actions for
    # it so a long flight truncates its own TAIL (late stops, already perceived
    # in flight) rather than starving the commit fly-to (which would silently
    # leave pred_xy at an intermediate pose, faking an SR regression). Sized to the planned descent distance in the
    # FLYING model's own ticks; capped at half the budget.
    #
    # COMMIT-ARRIVAL INTEGRITY (reasoning audit r7b, Action 1.iii): the static
    # reserve is sized from the PLANNED last-waypoint -> commit distance, but a
    # truncated tour leaves the drone somewhere ELSE — the actual descent leg can
    # be far longer than the estimate (496 genai10: committed 12.2m from GT, drone
    # ended 32.5m; the reserve priced the wrong start point). The whole replay is
    # pure deterministic kinematics, so on a truncated descent we RE-SYNTHESIZE
    # with an escalated reserve (dropping more of the already-perceived tour
    # tail — the same tail-drop principle the reserve was built on) until the
    # descent completes or the FULL budget is reserved. A commit reachable within
    # the budget therefore always completes its transit; only a genuinely
    # unreachable commit (start -> commit exceeds the whole budget) remains
    # short, and that is logged loudly (react_stats.descent_unreachable).
    base_reserve = 0
    if descent_xy is not None and tour_wps:
        _last = tour_wps[-1]
        _dd = math.hypot(descent_xy[0] - _last[0], descent_xy[1] - _last[1])
        base_reserve = min(int(_dd / _advance_m) + 6, max_frames // 2)
    react_descent_starved = False
    descent_truncated = False
    descent_replans = 0
    descent_unreachable = False

    def _walk_tour_discrete(tour_budget: int, descent_reserve: int) -> bool:
        """Drive to each tour stop in turn with the re-snap loop above.
        Aims at one stop at a time rather than pre-planning the whole
        stop list, so a leg's cost is re-derived from where the drone
        actually is. Stops at ``tour_budget`` (= max_frames - descent_reserve) so
        the descent below always runs — enforced BOTH between legs and
        MID-LEG (each tour leg flies with ``budget=tour_budget``): before
        the mid-leg cap, a single over-long leg ran to ``max_frames``,
        silently ate the descent reserve, and the commit was replaced by
        the mid-walk pose (round-2 #6450 w2, the 181m symptom). Returns
        True when the reserve truncated the stop tail."""
        starved = False
        for i, wp in enumerate(tour_wps):
            if steps >= tour_budget:
                starved = True
                log.warning(
                    "replay budget: the flown-stop walk hit the commit reserve at "
                    "step %d/%d (reserved %d) after %d/%d stops; remaining "
                    "tail stops dropped from the replay so the commit leg "
                    "is reached",
                    steps, max_frames, descent_reserve, i, len(tour_wps),
                )
                break
            leg_truncated = _drive_to(
                (float(wp[0]), float(wp[1])),
                altitude_m=(float(tour_alts[i]) if i < len(tour_alts) else None),
                budget=tour_budget,
            )
            if leg_truncated:
                starved = True
                log.warning(
                    "replay budget: leg %d/%d truncated MID-LEG at the commit "
                    "reserve (step %d/%d, reserved %d); remaining stops "
                    "dropped from the replay so the commit leg is reached",
                    i, len(tour_wps), steps, max_frames, descent_reserve,
                )
                break
        return starved

    def _fly_descent() -> bool:
        """The final transit to the committed point. Returns True when the
        action budget truncated it short of the commit."""
        if steps >= max_frames:
            # Defensive: with the mid-leg tour cap above, a non-empty tour
            # leaves at least the descent reserve; this fires only on a
            # zero-reserve edge (e.g. empty tour with steps somehow spent).
            return True
        # SR is 2D: traverse to the committed centroid, but only drop to
        # landing_altitude when descend_to_land is set. Otherwise stay at
        # cruise altitude (no vertical action budget spent for zero
        # scoring benefit).
        descent_alt = (
            float(cfg.landing_altitude_m)
            if cfg.descend_to_land
            else float(state.altitude_m)
        )
        # Final-approach overshoot, ALWAYS: the approach stops anywhere within
        # the model's arrival tolerance SHORT of the target, biasing the achieved
        # position away from the commit (knife-edge losses at NE 20.1-21.1 m). Aiming
        # one tolerance PAST the commit centers the achieved position on it.
        final_target = descent_xy
        px, py = float(state.pose_xy[0]), float(state.pose_xy[1])
        dx, dy = descent_xy[0] - px, descent_xy[1] - py
        dist = math.hypot(dx, dy)
        if dist > snap_xy_tol_m + 0.5:
            final_target = (
                descent_xy[0] + snap_xy_tol_m * dx / dist,
                descent_xy[1] + snap_xy_tol_m * dy / dist,
            )
        return _drive_to(final_target, altitude_m=descent_alt, budget=max_frames)

    descent_reserve = base_reserve
    while True:
        # Fresh deterministic replay per attempt: new sim from the start
        # pose, replay state reset (the closures rebind through the
        # enclosing scope).
        synth_sim = resolve_flight_model(
            cfg.flight_model,
            start_pose=start_pose_xyh,
            start_altitude_m=float(start_alt),
        )
        state.visited_xy = []
        steps = 0
        state.pose_xy = (float(start_pose_xyh[0]), float(start_pose_xyh[1]))
        state.heading_deg = float(start_pose_xyh[2])
        state.altitude_m = float(start_alt)
        react_descent_starved = False
        descent_truncated = False
        if tour_wps:
            react_descent_starved = _walk_tour_discrete(
                max_frames - descent_reserve, descent_reserve
            )
        if descent_xy is not None:
            descent_truncated = _fly_descent()
        if not descent_truncated or descent_xy is None:
            break
        _short_m = math.hypot(
            float(state.pose_xy[0]) - descent_xy[0],
            float(state.pose_xy[1]) - descent_xy[1],
        )
        if descent_reserve >= max_frames:
            # Even a descent-only flight (whole budget reserved, whole
            # tour dropped) cannot reach the commit: genuinely
            # unreachable within the action budget. Loud, not silent.
            descent_unreachable = True
            log.error(
                "COMPLIANT-PARALLEL: commit descent UNREACHABLE within the "
                "action budget even with the FULL budget reserved (whole "
                "tour dropped): pred_xy stays %.1fm short of the committed "
                "target %s (steps=%d/%d, replans=%d) "
                "(react_stats.descent_unreachable)",
                _short_m, descent_xy, steps, int(max_frames),
                descent_replans,
            )
            break
        descent_replans += 1
        _new_reserve = min(
            max_frames,
            max(2 * descent_reserve,
                descent_reserve + int(_short_m / _advance_m) + 12,
                16),
        )
        log.warning(
            "COMPLIANT-PARALLEL: commit descent truncated %.1fm short of "
            "the committed target %s (steps=%d/%d, reserve=%d) -> "
            "re-synthesizing the replay with reserve=%d (replan %d; the "
            "stop tail is already perceived, the commit transit is not "
            "optional)",
            _short_m, descent_xy, steps, int(max_frames),
            descent_reserve, _new_reserve, descent_replans,
        )
        descent_reserve = _new_reserve
    if descent_truncated and descent_xy is not None:
        _short_m = math.hypot(
            float(state.pose_xy[0]) - descent_xy[0],
            float(state.pose_xy[1]) - descent_xy[1],
        )
        log.warning(
            "COMPLIANT-PARALLEL: commit descent TRUNCATED by the action "
            "budget %.1fm short of the committed target %s (steps=%d/%d) "
            "-> pred_xy is the truncated pose, NOT the commit "
            "(react_stats.descent_truncated)",
            _short_m, descent_xy, steps, int(max_frames),
        )

    final_pos_xy = (float(state.pose_xy[0]), float(state.pose_xy[1]))
    pred_xy = final_pos_xy
    # Replay telemetry: descent_starved = the reserve truncated the STOP TAIL
    # (the descent itself still ran, so pred_xy carries the committed target);
    # descent_truncated = the DESCENT LEG was cut short by the budget, so
    # pred_xy is the truncated pose, NOT the commit (pred_xy stays pose-tied
    # by protocol — the flag makes the shortfall loud instead of silent).
    try:
        _rs = getattr(state, "react_stats", None)
        if isinstance(_rs, dict):
            if react_descent_starved:
                _rs["descent_starved"] = True
            if descent_truncated:
                _rs["descent_truncated"] = True
            if descent_replans:
                _rs["descent_replans"] = int(descent_replans)
            if descent_unreachable:
                _rs["descent_unreachable"] = True
    except Exception:  # noqa: BLE001 — telemetry only
        pass
    log.info(
        "COMPLIANT-PARALLEL trajectory (discrete 6-action): %d tour stops %s "
        "descent=%s frames=%d pred_xy=%s final_pos_xy=%s (budget=%d) "
        "descent_starved=%s descent_truncated=%s",
        len(tour_fids),
        tour_fids,
        descent_xy,
        steps,
        pred_xy,
        final_pos_xy,
        int(max_frames),
        react_descent_starved,
        descent_truncated,
    )

    workdir_env = os.environ.get("SEARCH_COMPILE_WORKDIR")
    if workdir_env:
        try:
            run_dir = Path(workdir_env) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "compliant_trajectory.json").write_text(
                json.dumps(
                    {
                        "episode_id": episode.episode_id,
                        "tour_fids": tour_fids,
                        "tour_xy": tour_wps,
                        "tour_alts": tour_alts,
                        "descent_xy": descent_xy,
                        "n_steps": steps,
                        "pred_xy": list(pred_xy),
                        "final_pos_xy": list(final_pos_xy),
                        "max_steps": int(cfg.max_steps),
                        # round-2 audit FIX B truncation observability
                        "descent_starved": bool(react_descent_starved),
                        "descent_truncated": bool(descent_truncated),
                        # r7b Action 1.iii commit-arrival integrity
                        "descent_replans": int(descent_replans),
                        "descent_unreachable": bool(descent_unreachable),
                    },
                    indent=2,
                )
            )
        except OSError as exc:
            log.warning("compliant_trajectory.json write failed: %s", exc)

    return steps
