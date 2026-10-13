"""Mission engine: one episode as ONE reasoner conversation.

The operator instruction is the single user message; every move FLIES the
moment it lands (:meth:`MissionSession.fly` renders, segments, and folds that
stop's perception into the live world-state); a plain no-tool-call reply ends
the bench mission. The commit is the LAST flight's target. Zero flights is an
honest zero the episode runner scores.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from benchmarks.citynav.episodes import atomic_write_json
from benchmarks.citynav.paths import CITYNAV_NATIVE_RGBD_DIR
from benchmarks.citynav.run_benchmark.osm_prior import _build_frame_enrich
from maps.llm.endpoint_spec import spec as endpoint_spec
from maps.perception.geometry.sensor_spec import served_description
from maps.perception.observation import crop_observation, ground_sides_m
from maps.perception.ortho.detection_stamping import project_and_stamp_ortho_detections
from maps.perception.ortho.ground_level import GROUND_LEVEL
from benchmarks.citynav.conversation import CONVERSATION_JSONL, ConversationLog
from maps.search.parse_compile.reasoner_loop import MissionSpec, run_mission
from maps.search.withhold import withheld
from maps.util.xy_token import parse_xy_token

log = logging.getLogger("run_benchmark")

#: The two non-feature tokens a flight target can carry: the drone's current
#: position and the marked region.
VIEW_POSITION_CURRENT = "current_position"
VIEW_POSITION_MARKER = "marked_region"


def _write_tour_cursor(flights: list[dict]) -> None:
    """Persist the flown stops (id and xy, in flight order) to
    ``<workdir>/flight/tour_cursor.json``, the record the flight readers
    (``benchmarks.flight.measures`` and ``replay``) take a run's flights
    from. No-op without ``SEARCH_COMPILE_WORKDIR``."""
    workdir_env = os.environ.get("SEARCH_COMPILE_WORKDIR")
    if not workdir_env:
        return
    try:
        atomic_write_json(Path(workdir_env) / "flight" / "tour_cursor.json", {
            "stops_visited": [{"stop_id": f["target"], "xy": f["resolved_xy"]} for f in flights]})
    except Exception:
        log.debug("tour_cursor write failed; non-fatal", exc_info=True)


# The wall-clock backstop: an absolute epoch the bench driver exports. Unset
# (the flight mission, a bare smoke) means no clock.
MISSION_DEADLINE_ENV = "CITYNAV_EPISODE_DEADLINE_EPOCH"


def _seed_base() -> int:
    """``SEARCH_SEED_BASE`` if set, else a wall-clock nanosecond stamp, as a
    32-bit seed."""
    env_seed_base = os.environ.get("SEARCH_SEED_BASE")
    if env_seed_base is not None:
        return int(env_seed_base) & 0xFFFFFFFF
    return time.time_ns() & 0xFFFFFFFF


def _mission_deadline_epoch() -> float | None:
    """The absolute epoch a pass must stop flying at, or ``None``."""
    raw = os.environ.get(MISSION_DEADLINE_ENV, "").strip()
    # The bench driver writes repr(float).
    return float(raw) if raw else None


def resolve_flight_target(
    token: str,
    *,
    node_xy,
    pose_xy: tuple[float, float],
    marker_xy: tuple[float, float] | None,
) -> dict:
    """Resolve one viewpoint target token to a world point + provenance.

    ``token`` is the item's ``target_fid``: a literal fid (``node_xy(fid) ->
    ((x, y), "detected"|"prior") | None`` over the LIVE graph), a
    view-position sentinel, or the canonical ``"[x, y]"`` world-point token.
    Returns ``{"xy": (x, y), "provenance": ...}`` or ``{"error": ...}`` (the
    fid is on neither graph). A named thing flies its middle; there is no aim.
    """
    xy = parse_xy_token(token)
    if xy is not None:
        return {"xy": (float(xy[0]), float(xy[1])), "provenance": "point"}
    if token == VIEW_POSITION_CURRENT:
        return {
            "xy": (float(pose_xy[0]), float(pose_xy[1])),
            "provenance": "point",
        }
    if token == VIEW_POSITION_MARKER:
        if marker_xy is None:
            return {"error": "no marked region on this episode"}
        return {
            "xy": (float(marker_xy[0]), float(marker_xy[1])),
            "provenance": "point",
        }
    resolved = node_xy(token)
    if resolved is None:
        return {"error": f"unknown fid {token!r}; it is not on the surface"}
    nxy, provenance = resolved
    return {"xy": (float(nxy[0]), float(nxy[1])),
            "provenance": str(provenance)}


# The flight report's head of fresh detections (geom_program's 12-row head).
_DETECTED_HEAD = 12


def detected_digest(dets: list[dict]) -> dict:
    """Report fields for THIS flight's folded detections: up to ``_DETECTED_HEAD``
    rows, round-robin over classes ordered by their strongest detection, each
    class's rows by confidence, so every class is represented;
    ``detected_more`` counts the rest.
    """
    if not dets:
        return {}
    by_class: dict = {}
    for _i, d in enumerate(dets):
        by_class.setdefault(str(d.get("class") or ""), []).append((_i, d))
    for rows in by_class.values():
        # conf descending; fold order breaks a tie.
        rows.sort(key=lambda t: (-float(t[1].get("conf") or 0.0), t[0]))
    lanes = sorted(by_class.values(),
                   key=lambda rows: (-float(rows[0][1].get("conf") or 0.0),
                                     rows[0][0]))
    head: list = []
    for _rank in range(max((len(r) for r in lanes), default=0)):
        for rows in lanes:
            if _rank < len(rows) and len(head) < _DETECTED_HEAD:
                head.append(rows[_rank][1])
        if len(head) >= _DETECTED_HEAD:
            break
    out: dict = {"detected": [
        {"fid": d["fid"], "class": d["class"], "xy": list(d["xy"])}
        for d in head]}
    if len(dets) > len(head):
        out["detected_more"] = len(dets) - len(head)
    return out


def echo_pose(ctx, report: dict) -> None:
    """After a successful flight the ToolContext's pose follows the drone.
    The context already holds the live ``state.runtime_graph``."""
    if "error" not in report:
        ctx.pose_xy = (float(report["pose_xy"][0]), float(report["pose_xy"][1]))


def perceive(h, obs, detections) -> list[dict]:
    """Project and stamp one observation's detections through its view square
    and fold them into the live world-state; returns the folded rows. The
    one ingest path a flight and the bench's retrospective boost share."""
    projected = dict(project_and_stamp_ortho_detections(
        detections, view_corners_xy=obs.ortho_view_corners_xy, rgb=obs.rgb, tile=h.tile_frame,
    )) if obs is not None else {}
    return h.search_loop.update_perception(detections=detections, projected_masks=projected)


class MissionSession:
    """The engine's half of one mission conversation over a hydrated
    episode (``episode_runner.Hydrated``): the ToolContext, the MissionSpec
    its verbs call back into, and the flights. The flight mission re-tasks a
    mission by driving another turn over the same session.

    ``observe`` (the flight mission's) returns the camera's current posed
    frame or ``None``; a flight then perceives that frame from its own pose
    and still serves the target as the stop. The bench passes none and
    renders the raster crop at the target.
    """

    def __init__(self, h, *, recorder=None, observe=None) -> None:
        self.h = h
        self.recorder = recorder
        self.observe = observe
        self.flights: list[dict] = []
        markers = h.episode.marker_positions or []
        self.marker_xy = (
            (float(sum(p[0] for p in markers) / len(markers)),
             float(sum(p[1] for p in markers) / len(markers)))
            if markers else None
        )
        loop = h.search_loop
        with loop._lock:
            self.ctx = loop._tool_ctx_for_examiner()
        state = h.state
        self.ctx.pose_xy = (float(state.pose_xy[0]), float(state.pose_xy[1]))
        ground = float(loop._ortho_ground_z_m)
        alt = float(state.altitude_m)
        sides = ground_sides_m(alt, ground, h.sensor)
        # The footprint's LONG side is the crop window's ceiling; its SHORT
        # side is the row pitch an explore's serpentine can rely on whichever
        # way the footprint is turned, and the look coverage gate (a crop on
        # ground no flight covered refuses).
        obs_span_m = max(sides)
        self.obs_minor_m = min(sides)
        self.mission_spec = MissionSpec(
            fly=self.fly,
            declare_concepts=loop.declare_mission_concepts,
            coverage_stops=self._coverage_stops,
            deadline_epoch=_mission_deadline_epoch(),
        )
        #: What every turn of the conversation takes, the first and each
        #: operator turn after it alike.
        self.run_kwargs = dict(
            ctx=self.ctx,
            mission_spec=self.mission_spec,
            look_renderer=loop._render_candidate_crops,
            obs_span_m=obs_span_m,
            obs_minor_m=self.obs_minor_m,
            boost_queue=loop._boost_queue_cb,
        )

    def run(self, instruction: str, **extra):
        """The conversation's first turn: ``run_mission`` over this session,
        the installed endpoint and the rig's served description. Returns
        ``(reply, reasoner)``."""
        h = self.h
        spec = endpoint_spec()
        return run_mission(
            instruction=instruction,
            reasoner_url=spec.base_url,
            endpoint_spec=spec,
            # "EMPTY" is the no-auth sentinel a local vLLM accepts.
            api_key=os.environ.get("SEARCH_REASONER_API_KEY", "EMPTY"),
            seed=_seed_base() & 0x7FFFFFFF,
            sensor_description=(served_description(
                h.sensor, agl_m=float(h.state.altitude_m) - float(h.search_loop._ortho_ground_z_m))
                if h.sensor is not None else ""),
            **self.run_kwargs,
            **extra,
        )

    def _coverage_stops(self, fid: str, spacing_m: float):
        # The flight budget bounds a pass: a stop IS a flight.
        return self.h.search_loop.node_coverage_stops(
            fid, spacing_m, max_stops=self.mission_spec.flight_budget, minor_spacing_m=self.obs_minor_m)

    def resolve(self, token: str) -> dict:
        """``token`` as :func:`resolve_flight_target` resolves it over the live graph."""
        return resolve_flight_target(token, node_xy=self.h.search_loop.node_xy, pose_xy=self.h.state.pose_xy,
                                     marker_xy=self.marker_xy)

    def fly(self, item: dict) -> dict:
        """One flight (``MissionSpec.fly``), then the pose echo follows it."""
        report = self._fly(item)
        echo_pose(self.ctx, report)
        return report

    def _fly(self, item: dict) -> dict:
        """Resolve -> teleport -> render -> segment -> fold -> enrich for ONE
        move, the moment it lands. Returns the model-facing flight report, or
        ``{"error": ...}`` when the target does not resolve. Appends the flight
        to ``flights``: the commit reads the last, and the discrete replay
        flies them all."""
        h = self.h
        state, episode, loop = h.state, h.episode, h.search_loop
        token = str(item.get("target_fid") or "")
        resolved = self.resolve(token)
        if "error" in resolved:
            return {"error": resolved["error"]}
        target_xy = resolved["xy"]
        new_xy = target_xy

        # Hold the cruise altitude, clamped ground+1 .. cap.
        _g = float(GROUND_LEVEL.get(episode.map_name, 0.0))
        new_alt = float(min(max(float(state.altitude_m), _g + 1.0), float(h.cfg.altitude_cap_m)))
        new_heading = float(state.heading_deg)

        # Queued boosts segment on THIS flight.
        try:
            loop.start_flight()
        except Exception:  # noqa: BLE001 — perception direction never kills a flight
            log.exception("boost start failed; flight proceeds un-boosted")

        step_idx = len(self.flights)
        stop_id = f"stop_{step_idx:03d}"
        state.step_idx = step_idx
        if self.observe is not None:
            obs = self.observe()
            if obs is None:
                return {"error": "no observation: the aircraft's recording served no frame of this flight's ground"}
            new_xy = (float(obs.pose_xy[0]), float(obs.pose_xy[1]))
            new_alt = float(obs.altitude_m)
            new_heading = float(obs.yaw_deg)
        state.pose_xy = new_xy
        state.altitude_m = new_alt
        state.heading_deg = new_heading

        # Rendered north-up (yaw 0), as the captions and the retrospective
        # re-render are; the pose keeps its heading for the replay.
        if self.observe is None:
            obs = crop_observation(
                new_xy,
                altitude_m=new_alt,
                yaw_deg=0.0,
                sensor=h.sensor,
                rgbd_dir=CITYNAV_NATIVE_RGBD_DIR,
                map_name=episode.map_name,
            )
        if obs is not None:
            obs.tile = h.tile_frame

        step_detections = perceive(h, obs, h.perception.detect(obs))

        if self.recorder is not None and obs is not None:
            try:
                self.recorder.write_step(step=step_idx, detections=step_detections)
            except Exception as exc:  # noqa: BLE001
                log.warning("mission step write failed step=%d: %s", step_idx, exc)

        # Prior colour enrichment: one frame, upserted onto the named priors.
        if obs is not None and obs.rgb is not None:
            _built = _build_frame_enrich(
                obs=obs,
                step_idx=step_idx,
                view_corners=(tuple(obs.ortho_view_corners_xy)
                              if obs.ortho_view_corners_xy is not None else None),
                assets=h.assets,
                build_prior=not withheld("prior"),
            )
            for pfid, pattrs, ppx in _built["prior"]:
                loop.upsert_prior_percept(pfid, pattrs, px=ppx)
            if _built["prior_exc"] is not None:
                loop.state.prior_enrich_fails += 1
                _nfail = loop.state.prior_enrich_fails
                if _nfail == 1 or _nfail % 50 == 0:
                    log.warning(
                        "prior percept enrichment failed (n=%d) step=%d: %s",
                        _nfail,
                        step_idx,
                        _built["prior_exc"],
                        exc_info=_built["prior_exc"],
                    )

        if self.observe is not None:
            # The frame's own pose fed the fold and the recorder; from here
            # the stop is the target asked for, as on the bench.
            new_xy = (float(target_xy[0]), float(target_xy[1]))
            state.pose_xy = new_xy
        # The flight is perceived: a boost can now address this stop.
        try:
            loop.record_flown_stop({"pose_xy": new_xy, "alt_m": new_alt, "stop_id": stop_id,
                                    "target_fids": [token] if token else []})
        except Exception:  # noqa: BLE001 — perception direction never kills a flight
            log.exception("flown-stop record failed (non-fatal)")

        self.flights.append(
            {
                "target": token,
                "resolved_xy": [float(new_xy[0]), float(new_xy[1])],
                "provenance": str(resolved.get("provenance", "unresolved")),
                "altitude_m": new_alt,
            }
        )
        _write_tour_cursor(self.flights)

        report: dict = {
            "stop_id": stop_id,
            "target": token,
            "pose_xy": [round(float(new_xy[0]), 1), round(float(new_xy[1]), 1)],
            "alt_m": round(new_alt, 1),
            "new_detections": len(step_detections),
            **detected_digest(loop.last_flight_detections()),
            # The commit is the last flight's target, verbatim.
            "answer_if_you_reply_now": token,
            # The operator instruction rides every flight report.
            "cue": episode.description,
        }
        if not withheld("look_for"):
            report["watchlist"] = list(h.perception.concept_provider.concepts)
        return report


def run_mission_episode(session: MissionSession):
    """Drive the bench's one conversation and bind the last flight's target
    as the commit (``state.confirmed_target_xy``) the trajectory replay
    reads. Returns the reply; zero flights leaves it unbound, the honest
    zero."""
    h = session.h
    workdir = os.environ.get("SEARCH_COMPILE_WORKDIR", "").strip() or None
    transcript = ConversationLog(Path(workdir) / CONVERSATION_JSONL if workdir else None)
    reply, _ = session.run(h.episode.description, transcript=transcript)
    flights = session.flights
    if flights:
        last = flights[-1]
        h.state.confirmed_target_xy = (float(last["resolved_xy"][0]), float(last["resolved_xy"][1]))
        log.info(
            "mission commit: fid=%s xy=%s flights=%d ended_by=%s",
            last["target"],
            last["resolved_xy"],
            len(flights),
            reply.ended_by,
        )
    else:
        log.warning(
            "mission ended with ZERO flights (ended_by=%s); honest zero",
            reply.ended_by,
        )
    return reply
