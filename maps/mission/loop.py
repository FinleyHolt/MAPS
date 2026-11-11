"""The mission conversation: one persistent reasoner, one served tool.

Each ``run`` call carries one block of Python; the interpreter
(``maps.library.interpreter``) runs it in the mission's one namespace on this
mission's verbs (ground, analyze, the acts, finish). ``go_to`` / ``explore``
fly the moment they land and the query surface re-derives from the world the
flight folded. The model's plain reply ends the mission and the last flight's
target is the answer (the engine owns the commit); ``MissionSpec`` holds the
liveness bounds the model never hears."""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

from maps.library.call_arguments import MissionEnv
from maps.library.interpreter import MissionPython
from maps.library.op_compiler import HANDLE_WORD_RE
from maps.library.result_handles import Carry
from maps.library.rows import UNNAMED
from maps.llm.http import CallCanceled
from maps.mission.acks import _Acks, _summarise_exc
from maps.mission.compactor import CompactionInfeasible, CompactionInvariantError
from maps.mission.fly_verbs import _FlyVerbs
from maps.mission.picture_verbs import _PictureVerbs
from maps.mission.program_verbs import _ProgramVerbs
from maps.mission.reasoner import Reasoner
from maps.mission.reply_repair import (
    MALFORMED_ENVELOPE_NUDGE, NO_FLIGHT_ABSTAIN_STREAK, NO_FLIGHT_REPLY_NUDGE,
    TRUNCATED_TURN_NUDGE, TRUNCATED_TURN_STREAK, _GO_TO_FIRST, _envelope_residue,
    _not_a_reply)
from maps.mission.serve_refusals import _REFUSAL_ENDINGS, _RETRIABLE_LLM_ERRORS
from maps.mission.served_text import served_tools
from maps.mission.spec import MissionReply, MissionSpec
from maps.mission.surface_state import _SurfaceState, _surface_from_ctx
from maps.mission.watch_verbs import _WatchVerbs
from maps.settings import Settings
from maps.withhold import active, withheld, withholding
from maps.world_state.context import QueryContext


log = logging.getLogger(__name__)


# The bench's transient-error budget; a live mission (a reasoner with a
# cancel_event) waits for the arm until it answers or the operator stops.
_MAX_TRANSIENT = 6


def run_mission(
    *,
    instruction: str,
    endpoint_spec,
    settings: Settings = Settings(),
    workdir: Path | None = None,
    routing_key: str | None = None,
    seed: int | None = None,
    sensor_description: str = "",
    cancel_event: "threading.Event | None" = None,
    reasoning_effort: str | None = None,
    rid_tag: str = "m0",
    **turn,
) -> tuple[MissionReply, "_Mission"]:
    """THE mission entry: build the Reasoner and the mission over it, and run
    the first operator turn (``turn`` is :class:`_Mission`'s). ``settings``
    is the run's (``maps.settings``), fixed for the mission; ``workdir`` is
    the mission's own directory (its wire log and shown frames; None records
    nothing) and ``routing_key`` its router-affinity key. Returns the reply
    and the mission, for :func:`continue_mission`."""
    # The mission is built under the set it withholds, which it keeps for every turn.
    with withholding(settings.withhold):
        reasoner = Reasoner(
            endpoint_spec=endpoint_spec,
            settings=settings,
            workdir=workdir,
            routing_key=routing_key,
            seed=seed,
            sensor_description=sensor_description,
            cancel_event=cancel_event,
            reasoning_effort=reasoning_effort,
        )
        mission = _Mission(reasoner=reasoner, workdir=workdir, rid_tag=rid_tag, preflight=True,
                           compute_namespace=_surface_from_ctx(turn["ctx"]), **turn)
    return mission.converse(instruction.strip()), mission


def continue_mission(mission: "_Mission", instruction: str, *,
                     rid_tag: str) -> MissionReply:
    """One more operator turn on the same mission: it follows a reply (the
    field path: the reply is a report), reads the surface the mission's
    ``ctx`` holds now, and keeps every name, handle, flown stop, staged view
    and watched class of the turns before it."""
    with withholding(mission.withhold):
        mission.rid_tag = rid_tag
        mission._world_moved()
        return mission.converse(instruction.strip())


class _Mission(_ProgramVerbs, _FlyVerbs, _PictureVerbs, _WatchVerbs,
               _SurfaceState, _Acks):
    """One mission's conversation state, across every operator turn, and the
    verbs the run surface calls on it."""

    def __init__(
        self,
        *,
        reasoner: Reasoner,
        ctx: QueryContext,
        compute_namespace: dict,
        mission_spec: MissionSpec,
        boost_queue: Callable[[list, list], dict],
        rid_tag: str = "h?",
        transcript: list[dict] | None = None,
        diag: dict | None = None,
        look_renderer: Callable[[list[dict]], list[dict]] | None = None,
        obs_span_m: float | None = None,
        obs_minor_m: float | None = None,
        preflight: bool = False,
        retask: Callable[[], str | None] | None = None,
        workdir: Path | None = None,
    ):
        self.reasoner = reasoner
        #: The mission's directory, where the frames it is shown are kept; None keeps none.
        self.workdir = workdir
        #: What the mission withholds, fixed when it is built: the set bound
        #: then (``maps.withhold``), which every turn runs under.
        self.withhold = active()
        self.ctx = ctx
        self.rid_tag = rid_tag
        self.transcript = transcript
        self.ns = compute_namespace
        self.diag = diag
        self.look_renderer = look_renderer
        self.obs_span_m = obs_span_m
        self.obs_minor_m = obs_minor_m
        self.boost_queue = boost_queue
        self.preflight = preflight
        self.mission_spec = mission_spec
        self.retask = retask

        # Each fid's latest result-enriched row, for the rejection census.
        self.row_latest: dict[str, dict] = {}
        # Successful self-contained programs by ``program_dedup_key`` ->
        # ``(surface_version, turn, ops)``: a repeat under the same version gets
        # the dedup note. A flight, a boost or a stance bumps the version.
        self._programs_seen: dict[str, tuple] = {}

        # Surface rows by fid, rebuilt in place after every flight and boost.
        self._ns_rows: dict[str, dict] = {}

        self._rebuild_ns_rows()
        self.stop_tools = list(served_tools())
        self.pending_vision: list[tuple[str, list[bytes]]] = []
        self._block_frames: list[str] = []

        # The staged viewpoints by vN handle; the latest is STANDING (its side
        # columns ride every find row, its facing orients what renders).
        self.staged_viewpoints: dict[str, dict] = {}
        self.standing_handle: str | None = None
        # The mission's flown stops, {stop_id: [x, y]}: each a sayable place.
        self._plan_stops: dict[str, list] = {}

        # The flatten is built once per surface version.
        self._map_feats_version = None
        self._map_feats: list[dict] = []

        self.carry = Carry(compute_namespace, name=lambda ref: (
            self._mission_py.ref_name(ref) or UNNAMED))
        # The mission's flights so far, over every operator turn.
        self.flights = 0
        # The watched classes, in declaration order.
        self.mission_sam: list[str] = []
        self._new_turn()

        # The set in scope (the last non-empty result, or what the last flight
        # added), the paging ledger, and the mission-global rN counter.
        self._those: str | None = None
        self._widened: dict = {}
        self.plan_rn: list = [0]
        # The lines of every block run, a running total on each result.
        self._script_lines = 0
        # Every rN in play (bound handles and those the running block spells):
        # the flight mint skips them.
        self._code_names: set = set()
        # The handles flights minted during the running block, for the code
        # scope to bind.
        self._code_minted: list = []
        # The fids explore passes have covered this mission.
        self._covered_fids: set = set()

        # A pass's stop spacing: half the footprint's long side, under the
        # short side for any sensor below 2:1, so stops overlap on both axes.
        self._cover_spacing = (obs_span_m / 2.0
                               if obs_span_m and obs_span_m > 0 else None)

        # The call in flight: what _ground_names noted and _analyze answered,
        # read by _finish.
        self._call: dict = {}

        self._mission_py = MissionPython(MissionEnv(
            rows_of_ref=lambda ref: list(self.carry.read(ref) or []),
            tied_of_ref=lambda ref: bool(
                getattr(self.carry.read(ref), "name_set", False)),
            version=lambda: self.carry.version,
            plan_rn=self.plan_rn, widened=self._widened, minted=self._code_minted,
            say_err=self._plan_err, row_of_fid=self._ns_rows.get,
            ground=self._ground_names, analyze=self._analyze, stand=self._stand,
            move=self._move_to, cover=self._cover, crops=self._crops, map=self._map,
            watch=self._watch, boost=self._boost, finish=self._finish))

    def _new_turn(self) -> None:
        """An operator turn's own state: its flight budget (the backstop's
        count; ``flights`` counts the mission's), its escalating loop
        breaker (see _loop_broken_error) and its dedup streak."""
        self.turn_flights = 0
        self._err_class_counts: dict[str, int] = {}
        self._err_sig, self._err_n = None, 0
        self._dedup_key, self._dedup_n = None, 0

    def _dispatch_run(self, args) -> dict:
        """One code block, run in the mission's persistent namespace; an
        error costs the line it is on."""
        if "__args_parse_error" in args:
            return {"error": "the call's arguments did not parse as JSON: "
                             f"{args['__args_parse_error']}; nothing ran"}
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return {"error": "run takes one block of code in code"}
        self._code_names.clear()
        self._code_names.update(HANDLE_WORD_RE.findall(code))
        self._code_names.update(self._mission_py.held_refs())
        self._code_minted.clear()
        self._script_lines += len(code.strip().splitlines())
        out = self._mission_py.run_block(code)
        out["script_lines"] = self._script_lines
        return out

    def _log(self, kind: str, **row) -> None:
        """One transcript row, when the mission keeps a transcript."""
        if self.transcript is not None:
            self.transcript.append({"kind": kind, **row})

    def _mission_end(self, ended_by: str, turns: int = 0) -> MissionReply:
        self._log("mission_end", ended_by=ended_by, turns=int(turns))
        return MissionReply(ended_by=ended_by, turns_used=int(turns))

    def converse(self, user_message: str) -> MissionReply:
        """The mission's turn loop, under the set the mission withholds:
        append ``user_message``, then serve ``run`` until the model's plain
        reply or a harness ending; an empty turn, a block of code and a reply
        with no flight behind it are repaired before a repeat is accepted as
        the ending."""
        with withholding(self.withhold):
            return self._converse(user_message)

    def _converse(self, user_message: str) -> MissionReply:
        self._new_turn()
        if self.reasoner.compaction_infeasible:
            return self._mission_end("compaction_infeasible")
        self.reasoner.append_user(user_message)
        # Channel misses, consecutive zero-flight replies
        # (NO_FLIGHT_ABSTAIN_STREAK), transient serve errors.
        no_call_streak = no_flight_streak = transient_failures = truncated_streak = 0
        for turn_idx in itertools.count():
            self.turn_idx = turn_idx
            if self.diag is not None:
                self.diag["turns_used"] = turn_idx + 1
            if turn_idx >= self.mission_spec.turn_cap:
                log.info("%s mission turn cap (%d) reached", self.rid_tag,
                         self.mission_spec.turn_cap)
                return self._mission_end("turn_cap", turns=turn_idx)
            # The operator's re-tasks land at the top of a turn, never between
            # a turn's tool calls and their results.
            if self.retask is not None:
                while (operator_turn := self.retask()) is not None:
                    log.info("%s operator re-task at turn %d (%d chars)",
                             self.rid_tag, turn_idx, len(operator_turn))
                    self.reasoner.append_user(operator_turn)
            try:
                r = self.reasoner.step_tool_call(self.stop_tools)
            except (CompactionInfeasible, CompactionInvariantError) as exc:
                # Terminal: flag the thread and drop this turn's unread message.
                self.reasoner.compaction_infeasible = True
                msgs = self.reasoner.messages
                if (turn_idx == 0 and msgs
                        and msgs[-1].get("role") == "user"
                        and msgs[-1].get("content") == user_message):
                    msgs.pop()
                log.warning(
                    "%s compaction infeasible -> mission ends "
                    "(thread flagged): %s",
                    self.rid_tag, _summarise_exc(exc))
                return self._mission_end("compaction_infeasible", turns=turn_idx + 1)
            except _RETRIABLE_LLM_ERRORS as exc:
                for refused, ended_by, why in _REFUSAL_ENDINGS:
                    if refused(exc):
                        log.warning("%s %s: %s", self.rid_tag, why,
                                    _summarise_exc(exc))
                        self._log(ended_by, error=_summarise_exc(exc))
                        return self._mission_end(ended_by, turns=turn_idx + 1)
                transient_failures += 1
                cancel = self.reasoner.cancel_event
                log.warning(
                    "%s transient LLM error %d%s: %s",
                    self.rid_tag,
                    transient_failures,
                    "" if cancel is not None else f"/{_MAX_TRANSIENT}",
                    _summarise_exc(exc),
                )
                self._log("transient_error", attempt=transient_failures,
                          error=_summarise_exc(exc))
                if cancel is None:
                    if transient_failures >= _MAX_TRANSIENT:
                        return self._mission_end("transient_failures",
                                                 turns=turn_idx + 1)
                    time.sleep(min(15.0, 0.5 + 2.0 * transient_failures))
                elif cancel.wait(min(30.0, 0.5 + 2.0 * transient_failures)):
                    raise CallCanceled("stopped while waiting for the reasoner")
                continue
            transient_failures = 0

            if r["kind"] == "text" and withheld("loop"):
                # No repair under the ``loop`` withhold: any text turn ends.
                log.info("%s mission ended by reply at turn %d (%d flights)",
                         self.rid_tag, turn_idx, self.turn_flights)
                return self._mission_end("reply", turns=turn_idx + 1)
            if r["kind"] == "text":
                content = str(r.get("content") or "")
                if _envelope_residue(content) or _envelope_residue(
                        str(r.get("reasoning") or "")):
                    log.info("%s MALFORMED_ENVELOPE at turn %d -> repair",
                             self.rid_tag, turn_idx)
                    self._log("malformed_envelope", turn_idx=turn_idx)
                    self.reasoner.append_repair(MALFORMED_ENVELOPE_NUDGE)
                    continue
                miss = _not_a_reply(content)
                cut = (miss is not None and miss[0] == "empty_turn"
                       and r.get("finish_reason") == "length")
                if not cut:
                    truncated_streak = 0
                elif truncated_streak < TRUNCATED_TURN_STREAK:
                    truncated_streak += 1
                    self._log("truncated_turn", turn_idx=turn_idx, streak=truncated_streak)
                    log.info("%s TRUNCATED turn %d -> nudge", self.rid_tag, turn_idx)
                    self.reasoner.append_repair(TRUNCATED_TURN_NUDGE)
                    continue
                if miss is None:
                    no_call_streak = 0
                else:
                    # Repaired once; a repeat (the streak is shared) is the
                    # ending.
                    kind, what, nudge = miss
                    no_call_streak += 1
                    self._log(kind, turn_idx=turn_idx, streak=no_call_streak)
                    if no_call_streak < 2:
                        log.info("%s %s %d -> repair", self.rid_tag, what, turn_idx)
                        self.reasoner.append_repair(nudge)
                        continue
                    log.info("%s %s %d repeated; mission ends", self.rid_tag, what,
                             turn_idx)
                # A clean prose reply before any flight answers nothing: hand
                # the turn back, up to NO_FLIGHT_ABSTAIN_STREAK.
                if no_call_streak == 0 and self.flights == 0:
                    no_flight_streak += 1
                    self._log("no_flight_reply", turn_idx=turn_idx,
                              streak=no_flight_streak)
                    if no_flight_streak < NO_FLIGHT_ABSTAIN_STREAK:
                        log.info("%s reply with ZERO flights at turn %d "
                                 "(streak %d) -> repair", self.rid_tag, turn_idx,
                                 no_flight_streak)
                        self.reasoner.append_repair(
                            "You have flown nowhere. " + _GO_TO_FIRST
                            if withheld("end_commit") else NO_FLIGHT_REPLY_NUDGE)
                        continue
                    log.info("%s reply with ZERO flights at turn %d, streak "
                             "%d -- abstain", self.rid_tag, turn_idx,
                             no_flight_streak)
                log.info("%s mission ended by reply at turn %d (%d flights)",
                         self.rid_tag, turn_idx, self.turn_flights)
                return self._mission_end("reply", turns=turn_idx + 1)
            no_call_streak = truncated_streak = 0
            no_flight_streak = 0

            # Dispatch each call in order, answering every call id.
            for tc in r["tool_calls"]:
                name = (tc.get("function") or {}).get("name") or ""
                cid = tc.get("id") or ""
                args = json.loads(tc["function"]["arguments"])
                if name == "run":
                    flights_before = self.flights
                    out = self._dispatch_run(args)
                    if not withheld("look_for"):
                        out.setdefault("watching", (
                            self._watching_trailer()
                            or "nothing -- no class is segmented on any flight "
                               'until one is looked for; call look_for("<class>").'))
                    self.reasoner.append_tool_result(cid, "run", out)
                    self._log(
                        "run",
                        turn_idx=turn_idx,
                        reasoning=str(r.get("reasoning") or "")[:4000],
                        # the whole block: a flight past any cut would
                        # vanish from the transcript
                        said=str(args.get("code") or ""),
                        # already capped (interpreter.PRINT_CHAR_CAP)
                        printed=str(out.get("printed") or ""),
                        errored="error" in out,
                        # stops this block flew, an errored block's included
                        flights=self.flights - flights_before,
                        # every image this block showed, by sha
                        frames=list(self._block_frames))
                    self._block_frames.clear()
                else:
                    # An unknown tool's id is answered, never executed.
                    self.reasoner.append_tool_result(cid, name or "unknown", {
                        "error": f"unknown tool {name!r}; available: run"})
            # Pictures ride a user turn after the turn's tool results.
            for vtext, vjpegs in self.pending_vision:
                self.reasoner.append_vision_user(vtext, vjpegs)
            self.pending_vision.clear()
