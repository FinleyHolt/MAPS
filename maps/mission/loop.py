"""The mission conversation: one persistent reasoner, one served tool.

Each ``run`` call carries one block of Python; the interpreter
(``maps.library.interpreter``) runs it in the mission's one namespace, and
each library call runs on one verb of this mission: a program
(``program_verbs``), a flight (``fly_verbs``), a picture (``picture_verbs``),
the watchlist (``watch_verbs``) or a stance (``surface_state``), all over one
:class:`Surface`. ``go_to`` / ``explore`` fly the moment they land and the
query surface re-derives from the world the flight folded. The model's plain
reply ends the mission and the last flight's target is the answer (the engine
owns the commit); ``MissionSpec`` holds the liveness bounds the model never
hears."""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

from dataclasses import dataclass

from maps.library.call_arguments import MissionEnv
from maps.library.interpreter import MissionPython
from maps.library.op_compiler import HANDLE_WORD_RE
from maps.library.rows import UNNAMED
from maps.llm.http import CallCanceled
from maps.mission.acks import Rejections, summarise_exc
from maps.mission.compactor import CompactionInfeasible, CompactionInvariantError
from maps.mission.fly_verbs import Flights
from maps.mission.picture_verbs import Pictures
from maps.mission.program_verbs import Programs
from maps.mission.reasoner import Reasoner
from maps.mission.reply_repair import (
    MALFORMED_ENVELOPE_NUDGE,
    NO_FLIGHT_ABSTAIN_STREAK,
    NO_FLIGHT_REPLY_NUDGE,
    TRUNCATED_TURN_NUDGE,
    TRUNCATED_TURN_STREAK,
    _envelope_residue,
    _not_a_reply,
)
from maps.mission.serve_refusals import _REFUSAL_ENDINGS, _RETRIABLE_LLM_ERRORS
from maps.mission.served_text import SERVED_TOOLS
from maps.mission.spec import MissionReply, MissionSpec
from maps.mission.surface_state import Progress, Stance, Surface, surface_from_ctx
from maps.mission.watch_verbs import Watchlist
from maps.settings import Settings
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
    mission = _Mission(
        reasoner=reasoner,
        workdir=workdir,
        rid_tag=rid_tag,
        preflight=True,
        compute_namespace=surface_from_ctx(turn["ctx"]),
        **turn,
    )
    return mission.converse(instruction.strip()), mission


def continue_mission(mission: "_Mission", instruction: str, *, rid_tag: str) -> MissionReply:
    """One more operator turn on the same mission: it follows a reply (the
    field path: the reply is a report), reads the surface the mission's
    ``ctx`` holds now, and keeps every name, handle, flown stop, staged view
    and watched class of the turns before it."""
    mission.progress.rid_tag = rid_tag
    mission.surface.world_moved()
    return mission.converse(instruction.strip())


@dataclass
class _Streaks:
    """An operator turn's consecutive turns that were not progress; a tool
    call ends the first three, any answered request the last."""

    #: Dropped messages: an empty turn or a block of code sent as text.
    no_call: int = 0
    #: Clean replies with no flight behind them (NO_FLIGHT_ABSTAIN_STREAK).
    no_flight: int = 0
    #: Empty turns the output cap cut off (TRUNCATED_TURN_STREAK).
    truncated: int = 0
    #: Transient request failures (_MAX_TRANSIENT).
    transient: int = 0


class _Mission:
    """One mission's conversation, across every operator turn: the reasoner,
    the interpreter its blocks run in, and the verbs the library's calls run
    on, each a component over the one surface."""

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
        self.transcript = transcript
        self.diag = diag
        self.mission_spec = mission_spec
        self.retask = retask
        self.tools = list(SERVED_TOOLS)
        # The lines of every block run, a running total on each result.
        self._script_lines = 0

        self.progress = Progress(rid_tag=rid_tag)
        # The interpreter is built last (it runs on the verbs), so the two
        # verbs that say the model's own name for a set ask it through these.
        self.surface = Surface(
            ctx,
            compute_namespace,
            preflight=preflight,
            name_of_ref=lambda ref: self.python.names.ref_name(ref) or UNNAMED,
        )
        self.rejections = Rejections(self.surface)
        self.programs = Programs(self.surface, self.rejections, self.progress)
        self.stance = Stance(self.surface, self.rejections)
        self.pictures = Pictures(
            self.surface,
            self.rejections,
            self.progress,
            look_renderer=look_renderer,
            workdir=workdir,
            obs_span_m=obs_span_m,
            obs_minor_m=obs_minor_m,
            name_of_ref=lambda ref: self.python.names.ref_name(ref),
            look_argument=lambda: self.python.names.call_arg("look"),
        )
        self.flights = Flights(
            mission_spec,
            self.surface,
            self.rejections,
            self.pictures,
            self.progress,
            obs_span_m=obs_span_m,
        )
        self.watchlist = Watchlist(
            mission_spec, self.surface, self.rejections, self.progress, boost_queue=boost_queue
        )

        # Each library call's verb: what the interpreter's calls run on.
        carry = self.surface.carry
        self.python = MissionPython(
            MissionEnv(
                rows_of_ref=lambda ref: list(carry.read(ref) or []),
                tied_of_ref=lambda ref: bool(getattr(carry.read(ref), "name_set", False)),
                version=lambda: carry.version,
                plan_rn=carry.next_rn,
                widened=carry.paged,
                minted=carry.minted,
                row_of_fid=self.surface.rows.get,
                say_err=self.rejections.error,
                ground=self.programs.ground,
                analyze=self.programs.analyze,
                finish=self.programs.finish,
                stand=self.stance.stand,
                move=self.flights.go_to,
                cover=self.flights.explore,
                crops=self.pictures.look,
                map=self.pictures.show_map,
                watch=self.watchlist.watch,
                boost=self.watchlist.boost,
            )
        )

    def _new_turn(self) -> None:
        """An operator turn's own state: its flight budget (the backstop's
        count; ``progress.flights`` counts the mission's), its escalating
        loop breaker and its dedup streak."""
        self.progress.turn_flights = 0
        self.rejections.new_turn()
        self.programs.new_turn()

    def _log(self, kind: str, **row) -> None:
        """One transcript row, when the mission keeps a transcript."""
        if self.transcript is not None:
            self.transcript.append({"kind": kind, **row})

    def _mission_end(self, ended_by: str, turns: int = 0) -> MissionReply:
        self._log("mission_end", ended_by=ended_by, turns=int(turns))
        return MissionReply(ended_by=ended_by, turns_used=int(turns))

    def converse(self, user_message: str) -> MissionReply:
        """The mission's turn loop: append ``user_message``, then serve
        ``run`` until the model's plain reply or a harness ending; an empty
        turn, a block of code and a reply with no flight behind it are
        repaired before a repeat is accepted as the ending."""
        rid_tag = self.progress.rid_tag
        self._new_turn()
        if self.reasoner.compaction_infeasible:
            return self._mission_end("compaction_infeasible")
        self.reasoner.append_user(user_message)
        streaks = _Streaks()
        for turn_idx in itertools.count():
            self.progress.turn_idx = turn_idx
            if self.diag is not None:
                self.diag["turns_used"] = turn_idx + 1
            if turn_idx >= self.mission_spec.turn_cap:
                log.info("%s mission turn cap (%d) reached", rid_tag, self.mission_spec.turn_cap)
                return self._mission_end("turn_cap", turns=turn_idx)
            self._take_retasks(turn_idx)
            try:
                r = self.reasoner.step_tool_call(self.tools)
            except (CompactionInfeasible, CompactionInvariantError) as exc:
                return self._compaction_infeasible(exc, turn_idx, user_message)
            except _RETRIABLE_LLM_ERRORS as exc:
                ending = self._request_failed(exc, streaks, turn_idx)
                if ending is not None:
                    return ending
                continue
            streaks.transient = 0

            if r["kind"] == "text":
                nudge = self._repair_for(r, streaks, turn_idx)
                if nudge is not None:
                    self.reasoner.append_repair(nudge)
                    continue
                log.info(
                    "%s mission ended by reply at turn %d (%d flights)",
                    rid_tag,
                    turn_idx,
                    self.progress.turn_flights,
                )
                return self._mission_end("reply", turns=turn_idx + 1)
            streaks.no_call = streaks.truncated = streaks.no_flight = 0
            self._answer_tool_calls(r, turn_idx)

    def _take_retasks(self, turn_idx: int) -> None:
        """The operator's re-tasks land at the top of a turn, never between
        a turn's tool calls and their results."""
        if self.retask is None:
            return
        while (operator_turn := self.retask()) is not None:
            log.info(
                "%s operator re-task at turn %d (%d chars)",
                self.progress.rid_tag,
                turn_idx,
                len(operator_turn),
            )
            self.reasoner.append_user(operator_turn)

    def _compaction_infeasible(
        self, exc: Exception, turn_idx: int, user_message: str
    ) -> MissionReply:
        """The transcript cannot be folded into the window. Terminal: flag
        the thread and drop this turn's unread message."""
        self.reasoner.compaction_infeasible = True
        msgs = self.reasoner.messages
        if (
            turn_idx == 0
            and msgs
            and msgs[-1].get("role") == "user"
            and msgs[-1].get("content") == user_message
        ):
            msgs.pop()
        log.warning(
            "%s compaction infeasible -> mission ends (thread flagged): %s",
            self.progress.rid_tag,
            summarise_exc(exc),
        )
        return self._mission_end("compaction_infeasible", turns=turn_idx + 1)

    def _request_failed(
        self, exc: Exception, streaks: _Streaks, turn_idx: int
    ) -> MissionReply | None:
        """A failed reasoner request: the ending of a rejection a retry
        would only repeat or of a spent transient budget; None after the
        wait before a retry."""
        rid_tag = self.progress.rid_tag
        for refused, ended_by, why in _REFUSAL_ENDINGS:
            if refused(exc):
                log.warning("%s %s: %s", rid_tag, why, summarise_exc(exc))
                self._log(ended_by, error=summarise_exc(exc))
                return self._mission_end(ended_by, turns=turn_idx + 1)
        streaks.transient += 1
        cancel = self.reasoner.cancel_event
        log.warning(
            "%s transient LLM error %d%s: %s",
            rid_tag,
            streaks.transient,
            "" if cancel is not None else f"/{_MAX_TRANSIENT}",
            summarise_exc(exc),
        )
        self._log("transient_error", attempt=streaks.transient, error=summarise_exc(exc))
        if cancel is None:
            if streaks.transient >= _MAX_TRANSIENT:
                return self._mission_end("transient_failures", turns=turn_idx + 1)
            time.sleep(min(15.0, 0.5 + 2.0 * streaks.transient))
        elif cancel.wait(min(30.0, 0.5 + 2.0 * streaks.transient)):
            raise CallCanceled("stopped while waiting for the reasoner")
        return None

    def _repair_for(self, r: dict, streaks: _Streaks, turn_idx: int) -> str | None:
        """The nudge a text turn draws, or None when it stands as the reply
        that ends the mission."""
        content = str(r.get("content") or "")
        if _envelope_residue(content) or _envelope_residue(str(r.get("reasoning") or "")):
            log.info("%s MALFORMED_ENVELOPE at turn %d -> repair", self.progress.rid_tag, turn_idx)
            self._log("malformed_envelope", turn_idx=turn_idx)
            return MALFORMED_ENVELOPE_NUDGE
        dropped = _not_a_reply(content)
        cut_off = (
            dropped is not None
            and dropped[0] == "empty_turn"
            and r.get("finish_reason") == "length"
        )
        if not cut_off:
            streaks.truncated = 0
        elif streaks.truncated < TRUNCATED_TURN_STREAK:
            return self._truncated_turn(streaks, turn_idx)
        if dropped is not None:
            return self._dropped_message(dropped, streaks, turn_idx)
        streaks.no_call = 0
        if self.progress.flights == 0:
            return self._reply_before_any_flight(streaks, turn_idx)
        return None

    def _truncated_turn(self, streaks: _Streaks, turn_idx: int) -> str:
        """A turn the output cap cut off empty: the serve's truncation, so
        it spends no repair."""
        streaks.truncated += 1
        self._log("truncated_turn", turn_idx=turn_idx, streak=streaks.truncated)
        log.info("%s TRUNCATED turn %d -> nudge", self.progress.rid_tag, turn_idx)
        return TRUNCATED_TURN_NUDGE

    def _dropped_message(
        self, dropped: tuple[str, str, str], streaks: _Streaks, turn_idx: int
    ) -> str | None:
        """An empty turn or a block of code sent as text: repaired once; a
        repeat (the streak is shared) is the ending."""
        kind, what, nudge = dropped
        streaks.no_call += 1
        self._log(kind, turn_idx=turn_idx, streak=streaks.no_call)
        if streaks.no_call < 2:
            log.info("%s %s %d -> repair", self.progress.rid_tag, what, turn_idx)
            return nudge
        log.info("%s %s %d repeated; mission ends", self.progress.rid_tag, what, turn_idx)
        return None

    def _reply_before_any_flight(self, streaks: _Streaks, turn_idx: int) -> str | None:
        """A clean prose reply before any flight answers nothing: hand the
        turn back, up to NO_FLIGHT_ABSTAIN_STREAK."""
        streaks.no_flight += 1
        self._log("no_flight_reply", turn_idx=turn_idx, streak=streaks.no_flight)
        if streaks.no_flight < NO_FLIGHT_ABSTAIN_STREAK:
            log.info(
                "%s reply with ZERO flights at turn %d (streak %d) -> repair",
                self.progress.rid_tag,
                turn_idx,
                streaks.no_flight,
            )
            return NO_FLIGHT_REPLY_NUDGE
        log.info(
            "%s reply with ZERO flights at turn %d, streak %d -- abstain",
            self.progress.rid_tag,
            turn_idx,
            streaks.no_flight,
        )
        return None

    def _answer_tool_calls(self, r: dict, turn_idx: int) -> None:
        """Dispatch each call of a turn in order, answering every call id;
        the pictures its blocks showed ride a user turn after the results."""
        for tc in r["tool_calls"]:
            name = (tc.get("function") or {}).get("name") or ""
            cid = tc.get("id") or ""
            args = json.loads(tc["function"]["arguments"])
            if name != "run":
                # An unknown tool's id is answered, never executed.
                self.reasoner.append_tool_result(
                    cid, name or "unknown", {"error": f"unknown tool {name!r}; available: run"}
                )
                continue
            flights_before = self.progress.flights
            out = self._run_block(args)
            out.setdefault(
                "watching",
                (
                    self.watchlist.trailer()
                    or "nothing -- no class is segmented on any flight "
                    'until one is looked for; call look_for("<class>").'
                ),
            )
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
                flights=self.progress.flights - flights_before,
                # every image this block showed, by sha
                frames=list(self.pictures.block_frames),
            )
            self.pictures.block_frames.clear()
        for caption, jpegs in self.pictures.pending:
            self.reasoner.append_vision_user(caption, jpegs)
        self.pictures.pending.clear()

    def _run_block(self, args: dict) -> dict:
        """One code block, run in the mission's persistent namespace; an
        error costs the line it is on."""
        if "__args_parse_error" in args:
            return {
                "error": "the call's arguments did not parse as JSON: "
                f"{args['__args_parse_error']}; nothing ran"
            }
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return {"error": "run takes one block of code in code"}
        # Every rN in play (bound handles and those this block spells): a
        # flight's mint skips them.
        carry = self.surface.carry
        carry.in_play.clear()
        carry.in_play.update(HANDLE_WORD_RE.findall(code))
        carry.in_play.update(self.python.names.held_refs())
        carry.minted.clear()
        self._script_lines += len(code.strip().splitlines())
        out = self.python.run_block(code)
        out["script_lines"] = self._script_lines
        return out
