"""The mission conversation: one persistent reasoner, one served tool.

Each ``run`` call carries one block of Python; ``python_surface`` runs it
in the mission's one namespace on this loop's verbs (ground, analyze, the
acts, finish). ``go_to`` / ``explore`` fly the moment they land and the
query surface re-derives from the world the flight folded. The model's
plain reply ends the mission and the last flight's target is the answer
(the engine owns the commit); ``MissionSpec`` holds the liveness bounds
the model never hears.
"""

from __future__ import annotations

import difflib
import hashlib
import itertools
import json
import os
import re
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from maps.llm.http import TRANSIENT_CLIENT_STATUS, CallCanceled
from maps.search.analyze_scene.compactor import (
    CompactionInfeasible,
    CompactionInvariantError,
)
from maps.search.render.look_render import (
    look_label,
    look_outline,
    look_xy,
    up_bearing_deg,
)
from maps.util.xy_token import format_xy_token, parse_xy_token
from maps.search.analyze_scene.tools import ToolContext
from maps.search.withhold import withheld
from maps.search.analyze_scene.text_match import (
    FUZZY_THRESHOLD as _FUZZY_FLOOR,
    unmatched_query_tokens as _unmatched_query_tokens,
)
from maps.search.analyze_scene.geom_program import (
    _BROAD_DUMP_SOFT_ROWS,
    build_surface_from_graph,
    MODEL_UNSERVED_COLUMNS,
)
from maps.search.parse_compile.carry import (
    Carry,
    _program_dedup_key,
    program_reads,
    run_ops,
)
from maps.search.parse_compile.prompts import served_tools
from maps.search.parse_compile.reasoner_agent import Reasoner
from maps.search.render.look_render import LOOK_MAX_BADGES as _MAP_MAX_MARKS
from maps.search.parse_compile import python_surface as _py_surface
from maps.search.parse_compile.passes import cover_items
from maps.search.parse_compile.views import (
    LOOK_MAX_CROPS,
    _crop_token_of,
    _look_vision_text,
    _rounded_xy,
    crop_requests,
    map_features,
    map_spec,
    map_vision_text,
)

# Transient LLM-side errors: a retry, never the end of the reasoner (a
# non-2xx, a link fault, or a ValueError: an undecodable body or a collapsed
# completion).
_RETRIABLE_LLM_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPStatusError,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    ValueError,
)

#: What a serve says when it rejects a request for its SIZE (lowercased
#: body substrings). A spelling this misses falls through to the retry path.
_OVERSIZE_REJECTION = (
    "maximum context length",
    "context length of",
    "longer than the maximum",
    "reduce the length",
)


def _is_oversize_rejection(exc: BaseException) -> bool:
    """Did the serve reject this request because it does not FIT? A retry
    would re-send the same bytes for the same answer."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    if not 400 <= exc.response.status_code < 500:
        return False
    try:
        body = exc.response.text.lower()
    except Exception:
        return False
    return any(s in body for s in _OVERSIZE_REJECTION)


def _is_refused_request(exc: BaseException) -> bool:
    """Did the serve refuse what was sent (a 4xx outside
    ``TRANSIENT_CLIENT_STATUS``, which re-sending fails the same way)?"""
    return (isinstance(exc, httpx.HTTPStatusError)
            and 400 <= exc.response.status_code < 500
            and exc.response.status_code not in TRANSIENT_CLIENT_STATUS)


def _summarise_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


# The dedup hit serves a one-line REFERENCE, never a fresh payload (a full
# re-render of an identical result is loop fuel). Every path that can mint a
# row bumps the surface version, so "nothing has entered the world" is true
# whenever it fires.
_DEDUP_NOTE = (
    "no new information since turn {turn}: nothing has entered the world "
    "since then -- no flight{no_look_for} -- so every program returns what "
    "it returned then, and this one is not re-rendered. Its handles stay "
    "live for later ops. Change the world (fly{look_for}) or "
    "ask something different."
)
_DEDUP_REPEAT_NOTE = (
    "still nothing new, a {n} time: the world has not moved since that turn, "
    "so no program can answer differently until something enters it -- a "
    "flight folds new detections. Live now: {counts}{cols}."
)


def _live_counts(c_n: int, a_n: int, sep: str = ",") -> str:
    """The census's fid counts; the prior term only when the prior is served."""
    return f"{c_n} detection fid(s)" + (
        "" if withheld("prior") else f"{sep} {a_n} prior map row fid(s)")


def _ordinal(n: int) -> str:
    """1 -> 1st, 2 -> 2nd, 3 -> 3rd, 11..13 -> th."""
    n = int(n)
    if 10 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


log = logging.getLogger(__name__)


def _deliver_frames(jpegs: list[bytes], sink: list[str]) -> None:
    """Record every image about to reach the model: its sha256 (the wire
    log's ``image_ref.sha256``) into ``sink`` and the bytes into
    ``<workdir>/frames/<sha256>.jpg`` when a mission is recording. Never
    raises: a full disk must not end the turn."""
    workdir = os.environ.get("SEARCH_COMPILE_WORKDIR", "").strip()
    for jpeg in jpegs:
        if not jpeg:
            continue
        sha = hashlib.sha256(jpeg).hexdigest()
        sink.append(sha)
        if not workdir:
            continue
        try:
            d = Path(workdir) / "frames"
            d.mkdir(parents=True, exist_ok=True)
            dst = d / f"{sha}.jpg"
            if dst.exists():
                continue
            tmp = d / f".{sha}.jpg.tmp"
            tmp.write_bytes(jpeg)
            os.replace(tmp, dst)
        except OSError:
            log.warning("frame %s not written under %s", sha[:12], workdir,
                        exc_info=True)



# Tool-call residue: a prose turn carrying one is a malformed envelope,
# repaired, never an ending.
ENVELOPE_RESIDUE = ("<tool_call", "</tool_call", "<function")

# Consecutive zero-flight replies that end the mission: the first two draw
# NO_FLIGHT_REPLY_NUDGE, the third is an abstain. Anything the model
# executes between two replies resets the streak.
NO_FLIGHT_ABSTAIN_STREAK = 3


@dataclass
class MissionSpec:
    """The engine half of the mission contract: ``fly`` flies one move item
    (``{"target_fid": <fid or "[x, y]">}``) and returns its report or
    ``{"error": ...}``; ``declare_concepts`` arms the watchlist and returns
    the error a failed arming serves, or None."""

    fly: "Callable[[dict], dict]"
    declare_concepts: "Callable[[list[str]], str | None]"
    # ``coverage_stops(fid, spacing_m)``: an explore's stops over that fid's
    # footprint, or None when it carries no cells.
    coverage_stops: "Callable[[str, float], list | None] | None" = None
    #: Which stops of a pass the recording already holds
    #: (``[(token, xy or None), ...] -> [bool, ...]``); a held stop flies
    #: first. None on the bench.
    held_now: "Callable[[list], list[bool]] | None" = None
    # Liveness backstops, never served; a stop of a pass is a flight.
    flight_budget: int = 1000
    turn_cap: int = 100
    # The wall-clock backstop as an absolute epoch the harness owns; None
    # (the flight default) means no clock.
    deadline_epoch: float | None = None


@dataclass(frozen=True)
class MissionReply:
    """How the conversation ended: ``ended_by`` is "reply" (the one
    model-owned ending) | "turn_cap" | "transient_failures" |
    "request_refused" | "oversize_rejection" | "compaction_infeasible"."""

    ended_by: str
    turns_used: int = 0


def _served_flight(report: dict) -> dict:
    """A flight's report as the model reads it: never where the aircraft
    was (``pose_xy`` and ``alt_m`` stay for the engine)."""
    return {k: v for k, v in report.items() if k not in ("pose_xy", "alt_m")}


def _named_feature_note(classes: list[str], names: set[str]) -> str | None:
    """A note when a watch class IS a world-state name (the watchlist takes
    visual classes); the arm stands as said."""
    named = [c for c in classes if c.strip().lower() in names]
    if not named:
        return None
    q = ", ".join(f"'{c}'" for c in named)
    return (f"{q}: a NAMED feature on this map, already a world-state row "
            f"-- the segmenter takes visual classes like 'house' or "
            f"'pool'. Watched as said.")


# How many of a pass's new rows ride back as crops (its catch): half the
# image budget, so the model's own looks keep room.
_CATCH_CROPS = LOOK_MAX_CROPS // 2
# What a look and a pass's catch say under the ``images`` withhold.
_PICTURE_WITHHELD_OF = ("the {}'s caption follows in the next message; "
                        "its picture is not served on this run")
_PICTURE_WITHHELD = _PICTURE_WITHHELD_OF.format("crop")


def _surface_from_ctx(ctx) -> dict:
    """The ``{cands, anchors, cell_m}`` surface of a :class:`ToolContext`'s
    graph."""
    cell_m = float(ctx.cell_size_m)
    cands, anchors = build_surface_from_graph(
        ctx.graph.nodes.items(), ctx.tile_origin_xy, cell_m
    )
    return {"cands": cands, "anchors": anchors, "cell_m": cell_m}


def _is_handle_token(token: str) -> bool:
    """True when ``token`` is a result handle, ``rN`` or ``rN[...]``."""
    return bool(re.fullmatch(r"r\d+(\[[^\]]*\])?", token))


def _envelope_residue(text: str) -> bool:
    """True when a prose turn carries tool-call envelope residue."""
    t = str(text or "")
    return any(m in t for m in ENVELOPE_RESIDUE)


EMPTY_TURN_NUDGE = (
    "Your last turn arrived empty -- no tool call and no text, so nothing "
    "executed. Re-emit it: the tool call you intended, or your plain-text "
    "reply."
)

CODE_REPLY_NUDGE = (
    "Your reply is code, but it arrived as plain text, not through the run "
    "tool, so nothing was executed. Send the block with the tool to run "
    "it; a plain reply in your own words ends the mission."
)

MALFORMED_ENVELOPE_NUDGE = (
    "Your reply contains tool-call markup that did not parse as a native "
    "tool call; an envelope written as text is not executed. Only a call on "
    "the API's tool-call channel executes."
)

NO_FLIGHT_REPLY_NUDGE = (
    "Your reply ends the mission, and your answer is the row of your LAST "
    "flight -- but you have flown nowhere, so it answers nothing. Naming "
    "the object commits nothing; only flying to it does. If you know which "
    "one it is, go_to() it first, then reply."
)

# The bench's transient-error budget; a live mission (a reasoner with a
# cancel_event) waits for the arm until it answers or the operator stops.
_MAX_TRANSIENT = 6

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|\n```\s*$")


def _code_reply(text: str) -> bool:
    """True when the whole reply is a block of the surface: it parses whole
    (prose narrating a call does not) and carries a library call (plain
    prose that happens to parse does not)."""
    t = _CODE_FENCE_RE.sub("", str(text or "").strip()).strip()
    return bool(t) and _py_surface.looks_like_code(t)


def _not_a_reply(content: str) -> tuple[str, str, str] | None:
    """A text turn that is a dropped message rather than a reply -- empty,
    or a block of code -- as ``(transcript kind, log label, nudge)``."""
    if not content.strip():
        return "empty_turn", "EMPTY turn", EMPTY_TURN_NUDGE
    if _code_reply(content):
        return "code_shaped_reply", "CODE-SHAPED reply turn", CODE_REPLY_NUDGE
    return None


def run_mission(
    *,
    instruction: str,
    reasoner_url: str,
    endpoint_spec,
    api_key: str = "EMPTY",
    seed: int | None = None,
    sensor_description: str = "",
    cancel_event: "threading.Event | None" = None,
    reasoning_effort: str | None = None,
    rid_tag: str = "m0",
    **turn,
) -> tuple[MissionReply, "_Mission"]:
    """THE mission entry: build the Reasoner and the mission over it, and run
    the first operator turn (``turn`` is :class:`_Mission`'s). Returns the
    reply and the mission, for :func:`continue_mission`."""
    reasoner = Reasoner(
        reasoner_url=reasoner_url,
        endpoint_spec=endpoint_spec,
        api_key=api_key,
        seed=seed,
        sensor_description=sensor_description,
        cancel_event=cancel_event,
        reasoning_effort=reasoning_effort,
    )
    mission = _Mission(reasoner=reasoner, rid_tag=rid_tag, preflight=True,
                       compute_namespace=_surface_from_ctx(turn["ctx"]), **turn)
    return mission.converse(instruction.strip()), mission


def continue_mission(mission: "_Mission", instruction: str, *,
                     rid_tag: str) -> MissionReply:
    """One more operator turn on the same mission: it follows a reply (the
    field path: the reply is a report), reads the surface the mission's
    ``ctx`` holds now, and keeps every name, handle, flown stop, staged view
    and watched class of the turns before it."""
    mission.rid_tag = rid_tag
    mission.preflight = False
    mission._world_moved()
    return mission.converse(instruction.strip())


class _Mission:
    """One mission's conversation state, across every operator turn, and the
    verbs the run surface calls on it."""

    def __init__(
        self,
        *,
        reasoner: Reasoner,
        ctx: ToolContext,
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
    ):
        self.reasoner = reasoner
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
        # Successful self-contained programs by ``_program_dedup_key`` ->
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

        self.carry = Carry(compute_namespace)
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

        # HALF THE OBSERVATION'S LONG SIDE: a pass's stop spacing. Half of
        # ``obs_span_m`` (the footprint's long side) sits under the short side
        # for any sensor below 2:1, so a ONE-DIMENSIONAL walk covers a shape that
        # is not one: consecutive stops overlap on the minor axis too.
        self._cover_spacing = (float(self.obs_span_m) / 2.0
                               if isinstance(self.obs_span_m, (int, float))
                               and self.obs_span_m > 0 else None)

        # The call in flight: what _ground_names noted and _analyze answered,
        # read by _finish.
        self._call: dict = {}

        self._mission_py = _py_surface.MissionPython(_py_surface.MissionEnv(
            rows_of_ref=lambda ref: list(self.carry.read(ref) or []),
            tied_of_ref=lambda ref: bool(
                getattr(self.carry.read(ref), "name_set", False)),
            plan_rn=self.plan_rn, widened=self._widened, minted=self._code_minted,
            say_err=self._plan_err, row_of_fid=self._ns_rows.get,
            ground=self._ground_names, analyze=self._analyze, stand=self._stand,
            move=self._move_to, cover=self._cover, crops=self._crops, map=self._map,
            watch=self._watch, boost=self._boost, finish=self._finish))

    def _new_turn(self) -> None:
        """An operator turn's own state: its flight budget (the backstop's
        count; the mission's flights are the reasoner's), its escalating loop
        breaker (see _loop_broken_error) and its dedup streak."""
        self.turn_flights = 0
        self._err_class_counts: dict[str, int] = {}
        self._err_sig, self._err_n = None, 0
        self._dedup_key, self._dedup_n = None, 0

    def _world_moved(self) -> None:
        """The world stepped: refresh both surface views and bump the
        version."""
        self.ns.update(_surface_from_ctx(self.ctx))
        self._rebuild_ns_rows()
        self.carry.version += 1

    def _rebuild_ns_rows(self) -> None:
        # cands win on fid collision.
        self._ns_rows.clear()
        for _key in ("cands", "anchors"):
            for _r in self.ns.get(_key) or []:
                self._ns_rows.setdefault(str(_r["fid"]), _r)

    def _resolve_stage_point(self, tok) -> list[float] | None:
        """A stage at/facing slot -> [x, y]: a live surface fid, a flown
        stop id, a staged handle, or the canonical '[x, y]' token; None
        otherwise."""
        if isinstance(tok, str):
            row = self._ns_rows.get(tok.strip())
            if row is not None:
                return look_xy(row)
            sp = self._plan_stops.get(tok.strip().lower())
            if sp is not None:
                return [round(sp[0], 2), round(sp[1], 2)]
            sv = self.staged_viewpoints.get(tok.strip().lower())
            if sv is not None:
                return [round(float(sv["from_xy"][0]), 2),
                        round(float(sv["from_xy"][1]), 2)]
            return _rounded_xy(tok)
        return None

    def _stage_apply(self, at_tok: str, facing_tok: str) -> dict:
        """stand(): mint a vN handle that becomes STANDING and bump the
        surface version (a program can answer differently under it)."""
        from maps.search.analyze_scene.geom_compute import COMPASS_UNIT  # noqa: PLC0415

        at_xy = self._resolve_stage_point(at_tok)
        if at_xy is None:
            return {"error": f"at {at_tok!r} is not a live surface fid or an "
                             f"[x, y] world point"}
        if facing_tok.strip().lower() in COMPASS_UNIT:
            up = list(COMPASS_UNIT[facing_tok.strip().lower()])
        else:
            f_xy = self._resolve_stage_point(facing_tok)
            if f_xy is None:
                return {"error": (f"facing {facing_tok!r} is not a live surface "
                                  f"fid, an [x, y] point, or one of "
                                  f"{'|'.join(COMPASS_UNIT)}")}
            dx = f_xy[0] - at_xy[0]
            dy = f_xy[1] - at_xy[1]
            n = (dx * dx + dy * dy) ** 0.5
            if n < 1e-6:
                return {"error": "at and facing coincide -- no look direction"}
            up = [dx / n, dy / n]
        h = f"v{len(self.staged_viewpoints) + 1}"
        brg = round(up_bearing_deg(up)) % 360
        self.staged_viewpoints[h] = {
            "handle": h,
            "from_xy": at_xy,
            "up_unit": [round(up[0], 4), round(up[1], 4)],
            "bearing_deg": brg,
        }
        self.standing_handle = h
        self.ns["staged"] = self.staged_viewpoints
        self.ns["standing"] = h
        self.carry.version += 1
        return {
            "staged": [{"handle": h, "at_xy": at_xy,
                        "facing_bearing_deg": brg}],
            "note": (f"standing viewpoint = {h}: 'left of {h}'/'right of "
                     f"{h}'/'ahead of {h}'/'behind {h}' ride every find "
                     f"row; {h} is a point ref ('distance to {h}'); crops "
                     f"and maps render its facing up."),
        }

    def _standing_frame(self) -> tuple[list | None, str]:
        """The STANDING viewpoint's image-up and the caption note stating it
        (never a silent rotation); ``(None, "")`` with no stance."""
        if self.standing_handle is None:
            return None, ""
        _sv = self.staged_viewpoints[self.standing_handle]
        return (list(_sv["up_unit"]),
                f"image-up = {_sv['handle']}'s facing "
                f"(bearing {_sv['bearing_deg']})")

    def _map_features_cached(self) -> list[dict]:
        if self._map_feats_version != self.carry.version:
            self._map_feats_version = self.carry.version
            self._map_feats = map_features(self.ctx)
        return self._map_feats

    def _loop_breaker_facts(self, _n: int) -> str:
        def _cap(fids: list[str]) -> str:
            return f"{fids[:8]}" + (
                f" (+{len(fids) - 8} more)" if len(fids) > 8 else "")
        c_n, a_n, cols = self._surface_census_counts()
        parts = [
            f" This error class has now occurred {_n}x.",
            f" Live surface: {_live_counts(c_n, a_n, ';')}.",
        ]
        if cols:
            parts.append(
                f" Columns computed on rows so far: {_cap(cols)}.")
        return "".join(parts)

    def _err_class_sig(self, err: str) -> str:
        # Same-class signature: quoted tokens and numbers stripped.
        s = re.sub(r"'[^']*'", "'_'", str(err))
        s = re.sub(r"\d+", "_", s)
        return s[:100]

    def _surface_census_counts(self) -> tuple[int, int, list[str]]:
        """Live counts: detections, priors, computed columns."""
        c_n = len(self.ns.get("cands") or [])
        a_n = len(self.ns.get("anchors") or [])
        cols = sorted({k for _row in self.row_latest.values()
                       for k, _v in _row.items()
                       if k not in MODEL_UNSERVED_COLUMNS
                       and _v not in (None, "")})
        return c_n, a_n, cols

    def _repeated_rejection_state(self, err: str, _n: int, _consec: int) -> str:
        """The 3rd-consecutive-rejection fact, served instead of the remedy
        and census again: the counts, the rejection's first clause and the
        live state."""
        _first = str(err).split(". ", 1)[0].strip().rstrip(".")
        c_n, a_n, cols = self._surface_census_counts()
        _cols_s = f"{cols[:12]}" + (
            f" (+{len(cols) - 12} more)" if len(cols) > 12 else "")
        return (
            f"rejected again -- the same error class, {_consec}x "
            f"consecutively ({_n}x): {_first}. The full "
            f"rejection text was served at its 1st occurrence and the "
            f"live-surface census at its 2nd (both above in this "
            f"conversation, unchanged). The executor is deterministic on "
            f"the same surface: re-issuing the same call unchanged "
            f"returns exactly this rejection. Live now: "
            f"{_live_counts(c_n, a_n)}"
            + (f"; columns computed on rows: {_cols_s}." if cols else "."))

    def _loop_broken_error(self, err: str, sig_of: str | None = None) -> str:
        """A repeated same-class error, escalating: verbatim first, then with
        the live census, then (3rd consecutive on) the repetition fact
        alone. A shape error gets only its count. A successful call resets
        the consecutive streak, never the class count."""
        _sig = self._err_class_sig(err if sig_of is None else sig_of)
        _n = self._err_class_counts[_sig] = self._err_class_counts.get(_sig, 0) + 1
        if self._err_sig == _sig:
            self._err_n += 1
        else:
            self._err_sig, self._err_n = _sig, 1
        _consec = self._err_n
        if _n < 2:
            return err
        _SHAPE_MARKERS = ("unknown ref", "is a result handle",
                          "syntax error")
        if any(m in err for m in _SHAPE_MARKERS):
            return err + f" This error class has now occurred {_n}x."
        if _consec >= 3:
            return self._repeated_rejection_state(err, _n, _consec)
        return err + self._loop_breaker_facts(_n)

    def _dispatch_analyze_scene(self, ops, *, answer_ref=""):
        _key = _program_dedup_key(ops)
        _prev = self._programs_seen.get(_key) if _key is not None else None
        if _prev is not None and _prev[0] == self.carry.version:
            # A one-line reference, never a payload; the emission's own
            # labels alias the stored handles so nothing downstream can tell
            # it from a re-run.
            self.carry.alias(_prev[2], ops)
            if self._dedup_key == _key:
                self._dedup_n += 1
            else:
                self._dedup_key, self._dedup_n = _key, 1
            if self._dedup_n >= 2:
                _c_n, _a_n, _dcols = self._surface_census_counts()
                _cols_s = ("; columns live: " + ", ".join(_dcols[:12])
                           if _dcols else "")
                return {"note": _DEDUP_REPEAT_NOTE.format(
                    n=_ordinal(self._dedup_n),
                    counts=_live_counts(_c_n, _a_n), cols=_cols_s)}
            _lf = not withheld("look_for")
            return {"note": _DEDUP_NOTE.format(
                turn=_prev[1], no_look_for=", no look_for(..., at=)" if _lf else "",
                look_for=", look_for(..., at=)" if _lf else "")}
        _raw_sink: dict = {}
        self.carry.ensure_fresh(program_reads(ops))
        out = run_ops(ops, namespace=self.ns,
                      raw_sink=_raw_sink,
                      seed_results=self.carry.rows,
                      preflight=self.preflight,
                      answer_ref=answer_ref)
        if "error" in out:
            # The failing op keeps one op's errors their own class.
            _err = str(out["error"])
            out = {**out, "error": self._loop_broken_error(
                _err, sig_of=f"{_err}|{_raw_sink.get('error_op', '')}")}
        else:
            self._err_sig, self._err_n = None, 0
            self._dedup_key, self._dedup_n = None, 0
            if _key is not None:
                self._programs_seen[_key] = (self.carry.version, self.turn_idx,
                                             ops)
            self.carry.record(ops, _raw_sink["results"])
            for _rows in _raw_sink["results"].values():
                for _row in _rows:
                    if _row.get("fid"):
                        self.row_latest[str(_row["fid"])] = _row
        return out

    def _watching_trailer(self) -> list:
        """The armed watchlist with each class's live detection count."""
        counts: dict[str, int] = {}
        for _r in self.ns.get("cands") or []:
            if str(_r["fid"]).startswith("detected_"):
                _c = str(_r.get("class") or "?")
                counts[_c] = counts.get(_c, 0) + 1
        return [f"{c} ({counts.get(c, 0)} detected)" for c in self.mission_sam]

    def _mint_rows(self, _new_fids: list, _what: str):
        """The rows a flight or a pass added -> the note naming their new rN
        handle; None (and no handle) when nothing was added."""
        if not _new_fids:
            return None
        while f"r{self.plan_rn[0]}" in self._code_names:
            self.plan_rn[0] += 1
        _h = f"r{self.plan_rn[0]}"
        self.plan_rn[0] += 1
        self._code_names.add(_h)
        self._code_minted.append(_h)
        self.carry.mint(_h, [self._ns_rows[_f] for _f in _new_fids])
        self._those = _h
        return (f"{_h} — the {len(_new_fids)} rows this {_what} added; "
                    f"the handle is bound, so {_h}, {_h}[0] and "
                    f"{_h}.order(...) all reach them")

    def _stop_xy(self, _it) -> list[float] | None:
        _t = str(_it.get("target_fid") or "")
        return parse_xy_token(_t) or look_xy(self._ns_rows.get(_t))

    def _pick_next(self, _queue: list, cover: str | None) -> int:
        """The queue index to fly next: in a pass, a stop the recording
        already holds flies first; otherwise the head."""
        if cover is None or len(_queue) < 2 or self.mission_spec.held_now is None:
            return 0
        try:
            _held = self.mission_spec.held_now(
                [(str(_it.get("target_fid") or ""), self._stop_xy(_it)) for _it in _queue])
        except Exception as _exc:  # noqa: BLE001 — the order is a convenience, never a fault
            log.warning("%s pass: held_now failed: %s", self.rid_tag, _summarise_exc(_exc))
            return 0
        _pick = next((_i for _i, _h in enumerate(_held) if _h), 0)
        if _pick:
            log.info("%s pass: %s taken first, the recording holds it",
                     self.rid_tag, _queue[_pick].get("target_fid"))
        return _pick

    def _snap_to_row(self, _item: dict) -> str | None:
        """A point target within 1 m of a row flies as that row's fid; the
        point it was asked as, or None when nothing snapped."""
        _pt = parse_xy_token(_item["target_fid"])
        if _pt is None:
            return None
        _best, _bd = None, 1.0
        for _bf, _brow in self._ns_rows.items():
            _bxy = look_xy(_brow)
            if _bxy is None:
                continue
            _d = ((_bxy[0] - _pt[0]) ** 2 + (_bxy[1] - _pt[1]) ** 2) ** 0.5
            if _d < _bd:
                _best, _bd = _bf, _d
        if _best is None:
            return None
        _snap_from = _item["target_fid"]
        _item["target_fid"] = _best
        return _snap_from

    def _flight_view(self, _item: dict, _report: dict) -> None:
        """THE FLIGHT SHOWS WHAT IT SAW: the target's own look, centred on
        the row it asked for (a stop flown again centres on the stop's
        point), queued for the next message and named in the report."""
        _tfid = str(_item.get("target_fid") or "")
        _trow = self._ns_rows.get(_tfid)
        _txy = look_xy(_trow)
        try:
            _fv = self.look_renderer([{
                "center_fid": _tfid if _txy else "",
                "center_xy": (list(_txy) if _txy
                              else list(_report["pose_xy"])),
                "center_label": (
                    look_label(_tfid, _trow, is_target=True)
                    if _txy else (_tfid or "the stop")),
                "center_boundary": look_outline(_trow) if _txy else None,
                "marks": [],
            }])
        except Exception as _exc:  # noqa: BLE001 — never crash a flight
            log.warning("%s flight view render failed: %s",
                        self.rid_tag, _summarise_exc(_exc))
            _fv = []
        _fj = [c["jpeg"] for c in _fv if c.get("jpeg")]
        if not _fj:
            return
        _fv_text = _look_vision_text(_fv)
        _shown = not withheld("images")
        _fv_note = ("the view follows." if _shown else
                    "the view's caption follows.")
        if _shown:
            _deliver_frames(_fj, self._block_frames)
        self.pending_vision.append((_fv_text, _fj if _shown else []))
        _report["note"] = (f"{_report['note']} {_fv_note}"
                           if _report.get("note") else _fv_note)

    def _after_flight(self, _report: dict, _fids_before: set,
                      cover: str | None) -> None:
        """A flight that flew: count it, record its stop, re-derive the
        surface, and (outside a pass) mint the rows it added."""
        self.turn_flights += 1
        self.reasoner.flights += 1
        _pxy = _report["pose_xy"]
        self._plan_stops[str(_report["stop_id"]).strip().lower()] = [
            float(_pxy[0]), float(_pxy[1])]
        self._world_moved()
        # The flight's new rows (the surface diff) mint a handle; a pass
        # mints once, around the whole of itself.
        if cover is None:
            _m = self._mint_rows([_f for _f in self._ns_rows
                                  if _f not in _fids_before], "flight")
            if _m is not None:
                _report["new_rows"] = _m
        self.preflight = False

    def _fly_moves(self, _accepted: list, _item_errors: list,
                   cover: str | None = None) -> dict:
        """Fly each entry in order; after each flight the surface re-derives
        and the rows it added mint an rN handle. ``cover`` (an explore) makes
        the loop ONE pass: no per-stop view, one handle, one ack."""
        _reports: list[dict] = []
        _cover_before = set(self._ns_rows) if cover is not None else None
        _clock_cut = False
        _queue = list(_accepted)
        while _queue:
            _item = _queue.pop(self._pick_next(_queue, cover))
            if cover is not None:
                # The rest of the pass rides every stop, for the crew.
                _item["pass_ahead"] = [[float(_xy[0]), float(_xy[1])]
                                       for _xy in map(self._stop_xy, _queue) if _xy]
            if self.turn_flights >= self.mission_spec.flight_budget:
                log.warning("%s flight backstop hit (%d flights)",
                            self.rid_tag, self.mission_spec.flight_budget)
                _item_errors.append({"target_fid": _item["target_fid"],
                                     "error": "not flown."})
                continue
            if (self.mission_spec.deadline_epoch is not None
                    and time.time() >= self.mission_spec.deadline_epoch):
                # The wall-clock backstop stops the flying only; the model
                # keeps its turns and its own commit.
                # ponytail: checked between stops, so one slow stop still
                # overruns by its own duration.
                if not _clock_cut:
                    log.warning(
                        "%s wall-clock backstop hit (%d flights, %.0f s past "
                        "the deadline)", self.rid_tag, self.turn_flights,
                        time.time() - self.mission_spec.deadline_epoch)
                _clock_cut = True
                _item_errors.append({"target_fid": _item["target_fid"],
                                     "error": "not flown."})
                continue
            _snap_from = self._snap_to_row(_item)
            _fids_before = set(self._ns_rows)
            _report = self.mission_spec.fly(dict(_item))
            if _snap_from:
                _report.setdefault(
                    "note",
                    f"{_snap_from} is {_item['target_fid']}"
                    "'s own position -- flown as that fid.")
            if "error" not in _report:
                if (not _report.get("new_detections") and not self.mission_sam
                        and not withheld("look_for")):
                    _wl_note = ('the watchlist is empty -- nothing is '
                                'segmented on any flight; call '
                                'look_for("<class>") to see detections')
                    _report["note"] = (f"{_report['note']} {_wl_note}"
                                       if _report.get("note")
                                       else _wl_note)
                # A pass renders no per-stop view.
                if cover is None:
                    self._flight_view(_item, _report)
                self._after_flight(_report, _fids_before, cover)
            _reports.append(_report)
        if cover is not None:
            return self._cover_ack(cover, _reports, _cover_before, _item_errors)
        _mv_ack: dict = {"flown": [_served_flight(r) for r in _reports]}
        if _item_errors:
            _mv_ack["rejected"] = _item_errors
        if not _accepted and not _item_errors:
            _mv_ack = {"error": "nothing to fly"}
        return _mv_ack

    def _cover_ack(self, cover: str, _reports: list, _before: set,
                   _item_errors: list) -> dict:
        """One explore's flight reports -> the one ack of the pass. The ack
        states that the standing answer is the pass's last stop."""
        _ok = [r for r in _reports if "error" not in r]
        _ack: dict = {"explored": cover, "stops": len(_ok)}
        if _ok:
            _ack["answer_if_you_reply_now"] = (
                f"the last stop of the pass over {cover}; go_to() the row "
                f"you mean before you reply")
        _ack["new_detections"] = sum(int(r.get("new_detections") or 0)
                                     for r in _ok)
        _m = self._mint_rows([_f for _f in self._ns_rows if _f not in _before], "pass")
        if _m is not None:
            _ack["new_rows"] = _m
        _errs = [str(r["error"]) for r in _reports if r.get("error")]
        if _errs:
            _ack["not_flown"] = _errs[:4]
        if _item_errors:
            _ack["rejected"] = _item_errors
        if not _ok:
            return {"error": (_errs[0] if _errs
                              else f"nothing of {cover} could be flown")}
        return _ack

    def _watch_apply(self, adds: list[str]) -> dict:
        """look_for(): add classes to the watchlist (a class declared again
        moves to the end; the list only grows) and arm it with the engine."""
        for _c in adds:
            self.mission_sam[:] = [c for c in self.mission_sam if c != _c]
            self.mission_sam.append(_c)
        _err = self.mission_spec.declare_concepts(list(self.mission_sam))
        ack: dict = {"now": [f"{c} (target)" for c in self.mission_sam]}
        if _err:
            ack["error"] = _err
        _names = {s for s in (
            str(n.name or "").strip().lower()
            for n in self.ctx.graph.nodes.copy().values()) if s}
        _note = _named_feature_note(list(adds), _names)
        if _note:
            ack["note"] = _note
        log.info("%s SAY watch: +%s -> %s", self.rid_tag, adds, self.mission_sam)
        return ack

    def _boost(self, where: str, concepts: list) -> dict:
        """look_for(..., at=) -- the targeted boost."""
        res = self.boost_queue(list(concepts), [where]) or {}
        if "error" not in res and res.get("queued"):
            # The placed form also arms the class it names.
            _adds = [str(c) for c in concepts if c]
            if _adds:
                _w = self._watch_apply(_adds)
                res["watching"] = _w.get("error") or _w.get("now")
            res["note"] = ("boost queued -- extra segmentation at the "
                           "asked spot fires on your next flight; the "
                           "class is also on the standing watchlist for "
                           "every later flight")
        if "error" not in res:
            # Any accepted boost refreshes the surface and bumps its version,
            # as a flight does (a retro boost has ingested detections).
            self._world_moved()
        if res.get("error"):
            return self._plan_err({"boost": res}, str(res["error"]))
        return {"boost": res}

    def _ship(self, ack: dict, kind: str, jpegs: list, vis_text: str) -> bool:
        """Queue one look's or map's pictures (or, under the images
        withhold, its caption alone) for the next message; ``ack`` states
        what follows. True when pictures shipped."""
        ack["delivered"] = len(jpegs)
        if jpegs and withheld("images"):
            ack["delivered"] = 0
            ack["note"] = _PICTURE_WITHHELD_OF.format(kind)
            self.pending_vision.append((vis_text, []))
            return False
        if jpegs:
            ack["note"] = f"the {kind} follows in the next message"
            _deliver_frames(jpegs, self._block_frames)
            self.pending_vision.append((vis_text, jpegs))
            return True
        ack["error"] = f"render returned no image bytes; no {kind} will follow"
        return False

    def _crops_of(self, fid, marks: list | None = None,
                   preamble: str = "") -> dict:
        # A list asks for one crop per row.
        _fids = [str(f) for f in fid] if isinstance(fid, (list, tuple)) \
            else [str(fid)]
        reqs, skipped, ref_notes = crop_requests(
            _fids, marks, rows=self._ns_rows, carry=self.carry,
            stops=self._plan_stops, obs_minor_m=self.obs_minor_m,
            obs_span_m=self.obs_span_m, frame=self._standing_frame())
        ack: dict = {"looked": len(reqs)}
        if skipped:
            ack["skipped"] = list(dict.fromkeys(skipped))[:20]
        if ref_notes:
            ack["ref_notes"] = ref_notes[:20]
        if not reqs:
            ack["error"] = ("; ".join(ref_notes) if ref_notes
                            else f"{', '.join(_fids)} is not on the live "
                                 "surface")
            return ack
        try:
            crops = self.look_renderer(reqs)
        except Exception as exc:  # noqa: BLE001 — render never crashes the mission
            log.warning("%s SAY crops render failed: %s", self.rid_tag,
                        _summarise_exc(exc))
            return {"error": f"crop render failed: {_summarise_exc(exc)}"}
        jpegs = [c["jpeg"] for c in crops if c.get("jpeg")]
        vis_text = preamble + _look_vision_text(crops)
        if self._ship(ack, "crop", jpegs, vis_text) and marks:
            # Badges drawn (the legend IS the drawn set) and the renderer's
            # own out-of-frame list, on a crop that shipped.
            ack["badged"] = sum(
                len(c.get("legend") or []) for c in crops if c.get("jpeg"))
            ack["not_drawn"] = [
                str(f) for c in crops if c.get("jpeg")
                for f in (c.get("marks_out_of_view") or [])][:20]
        return ack

    def _crops_set(self, handle: str, marks: list | None = None) -> dict:
        """look(rows): one crop per row of the handle, in its own order, each
        badged with the row it shows; the crop budget is stated when it
        bites."""
        rows = [r for r in (self.carry.read(handle) or []) if r.get("fid")]
        if not rows:
            _mp = self._mission_py
            _name = _mp.ref_name(handle) or _mp.call_arg("look")
            return {"error": (f"{_name or handle} holds no rows to look at; "
                              "name a handle that has some, or find() one first")}
        fids = [_crop_token_of(r, str(r["fid"]) in self._ns_rows) for r in rows]
        return self._crops_of(
            fids, marks=marks,
            # No handle named: the executor's ref may be unbound.
            preamble=("One crop per row, in order, each badged with the "
                      "row it shows. "))

    def _map_of(self, fid: str, mark_fids: list | None = None) -> dict:
        spec, m_skipped = map_spec(fid, mark_fids or [], rows=self._ns_rows,
                                   resolve=self._resolve_stage_point,
                                   frame=self._standing_frame())
        ack: dict = {"maps": int(spec is not None)}
        if m_skipped:
            ack["skipped"] = m_skipped[:20]
        if spec is None:
            ack["error"] = "no resolvable map center"
            return ack
        try:
            from maps.search.render.map_render import (  # noqa: PLC0415
                render_map_from_spec,
            )
            _m = render_map_from_spec(
                spec, self._map_features_cached(),
                tile_origin_xy=self.ctx.tile_origin_xy,
                cell_size_m=float(self.ctx.cell_size_m),
                grid_shape=tuple(self.ctx.grid_shape))
            maps = [_m] if _m is not None else []
        except Exception as exc:  # noqa: BLE001 — render never crashes the mission
            log.warning("%s SAY map render failed: %s", self.rid_tag,
                        _summarise_exc(exc))
            return {"error": f"map render failed: {_summarise_exc(exc)}"}
        jpegs = [m["jpeg"] for m in maps if m.get("jpeg")]
        self._ship(ack, "map", jpegs, map_vision_text(maps))
        return ack

    def _plan_err(self, ack: dict, msg: str) -> dict:
        """A plan error, through the escalating loop breaker."""
        return {**ack, "error": self._loop_broken_error(str(msg))}

    def _flown_stop(self, tok: str) -> tuple[str, str | None]:
        """A flown stop id -> its recorded point; an unflown one is refused.
        ``(token, error)``."""
        _sp = self._plan_stops.get(tok.strip().lower())
        if _sp is not None:
            return format_xy_token(_sp[0], _sp[1]), None
        if re.fullmatch(r"stop_[0-9]{1,4}", tok.strip().lower()):
            _known = ", ".join(sorted(self._plan_stops)) or "none yet"
            return tok, (f"{tok} has not been flown; the flown stops are: "
                         f"{_known}")
        return tok, None

    def _closest_names(self, what: str) -> str:
        """The 3 nearest named rows to a name that matched nothing."""
        names: dict[str, tuple] = {}
        for _f, _r in self._ns_rows.items():
            _nm = str(_r.get("name") or "").strip()
            if _nm:
                names.setdefault(_nm.lower(), (_f, _nm))
        close = difflib.get_close_matches(
            (what or "").lower(), list(names), n=3, cutoff=0.55)
        if not close:
            return ""
        return "; closest named rows: " + ", ".join(
            f"\"{names[c][1]}\" ({names[c][0]})" for c in close)

    def _call_minted(self):
        """The handle this call's last flight minted, or None."""
        return (self._code_minted[self._call.get("minted", 0):] or [None])[-1]

    def _ground_names(self, names: dict) -> dict:
        """Open a call and ground its names (``{name: why}``) under the
        name-match floor: ``{"ties": {name: top-tie fids}}``, or the first
        refusal."""
        self._call.clear()
        self._call["notes"] = notes = []
        self._call["minted"] = len(self._code_minted)   # this call's flights mint past it
        ties: dict = {}
        if not names:
            return {"ties": ties}
        _sink: dict = {}
        g_out = run_ops(
            [{"ref": f"r{i}", "op": "find", "args": {"query": q.lower()}}
             for i, q in enumerate(names)],
            namespace=self.ns,
            raw_sink=_sink, preflight=self.preflight)
        if "error" in g_out:
            return self._plan_err({}, str(g_out["error"]))
        g_res = _sink.get("results") or {}
        _WHY_VERB = {"stage": "standing at", "move": "flying",
                     "map": "mapping around", "crops": "looking at",
                     "boost": "looking around", "facing": "facing"}
        for i, (q, why) in enumerate(names.items()):
            rows = [r for r in (g_res.get(f"r{i}") or [])
                    if r.get("fid")]
            rows = [r for r in rows
                    if r.get("name_match") is None
                    or float(r["name_match"]) >= _FUZZY_FLOOR]
            _what = q.lower() or "that"
            if not rows:
                return self._plan_err({}, (
                    f"nothing named \"{_what}\" is on "
                    f"this surface{self._closest_names(_what)}; say it "
                    "differently, say a fid off a served row, or "
                    + ("fly" if withheld("look_for") else "look_for + fly")
                    + " more first"))
            # every name keeps only its top name_match tie group
            _top = rows[0].get("name_match")
            ties[q] = [str(r["fid"]) for r in rows
                       if r.get("name_match") == _top] \
                or [str(rows[0]["fid"])]
            if why == "stage" and len(ties[q]) > 1:
                _f = ", ".join(ties[q][:4])
                return self._plan_err({}, (
                    f"a viewpoint is one point; "
                    f"\"{_what}\" has {len(ties[q])} equally good "
                    f"name matches ({_f}); say which one — a fid "
                    "works"))
            # A token no matched row carries gets a note, never a refusal.
            _dead = _unmatched_query_tokens(q.lower(), rows)
            if _dead:
                notes.append(
                    f"no row matching \"{_what}\" carries "
                    + ", ".join(repr(t) for t in _dead[:4])
                    + " -- those words did not narrow anything")
            # a covered name skips this note: the pass flies every top tie
            if why not in ("anchor", "cover") and len(rows) > 1:
                notes.append(
                    f"{len(rows)} rows match \"{_what}\"; "
                    f"{_WHY_VERB.get(why, 'using')} the best "
                    f"name match, {rows[0]['fid']} — say a fid to "
                    "choose precisely")
        return {"ties": ties}

    def _analyze(self, ops: list, rn_end: int, answer, page) -> dict:
        """Run one compiled program; ``answer`` names the ref whose envelope
        is served, ``page`` the ``(ref, start, the model's name for it)`` a
        more_of window pages."""
        if page:
            # Paging a handle with nothing left to page is refused here.
            _ref, _lo, _h = page
            _held = len(self.carry.read(_ref) or [])
            if _held and (_held <= _BROAD_DUMP_SOFT_ROWS or _lo >= _held):
                return self._plan_err({}, (
                    f"all {_held} row(s) of {_h} are already served -- there "
                    f"is no further block to serve. Narrow it "
                    f"({_h}.order(color=\"<colour>\"), "
                    f"{_h}.order(nearest=\"<fid>\"), "
                    f"{_h}.filter(within=\"<place>\"))"
                    + ("." if withheld("look") else f" or look({_h}).")))
        out = self._dispatch_analyze_scene(ops, answer_ref=str(answer or ""))
        if "error" in out:
            return {"error": out["error"]}
        self.plan_rn[0] = max(self.plan_rn[0], int(rn_end or 0))
        # The column this call ordered by, so _finish can say when the
        # ordering separated nothing.
        _sb = next((o for o in reversed(ops) if o["op"] == "score_by"), None)
        self._call.update(answer=answer, page=page,
                          rows=self.carry.rows.get(answer),
                          order=str(_sb["args"]["emphasis"][0]["key"])
                          if _sb else None)
        return {"result": out}

    def _stand(self, at: str, facing: str) -> dict:
        st = self._stage_apply(at, facing)
        if "error" in st:
            return self._plan_err({}, f"the stance did not bind: {st['error']}")
        return {"stance": st}

    def _move_to(self, tok: str) -> dict:
        tok, _stop_err = self._flown_stop(tok)
        if _stop_err:
            return self._plan_err({}, _stop_err)
        tok = tok.strip()
        if _is_handle_token(tok):
            mv = self._fly_moves([], [{"index": 0, "error": (
                f"{tok!r} is a result handle; move takes a literal fid "
                "off the rows, a flown stop id, or an [x, y] world point")}])
        else:
            mv = self._fly_moves([{"target_fid": tok}], [])
        _ok = [r for r in mv.get("flown") or [] if "error" not in r]
        if not _ok:
            # nothing flew: the failure is the result
            _msg = next(
                (str(r["error"]) for r in mv.get("flown") or []
                 if r.get("error")),
                None) or next(
                (str(e["error"]) for e in mv.get("rejected") or []
                 if e.get("error")),
                None) or str(mv.get("error") or "nothing flew")
            return self._plan_err({"flown": mv}, _msg)
        return {"flown": mv}

    def _cover(self, members: list, said_to: str) -> dict:
        """explore(): one pass of flights over a place or a set, whose
        members arrive as a name's top tie group (a list), a handle's rows
        or a token."""
        _raw_members: list = []
        for _t in members:
            if isinstance(_t, list):
                _raw_members.extend(_t)
                continue
            _t, _stop_err = self._flown_stop(_t)
            if _stop_err:
                return self._plan_err({}, _stop_err)
            if _is_handle_token(_t):
                _hr = [str(r["fid"]) for r in (self.carry.read(_t) or [])
                       if r.get("fid")]
                if not _hr:
                    return self._plan_err({}, (
                        f"{_t} holds 0 fid-bearing row(s) "
                        "here; there is nothing to cover."))
                _raw_members.extend(_hr)
                continue
            _raw_members.append(_t)
        _accepted = [{"target_fid": _t.strip()}
                     for _t in dict.fromkeys(_raw_members)]
        if not _accepted:
            return self._plan_err({}, "nothing to cover")
        _accepted, _cov_note, _subject, _segs = cover_items(
            _accepted, said_to, spec=self.mission_spec,
            spacing=self._cover_spacing, rows=self._ns_rows,
            covered=self._covered_fids, pose_xy=self.ctx.pose_xy)
        if not _accepted:
            return self._plan_err({}, _cov_note)
        mv = self._fly_moves(_accepted, [], cover=_subject)
        if "error" in mv:
            return self._plan_err({"explored": mv}, str(mv["error"]))
        if _segs > 1:
            mv["segments"] = _segs
        # A truncated pass reads as truncated.
        if mv["stops"] < len(_accepted):
            _cov_note = (f"the pass ended after "
                         f"{mv['stops']} of its stops.")
        if _cov_note:
            mv["note"] = (f"{_cov_note} {mv['note']}"
                          if mv.get("note") else _cov_note)
        # THE PASS SHOWS ITS CATCH: one crop per row it added, pass order.
        _h = self._call_minted()
        _rows = [r for r in (self.carry.rows.get(_h) or [])
                 if r.get("fid")] if _h else []
        _catch = _rows[:_CATCH_CROPS]
        if _catch:
            _cack = self._crops_of(
                [_crop_token_of(r, str(r["fid"]) in self._ns_rows)
                 for r in _catch],
                preamble=("The pass's catch, one crop per "
                          "row it added, in pass order: "))
            if not _cack.get("error"):
                mv["catch"] = {k: _cack[k] for k in ("delivered", "not_drawn")
                               if k in _cack}
                if len(_catch) < len(self.carry.rows.get(_h) or []):
                    mv["catch"]["shown"] = (
                        f"first {len(_catch)} of "
                        f"{len(self.carry.rows[_h])}" + (
                            "" if withheld("look") else
                            f" -- look({_h}[a:b]) pages the rest"))
                if withheld("images"):
                    mv["catch"]["note"] = _PICTURE_WITHHELD
        return {"explored": mv}

    def _crops(self, of, fid, marks) -> dict:
        _mk = [m for m in marks or [] if isinstance(m, str) and m.strip()]
        crop = (self._crops_set(str(of), marks=_mk or None) if of
                else self._crops_of(fid, marks=_mk or None))
        if crop.get("error") and not crop.get("delivered"):
            return self._plan_err({"crop": crop}, str(crop["error"]))
        return {"crop": crop}

    def _map(self, around: str) -> dict:
        # The map badges the set in scope.
        _h = self._those
        _mrows = [r for r in (self.carry.read(_h) or [])
                  if r.get("fid")] if _h else []
        _mfids = [str(r["fid"]) for r in _mrows[:_MAP_MAX_MARKS]]
        mp = self._map_of(around, mark_fids=_mfids)
        if _mfids:
            # The set under the model's own name for it.
            _name = self._mission_py.ref_name(_h)
            mp["badged_from"] = _name or "the set in scope"
            if len(_mrows) > len(_mfids):
                mp.setdefault("ref_notes", []).append(
                    f"{len(_mrows)} rows in {_name or 'the set in scope'}; "
                    f"this map badges the first {len(_mfids)}")
        if mp.get("error") and not mp.get("delivered"):
            return self._plan_err({"map": mp}, str(mp["error"]))
        return {"map": mp}

    def _watch(self, concepts: list) -> dict:
        w = self._watch_apply(list(concepts))
        if w.get("error"):
            return self._plan_err({"watching": w}, str(w["error"]))
        return {"watching": w}

    def _finish(self, ack: dict) -> dict:
        """Close a call nothing refused: the tie note, the grounding notes,
        the streak reset, the scope and paging writes."""
        # An ordering that separated nothing says how many rows tie.
        _ok_key = self._call.get("order")
        if _ok_key:
            _rows = self._call.get("rows") or []
            if len(_rows) >= 2:
                _v0 = _rows[0].get(_ok_key)
                if _v0 is not None:
                    _tied = [r for r in _rows if r.get(_ok_key) == _v0]
                    if len(_tied) >= 2:
                        _t = f"{_v0:g}" if isinstance(_v0, (int, float)) \
                            else str(_v0)
                        ack["note"] = (
                            (f"{ack['note']}\n" if ack.get("note") else "")
                            + f"{len(_tied)} of {len(_rows)} rows tie at "
                              f"{_t} on '{_ok_key}' -- this ordering does "
                              f"not separate them, so its rank 0 is "
                              f"arbitrary among the tie")
        if self._call.get("notes"):
            ack["note"] = (f"{ack['note']}\n" if ack.get("note")
                           else "") + "; ".join(self._call["notes"])
        self._err_sig, self._err_n = None, 0
        if self._call.get("page"):
            self._widened.update([self._call["page"][:2]])
        # The set in scope: the call's non-empty answer, or what its last
        # flight added.
        if self._call.get("answer") and self._call.get("rows"):
            self._those = self._call["answer"]
        if self._call_minted():
            self._those = self._call_minted()
        return ack

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
        self._code_names.update(re.findall(r"\br\d+\b", code))
        self._code_names.update(
            v.ref for v in self._mission_py.ns.values()
            if isinstance(v, _py_surface.Rows))
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
        """The mission's turn loop: append ``user_message``, then serve
        ``run`` until the model's plain reply or a harness ending; an empty
        turn, a block of code and a reply with no flight behind it are
        repaired before a repeat is accepted as the ending."""
        self._new_turn()
        if self.reasoner.compaction_infeasible:
            return self._mission_end("compaction_infeasible")
        self.reasoner.append_user(user_message)
        # Channel misses, consecutive zero-flight replies
        # (NO_FLIGHT_ABSTAIN_STREAK), transient serve errors.
        no_call_streak = no_flight_streak = transient_failures = 0
        for turn_idx in itertools.count():
            self.turn_idx = turn_idx
            if self.diag is not None:
                self.diag["turns_used"] = turn_idx + 1
            # The operator's re-tasks land at the top of a turn, never between
            # a turn's tool calls and their results.
            if self.retask is not None:
                while (operator_turn := self.retask()) is not None:
                    log.info("%s operator re-task at turn %d (%d chars)",
                             self.rid_tag, turn_idx, len(operator_turn))
                    self.reasoner.append_user(operator_turn)
            if turn_idx >= self.mission_spec.turn_cap:
                log.info("%s mission turn cap (%d) reached", self.rid_tag,
                         self.mission_spec.turn_cap)
                return self._mission_end("turn_cap", turns=turn_idx)
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
                if _is_oversize_rejection(exc):
                    log.warning(
                        "%s request does not fit the served window "
                        "-> mission ends (retry would re-send the same bytes): %s",
                        self.rid_tag, _summarise_exc(exc))
                    self._log("oversize_rejection", error=_summarise_exc(exc))
                    return self._mission_end("oversize_rejection", turns=turn_idx + 1)
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
                        return self._mission_end(
                            "request_refused" if _is_refused_request(exc)
                            else "transient_failures", turns=turn_idx + 1)
                    time.sleep(min(15.0, 0.5 + 2.0 * transient_failures))
                elif cancel.wait(min(30.0, 0.5 + 2.0 * transient_failures)):
                    raise CallCanceled("stopped while waiting for the reasoner")
                continue
            transient_failures = 0

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
                if no_call_streak == 0 and self.reasoner.flights == 0:
                    no_flight_streak += 1
                    self._log("no_flight_reply", turn_idx=turn_idx,
                              streak=no_flight_streak)
                    if no_flight_streak < NO_FLIGHT_ABSTAIN_STREAK:
                        log.info("%s reply with ZERO flights at turn %d "
                                 "(streak %d) -> repair", self.rid_tag, turn_idx,
                                 no_flight_streak)
                        self.reasoner.append_repair(NO_FLIGHT_REPLY_NUDGE)
                        continue
                    log.info("%s reply with ZERO flights at turn %d, streak "
                             "%d -- abstain", self.rid_tag, turn_idx,
                             no_flight_streak)
                log.info("%s mission ended by reply at turn %d (%d flights)",
                         self.rid_tag, turn_idx, self.turn_flights)
                return self._mission_end("reply", turns=turn_idx + 1)
            no_call_streak = 0
            no_flight_streak = 0

            # Dispatch each call in order, answering every call id.
            for tc in r["tool_calls"]:
                name = (tc.get("function") or {}).get("name") or ""
                cid = tc.get("id") or ""
                args = json.loads(tc["function"]["arguments"])
                if name == "run":
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
                        said=str(args.get("code") or "")[:400],
                        # already capped (python_surface.PRINT_CHAR_CAP)
                        printed=str(out.get("printed") or ""),
                        errored="error" in out,
                        # every image this block showed, by sha
                        frames=list(self._block_frames))
                    self._block_frames.clear()
                else:
                    # An unknown tool's id is answered, never executed.
                    self.reasoner.append_tool_result(cid, name or "unknown", {
                        "error": f"unknown tool {name!r}; available: run"})
            # Pictures ride a user turn after the turn's tool results.
            for _vtext, _vjpegs in self.pending_vision:
                self.reasoner.append_vision_user(_vtext, _vjpegs)
            self.pending_vision.clear()
