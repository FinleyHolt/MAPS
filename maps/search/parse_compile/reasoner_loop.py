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
import math
import os
import re
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from maps.llm.http import TRANSIENT_CLIENT_STATUS, CallCanceled, DegenerateCompletion
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
from maps.util.xy_token import format_xy_token, is_xy, parse_xy_token
from maps.search.analyze_scene.tools import ToolContext
from maps.search.withhold import withheld
from maps.search.analyze_scene.text_match import (
    FUZZY_THRESHOLD as _FUZZY_FLOOR,
    unmatched_query_tokens as _unmatched_query_tokens,
)
from maps.search.analyze_scene.geom_compute import normalize_color as _normalize_color
from maps.search.analyze_scene.geom_program import (
    _BROAD_DUMP_SOFT_ROWS,
    MODEL_UNSERVED_COLUMNS,
    ProgramError as _ProgramError,
    _is_color_column_key,
    compact_results as _compact_results,
    join_hints,
    run_program as _run_program,
)
from maps.search.parse_compile.reasoner_agent import Reasoner
from maps.search.parse_compile import python_surface as _py_surface

# Transient LLM-side errors: a retry, never the end of the reasoner (a
# non-2xx, a link fault, a collapsed completion, an undecodable body).
_RETRIABLE_LLM_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPStatusError,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    DegenerateCompletion,
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


def _program_dedup_key(program: dict) -> str | None:
    """Canonical identity of a SELF-CONTAINED program (each ref renamed to
    its emission index), or ``None`` when it reads a handle it does not
    define and is therefore never deduped."""
    ops = program.get("ops") or []
    if not ops:
        return None
    ren = {op["ref"]: f"r{i}" for i, op in enumerate(ops)}
    blob = json.dumps(ops, sort_keys=True, default=str)
    if any(t not in ren for t in re.findall(r"\br\d+\b", blob)):
        return None
    return re.sub(r"\br\d+\b", lambda m: ren[m.group(0)], blob)


# The dedup hit serves a one-line REFERENCE, never a fresh payload (a full
# re-render of an identical result is loop fuel). Every path that can mint a
# row bumps the surface version, so "nothing has entered the world" is true
# whenever it fires.
_DEDUP_NOTE = (
    "no new information since turn {turn}: nothing has entered the world "
    "since then -- no flight, no perceive -- so every program returns what "
    "it returned then, and this one is not re-rendered. Its handles stay "
    "live for later ops. Change the world (fly, perceive) or "
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


log = logging.getLogger(__name__)


# Tool-call residue: a prose turn carrying one is a malformed envelope,
# repaired, never an ending.
ENVELOPE_RESIDUE = ("<tool_call", "</tool_call", "<function")

# Consecutive zero-flight replies that end the mission: the first two draw
# NO_FLIGHT_REPLY_NUDGE, the third is an abstain. Anything the model
# executes between two replies resets the streak.
NO_FLIGHT_ABSTAIN_STREAK = 3


@dataclass
class MissionSpec:
    """The engine half of the mission contract.

    ``fly`` executes ONE move item (``{"target_fid": <fid or "[x, y]">}``)
    now and returns the flight report, or ``{"error": ...}``;
    ``declare_concepts`` arms the full watchlist and returns its receipt.
    """

    fly: "Callable[[dict], dict]"
    declare_concepts: "Callable[[list[dict]], dict]"
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


def _legend_line(entries: list[dict]) -> str:
    return " ; ".join(
        f"{e['digit']} = {e['label']} ({e['role']})" for e in entries)


def _window_caption(c: dict) -> str:
    """One rendered image's window: span, orientation and world extent."""
    # ORIENTATION, per image and derived, never assumed: a rotated image
    # states its true up (bearing + the frame note); only a genuinely
    # unrotated one says north-up. The header claims nothing.
    _fu = c.get("frame_up")
    if isinstance(_fu, (list, tuple)) and len(_fu) == 2:
        _fn = str(c.get("frame_note") or "").strip()
        _up = (f"up {round(up_bearing_deg(_fu)) % 360} deg"
               + (f" -- {_fn}" if _fn else ""))
    else:
        _up = "north-up"
    _cm = c.get("context_m")
    _span = (f" ({_cm:g} m window; {_up})"
             if isinstance(_cm, (int, float)) else f" ({_up})")
    # The numeric extent makes a pixel position back-projectable.
    _ww = c.get("world_window")
    if isinstance(_ww, (list, tuple)) and len(_ww) == 4:
        _span += (f" [x {_ww[0]:g}..{_ww[2]:g}, "
                  f"y {_ww[1]:g}..{_ww[3]:g}]")
    return _span


def _look_vision_text(crops: list[dict]) -> str:
    # The digit -> fid binding lives in this text legend; color is left to
    # the pixels, and each crop line states its delivered span.
    from maps.search.render.look_render import (  # noqa: PLC0415
        LOOK_MAX_BADGES,
    )
    # The caption names the subject: center_fid, else (a stop id on a flight
    # view, or absent) the legend's own center entry.
    def _subject(c):
        f = c.get("center_fid")
        if f and not str(f).startswith("stop_"):
            return str(f)
        for e in c.get("legend") or []:
            if e.get("role") == "center" and e.get("fid"):
                return str(e["fid"])
        return str(f) if f else "?"

    order = ", ".join(_subject(c) for c in crops)
    # GSD from the crop records themselves, never a literal.
    _gsds = {round(float(c["gsd_m_per_px"]), 2)
             for c in crops if c.get("gsd_m_per_px")}
    _gsd = (f"{min(_gsds):.2f} m/px" if len(_gsds) == 1
            else f"{min(_gsds):.2f}-{max(_gsds):.2f} m/px" if _gsds
            else "m/px unknown")
    # Resampling adds pixels, never detail: the source imagery's GSD bounds
    # what the caption may claim.
    _src = max((float(c["source_gsd_m"]) for c in crops
                if c.get("source_gsd_m")), default=0.0)
    _detail = (f"{_gsd}; detail limited by {_src:.2f} m source imagery"
               if _src and _src > min(_gsds or {_src}) else _gsd)
    base = (f"Overhead crops ({_detail}, nadir), one per fid "
            f"in order: [{order}]. White outline marks, numbered badges; "
            f"per-crop badge legend. Altitude does NOT change crop detail: "
            f"a crop is a digital crop of imagery already captured, so "
            f"flying lower buys none. What it buys is a NEW view square "
            f"somewhere else:")
    for k, c in enumerate(crops):
        entries = c.get("legend") or []
        if not entries:
            continue
        line = _legend_line(entries)
        _span = _window_caption(c)
        # The pixel grid, per image: with the window it converts what the
        # model sees into a world point.
        _px = c.get("side_px") or c.get("width_px")
        if _px:
            _span += f" ({int(_px)}x{int(_px)} px)"
        base += f"\n  crop {k + 1}{_span}: {line}"
        if c.get("subject_in_window") is False:
            _sxy = c.get("world_xy") or ["?", "?"]
            _sw = c.get("world_window") or ["?"] * 4
            base += (" ; WARNING: neither this fid's position "
                     f"[{_sxy[0]}, {_sxy[1]}] nor any part of its "
                     f"footprint is inside this crop's window "
                     f"[x {_sw[0]}..{_sw[2]}, y {_sw[1]}..{_sw[3]}] "
                     "(edge-clamped or off-tile) -- the subject is not "
                     "in this image; do not read color or arrangement "
                     "from it (the position and window bounds here say "
                     "where it sits relative to this crop).")
        elif c.get("subject_centroid_in_window") is False:
            _sxy = c.get("world_xy") or ["?", "?"]
            base += (" ; note: this fid's recorded position "
                     f"[{_sxy[0]}, {_sxy[1]}] is outside this crop's "
                     "window (edge-clamped at the tile boundary), but "
                     "part of its footprint IS in frame -- the outline/"
                     "circle marks the in-frame part.")
        elif c.get("subject_on_data") is False and c.get("source") == "run":
            _sxy = c.get("world_xy") or ["?", "?"]
            base += (" ; WARNING: this fid's position "
                     f"[{_sxy[0]}, {_sxy[1]}] is inside this crop's "
                     "window but this run has not imaged that ground yet "
                     "-- the black region is unobserved, not a dark "
                     "object; its color/shape cannot be read until a leg "
                     "flies over it (go_to it, then look).")
        elif c.get("subject_on_data") is False:
            _sxy = c.get("world_xy") or ["?", "?"]
            base += (" ; WARNING: this fid's position "
                     f"[{_sxy[0]}, {_sxy[1]}] is inside this crop's "
                     "window but past the map raster's DATA edge -- "
                     "the black region is off-raster no-data, not a "
                     "dark object; the raster holds no pixels for "
                     "this subject, so its color/shape cannot be "
                     "read from this image.")
        # A subject bigger than the frame: state its size, the ink is cut.
        _ext = c.get("subject_extent_m")
        if c.get("subject_outline_cut") and isinstance(_ext, (list, tuple)):
            base += (f" ; note: this fid's footprint is {_ext[0]:g} x "
                     f"{_ext[1]:g} m and runs past this window, so the "
                     "outline marks only where its edge crosses the frame "
                     "-- not a shape you can read whole here. Its size is "
                     "the fact; the ink is a fragment.")
        _vf = c.get("window_valid_frac")
        if (isinstance(_vf, (int, float)) and _vf < 0.99
                and c.get("subject_on_data") is not False):
            base += (f" ; note: {round((1 - _vf) * 100)}% of this "
                     + ("window is ground this run has not imaged yet "
                        "(black there = unobserved, not shadow)."
                        if c.get("source") == "run" else
                        "window lies past the map raster's data edge "
                        "(black there = no data, not shadow)."))
        if int(c.get("marks_dropped") or 0):
            base += (f" ; +{int(c['marks_dropped'])} requested mark(s) "
                     f"not drawn (badge cap {LOOK_MAX_BADGES}/crop -- "
                     f"request them in another crops call)")
        _mof = c.get("marks_out_of_view") or []
        if _mof:
            base += (f" ; requested mark(s) {', '.join(_mof)} lie "
                     f"outside this crop's frame and are NOT drawn "
                     f"(absent from the image, not unlabeled)")
        _muf = c.get("marks_unbadged_in_frame") or []
        if _muf:
            base += (f" ; requested mark(s) {', '.join(_muf)}: part of "
                     f"the footprint IS in frame but the centroid badge "
                     f"position falls outside it, so no badge is drawn "
                     f"(in the image, unlabeled)")
    return base


# A framed map's slack around what was asked about, and its tightest frame
# (below it the renderer's GSD floor binds and a tighter frame only removes
# context).
_MAP_FRAME_MARGIN = 1.6
_MAP_FRAME_MIN_M = 200.0


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
# 16-image budget, so the model's own looks keep room.
_CATCH_CROPS = 8
# What a look and a pass's catch say under the ``images`` withhold.
_PICTURE_WITHHELD_OF = ("the {}'s caption follows in the next message; "
                        "its picture is not served on this run")
_PICTURE_WITHHELD = _PICTURE_WITHHELD_OF.format("crop")


def _mission_sanitize_moves(raw: list[str]) -> tuple[list[dict], list[dict]]:
    """The move targets as ``{"target_fid": token}`` items, a result handle
    refused: ``(accepted, errors)``, each error carrying its index."""
    accepted: list[dict] = []
    errors: list[dict] = []
    for i, token in enumerate(raw):
        token = token.strip()
        if _is_handle_token(token):
            errors.append({"index": i, "error": (
                f"{token!r} is a result handle; move takes a literal fid "
                "off the rows, a flown stop id, or an [x, y] world point")})
            continue
        accepted.append({"target_fid": token})
    return accepted, errors


# Served on a preflight color-column request that attached to no row:
# before landing no row carries perceived color.
_PREFLIGHT_COLOR_NOTE = (
    "priors carry no perceived color before landing, so no color column "
    "attaches yet; color enriches per-stop after arrival."
)


def _color_words_of_op(op: dict) -> list[str]:
    """The palette words this op requests a color column for (a filter key,
    a score_by emphasis key)."""
    args = op["args"]
    words: list[str] = []
    for k in [args.get("key")] + [e["key"] for e in args.get("emphasis") or []]:
        if _is_color_column_key(k):
            c = _normalize_color(k[len("color_"):].strip().lower())
            if c and c not in words:
                words.append(c)
    return words


def _program_requests_color(program: dict) -> bool:
    return any(_color_words_of_op(op) for op in program["ops"])


def run_geom_program_tool(args: dict, *, namespace: dict | None = None,
                          raw_sink: dict | None = None,
                          seed_results: dict | None = None,
                          preflight: bool = False,
                          answer_ref: str = "") -> dict:
    """Run one op-program over the namespace's ``(cands, anchors)`` surface
    (and its standing viewpoint) and compact the results for serving.

    ``raw_sink["results"]`` receives the pre-compaction results. A refused
    program returns ``{"error": ...}``, never raises.
    """
    program = (args or {}).get("program")
    if not isinstance(program, dict) or not program.get("ops"):
        return {"error": "program must be an object with a non-empty 'ops' array"}
    ns = namespace or {}
    cands = ns.get("cands") or []
    anchors = ns.get("anchors") or []
    try:
        out = _run_program(cands, anchors, program,
                           seed_results=seed_results,
                           staged=ns.get("staged"),
                           standing=ns.get("standing"))
    except _ProgramError as exc:
        _err = str(exc)
        if raw_sink is not None:
            raw_sink["error_op"] = getattr(exc, "op", "")
        if preflight and _program_requests_color(program):
            # The color request errored before any envelope existed (e.g.
            # the emphasis absent-color guard) -- the phase fact rides the
            # error so "no color" reads as WHEN, not broken.
            _err = join_hints(_err, _PREFLIGHT_COLOR_NOTE)
        return {"error": _err}
    except Exception as exc:  # noqa: BLE001  — surface the failure to the agent, don't crash the round
        return {"error": f"program failed: {exc}"}
    _raw_results = out["results"]
    if raw_sink is not None:
        raw_sink["results"] = _raw_results
    # ``answer_ref`` names the one table a plan returns; the other ops' hints
    # ride it.
    out["results"] = _compact_results(_raw_results, answer_ref=answer_ref)
    if preflight:
        # A color-column request whose raw rows carry none of its columns
        # gets the phase note.
        for _op in (program.get("ops") or []):
            _words = _color_words_of_op(_op)
            if not _words:
                continue
            _rows = _raw_results.get(_op.get("ref"))
            _attached = isinstance(_rows, list) and any(
                isinstance(_it, dict) and any(
                    _it.get("color_" + _w) is not None for _w in _words)
                for _it in _rows)
            _env = out["results"].get(_op.get("ref"))
            if not _attached and isinstance(_env, dict):
                _env["hint"] = join_hints(_env.get("hint"),
                                          _PREFLIGHT_COLOR_NOTE)
    return out


def _surface_from_ctx(ctx) -> dict:
    """The ``{cands, anchors, cell_m}`` surface of a :class:`ToolContext`'s
    graph."""
    from maps.search.analyze_scene.geom_program import build_surface_from_graph

    cell_m = float(ctx.cell_size_m)
    if ctx.graph is None:
        return {"cands": [], "anchors": [], "cell_m": cell_m}
    cands, anchors = build_surface_from_graph(
        ctx.graph.nodes.items(), ctx.tile_origin_xy, cell_m
    )
    return {"cands": cands, "anchors": anchors, "cell_m": cell_m}


def _is_handle_token(token: str) -> bool:
    """True when ``token`` is a result handle, ``rN`` or ``rN[...]``."""
    return bool(re.fullmatch(r"r\d+(\[[^\]]*\])?", token))


def _crop_token_of(row: dict, live: bool) -> str:
    """A crops token for one carried row: its fid, except a derived
    ``end_N`` row off the surface crops its own ``[x, y]`` (an evicted fid
    stays on the skip path)."""
    fid = str(row.get("fid"))
    xy = row.get("xy")
    if not live and re.fullmatch(r"end_\d+", fid) and is_xy(xy):
        return format_xy_token(float(xy[0]), float(xy[1]))
    return fid


def _rounded_xy(token) -> list[float] | None:
    """An ``[x, y]`` token -> the point at centimetre precision, else None."""
    p = parse_xy_token(token)
    return [round(p[0], 2), round(p[1], 2)] if p else None


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

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|\n```\s*$")


def _code_reply(text: str) -> bool:
    """True when the whole reply is a block of the surface: it parses whole
    (prose narrating a call does not) and carries a library call (plain
    prose that happens to parse does not)."""
    t = _CODE_FENCE_RE.sub("", str(text or "").strip()).strip()
    return bool(t) and _py_surface.looks_like_code(t)


def run_mission(
    *,
    instruction: str,
    ctx: "ToolContext",
    mission_spec: MissionSpec,
    reasoner_url: str,
    endpoint_spec,
    api_key: str = "EMPTY",
    seed: int | None = None,
    sensor_description: str = "",
    cancel_event: "threading.Event | None" = None,
    look_renderer: "Callable[[list[dict]], list[dict]] | None" = None,
    obs_span_m: float | None = None,
    obs_minor_m: float | None = None,
    boost_queue: "Callable[[list, list], dict]",
    transcript: list[dict] | None = None,
    diag: dict | None = None,
    rid_tag: str = "m0",
    retask: "Callable[[], str | None] | None" = None,
    reasoning_effort: str | None = None,
) -> tuple[MissionReply, Reasoner]:
    """THE mission entry: build the Reasoner, append the instruction as the
    one user message and run the conversation to its end. Returns the reply
    and the Reasoner (for :func:`continue_mission`)."""
    reasoner = Reasoner(
        reasoner_url=reasoner_url,
        endpoint_spec=endpoint_spec,
        api_key=api_key,
        seed=seed,
        sensor_description=sensor_description,
        cancel_event=cancel_event,
        reasoning_effort=reasoning_effort,
    )
    reply = advance_reasoner_one_round(
        reasoner=reasoner,
        user_message=instruction.strip(),
        ctx=ctx,
        rid_tag=rid_tag,
        transcript=transcript,
        compute_namespace=_surface_from_ctx(ctx),
        diag=diag,
        look_renderer=look_renderer,
        obs_span_m=obs_span_m,
        obs_minor_m=obs_minor_m,
        boost_queue=boost_queue,
        preflight=True,
        mission_spec=mission_spec,
        retask=retask,
    )
    assert isinstance(reply, MissionReply)
    return reply, reasoner


def continue_mission(
    *,
    reasoner: Reasoner,
    instruction: str,
    ctx: "ToolContext",
    mission_spec: MissionSpec,
    look_renderer: "Callable[[list[dict]], list[dict]] | None" = None,
    obs_span_m: float | None = None,
    obs_minor_m: float | None = None,
    boost_queue: "Callable[[list, list], dict]",
    transcript: list[dict] | None = None,
    diag: dict | None = None,
    rid_tag: str = "m1",
    retask: "Callable[[], str | None] | None" = None,
) -> MissionReply:
    """Another operator turn on a mission that ended by reply (the field
    path: the reply is a report). The same Reasoner keeps every earlier
    turn, and the surface re-derives from ``ctx`` as it now stands."""
    reply = advance_reasoner_one_round(
        reasoner=reasoner,
        user_message=instruction.strip(),
        ctx=ctx,
        rid_tag=rid_tag,
        transcript=transcript,
        compute_namespace=_surface_from_ctx(ctx),
        diag=diag,
        look_renderer=look_renderer,
        obs_span_m=obs_span_m,
        obs_minor_m=obs_minor_m,
        boost_queue=boost_queue,
        preflight=False,
        mission_spec=mission_spec,
        retask=retask,
    )
    assert isinstance(reply, MissionReply)
    return reply


def advance_reasoner_one_round(
    *,
    reasoner: Reasoner,
    user_message: str,
    ctx: ToolContext,
    rid_tag: str = "h?",
    transcript: list[dict] | None = None,
    compute_namespace: dict | None = None,
    diag: dict | None = None,
    look_renderer: Callable[[list[dict]], list[dict]] | None = None,
    obs_span_m: float | None = None,
    obs_minor_m: float | None = None,
    boost_queue: Callable[[list, list], dict],
    preflight: bool = False,
    mission_spec: MissionSpec,
    retask: Callable[[], str | None] | None = None,
) -> MissionReply:
    """The mission's turn loop: append ``user_message``, then serve ``run``
    until the model's plain reply or a harness ending.

    An empty turn, a reply that is a block of code and a reply with zero
    flights behind it are repaired before a repeat is accepted as the
    ending. ``retask`` (the field console) yields operator turns drained at
    the top of each turn; ``diag["turns_used"]`` is stamped every turn.
    """
    # A thread the compactor declared infeasible stays so: no message grows
    # it.
    thread_infeasible = bool(getattr(reasoner, "compaction_infeasible", False))
    if not thread_infeasible:
        reasoner.append_user(user_message)

    # Each fid's latest result-enriched row, for the rejection census.
    row_latest: dict[str, dict] = {}
    # Successful self-contained programs by ``_program_dedup_key`` ->
    # ``(surface_version, turn, ops)``: a repeat under the same version gets
    # the dedup note. A flight, a boost or a stance bumps the version.
    _programs_seen: dict[str, tuple] = {}
    _surface_version = [0]

    no_call_streak = 0
    # Consecutive zero-flight replies (NO_FLIGHT_ABSTAIN_STREAK); the
    # no-call streak counts channel misses.
    no_flight_streak = 0

    # Surface rows by fid, rebuilt in place after every flight and boost.
    _ns_rows: dict[str, dict] = {}

    def _rebuild_ns_rows() -> None:
        # cands win on fid collision.
        _ns_rows.clear()
        for _key in ("cands", "anchors"):
            for _r in (compute_namespace or {}).get(_key) or []:
                if isinstance(_r, dict) and _r.get("fid"):
                    _ns_rows.setdefault(str(_r["fid"]), _r)

    _rebuild_ns_rows()

    # ONE served tool, `run`, with no decoder mask: the parser's refusals are
    # ordinary tool results. Pictures ride a user turn after the turn's tool
    # results.
    from maps.search.parse_compile.prompts import served_tools
    stop_tools = list(served_tools())
    pending_vision: list[tuple[str, list[bytes]]] = []
    _block_frames: list[str] = []
    _LOOK_MAX_CROPS = 16
    # How many of a set's rows a map badges: the crop renderer's cap.
    from maps.search.render.look_render import (  # noqa: PLC0415
        LOOK_MAX_BADGES as _MAP_MAX_MARKS,
    )

    # The staged viewpoints by vN handle; the latest is STANDING (its side
    # columns ride every find row, its facing orients what renders).
    staged_viewpoints: dict[str, dict] = {}
    standing_handle: list = [None]
    # The mission's flown stops, {stop_id: [x, y]}: each a sayable place.
    _plan_stops: dict[str, list] = {}

    def _resolve_stage_point(tok) -> list[float] | None:
        """A stage at/facing slot -> [x, y]: a live surface fid, a flown
        stop id, a staged handle, or the canonical '[x, y]' token; None
        otherwise."""
        if isinstance(tok, str):
            row = _ns_rows.get(tok.strip())
            if row is not None:
                return look_xy(row)
            sp = _plan_stops.get(tok.strip().lower())
            if sp is not None:
                return [round(sp[0], 2), round(sp[1], 2)]
            sv = staged_viewpoints.get(tok.strip().lower())
            if sv is not None:
                return [round(float(sv["from_xy"][0]), 2),
                        round(float(sv["from_xy"][1]), 2)]
            return _rounded_xy(tok)
        return None

    def _stage_apply(at_tok: str, facing_tok: str) -> dict:
        """stand(): mint a vN handle that becomes STANDING and bump the
        surface version (a program can answer differently under it)."""
        from maps.search.analyze_scene.geom_compute import COMPASS_UNIT  # noqa: PLC0415

        def _rejected(error: str) -> dict:
            return {"rejected": [{"index": 0, "error": error}]}

        at_xy = _resolve_stage_point(at_tok)
        if at_xy is None:
            return _rejected(f"at {at_tok!r} is not a live surface fid or an "
                             f"[x, y] world point")
        if facing_tok.strip().lower() in COMPASS_UNIT:
            facing_desc = facing_tok.strip().lower()
            up = list(COMPASS_UNIT[facing_desc])
        else:
            f_xy = _resolve_stage_point(facing_tok)
            if f_xy is None:
                return _rejected(
                    f"facing {facing_tok!r} is not a live surface "
                    f"fid, an [x, y] point, or one of "
                    f"{'|'.join(COMPASS_UNIT)}")
            dx = f_xy[0] - at_xy[0]
            dy = f_xy[1] - at_xy[1]
            n = (dx * dx + dy * dy) ** 0.5
            if n < 1e-6:
                return _rejected("at and facing coincide -- no look "
                                 "direction")
            up = [dx / n, dy / n]
            facing_desc = facing_tok.strip()
        h = f"v{len(staged_viewpoints) + 1}"
        brg = round(up_bearing_deg(up)) % 360
        staged_viewpoints[h] = {
            "handle": h,
            "from_xy": at_xy,
            "up_unit": [round(up[0], 4), round(up[1], 4)],
            "at": at_tok.strip(),
            "facing": facing_desc,
            "bearing_deg": brg,
        }
        standing_handle[0] = h
        if compute_namespace is not None:
            compute_namespace["staged"] = staged_viewpoints
            compute_namespace["standing"] = h
        _surface_version[0] += 1
        return {
            "staged": [{"handle": h, "at_xy": at_xy,
                        "facing_bearing_deg": brg}],
            "note": (f"standing viewpoint = {h}: 'left of {h}'/'right of "
                     f"{h}'/'ahead of {h}'/'behind {h}' ride every find "
                     f"row; {h} is a point ref ('distance to {h}'); crops "
                     f"and maps render its facing up."),
        }

    def _standing_frame() -> tuple[list | None, str]:
        """The STANDING viewpoint's image-up and the caption note stating it
        (never a silent rotation); ``(None, "")`` with no stance."""
        if standing_handle[0] is None:
            return None, ""
        _sv = staged_viewpoints[standing_handle[0]]
        return (list(_sv["up_unit"]),
                f"image-up = {_sv['handle']}'s facing "
                f"(bearing {_sv['bearing_deg']})")

    def _look_strs(v: object) -> list[str]:
        return [str(x) for x in v if isinstance(x, str) and x.strip()] \
            if isinstance(v, list) else []

    def _build_crop_requests(
            items: list) -> tuple[list[dict], list[str], list[str]]:
        """One look's items ``{fid, mark_fids?, target_fid?}`` -> per-crop
        render specs off ``_ns_rows``, capped at the server's 16 images:
        ``(requests, skipped_fids, ref_notes)``."""
        skipped: list[str] = []
        requests: list[dict] = []
        ref_notes: list[str] = []
        # PIXELS ARE EARNED BY FLYING: a crop centred on ground no flight's
        # inscribed circle (half the footprint's short side) has covered is
        # refused.
        uncovered: list[str] = []

        def _covered(_xy) -> bool:
            if not (isinstance(obs_minor_m, (int, float))
                    and obs_minor_m > 0):
                return True
            _r = float(obs_minor_m) / 2.0
            return any(math.dist(_xy, _p) <= _r
                       for _p in _plan_stops.values())
        for _i, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            if len(requests) >= _LOOK_MAX_CROPS:
                ref_notes.append(
                    f"only the first {_LOOK_MAX_CROPS} crops of this call "
                    f"rendered ({len(items) - _i} item(s) dropped); the "
                    f"server takes {_LOOK_MAX_CROPS} images per request. "
                    "Ask for the rest in a later view call.")
                break
            f = str(it.get("fid") or "").strip()
            if not f:
                continue
            target_fid = it.get("target_fid")
            target_fid = (str(target_fid).strip()
                          if isinstance(target_fid, str) else "")
            mark_fids = _look_strs(it.get("mark_fids"))
            annotate = bool(mark_fids or target_fid)
            marks: list[dict] = []
            if annotate:
                if target_fid:
                    mark_fids = mark_fids + [target_fid]
                # A handle is a legal mark (a street is its ways).
                _mfx: list[str] = []
                for _mf in mark_fids:
                    if re.fullmatch(r"r\d+", _mf):
                        _mfx.extend(
                            str(_r["fid"])
                            for _r in (_carry_read(_mf) or [])
                            if isinstance(_r, dict) and _r.get("fid"))
                    else:
                        _mfx.append(_mf)
                mark_fids = _mfx
                seen_m: set[str] = set()
                for mf in mark_fids:
                    if mf in seen_m:
                        continue
                    seen_m.add(mf)
                    row = _ns_rows.get(mf)
                    xy = look_xy(row)
                    if xy is None:
                        xy = _rounded_xy(mf)
                    if xy is None:
                        skipped.append(mf)
                        continue
                    marks.append({
                        "fid": mf,
                        "xy": xy,
                        "boundary": look_outline(row),
                        "label": look_label(
                            mf, row, is_target=(mf == target_fid)),
                        "is_target": bool(mf == target_fid),
                    })
            row = _ns_rows.get(f)
            xy = look_xy(row)
            if xy is None:
                xy = _rounded_xy(f)
            if xy is None:
                skipped.append(f)
                continue
            if not _covered(xy):
                uncovered.append(f)
                continue
            frame_up, frame_note = _standing_frame()
            requests.append({
                "center_fid": f,
                "center_xy": xy,
                "frame_up": frame_up,
                "frame_note": frame_note,
                "center_label": look_label(
                    f, row, is_target=(f == target_fid)),
                "center_boundary": look_outline(row),
                "annotate": annotate,
                "marks": marks,
                # The observation extent: the renderer's window ceiling.
                "obs_span_m": obs_span_m,
            })
        if uncovered:
            _u = ", ".join(uncovered[:6])
            _um = (f", +{len(uncovered) - 6} more"
                   if len(uncovered) > 6 else "")
            ref_notes.append(
                f"{len(uncovered)} crop(s) not rendered ({_u}{_um}): that "
                "ground has never been under the sensor. A look shows what "
                "some flight has covered -- explore the place so a pass "
                "covers it, then look.")
        return requests, skipped, ref_notes

    def _map_features_from_ctx() -> list[dict]:
        """The live graph as the map painter's feature dicts: label and paint
        kind, cells, and a road's centerline or a building's contour in world
        metres."""
        import numpy as np  # noqa: PLC0415
        from maps.perception.viz.osm_palette import (  # noqa: PLC0415
            osm_kind,
            osm_label,
        )
        nodes = getattr(ctx.graph, "nodes", None) or {}
        _ox, _oy = ctx.tile_origin_xy
        _cs = float(ctx.cell_size_m)

        def _grid_to_world(pts):
            # astype truncates toward zero, like int().
            a = np.asarray(pts, dtype=float).reshape(-1, 2).astype(np.int64)
            return np.stack([float(_ox) + (a[:, 1] + 0.5) * _cs,
                             float(_oy) + (a[:, 0] + 0.5) * _cs],
                            axis=1).tolist()

        feats: list[dict] = []
        for fid, node in nodes.copy().items():
            # The node's own cell array rides as is (no per-map list copy).
            cells = getattr(node, "cells", None)
            if cells is None or len(cells) == 0:
                c = getattr(node, "centroid", None)
                cells = ([[int(c[0]), int(c[1])]] if c is not None else [])
            prov = getattr(node, "provenance", "prebaked")
            tags = getattr(node, "tags", None) or {}
            if prov == "detected":
                label = str(getattr(node, "class_name", "") or "detection")
                kind = "detection"
            else:
                label = osm_label(
                    tags, str(getattr(node, "class_name", "") or "prior"))
                kind = osm_kind(tags)
            feat = {
                "fid": str(fid), "label": label, "kind": kind,
                "cells": cells,
                "provenance": ("detected" if prov == "detected"
                               else "prebaked"),
            }
            # The name rides beside the class label the palette keys off.
            _nm = tags.get("name") if isinstance(tags, dict) else None
            if _nm:
                feat["name"] = str(_nm)
            if kind == "road":
                cl = getattr(node, "centerline", None)
                if cl is not None and len(cl) >= 2:
                    feat["centerline"] = _grid_to_world(cl)
            elif kind == "building":
                wc = getattr(node, "world_contour", None)
                if wc is not None and len(wc) >= 3:
                    feat["contour"] = [[float(x), float(y)]
                                       for x, y in wc]
            feats.append(feat)
        return feats

    # The flatten is built once per surface version.
    _map_feats: list = [None, None]

    def _map_features_cached() -> list[dict]:
        if _map_feats[0] != _surface_version[0]:
            _map_feats[:] = [_surface_version[0], _map_features_from_ctx()]
        return _map_feats[1]

    def _map_spec(around: str, mark_fids: list) -> tuple[dict | None, list[str]]:
        """One show_map's render spec, centred on ``around``, marks resolved
        off ``_ns_rows`` exactly like crops: ``(spec or None, skipped_fids)``."""
        cxy = _resolve_stage_point(around)
        if cxy is None:
            return None, [str(around)]
        skipped: list[str] = []
        center_fid = (str(around).strip()
                      if isinstance(around, str)
                      and str(around).strip() in _ns_rows else "")
        center_label = (look_label(center_fid, _ns_rows.get(center_fid),
                                   is_target=False)
                        if center_fid else f"[{cxy[0]}, {cxy[1]}]")
        marks: list[dict] = []
        for mf in dict.fromkeys(_look_strs(mark_fids)):
            row = _ns_rows.get(mf)
            xy = look_xy(row)
            if xy is None:
                skipped.append(mf)
                continue
            marks.append({"fid": mf, "xy": xy,
                          "boundary": look_outline(row),
                          "label": look_label(mf, row, is_target=False),
                          "is_target": False})
        # The frame follows the request: the marks' and centre's extent
        # derives the zoom; a bare point gets the floor.
        _cb = look_outline(_ns_rows.get(center_fid)) if center_fid else None
        _fpts: list = [m["xy"] for m in marks if m.get("xy")]
        for _src in [m.get("boundary") for m in marks] + [_cb]:
            for _b in (_src or []):
                if isinstance(_b, (list, tuple)) and len(_b) >= 2:
                    _fpts.append([float(_b[0]), float(_b[1])])
        span = _MAP_FRAME_MIN_M
        if _fpts:
            _fx = [float(p[0]) for p in _fpts]
            _fy = [float(p[1]) for p in _fpts]
            cxy = [round((min(_fx) + max(_fx)) / 2.0, 2),
                   round((min(_fy) + max(_fy)) / 2.0, 2)]
            span = max((max(_fx) - min(_fx)) * _MAP_FRAME_MARGIN,
                       (max(_fy) - min(_fy)) * _MAP_FRAME_MARGIN,
                       _MAP_FRAME_MIN_M)
        spec: dict = {
            "center_xy": cxy,
            "context_m": round(float(span), 2),
            "center_fid": center_fid,
            "center_label": center_label,
            "center_boundary": _cb,
            "marks": marks,
        }
        frame_up, frame_note = _standing_frame()
        if frame_up is not None:
            spec["frame_up"], spec["frame_note"] = frame_up, frame_note
        return spec, skipped

    def _map_vision_text(maps: list[dict]) -> str:
        """The map message's text: what the picture is (world state, not
        imagery), and each map's window, badge legend and class colors."""
        order = ", ".join(str(m.get("center_fid") or "map") for m in maps)
        base = (f"World-state maps -- the survey plus everything detected "
                f"so far, not a photograph: light-gray lines are roads, "
                f"tan shapes are surveyed buildings, muted regions are "
                f"landuse by class, dark is nothing recorded. Detections "
                f"are the bright class-colored shapes, each carrying a "
                f"small dN chip -- dN is detection fid detected_N, so any "
                f"one of them can be named. Surveyed buildings big enough "
                f"at this zoom carry their own pNNNN chip: pNNNN is "
                f"prior fid prior_NNNN (narrow the window to chip more of "
                f"them). A detection lying on a surveyed footprint shows "
                f"tan hatching through its fill, and that survey's "
                f"pNNNN chip sits stacked under the dN chip -- "
                f"co-location drawn, identity still the graph's call. "
                f"The survey part does NOT have "
                f"the shape of a cul-de-sac bulb, a driveway, a pool, a "
                f"car, or anything built since the survey -- no zoom "
                f"shows those; what has been DETECTED is drawn where it "
                f"was found. To see what is actually there, "
                f"{'fly: ' if withheld('look') else 'fly and look: '}"
                f"the camera is the live check on this. One per item in "
                f"order: [{order}]. Numbered badges; per-map legend:")
        for k, m in enumerate(maps):
            entries = m.get("legend") or []
            line = _legend_line(entries)
            _span = _window_caption(m)
            base += f"\n  map {k + 1}{_span}: {line}" if line else \
                f"\n  map {k + 1}{_span}:"
            _cls = m.get("classes") or []
            if _cls:
                base += (" ; classes: " + ", ".join(
                    f"{c['label']}={c['color']}" for c in _cls))
            elif m.get("features_painted") == 0:
                base += (" ; NOTE: 0 recorded features in this window -- "
                         "the dark image is an empty window, not a broken "
                         "render; re-center the map where rows are.")
            _pu = int(m.get("priors_unchipped") or 0)
            if _pu:
                base += (f" ; {int(m.get('priors_chipped') or 0)} "
                         f"buildings carry fid chips, {_pu} more in frame "
                         f"are unchipped at this zoom -- narrow the window "
                         f"to id them")
            _mof = m.get("marks_out_of_view") or []
            if _mof:
                base += (f" ; requested mark(s) {', '.join(_mof)} lie "
                         f"outside this map's window and are NOT drawn")
        return base

    # The escalating loop breaker's counts (see _loop_broken_error), and the
    # dedup path's own consecutive streak.
    _err_class_counts: dict[str, int] = {}
    _err_streak: dict = {"sig": None, "n": 0}
    _dedup_streak: dict = {"key": None, "n": 0}

    def _loop_breaker_facts(_ns, _n: int) -> str:
        def _cap(fids: list[str]) -> str:
            return f"{fids[:8]}" + (
                f" (+{len(fids) - 8} more)" if len(fids) > 8 else "")
        c_n, a_n, cols = _surface_census_counts(_ns)
        parts = [
            f" This error class has now occurred {_n}x.",
            f" Live surface: {_live_counts(c_n, a_n, ';')}.",
        ]
        if cols:
            parts.append(
                f" Columns computed on rows so far: {_cap(cols)}.")
        return "".join(parts)

    def _err_class_sig(err: str) -> str:
        # Same-class signature: quoted tokens and numbers stripped.
        s = re.sub(r"'[^']*'", "'_'", str(err))
        s = re.sub(r"\d+", "_", s)
        return s[:100]

    def _surface_census_counts(_ns) -> tuple[int, int, list[str]]:
        """Live counts: detections, priors, computed columns."""
        c_n = sum(1 for r in (_ns.get("cands") or [])
                  if isinstance(r, dict) and r.get("fid"))
        a_n = sum(1 for r in (_ns.get("anchors") or [])
                  if isinstance(r, dict) and r.get("fid"))
        cols = sorted({k for _row in row_latest.values()
                       if isinstance(_row, dict)
                       for k, _v in _row.items()
                       if k not in MODEL_UNSERVED_COLUMNS
                       and _v not in (None, "")})
        return c_n, a_n, cols

    def _repeated_rejection_state(err: str, _n: int, _consec: int) -> str:
        """The 3rd-consecutive-rejection fact, served instead of the remedy
        and census again: the counts, the rejection's first clause and the
        live state."""
        _first = str(err).split(". ", 1)[0].strip().rstrip(".")
        c_n, a_n, cols = _surface_census_counts(compute_namespace or {})
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

    def _loop_broken_error(err: str, sig_of: str | None = None) -> str:
        """A repeated same-class error, escalating: verbatim first, then with
        the live census, then (3rd consecutive on) the repetition fact
        alone. A shape error gets only its count. A successful call resets
        the consecutive streak, never the class count."""
        _sig = _err_class_sig(err if sig_of is None else sig_of)
        _n = _err_class_counts[_sig] = _err_class_counts.get(_sig, 0) + 1
        if _err_streak["sig"] == _sig:
            _err_streak["n"] += 1
        else:
            _err_streak["sig"] = _sig
            _err_streak["n"] = 1
        _consec = _err_streak["n"]
        if _n < 2:
            return err
        _SHAPE_MARKERS = ("unknown ref", "is a result handle",
                          "syntax error")
        if any(m in err for m in _SHAPE_MARKERS):
            return err + f" This error class has now occurred {_n}x."
        if _consec >= 3:
            return _repeated_rejection_state(err, _n, _consec)
        return err + _loop_breaker_facts(compute_namespace or {}, _n)

    # Raw results of every successful program, by ref.
    carry_results: dict = {}

    # A HANDLE IS A QUERY, NOT A SNAPSHOT: after the surface moves, a read
    # of a handle re-runs the ops that produced it (and those they read)
    # over the current surface. A replay that fails keeps the old rows.
    _op_history: list = []           # every op, in execution order
    _op_reads: list = []             # per op: the handle bases its args name
    _producers: dict = {}            # ref -> indices into _op_history
    # The surface version the carry is armed for: the first read under a new
    # version claims it, and every read after re-derives what it reads.
    _carry_version = [0]
    _fresh: set = set()              # refs re-derived, or written, under it
    # A flight-minted handle records an event (the rows one flight added)
    # and never re-derives.
    _event_handles: set = set()

    def _refs_read(op: dict) -> set:
        """The handles an op's args read (``rN`` or ``rN[a:b]``)."""
        return set(re.findall(r'"(r\d+)(?:\[[^"\]]*\])?"',
                              json.dumps(op["args"], default=str))) - {op["ref"]}

    def _program_reads(program: dict) -> set:
        """The carry handles a program reads and does not define."""
        ops = program["ops"]
        return set().union(*(_refs_read(o) for o in ops)) \
            - {o["ref"] for o in ops}

    def _ensure_fresh(refs) -> None:
        """Re-derive ``refs`` and every op they read against the current
        surface, once per surface version."""
        if _carry_version[0] != _surface_version[0]:
            _carry_version[0] = _surface_version[0]
            _fresh.clear()
            _fresh.update(_event_handles)
        need: set = set()
        todo = [r for r in refs if r not in _fresh]
        while todo:
            r = todo.pop()
            if r in need or r in _fresh:
                continue
            idxs = _producers.get(r)
            if not idxs:
                continue    # an event handle, an alias, an unknown: nothing to re-derive
            need.add(r)
            for i in idxs:
                todo.extend(_op_reads[i])
        if not need:
            return
        try:
            _sink: dict = {}
            # Seed everything else (a flight-minted handle has no op).
            _out = run_geom_program_tool(
                {"program": {"ops": [
                    op for op in _op_history if op["ref"] in need]}},
                namespace=compute_namespace or {}, raw_sink=_sink,
                seed_results={h: v for h, v in carry_results.items()
                              if h not in need})
            if isinstance(_out, dict) and _out.get("error"):
                # A refused replay keeps the last rows, unclaimed.
                log.debug("carry refresh refused; handles keep their last "
                          "rows: %s", str(_out["error"])[:200])
                return
            carry_results.update(
                {k: v for k, v in (_sink.get("results") or {}).items()
                 if k in need})
            _fresh.update(need)
        except Exception:  # noqa: BLE001 — a failed replay keeps the old rows
            log.debug("carry refresh failed; handles keep their last rows",
                      exc_info=True)

    def _carry_read(ref):
        """A handle's rows, re-derived first when the version is armed."""
        if _carry_version[0] == _surface_version[0]:
            _ensure_fresh([ref])
        return carry_results.get(ref)

    def _alias_deduped_refs(prev_ops, cur_ops) -> None:
        """Point this emission's refs at the handles the first run stored,
        op by op; a label already live is never overwritten."""
        for prev, cur in zip(prev_ops, cur_ops):
            if cur["ref"] not in carry_results \
                    and prev["ref"] in carry_results:
                carry_results[cur["ref"]] = carry_results[prev["ref"]]

    def _dispatch_analyze_scene(a, *, answer_ref=""):
        _key = _program_dedup_key(a["program"])
        _prev = _programs_seen.get(_key) if _key is not None else None
        if _prev is not None and _prev[0] == _surface_version[0]:
            # A one-line reference, never a payload; the emission's own
            # labels alias the stored handles so nothing downstream can tell
            # it from a re-run.
            _alias_deduped_refs(_prev[2], a["program"]["ops"])
            if _dedup_streak["key"] == _key:
                _dedup_streak["n"] += 1
            else:
                _dedup_streak["key"], _dedup_streak["n"] = _key, 1
            if _dedup_streak["n"] >= 2:
                _c_n, _a_n, _dcols = _surface_census_counts(
                    compute_namespace or {})
                _cols_s = ("; columns live: " + ", ".join(_dcols[:12])
                           if _dcols else "")
                return {"note": _DEDUP_REPEAT_NOTE.format(
                    n=_ordinal(_dedup_streak["n"]),
                    counts=_live_counts(_c_n, _a_n), cols=_cols_s)}
            return {"note": _DEDUP_NOTE.format(turn=_prev[1])}
        _raw_sink: dict = {}
        _ensure_fresh(_program_reads(a["program"]))
        out = run_geom_program_tool(a, namespace=compute_namespace or {},
                                    raw_sink=_raw_sink,
                                    seed_results=carry_results,
                                    preflight=preflight,
                                    answer_ref=answer_ref)
        if isinstance(out, dict) and "error" in out:
            # The failing op keeps one op's errors their own class.
            _err = str(out.get("error"))
            out = {**out, "error": _loop_broken_error(
                _err, sig_of=f"{_err}|{_raw_sink.get('error_op', '')}")}
        if isinstance(out, dict) and "error" not in out:
            _err_streak["sig"] = None
            _err_streak["n"] = 0
            _dedup_streak["key"] = None
            _dedup_streak["n"] = 0
            if _key is not None:
                _programs_seen[_key] = (_surface_version[0], turn_idx,
                                        a["program"]["ops"])
            carry_results.update(_raw_sink.get("results") or {})
            _fresh.update(_raw_sink.get("results") or {})
            for _op in a["program"]["ops"]:
                _producers.setdefault(_op["ref"], []).append(len(_op_history))
                _op_history.append(_op)
                _op_reads.append(_refs_read(_op))
            try:
                for _v in (_raw_sink.get("results") or {}).values():
                    _rows = _v if isinstance(_v, list) else (
                        [_v] if isinstance(_v, dict) else [])
                    for _row in _rows:
                        if isinstance(_row, dict) and _row.get("fid"):
                            row_latest[str(_row["fid"])] = _row
            except Exception:  # noqa: BLE001 — evidence plumbing never perturbs the round
                pass
        return out

    transient_failures = 0
    # The bench's budget; a live mission (a reasoner with a cancel_event)
    # waits for the arm until it answers or the operator stops.
    MAX_ROUND_TRANSIENT = 6

    mission_flights = 0
    mission_sam: list[dict] = []

    # The set in scope (the last non-empty result, or what the last flight
    # added), the paging ledger, and the mission-global rN counter.
    _those: list = [None]
    _widened: dict = {}
    plan_rn: list = [0]
    # Every accepted block, in order (counted on each result).
    _episode_script: list = []
    # Every rN in play (bound handles and those the running block spells):
    # the flight mint skips them.
    _code_names: set = set()
    # The handles flights minted during the running block, for the code
    # scope to bind.
    _code_minted: list = []
    # The fids explore passes have covered this mission.
    _covered_fids: set = set()

    def _watching_trailer() -> list:
        """The armed watchlist with each class's live detection count."""
        counts: dict[str, int] = {}
        for _r in (compute_namespace or {}).get("cands") or []:
            if isinstance(_r, dict) \
                    and str(_r.get("fid") or "").startswith("detected_"):
                _c = str(_r.get("class") or "?")
                counts[_c] = counts.get(_c, 0) + 1
        return [f"{c['class']} ({counts.get(c['class'], 0)} detected)"
                for c in mission_sam]

    def _mint_rows(_new_fids: list, _what: str):
        """The rows a flight or a pass added -> ``(rN handle, the note
        naming it)``; None (and no handle) when nothing was added."""
        if not _new_fids:
            return None
        while f"r{plan_rn[0]}" in _code_names:
            plan_rn[0] += 1
        _h = f"r{plan_rn[0]}"
        plan_rn[0] += 1
        _code_names.add(_h)
        _code_minted.append(_h)
        carry_results[_h] = [_ns_rows[_f] for _f in _new_fids]
        _event_handles.add(_h)
        _fresh.add(_h)
        _those[0] = _h
        return _h, (f"{_h} — the {len(_new_fids)} rows this {_what} added; "
                    f"the handle is bound, so {_h}, {_h}[0] and "
                    f"{_h}.order(...) all reach them")

    def _fly_moves(_accepted: list, _item_errors: list,
                   cover: str | None = None) -> dict:
        """Fly each entry in order; after each flight the surface re-derives
        and the rows it added mint an rN handle.

        ``cover`` (an explore) makes the loop ONE pass: no per-stop view,
        one handle over everything the pass folded, one ack.
        """
        nonlocal mission_flights, preflight
        _reports: list[dict] = []
        _cover_before = set(_ns_rows) if cover is not None else None
        _clock_cut = False

        def _stop_xy(_it):
            _t = str(_it.get("target_fid") or "")
            return parse_xy_token(_t) or look_xy(_ns_rows.get(_t) or {})

        _queue = list(_accepted)
        while _queue:
            _pick = 0
            if cover is not None and len(_queue) > 1 and mission_spec.held_now is not None:
                # A stop the recording already holds flies first.
                try:
                    _held = mission_spec.held_now(
                        [(str(_it.get("target_fid") or ""), _stop_xy(_it)) for _it in _queue])
                except Exception as _exc:  # noqa: BLE001 — the order is a convenience, never a fault
                    log.warning("%s pass: held_now failed: %s", rid_tag, _summarise_exc(_exc))
                    _held = []
                _pick = next((_i for _i, _h in enumerate(_held) if _h), 0)
                if _pick:
                    log.info("%s pass: %s taken first, the recording holds it",
                             rid_tag, _queue[_pick].get("target_fid"))
            _item = _queue.pop(_pick)
            if cover is not None:
                # The rest of the pass rides every stop, for the crew.
                _item["pass_ahead"] = [[float(_xy[0]), float(_xy[1])]
                                       for _xy in map(_stop_xy, _queue) if _xy]
            if mission_flights >= mission_spec.flight_budget:
                log.warning("%s flight backstop hit (%d flights)",
                            rid_tag, mission_spec.flight_budget)
                _item_errors.append({"target_fid": _item["target_fid"],
                                     "error": "not flown."})
                continue
            if (mission_spec.deadline_epoch is not None
                    and time.time() >= mission_spec.deadline_epoch):
                # The wall-clock backstop stops the flying only; the model
                # keeps its turns and its own commit.
                # ponytail: checked between stops, so one slow stop still
                # overruns by its own duration.
                if not _clock_cut:
                    log.warning(
                        "%s wall-clock backstop hit (%d flights, %.0f s past "
                        "the deadline)", rid_tag, mission_flights,
                        time.time() - mission_spec.deadline_epoch)
                _clock_cut = True
                _item_errors.append({"target_fid": _item["target_fid"],
                                     "error": "not flown."})
                continue
            _snap_from = None
            _pt = parse_xy_token(_item["target_fid"])
            if _pt is not None:
                # Snap a point within 1 m of a row to that row's fid.
                _best, _bd = None, 1.0
                for _bf, _brow in _ns_rows.items():
                    _bxy = look_xy(_brow)
                    if _bxy is None:
                        continue
                    _d = ((_bxy[0] - _pt[0]) ** 2
                          + (_bxy[1] - _pt[1]) ** 2) ** 0.5
                    if _d < _bd:
                        _best, _bd = _bf, _d
                if _best is not None:
                    _snap_from = _item["target_fid"]
                    _item["target_fid"] = _best
            _fids_before = set(_ns_rows)
            _report = mission_spec.fly(dict(_item))
            if not isinstance(_report, dict):
                _report = {"error": str(_report)}
            if _snap_from:
                _report.setdefault(
                    "note",
                    f"{_snap_from} is {_item['target_fid']}"
                    "'s own position -- flown as that fid.")
            if "error" not in _report:
                if (not _report.get("new_detections") and not mission_sam
                        and not withheld("look_for")):
                    _wl_note = ('the watchlist is empty -- nothing is '
                                'segmented on any flight; call '
                                'look_for("<class>") to see detections')
                    _report["note"] = (f"{_report['note']} {_wl_note}"
                                       if _report.get("note")
                                       else _wl_note)
                # THE FLIGHT SHOWS WHAT IT SAW: the target's own look,
                # centred on the row it asked for (a stop flown again centres
                # on the stop's point). A pass renders no per-stop view.
                if _report.get("pose_xy") and cover is None:
                    _tfid = str(_item.get("target_fid") or "")
                    _trow = _ns_rows.get(_tfid)
                    _tout = look_outline(_trow)
                    _txy = look_xy(_trow)
                    try:
                        _fv = look_renderer([{
                            "center_fid": _tfid if _txy else "",
                            "center_xy": (list(_txy) if _txy
                                          else list(_report["pose_xy"])),
                            "center_label": (
                                look_label(_tfid, _trow, is_target=True)
                                if _txy else (_tfid or "the stop")),
                            "center_boundary": _tout if _txy else None,
                            "annotate": False,
                            "marks": [],
                        }])
                    except Exception as _exc:  # noqa: BLE001 — never crash a flight
                        log.warning("%s flight view render failed: %s",
                                    rid_tag, _summarise_exc(_exc))
                        _fv = []
                    _fj = [c["jpeg"] for c in (_fv or [])
                           if isinstance(c, dict) and c.get("jpeg")]
                    if _fj:
                        _fv_text = _look_vision_text(_fv)
                        _shown = not withheld("images")
                        _fv_note = ("the view follows." if _shown else
                                    "the view's caption follows.")
                        if _shown:
                            _deliver_frames(_fj, _block_frames)
                        pending_vision.append((_fv_text, _fj if _shown else []))
                        _report["note"] = (
                            f"{_report['note']} {_fv_note}"
                            if _report.get("note") else _fv_note)
                mission_flights += 1
                _sid = _report.get("stop_id")
                _pxy = _report.get("pose_xy")
                if (_sid and isinstance(_pxy, (list, tuple))
                        and len(_pxy) == 2):
                    _plan_stops[str(_sid).strip().lower()] = [
                        float(_pxy[0]), float(_pxy[1])]
                # The world stepped: refresh both surface views and bump
                # the version.
                (compute_namespace or {}).update(_surface_from_ctx(ctx))
                _rebuild_ns_rows()
                _surface_version[0] += 1
                # The flight's new rows (the surface diff) mint a handle; a
                # pass mints once, around the whole of itself.
                if cover is None:
                    _m = _mint_rows([_f for _f in _ns_rows
                                     if _f not in _fids_before], "flight")
                    if _m is not None:
                        _report["new_rows"] = _m[1]
                preflight = False
            _reports.append(_report)
        if cover is not None:
            return _cover_ack(cover, _reports, _cover_before, _item_errors)
        _mv_ack: dict = {"flown": [_served_flight(r) for r in _reports]}
        if _item_errors:
            _mv_ack["rejected"] = _item_errors
        if not _accepted and not _item_errors:
            _mv_ack = {"error": "nothing to fly"}
        return _mv_ack

    def _cover_ack(cover: str, _reports: list, _before: set,
                   _item_errors: list) -> dict:
        """One explore's flight reports -> the one ack of the pass. The ack
        states that the standing answer is the pass's last stop."""
        _ok = [r for r in _reports
               if isinstance(r, dict) and "error" not in r]
        _stops = [r for r in _reports
                  if isinstance(r, dict) and r.get("pose_xy")]
        _ack: dict = {"explored": cover, "stops": len(_stops)}
        if _stops:
            _ack["answer_if_you_reply_now"] = (
                f"the last stop of the pass over {cover}; go_to() the row "
                f"you mean before you reply")
        _ack["new_detections"] = sum(int(r.get("new_detections") or 0)
                                     for r in _ok)
        _m = _mint_rows([_f for _f in _ns_rows if _f not in _before], "pass")
        if _m is not None:
            _ack["new_rows"] = _m[1]
        _errs = [str(r["error"]) for r in _reports
                 if isinstance(r, dict) and r.get("error")]
        if _errs:
            _ack["not_flown"] = _errs[:4]
        if _item_errors:
            _ack["rejected"] = _item_errors
        if not _ok:
            return {"error": (_errs[0] if _errs
                              else f"nothing of {cover} could be flown")}
        return _ack

    # HALF THE OBSERVATION'S LONG SIDE: a pass's stop spacing. Half of
    # ``obs_span_m`` (the footprint's long side) sits under the short side
    # for any sensor below 2:1, so a ONE-DIMENSIONAL walk covers a shape that
    # is not one: consecutive stops overlap on the minor axis too.
    _cover_spacing = (float(obs_span_m) / 2.0
                      if isinstance(obs_span_m, (int, float))
                      and obs_span_m > 0 else None)

    def _coverage_stops(_tok: str):
        """The engine's stops over one fid's footprint; None for a point,
        a fid with no cells, or a run without an observation extent."""
        if (parse_xy_token(_tok) is None and _cover_spacing is not None
                and mission_spec.coverage_stops is not None):
            return mission_spec.coverage_stops(_tok, _cover_spacing)
        return None

    def _uncovered_segments(_nm: str) -> str:
        """`` '<name>' has N more segment(s) no pass has covered (...).``,
        or "" when every segment bearing the name is covered."""
        _sibs = sorted(f for f, r in _ns_rows.items()
                       if isinstance(r, dict) and r.get("name") == _nm
                       and f not in _covered_fids)
        if not _sibs:
            return ""
        _head = ", ".join(_sibs[:6])
        _more = f", +{len(_sibs) - 6} more" if len(_sibs) > 6 else ""
        return (f" {_nm!r} has {len(_sibs)} more segment(s) no pass has "
                f"covered ({_head}{_more}).")

    def _cover_items(_accepted_items: list, _said=None):
        """One explore -> ``(move items, note, subject, segments)``: the stops
        over the subject's footprint in shape order and nothing after them,
        and a note saying how much of the shape they reach. A set is one
        pass over the union (``_cover_set_items``)."""
        if len(_accepted_items) > 1:
            return _cover_set_items(_accepted_items, _said)
        _item = _accepted_items[0]
        _tok = str(_item.get("target_fid") or "")
        _span = _cover_spacing
        _stops = _coverage_stops(_tok)
        _note = ""
        if _stops:
            _walk = sum(math.dist(a, b) for a, b in zip(_stops, _stops[1:]))
            _reached = min(1.0, len(_stops) * _span / _walk) if _walk else 1.0
            if len(_stops) == 1:
                _note = (f"{_tok} fits inside one observation; covered it "
                         "from the middle in one stop.")
            elif _reached >= 1.0:
                _note = (f"covered {_walk:.0f} m of {_tok} in "
                         f"{len(_stops)} stops.")
            else:
                _note = (f"covered {_reached:.0%} of {_tok}'s "
                         f"{_walk:.0f} m in {len(_stops)} stops.")
            _items = [{"target_fid": format_xy_token(float(p[0]),
                                                     float(p[1]))}
                      for p in _stops]
        else:
            # A point, a fid with no cells, or no observation extent: the
            # pass is the one stop it is.
            _note = (f"{_tok} has no footprint to cover, so this pass is "
                     "the one stop it is.")
            _items = [dict(_item)]
        # A named feature is often many segments: the note names those no
        # pass has covered.
        _covered_fids.add(str(_tok))
        _nm = (_ns_rows.get(str(_tok)) or {}).get("name")
        if _nm:
            _note += _uncovered_segments(_nm)
        return _items, _note, _tok, 1

    def _cover_set_items(_accepted_items: list, _said):
        """The multi-member half of ``_cover_items``: each member's own walk,
        chained greedily by nearest endpoint from the live pose (a member
        reversed when its far end is nearer), dropping a stop within half a
        spacing of one already kept."""
        # ponytail: O(n*m) greedy chaining, not a TSP; order the members
        # optimally if a pass's flown length ever matters.
        _span = _cover_spacing
        _members: list[dict] = []
        for _it in _accepted_items:
            _tok = str(_it.get("target_fid") or "")
            _pt = parse_xy_token(_tok)
            _stops = _coverage_stops(_tok)
            if _stops:
                _pts = [[float(p[0]), float(p[1])] for p in _stops]
                _members.append({
                    "tok": _tok, "pts": _pts,
                    "items": [{"target_fid": format_xy_token(p[0], p[1])}
                              for p in _pts],
                    "walk": sum(math.dist(a, b)
                                for a, b in zip(_pts, _pts[1:]))})
            else:
                # one stop; its recorded position anchors the chain.
                _xy = ((_ns_rows.get(_tok) or {}).get("xy")
                       if _pt is None else list(_pt))
                _p = ([float(_xy[0]), float(_xy[1])]
                      if isinstance(_xy, (list, tuple)) and len(_xy) == 2
                      else None)
                _members.append({"tok": _tok, "pts": [_p] if _p else [],
                                 "items": [dict(_it)], "walk": 0.0})
        # greedy nearest-endpoint chain, seeded from where the drone is
        _cur = ctx.pose_xy
        _cur = [float(_cur[0]), float(_cur[1])] if _cur else None
        _chained: list[dict] = []
        _rest = list(_members)
        while _rest:
            if _cur is None:
                _next = _rest.pop(0)
            else:
                def _gap(_m):
                    if not _m["pts"]:
                        return float("inf")
                    return min(math.dist(_cur, _m["pts"][0]),
                               math.dist(_cur, _m["pts"][-1]))
                _next = min(_rest, key=_gap)
                _rest.remove(_next)
                if (len(_next["pts"]) > 1
                        and math.dist(_cur, _next["pts"][-1])
                        < math.dist(_cur, _next["pts"][0])):
                    _next["pts"].reverse()
                    _next["items"].reverse()
            _chained.append(_next)
            if _next["pts"]:
                _cur = _next["pts"][-1]
        # flatten in chain order, dropping stops a kept stop already covers
        _kept_pts: list = []
        _flat: list[tuple[dict, dict]] = []
        for _m in _chained:
            for _i, _mv in enumerate(_m["items"]):
                _p = _m["pts"][_i] if _i < len(_m["pts"]) else None
                if (_p is not None and _span is not None
                        and any(math.dist(_p, _q) < _span / 2.0
                                for _q in _kept_pts)):
                    continue
                _flat.append((_m, _mv))
                if _p is not None:
                    _kept_pts.append(_p)
        _set_cap = mission_spec.flight_budget
        if len(_flat) > _set_cap:
            _n = _set_cap
            _stride = (len(_flat) - 1) / (_n - 1)
            _keep = sorted({int(round(_i * _stride)) for _i in range(_n)})
            _flat = [_flat[_i] for _i in _keep]
        for _m in _chained:
            _m["kept"] = sum(1 for _f, _ in _flat if _f is _m)
        _total = sum(_m["walk"] for _m in _chained)
        _n_stops = len(_flat)
        _reached = (min(1.0, _n_stops * _span / _total)
                    if _total and _span else 1.0)
        _subject = str(_said) if _said else f"{len(_chained)} places"
        if _reached >= 1.0:
            _note = (
                (f"covered {len(_chained)} places in {_n_stops} stops; each "
                 f"fits inside one observation.")
                if not _total else
                (f"covered {_total:.0f} m of {_subject} across "
                 f"{len(_chained)} segments in {_n_stops} stops."))
        else:
            _note = (f"covered {_reached:.0%} of {_subject}'s "
                     f"{_total:.0f} m.")

        def _pct(_m):
            if not _m["walk"] or _span is None:
                return 1.0
            return min(1.0, max(_m["kept"], 0) * _span / _m["walk"])
        _mm = ", ".join(f"{_m['tok']} {_pct(_m):.0%}" for _m in _chained[:8])
        _mm_more = (f", +{len(_chained) - 8} more"
                    if len(_chained) > 8 else "")
        _note = f"{_note} ({_mm}{_mm_more})"
        for _m in _chained:
            _covered_fids.add(_m["tok"])
        _names = {(_ns_rows.get(_m["tok"]) or {}).get("name")
                  for _m in _chained}
        _names.discard(None)
        _names.discard("")
        for _nm in sorted(_names):
            _note += _uncovered_segments(_nm)
        return ([_mv for _, _mv in _flat], _note, _subject, len(_chained))

    def _watch_apply(adds: list) -> dict:
        """look_for(): add classes to the watchlist (latest declaration of
        a class wins; the list only grows) and arm it with the engine."""
        adds = [dict(e) for e in adds]
        for _e in adds:
            mission_sam[:] = [c for c in mission_sam
                              if c["class"] != _e["class"]]
            mission_sam.append(_e)
        _rcpt = mission_spec.declare_concepts(list(mission_sam))
        ack: dict = {"now": [f"{c['class']} ({c['role']})"
                             for c in mission_sam]}
        if isinstance(_rcpt, dict) and "error" in _rcpt:
            ack["error"] = _rcpt["error"]
        _names = {s for s in (
            str(getattr(n, "name", "") or "").strip().lower()
            for n in (getattr(ctx.graph, "nodes", None)
                      or {}).copy().values()) if s}
        _note = _named_feature_note([c["class"] for c in adds], _names)
        if _note:
            ack["note"] = _note
        log.info("%s SAY watch: +%s -> %s", rid_tag,
                 [c["class"] for c in adds],
                 [c["class"] for c in mission_sam])
        return ack

    def _boost(where: str, concepts: list) -> dict:
        """look_for(..., at=) -- the targeted boost."""
        res = boost_queue(list(concepts), [where]) or {}
        if "error" not in res and res.get("queued"):
            # The placed form also arms the class it names.
            _adds = [{"class": str(c), "role": "target"}
                     for c in concepts if c]
            if _adds:
                _w = _watch_apply(_adds)
                res["watching"] = _w.get("error") or _w.get("now")
            res["note"] = ("boost queued -- extra segmentation at the "
                           "asked spot fires on your next flight; the "
                           "class is also on the standing watchlist for "
                           "every later flight")
        if "error" not in res:
            # A retro boost ingests detections now: refresh like a flight.
            (compute_namespace or {}).update(_surface_from_ctx(ctx))
            _rebuild_ns_rows()
            _surface_version[0] += 1
        if res.get("error"):
            return _plan_err({"boost": res}, str(res["error"]))
        return {"boost": res}

    def _ship(ack: dict, kind: str, jpegs: list, vis_text: str) -> bool:
        """Queue one look's or map's pictures (or, under the images
        withhold, its caption alone) for the next message; ``ack`` states
        what follows. True when pictures shipped."""
        ack["delivered"] = len(jpegs)
        if jpegs and withheld("images"):
            ack["delivered"] = 0
            ack["note"] = _PICTURE_WITHHELD_OF.format(kind)
            pending_vision.append((vis_text, []))
            return False
        if jpegs:
            ack["note"] = f"the {kind} follows in the next message"
            _deliver_frames(jpegs, _block_frames)
            pending_vision.append((vis_text, jpegs))
            return True
        ack["error"] = f"render returned no image bytes; no {kind} will follow"
        return False

    def _crops_of(fid, marks: list | None = None,
                   preamble: str = "") -> dict:
        # A list asks for one crop per row.
        _fids = [str(f) for f in fid] if isinstance(fid, (list, tuple)) \
            else [str(fid)]
        reqs, skipped, ref_notes = _build_crop_requests(
            [{"fid": f, "target_fid": f,
              **({"mark_fids": list(marks)} if marks else {})}
             for f in _fids])
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
            crops = look_renderer(reqs)
        except Exception as exc:  # noqa: BLE001 — render never crashes the mission
            log.warning("%s SAY crops render failed: %s", rid_tag,
                        _summarise_exc(exc))
            return {"error": f"crop render failed: {_summarise_exc(exc)}"}
        jpegs = [c["jpeg"] for c in crops
                 if isinstance(c, dict) and c.get("jpeg")]
        vis_text = preamble + _look_vision_text(crops)
        if _ship(ack, "crop", jpegs, vis_text) and marks:
            # Badges drawn (the legend IS the drawn set) and the renderer's
            # own out-of-frame list, on a crop that shipped.
            ack["badged"] = sum(
                len(c.get("legend") or []) for c in crops
                if isinstance(c, dict) and c.get("jpeg"))
            ack["not_drawn"] = [
                str(f) for c in crops
                if isinstance(c, dict) and c.get("jpeg")
                for f in (c.get("marks_out_of_view") or [])][:20]
        return ack

    def _crops_set(handle: str, marks: list | None = None) -> dict:
        """look(rows): one crop per row of the handle, in its own order, each
        badged with the row it shows; the crop budget is stated when it
        bites."""
        rows = [r for r in (_carry_read(handle) or [])
                if isinstance(r, dict) and r.get("fid")]
        if not rows:
            return {"error": (f"{handle} holds no rows to look at; name a "
                              "handle that has some, or find() one first")}
        fids = [_crop_token_of(r, str(r["fid"]) in _ns_rows) for r in rows]
        ack = _crops_of(
            fids, marks=marks,
            # No handle named: the executor's ref may be unbound.
            preamble=("One crop per row, in order, each badged with the "
                      "row it shows. "))
        if not withheld("images") and len(rows) > int(ack.get("delivered") or 0):
            ack.setdefault("ref_notes", []).append(
                f"{len(rows)} rows in {handle}: this call shows the first "
                f"{ack.get('delivered')} in the standing order; slice the "
                "handle to see the rest")
        return ack

    def _map_of(fid: str, mark_fids: list | None = None) -> dict:
        spec, m_skipped = _map_spec(fid, mark_fids or [])
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
                spec, _map_features_cached(),
                tile_origin_xy=ctx.tile_origin_xy,
                cell_size_m=float(ctx.cell_size_m),
                grid_shape=tuple(ctx.grid_shape))
            maps = [_m] if _m is not None else []
        except Exception as exc:  # noqa: BLE001 — render never crashes the mission
            log.warning("%s SAY map render failed: %s", rid_tag,
                        _summarise_exc(exc))
            return {"error": f"map render failed: {_summarise_exc(exc)}"}
        jpegs = [m["jpeg"] for m in maps if m.get("jpeg")]
        _ship(ack, "map", jpegs, _map_vision_text(maps))
        return ack

    def _plan_err(ack: dict, msg: str) -> dict:
        """A plan error, through the escalating loop breaker."""
        return {**ack, "error": _loop_broken_error(str(msg))}

    def _flown_stop(tok: str) -> tuple[str, str | None]:
        """A flown stop id -> its recorded point; an unflown one is refused.
        ``(token, error)``."""
        _sp = _plan_stops.get(tok.strip().lower())
        if _sp is not None:
            return format_xy_token(_sp[0], _sp[1]), None
        if re.fullmatch(r"stop_[0-9]{1,4}", tok.strip().lower()):
            _known = ", ".join(sorted(_plan_stops)) or "none yet"
            return tok, (f"{tok} has not been flown; the flown stops are: "
                         f"{_known}")
        return tok, None

    def _closest_names(what: str) -> str:
        """The 3 nearest named rows to a name that matched nothing."""
        names: dict[str, tuple] = {}
        for _key in ("cands", "anchors"):
            for _r in (compute_namespace or {}).get(_key) or []:
                if not isinstance(_r, dict) or not _r.get("fid"):
                    continue
                _nm = str(_r.get("name") or "").strip()
                if _nm:
                    names.setdefault(_nm.lower(), (str(_r["fid"]), _nm))
        close = difflib.get_close_matches(
            (what or "").lower(), list(names), n=3, cutoff=0.55)
        if not close:
            return ""
        return "; closest named rows: " + ", ".join(
            f"\"{names[c][1]}\" ({names[c][0]})" for c in close)

    # The call in flight: what _ground_names noted and _analyze answered,
    # read by _finish.
    _call: dict = {}

    def _call_minted():
        """The handle this call's last flight minted, or None."""
        return (_code_minted[_call.get("minted", 0):] or [None])[-1]

    def _ground_names(names: dict) -> dict:
        """Open a call and ground its names (``{name: why}``) under the
        name-match floor: ``{"ties": {name: top-tie fids}}``, or the first
        refusal."""
        _call.clear()
        _call["notes"] = notes = []
        _call["minted"] = len(_code_minted)   # this call's flights mint past it
        ties: dict = {}
        if not names:
            return {"ties": ties}
        _sink: dict = {}
        g_out = run_geom_program_tool(
            {"program": {"ops": [
                {"ref": f"r{i}", "op": "find",
                 "args": {"query": q.lower()}}
                for i, q in enumerate(names)]}},
            namespace=(compute_namespace or {}),
            raw_sink=_sink, preflight=preflight)
        if isinstance(g_out, dict) and "error" in g_out:
            return _plan_err({}, str(g_out["error"]))
        g_res = _sink.get("results") or {}
        _WHY_VERB = {"stage": "standing at", "move": "flying",
                     "map": "mapping around", "crops": "looking at",
                     "boost": "looking around", "facing": "facing"}
        for i, (q, why) in enumerate(names.items()):
            rows = [r for r in (g_res.get(f"r{i}") or [])
                    if isinstance(r, dict) and r.get("fid")]
            rows = [r for r in rows
                    if r.get("name_match") is None
                    or float(r["name_match"]) >= _FUZZY_FLOOR]
            _what = q.lower() or "that"
            if not rows:
                return _plan_err({}, (
                    f"rule N: nothing named \"{_what}\" is on "
                    f"this surface{_closest_names(_what)}; say it "
                    "differently, say a fid off a served row, or "
                    "perceive more first"))
            # every name keeps only its top name_match tie group
            _top = rows[0].get("name_match")
            ties[q] = [str(r["fid"]) for r in rows
                       if r.get("name_match") == _top] \
                or [str(rows[0]["fid"])]
            if why == "stage" and len(ties[q]) > 1:
                _f = ", ".join(ties[q][:4])
                return _plan_err({}, (
                    f"rule F: a viewpoint is one point; "
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

    def _analyze(ops: list, rn_end: int, answer, page) -> dict:
        """Run one compiled program; ``answer`` names the ref whose envelope
        is served, ``page`` the ``(ref, start)`` a more_of window pages."""
        if page:
            # Paging a handle with nothing left to page is refused here.
            _h, _lo = page
            _held = len(_carry_read(_h) or [])
            if _held and (_held <= _BROAD_DUMP_SOFT_ROWS or _lo >= _held):
                return _plan_err({}, (
                    f"all {_held} row(s) of {_h} are already served -- there "
                    f"is no further block to serve. Narrow it "
                    f"({_h}.order(color=\"<colour>\"), "
                    f"{_h}.order(nearest=\"<fid>\"), "
                    f"{_h}.filter(within=\"<place>\"))"
                    + ("." if withheld("look") else f" or look({_h}).")))
        out = _dispatch_analyze_scene({"program": {"ops": ops}},
                                      answer_ref=str(answer or ""))
        if isinstance(out, dict) and "error" in out:
            return {"error": out["error"]}
        plan_rn[0] = max(plan_rn[0], int(rn_end or 0))
        # The column this call ordered by, so _finish can say when the
        # ordering separated nothing.
        _sb = next((o for o in reversed(ops) if o["op"] == "score_by"), None)
        _em = ((_sb["args"].get("emphasis") or [None])[0]) if _sb else None
        _call.update(answer=answer, page=page,
                     rows=carry_results.get(answer),
                     order=str(_em["key"]) if isinstance(_em, dict)
                     and _em.get("key") else None)
        return {"result": out}

    def _stand(at: str, facing: str) -> dict:
        st = _stage_apply(at, facing)
        if not st.get("staged"):
            return _plan_err({}, (
                "rule F: the stance did not bind: "
                + str(st.get("rejected") or st.get("error"))))
        return {"stance": st}

    def _move_to(tok: str) -> dict:
        tok, _stop_err = _flown_stop(tok)
        if _stop_err:
            return _plan_err({}, _stop_err)
        _accepted, _item_errors = _mission_sanitize_moves([tok])
        mv = _fly_moves(_accepted, _item_errors)
        _ok = [r for r in mv.get("flown") or []
               if isinstance(r, dict) and "error" not in r]
        if not (_ok and _accepted):
            # nothing flew: the failure is the result
            _msg = next(
                (str(r.get("error")) for r in mv.get("flown") or []
                 if isinstance(r, dict) and r.get("error")),
                None) or next(
                (str(e.get("error")) for e in mv.get("rejected") or []
                 if isinstance(e, dict) and e.get("error")),
                None) or str(mv.get("error") or "nothing flew")
            return _plan_err({"flown": mv}, _msg)
        return {"flown": mv}

    def _cover(members: list, said_to: str) -> dict:
        """explore(): one pass of flights over a place or a set, whose
        members arrive as a name's top tie group (a list), a handle's rows
        or a token."""
        _raw_members: list = []
        for _t in members:
            if isinstance(_t, list):
                _raw_members.extend(_t)
                continue
            _t, _stop_err = _flown_stop(_t)
            if _stop_err:
                return _plan_err({}, _stop_err)
            if _is_handle_token(_t):
                _hr = [str(r["fid"]) for r in (_carry_read(_t) or [])
                       if isinstance(r, dict) and r.get("fid")]
                if not _hr:
                    return _plan_err({}, (
                        f"{_t} holds 0 fid-bearing row(s) "
                        "here; there is nothing to cover."))
                _raw_members.extend(_hr)
                continue
            _raw_members.append(_t)
        _accepted, _item_errors = _mission_sanitize_moves(
            list(dict.fromkeys(_raw_members)))
        if not _accepted:
            return _plan_err({}, next(
                (str(e.get("error")) for e in _item_errors
                 if isinstance(e, dict) and e.get("error")),
                "nothing to cover"))
        _accepted, _cov_note, _subject, _segs = _cover_items(
            _accepted, _said=said_to)
        if not _accepted:
            return _plan_err({}, _cov_note)
        mv = _fly_moves(_accepted, _item_errors, cover=_subject)
        if "error" in mv:
            return _plan_err({"explored": mv}, str(mv["error"]))
        if _segs > 1:
            mv["segments"] = _segs
        # A truncated pass reads as truncated.
        if mv.get("stops", len(_accepted)) < len(_accepted):
            _cov_note = (f"the pass ended after "
                         f"{mv.get('stops', 0)} of its stops.")
        if _cov_note:
            mv["note"] = (f"{_cov_note} {mv['note']}"
                          if mv.get("note") else _cov_note)
        # THE PASS SHOWS ITS CATCH: one crop per row it added, pass order.
        _h = _call_minted()
        _rows = [r for r in (carry_results.get(_h) or [])
                 if isinstance(r, dict) and r.get("fid")] if _h else []
        _catch = _rows[:_CATCH_CROPS]
        if _catch:
            _cack = _crops_of(
                [_crop_token_of(r, str(r["fid"]) in _ns_rows)
                 for r in _catch],
                preamble=("The pass's catch, one crop per "
                          "row it added, in pass order: "))
            if not _cack.get("error"):
                mv["catch"] = {k: _cack[k] for k in ("delivered", "not_drawn")
                               if k in _cack}
                if len(_catch) < len(carry_results.get(_h) or []):
                    mv["catch"]["shown"] = (
                        f"first {len(_catch)} of "
                        f"{len(carry_results[_h])}" + (
                            "" if withheld("look") else
                            f" -- look({_h}[a:b]) pages the rest"))
                if withheld("images"):
                    mv["catch"] = {"note": _PICTURE_WITHHELD}
        return {"explored": mv}

    def _crops(of, fid, marks) -> dict:
        _mk = [m for m in marks or [] if isinstance(m, str) and m.strip()]
        crop = (_crops_set(str(of), marks=_mk or None) if of
                else _crops_of(fid, marks=_mk or None))
        if crop.get("error") and not crop.get("delivered"):
            return _plan_err({"crop": crop}, str(crop["error"]))
        return {"crop": crop}

    def _map(around: str) -> dict:
        # The map badges the set in scope.
        _h = _those[0]
        _mrows = [r for r in (_carry_read(_h) or [])
                  if isinstance(r, dict) and r.get("fid")] if _h else []
        _mfids = [str(r["fid"]) for r in _mrows[:_MAP_MAX_MARKS]]
        mp = _map_of(around, mark_fids=_mfids)
        if _mfids:
            mp["badged_from"] = _h
            if len(_mrows) > len(_mfids):
                mp.setdefault("ref_notes", []).append(
                    f"{len(_mrows)} rows in {_h}; this map badges the "
                    f"first {len(_mfids)}")
        if mp.get("error") and not mp.get("delivered"):
            return _plan_err({"map": mp}, str(mp["error"]))
        return {"map": mp}

    def _watch(concepts: list) -> dict:
        w = _watch_apply([{"class": c, "role": "target"} for c in concepts])
        if w.get("error"):
            return _plan_err({"watching": w}, str(w["error"]))
        return {"watching": w}

    def _finish(ack: dict) -> dict:
        """Close a call nothing refused: the tie note, the grounding notes,
        the streak reset, the scope and paging writes."""
        # An ordering that separated nothing says how many rows tie.
        _ok_key = _call.get("order")
        if _ok_key:
            _rows = [r for r in (_call.get("rows") or [])
                     if isinstance(r, dict)]
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
        if _call.get("notes"):
            ack["note"] = (f"{ack['note']}\n" if ack.get("note")
                           else "") + "; ".join(_call["notes"])
        _err_streak["sig"] = None
        _err_streak["n"] = 0
        if _call.get("page"):
            _widened.update([_call["page"]])
        # The set in scope: the call's non-empty answer, or what its last
        # flight added.
        if _call.get("answer") and _call.get("rows"):
            _those[0] = _call["answer"]
        if _call_minted():
            _those[0] = _call_minted()
        return ack

    _mission_py: list = [None]

    def _dispatch_run(args) -> dict:
        """One code block, run in the mission's persistent namespace; an
        error costs the line it is on."""
        if "__args_parse_error" in (args or {}):
            return {"error": "the call's arguments did not parse as JSON: "
                             f"{args['__args_parse_error']}; nothing ran"}
        code = (args or {}).get("code")
        if not isinstance(code, str) or not code.strip():
            return {"error": "run takes one block of code in code"}
        if _mission_py[0] is None:
            _mission_py[0] = _py_surface.MissionPython(_py_surface.MissionEnv(
                # A read of a handle's rows re-derives it first.
                rows_of_ref=lambda ref: (
                    _ensure_fresh([ref]),
                    [r for r in (carry_results.get(ref) or [])
                     if isinstance(r, dict)])[1],
                tied_of_ref=lambda ref: bool(
                    getattr(_carry_read(ref), "name_set", False)),
                plan_rn=plan_rn, widened=_widened, minted=_code_minted,
                say_err=_plan_err, row_of_fid=_ns_rows.get,
                ground=_ground_names, analyze=_analyze, stand=_stand,
                move=_move_to, cover=_cover, crops=_crops, map=_map,
                watch=_watch, boost=_boost, finish=_finish))
        _code_names.clear()
        _code_names.update(re.findall(r"\br\d+\b", code))
        _code_names.update(
            v.ref for v in _mission_py[0].ns.values()
            if isinstance(v, _py_surface.Rows))
        _code_minted.clear()
        _episode_script.append(code.strip())
        out = _mission_py[0].run_block(code)
        out["script_lines"] = sum(
            len(b.splitlines()) for b in _episode_script)
        return out

    def _mission_end(ended_by: str, turns: int = 0) -> MissionReply:
        if transcript is not None:
            transcript.append({"kind": "mission_end",
                               "ended_by": ended_by, "turns": int(turns)})
        return MissionReply(ended_by=ended_by, turns_used=int(turns))

    if thread_infeasible:
        return _mission_end("compaction_infeasible")

    for turn_idx in itertools.count():
        if diag is not None:
            diag["turns_used"] = turn_idx + 1
        # The operator's re-tasks land at the top of a turn, never between
        # a turn's tool calls and their results.
        if retask is not None:
            while (operator_turn := retask()) is not None:
                log.info("%s operator re-task at turn %d (%d chars)",
                         rid_tag, turn_idx, len(operator_turn))
                reasoner.append_user(operator_turn)
        if turn_idx >= mission_spec.turn_cap:
            log.info("%s mission turn cap (%d) reached", rid_tag,
                     mission_spec.turn_cap)
            return _mission_end("turn_cap", turns=turn_idx)
        try:
            r = reasoner.step_tool_call(stop_tools)
        except (CompactionInfeasible, CompactionInvariantError) as exc:
            # Terminal: flag the thread and drop this turn's unread message.
            reasoner.compaction_infeasible = True
            msgs = getattr(reasoner, "messages", None)
            if (turn_idx == 0 and msgs
                    and msgs[-1].get("role") == "user"
                    and msgs[-1].get("content") == user_message):
                msgs.pop()
            log.warning(
                "%s stop_round compaction-infeasible -> terminal carry "
                "(thread flagged; remaining rounds skip): %s",
                rid_tag, _summarise_exc(exc))
            return _mission_end("compaction_infeasible", turns=turn_idx + 1)
        except _RETRIABLE_LLM_ERRORS as exc:
            if _is_oversize_rejection(exc):
                log.warning(
                    "%s stop_round request does not fit the served window "
                    "-> terminal (retry would re-send the same bytes): %s",
                    rid_tag, _summarise_exc(exc))
                if transcript is not None:
                    transcript.append({
                        "kind": "oversize_rejection",
                        "error": _summarise_exc(exc),
                    })
                return _mission_end("oversize_rejection", turns=turn_idx + 1)
            transient_failures += 1
            cancel = getattr(reasoner, "cancel_event", None)
            log.warning(
                "%s stop_round transient LLM error %d%s: %s",
                rid_tag,
                transient_failures,
                "" if cancel is not None else f"/{MAX_ROUND_TRANSIENT}",
                _summarise_exc(exc),
            )
            if transcript is not None:
                transcript.append({
                    "kind": "transient_error",
                    "attempt": transient_failures, "error": _summarise_exc(exc),
                })
            if cancel is None:
                if transient_failures >= MAX_ROUND_TRANSIENT:
                    return _mission_end(
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
                         rid_tag, turn_idx)
                if transcript is not None:
                    transcript.append({"kind": "malformed_envelope",
                                       "turn_idx": turn_idx})
                reasoner.append_repair(MALFORMED_ENVELOPE_NUDGE)
                continue
            if not content.strip():
                # An empty turn is a dropped message: repaired once.
                no_call_streak += 1
                if transcript is not None:
                    transcript.append({"kind": "empty_turn",
                                       "turn_idx": turn_idx,
                                       "streak": no_call_streak})
                if no_call_streak < 2:
                    log.info("%s EMPTY turn %d -> repair", rid_tag,
                             turn_idx)
                    reasoner.append_repair(EMPTY_TURN_NUDGE)
                    continue
                log.info("%s EMPTY turn %d repeated; mission ends",
                         rid_tag, turn_idx)
            elif _code_reply(content):
                # A block in the reply channel: repaired once (the streak
                # is shared with the empty turn).
                no_call_streak += 1
                if transcript is not None:
                    transcript.append({"kind": "code_shaped_reply",
                                       "turn_idx": turn_idx,
                                       "streak": no_call_streak})
                if no_call_streak < 2:
                    log.info("%s CODE-SHAPED reply turn %d -> "
                             "repair", rid_tag, turn_idx)
                    reasoner.append_repair(CODE_REPLY_NUDGE)
                    continue
                log.info("%s CODE-SHAPED reply turn %d repeated; "
                         "mission ends", rid_tag, turn_idx)
            else:
                no_call_streak = 0
            # A clean prose reply before any flight answers nothing: hand
            # the turn back, up to NO_FLIGHT_ABSTAIN_STREAK.
            if no_call_streak == 0 and mission_flights == 0:
                no_flight_streak += 1
                if transcript is not None:
                    transcript.append({"kind": "no_flight_reply",
                                       "turn_idx": turn_idx,
                                       "streak": no_flight_streak})
                if no_flight_streak < NO_FLIGHT_ABSTAIN_STREAK:
                    log.info("%s reply with ZERO flights at turn %d "
                             "(streak %d) -> repair", rid_tag, turn_idx,
                             no_flight_streak)
                    reasoner.append_repair(NO_FLIGHT_REPLY_NUDGE)
                    continue
                log.info("%s reply with ZERO flights at turn %d, streak "
                         "%d -- abstain", rid_tag, turn_idx,
                         no_flight_streak)
            log.info("%s mission ended by reply at turn %d (%d flights)",
                     rid_tag, turn_idx, mission_flights)
            return _mission_end("reply", turns=turn_idx + 1)
        no_call_streak = 0
        no_flight_streak = 0

        # Dispatch each call in order, answering every call id.
        for tc in r["tool_calls"]:
            name = (tc.get("function") or {}).get("name") or ""
            cid = tc.get("id") or ""
            args = json.loads(tc["function"]["arguments"])
            if name == "run":
                out = _dispatch_run(args)
                if isinstance(out, dict) and not withheld("look_for"):
                    out.setdefault("watching", (
                        _watching_trailer()
                        or "nothing -- no class is segmented on any flight "
                           'until one is looked for; call look_for("<class>").'))
                reasoner.append_tool_result(cid, "run", out)
                if transcript is not None:
                    transcript.append({
                        "kind": "run",
                        "turn_idx": turn_idx,
                        "reasoning": str(r.get("reasoning") or "")[:4000],
                        "said": str((args or {}).get("code") or "")[:400],
                        # already capped (python_surface.PRINT_CHAR_CAP)
                        "printed": (str(out.get("printed") or "")
                                    if isinstance(out, dict) else ""),
                        "errored": bool(isinstance(out, dict)
                                        and "error" in out),
                        # every image this block showed, by sha
                        "frames": list(_block_frames)})
                _block_frames.clear()
            else:
                # An unknown tool's id is answered, never executed.
                reasoner.append_tool_result(cid, name or "unknown", {
                    "error": f"unknown tool {name!r}; available: run"})
        # Pictures ride a user turn after the turn's tool results.
        for _vtext, _vjpegs in pending_vision:
            reasoner.append_vision_user(_vtext, _vjpegs)
        pending_vision.clear()
