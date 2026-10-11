"""The mission conversation: one persistent reasoner, one served tool.

``run_mission`` (the CityNav bench) opens ONE conversation per mission
over ``prompts.CODE_TOOLS`` ([run]) and serves ``CODE_SYSTEM_PROMPT``
beside it. Every model act is a native tool call; everything else is
thinking, and there is no structured-output emission on the agent path.

Each call carries one BLOCK of Python. ``_dispatch_run`` hands it to
``python_surface``, whose library calls compile through ``code_surface``
and run on this loop's verbs (``_ground_names``, ``_analyze``, the acts,
``_finish``):

  * ``go_to(...)`` / ``explore(...)`` FLY the moment the block lands
    (``MissionSpec.fly``), and the query surface re-derives from the world
    the flight just folded.
  * ``stand(at=, facing=)`` adopts the STANDING viewpoint (vN handles; its
    side columns ride every find row, its facing orients what renders).
  * ``find`` / ``intersection`` / ``.order`` / ``.filter`` run op-programs
    over the mission-global rN chain.
  * ``look_for`` directs the segmenter; ``look`` / ``show_map`` fetch a
    crop and the map.

Names bind across blocks for the whole mission. The model's plain
no-tool-call reply ends it, and the LAST flight's target is the answer
(the engine owns the commit). Liveness bounds the model never hears:
``MissionSpec.turn_cap``, ``flight_budget``, ``deadline_epoch``.
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

# Transient LLM-side errors: a retry, never the end of the reasoner.
#   - HTTPStatusError: the serve returned a non-2xx (a grammar refusal of
#     a malformed args envelope, or 429 backpressure).
#   - ConnectError / ReadError / TimeoutException (connect, read, write,
#     pool, and the ConnectTimeout of a dead relay port): a link fault the
#     reasoner recovers from after a short backoff.
_RETRIABLE_LLM_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPStatusError,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    # A collapsed completion: a repeated-token run with no content and no
    # call. Named, so it does not depend on its ValueError base.
    DegenerateCompletion,
    # ValueError covers a transport-level decode failure (empty body /
    # non-JSON response). A fresh generation usually sticks.
    ValueError,
)

#: What a serve says when it rejects a request for its SIZE. Lowercased
#: substrings, matched against the response body.
#:
#: A grammar rejection turns on what the model GENERATED, and a fresh
#: generation differs; an oversize rejection turns on what was SENT, and a
#: retry re-sends the same bytes. A spelling this misses falls through to
#: the retry path: it can cost a wasted retry, never turn a retriable error
#: permanent.
_OVERSIZE_REJECTION = (
    "maximum context length",
    "context length of",
    "longer than the maximum",
    "reduce the length",
)


def _is_oversize_rejection(exc: BaseException) -> bool:
    """Did the serve reject this request because it does not FIT?

    Deterministic in the request, so retrying re-sends the same bytes and
    gets the same answer. The compactor is what should have prevented it.
    """
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
    ``TRANSIENT_CLIENT_STATUS``)?

    A timeout, a rate limit or a spent quota is the serve's weather; any
    other 4xx is a request the method built wrong, and re-sending it fails
    the same way."""
    return (isinstance(exc, httpx.HTTPStatusError)
            and 400 <= exc.response.status_code < 500
            and exc.response.status_code not in TRANSIENT_CLIENT_STATUS)


def _summarise_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _program_dedup_key(program: dict) -> str | None:
    """Canonical identity of a SELF-CONTAINED program, or ``None`` when it
    reads a handle it does not define and is therefore never deduped.

    While the surface version holds, an identical self-contained program is
    a pure re-render -- its result cannot change. Every op's ref is an
    ``rN``; identity is judged with each renamed to its emission index
    (``r0``, ``r1``, ...) wherever it is spelled, so an incrementing ``rN``
    does not defeat it while emission-order permutations stay distinct. A
    program that reads an ``rN`` it does not define resolves against the
    mission's ACCUMULATED results, so it stays dedup-exempt.
    """
    ops = program.get("ops") or []
    if not ops:
        return None
    ren = {op["ref"]: f"r{i}" for i, op in enumerate(ops)}
    blob = json.dumps(ops, sort_keys=True, default=str)
    if any(t not in ren for t in re.findall(r"\br\d+\b", blob)):
        return None
    return re.sub(r"\br\d+\b", lambda m: ren[m.group(0)], blob)


# The dedup hit serves a one-line REFERENCE, never a fresh payload (a full
# re-render of an identical result is loop fuel). {turn} = the turn the
# original ran on. The note echoes nothing of the submitted program: the
# identity fact ("turn {turn}") already names the call.
#
# IT LEADS WITH THE WORLD, NOT WITH THE SUBMISSION. It fires when the
# surface has not moved since that turn, which says NO program would answer
# differently, so only changing the world helps. Every path that can mint a
# row bumps the version (a flight, a perceive, a retro boost, a stance), so
# "nothing has entered the world" is exactly true whenever this fires.
_DEDUP_NOTE = (
    "no new information since turn {turn}: nothing has entered the world "
    "since then -- no flight, no perceive -- so every program returns what "
    "it returned then, and this one is not re-rendered. Its handles"
    "{handles} stay live for later ops. Change the world (fly, perceive) or "
    "ask something different."
)
_DEDUP_REPEAT_NOTE = (
    "still nothing new, a {n} time: the world has not moved since that turn, "
    "so no program can answer differently until something enters it -- a "
    "flight folds new detections. Live now: {c_n} detection fid(s), {a_n} "
    "prior map row fid(s){cols}."
)


def _ordinal(n: int) -> str:
    """1 -> 1st, 2 -> 2nd, 3 -> 3rd, 11..13 -> th."""
    n = int(n)
    if 10 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _persist_look_audit(
    *, audit_dir: str, stop_id: str, step_idx: int, rid_tag: str, cid: str,
    annotate: bool, tool_ack: dict, vision_text: str, crops: list[dict],
) -> None:
    """Persist EXACTLY what one ``look`` call delivered to the model, so a
    completed run can be audited on what each tool call provided in text and
    images (a bad crop -- mis-centred, target out of frame, unreadable -- is
    only checkable by seeing what the model saw).

    Writes ``manifest.json`` under ``<audit_dir>/<stop_id>/<cid>/`` (keyed by
    stop_id + the tool-call id): the
    tool-ack + the vision-user message TEXT, plus per crop the FULL render
    record minus the jpeg bytes — the resolved spec
    (``look_render.render_look_from_spec``'s input: map_name / center / marks /
    boundaries / pose / context) plus the derived audit geometry and the
    ``sha256`` of the delivered bytes. The pixels themselves are NOT stored:
    the record is the re-render input, and the sha is the proof the re-render
    shows what the model saw (a hash mismatch surfaces as renderer/raster
    drift, keyed by ``renderer_version`` / ``renderer_env``). ``jpg`` stays as
    the crop's virtual filename — the dashboard's URL + delivery-order key.
    The manifest lands atomically (tmp + replace) so a mid-run reader never
    parses a partial file. Never raises (audit is best-effort; it must not
    perturb the round)."""
    try:
        d = Path(audit_dir) / str(stop_id) / str(cid)
        d.mkdir(parents=True, exist_ok=True)
        manifest: dict = {
            "stop_id": str(stop_id), "step_idx": int(step_idx),
            "rid": str(rid_tag), "cid": str(cid), "annotate": bool(annotate),
            "tool_ack": tool_ack, "vision_user_text": vision_text,
            "n_crops": len(crops), "crops": [],
        }
        for i, c in enumerate(crops):
            fid = str(c.get("center_fid") or f"crop{i}")
            rec = {k: v for k, v in c.items() if k != "jpeg"}
            rec["jpg"] = f"{i:02d}_{fid}.jpg"
            manifest["crops"].append(rec)
        tmp = d / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, default=str, indent=2))
        os.replace(tmp, d / "manifest.json")
    except Exception:  # noqa: BLE001 — audit persistence never perturbs the round
        # WARNING, not debug: a persist failure must be visible in the
        # runner log.
        log.warning("look audit persist failed at %s", audit_dir, exc_info=True)


def _deliver_frames(jpegs: list[bytes], sink: list[str]) -> None:
    """Record every image about to reach the model: its sha256 into ``sink``
    (the turn's ``frames`` list in ``conversation.jsonl``) and the exact bytes
    into ``<workdir>/frames/<sha256>.jpg`` when a mission is recording.

    The sha is of the delivered bytes, so it equals the wire log's
    ``image_ref.sha256`` (``maps.llm.http._scrub_image_parts``) and the look
    manifest's per-crop ``sha256``: one name for one image across all three
    records. A file already there is the same bytes and is left alone. Never
    raises -- the pixel copy is evidence, and a full disk must not end the
    turn that produced it."""
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


# Tool-call residue markers: a prose turn carrying one of these is a
# MALFORMED ENVELOPE (a parser miss / an envelope written into content),
# never a deliberate no-call: repair it, never carry on it.
ENVELOPE_RESIDUE = ("<tool_call", "</tool_call", "<function")

# Consecutive replies carrying ZERO flights that end the mission: the first
# two draw NO_FLIGHT_REPLY_NUDGE, the third is a CONFIDENT ABSTAIN and ends
# it. Turns are not free, and a second ask collects most of what the first
# leaves.
#
# It is a STREAK: anything the model EXECUTES between two replies resets it
# (see the tool-call site), so a model doing real work is never cornered by
# it. The harness still never picks a target: zero flights commits nothing.
NO_FLIGHT_ABSTAIN_STREAK = 3


# ---------------------------------------------------------------------------
# The mission contract (the CityNav bench path): fly-on-call, one
# conversation, the plain reply ends the mission.
# ---------------------------------------------------------------------------


@dataclass
class MissionSpec:
    """The engine half of the mission contract, injected into the loop.

    ``fly`` takes ONE normalized move item ({"target_fid": <fid or
    canonical "[x, y]" token>}) and executes it NOW — resolve to a nadir
    stop, render, segment, fold — returning the model-facing flight report
    dict (the engine owns its content, including the operator-cue echo).
    A flight that cannot resolve returns {"error": ...} instead. After
    every fly the loop refreshes its analyze surface from ``ctx`` and
    bumps the dedup surface version.

    ``declare_concepts`` records the standing watchlist ({class, role}
    list, the full set latest-wins) with the engine (segmenter re-arm)
    and returns the receipt dict. The perceive.watchlist channel calls
    it with the maintained full set on every add/remove.
    """

    fly: "Callable[[dict], dict]"
    declare_concepts: "Callable[[list[dict]], dict]"
    # ``coverage_stops(fid, spacing_m) -> [[x, y], ...] | None`` — the
    # stops an ``explore`` flies over that fid's own footprint, one per
    # ``spacing_m`` of its length, bounded by the engine. ``None`` when the
    # fid is on neither graph or carries no cells. The loop owns the pass
    # itself (it flies every stop and the return leg through ``fly``); only
    # the geometry lives with the world-state.
    coverage_stops: "Callable[[str, float], list | None] | None" = None
    #: The station's word on which stops of a pass the recording holds
    #: ALREADY: ``[(target token, (x, y) or None), ...] -> [bool, ...]``.
    #: The pass takes a held stop first, whatever the model's order, so the
    #: crew's one sweep is read the moment it lands instead of after every
    #: stop before it. None on the bench, where a stop costs no flight and
    #: the order is the model's.
    held_now: "Callable[[list], list[bool]] | None" = None
    # Liveness backstops, NEVER served: the model hears about no budget.
    # The turn cap is the ONLY ending the model does not own; both bounds
    # sit far above a working episode and exist so a pathological
    # conversation cannot run unbounded. The flight budget also bounds an
    # explore pass: a stop IS a flight.
    flight_budget: int = 1000
    turn_cap: int = 100
    # The wall-clock half of the same backstop, as an ABSOLUTE epoch — a
    # count of stops cannot bound a count of seconds, and the harness kills
    # on seconds.
    #
    # ABSOLUTE, not a duration: the harness owns the one true start and the
    # cap, so nothing here re-derives an elapsed that would drift past
    # pre-flight pose resolution and ingest.
    #
    # ``None`` means NO CLOCK, and it is the default because the field
    # station has no harness cap: a flight mission flies for
    # as long as the crew is flying and must never stop on a timer.
    deadline_epoch: float | None = None


@dataclass(frozen=True)
class MissionReply:
    """How the mission's conversation ended.

    ``ended_by`` = "reply" (the model's plain no-tool-call turn — the ONE
    model-owned ending) | "turn_cap" | "transient_failures" |
    "request_refused" | "oversize_rejection" | "compaction_infeasible".
    The commit is the engine's job (it tracked
    every flight it executed); this reply carries the conversation facts.
    """

    ended_by: str
    reply_text: str = ""
    turns_used: int = 0
    flights_flown: int = 0
    no_call_turns: int = 0


def _served_flight(report: dict) -> dict:
    """A flight's report as the model reads it: what it asked for and what
    the flight found, never where the aircraft was. ``pose_xy`` and
    ``alt_m`` stay on the report for the engine -- the flight view, a stop
    flown again, the cover walk -- and leave here: served, they read as a
    placement to critique and a reason to fly the same row again."""
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
    # The numeric extent makes a pixel position back-projectable — an
    # edge-shifted window silently breaks the centred-window assumption
    # otherwise.
    _ww = c.get("world_window")
    if isinstance(_ww, (list, tuple)) and len(_ww) == 4:
        _span += (f" [x {_ww[0]:g}..{_ww[2]:g}, "
                  f"y {_ww[1]:g}..{_ww[3]:g}]")
    return _span


def _look_vision_text(crops: list[dict]) -> str:
    # Thin white ellipses + numbered badges on the image; the digit -> fid
    # binding lives HERE in the text legend. Color is not stated: the model
    # reads it from the unobscured pixels. Each crop line states its
    # delivered span, so a wide frame never implies a close look.
    from maps.search.render.look_render import (  # noqa: PLC0415
        LOOK_MAX_BADGES,
    )
    # THE CAPTION NAMES WHAT THE PICTURE IS OF. center_fid is a STOP id on a
    # flight view and absent on some crops; the legend's own center entry is
    # the subject the renderer actually drew, so it is the fallback.
    def _subject(c):
        f = c.get("center_fid")
        if f and not str(f).startswith("stop_"):
            return str(f)
        for e in c.get("legend") or []:
            if e.get("role") == "center" and e.get("fid"):
                return str(e["fid"])
        return str(f) if f else "?"

    order = ", ".join(_subject(c) for c in crops)
    # GSD from the crop records themselves, never a literal: the renderer
    # states what it delivered, and a model judging whether a small object is
    # resolvable must not be told a sharper picture than it was given.
    _gsds = {round(float(c["gsd_m_per_px"]), 2)
             for c in crops if c.get("gsd_m_per_px")}
    _gsd = (f"{min(_gsds):.2f} m/px" if len(_gsds) == 1
            else f"{min(_gsds):.2f}-{max(_gsds):.2f} m/px" if _gsds
            else "m/px unknown")
    # HOW SHARP THE PICTURE ACTUALLY IS, not how many pixels it has. The
    # delivered GSD is grid spacing divided by an upscale, and the grid can
    # be finer than the camera; resampling adds pixels, never detail, so
    # the source imagery's GSD bounds what the caption may claim.
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
        # THE PIXEL GRID, per image, because it is not constant. With the
        # grid and the window both named, what the model SEES converts into
        # a world point.
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
            # the run's own imagery: nothing has been flown over this ground yet.
            _sxy = c.get("world_xy") or ["?", "?"]
            base += (" ; WARNING: this fid's position "
                     f"[{_sxy[0]}, {_sxy[1]}] is inside this crop's "
                     "window but this run has not imaged that ground yet "
                     "-- the black region is unobserved, not a dark "
                     "object; its color/shape cannot be read until a leg "
                     "flies over it (go_to it, then look).")
        elif c.get("subject_on_data") is False:
            # subject-centered window past the raster's data edge: the
            # position IS in frame but the raster holds no pixels there.
            _sxy = c.get("world_xy") or ["?", "?"]
            base += (" ; WARNING: this fid's position "
                     f"[{_sxy[0]}, {_sxy[1]}] is inside this crop's "
                     "window but past the map raster's DATA edge -- "
                     "the black region is off-raster no-data, not a "
                     "dark object; the raster holds no pixels for "
                     "this subject, so its color/shape cannot be "
                     "read from this image.")
        # The subject is IN frame and still bigger than the frame: the badge
        # alone binds its digit to whichever fragment of the edge crosses
        # the window, so state the size and say the ink is cut.
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


# A map framed to what was asked about keeps this much slack around it, so a
# set sits in its surroundings rather than cropped to its own edge — the
# surroundings are usually what the cue is relating it to.
_MAP_FRAME_MARGIN = 1.6
# ...and never frames tighter than this. Below ~358 m the renderer's
# 0.35 m/px GSD floor is already binding at 1024 px, so a tighter frame buys
# no detail and only removes context.
_MAP_FRAME_MIN_M = 200.0


def _named_feature_note(classes: list[str], names: set[str]) -> str | None:
    """A factual note when a watch class IS a world-state NAME.

    Class-blind name membership, nothing else: the watchlist takes a
    visual class, and a proper noun there ('airway boulevard') buys a
    segmenter pass for a thing the world state already carries as a row.
    The note DESCRIBES; the arm stands exactly as said.
    """
    named = [c for c in classes if c.strip().lower() in names]
    if not named:
        return None
    q = ", ".join(f"'{c}'" for c in named)
    return (f"{q}: a NAMED feature on this map, already a world-state row "
            f"-- the segmenter takes visual classes like 'house' or "
            f"'pool'. Watched as said.")


# How many of a pass's own new rows ride back as crops on its ack -- the
# catch. Half the 16-image context budget, so the model's own later looks
# keep room; the note pages the rest.
_CATCH_CROPS = 8
# What a look and a pass's catch say under the ``images`` withhold.
_PICTURE_WITHHELD = ("the crop's caption follows in the next message; "
                     "its picture is not served on this run")


def _mission_sanitize_moves(raw: list[str]) -> tuple[list[dict], list[dict]]:
    """The move targets as flight items: each a literal fid or the
    canonical ``"[x, y]"`` token, under ``target_fid``. A result handle is
    refused. Returns ``(accepted_items, item_errors)`` -- per-entry errors
    carry the entry's list index so valid siblings still fly.
    """
    accepted: list[dict] = []
    errors: list[dict] = []
    for i, token in enumerate(raw):
        token = token.strip()
        if _is_handle_token(token):
            errors.append({"index": i, "error": (
                f"{token!r} is a result handle; move takes a staged "
                "handle (vN), a literal fid off the rows, or an [x, y] "
                "world point")})
            continue
        accepted.append({"target_fid": token})
    return accepted, errors


# One factual line on a PREFLIGHT color-column request that attached to no
# row: before landing the prior surface carries no perceived color, so no
# color_<word> column attaches anywhere -- a fact about WHEN color exists,
# stated where the agent reads the rows (the per-stop surface enriches after
# arrival). Keyed on the ATTACH REQUEST, wherever it rode (a find ``color``
# arg, a score_by emphasis key, a filter key) -- never on one op shape.
_PREFLIGHT_COLOR_NOTE = (
    "priors carry no perceived color before landing, so no color column "
    "attaches yet; color enriches per-stop after arrival."
)


def _color_words_of_op(op: dict) -> list[str]:
    """The canonical palette words this op requests a color COLUMN for,
    read off every slot that can name one — a ``filter_items`` key, a
    ``score_by`` emphasis key — so a phase note keys on the attach request
    itself, never on one op shape."""
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
    """Dispatch one ``analyze_scene(program)`` call — the ONE geometry channel.

    Runs the model's typed op-program through the SHARED
    :func:`maps.search.analyze_scene.geom_program.run_program` over the
    ``(cands, anchors)`` surface (``_surface_from_ctx``), then truncates list
    results with ``compact_results``.

    ``raw_sink`` (harness-side, never model-facing): when given, receives the
    PRE-compaction results under ``raw_sink["results"]`` for the carry store
    and the latest enriched rows.

    The namespace's ``staged``/``standing`` thread through to
    :func:`run_program` (the staged-viewpoint side columns ride every find
    result).

    Pure + in-process (no subprocess): the op vocabulary is the only affordance,
    so there is nothing to sandbox. A malformed program returns ``{"error": ...}``
    for the agent to read and retry (self-recovery), never raises into the loop.
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
        if preflight and _program_requests_color(program):
            # The color request errored before any envelope existed (e.g.
            # the emphasis absent-color guard) -- the phase fact rides the
            # error so "no color" reads as WHEN, not broken.
            _err = join_hints(_err, _PREFLIGHT_COLOR_NOTE)
        return {"error": _err}
    except Exception as exc:  # noqa: BLE001  — surface the failure to the agent, don't crash the round
        return {"error": f"program failed: {exc}"}
    # The agent owns the order: everything downstream reads the program's
    # OWN result order; nothing is harness-reordered.
    _raw_results = out["results"]
    if raw_sink is not None:
        raw_sink["results"] = _raw_results
    # ``answer_ref`` names the one table a plan returns; the other ops' hints
    # ride it.
    out["results"] = _compact_results(_raw_results,
                                      widen_form="more_of(<rN>)",
                                      answer_ref=answer_ref)
    if preflight:
        # The pre-takeoff phase note, keyed on what each op's result
        # ACTUALLY holds (never an op shape): a color-column request
        # (emphasis key / filter key) whose raw rows carry NONE of the
        # requested columns states the color-availability fact.
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
    """Build the ``analyze_scene`` ``{cands, anchors, cell_m}`` surface from a
    :class:`ToolContext`'s unified graph (``build_surface_from_graph``).
    Before the first flight ``cands`` is typically empty and ``anchors`` are
    the prebaked priors."""
    from maps.search.analyze_scene.geom_program import build_surface_from_graph

    cell_m = float(getattr(ctx, "cell_size_m", 0.0) or 0.0)
    g = getattr(ctx, "graph", None)
    if g is None or not hasattr(g, "nodes"):
        return {"cands": [], "anchors": [], "cell_m": cell_m}
    cands, anchors = build_surface_from_graph(
        g.nodes.items(), getattr(ctx, "tile_origin_xy", (0.0, 0.0)), cell_m
    )
    return {"cands": cands, "anchors": anchors, "cell_m": cell_m}


def _is_handle_token(token: str) -> bool:
    """True when ``token`` is a result handle -- ``rN`` or a window
    ``rN[...]`` of one. Handles are REFUSED wherever a literal fid/[x,y]
    is wanted (move targets)."""
    return bool(re.fullmatch(r"r\d+(\[[^\]]*\])?", token))


def _crop_token_of(row: dict, live: bool) -> str:
    """A crops token for one carried row: its fid, except a DERIVED end
    row (``end_N`` — the ``ends`` op's mint, never on the live surface)
    crops its own ``[x, y]``, the xy-token form ``_build_crop_requests``
    crops directly. ONLY that family coerces — a once-live fid gone from the surface
    (an evicted detection) must stay on the skip path, because cropping
    its stale carried xy would report a row the surface no longer
    holds."""
    fid = str(row.get("fid"))
    xy = row.get("xy")
    if not live and re.fullmatch(r"end_\d+", fid) and is_xy(xy):
        return format_xy_token(float(xy[0]), float(xy[1]))
    return fid


def _rounded_xy(token) -> list[float] | None:
    """An ``[x, y]`` token -> the point at centimetre precision, else None."""
    p = parse_xy_token(token)
    return [round(p[0], 2), round(p[1], 2)] if p else None


def _pose_xy_of_ctx(ctx) -> tuple[float, float] | None:
    """The current pose carried on a ToolContext (None when absent)."""
    pose = getattr(ctx, "robot_pose", None)
    if pose is None:
        return None
    try:
        return (float(pose.x), float(pose.y))
    except (AttributeError, TypeError, ValueError):
        return None


def _envelope_residue(text: str) -> bool:
    """True when a PROSE turn carries tool-call envelope residue — a parser
    miss (the envelope swallowed into content/reasoning) or the model
    writing the XML by hand: a malformed-envelope repair."""
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
    """True when the whole reply is a BLOCK of the surface -- the model
    wrote its next call into the reply channel.

    A block written there would end the mission silently, short, with the
    flights it was about to make never made.

    Both halves are load-bearing, and each kills a false positive the
    other lets through: prose that narrates a call ("I flew to the church
    with go_to(r5)") does not PARSE WHOLE and is a real ending that must
    not be repaired; real Python parses far more than any surface serves,
    so CARRIES A LIBRARY CALL is what separates a block written into the
    reply channel from a plain reply that happens to parse.
    """
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
    look_audit_dir: str | None = None,
    obs_span_m: float | None = None,
    obs_minor_m: float | None = None,
    boost_queue: "Callable[[list, list], dict]",
    transcript: list[dict] | None = None,
    diag: dict | None = None,
    rid_tag: str = "m0",
    retask: "Callable[[], str | None] | None" = None,
    reasoning_effort: str | None = None,
) -> tuple[MissionReply, Reasoner]:
    """THE mission entry (the CityNav bench path): one conversation.

    Builds the mission Reasoner (live mission system prompt + tools),
    appends the operator's instruction verbatim as the ONE user message,
    and runs the conversation until the model replies with no tool call
    (or a liveness bound fires). A move flies through ``mission_spec.fly``
    the moment it lands; the surface is re-derived from ``ctx``'s live
    graph after every flight. The commit is
    the ENGINE's job — it executed every flight and knows the last target;
    this returns the conversation facts plus the Reasoner (for transcript
    persistence, and for :func:`continue_mission`).

    ``retask`` is the FIELD console's channel and the bench passes none:
    a callable yielding operator text typed since the last turn, drained
    at the top of each turn. With it unset the instruction is the single
    user message.
    """
    reasoner = Reasoner(
        reasoner_url=reasoner_url,
        endpoint_spec=endpoint_spec,
        api_key=api_key,
        seed=seed,
        reasoner_index=0,
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
        look_audit_dir=look_audit_dir,
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
    look_audit_dir: str | None = None,
    obs_span_m: float | None = None,
    obs_minor_m: float | None = None,
    boost_queue: "Callable[[list, list], dict]",
    transcript: list[dict] | None = None,
    diag: dict | None = None,
    rid_tag: str = "m1",
    retask: "Callable[[], str | None] | None" = None,
) -> MissionReply:
    """Another operator turn on a mission that already ended by reply.

    THE FIELD PATH ONLY, and nothing on the bench calls it: a bench
    episode is one conversation and its reply is the end of it. In flight
    the reply is a REPORT -- the authored ISR tasking says "report what
    you see. Do not commit -- await tasking" -- so the console holds the
    session and hands the operator's answer back to the SAME reasoner.

    The conversation carries: this appends ``instruction`` as a user turn
    onto the existing message list, so the model keeps every observation
    the earlier turns folded. The surface is re-derived from ``ctx``,
    which the flights have been mutating all along, so the model reasons
    against the world as it now stands rather than as it was at takeoff.
    """
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
        look_audit_dir=look_audit_dir,
        boost_queue=boost_queue,
        # Takeoff happened on the first turn. Preflight notes on an
        # aircraft already airborne would state a phase it has left.
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
    look_audit_dir: str | None = None,
    boost_queue: Callable[[list, list], dict],
    preflight: bool = False,
    mission_spec: MissionSpec,
    retask: Callable[[], str | None] | None = None,
) -> MissionReply:
    """The mission's turn loop: append the instruction, then serve `run`.

    Appends ``user_message`` as the ONE user turn, then loops native tool
    calls over ``prompts.CODE_TOOLS``. Each ``run`` call carries a block of
    Python; the block executes in one persistent namespace and each library
    call runs on the loop's verbs, so a move inside it flies before the next
    turn is built.

    The model's plain prose reply is the ONE model-owned ending, and three
    repairs stand in front of it so a dropped message is never mistaken for
    one: an EMPTY turn, a reply that is itself a block of code
    (``_code_reply``), and a reply with ZERO flights behind it each get a
    nudge before the second consecutive occurrence is accepted. Envelope
    residue in a prose turn gets the malformed-envelope repair.

    Harness-owned endings, none of which the model hears about:
    ``turn_cap``, ``oversize_rejection``, ``transient_failures``,
    ``request_refused``, ``compaction_infeasible``. ``MissionSpec.flight_budget`` and
    ``deadline_epoch`` stop flights, not the conversation.

    ``retask`` is the FIELD console's channel, drained at the top of each
    turn; the bench passes none, so the instruction is the single user
    message.

    When ``diag`` is provided, ``diag["turns_used"]`` is stamped each turn
    (failure-mode diagnostics; survives every return path).
    """
    # A thread the compactor has declared infeasible stays so for the rest
    # of the episode (the residual head/prose mass it cannot touch only
    # grows), so don't grow it further with a message no dispatch will read.
    thread_infeasible = bool(getattr(reasoner, "compaction_infeasible", False))
    if not thread_infeasible:
        reasoner.append_user(user_message)

    # Programs run through the SHARED ``run_program`` over the (cands,
    # anchors) surface in ``compute_namespace``, so the live path and any
    # offline replay serve byte-for-byte the same surface.
    # The fid's LATEST result-enriched row across the successful programs (a
    # compare/sort output carries the attached columns) — the
    # rejection-escalation surface census reads it to state live facts.
    row_latest: dict[str, dict] = {}
    # Anti-spiral result reuse: successful SELF-CONTAINED programs, keyed by
    # canonical ops JSON (``_program_dedup_key``) -> ``(surface_version,
    # out, turn, ops)``. A repeat under the same surface version gets the
    # one-line dedup note: the re-run cannot say anything new. Anything that
    # changes the world (a flight, a boost, a stance) bumps the version and
    # re-arms every program.
    _programs_seen: dict[str, tuple] = {}
    _surface_version = [0]

    # Protocol telemetry.
    no_call_turns = 0
    no_call_streak = 0
    # CONSECUTIVE replies with ZERO flights behind them (see
    # NO_FLIGHT_ABSTAIN_STREAK). Its own counter rather than the no-call
    # streak: that one counts channel misses, and this reply is on the right
    # channel and is a legal ending.
    no_flight_streak = 0

    # Surface rows by fid (the same rows the programs read). A flight steps
    # the world mid-conversation, so this is REBUILT after every flight and
    # boost; a frozen snapshot would hide every runtime detection from
    # crops and marks.
    _ns_rows: dict[str, dict] = {}

    def _rebuild_ns_rows() -> None:
        # Mutated in place: the closures read this dict. cands win on fid
        # collision (setdefault, cands first).
        _ns_rows.clear()
        for _key in ("cands", "anchors"):
            for _r in (compute_namespace or {}).get(_key) or []:
                if isinstance(_r, dict) and _r.get("fid"):
                    _ns_rows.setdefault(str(_r["fid"]), _r)

    _rebuild_ns_rows()

    # ---- look: the crop path ----------------------------------------------
    # Each crop item resolves its fid to the surface xy (+ footprint/label for
    # the mark overlay) from ``_ns_rows``, clamps to the server's 16-image
    # budget, and hands fully-resolved crop specs to ``look_renderer``. The
    # rendered crops ride a USER turn appended AFTER the turn's tool results
    # (so tool-result id-closure stays contiguous).
    # ONE served tool: `run`, carrying a block of Python.
    #
    # AND NO DECODER MASK. Python is not EBNF-able here: `run` is
    # constrained by `strict: true` on the tool envelope and by the
    # parser's own refusals, which are ordinary tool results the model can
    # act on. A card declaring a decoder dialect would grammar-lock the
    # model out of emitting Python and kill the episode at the first call.
    from maps.search.parse_compile.prompts import served_tools
    stop_tools = list(served_tools())
    pending_vision: list[tuple[str, list[bytes]]] = []
    _block_frames: list[str] = []
    _LOOK_MAX_CROPS = 16
    # How many of a set's rows a MAP badges: the map shares the crop
    # renderer's digit->fid legend contract, so the ceiling is the one
    # look_render already enforces.
    from maps.search.render.look_render import (  # noqa: PLC0415
        LOOK_MAX_BADGES as _MAP_MAX_MARKS,
    )

    # ---- stand: the staged ground viewpoints -------------------------------
    # {handle: {"handle", "from_xy", "up_unit", "at", "facing",
    # "bearing_deg"}}; the LATEST stage is the STANDING viewpoint. One
    # state, three readers: the analyze namespace (executor staged/standing
    # kwargs -> vN point refs + the standing side columns), the crop/map
    # renderers (frame-up), and the move sanitizer (vN targets).
    staged_viewpoints: dict[str, dict] = {}
    standing_handle: list = [None]
    # The mission's flown stops, {stop_id: [x, y]} — a stop id the flight
    # reports serve is a sayable place (stand at it, fly to it, map it).
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
                # a staged handle IS a point: a map centres on it, a stance
                # re-stands there
                return [round(float(sv["from_xy"][0]), 2),
                        round(float(sv["from_xy"][1]), 2)]
            return _rounded_xy(tok)
        return None

    def _stage_apply(at_tok: str, facing_tok: str) -> dict:
        """stand(): mint a vN handle that becomes STANDING. The receipt
        states the columns/refs/render facts the stance turns on. Bumps the
        surface version -- a program re-run under a new stance can say
        something new."""
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
        # a new stance changes what the SAME program returns: thread it
        # into the analyze namespace and re-arm the dedup.
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

    def _look_strs(v: object) -> list[str]:
        return [str(x) for x in v if isinstance(x, str) and x.strip()] \
            if isinstance(v, list) else []

    def _build_crop_requests(
            items: list) -> tuple[list[dict], list[str], list[str]]:
        """Resolve one look's items into per-crop specs off ``_ns_rows``.

        Each item ``{fid, mark_fids?, target_fid?}`` yields one request
        ``{center_fid, center_xy, annotate, marks}`` where ``marks`` are that
        ITEM's resolved footprints ``{xy, boundary, label, is_target}`` (the
        renderer clips each mark to its own frame); providing mark_fids /
        target_fid on an item is what turns its relational overlay on (the
        center identification is unconditional). Centers clamp to the
        16-image server budget; a fid absent from the surface is skipped
        (never a hand-rolled cells->pixel). Returns
        ``(requests, skipped_fids, ref_notes)``."""
        skipped: list[str] = []
        requests: list[dict] = []
        ref_notes: list[str] = []
        # PIXELS ARE EARNED BY FLYING. The sensor holds imagery only where
        # a flight's view square has been; a crop centered on ground no
        # flight has covered refuses toward the verb that earns it. Armed
        # only when the engine states the footprint's SHORT side; the
        # inscribed-circle test is orientation-free and refuses corners it
        # could have served -- the honest direction.
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
                # The server's per-request image cap, stated WHEN IT BITES: a
                # silent drop of the tail would read as "these fids rendered
                # nothing".
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
                # A HANDLE IS A LEGAL MARK, because the thing a candidate
                # is supposed to be ON is a SET: a street is its ways. The
                # renderer drops marks whose footprint never reached the
                # pixels BEFORE the badge cap, so naming a many-way road
                # costs the crop only the ways actually in its window.
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
                        # an [x,y] mark token marks that world point (a
                        # derived crossing/terminus) — no row needed.
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
                # an [x,y] center token crops that world point directly.
                xy = _rounded_xy(f)
            if xy is None:
                skipped.append(f)
                continue
            if not _covered(xy):
                uncovered.append(f)
                continue
            # The crop is centred on this fid; the renderer gets the fid's own
            # label + footprint to bracket-and-label the CENTER subject
            # without occluding it.
            frame_up = None
            frame_note = ""
            if standing_handle[0] is not None:
                # the STANDING viewpoint orients every unframed view; the
                # caption + band state it (never a silent rotation).
                _sv = staged_viewpoints[standing_handle[0]]
                frame_up = list(_sv["up_unit"])
                frame_note = (f"image-up = {_sv['handle']}'s facing "
                              f"(bearing {_sv['bearing_deg']})")
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
                # The observation extent — the renderer's window ceiling (a
                # crop selects less of what was captured, never more). ONE
                # value per mission because the mission flies ONE cruise
                # altitude. On the CityNav orthos the pixel-budget ceiling
                # binds first; this one binds where the frustum is small (a
                # low AGL). test_look_obs_span_ceiling.py and
                # test_start_altitude_span.py pin it.
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

    # ---- show_map: the world-state map path --------------------------------
    def _map_features_from_ctx() -> list[dict]:
        """Flatten the live graph to the map painter's plain feature dicts
        (label from raw OSM tags via the one label rule; a detection's
        label is its detector class; ``kind`` names the paint rule; roads
        carry their centerline and buildings their recorded contour, both
        in world metres, for the stroke/fill ink)."""
        import numpy as np  # noqa: PLC0415
        from maps.perception.viz.osm_palette import (  # noqa: PLC0415
            osm_kind,
            osm_label,
        )
        g = getattr(ctx, "graph", None)
        nodes = getattr(g, "nodes", None) or {}
        _ox, _oy = (getattr(ctx, "tile_origin_xy", None) or (0.0, 0.0))
        _cs = float(getattr(ctx, "cell_size_m", 0.0) or 2.0)

        def _grid_to_world(pts):
            # astype truncates toward zero, like int(): ox + (c + 0.5) * cs
            # per vertex, vectorised.
            a = np.asarray(pts, dtype=float).reshape(-1, 2).astype(np.int64)
            return np.stack([float(_ox) + (a[:, 1] + 0.5) * _cs,
                             float(_oy) + (a[:, 0] + 0.5) * _cs],
                            axis=1).tolist()

        feats: list[dict] = []
        for fid, node in nodes.copy().items():
            # The node's own (N, 2) int array rides as is: the painter takes
            # an array or a list, and converting millions of cells to a list
            # per map would be the map's whole cost.
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
            # The NAME rides beside the class label. `label` is the OSM
            # CLASS ("residential") because that is what the palette and
            # the legend word key off; the name is what the CUE is written
            # in.
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

    # THE FLATTEN IS PER SURFACE VERSION. The graph moves only where the
    # version does (a flight, a boost), so the feature dicts a map paints
    # from are built once per version and every map under it reads them.
    _map_feats: list = [None, None]

    def _map_features_cached() -> list[dict]:
        if _map_feats[0] != _surface_version[0]:
            _map_feats[:] = [_surface_version[0], _map_features_from_ctx()]
        return _map_feats[1]

    def _map_default_span_m() -> float:
        gs = getattr(ctx, "grid_shape", None) or (0, 0)
        cs = float(getattr(ctx, "cell_size_m", 0.0) or 2.0)
        return max(float(gs[0]), float(gs[1])) * cs or 200.0

    def _build_map_requests(items: list) -> tuple[list[dict], list[str]]:
        """Resolve show_map items into render specs. Each item
        ``{around?, mark_fids?}`` centres one map (around absent = the whole
        tile); marks resolve off ``_ns_rows`` exactly like crops. Returns
        ``(specs, skipped_fids)``."""
        specs: list[dict] = []
        skipped: list[str] = []
        gs = getattr(ctx, "grid_shape", None) or (0, 0)
        cs = float(getattr(ctx, "cell_size_m", 0.0) or 2.0)
        ox, oy = getattr(ctx, "tile_origin_xy", (0.0, 0.0))
        tile_center = [round(float(ox) + float(gs[1]) * cs / 2.0, 2),
                       round(float(oy) + float(gs[0]) * cs / 2.0, 2)]
        for it in items:
            if not isinstance(it, dict):
                continue
            around = it.get("around")
            center_fid = ""
            if around is None:
                cxy = tile_center
                span = _map_default_span_m()
                center_label = "map center"
            else:
                cxy = _resolve_stage_point(around)
                if cxy is None:
                    skipped.append(str(around))
                    continue
                center_fid = (str(around).strip()
                              if isinstance(around, str)
                              and str(around).strip() in _ns_rows else "")
                center_label = (look_label(center_fid,
                                            _ns_rows.get(center_fid),
                                            is_target=False)
                                if center_fid else
                                f"[{cxy[0]}, {cxy[1]}]")
                span = _map_default_span_m()
            marks: list[dict] = []
            _seen_mm: set[str] = set()
            _mm_raw = _look_strs(it.get("mark_fids"))
            for mf in _mm_raw:
                if mf in _seen_mm:
                    continue
                _seen_mm.add(mf)
                row = _ns_rows.get(mf)
                xy = look_xy(row)
                if xy is None:
                    skipped.append(mf)
                    continue
                marks.append({"fid": mf, "xy": xy,
                              "boundary": look_outline(row),
                              "label": look_label(mf, row,
                                                   is_target=False),
                              "is_target": False})
            # THE FRAME FOLLOWS THE REQUEST: the marks' and the centre's
            # extent derives the zoom, so a narrowed set renders at a scale
            # that resolves it instead of as a few pixels of the whole tile.
            _cb = (look_outline(_ns_rows.get(center_fid))
                   if center_fid else None)
            _fpts: list = [m["xy"] for m in marks if m.get("xy")]
            for _src in [m.get("boundary") for m in marks] + [_cb]:
                for _b in (_src or []):
                    if isinstance(_b, (list, tuple)) and len(_b) >= 2:
                        _fpts.append([float(_b[0]), float(_b[1])])
            if not _fpts and around is not None:
                # A map around a bare point is a LOCAL ask with nothing to
                # frame from: it gets the same floor a framed set does.
                span = _MAP_FRAME_MIN_M
            elif _fpts:
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
            if standing_handle[0] is not None:
                _sv = staged_viewpoints[standing_handle[0]]
                spec["frame_up"] = list(_sv["up_unit"])
                spec["frame_note"] = (
                    f"image-up = {_sv['handle']}'s facing "
                    f"(bearing {_sv['bearing_deg']})")
            specs.append(spec)
        return specs, skipped

    def _map_vision_text(maps: list[dict]) -> str:
        """The map message's text: what the picture IS (world state, no
        imagery), each map's window + orientation + badge legend, and the
        class->color legend the digest palette needs."""
        order = ", ".join(str(m.get("center_fid") or "map") for m in maps)
        # WHAT THE MAP IS, AND WHAT IT IS NOT. It is the PRIOR -- a survey,
        # drawn from OSM, filled by class -- plus the detections, not a
        # picture of the world as it is now: no zoom shows what the survey
        # never recorded. The camera is the live check, and a flight is how
        # it is taken.
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

    # Identical-error loop breaker, ESCALATING: from the 2nd same-class
    # rejection the error text also states the repetition + the live surface
    # facts (fid counts, columns computed so far), ONCE. From the 3rd
    # CONSECUTIVE one the remedy+census bytes are not re-issued verbatim (a
    # repeated identical byte-string does not break a rejection loop): the
    # escalated form serves the repetition FACT, the rejection's identifying
    # clause, where the full remedy/census already sit, and the compact
    # live-state counts. Facts only, no forcing, no lockout.
    _err_class_counts: dict[str, int] = {}
    _err_streak: dict = {"sig": None, "n": 0}
    # Consecutive identical-dedup streak — the dedup path's own escalation:
    # from the 2nd consecutive hit the note escalates to the changing
    # live-state facts, as _loop_broken_error does for rejections.
    _dedup_streak: dict = {"key": None, "n": 0}

    def _loop_breaker_facts(_ns, _n: int) -> str:
        def _cap(fids: list[str]) -> str:
            return f"{fids[:8]}" + (
                f" (+{len(fids) - 8} more)" if len(fids) > 8 else "")
        c_n, a_n, cols = _surface_census_counts(_ns)
        parts = [
            f" This error class has now occurred {_n}x.",
            # Counts only: a fid sample is noise.
            f" Live surface: {c_n} detection fid(s);"
            f" {a_n} prior map row fid(s).",
        ]
        if cols:
            parts.append(
                f" Columns computed on rows so far: {_cap(cols)}.")
        return "".join(parts)

    def _err_class_sig(err: str) -> str:
        # Same-CLASS signature: strip quoted tokens/numbers so the same error
        # with a different fid/value still matches.
        s = re.sub(r"'[^']*'", "'_'", str(err))
        s = re.sub(r"\d+", "_", s)
        return s[:100]

    def _surface_census_counts(_ns) -> tuple[int, int, list[str]]:
        """Compact live-state counts: n detections, n priors, computed
        columns."""
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
        """The Nth-consecutive-identical-rejection FACT, served INSTEAD of
        the verbatim remedy+census re-issue: the consecutive + total counts,
        the rejection's identifying first clause, where the full remedy text
        and the live-surface census were already served, the
        executor-determinism fact, and the live counts + computed-column
        names. No command, no candidate gate."""
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
            f"returns exactly this rejection. Live now: {c_n} "
            f"detection fid(s), {a_n} prior map row fid(s)"
            + (f"; columns computed on rows: {_cols_s}." if cols else "."))

    def _loop_broken_error(err: str) -> str:
        """Fold the live-surface facts into a repeated same-class error,
        escalating instead of repeating itself verbatim.

        1st serve: the error verbatim. 2nd: + the surface census (fid
        COUNTS + the computed column names — the fix material for an
        absent column), served ONCE in full. 3rd consecutive onward: the
        repetition fact + compact state (``_repeated_rejection_state``)
        replaces the verbatim remedy+census re-issue. A successful program
        resets the CONSECUTIVE streak; the class count keeps accumulating.
        A structural SHAPE error (an unknown ref / a handle used where a
        literal fid is required / a syntax error, whose echoed offending
        line is its own fix material) repeats for program-shape reasons,
        and a fid inventory is noise there — those get only the count."""
        _sig = _err_class_sig(err)
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

    # Cross-call ref persistence: raw results of every successful program,
    # keyed by ref — later calls resolve earlier rN from it (never
    # re-rendered).
    carry_results: dict = {}

    # A HANDLE IS A QUERY, NOT A SNAPSHOT. Every op the mission has run,
    # in order, plus the surface version those results reflect. A flight
    # moves the surface; the next read of any handle re-runs the history
    # against the world the flight folded, so a handle cannot answer about
    # ground that had not been covered when it was bound.
    #
    # Ops are pure reads over (cands, anchors), so replaying them is safe
    # and order-correct by construction: the history IS the dependency
    # order. A replay that raises leaves the old rows standing, because a
    # stale handle beats no handle.
    #
    # THE REPLAY IS PER HANDLE, ON READ: a read of one handle re-derives
    # that handle and the ops it reads, and nothing else (a whole-history
    # replay per flight grows with every block). The ops are pure, so the
    # rows come out the same as a full replay would have made them.
    _op_history: list = []           # every op, in execution order
    _op_reads: list = []             # per op: the handle bases its args name
    _producers: dict = {}            # ref -> indices into _op_history
    # The surface version the carry is ARMED for: the first read under a new
    # version claims it and clears the fresh set, and every read after that
    # re-derives what it reads. Before that first read a handle answers with
    # the rows it held.
    _carry_version = [0]
    _fresh: set = set()              # refs re-derived, or written, under it
    # A FLIGHT-MINTED handle is not a query. It records the rows ONE
    # flight added, which is a fact about an event and cannot re-derive:
    # replaying ops over the new surface would hand "the rows this flight
    # added" every row a later flight added too.
    _event_handles: set = set()

    def _refs_read(op: dict) -> set:
        """The handles an op's args read: every string that is an ``rN``
        or an ``rN[a:b]`` window of one."""
        return set(re.findall(r'"(r\d+)(?:\[[^"\]]*\])?"',
                              json.dumps(op["args"], default=str))) - {op["ref"]}

    def _program_reads(program: dict) -> set:
        """The carry handles a program reads and does not define."""
        ops = program["ops"]
        return set().union(*(_refs_read(o) for o in ops)) \
            - {o["ref"] for o in ops}

    def _ensure_fresh(refs) -> None:
        """Re-derive ``refs`` -- and every op they read -- against the
        current surface, once per surface version. THE trigger: a read
        through here arms the version for every read after it."""
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
            # SEED EVERYTHING ELSE. A flight-minted handle is produced by
            # no op -- it is written straight into carry_results -- so a
            # replay that does not seed it raises "unknown ref" the moment
            # an op reads one, and the flight ack invites exactly that.
            _out = run_geom_program_tool(
                {"program": {"ops": [
                    op for op in _op_history if op["ref"] in need]}},
                namespace=compute_namespace or {}, raw_sink=_sink,
                seed_results={h: v for h, v in carry_results.items()
                              if h not in need})
            if isinstance(_out, dict) and _out.get("error"):
                # An executor refusal is a failed replay too: the handles
                # keep their last rows and stay UNCLAIMED, so the next read
                # tries again.
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
        """Point THIS emission's refs at the handles the first run stored
        under its own labels, op by op -- the emission-index correspondence
        the dedup key judged identity on. A label already live is never
        overwritten: the emission that defines it owns it.
        """
        for prev, cur in zip(prev_ops, cur_ops):
            if cur["ref"] not in carry_results \
                    and prev["ref"] in carry_results:
                carry_results[cur["ref"]] = carry_results[prev["ref"]]

    def _dispatch_analyze_scene(a, _ns=(compute_namespace or {}), *,
                                answer_ref=""):
        _key = _program_dedup_key(a["program"])
        _prev = _programs_seen.get(_key) if _key is not None else None
        if _prev is not None and _prev[0] == _surface_version[0]:
            # ONE-LINE REFERENCE, never a payload and never a lockout: the
            # original's rN handles stay live in carry_results (a new
            # program can slice/chain them), so nothing is locked away —
            # only the re-render is defunded.
            #
            # THE DEDUP OWES THIS EMISSION ITS OWN LABELS. Identity is judged
            # with ref labels NORMALIZED OUT, so the re-issue's labels are not
            # the ones the first run stored under, and a caller resolving a
            # slot from `carry_results[<its own ref>]` would find nothing.
            # Alias by the key's own emission-index correspondence so a
            # deduped program is indistinguishable from a re-run to
            # everything downstream of it.
            _alias_deduped_refs(_prev[3], a["program"]["ops"])
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
                    n=_ordinal(_dedup_streak["n"]), c_n=_c_n, a_n=_a_n,
                    cols=_cols_s)}
            # name the live handles so the reference is actionable
            _prev_out = _prev[1] if isinstance(_prev[1], dict) else {}
            _hs = sorted((k for k in {**_prev_out,
                                      **(_prev_out.get("result") or {}
                                         if isinstance(
                                             _prev_out.get("result"),
                                             dict) else {})}
                          if re.fullmatch(r"r\d+", str(k))),
                         key=lambda h: int(h[1:]))
            return {"note": _DEDUP_NOTE.format(
                turn=_prev[2],
                handles=f" ({', '.join(_hs)})" if _hs else "")}
        _raw_sink: dict = {}
        # Re-derive every handle this program reads against the CURRENT
        # surface first, so it reads the world as it is, not as it was.
        _ensure_fresh(_program_reads(a["program"]))
        out = run_geom_program_tool(a, namespace=_ns, raw_sink=_raw_sink,
                                    seed_results=carry_results,
                                    preflight=preflight,
                                    answer_ref=answer_ref)
        if isinstance(out, dict) and "error" in out:
            out = {**out, "error": _loop_broken_error(str(out.get("error")))}
        if isinstance(out, dict) and "error" not in out:
            # a fresh successful program ends the CONSECUTIVE-rejection
            # streak: the next same-class error serves its full text again.
            _err_streak["sig"] = None
            _err_streak["n"] = 0
            _dedup_streak["key"] = None
            _dedup_streak["n"] = 0
            if _key is not None:
                # Keyed to the surface version; the turn label feeds the
                # one-line reference an identical re-issue gets back, and
                # the ops let a later dedup hit alias its own labels onto
                # the handles this run stored (_alias_deduped_refs).
                _programs_seen[_key] = (_surface_version[0], out, turn_idx,
                                        a["program"]["ops"])
            carry_results.update(_raw_sink.get("results") or {})
            _fresh.update(_raw_sink.get("results") or {})
            for _op in a["program"]["ops"]:
                _producers.setdefault(_op["ref"], []).append(len(_op_history))
                _op_history.append(_op)
                _op_reads.append(_refs_read(_op))
            # Latest enriched row per fid (raw, pre-compaction) — the
            # rejection-escalation surface census reads it.
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
    outage_since: float | None = None
    # The bench's budget. A LIVE mission (the reasoner carries a
    # cancel_event) has none: the arm going down mid-flight is a link to
    # bring back, not a verdict on the run, so the loop holds its turn
    # until the reasoner answers or the operator presses Stop.
    MAX_ROUND_TRANSIENT = 6

    # Mission-path local state + the ONE mission-end constructor. The
    # mission's model-owned ending is the plain reply; everything else here
    # is a harness liveness bound, named truthfully in ``ended_by``.
    mission_flights = 0
    _tr_stage = "mission"
    mission_sam: list[dict] = []

    # THE SET IN SCOPE (the last non-empty result, or what the last flight
    # added), the paging ledger (executor ref -> rows already asked past,
    # written only by a call that executed), and the mission-global rN chain
    # that carry_results resolves against.
    _those: list = [None]
    _widened: dict = {}
    plan_rn: list = [0]
    # THE EPISODE'S SCRIPT, append-only: each accepted block in order;
    # nothing already executed is ever re-run (that would re-mint handles
    # and diverge from what the model was shown).
    _episode_script: list = []
    # EVERY rN THE CODE SURFACE HAS IN PLAY: the handles earlier blocks
    # bound, plus every rN the block now running spells. Both writers into
    # the rN namespace -- the compiler and the flight mint below -- skip
    # them. The dispatcher refreshes it per block.
    _code_names: set = set()
    # The handles flights minted DURING the call being dispatched, so the
    # code scope can bind them: the flight's report names the handle, and a
    # name the surface serves has to be a name the surface can resolve. The
    # surface drains it after every call.
    _code_minted: list = []
    # The fids explore passes have covered this mission, for the named-
    # feature coverage note below.
    _covered_fids: set = set()

    def _watching_trailer() -> list:
        """The armed watchlist WITH each class's runtime detection count.

        The count is the missing fact behind the prior-graph-as-world
        inference: a find over a watched class returns only OSM priors
        until a flight covers candidate ground, and "pool (0 detected)"
        on every receipt is what says the pixels have not been asked yet.
        Counted live off the candidate surface (detections fold into
        ``cands`` in place), so it is a measurement, never a cache.
        """
        counts: dict[str, int] = {}
        for _r in (compute_namespace or {}).get("cands") or []:
            if isinstance(_r, dict) \
                    and str(_r.get("fid") or "").startswith("detected_"):
                _c = str(_r.get("class") or "?")
                counts[_c] = counts.get(_c, 0) + 1
        return [f"{c['class']} ({counts.get(c['class'], 0)} detected)"
                for c in mission_sam]

    def _mint_rows(_new_fids: list, _what: str):
        """The rows a flight or a pass ADDED -> an ordinary rN handle.

        ``(handle, the note that names it)``, or ``None`` when nothing was
        added. ZERO new rows mints NOTHING and leaves "those" standing: a
        live-looking empty handle would read as a real empty answer.
        """
        if not _new_fids:
            return None
        # THE MINT SKIPS THE NAMES IN PLAY, as the compiler does: a flight
        # taking `r3` while the model holds r3 for something else would name
        # one handle for two sets.
        while f"r{plan_rn[0]}" in _code_names:
            plan_rn[0] += 1
        _h = f"r{plan_rn[0]}"
        plan_rn[0] += 1
        _code_names.add(_h)
        _code_minted.append(_h)
        carry_results[_h] = [_ns_rows[_f] for _f in _new_fids]
        _event_handles.add(_h)
        _fresh.add(_h)
        # bound NOW so a later refusal still leaves the scope on what was
        # shown, and re-applied when the call finishes.
        _those[0] = _h
        return _h, (f"{_h} — the {len(_new_fids)} rows this {_what} added; "
                    f"the handle is bound, so {_h}, {_h}[0] and "
                    f"{_h}.order(...) all reach them")

    def _fly_moves(_accepted: list, _item_errors: list,
                   cover: str | None = None) -> dict:
        """The move core — each entry FLIES the moment it lands, in order;
        after every successful flight the surface re-derives from the live
        graph, and the rows the flight ADDED are minted as an ordinary rN
        result handle.

        ``cover`` names the place an ``explore`` is covering, and turns the
        SAME loop into ONE pass over it. Every entry is a stop and the pass
        ends on the last of them. Three things make a pass one act rather
        than N flights:

          * NO per-stop view: a dozen images in one turn would spend a large
            share of the mission's token budget. The pass shows its catch
            instead (see ``_cover``).
          * ONE handle, over everything the whole pass folded.
          * ONE ack. The pass is an act; N flight reports are its
            mechanism, not its result.
        """
        nonlocal mission_flights, preflight
        _reports: list[dict] = []
        # The pass's rows are what it added between its first stop and its
        # last, so the diff is taken ONCE around the whole of it.
        _cover_before = set(_ns_rows) if cover is not None else None
        _clock_cut = False

        def _stop_xy(_it):
            _t = str(_it.get("target_fid") or "")
            return parse_xy_token(_t) or look_xy(_ns_rows.get(_t) or {})

        _queue = list(_accepted)
        while _queue:
            _pick = 0
            if cover is not None and len(_queue) > 1 and mission_spec.held_now is not None:
                # GREEDY ON THE PASS: the crew fly the whole sweep in one
                # go, so a later stop is often held before the first is. A
                # stop the recording holds is taken now, whatever its place
                # in the model's order; the rest keep theirs.
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
                # THE REST OF THE PASS RIDES EVERY STOP, for the crew: the
                # station posts each leg with the stops behind it, so the
                # whole pass can be flown in one go. The bench's fly reads
                # ``target_fid`` alone, and nothing of this reaches the
                # model.
                _item["pass_ahead"] = [[float(_xy[0]), float(_xy[1])]
                                       for _xy in map(_stop_xy, _queue) if _xy]
            if mission_flights >= mission_spec.flight_budget:
                # The backstop fires without a NUMBER: the model never
                # hears a budget fact; the count logs.
                log.warning("%s flight backstop hit (%d flights)",
                            rid_tag, mission_spec.flight_budget)
                _item_errors.append({"target_fid": _item["target_fid"],
                                     "error": "not flown."})
                continue
            if (mission_spec.deadline_epoch is not None
                    and time.time() >= mission_spec.deadline_epoch):
                # THE SAME BACKSTOP, IN THE UNIT THAT KILLS: per-stop cost
                # moves with tile density, watchlist width and the serving
                # plane's health, so a count of stops cannot bound seconds.
                #
                # Stopping the FLYING is the whole of it: the model keeps
                # its remaining turns and ends the mission with its own
                # commit, so the episode is scored instead of lost. Nothing
                # here reaches the served bytes -- the model reads the same
                # "not flown." the flight arm writes, and the clock goes to
                # the log only.
                #
                # ponytail: checked BETWEEN stops, so one pathological stop
                # still overruns by its own duration; bounding a single
                # stop's cost is what shrinks that residual.
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
                # SNAP a point that names a known row: flying is
                # identical; the fid keeps its identity.
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
                    # the watchlist gate is mechanical and invisible, so
                    # state the mechanism.
                    _wl_note = ('the watchlist is empty -- nothing is '
                                'segmented on any flight; call '
                                'look_for("<class>") to see detections')
                    _report["note"] = (f"{_report['note']} {_wl_note}"
                                       if _report.get("note")
                                       else _wl_note)
                # THE FLIGHT SHOWS WHAT IT SAW. A count of the watched
                # classes reads "0 detections" as absence when it means
                # "nothing you asked for"; the view the flight saw is the
                # evidence. Its caption carries the world window, so what
                # the model sees converts straight into a point.
                #
                # THE VIEW IS THE TARGET'S OWN LOOK: centred on the row the
                # flight asked for, its footprint outlined, and nothing
                # about where the aircraft was. It is the one picture that
                # reaches the model before its next aim without costing a
                # turn. A shape too big for its window is clipped to the
                # in-frame arcs (the caption's ``subject_outline_cut``). A
                # target that is not a row (a stop flown again) centres on
                # the stop's point under the token it was asked for.
                #
                # A pass renders no per-stop view — see ``cover``.
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
                        # AUDIT: a flight view is a picture the model was
                        # SHOWN, so it is persisted like any other: a
                        # caption cannot state a mark's footprint, and a
                        # detected_* mark's mask lives in the runtime object
                        # graph and nowhere on disk.
                        if look_audit_dir and _shown:
                            _persist_look_audit(
                                audit_dir=look_audit_dir, stop_id="",
                                step_idx=-1, rid_tag=rid_tag,
                                cid=f"flight_{_report.get('stop_id') or ''}",
                                annotate=False,
                                tool_ack={"flight_view": 1,
                                          "stop_id": _report.get("stop_id")},
                                vision_text=_fv_text, crops=_fv)
                mission_flights += 1
                _sid = _report.get("stop_id")
                _pxy = _report.get("pose_xy")
                if (_sid and isinstance(_pxy, (list, tuple))
                        and len(_pxy) == 2):
                    _plan_stops[str(_sid).strip().lower()] = [
                        float(_pxy[0]), float(_pxy[1])]
                # The world stepped: refresh BOTH surface views in
                # place, re-key the dedup, leave the takeoff phase.
                (compute_namespace or {}).update(_surface_from_ctx(ctx))
                _rebuild_ns_rows()
                _surface_version[0] += 1
                # A VIEW MINTS THE SET IT SHOWS. The flight's own new rows
                # become an ordinary rN handle, so order_by(rN, ...) and
                # look(rN) reach them while the drone is still standing over
                # them. The surface diff, not the engine's detection count:
                # a re-sighting merges into a row that already exists, and
                # the addressable rows are the truth here. A PASS mints
                # once, around the whole of itself.
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
        """One explore's N flight reports -> the ONE act they were.

        EVERY report is a stop, and the pass ends on the last of them. SR
        reads ``final_pos_xy`` (benchmarks/citynav/eval.py) -- the last
        flight's target -- so an explore commits to where its walk
        finished, and the ack says so rather than leaving the model to
        infer it: the standing answer is the pass's last stop over the
        thing explored, never the thing itself, and no coordinate is
        served for it. Naming the row it meant and going to it is how
        that is undone; the ack's own handle is what makes the rows
        reachable.
        """
        _ok = [r for r in _reports
               if isinstance(r, dict) and "error" not in r]
        _stops = [r for r in _reports
                  if isinstance(r, dict) and r.get("pose_xy")]
        _ack: dict = {"explored": cover, "stops": len(_stops)}
        if _stops:
            # THE STANDING ANSWER, stated as fact: an explore that is not
            # followed by a go_to answers with its own last stop, which the
            # model must be able to read, not deduce.
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
            # A pass that could not fly is not a pass that found nothing.
            _ack["not_flown"] = _errs[:4]
        if _item_errors:
            _ack["rejected"] = _item_errors
        if not _ok:
            return {"error": (_errs[0] if _errs
                              else f"nothing of {cover} could be flown")}
        # A PASS SHOWS ITS CATCH, not its coverage: a wide frame of the
        # whole pass resolves nothing, so no coverage frame is drawn HERE;
        # the explore branch serves one crop per row the pass ADDED
        # (_CATCH_CROPS of them, pass order).
        return _ack

    # HALF THE OBSERVATION'S LONG SIDE: a pass's stop spacing.
    # ``observation_extent_m`` returns the sensor footprint's LONG side, so
    # half of it sits under the short side for any sensor below 2:1 -- which
    # is what lets a ONE-DIMENSIONAL walk cover a shape that is not one:
    # consecutive stops overlap on the minor axis too. The number is the
    # run's own: a bake or an altitude change moves it.
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
        """One explore -> ``(move items, note, subject, segments)``.

        The items are the stops over the subject's own footprint, in shape
        order, and NOTHING after them: the pass ends where it finished.
        SR is scored on ``final_pos_xy``, so an explore commits to its last
        stop -- which is the honest cost of covering something, paid in
        where you end and undone by naming the row you meant and going to
        it. A return leg would spend a flight putting the drone somewhere
        that is not an answer either.

        The note carries HOW MUCH of the shape the stops reached. The
        stops run end to end along the shape, so the walk's own length is
        its reach and each stop is credited with one observation of it --
        a pass the engine's stop bound truncated reads as truncated
        instead of as "that is all there was".

        A SET is one place to cover: many members become ONE pass over
        their union, per-member walks chained end to end so the whole
        thing flies a line.
        """
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
                # A THING SMALLER THAN ONE OBSERVATION is covered by
                # standing in the middle of it, and saying "covered 0 m"
                # about that reads as a failure. One stop is the honest
                # whole of the pass, not a short one.
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
            # NOTHING TO COVER, and the honest pass is the one stop it is:
            # a bare point has no footprint, a fid may carry no cells, and
            # without an observation extent there is no spacing that is
            # not invented. It still gives the pose back.
            _note = (f"{_tok} has no footprint to cover, so this pass is "
                     "the one stop it is.")
            _items = [dict(_item)]
        # A NAMED FEATURE IS OFTEN MANY SEGMENTS, and a pass covers ONE.
        # Which segments no pass has covered is world-state, so the pass
        # note says it; the model decides whether to keep covering.
        # Computed HERE, before the flight refreshes the surface views.
        _covered_fids.add(str(_tok))
        _nm = (_ns_rows.get(str(_tok)) or {}).get("name")
        if _nm:
            _note += _uncovered_segments(_nm)
        return _items, _note, _tok, 1

    def _cover_set_items(_accepted_items: list, _said):
        """The multi-member half of ``_cover_items``: one pass over the
        union.

        Per member the stops are the SAME walk a single explore flies
        (``coverage_stops``); the pass then chains members greedily by
        nearest endpoint from the live pose, reversing a member whose far
        end is nearer, so consecutive members join where they touch and
        the whole pass flies a line, not a scribble. Stops landing within
        half a stop-spacing of one already kept are dropped -- that is the
        shared endpoint of consecutive street segments, seen twice.
        """
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
                # a point member, or a fid with no cells: its pass is the
                # one stop it is (the fid flies itself); its recorded
                # position anchors the chain when one exists.
                _xy = ((_ns_rows.get(_tok) or {}).get("xy")
                       if _pt is None else list(_pt))
                _p = ([float(_xy[0]), float(_xy[1])]
                      if isinstance(_xy, (list, tuple)) and len(_xy) == 2
                      else None)
                _members.append({"tok": _tok, "pts": [_p] if _p else [],
                                 "items": [dict(_it)], "walk": 0.0})
        # greedy nearest-endpoint chain, seeded from where the drone is
        _cur = _pose_xy_of_ctx(ctx)
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
            # A MEMBER SMALLER THAN ONE OBSERVATION WALKS ZERO METRES, so
            # length is the wrong fact for a set of footprints; the count is
            # the right one.
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
        """look_for() — the ONE segmenter channel (latest declaration of a
        class wins), reached by the bare form directly and by the placed
        form on its way to the boost. The watchlist only GROWS: a removal
        buys a cheaper segmenter pass, never an answer."""
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
            for n in (getattr(getattr(ctx, "graph", None), "nodes", None)
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
            # THE PLACED FORM MEANS WHAT A PLANNER MEANS: standing attention
            # on the class, with localized emphasis where it was asked. One
            # verb, one semantics: the model's own call names the class, so
            # arming it is the call's meaning -- nothing arms that the model
            # did not say.
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
            # a retro boost ingests detections NOW: refresh the retrieval
            # surface like a flight does, or the new fids are findable
            # only after the next flight.
            (compute_namespace or {}).update(_surface_from_ctx(ctx))
            _rebuild_ns_rows()
            _surface_version[0] += 1
        if res.get("error"):
            return _plan_err({"boost": res}, str(res["error"]))
        return {"boost": res}

    def _crops_of(fid, cid: str, marks: list | None = None,
                   preamble: str = "") -> dict:
        # A LIST asks for one crop PER ROW -- each its own subject, each at
        # the single-look scale where a distinction is resolvable.
        _fids = [str(f) for f in fid] if isinstance(fid, (list, tuple)) \
            else [str(fid)]
        # MARKS RIDE EVERY CROP: what marks= is FOR is telling candidate
        # crops apart.
        reqs, skipped, ref_notes = _build_crop_requests(
            [{"fid": f, "target_fid": f,
              **({"mark_fids": list(marks)} if marks else {})}
             for f in _fids])
        ack: dict = {"looked": len(reqs)}
        if skipped:
            # deduped: one crop PER ROW asks per fid, so a row gone from the
            # surface would otherwise be reported once for every request that
            # named it.
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
        # The preamble rides the message the PIXELS are in, not only the
        # tool result: a badged frame whose contents cannot be named
        # afterwards is worth much less than one whose badges are the rows
        # of a handle the model already holds.
        vis_text = preamble + _look_vision_text(crops)
        ack["delivered"] = len(jpegs)
        if jpegs and withheld("images"):
            ack["delivered"] = 0
            ack["note"] = _PICTURE_WITHHELD
            pending_vision.append((vis_text, []))
        elif jpegs:
            ack["note"] = "the crop follows in the next message"
            _deliver_frames(jpegs, _block_frames)
            pending_vision.append((vis_text, jpegs))
            if marks:
                # DRAWN, not requested, and only on a crop that SHIPPED.
                # The renderer CLIPS at the window: a mark outside the
                # delivered frame is named in the legend text and no badge
                # is drawn for it. The legend IS the drawn set, so count it.
                ack["badged"] = sum(
                    len(c.get("legend") or []) for c in crops
                    if isinstance(c, dict) and c.get("jpeg"))
                # A requested mark goes undrawn for FOUR different reasons
                # and only ONE of them is distance, so serve the renderer's
                # own out-of-frame list rather than requested-minus-badged
                # (a surface eviction is ``skipped``, already on this ack).
                ack["not_drawn"] = [
                    str(f) for c in crops
                    if isinstance(c, dict) and c.get("jpeg")
                    for f in (c.get("marks_out_of_view") or [])][:20]
        else:
            ack["error"] = ("render returned no image bytes; no crop "
                            "will follow")
        if look_audit_dir:
            _persist_look_audit(
                audit_dir=look_audit_dir, stop_id="",
                step_idx=-1, rid_tag=rid_tag, cid=cid,
                annotate=True, tool_ack=ack, vision_text=vis_text,
                crops=crops)
        return ack

    def _crops_set(handle: str, cid: str, marks: list | None = None) -> dict:
        """look(rows) — one crop per row of the handle, in the handle's own
        order, each badged with the row it shows.

        Every badge is a row the model can still name afterwards (its fid
        rides the legend text, and the handle reaches the whole set again);
        that correspondence is SERVED on the message the pixels are in. Any
        handle qualifies, so a set minted by something other than a find
        looks the same way. The crop budget is stated only WHEN IT BITES:
        a silent drop would read as "those rows are not there"."""
        rows = [r for r in (_carry_read(handle) or [])
                if isinstance(r, dict) and r.get("fid")]
        if not rows:
            return {"error": (f"{handle} holds no rows to look at; name a "
                              "handle that has some, or find() one first")}
        # ONE CROP PER ROW, not one frame over the set: a frame over a
        # spread set renders at map scale and resolves nothing, while a
        # single-row crop is the one every flight ships. One badge per
        # image makes badge->fid unambiguous by construction.
        fids = [_crop_token_of(r, str(r["fid"]) in _ns_rows) for r in rows]
        ack = _crops_of(
            fids, cid, marks=marks,
            # The preamble names no handle: ``handle`` is the EXECUTOR's
            # ref, which the model may never have bound. Each crop names its
            # own row, and naming a row is what the next call needs.
            preamble=("One crop per row, in order, each badged with the "
                      "row it shows. "))
        if not withheld("images") and len(rows) > int(ack.get("delivered") or 0):
            ack.setdefault("ref_notes", []).append(
                f"{len(rows)} rows in {handle}: this call shows the first "
                f"{ack.get('delivered')} in the standing order; slice the "
                "handle to see the rest")
        return ack

    def _map_of(fid: str, cid: str, mark_fids: list | None = None) -> dict:
        _req: dict = {"around": fid}
        if mark_fids:
            _req["mark_fids"] = list(mark_fids)
        specs, m_skipped = _build_map_requests([_req])
        ack: dict = {"maps": len(specs)}
        if m_skipped:
            ack["skipped"] = m_skipped[:20]
        if not specs:
            ack["error"] = "no resolvable map center"
            return ack
        try:
            from maps.search.render.map_render import (  # noqa: PLC0415
                render_map_from_spec,
            )
            _feats = _map_features_cached()
            maps = [m for m in (
                render_map_from_spec(
                    sp, _feats,
                    tile_origin_xy=getattr(ctx, "tile_origin_xy",
                                           (0.0, 0.0)),
                    cell_size_m=float(getattr(ctx, "cell_size_m", 0.0)
                                      or 2.0),
                    grid_shape=tuple(getattr(ctx, "grid_shape", (0, 0))))
                for sp in specs) if m is not None]
        except Exception as exc:  # noqa: BLE001 — render never crashes the mission
            log.warning("%s SAY map render failed: %s", rid_tag,
                        _summarise_exc(exc))
            return {"error": f"map render failed: {_summarise_exc(exc)}"}
        jpegs = [m["jpeg"] for m in maps if m.get("jpeg")]
        vis_text = _map_vision_text(maps)
        ack["delivered"] = len(jpegs)
        if jpegs and withheld("images"):
            ack["note"] = ("the map's caption follows in the next message; "
                           "its picture is not served on this run")
            pending_vision.append((vis_text, []))
        elif jpegs:
            ack["note"] = "the map follows in the next message"
            _deliver_frames(jpegs, _block_frames)
            pending_vision.append((vis_text, jpegs))
        else:
            ack["error"] = ("render returned no image bytes; no map "
                            "will follow")
        if look_audit_dir:
            _persist_look_audit(
                audit_dir=look_audit_dir, stop_id="",
                step_idx=-1, rid_tag=rid_tag, cid=cid,
                annotate=False, tool_ack=ack, vision_text=vis_text,
                crops=maps)
        return ack

    def _plan_err(ack: dict, msg: str) -> dict:
        """A plan error result, routed through the SAME escalating loop
        breaker the analyze path uses."""
        return {**ack, "error": _loop_broken_error(str(msg))}

    def _flown_stop(tok: str) -> tuple[str, str | None]:
        """A flown stop id flies its recorded point; an unflown stop id is
        refused, naming the stops that have flown. ``(token, error)``."""
        _sp = _plan_stops.get(tok.strip().lower())
        if _sp is not None:
            return format_xy_token(_sp[0], _sp[1]), None
        if re.fullmatch(r"stop_[0-9]{1,4}", tok.strip().lower()):
            _known = ", ".join(sorted(_plan_stops)) or "none yet"
            return tok, (f"{tok} has not been flown; the flown stops are: "
                         f"{_known}")
        return tok, None

    def _closest_names(what: str) -> str:
        """The nearest NAMED rows to a grounding that matched nothing
        (fuzzy, top 3)."""
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

    # THE CALL IN FLIGHT: what _ground_names noted and what _analyze
    # answered, read by _finish. _ground_names opens every call.
    _call: dict = {}

    def _ground_names(names: dict) -> dict:
        """Open a call and ground its NAMES (``{name: why}``, first use
        first) under the name-match floor: ``{"ties": {name: top-tie
        fids}}``, or the first refusal. One plumbing find per name, raw,
        never dedup'd, never rendered."""
        _call.clear()
        _call["notes"] = notes = []
        ties: dict = {}
        if not names:
            return {"ties": ties}
        _sink: dict = {}
        g_out = run_geom_program_tool(
            {"program": {"ops": [
                {"ref": f"r{i}", "op": "find",
                 "args": {"provenance": "any", "query": q.lower()}}
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
            # the same floor the find head's retrieval uses: a below-
            # threshold "best match" is a guess
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
            # every name keeps only the TOP name_match tie group (anchors:
            # the labeled union); the best match leads it
            _top = rows[0].get("name_match")
            ties[q] = [str(r["fid"]) for r in rows
                       if r.get("name_match") == _top] \
                or [str(rows[0]["fid"])]
            if why == "stage" and len(ties[q]) > 1:
                # rule F refuses only a genuine TOP-MATCH tie
                _f = ", ".join(ties[q][:4])
                return _plan_err({}, (
                    f"rule F: a viewpoint is one point; "
                    f"\"{_what}\" has {len(ties[q])} equally good "
                    f"name matches ({_f}); say which one — a fid "
                    "works"))
            # A token NO matched row carries -- the same note the query find
            # serves, on the grounding path, so glue in a grounding does not
            # bind silently. A NOTE, never an error and never a filter: a
            # real query often names an attribute no row spells.
            _dead = _unmatched_query_tokens(q.lower(), rows)
            if _dead:
                notes.append(
                    f"no row matching \"{_what}\" carries "
                    + ", ".join(repr(t) for t in _dead[:4])
                    + " -- those words did not narrow anything")
            # a covered name skips this note: the pass flies EVERY top-tie
            # segment, so "flying the best name match" would state the thing
            # it just stopped being true of
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
            # Paging a handle with nothing left to page -- the window walked
            # past the end, or the handle was never truncated -- is refused
            # HERE, where the rows are; a refusal leaves the ledger alone.
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
            # keep the mechanical fact, drop the op-signature tail (it
            # teaches op names this surface cannot spell)
            return {"error": str(out["error"]).split("; signature: ")[0]}
        plan_rn[0] = max(plan_rn[0], int(rn_end or 0))
        # THE COLUMN THIS CALL ORDERED BY -- the last score_by's leading
        # emphasis -- so _finish can say when the ordering did not separate
        # anything (every house on a 4 km road is 0 m from it).
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
        # a flown stop is a place: fly its recorded point (the xy→fid snap
        # keeps a row's identity if one stands there)
        tok, _stop_err = _flown_stop(tok)
        if _stop_err:
            return _plan_err({}, _stop_err)
        _accepted, _item_errors = _mission_sanitize_moves([tok])
        mv = _fly_moves(_accepted, _item_errors)
        _ok = [r for r in mv.get("flown") or []
               if isinstance(r, dict) and "error" not in r]
        if not (_ok and _accepted):
            # nothing flew: the failure is THE result, top-level, never a
            # flight-shaped envelope that never flew
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
        """AN EXPLORE IS ONE ACT MADE OF FLIGHTS, expanded HERE because
        none of it exists at compile time: the stops come off each place's
        own cells and the chain off the live pose. A SET is one place to
        cover, and its members arrive three ways -- a NAME's whole top tie
        group (a list), a result handle's fid-bearing rows, a token."""
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
        # A TRUNCATED PASS READS AS TRUNCATED: ``_cov_note`` counts the stops
        # planned, ``_cover_ack`` the stops flown.
        if mv.get("stops", len(_accepted)) < len(_accepted):
            _cov_note = (f"the pass ended after "
                         f"{mv.get('stops', 0)} of its stops.")
        if _cov_note:
            mv["note"] = (f"{_cov_note} {mv['note']}"
                          if mv.get("note") else _cov_note)
        # THE PASS SHOWS ITS CATCH: one crop per row it added, in the pass's
        # own (spatial) order, capped at _CATCH_CROPS so the model's own
        # later looks keep room.
        _h = _code_minted[-1] if _code_minted else None
        _rows = [r for r in (carry_results.get(_h) or [])
                 if isinstance(r, dict) and r.get("fid")] if _h else []
        _catch = _rows[:_CATCH_CROPS]
        if _catch:
            _cack = _crops_of(
                [_crop_token_of(r, str(r["fid"]) in _ns_rows)
                 for r in _catch], cid,
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
        # the marks a crop request carries: only real tokens
        _mk = [m for m in marks or [] if isinstance(m, str) and m.strip()]
        crop = (_crops_set(str(of), cid, marks=_mk or None) if of
                else _crops_of(fid, cid, marks=_mk or None))
        if crop.get("error") and not crop.get("delivered"):
            return _plan_err({"crop": crop}, str(crop["error"]))
        return {"crop": crop}

    def _map(around: str) -> dict:
        # THE MAP BADGES THE SET IN SCOPE. Unmarked, every detection paints
        # as one white class (map_render's _DETECTION_RGB), so the picture
        # could not answer "which of these rows is it" -- the one question a
        # narrowed set is asking.
        _h = _those[0]
        _mrows = [r for r in (_carry_read(_h) or [])
                  if isinstance(r, dict) and r.get("fid")] if _h else []
        _mfids = [str(r["fid"]) for r in _mrows[:_MAP_MAX_MARKS]]
        mp = _map_of(around, cid, mark_fids=_mfids)
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
        # AN ORDERING THAT DID NOT ORDER ANYTHING. A big prior is a big
        # relation: every row along a many-way road is 0 m from it, so rank
        # 0 can mean nothing. It is a FACT, not advice: how many rows tie,
        # and on what column; narrowing further is the model's to choose.
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
        # a call that executed ends the consecutive-rejection streak (mirror
        # of the analyze path's reset)
        _err_streak["sig"] = None
        _err_streak["n"] = 0
        if _call.get("page"):
            _widened.update([_call["page"]])
        # THE SET IN SCOPE: the call's answer, NON-EMPTY only -- an empty
        # tail must not take the slot, or the rows it failed to reach become
        # unreachable. A flight this call made owns it over that: a call
        # that both builds a set and flies ends standing over what the
        # flight revealed, and the set it built keeps its own handle.
        if _call.get("answer") and _call.get("rows"):
            _those[0] = _call["answer"]
        if _code_minted:
            _those[0] = _code_minted[-1]
        return ack

    _mission_py: list = [None]

    def _dispatch_run(args, cid: str) -> dict:
        """One code block, run in the mission's ONE persistent namespace on
        the verbs above. An error costs the line it is on: everything the
        block ran before it stands, receipts included."""
        code = (args or {}).get("code")
        if not isinstance(code, str) or not code.strip():
            return {"error": "run takes one block of code in code"}
        if _mission_py[0] is None:
            _mission_py[0] = _py_surface.MissionPython(_py_surface.MissionEnv(
                # THE CODE ARM READS THE FLOWN WORLD. Python evaluates
                # go_to(red_cars[0])'s subscript BEFORE go_to dispatches,
                # and len(), print() and iteration read rows the same way,
                # so a read re-derives the handle against the current
                # surface first.
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
        # The names no OTHER writer may mint: every rN this block spells
        # plus every ref a variable already holds, so a flight minting a
        # result handle never collides with a name the model is using.
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

    def _mission_end(ended_by: str, reply_text: str = "",
                     turns: int = 0) -> MissionReply:
        if transcript is not None:
            transcript.append({"stage": "mission", "kind": "mission_end",
                               "ended_by": ended_by, "turns": int(turns)})
        return MissionReply(
            ended_by=ended_by, reply_text=reply_text, turns_used=int(turns),
            flights_flown=mission_flights, no_call_turns=no_call_turns,
        )

    if thread_infeasible:
        # No dispatch: re-scanning a known-infeasible thread costs a
        # full-transcript tokenize and re-raises every time.
        return _mission_end("compaction_infeasible")

    for turn_idx in itertools.count():
        # Failure-mode diagnostics: stamp turns-used each iteration so it
        # survives whichever return point fires.
        if diag is not None:
            diag["turns_used"] = turn_idx + 1
        # THE OPERATOR'S CHANNEL INTO A FLYING MISSION. A field console
        # hands over a callable yielding whatever the operator typed since
        # the last turn; each string lands as a plain user turn and the
        # model re-decides against it on this turn's call. Drained at the
        # TOP of a turn so a re-task never splits a turn's tool calls from
        # their results. The transcript row is the console's, written when
        # the operator spoke, not when the loop heard them.
        #
        # The bench passes nothing: ``retask is None`` appends not one
        # byte, so the operator instruction stays the single user message.
        if retask is not None:
            while (operator_turn := retask()) is not None:
                log.info("%s operator re-task at turn %d (%d chars)",
                         rid_tag, turn_idx, len(operator_turn))
                reasoner.append_user(operator_turn)
        if turn_idx >= mission_spec.turn_cap:
            # The mission's only harness-owned ending: nothing but the
            # reply terminates, so a conversation that never replies needs
            # a liveness bound. The engine commits whatever flew.
            log.info("%s mission turn cap (%d) reached", rid_tag,
                     mission_spec.turn_cap)
            return _mission_end("turn_cap", turns=turn_idx)
        try:
            r = reasoner.step_tool_call(stop_tools)
        except (CompactionInfeasible, CompactionInvariantError) as exc:
            # CompactionInvariantError is a sibling of Infeasible, not a
            # subclass; both mean this conversation cannot be served, so
            # both end it the same way. Terminal for the EPISODE: nothing
            # removes the residue the compactor cannot touch, so flag the
            # thread (a later turn on it returns at once, above) and drop
            # this turn's message, which no dispatch will ever read.
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
                # Terminal for the EPISODE, and for the same reason
                # CompactionInfeasible above is: the conversation cannot be
                # made to fit, and nothing this loop does next changes that.
                # Retrying re-sends the same bytes for the same answer.
                log.warning(
                    "%s stop_round request does not fit the served window "
                    "-> terminal (retry would re-send the same bytes): %s",
                    rid_tag, _summarise_exc(exc))
                if transcript is not None:
                    transcript.append({
                        "stage": _tr_stage, "kind": "oversize_rejection",
                        "error": _summarise_exc(exc),
                    })
                return _mission_end("oversize_rejection", turns=turn_idx + 1)
            transient_failures += 1
            if outage_since is None:
                outage_since = time.time()
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
                    "stage": _tr_stage, "kind": "transient_error",
                    "attempt": transient_failures, "error": _summarise_exc(exc),
                    "since": outage_since,
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
        outage_since = None

        if r["kind"] == "text":
            content = str(r.get("content") or "")
            # Malformed-envelope repair: ANY tool-call residue in a prose
            # turn is a parser miss / hand-written envelope — repair it,
            # never end on it.
            if _envelope_residue(content) or _envelope_residue(
                    str(r.get("reasoning") or "")):
                log.info("%s MALFORMED_ENVELOPE at turn %d -> repair",
                         rid_tag, turn_idx)
                if transcript is not None:
                    transcript.append({"stage": _tr_stage,
                                       "kind": "malformed_envelope",
                                       "turn_idx": turn_idx})
                reasoner.append_repair(MALFORMED_ENVELOPE_NUDGE)
                continue
            if not content.strip():
                # An EMPTY assistant turn is not a reply: every real
                # ending carries text, and an empty turn is a dropped
                # message (a call lost between the think span and either
                # channel). Repair once; a second consecutive empty still
                # ends the mission -- the ending stays model-owned.
                no_call_turns += 1
                no_call_streak += 1
                if transcript is not None:
                    transcript.append({"stage": _tr_stage,
                                       "kind": "empty_turn",
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
                # The model wrote a BLOCK into the reply channel -- a
                # dropped message, not an ending. Repair once; the streak
                # is shared with the empty turn, so a repeated
                # not-a-reply still ends and the ending stays model-owned.
                no_call_turns += 1
                no_call_streak += 1
                if transcript is not None:
                    transcript.append({"stage": _tr_stage,
                                       "kind": "code_shaped_reply",
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
            # A REPLY BEFORE ANY FLIGHT answers nothing, so say so and
            # hand the turn back (NO_FLIGHT_REPLY_NUDGE), up to
            # NO_FLIGHT_ABSTAIN_STREAK; the ending stays model-owned.
            #
            # ONLY A CLEAN PROSE REPLY. The empty turn and the
            # code-shaped reply fall through to this same ending after
            # their own repair, and neither is the model saying "this is
            # my answer".
            if no_call_streak == 0 and mission_flights == 0:
                no_flight_streak += 1
                if transcript is not None:
                    transcript.append({"stage": _tr_stage,
                                       "kind": "no_flight_reply",
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
            # The mission contract's ONE model-owned ending: a clean
            # PROSE reply. (Envelope residue and a reply that IS a block
            # of the surface are channel misses, repaired above.)
            log.info("%s mission ended by reply at turn %d (%d flights)",
                     rid_tag, turn_idx, mission_flights)
            return _mission_end("reply", reply_text=content,
                                turns=turn_idx + 1)
        no_call_streak = 0
        # THE MODEL DID SOMETHING, so the abstain streak restarts.
        no_flight_streak = 0

        # Tool-call turn: dispatch each call IN ORDER, answering EVERY call
        # id (the compactor's id-closure sweep requires a result per id).
        for tc in r["tool_calls"]:
            name = (tc.get("function") or {}).get("name") or ""
            cid = tc.get("id") or ""
            args = json.loads(tc["function"]["arguments"])
            if name == "run":
                out = _dispatch_run(args, cid)
                if isinstance(out, dict) and not withheld("look_for"):
                    out.setdefault("watching", (
                        _watching_trailer()
                        or "nothing -- no class is segmented on any flight "
                           'until one is looked for; call look_for("<class>").'))
                reasoner.append_tool_result(cid, "run", out)
                if transcript is not None:
                    transcript.append({
                        "stage": _tr_stage, "kind": "run",
                        "turn_idx": turn_idx,
                        # The model's own reasoning-channel text, ahead of
                        # the code it decided on.
                        "reasoning": str(r.get("reasoning") or "")[:4000],
                        "said": str((args or {}).get("code") or "")[:400],
                        # print()'d text from running the block -- already
                        # capped upstream (python_surface.PRINT_CHAR_CAP),
                        # so passed through rather than re-truncated here.
                        "printed": (str(out.get("printed") or "")
                                    if isinstance(out, dict) else ""),
                        "errored": bool(isinstance(out, dict)
                                        and "error" in out),
                        # Every image this block put in front of the
                        # model, by the sha its bytes hash to: the
                        # frames/<sha>.jpg file and the wire log's
                        # image_ref share the name.
                        "frames": list(_block_frames)})
                _block_frames.clear()
            else:
                # A mangled/unknown function name still gets its id answered
                # (id-closure) with an error naming the tool that IS served;
                # never executed.
                reasoner.append_tool_result(cid, name or "unknown", {
                    "error": f"unknown tool {name!r}; available: run"})
        # Two-message look delivery: the images ride a USER turn AFTER the
        # turn's tool results (keeps tool-result id-closure contiguous).
        for _vtext, _vjpegs in pending_vision:
            reasoner.append_vision_user(_vtext, _vjpegs)
        pending_vision.clear()
