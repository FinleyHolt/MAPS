"""The mission conversation: one persistent reasoner, one served tool.

``run_mission`` (the CityNav bench) opens ONE conversation per mission
over ``prompts.CODE_TOOLS`` ([run]) and serves ``CODE_SYSTEM_PROMPT``
beside it. Every model act is a native tool call; everything else is
thinking, and there is no structured-output emission on the agent path.

Each call carries one BLOCK of Python. ``_dispatch_run`` hands it to
``python_surface``, whose library calls record steps that ``code_surface``
compiles onto a plan; ``_run_plan`` executes it:

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

from maps.llm.http import CallCanceled, DegenerateCompletion
from maps.llm.net_meter import stage as net_meter_stage
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
from maps.search.parse_compile.discourse import (
    echo as _echo,
    new_discourse as _new_discourse,
    update_discourse as _update_discourse,
)
from maps.search.parse_compile import python_surface as _py_surface

# Transient LLM-side errors. Treated as retry rounds rather than
# reasoner kills:
#   - HTTPStatusError: vLLM returned a non-2xx (typically 400 from
#     xgrammar refusing a malformed args envelope, or 429 backpressure).
#   - ConnectError / ReadError / TimeoutException (connect, read, write,
#     pool): tunnel flap, SSH drop, VPN cookie expiry, vLLM worker
#     restart. These are the bulk of retriable failures; retrying after
#     a short backoff lets the reasoner recover instead of marking it
#     unresolved. A ConnectTimeout is the shape a dead relay port takes,
#     so it rides with the rest.
_RETRIABLE_LLM_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPStatusError,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    # A collapsed completion: the serve's sampler returned a repeated-token
    # run with no content and no call. NAMED rather than left to ride its
    # ValueError base, so tightening that base later cannot silently drop it.
    DegenerateCompletion,
    # ValueError covers a transport-level decode failure (empty body /
    # non-JSON response). A fresh generation usually sticks.
    ValueError,
)

#: What a serve says when it rejects a request for its SIZE. Lowercased
#: substrings, matched against the response body.
#:
#: A 4xx above is retriable because a grammar rejection turns on what the
#: model GENERATED, and a fresh generation differs. An oversize rejection
#: turns on what we SENT, and the retry re-sends the same bytes: the loop
#: burned six identical 66,644-token requests per occurrence, then ended the
#: episode `transient_failures`. On the reasoning-effort ladder that hit 7 of
#: 32 episodes on the max-effort arm and ZERO on the other two, so the defect
#: landed entirely on one arm of a three-arm comparison.
#:
#: Prose matching, and safe as prose matching goes: a spelling this misses
#: falls through to the retry path, which is exactly today's behaviour. It can
#: cost a wasted retry; it cannot turn a retriable error permanent.
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
    """Did the serve refuse what was sent (a 4xx other than 408/429)?

    A timeout or a rate limit is the serve's weather; any other 4xx is a
    request the method built wrong, and re-sending it fails the same way."""
    return (isinstance(exc, httpx.HTTPStatusError)
            and 400 <= exc.response.status_code < 500
            and exc.response.status_code not in (408, 429))


def _summarise_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


# Carry-state tokens: a program reading these consumes results from EARLIER
# calls, so an identical re-issue can legitimately return new rows.
_DEDUP_CARRY_TOKENS = ("last", "prev", "previous")


def _dedup_ref_norm(ops: object) -> dict:
    """``{normalized label -> the ref as WRITTEN}``, by emission index.

    THE correspondence :func:`_program_dedup_key` judges identity on, read
    from one place by both the key and the alias that repairs a dedup hit
    (``_alias_deduped_refs``). Two copies would let an alias mean something
    the identity did not, which is the whole defect class the alias exists
    to close.
    """
    if not isinstance(ops, list):
        return {}
    return {
        f"r{i}": str(op.get("ref")).lstrip("$")
        for i, op in enumerate(ops)
        if isinstance(op, dict) and isinstance(op.get("ref"), str)
        and re.fullmatch(r"\$?r[0-9]{1,3}", str(op.get("ref")))
    }


def _program_dedup_key(program: object, carry_refs: object = ()) -> str | None:
    """Canonical identity of a SELF-CONTAINED program, or ``None`` when it
    reads carry state and is therefore never deduped.

    Within one per-stop round the (cands, anchors) surface is frozen, so an
    identical self-contained program is a pure re-render — its result cannot
    change. Identity is judged on the program's SHAPE with ref LABELS
    normalized out (each rN-grammar ref renamed to its emission index, rN
    tokens inside args strings rewritten to match): an incrementing ``rN``
    defeated the byte-identity guard entirely (356/357 identical calls, 0
    firings, measured). Emission-order permutations stay distinct — order
    freedom is a feature, not a rename. Programs that read a handle they do
    not define — a bare ``rN``, the ``last``/``prev`` tokens, or ANY live
    carry ref (which covers off-grammar ref names another call may have
    stored) — resolve against the round's ACCUMULATED results and stay
    dedup-exempt (all checks run against the ORIGINAL names).
    """
    if not isinstance(program, dict):
        return None
    ops = program.get("ops")
    if not isinstance(ops, list) or not ops:
        return None
    try:
        blob = json.dumps(ops, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 — unserialisable program: just don't dedup
        return None
    defined = {
        str(op.get("ref")).lstrip("$")
        for op in ops
        if isinstance(op, dict) and isinstance(op.get("ref"), str)
    }
    for tok in re.findall(r"\br[0-9]{1,3}\b", blob):
        if tok not in defined:
            return None
    low = blob.lower()
    if any(f'"{t}"' in low or f'"${t}"' in low for t in _DEDUP_CARRY_TOKENS):
        return None
    for name in carry_refs or ():
        name = str(name)
        if name in defined:
            continue  # locally shadowed — the seed is never read
        if (f'"{name}"' in blob or f'"${name}' in blob
                or f'"{name}[' in blob):
            return None
    # Ref-label normalization AFTER the carry checks: rename each rN-grammar
    # ref to its emission index in ONE pass (no chaining), leaving
    # off-grammar names untouched (renaming a definition without its
    # free-text uses could collide two genuinely different programs).
    ren = {actual: norm for norm, actual in _dedup_ref_norm(ops).items()}

    def _rw(v):
        if isinstance(v, str):
            return re.sub(r"\br[0-9]{1,3}\b",
                          lambda m: ren.get(m.group(0), m.group(0)), v)
        if isinstance(v, dict):
            return {k: _rw(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_rw(x) for x in v]
        return v

    canon = []
    for op in ops:
        if isinstance(op, dict):
            op2 = {k: _rw(v) for k, v in op.items() if k != "ref"}
            # NULL-NORMALIZE: the required-form schema fills unused args
            # with null; the same program written with and without the
            # sentinels is the same program.
            if isinstance(op2.get("args"), dict):
                op2["args"] = {k: v for k, v in op2["args"].items()
                               if v is not None}
            if "ref" in op:
                op2["ref"] = ren.get(str(op.get("ref")).lstrip("$"),
                                     op["ref"])
            canon.append(op2)
        else:
            canon.append(op)
    try:
        return json.dumps(canon, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 — unserialisable program: just don't dedup
        return None


# The dedup hit serves a one-line REFERENCE, never a fresh payload: the full
# re-render was measured pure loop fuel (~1,750 tokens per identical result,
# consumed without breaking any loop ×1,298). {turn} = the turn the original
# ran on this round. The note carries NO echo of the submitted program:
# echoing it back served raw op-JSON on the say channel (48 times over 47
# episodes of saydialect_1000) -- the one register that surface must never
# read -- and the identity fact ("turn {turn}") already names the call.
#
# IT LEADS WITH THE WORLD, NOT WITH THE SUBMISSION. The firing condition is
# `_prev[0] == _surface_version[0]` -- the surface has not moved since that
# turn -- and that is a strictly stronger fact than "your program was
# identical": it says NO program would answer differently, so re-asking
# anything is the wrong move and only changing the world is not. Leading with
# "identical to the program you ran" described the submission and invited the
# reading that the executor was declining to work, which is the one thing
# that is not happening. Every path that can mint a row bumps the version (a
# flight, a perceive, a retro boost, a stance, an anchor), so "nothing has
# entered the world" is exactly true whenever this fires -- and a re-flight
# is NOT what this is about: a flight bumps the version, so an act never
# reaches here.
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
    """1 -> 1st, 2 -> 2nd, 3 -> 3rd, 11..13 -> th (the served dedup note
    said "a 2th time" 146 times in framestage_1000)."""
    n = int(n)
    if 10 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _persist_look_audit(
    *, audit_dir: str, stop_id: str, step_idx: int, rid_tag: str, cid: str,
    annotate: bool, tool_ack: dict, vision_text: str, crops: list[dict],
) -> None:
    """Persist EXACTLY what one ``look`` call delivered to the model, so a
    completed run can be audited on WHAT each tool call provided in TEXT and
    IMAGES (load-bearing for diagnosing vision SR: a neutral/negative result is
    first suspected as a BAD crop — mis-centered, GT not in frame, wrong
    candidate, unreadable — not "vision doesn't help", and that is only checkable
    by SEEING what the model saw).

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
        # WARNING (not debug): the look audit is load-bearing for vision
        # diagnosis ([[feedback_vision_helps_prior_audit_payload]]) — a
        # persist failure must be visible in the runner log.
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
# it. Three, because the turns are not free -- an episode arguing about
# whether to fly is not searching, against a 100-turn cap -- and because the
# sibling empty-turn nudge measures 84% next-turn recovery, so a second ask
# collects most of what the first leaves. A tenth would fire only on a model
# stuck for some other reason, where repetition is not the fix and the
# streak itself is the signal worth reading.
#
# It is a STREAK: anything the model EXECUTES between two replies resets it
# (see the tool-call site), so a find that turns up nothing still counts as
# working and only replies with nothing between them accumulate. That is
# what stops a nudge loop without a magic number -- a model doing real work
# can never be cornered by this, and one doing nothing reaches the abstain
# in three turns.
#
# The harness still never picks a target. "Zero flights is an honest zero"
# is a rule about the COMMIT -- do not invent one where nothing flew -- and
# asking again is not inventing.
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
    # Liveness backstops, NEVER served: the model hears about no budget
    # (a stated cap is an arbitrary head-tax on an agentic loop). The turn
    # cap is the ONLY ending the model does not own; both bounds sit far
    # above every observed episode (measured maxima: 10 flights, 48 turns
    # only under a since-fixed friction class) and exist purely so a
    # pathological conversation cannot run unbounded.
    flight_budget: int = 96
    turn_cap: int = 100
    # The wall-clock half of the same backstop, as an ABSOLUTE epoch — a
    # count of stops cannot bound a count of seconds, and the harness kills
    # on seconds. Two Cal City episodes flew 262 and 264 stops of one
    # ``explore`` at ~40 s each and were SIGKILLed at the 4 h cap with no
    # prediction row at all; the 1000-flight backstop above is ~11 h at that
    # rate, so it could never fire first.
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
    flown again, the cover walk -- and leave here. Served, they were read
    as a placement to critique and a reason to fly the same row again
    (sortie 20260911T201524Z run 3, six requests for one row)."""
    return {k: v for k, v in report.items() if k not in ("pose_xy", "alt_m")}


def _legend_line(entries: list[dict]) -> str:
    return " ; ".join(
        f"{e['digit']} = {e['label']} ({e['role']})" for e in entries)


def _window_caption(c: dict) -> str:
    """One rendered image's window: span, orientation and world extent."""
    # ORIENTATION, per image and derived, never assumed: a rotated image
    # states its true up (bearing + the frame note); only a genuinely
    # unrotated one says north-up. The header claims nothing -- the
    # hardcoded "north-up" there captioned 1,129 rotated crops in 327
    # episodes of flygoal_1000.
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
    # Thin white ellipses + numbered badges on the image; the digit ->
    # fid binding lives HERE in the text legend (the strongest measured
    # lever for mark binding). Color is deliberately
    # NOT stated -- the model reads it from the unobscured pixels.
    # Each crop line states its delivered span, so a wide relational
    # frame can never silently imply a close look.
    from maps.search.render.look_render import (  # noqa: PLC0415
        LOOK_MAX_BADGES,
    )
    # THE CAPTION NAMES WHAT THE PICTURE IS OF. center_fid is a STOP id on a
    # flight view and absent on some crops, so this line read
    # "one per fid in order: [?]" on the majority of served captions -- the
    # one place a picture says which row it shows, saying nothing. The
    # legend's own center entry is the subject the renderer actually drew,
    # so it is the fallback.
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
    # delivered GSD above is grid spacing divided by a Lanczos upscale, and
    # on this bake the grid is finer than the camera: 0.3 m/px over 0.6 m
    # optical NAIP, upscaled up to 3x, captioned "0.10 m/px" — six times
    # sharper than anything the sensor saw. The comment above already
    # forbids telling the model a sharper picture than it was given; this
    # is what keeps it true. Resampling adds pixels and tokens, never
    # detail, and 54% of episodes said out loud they could not see.
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
        # THE PIXEL GRID, per image, because it is not constant: a crop's
        # upscale is clamp(round(450/native), 1, 3) and a map's side is
        # clamp(span/0.35, 320, 1024). Stating it lets the model convert
        # what it SEES into a world point by whichever route it reasons in
        # -- "300 of 1000 down" or "three tenths down" both land on the
        # same metre once the grid and the window are both named.
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
        # The subject is IN frame and still bigger than the frame. None of
        # the cases above fires here, and the badge alone then binds its
        # digit to whichever fragment of the edge happens to cross the
        # window -- a reader cannot tell a whole 80 m building from a
        # 2.6 km landuse relation. State the size and say the ink is cut.
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
    The note DESCRIBES; the arm stands exactly as said -- no filter, no
    rewrite, no gate (a colour-style cut here could drop a legitimate
    open-vocab class the same way a colour cut can drop ground truth).
    """
    named = [c for c in classes if c.strip().lower() in names]
    if not named:
        return None
    q = ", ".join(f"'{c}'" for c in named)
    return (f"{q}: a NAMED feature on this map, already a world-state row "
            f"-- the segmenter takes visual classes like 'house' or "
            f"'pool'. Watched as said.")


# A SET pass covers many members in one walk, and its bound is THE
# MISSION'S OWN -- ``mission_spec.flight_budget``, read at the call site,
# because a stop IS a flight and a second smaller number is a budget by
# another name. This default only covers a caller that states none.
#
# It was 24, a quarter of the mission's flights: a 64-way road truncated
# to a quarter of a pass and reported itself SAMPLED, and a served budget
# makes the model settle -- which is exactly what calcity::26#26 did.
_COVER_SET_MAX_STOPS = 1000

# How many of a pass's own new rows ride back as crops on its ack -- the
# catch. Half the 16-image context budget, so the model's own later looks
# keep room; the note pages the rest.
_CATCH_CROPS = 8
# What a look and a pass's catch say under the ``images`` withhold.
_PICTURE_WITHHELD = ("the crop's caption follows in the next message; "
                     "its picture is not served on this run")


def _mission_sanitize_moves(raw, staged) -> tuple[list[dict], list[dict]]:
    """Normalize the mission act's ``move`` list.

    Each entry is ONE slot — a staged handle (vN, resolved to its stand
    point), a literal fid / view-position sentinel, or the canonical
    ``"[x, y]"`` token. A staged handle normalizes to the canonical token
    under ``target_fid`` (``maps.util.xy_token``) so every downstream
    consumer keeps reading the ONE wire grammar; a
    vN-resolved item carries ``via`` so the flight report can echo the
    handle. Returns ``(accepted_items, item_errors)`` — per-entry errors
    carry the entry's list index so valid siblings still fly. The round
    era's target_fid-XOR-target_xy item is retired with the XOR error
    class it needed.
    """
    items = raw if isinstance(raw, list) else []
    accepted: list[dict] = []
    errors: list[dict] = []
    for i, it in enumerate(items):
        if not isinstance(it, str) or not it.strip():
            errors.append({"index": i, "error": (
                "each move entry is a staged handle (vN), a literal fid, "
                "or an [x, y] world point")})
            continue
        token = it.strip()
        sv = (staged or {}).get(token)
        if sv is not None:
            accepted.append({"target_fid": format_xy_token(
                float(sv["from_xy"][0]), float(sv["from_xy"][1])),
                "via": token})
            continue
        if re.fullmatch(r"v\d{1,3}", token):
            errors.append({"index": i, "error": (
                f"{token!r} names no staged viewpoint (stage one first, "
                f"or pass a fid / [x, y] point)")})
            continue
        if _is_handle_token(token):
            errors.append({"index": i, "error": (
                f"{token!r} is a result handle; move takes a staged "
                "handle (vN), a literal fid off the rows, or an [x, y] "
                "world point")})
            continue
        accepted.append({"target_fid": token})
    return accepted, errors


def _single_fid_of_geom_result_value(v) -> str | None:
    """The fid IFF this one op result isolated EXACTLY one fid-bearing node, else None.

    Two compacted shapes count as "one fid":
      * a single result-dict bearing a ``fid`` key — a ``pick``
        reducing to one node (survives ``compact_results`` untouched);
      * a LIST result that ``compact_results`` rendered as ``{"len": 1, "items": [d]}``
        where the one item ``d`` carries a ``fid`` — i.e. a ``find`` /
        ``score_by`` / ``filter_items`` that reduced to exactly one car
        WITHOUT a trailing ``pick``. (``compact_results`` returns the full
        list up to a high safety cap, so a len-1 list is always fully preserved.)
    A multi-element list, a scalar, or a fid-less dict is NOT one isolated fid."""
    if isinstance(v, dict) and "len" not in v and "items" not in v \
            and "fids" not in v:
        fid = v.get("fid")
        return fid if isinstance(fid, str) and fid else None
    # Compacted list shape: a len-1 list whose single row carries a fid counts
    # as the program isolating one object. A list result carries its ordered
    # fids as ``fids`` (the rows render as a grid string); an ``items`` list is
    # also read, for older payloads. A
    # ``compare`` result (which has ``refs``) is an ATTACH, never an isolation.
    if isinstance(v, dict) and v.get("len") == 1 and "refs" not in v:
        items = v.get("items")
        if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
            fid = items[0].get("fid")
            return fid if isinstance(fid, str) and fid else None
        fids = v.get("fids")
        if isinstance(fids, list) and len(fids) == 1 \
                and isinstance(fids[0], str) and fids[0]:
            return fids[0]
    return None


def _terminal_fid_of_geom_result(out: dict) -> str | None:
    """The fid stamped by the LAST op that reduced the program to ONE fid.

    PURE OBSERVER — no behavior depends on the return value. Reads the
    ``run_geom_program_tool`` payload's ``results`` map (which preserves op order:
    ``run_program`` inserts each op's result by ref in execution order, and
    ``compact_results`` is an order-preserving dict comprehension that leaves a
    single result-dict untouched — only LIST results are truncated to
    ``{len, items}``). A terminal single-fid op (``pick``
    -> a fid-bearing dict, OR a ``find`` / ``score_by``
    that reduced to exactly one fid-bearing item) leaves a
    one-fid result — the fid carried through from the surface.

    Returns the fid of the LAST such op. Scaffold-then-prose programs that never
    reduced to one fid return ``None`` (only multi-element lists, scalars, or
    fid-less dicts remain)."""
    results = (out or {}).get("results")
    if not isinstance(results, dict):
        return None
    terminal_fid: str | None = None
    for v in results.values():
        fid = _single_fid_of_geom_result_value(v)
        if fid is not None:
            terminal_fid = fid
    return terminal_fid




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

# One factual line on a PREFLIGHT runtime-surface retrieval: the detection
# surface is empty before takeoff -- a fact about WHEN detections exist,
# stated on the result the agent actually reads (the same fact the
# round-0 viewpoint machinery states on its own surface).
_PREFLIGHT_RUNTIME_NOTE = (
    "no runtime detections exist before takeoff; the detection surface "
    "populates per flown stop."
)


def _color_words_of_op(op) -> list[str]:
    """The canonical palette words this op requests a color COLUMN for,
    read off every slot that can name one — the find ``color`` arg, a
    ``filter_items`` key, a ``score_by`` emphasis key — so a phase note keys
    on the attach request itself, never on one op shape."""
    if not isinstance(op, dict):
        return []
    args = op.get("args") if isinstance(op.get("args"), dict) else {}
    words: list[str] = []

    def _add(w):
        c = _normalize_color(str(w).strip().lower())
        if c and c not in words:
            words.append(c)

    c = args.get("color")
    if isinstance(c, str) and c.strip():
        _add(c)
    keys = [args.get("key")] + [
        e.get("key") for e in (args.get("emphasis") or [])
        if isinstance(e, dict)]
    for k in keys:
        if _is_color_column_key(k):
            _add(k[len("color_"):])
    return words


def _program_requests_color(program) -> bool:
    ops = program.get("ops") if isinstance(program, dict) else None
    return any(_color_words_of_op(op) for op in ops or [])


def run_geom_program_tool(args: dict, *, namespace: dict | None = None,
                          raw_sink: dict | None = None,
                          seed_results: dict | None = None,
                          preflight: bool = False,
                          answer_ref: str = "",
                          geometry_sink: dict | None = None) -> dict:
    """Dispatch one ``analyze_scene(program)`` call — the ONE geometry channel.

    Runs the model's typed op-program through the SHARED
    :func:`maps.search.analyze_scene.geom_program.run_program` over the round's
    ``(cands, anchors)`` surface (``_surface_from_ctx``), then truncates list
    results with ``compact_results``.

    ``raw_sink`` (harness-side evidence plumbing, never model-facing): when
    given, receives the PRE-compaction results under ``raw_sink["results"]`` so
    the round can fold the model's own ``compare`` columns + latest enriched
    rows in (``geom_program.compare_columns``) without re-deriving them.
    ``geometry_sink`` is its dashboard-audit sibling: the resolved compare
    geometry under ``sink["ops"]`` (see :func:`run_program`), never model-facing.

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
                           geometry_sink=geometry_sink,
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
    # The agent owns the order: the surface, the crop feed and the P3
    # program-terminal commit input all read the program's OWN result order
    # (score_by fuses exactly the columns the agent's emphasis weights;
    # nothing is harness-reordered).
    _raw_results = out["results"]
    if raw_sink is not None:
        raw_sink["results"] = _raw_results
    # The widening form is the SAY channel's; the round-era surface has its
    # own slice syntax in its own system prompt and must not be handed a
    # sentence it cannot spell.
    # ``answer_ref`` is the SAY channel's: a sentence asks one question, so
    # its return is one table and the other ops' hints ride it. The round-era
    # surface writes its own op-programs and reads every rN it named, so it
    # passes none and keeps them all.
    out["results"] = _compact_results(_raw_results,
                                      widen_form='"more of <rN>."',
                                      answer_ref=answer_ref)
    if preflight:
        # Pre-takeoff phase notes, keyed on what each op's
        # result ACTUALLY holds (never an op shape):
        #   * a color-column request (find color / emphasis key / filter
        #     key) whose raw rows carry NONE of the requested columns
        #     states the color-availability fact;
        #   * a runtime-surface find over the empty pre-takeoff detection
        #     pool states the detection-availability fact.
        for _op in (program.get("ops") or []):
            if not isinstance(_op, dict):
                continue
            _notes: list[str] = []
            _words = _color_words_of_op(_op)
            if _words:
                _rows = _raw_results.get(_op.get("ref"))
                _attached = isinstance(_rows, list) and any(
                    isinstance(_it, dict) and any(
                        _it.get("color_" + _w) is not None for _w in _words)
                    for _it in _rows)
                if not _attached:
                    _notes.append(_PREFLIGHT_COLOR_NOTE)
            _args = _op.get("args") if isinstance(_op.get("args"), dict) else {}
            if (_op.get("op") == "find" and not cands
                    and str(_args.get("provenance") or "").strip().lower()
                    in ("runtime", "detected", "detection", "detections")):
                _notes.append(_PREFLIGHT_RUNTIME_NOTE)
            if _notes:
                _env = out["results"].get(_op.get("ref"))
                if isinstance(_env, dict):
                    for _n in _notes:
                        _env["hint"] = join_hints(_env.get("hint"), _n)
    return out


def _surface_from_ctx(ctx) -> dict:
    """Build the ``analyze_scene`` ``{cands, anchors, cell_m}`` surface from a
    :class:`ToolContext`'s unified graph — the round-0 entry to the
    ``build_surface_from_graph`` path every round reaches. Round 0 has no
    detections yet, so
    ``cands`` is typically empty and ``anchors`` are the prebaked priors; the
    reasoner can still ground named-landmark geometry (corners, projections)."""
    from maps.search.analyze_scene.geom_program import build_surface_from_graph

    cell_m = float(getattr(ctx, "cell_size_m", 0.0) or 0.0)
    g = getattr(ctx, "graph", None)
    if g is None or not hasattr(g, "nodes"):
        return {"cands": [], "anchors": [], "cell_m": cell_m}
    cands, anchors = build_surface_from_graph(
        g.nodes.items(), getattr(ctx, "tile_origin_xy", (0.0, 0.0)), cell_m
    )
    return {"cands": cands, "anchors": anchors, "cell_m": cell_m}


def _ref_key(ref_token) -> str:
    """Normalize a bare ``rN`` result-handle token to its carry-dict key,
    absorbing the ``$``/``ref:``/``ref_`` slips the program resolver forwards
    (a ``$``-prefixed form still resolves; the canonical served form is bare
    ``rN``)."""
    key = str(ref_token or "").strip()
    if key.startswith("$"):
        key = key[1:]
    for _pre in ("ref:", "ref_", "ref-", "ref."):
        if key.startswith(_pre):
            key = key[len(_pre):]
            break
    return key


_VP_TOKEN_RE = re.compile(r"^(?P<base>[^\[\]]+?)(?:\[(?P<idx>[^\]]*)\])?$")

# A result-handle base is bare ``rN`` (optionally windowed ``rN[i]``/``rN[a:b]``)
# or a reserved previous-result word. Literal fids (building_/road_/parking_/
# detected_) and the view-position sentinels do NOT match, so they pass through
# expansion untouched.
_HANDLE_BASE_RE = re.compile(r"^r[0-9]{1,3}$")
_HANDLE_WORDS = ("last", "prev", "previous", "ref")


def _is_handle_token(token) -> bool:
    """True when ``token`` is a result handle — bare ``rN`` / ``rN[i]`` /
    ``rN[a:b]`` or a reserved previous-result word (``last``). Handles are
    REFUSED wherever a literal fid/[x,y] is wanted (select targets,
    viewpoint targets — the fan-out forms are retired). A ``$``-prefixed
    form still matches (typo absorption). A literal fid or a view-position
    sentinel returns False."""
    if not isinstance(token, str):
        return False
    m = _VP_TOKEN_RE.match(token.strip().lstrip("$"))
    if not m:
        return False
    base = _ref_key(m.group("base"))
    return bool(_HANDLE_BASE_RE.match(base)) or base in _HANDLE_WORDS


def _crop_token_of(row: dict, live: bool) -> str:
    """A crops token for one carried row: its fid, except a DERIVED end
    row (``end_N`` — the ``ends`` op's mint, never on the live surface)
    crops its own ``[x, y]``: serving "end_0 is not on the live surface"
    was the whole look(ends_of(...)) kickback family, and the xy-token
    form is the one ``_build_crop_requests`` already crops directly.
    ONLY that family coerces — a once-live fid gone from the surface
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


def _tool_call_args(tc: dict) -> dict:
    """Parse one normalized tool_call's arguments string to a dict.
    ``reasoner_agent._normalize_tool_call_arguments`` guarantees the string
    parses to a mapping; belt-and-braces for direct/offline callers."""
    raw = (tc.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
    except json.JSONDecodeError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _envelope_residue(text: str) -> bool:
    """True when a PROSE turn carries tool-call envelope residue — a parser
    miss (vLLM #39056-style: the envelope swallowed into content/reasoning) or
    the model writing the XML by hand. This is a malformed-envelope
    repair, NEVER a carry."""
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

    The CODE arm's twin of ``_language_reply``, and the failure it catches
    is the larger one: a fleet ended 51% of its episodes on a sentence
    written into the reply channel, mid-search, most of them a `go to`. A
    block written there ends the mission the same silent way -- short,
    with the flights it was about to make never made.

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
    look_enabled: bool,
    look_renderer: "Callable[[list[dict]], list[dict]] | None" = None,
    look_audit_dir: str | None = None,
    obs_span_m: float | None = None,
    obs_minor_m: float | None = None,
    watchlist_apply: "Callable[[list, list], dict] | None" = None,
    boost_queue: "Callable[[list, list], dict] | None" = None,
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
    (or a liveness bound fires). ``act.viewpoints`` flies through
    ``mission_spec.fly`` the moment it lands; the analyze surface is
    re-derived from ``ctx``'s live graph after every flight. The commit is
    the ENGINE's job — it executed every flight and knows the last target;
    this returns the conversation facts plus the Reasoner (for transcript
    persistence, and for :func:`continue_mission`).

    ``retask`` is the FIELD console's channel and the bench passes none:
    a callable yielding operator text typed since the last turn, drained
    at the top of each turn. With it unset the instruction is the single
    user message, exactly as measured.
    """
    # THE ARM SERVES ONE SURFACE IN BOTH CHANNELS. Serving the sentence
    # prompt beside the code tool served two contradictory surfaces at
    # once: the prompt taught sentences the parser refuses, and every
    # footer named a form the arm does not have.
    #
    # ponytail: the code prompt states the crops capability unconditionally
    # rather than carrying reasoner_agent's `_NO_CROPS_FACT` needle for the
    # swap, because SEARCH_LOOK_TOOL=0 is a wiring no config or run script
    # in this tree declares. The assert is what makes that true rather than
    # assumed; give the prompt the needle and a code-spelled capability
    # fact if a crops-less code arm is ever wanted.
    assert look_enabled, (
        "the code arm's system prompt states the look()/show_map() "
        "capability unconditionally; a crops-less run would serve a legend "
        "for calls it cannot make")
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
        watchlist_apply=watchlist_apply,
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
    watchlist_apply: "Callable[[list, list], dict] | None" = None,
    boost_queue: "Callable[[list, list], dict] | None" = None,
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
        watchlist_apply=watchlist_apply,
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
    watchlist_apply: Callable[[list, list], dict] | None = None,
    boost_queue: Callable[[list, list], dict] | None = None,
    preflight: bool = False,
    mission_spec: MissionSpec,
    retask: Callable[[], str | None] | None = None,
) -> MissionReply:
    """The mission's turn loop: append the instruction, then serve `run`.

    Appends ``user_message`` as the ONE user turn, then loops native tool
    calls over ``prompts.CODE_TOOLS``. Each ``run`` call carries a block of
    Python; the block executes in one persistent namespace and its plan
    runs through ``_run_plan``, so a move inside it flies before the next
    turn is built.

    The model's plain prose reply is the ONE model-owned ending, and three
    repairs stand in front of it so a dropped message is never mistaken for
    one: an EMPTY turn, a reply that is itself a block of code
    (``_code_reply``), and a reply with ZERO flights behind it each get a
    nudge before the second consecutive occurrence is accepted. Envelope
    residue in a prose turn gets the malformed-envelope repair.

    Harness-owned endings, none of which the model hears about:
    ``turn_cap``, ``oversize_rejection``, ``transient_failures``,
    ``compaction_infeasible``. ``MissionSpec.flight_budget`` and
    ``deadline_epoch`` stop flights, not the conversation.

    ``retask`` is the FIELD console's channel, drained at the top of each
    turn; the bench passes none, so the instruction is the single user
    message exactly as measured.

    When ``diag`` is provided, ``diag["turns_used"]`` is stamped each turn
    (failure-mode diagnostics; survives every return path).
    """
    # Text-only round: the reasoner reasons over the structured world-state
    # (SAM detections + perceived attributes on the runtime graph) via
    # analyze_scene alone (the ONE query tool). No images are attached.
    # A thread the compactor has declared infeasible stays so for the rest
    # of the episode (the residual head/prose mass it cannot touch only
    # grows), so don't grow it further with a packet no dispatch will read.
    thread_infeasible = bool(getattr(reasoner, "compaction_infeasible", False))
    # Episode-scoped union of every evidence-column name the round's programs
    # attach, kept on the immortal reasoner across stops. Instrument only —
    # nothing served reads it.
    evidence_columns: set = getattr(reasoner, "evidence_columns", None)
    if evidence_columns is None:
        evidence_columns = set()
        reasoner.evidence_columns = evidence_columns
    if not thread_infeasible:
        reasoner.append_user(user_message)

    # THE canonical in-flight surface (flagless): ``analyze_scene`` — a typed
    # op-program run through the SHARED ``run_program`` over this round's
    # (cands, anchors) surface (``compute_namespace``), so the live path and any
    # offline replay serve byte-for-byte the same surface. Retrieval folds INTO
    # the program (the find op).
    # The fid's LATEST result-enriched row across the round's successful
    # programs (a compare/sort output carries the attached columns) — the
    # rejection-escalation surface census reads it to state live facts.
    row_latest: dict[str, dict] = {}
    # Anti-spiral result reuse: successful SELF-CONTAINED programs seen this
    # round, keyed by canonical ops JSON (``_program_dedup_key``) ->
    # ``(surface_version, compacted_out)``. A repeat RE-RENDERS the stored
    # compacted envelope with the dedup fact — the surface is frozen within a
    # stop, so the re-run cannot say anything new (the #18493-class spiral
    # re-issued one program 18x and blew the compaction budget), but the
    # agent can always re-read its own answer (never a lockout after a
    # summarized/compacted-away result). A ``view`` call can mutate the
    # round's evidence (retro boost ingest), so it bumps the version and
    # re-arms every program.
    _programs_seen: dict[str, tuple] = {}
    _surface_version = [0]

    # Protocol telemetry.
    no_call_turns = 0
    no_call_streak = 0
    # CONSECUTIVE replies that arrived with ZERO flights behind them. A
    # STREAK, not a total: doing something between two replies -- a find, a
    # look, anything that executes -- resets it at the tool-call site below,
    # because a model that is working and has not yet flown is not the model
    # this counts. Three in a row with nothing in between is a confident
    # abstain and the mission ends on it.
    #
    # Its own counter rather than the no-call streak: that one counts channel
    # misses, and this reply is on the right channel and is a legal ending.
    no_flight_streak = 0

    # Surface rows for the fid∈surface parity check (the same rows the
    # programs read). On the mission path a flight steps the world
    # mid-conversation and the act arm REBUILDS this after every fly — a
    # frozen snapshot made every runtime detection invisible to
    # crops/frames/marks (measured: 186 crop skips over 73/101 episodes).
    _ns_rows: dict[str, dict] = {}

    def _rebuild_ns_rows() -> None:
        # Mutated in place: the crop/frame/nearest closures read this dict.
        # cands win on fid collision (setdefault, cands first) — the same
        # precedence the original one-shot build had.
        _ns_rows.clear()
        for _key in ("cands", "anchors"):
            for _r in (compute_namespace or {}).get(_key) or []:
                if isinstance(_r, dict) and _r.get("fid"):
                    _ns_rows.setdefault(str(_r["fid"]), _r)

    _rebuild_ns_rows()

    # ---- look: the crop path ----------------------------------------------
    # Each crop item resolves its fid to the surface xy (+ footprint/label for
    # the mark overlay) from THIS round's ``_ns_rows``, clamps to the
    # server's 16-image budget, and hands fully-resolved crop specs to
    # ``look_renderer``. Non-terminal: the rendered crops ride a USER turn
    # appended AFTER the turn's tool results (so tool-result id-closure stays
    # contiguous) and the round continues.
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
    # How many of a set's rows a MAP badges. The map shares the crop
    # renderer's digit->fid legend contract, and the SoM probe measured
    # ordering over a badged set flat to 25 badges, so the ceiling is the
    # one look_render already enforces.
    from maps.search.render.look_render import (  # noqa: PLC0415
        LOOK_MAX_BADGES as _MAP_MAX_MARKS,
    )

    # ---- act{stage}: the staged ground viewpoints (mission path) -----------
    # {handle: {"handle", "from_xy", "up_unit", "at", "facing",
    # "bearing_deg"}}; the LATEST stage is the STANDING viewpoint. One
    # state, three readers: the analyze namespace (executor staged/standing
    # kwargs -> vN point refs + the standing side columns), the crop/map
    # renderers (frame-up), and the move sanitizer (vN targets).
    staged_viewpoints: dict[str, dict] = {}
    standing_handle: list = [None]
    # The mission's flown stops, {stop_id: [x, y]} — a stop id the flight
    # reports serve is a sayable place (stand at it, fly to it, map it);
    # pilot-2 measured 59 attempts at unsayable stop references.
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
                # a staged handle IS a point: "show me v1." maps around
                # it, "from v1, looking east:" re-stands there (pilot-3:
                # the handle namespace was half-wired — 6 inner errors)
                return [round(float(sv["from_xy"][0]), 2),
                        round(float(sv["from_xy"][1]), 2)]
            return _rounded_xy(tok)
        return None

    def _stage_apply(items: list) -> dict:
        """act.stage: mint vN handles; the latest staged becomes STANDING.
        The receipt states the columns/refs/render facts the stance turns
        on; a re-stage is one more item. Bumps the surface version — a
        program re-run under a new stance can say something new."""
        from maps.search.analyze_scene.geom_compute import COMPASS_UNIT  # noqa: PLC0415
        staged_acks: list[dict] = []
        errors: list[dict] = []
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                errors.append({"index": i, "error":
                               "each stage item is {at, facing}"})
                continue
            at_tok, facing_tok = it.get("at"), it.get("facing")
            at_xy = _resolve_stage_point(at_tok)
            if at_xy is None:
                errors.append({"index": i, "error": (
                    f"at {at_tok!r} is not a live surface fid or an "
                    f"[x, y] world point")})
                continue
            if (isinstance(facing_tok, str)
                    and facing_tok.strip().lower() in COMPASS_UNIT):
                w = facing_tok.strip().lower()
                up = list(COMPASS_UNIT[w])
                facing_desc = w
            else:
                f_xy = _resolve_stage_point(facing_tok)
                if f_xy is None:
                    errors.append({"index": i, "error": (
                        f"facing {facing_tok!r} is not a live surface "
                        f"fid, an [x, y] point, or one of "
                        f"{'|'.join(COMPASS_UNIT)}")})
                    continue
                dx = f_xy[0] - at_xy[0]
                dy = f_xy[1] - at_xy[1]
                n = (dx * dx + dy * dy) ** 0.5
                if n < 1e-6:
                    errors.append({"index": i, "error":
                                   "at and facing coincide -- no look "
                                   "direction"})
                    continue
                up = [dx / n, dy / n]
                facing_desc = (facing_tok.strip()
                               if isinstance(facing_tok, str)
                               else f"[{f_xy[0]}, {f_xy[1]}]")
            h = f"v{len(staged_viewpoints) + 1}"
            brg = round(up_bearing_deg(up)) % 360
            staged_viewpoints[h] = {
                "handle": h,
                "from_xy": at_xy,
                "up_unit": [round(up[0], 4), round(up[1], 4)],
                "at": (at_tok.strip() if isinstance(at_tok, str)
                       else list(at_xy)),
                "facing": facing_desc,
                "bearing_deg": brg,
            }
            standing_handle[0] = h
            staged_acks.append({"handle": h, "at_xy": at_xy,
                                "facing_bearing_deg": brg})
        if not staged_acks and not errors:
            return {"error": "stage is an empty list of {at, facing} items"}
        ack: dict = {}
        if staged_acks:
            ack["staged"] = staged_acks
            _h = standing_handle[0]
            ack["note"] = (
                f"standing viewpoint = {_h}: 'left of {_h}'/'right of "
                f"{_h}'/'ahead of {_h}'/'behind {_h}' ride every find "
                f"row; {_h} is a point ref ('distance to {_h}'); crops "
                f"and maps render its facing up.")
        if errors:
            ack["rejected"] = errors
        if staged_acks:
            # a new stance changes what the SAME program returns: thread
            # it into the analyze namespace and re-arm the dedup.
            if compute_namespace is not None:
                compute_namespace["staged"] = staged_viewpoints
                compute_namespace["standing"] = standing_handle[0]
            _surface_version[0] += 1
        return ack

    def _look_strs(v: object) -> list[str]:
        return [str(x) for x in v if isinstance(x, str) and x.strip()] \
            if isinstance(v, list) else []

    def _build_crop_requests(
            items: list) -> tuple[list[dict], list[str], list[str]]:
        """Resolve one ``view.crops`` list into per-crop specs off ``_ns_rows``.

        Each item ``{fid, mark_fids?, target_fid?}`` yields one request
        ``{center_fid, center_xy, annotate, marks}`` where ``marks`` are that
        ITEM's resolved footprints ``{xy, boundary, label, is_target}`` (the
        renderer clips each mark to its own frame); providing mark_fids /
        target_fid on an item is what turns its relational overlay on (the
        center identification is unconditional). Centers clamp to the
        16-image server budget; a fid absent from this round's surface is
        skipped (never a hand-rolled cells->pixel). Returns
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
                # The server's per-request image cap. Stated WHEN IT BITES,
                # never as prompt-time advice: a limit the agent has not hit
                # costs context on every round to teach, while a silent drop
                # of the tail would read as "these fids rendered nothing".
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
                # is supposed to be ON is a SET: a street is its ways, and
                # marks=poppy[0] would put back the arbitrary-piece index
                # the surface exists to prevent. The renderer drops marks
                # whose footprint never reached the pixels BEFORE the badge
                # cap, so naming a 64-way road costs the crop only the ways
                # actually in its window.
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
            # The crop is centered on this fid; hand the renderer the fid's own
            # label + footprint so it can bracket-and-label the CENTER car (the
            # fid<->car identification) without occluding it.
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
                # crop selects less of what was captured, never more).
                # ONE value per mission because the mission flies ONE
                # cruise altitude: mission_engine's fly_one holds
                # state.altitude_m through min(max(alt, ground+1), cap),
                # which the start altitude already satisfies, so the
                # pre-mission value IS the current-pose value at every
                # crop. Measured end to end on the birmingham_block_1::0#0
                # smoke: 4 flights, all at 141.3 m, obs_span_m 262.6 m at
                # each. The round/field path (search_loop) passes its own
                # per-stop alt_m, which DOES vary — a real drone's frustum.
                # On the CityNav splits this ceiling never binds. The
                # start altitude is a per-episode sample of the 100-150 m
                # AMSL band (episode_runner.start_altitude_m), so
                # 2*(alt - ground) is 112.4-291.0 m (median 192) across all
                # 29,811 episodes of train_seen/val_seen/test_unseen —
                # always above the pixel-budget ceiling (100 m on these
                # 0.1 m/px orthos), which is what
                # actually clamps the window. 16,910 pulled look manifests
                # agree: not ONE crop record was clamped by obs_span_m.
                # An earlier note here recorded a 25.8 m median: that read
                # cfg.altitude_m (50 m), the inert fallback the runner
                # overwrites, and no episode has ever flown it.
                # It DOES bind where the frustum is small, which is why it
                # exists: the stanford car field test flies a fixed 50 m
                # AMSL over a 16.74 m datum, 33 m AGL, and a nadir sensor
                # there really does see only 66.5 m across.
                # maps/search/tests/test_look_obs_span_ceiling.py and
                # benchmarks/citynav/tests/test_start_altitude_span.py pin
                # all of it.
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

    # ---- perceive{map}: the world-state map path ---------------------------
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
            # int() truncates toward zero and so does astype; the same
            # ox + (c + 0.5) * cs, in the same order, so the doubles match
            # the per-vertex form this replaces (1 s over 1,922 roads).
            a = np.asarray(pts, dtype=float).reshape(-1, 2).astype(np.int64)
            return np.stack([float(_ox) + (a[:, 1] + 0.5) * _cs,
                             float(_oy) + (a[:, 0] + 0.5) * _cs],
                            axis=1).tolist()

        feats: list[dict] = []
        for fid, node in nodes.copy().items():
            # The node's own (N, 2) int array rides as is: the painter takes
            # an array or a list, and converting 3.9 M cells to a list per
            # show_map was the map's whole cost on the Cal City bake.
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
            # in, and dropping it here is why every street drew as an
            # anonymous gray line.
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

    def _build_map_requests(
            items: list) -> tuple[list[dict], list[str], list[str]]:
        """Resolve one ``perceive.map`` list into render specs. Each item
        ``{around?, radius_m?, mark_fids?}`` centers one map (around
        absent = the whole tile); marks resolve off ``_ns_rows`` exactly
        like crops. Returns ``(specs, skipped_fids, ref_notes)``."""
        specs: list[dict] = []
        skipped: list[str] = []
        notes: list[str] = []
        gs = getattr(ctx, "grid_shape", None) or (0, 0)
        cs = float(getattr(ctx, "cell_size_m", 0.0) or 2.0)
        ox, oy = getattr(ctx, "tile_origin_xy", (0.0, 0.0))
        tile_center = [round(float(ox) + float(gs[1]) * cs / 2.0, 2),
                       round(float(oy) + float(gs[0]) * cs / 2.0, 2)]
        for _i, it in enumerate(items):
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
                    continue  # a degenerate emission repeated one fid 675x
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
            # THE FRAME FOLLOWS THE REQUEST. The span was a fixed radius or
            # the whole tile, whatever the map was OF: the default is 6576 m
            # at 1024 px, so a 15 m house renders 2.3 px and a 2 m road
            # 0.31 px -- and this verb runs in 94.5% of episodes at 1.5 per
            # episode. The marks were resolved and badged and then ignored
            # by the framing, so narrowing to three houses and asking to see
            # them returned the whole tile.
            # Framing to what was asked about is 8-18x linear resolution on
            # it (a block-sized set puts a house at ~43 px), and it needs no
            # numeral in the language: the extent derives the zoom.
            _cb = (look_outline(_ns_rows.get(center_fid))
                   if center_fid else None)
            _fpts: list = [m["xy"] for m in marks if m.get("xy")]
            for _src in [m.get("boundary") for m in marks] + [_cb]:
                for _b in (_src or []):
                    if isinstance(_b, (list, tuple)) and len(_b) >= 2:
                        _fpts.append([float(_b[0]), float(_b[1])])
            if not _fpts and around is not None:
                # `show me the map near 2830 3600.` is an unambiguously
                # LOCAL ask with nothing to frame from, and it fell through
                # to the 6576 m tile — 7-10% of map requests across two
                # runs. A named point gets the same floor a framed set does.
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
        return specs, skipped, notes

    def _map_vision_text(maps: list[dict]) -> str:
        """The map message's text: what the picture IS (world state, no
        imagery), each map's window + orientation + badge legend, and the
        class->color legend the digest palette needs."""
        order = ", ".join(str(m.get("center_fid") or "map") for m in maps)
        # WHAT THE MAP IS, AND WHAT IT IS NOT. It is the PRIOR -- a survey,
        # drawn from OSM, filled by class -- not a picture of the world as
        # it is now. It carries footprints, road runs, landuse and names.
        # It does not carry the shape of a cul-de-sac bulb, a driveway, a
        # pool, a parked car, or anything built since the survey, and no
        # zoom will make those appear because they were never recorded.
        # A log read found the model retreating to this instrument when
        # crops were unreadable and then guessing an end-shape off a
        # schematic that structurally cannot show one. The camera is the
        # live check, and a flight is how it is taken.
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

    # Identical-error loop breaker, ESCALATING: after 2 same-class
    # ProgramError / select rejections in one round, the error text
    # additionally states the repetition + the live surface facts (fids
    # present, columns computed this round) — the census vehicle carries the
    # full state ONCE. From the 3rd CONSECUTIVE same-class rejection the
    # harness stops re-issuing the remedy+census bytes verbatim: measured
    # (armC7 episode 7181), the identical color-as-filter rejection was
    # re-served 195x consecutively (69MB served, 205 turns) and broke
    # nothing — a repeated identical byte-string is the one served form
    # measured NOT to break a rejection loop. The escalated form serves the
    # repetition FACT (consecutive + round counts), the rejection's
    # identifying clause, where the full remedy/census already sit, and the
    # compact live-state counts. Facts-only, no forcing, no lockout, no turn
    # caps — the agent stays free to emit anything, including the same call
    # again (it then reads the incremented fact, never the same bytes).
    _err_class_counts: dict[str, int] = {}
    _err_streak: dict = {"sig": None, "n": 0}
    # Consecutive identical-dedup streak — the dedup path's own escalation
    # (measured: 348 dedup notes, streaks to 15; a byte-identical note is
    # the served form measured NOT to break a repeat loop, so from the 2nd
    # consecutive hit the note escalates to the changing live-state facts,
    # exactly as _loop_broken_error does for rejections).
    _dedup_streak: dict = {"key": None, "n": 0}

    def _loop_breaker_facts(_ns, _n: int) -> str:
        def _cap(fids: list[str]) -> str:
            return f"{fids[:8]}" + (
                f" (+{len(fids) - 8} more)" if len(fids) > 8 else "")
        c_n, a_n, cols = _surface_census_counts(_ns)
        parts = [
            f" This error class has now occurred {_n}x.",
            # Counts only: the first-8-alphabetical fid sample was always
            # ['prior_0', 'prior_1', ...] regardless of relevance -- pure
            # token noise on every 2nd serve (measured 199 payloads).
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
        """Compact live-state counts (n detections, n priors, computed
        columns) — the same facts ``_loop_breaker_facts`` serves in full."""
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
        the verbatim remedy+census re-issue (which each rejection >=3 would
        otherwise repeat byte-for-byte but for the count digit). States: the
        consecutive + per-round counts, the rejection's identifying first
        clause, where the full remedy text and the live-surface census were
        already served (rejections 1 and 2 — both still in this
        conversation), the executor-determinism fact, and the round's live
        alternatives as compact counts + the computed-column names. All
        facts the harness knows; no command, no candidate gate."""
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
        absent column; the old first-8-alphabetical fid sample was
        noise), served ONCE in full. 3rd consecutive
        onward: the repetition fact + compact state
        (``_repeated_rejection_state``) replaces the verbatim remedy+census
        re-issue. A successful program resets the CONSECUTIVE streak (the
        next same-class error gets the full text again); the per-round
        class count keeps accumulating. A structural op-SHAPE error
        (unknown top-level key / bad op / missing ref / a handle used where
        a literal fid is required / a syntax error, whose echoed offending
        line is its own fix material) repeats for program-shape reasons, and
        an inventory of fids is pure token noise there (gt_mi first-dozen
        audit S6, 11 observed) — those get only the count."""
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
        # A syntax error is a code-shape fact too: the echoed offending
        # line (scan_block serves it with a caret) is the fix material,
        # and a fid inventory is pure token noise beside it.
        _SHAPE_MARKERS = ("top-level key(s)", "not a known op",
                          "lacks its required 'ref'", "unknown ref",
                          "unknown handle", "is a result handle",
                          "syntax error")
        if any(m in err for m in _SHAPE_MARKERS):
            return err + f" This error class has now occurred {_n}x."
        if _consec >= 3:
            return _repeated_rejection_state(err, _n, _consec)
        return err + _loop_breaker_facts(compute_namespace or {}, _n)

    # Cross-call ref persistence (S4): raw results of every successful
    # program this round, keyed by ref — later calls resolve earlier $rN
    # from it (resolution-only; never re-rendered).
    carry_results: dict = {}

    # A HANDLE IS A QUERY, NOT A SNAPSHOT. Every op the mission has run,
    # in order, plus the surface version those results reflect. A flight
    # moves the surface; the next read of any handle re-runs the history
    # against the world the flight folded, so a handle cannot answer about
    # ground that had not been covered when it was bound.
    #
    # calcity::200#198 is the cost of the snapshot: the model bound
    # red_cars = find("car", color="red", near=poppy) BEFORE flying the
    # ground the target sat on, explored three more houses, and committed
    # out of that frozen list -- while the ground-truth car, folded five
    # times at conf 0.88 and 1.5 m from truth, was never in any row it
    # could see.
    #
    # Ops are pure reads over (cands, anchors), so replaying them is safe
    # and order-correct by construction: the history IS the dependency
    # order. A replay that raises leaves the old rows standing, because a
    # stale handle beats no handle.
    #
    # THE REPLAY IS PER HANDLE, ON READ. Replaying the whole history at
    # once cost a flight the whole mission so far: every compare the mission
    # had run, over every row of the flown surface, before the block that
    # flew could finish -- 32 s on the Cal City surface with 60 blocks of
    # history, and growing with every block. A read of one handle re-derives
    # that handle and the ops it reads, and nothing else; the ops are pure,
    # so the rows come out the same as a full replay would have made them.
    _op_history: list = []           # every op, in execution order
    _op_reads: list = []             # per op: the handle bases its args name
    _producers: dict = {}            # ref -> indices into _op_history
    # The surface version the carry is ARMED for: the first read under a new
    # version claims it and clears the fresh set, and every read after that
    # re-derives what it reads. Before that first read a handle answers with
    # the rows it held, the same as before.
    _carry_version = [0]
    _fresh: set = set()              # refs re-derived, or written, under it
    # A FLIGHT-MINTED handle is not a query. It records the rows ONE
    # flight added, which is a fact about an event and cannot re-derive:
    # replaying ops over the new surface would hand "the rows this flight
    # added" every row a later flight added too.
    _event_handles: set = set()

    def _refs_read(op: dict) -> set:
        """The handle bases an op's args name -- every string that could
        reach ``run_program``'s resolver -- plus ``*`` for the
        previous-result words, which read whatever ran before them."""
        found: set = set()

        def _walk(v):
            if isinstance(v, str):
                m = re.match(r"^([^.\[\]]+)", v.lstrip("$"))
                b = _ref_key(m.group(1) if m else v)
                if b.lower() in _HANDLE_WORDS:
                    found.add("*")
                elif re.fullmatch(r"r\d+", b) or b in carry_results:
                    found.add(b)
            elif isinstance(v, dict):
                for x in v.values():
                    _walk(x)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    _walk(x)

        for k, v in op.items():
            if k not in ("ref", "op"):
                _walk(v)
        found.discard(str(op.get("ref")))
        return found

    def _program_reads(program) -> set:
        """The carry handles a program reads and does not define."""
        ops = [o for o in ((program or {}).get("ops") or [])
               if isinstance(o, dict)]
        defined = {str(o.get("ref")) for o in ops}
        return set().union(*(_refs_read(o) for o in ops)) - defined

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
            if r == "*":
                todo.extend(_producers)
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
            # an op reads one, and the surface INVITES that: the flight ack
            # advertises rN[0] and order_by(rN, ...), and the catch note
            # advertises look(rN[a:b]). Measured: 34% of basket episodes ran
            # such an op, and every one of them froze this replay for the
            # rest of the mission.
            _out = run_geom_program_tool(
                {"program": {"ops": [
                    op for op in _op_history if str(op.get("ref")) in need]}},
                namespace=compute_namespace or {}, raw_sink=_sink,
                seed_results={h: v for h, v in carry_results.items()
                              if h not in need})
            if isinstance(_out, dict) and _out.get("error"):
                # An executor refusal is a failed replay too: the handles
                # keep their last rows and stay UNCLAIMED, so the next read
                # tries again. Accepting it as success marked stale rows
                # fresh for the rest of the mission, silently.
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

    def _alias_deduped_refs(prev_ops, cur_program) -> None:
        """Point THIS emission's grammar refs at the handles the first run
        stored under its own labels.

        Joined on ``_dedup_ref_norm`` -- the SAME normalization the dedup key
        is computed with -- so the alias cannot drift from the identity that
        justified it. A label already live is never overwritten: the emission
        that defines it owns it.
        """
        prev_map = _dedup_ref_norm(prev_ops)
        cur_map = _dedup_ref_norm((cur_program or {}).get("ops"))
        for norm, cur_ref in cur_map.items():
            prev_ref = prev_map.get(norm)
            if prev_ref is not None and cur_ref not in carry_results \
                    and prev_ref in carry_results:
                carry_results[cur_ref] = carry_results[prev_ref]

    def _dispatch_analyze_scene(a, _ns=(compute_namespace or {}), *,
                                answer_ref=""):
        _key = _program_dedup_key((a or {}).get("program"),
                                  carry_refs=carry_results.keys())
        _prev = _programs_seen.get(_key) if _key is not None else None
        if _prev is not None and _prev[0] == _surface_version[0]:
            # ONE-LINE REFERENCE, never a payload and never a lockout: the
            # original's rN handles stay live in carry_results (a new
            # program can slice/chain them), so nothing is locked away —
            # only the re-render is defunded.
            #
            # THE DEDUP OWES THIS EMISSION ITS OWN LABELS. Identity is judged
            # with ref labels NORMALIZED OUT (an incrementing rN defeated the
            # byte-identity guard entirely, 356/357 calls), so the re-issue's
            # labels are not the ones the first run stored under. A caller
            # that resolves a slot from `carry_results[<its own ref>]` then
            # finds nothing — and reports it as nothing, which is not the
            # same fact. Measured on calcity_v3base: `go to the intersection
            # of A and B.` answered "0 distinct crossings" for two roads the
            # SAME episode had just resolved with `find the intersection of A
            # and B.`, 49 says over 47 episodes, the prior success being the
            # precondition. Alias by the key's own emission-index
            # correspondence so a deduped program is indistinguishable from a
            # re-run to everything downstream of it.
            _alias_deduped_refs(_prev[3], (a or {}).get("program"))
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
            # (pilot-4: 11 dedup returns showed the model nothing at all)
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
        _ensure_fresh(_program_reads((a or {}).get("program")))
        out = run_geom_program_tool(a, namespace=_ns, raw_sink=_raw_sink,
                                    seed_results=carry_results,
                                    preflight=preflight,
                                    answer_ref=answer_ref)
        if isinstance(out, dict) and "error" in out:
            out = {**out, "error": _loop_broken_error(str(out.get("error")))}
        if diag is not None:
            try:
                diag["sel_geom_calls"] = int(diag.get("sel_geom_calls", 0)) + 1
                errored = isinstance(out, dict) and "error" in out
                diag["sel_geom_last_errored"] = bool(errored)
                if errored:
                    diag["sel_geom_last_error"] = str(out.get("error"))[:200]
                elif isinstance(out, dict):
                    diag["sel_geom_ok"] = int(diag.get("sel_geom_ok", 0)) + 1
                tfid = _terminal_fid_of_geom_result(out)
                if tfid is not None:
                    diag["sel_geom_terminal_fid"] = tfid
            except Exception:  # noqa: BLE001 — diagnostics never perturb the round
                log.debug("geom observer stamp failed (non-fatal)", exc_info=True)
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
                                        ((a or {}).get("program") or {})
                                        .get("ops"))
            carry_results.update(_raw_sink.get("results") or {})
            _fresh.update(_raw_sink.get("results") or {})
            _ops_now = ((a or {}).get("program") or {}).get("ops")
            if isinstance(_ops_now, list):
                for _op in _ops_now:
                    if isinstance(_op, dict):
                        _producers.setdefault(str(_op.get("ref")), []).append(
                            len(_op_history))
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
                # The episode's acquisition record: the same column vocabulary
                # ``_loop_breaker_facts`` discloses, unioned across the episode
                # and carried to metrics.evidence_columns.
                evidence_columns.update(
                    _surface_census_counts(compute_namespace or {})[2])
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

    # ONE discourse dict for the mission (what set is in scope, how far
    # past a handle's head the agent has asked) and the mission-global rN
    # chain that carry_results resolves against.
    discourse: dict = _new_discourse()
    plan_rn: list = [0]
    # THE EPISODE'S SCRIPT, append-only. Each accepted block is recorded in
    # order and nothing already executed is ever re-run: re-running turn 1's
    # find on turn 6 would re-mint handles and diverge from what the model
    # was shown. The artifact is a record to READ -- one program per episode
    # instead of tens of disconnected calls -- and its map half is a pure
    # function of the bake, so the analysis half re-reads deterministically.
    _episode_script: list = []
    # EVERY rN THE CODE SURFACE HAS IN PLAY: the handles earlier blocks
    # bound, plus every rN the block now running spells. There are two
    # writers into the rN namespace -- the compiler (which already skips
    # these) and the flight mint below -- and only one of them was reading
    # the model's names. The dispatcher refreshes it per block; the say
    # channel leaves it empty and the mint behaves as it always did.
    _code_names: set = set()
    # The handles flights minted DURING the block being dispatched, so the
    # code scope can bind them: the flight's report names the handle, and a
    # name the surface serves has to be a name the surface can resolve.
    _code_minted: list = []
    # The handle the LAST flight of this sentence minted over its own new
    # rows, applied to the discourse AFTER update_discourse so the flight —
    # the most recent thing that happened — owns "those".
    _flight_minted: list = [None]
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
        added. ZERO new rows mints NOTHING and leaves "those" standing --
        a live-looking empty handle is this repo's named hazard, and rule
        T's own "non-empty only" rule says the same.
        """
        if not _new_fids:
            return None
        # THE MINT SKIPS THE NAMES IN PLAY. The compiler already refuses to
        # mint an rN a model name spells (the two namespaces are disjoint
        # by construction); this writer did not, so a flight could take
        # `r3` while the model held r3 for something else -- and the report
        # then told it "r3 names the rows this flight added" while r3 still
        # resolved to what the model bound. A wrong commit, with nothing
        # anywhere saying so.
        while f"r{plan_rn[0]}" in _code_names:
            plan_rn[0] += 1
        _h = f"r{plan_rn[0]}"
        plan_rn[0] += 1
        _code_names.add(_h)
        _code_minted.append(_h)
        carry_results[_h] = [_ns_rows[_f] for _f in _new_fids]
        _event_handles.add(_h)
        _fresh.add(_h)
        # bound NOW so a later step's refusal still leaves the scope on
        # what was shown, and re-applied after the plan's own discourse
        # writes below.
        discourse["those"] = _h
        _flight_minted[0] = _h
        return _h, (f"{_h} — the {len(_new_fids)} rows this {_what} added; "
                    f"the handle is bound, so {_h}, {_h}[0] and "
                    f"{_h}.order(...) all reach them")

    def _fly_moves(_accepted: list, _item_errors: list,
                   cover: str | None = None) -> dict:
        """The act.move core — each entry FLIES the moment it lands, in
        order; after every successful flight the analyze surface
        re-derives from the live graph, and the rows the flight ADDED are
        minted as an ordinary rN result handle. (The retired mission-act
        branch's body, plus the mint; the sentence channel is its one
        caller.)

        ``cover`` names the place an ``explore`` is covering, and turns the
        SAME loop into ONE pass over it. Every entry is a stop and the pass
        ends on the last of them. Three things change, and they are the
        three that make a pass one act rather than N flights:

          * ONE picture, not one per stop. A per-stop view would put a
            dozen images into a single turn and spend, in one act, a large
            share of the token budget the rest of the mission has to fit
            inside. The pass gets one wide frame with its stops badged on
            it, and close looking stays what ``look()`` is for.
          * ONE handle, over everything the whole pass folded. Minting per
            stop hands the model a name for stop 5's rows and calls it the
            explore's product.
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
                # whole pass is on the tablet from its first leg and the
                # C-12 flies it in one go instead of a waypoint at a time.
                # The bench's fly reads ``target_fid`` alone, and nothing
                # of this reaches the model.
                _item["pass_ahead"] = [[float(_xy[0]), float(_xy[1])]
                                       for _xy in map(_stop_xy, _queue) if _xy]
            if mission_flights >= mission_spec.flight_budget:
                # The backstop fires without a NUMBER: the model never
                # hears a budget fact (CLAUDE.md law); the count logs.
                log.warning("%s flight backstop hit (%d flights)",
                            rid_tag, mission_spec.flight_budget)
                _item_errors.append({"target_fid": _item["target_fid"],
                                     "error": "not flown."})
                continue
            if (mission_spec.deadline_epoch is not None
                    and time.time() >= mission_spec.deadline_epoch):
                # THE SAME BACKSTOP, IN THE UNIT THAT KILLS. A count of
                # stops cannot bound a count of seconds: per-stop cost
                # moves with tile density, watchlist width and the serving
                # plane's health, so the flight arm above is ~11 h of wall
                # at one rate and minutes at another. calcity::6#6 and
                # ::250#248 flew 262 and 264 stops of ONE explore at ~40 s
                # each, hit the harness's 4 h SIGKILL, and were the only
                # two episodes of 209 to record no prediction row at all --
                # while the flight arm sat 738 flights away from firing.
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
            _via = _item.pop("via", None)
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
            if _via:
                _report.setdefault("via", _via)
            if _snap_from:
                _report.setdefault(
                    "note",
                    f"{_snap_from} is {_item['target_fid']}"
                    "'s own position -- flown as that fid.")
            if "error" not in _report:
                if (not _report.get("new_detections") and not mission_sam
                        and not withheld("look_for")):
                    # the watchlist gate is mechanical and invisible:
                    # pilot-3 measured 240 zero-detection flights, 99%
                    # with an empty watchlist, and 388 altitude
                    # misdiagnoses with no exit. State the mechanism.
                    _wl_note = ('the watchlist is empty -- nothing is '
                                'segmented on any flight; call '
                                'look_for("<class>") to see detections')
                    _report["note"] = (f"{_report['note']} {_wl_note}"
                                       if _report.get("note")
                                       else _wl_note)
                # THE FLIGHT SHOWS WHAT IT SAW. The harness already
                # renders this exact view to segment it and then throws the
                # pixels away, handing back a COUNT of the classes it was
                # told to watch. That count is what makes "0 detections"
                # read as absence when it means "nothing you asked for":
                # a log study over 146 reached-then-missed episodes found
                # the model trusting it and leaving a target it was
                # standing on. And the model pays to undo it -- 45% of
                # 4,691 flights were followed immediately by a look or map
                # call asking to see where it had just gone, ~2 turns an
                # episode spent re-requesting a picture that existed.
                # Serving the view costs no render the harness did not
                # already do, and the caption carries the world window, so
                # what the model sees converts straight into a point it can
                # fly to.
                #
                # AND THE THING IT FLEW TO IS OUTLINED ON IT. This is the
                # only picture that reaches the model BEFORE its next aim
                # without costing a turn — tool results within a turn all
                # arrive together, so a `look at` cannot inform an aim in the
                # same turn — and its caption already names the world window
                # in metres. Drawing the target's footprint on it is what
                # makes the free picture the pointing instrument: the shape,
                # the frame and the metres land on ONE image at the moment
                # the model decides whether the point it named was the part
                # it meant. Measured on the aimed commits of one arm, the
                # model's own point beat the fid's centroid 47.0% to 31.2%
                # while the best point on that same mask scored 65.8% — it
                # picks the right thing and the wrong part of it.
                #
                # THE VIEW IS THE TARGET'S OWN LOOK: centred on the row the
                # flight asked for, its footprint the boundary, so what the
                # model reads after a flight is the thing it asked to see
                # and nothing about where the aircraft was. A shape too big
                # for its window still rides: ``_proj_poly`` CLIPS it and
                # draws the in-frame arcs, and the caption's own
                # ``subject_outline_cut`` says the footprint runs past the
                # frame. A target that is not a row (a stop flown again)
                # centres on the stop's point under the token it was asked
                # for. Badged with where the flight put the aircraft, the
                # picture was read as a placement to critique (sortie
                # 20260911T201524Z run 3, six requests for one row).
                #
                # A PASS RENDERS ONCE, not once a stop — see ``cover``.
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
                # look(rN) reach them while the drone is
                # still standing over them. Without it the flight returned a
                # COUNT and no address, and the model had to re-issue
                # find("<noun>") over the whole accumulated world and hope
                # the new rows ranked in (measured on pilot-10: 126 of 300 episodes
                # reason explicitly about a handle the flight never named).
                # The surface diff, not the engine's detection count: those
                # differ (a re-sighting merges into a row that already
                # exists) and the addressable rows are the truth here. A
                # PASS mints once, around the whole of itself.
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
        # A PASS SHOWS ITS CATCH, not its coverage. A wide frame of the
        # whole pass is the wrong picture for the question a pass is asked
        # (three pools kilometres apart are a few pixels each), so no
        # coverage frame is drawn HERE; the move branch instead serves one
        # crop per row the pass ADDED (_CATCH_CROPS of them, pass order) --
        # the image budget spent on the rows the coverage found, which is
        # what vision is actually needed for. The 140#138 reflies measured
        # the alternative: the model paged look(rows) by hand or skipped
        # it and committed [0].
        return _ack

    # HALF THE OBSERVATION'S LONG SIDE: a pass's stop spacing.
    # ``observation_extent_m`` returns the sensor footprint's LONG side
    # (320.82 m on this bake; the short side is 240.77), so half of it sits
    # under the short side for any sensor below 2:1 -- which is what lets a
    # ONE-DIMENSIONAL walk cover a shape that is not one: consecutive stops
    # overlap on the minor axis too. The number is the run's own, never a
    # constant here: a bake or an altitude change moves it.
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
        that is not an answer either (smoke 25712881's pre-explore pose was
        the episode start, 4 km from the target).

        The note carries HOW MUCH of the shape the stops reached. The
        stops run end to end along the shape, so the walk's own length is
        its reach and each stop is credited with one observation of it --
        a pass the engine's stop bound truncated reads as truncated
        instead of as "that is all there was".

        A SET is one place to cover: many members become ONE pass over
        their union, per-member walks chained end to end so the whole
        thing flies a line. The single-member path below is byte-for-byte
        the pass it always was.
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
        # 147#145's method-correct retry explored 2 of 85th Street's 10
        # fragments, committed on the covered stretch, and missed by
        # 573 m -- the target sat two segments further on. Which
        # fragments no pass has covered is world-state, so the pass note
        # says it; the model decides whether to keep covering. Computed
        # HERE, before the flight refreshes the surface views.
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
        # ponytail: O(n*m) greedy chaining; a TSP is not worth it under a
        # 24-stop pass.
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
        _set_cap = int(getattr(mission_spec, "flight_budget", 0)
                       or _COVER_SET_MAX_STOPS)
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
            # A MEMBER SMALLER THAN ONE OBSERVATION WALKS ZERO METRES, so a
            # pass over five houses summed to 0 and reported "covered 0 m of
            # r4 across 5 segments in 3 stops" -- the one receipt on the
            # behaviour being installed, reading as a null result. Measured:
            # 40 of 340 explore acks in the basket said it. Length is the
            # wrong fact for a set of footprints; the count is the right one.
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

    def _sub_anchors(prog: dict, ties: dict) -> dict:
        """Substitute grounded anchor slots into compare refs: one fid
        stays a labeled single ref; top-tie multiples become the labeled
        UNION ref {ref:[fid,…], as} — one label, exact min gap over
        members, never a per-instance fan-out."""
        import copy as _copy
        out = _copy.deepcopy(prog)
        for _op in out.get("ops") or []:
            refs = (_op.get("args") or {}).get("refs")
            if not isinstance(refs, list):
                continue
            for _i, _el in enumerate(refs):
                if not isinstance(_el, dict):
                    continue
                _r = str(_el.get("ref", ""))
                if _r.startswith("<anchor:"):
                    slot = _r[len("<anchor:"):-1]
                    fids = ties.get(slot) or []
                    refs[_i] = {"ref": fids[0] if len(fids) == 1 else fids,
                                "as": _el.get("as")}
        return out

    def _watch_apply(step: dict) -> dict:
        """`look for` — the ONE segmenter channel (latest declaration of a
        class wins), reached by the bare form directly and by the
        near-form on its way to the boost. The watchlist only GROWS:
        `stop watching for` was said 5 times in 3,044 episodes, and a
        removal buys a cheaper segmenter pass, never an answer."""
        adds = [dict(e) for e in step.get("add") or []]
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

    def _boost_apply(step: dict, where: list) -> dict:
        if boost_queue is None:
            return {"error": (
                "targeted boost is not wired on this run: call "
                "look_for(\"<class>\") without near= instead — the "
                "watchlist is active at every later flight.")}
        res = boost_queue(list(step.get("concepts") or []), where) or {}
        if "error" not in res and res.get("queued"):
            # THE NEAR-FORM MEANS WHAT A PLANNER MEANS: standing attention
            # on the class, with localized emphasis where it was asked.
            # One verb, one semantics. Before this, `look for X near Y.`
            # was a one-shot that left the standing watchlist EMPTY -- 40
            # of the facing family's 49 never-armed episodes used ONLY
            # this form and read its receipt as an arm
            # (f40088a:calcity_full_replyrepair, arming read), and both
            # corrective notes measured behavior-flat on matched episodes
            # (schema form n=419: boost-dark 35 vs 38). An affordance
            # mismatch is fixed in the tool, not narrated around: the
            # model's own sentence names the class, so arming it IS the
            # sentence's meaning -- nothing arms that the model did not
            # say.
            _adds = [{"class": str(c), "role": "target"}
                     for c in step.get("concepts") or [] if c]
            if _adds:
                _w = _watch_apply({"add": _adds})
                res["watching"] = _w.get("error") or _w.get("now")
            res["note"] = ("boost queued -- extra segmentation at the "
                           "asked spot fires on your next flight; the "
                           "class is also on the standing watchlist for "
                           "every later flight")
        if "error" not in res:
            # a retro boost ingests detections NOW: refresh the retrieval
            # surface like a flight does, or the new fids are findable
            # only after the next flight (measured on the pilot: 514 new
            # detections a following find could not see).
            (compute_namespace or {}).update(_surface_from_ctx(ctx))
            _rebuild_ns_rows()
            _surface_version[0] += 1
        return res

    def _crops_of(fid, cid: str, marks: list | None = None,
                   preamble: str = "") -> dict:
        # A LIST asks for one crop PER ROW -- each its own subject, each at
        # the single-look scale where a distinction is resolvable. A bare
        # fid keeps the one-crop form (and its marks) unchanged.
        _fids = [str(f) for f in fid] if isinstance(fid, (list, tuple)) \
            else [str(fid)]
        # MARKS RIDE EVERY CROP. The gate was len(_fids) == 1, so the
        # multi-row look -- the one the worked example teaches and the one
        # a candidate set is read through -- silently dropped them. What
        # marks= is FOR is telling ten candidate crops apart.
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
                # The renderer frames the hull of the request's points and
                # then CLIPS at the window: a mark outside the delivered
                # frame is named in the legend text and no badge is drawn
                # for it. The legend (truthful) and a caller counting its
                # own request (not) served the same turn on pilot-10 -- 463
                # rows claimed badged, 215 drawn. The legend IS the drawn
                # set, so count it and let no caller report the ask as the
                # delivery.
                ack["badged"] = sum(
                    len(c.get("legend") or []) for c in crops
                    if isinstance(c, dict) and c.get("jpeg"))
                # A requested mark goes undrawn for FOUR different reasons
                # and only ONE of them is distance. Serve the renderer's own
                # out-of-frame list rather than a residual: a caller
                # subtracting badged from requested attributes a surface
                # eviction (``skipped``, already on this ack), a
                # footprint-in-frame centroid-out mark, and a badge-cap drop
                # all to "further away than one frame reaches" -- a row 6 m
                # from the subject in a 30 m frame gets called distant, and
                # the legend the note points at does not name it.
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
        """"look at them." — ONE frame centred on the SET in scope, badging
        the rows that fall inside it.

        The frame and the result table are the SAME object: the marks are
        the handle's rows, in the handle's own order, so every badge is a
        row the model can still name afterwards (its fid rides the legend
        text, and the handle reaches the whole set again). That
        correspondence is SERVED, on the message the pixels are in — not
        left to be inferred from a picture. Any handle qualifies, so a set
        minted by something other than a find badges the same way.

        The rank-0 row is the frame's subject (the standing order's own
        answer, and the row the renderer's recenter cap keeps in frame).
        EVERY ROW IS NOT AN INVARIANT: the window fits the set's hull only
        up to the observation and pixel ceilings, and a set spread wider
        than the one that binds is clipped to rank 0's neighbourhood — so
        the note counts what the renderer DREW (``badged``) and names the
        rows the renderer itself reports OUT OF FRAME (``not_drawn``).
        Never a residual of requested-minus-drawn: a row can also be gone
        from the surface (``skipped``, served on this same ack), be in
        frame with its centroid outside it, or lose its badge to the cap,
        and calling any of those "further away than one frame reaches" is
        the same false-precision the drawn count was added to remove. Every
        budget is stated only WHEN IT BITES — the same doctrine the 16-crop
        cap follows: a limit the model has not hit costs context every
        round to teach, and a silent drop would read as "those rows are not
        there"."""
        rows = [r for r in (_carry_read(handle) or [])
                if isinstance(r, dict) and r.get("fid")]
        if not rows:
            return {"error": (f"{handle} holds no rows to look at; name a "
                              "handle that has some, or find() one first")}
        # ONE CROP PER ROW, not one frame over the set. The set frame was
        # served at MAP SCALE and could not resolve what these cues turn on:
        # median 300 m window, 70% of them 300 m or wider, the widest
        # 37,243 m at 36.37 m/px. A single-row look is a median 78.6 m at
        # 0.15 m/px. So the only form that named more than one row could not
        # tell them apart, and the only form that could resolve a difference
        # named exactly the row already picked -- which is why the run flew
        # 2,224 times against 304 looks: a flight was the only way to get a
        # sharp picture of a candidate, and flying to one makes it the
        # answer.
        #
        # This is not new capability. It is the crop every flight already
        # ships, without the flight. One badge per image makes badge->fid
        # unambiguous by construction rather than by a legend that got it
        # wrong 33.8% of the time, and the answer to "which one has a pool?"
        # becomes a fid that is live on the surface.
        fids = [_crop_token_of(r, str(r["fid"]) in _ns_rows) for r in rows]
        ack = _crops_of(
            fids, cid, marks=marks,
            # NO SPELLING IS INVITED HERE. This said "of {handle}, …", which
            # is sentence-surface phrasing AND names the EXECUTOR's ref: the
            # receipt rename maps that ref back to the model's own label
            # everywhere else, so a code-surface model was being invited to
            # say a handle it never bound. Each crop names its own row, and
            # naming a row is what the next call needs.
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
        specs, m_skipped, m_notes = _build_map_requests([_req])
        ack: dict = {"maps": len(specs)}
        if m_skipped:
            ack["skipped"] = m_skipped[:20]
        if m_notes:
            ack["ref_notes"] = m_notes[:20]
        if not specs:
            ack["error"] = ("; ".join(m_notes) if m_notes
                            else "no resolvable map center")
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
        """A say error result, routed through the SAME escalating loop
        breaker the analyze path uses — pilot-2 measured sentences
        repeated 23x with the guard never firing because say errors
        bypassed it."""
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
        """Rule N's retry-fuel killer: the nearest NAMED rows to a
        grounding that matched nothing (fuzzy, top 3)."""
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

    def _run_plan(plan: dict, text: str) -> dict:
        """Execute a compiled plan and echo it. THE shared body: the say
        surface and the code surface differ only in what produced the
        plan, so everything mission law lives in -- grounding under the
        name-match floor, the analyze execution, flights, the watchlist,
        the discourse writes -- is reached by both and duplicated by
        neither."""
        ack: dict = {"said": text}
        # `more of rN.` with nothing left to page. compile_sentence sees the
        # parse node, the discourse dict and rn_start and nothing else — no
        # row count reaches it — so the length check lives HERE, where the
        # rows are. plan["widened"][h] IS the window's start index (the
        # compiler writes seen + _BROAD_HEAD_ROWS, which is exactly `lo`).
        # A refusal returns before _update_discourse, so the handle's counter
        # does not move and a repeat gets the same stable answer. An unknown
        # handle (_held == 0) falls through to the resolver's own error.
        # Two ways to have nothing left: the window walked past the end, or
        # the handle was never truncated at all — the serve law renders a
        # result of _BROAD_DUMP_SOFT_ROWS rows or fewer WHOLE, so `more of`
        # there would re-serve rows already on screen as if new (207 of 451
        # flown calls were exactly that).
        for _h, _lo in (plan.get("widened") or {}).items():
            _held = len(_carry_read(_h) or [])
            if _held and (_held <= _BROAD_DUMP_SOFT_ROWS or _lo >= _held):
                return _plan_err(ack, (
                    f"all {_held} row(s) of {_h} are already served -- there "
                    f"is no further block to serve. Narrow it "
                    f"(order_by({_h}, color=\"<colour>\"), "
                    f"order_by({_h}, nearest=\"<fid>\"), "
                    f"narrow({_h}, inside=\"<place>\"))"
                    + ("." if withheld("look") else f" or look({_h}).")))
        slot_fid: dict[str, str] = {}
        slot_ties: dict[str, list] = {}
        ground_notes: list[str] = []
        if plan["groundings"]:
            # plumbing finds: raw, never dedup'd, never rendered
            _sink: dict = {}
            g_prog = {"program": {"ops": [
                {"ref": f"r{i}", "op": "find",
                 "args": {"provenance": g["provenance"],
                          "query": g["query"]}}
                for i, g in enumerate(plan["groundings"])]}}
            g_out = run_geom_program_tool(
                g_prog, namespace=(compute_namespace or {}),
                raw_sink=_sink, preflight=preflight)
            if isinstance(g_out, dict) and "error" in g_out:
                return _plan_err(ack, str(g_out["error"]))
            g_res = _sink.get("results") or {}
            _WHY_VERB = {"stage": "standing at", "move": "flying",
                         "map": "mapping around", "crops": "looking at",
                         "boost": "looking around", "facing": "facing"}
            for i, g in enumerate(plan["groundings"]):
                rows = [r for r in (g_res.get(f"r{i}") or [])
                        if isinstance(r, dict) and r.get("fid")]
                # the same floor the find head's retrieval uses: a
                # below-threshold "best match" is a guess, and pilot-3
                # measured 5 silent mis-flights (to 193 m) plus 527-row
                # "matches" of junk queries riding rank-0
                rows = [r for r in rows
                        if r.get("name_match") is None
                        or float(r["name_match"]) >= _FUZZY_FLOOR]
                _what = g["query"] or "that"
                if not rows:
                    return _plan_err(ack, (
                        f"rule N: nothing named \"{_what}\" is on "
                        f"this surface{_closest_names(_what)}; say it "
                        "differently, say a fid off a served row, or "
                        "perceive more first"))
                # every slot keeps only the TOP name_match tie group
                # (anchors: the labeled union; the whole fuzzy set
                # walled compare with 199-instance fan-outs on pilot-1)
                _top = rows[0].get("name_match")
                ties = [str(r["fid"]) for r in rows
                        if r.get("name_match") == _top] \
                    or [str(rows[0]["fid"])]
                slot_ties[g["slot"]] = ties
                if g["why"] == "stage" and len(ties) > 1:
                    # rule F refuses only a genuine TOP-MATCH tie (the
                    # pilot-2 form refused any multi-row name — 31
                    # refusals, one naming 219 places)
                    _f = ", ".join(ties[:4])
                    return _plan_err(ack, (
                        f"rule F: a viewpoint is one point; "
                        f"\"{_what}\" has {len(ties)} equally good "
                        f"name matches ({_f}); say which one — a fid "
                        "works"))
                slot_fid[g["slot"]] = str(rows[0]["fid"])
                # A token NO matched row carries -- the same note the
                # query find serves, on the grounding path. Without it,
                # glue landing in a GROUNDING ("go to <name>", "show
                # me <place>", "from <site>, looking …", an anchor)
                # rides the top name_match tie and binds silently, and
                # episodes carrying glue score lower. A NOTE,
                # never an error and never a filter: the match stays
                # untouched because a real query often names an attribute
                # no row spells.
                _dead = _unmatched_query_tokens(g["query"] or "", rows)
                if _dead:
                    ground_notes.append(
                        f"no row matching \"{_what}\" carries "
                        + ", ".join(repr(t) for t in _dead[:4])
                        + " -- those words did not narrow anything")
                # a covered slot skips this note: the pass flies EVERY
                # top-tie segment, so "flying the best name match" would
                # state the thing it just stopped being true of
                if g["why"] not in ("anchor", "cover") and len(rows) > 1:
                    ground_notes.append(
                        f"{len(rows)} rows match \"{_what}\"; "
                        f"{_WHY_VERB.get(g['why'], 'using')} the best "
                        f"name match, {rows[0]['fid']} — say a fid to "
                        "choose precisely")

        def _refval(ref):
            """A compiler step ref -> the literal token to execute with."""
            if isinstance(ref, str):
                return ref
            kind, val = ref
            if kind == "fid":
                return slot_fid[val]
            if kind == "xy":
                # two bare numbers in a place slot = a world point,
                # spoken ("go to 700 300."); the canonical token flies
                # through the ordinary snap path
                return format_xy_token(val[0], val[1])
            return str(val)  # "lit" (a spoken fid) / "it"

        final_rows = None
        _flight_minted[0] = None   # this sentence's flight only
        for step in plan["steps"]:
            do = step["do"]
            if do == "stage":
                facing = (step["facing"]
                          if isinstance(step["facing"], str)
                          else _refval(step["facing"]))
                st = _stage_apply([{"at": _refval(step["at"]),
                                    "facing": facing}])
                if not st.get("staged"):
                    return _plan_err(ack, (
                        "rule F: the stance did not bind: "
                        + str(st.get("rejected") or st.get("error"))))
                ack["stance"] = st
            elif do == "analyze":
                prog = step["program"]
                prog = _sub_anchors(prog, slot_ties)
                out = _dispatch_analyze_scene(
                    {"program": prog}, answer_ref=str(plan["last_ref"] or ""))
                if isinstance(out, dict) and "error" in out:
                    # in-register: keep the mechanical fact, drop the
                    # op-signature tail (it teaches op names the language
                    # cannot say — measured register break on the pilot)
                    _e = str(out["error"]).split("; signature: ")[0]
                    return {**ack, "error": _e}
                ack["result"] = out
                plan_rn[0] = max(plan_rn[0], int(plan.get("rn_end") or 0))
                final_rows = carry_results.get(plan["last_ref"])
            elif do == "move":
                if step.get("cover"):
                    # AN EXPLORE IS ONE ACT MADE OF FLIGHTS, expanded HERE
                    # because none of it exists at compile time: the stops
                    # come off each place's own cells and the chain off the
                    # live pose. A SET is one place to cover, and its
                    # members arrive three ways -- a grounded NAME's whole
                    # top tie group, a result handle's fid-bearing rows, a
                    # compiled list -- all expanded where slot_ties and the
                    # live handles are.
                    _to = step["to"]
                    _raw_members: list = []
                    _empty_handle = None
                    for _s in (_to if isinstance(_to, list) else [_to]):
                        if (isinstance(_s, tuple) and len(_s) == 2
                                and _s[0] == "fid"
                                and len(slot_ties.get(_s[1]) or []) > 1):
                            _raw_members.extend(slot_ties[_s[1]])
                            continue
                        _t = _refval(_s)
                        if isinstance(_t, str):
                            _t, _stop_err = _flown_stop(_t)
                            if _stop_err:
                                return _plan_err(ack, _stop_err)
                            if _is_handle_token(_t):
                                _hr = [str(r["fid"]) for r in
                                       (_carry_read(_t) or [])
                                       if isinstance(r, dict)
                                       and r.get("fid")]
                                if not _hr:
                                    _empty_handle = _t
                                    break
                                _raw_members.extend(_hr)
                                continue
                        _raw_members.append(_t)
                    if _empty_handle is not None:
                        return _plan_err(ack, (
                            f"{_empty_handle} holds 0 fid-bearing row(s) "
                            "here; there is nothing to cover."))
                    _uniq: list = []
                    for _t in _raw_members:
                        if _t not in _uniq:
                            _uniq.append(_t)
                    _accepted, _item_errors = _mission_sanitize_moves(
                        _uniq, staged_viewpoints)
                    if not _accepted:
                        _msg = next(
                            (str(e.get("error")) for e in _item_errors
                             if isinstance(e, dict) and e.get("error")),
                            "nothing to cover")
                        return _plan_err(ack, _msg)
                    _accepted, _cov_note, _cover, _segs = _cover_items(
                        _accepted, _said=step.get("said_to"))
                    if not _accepted:
                        return _plan_err(ack, _cov_note)
                    mv = _fly_moves(_accepted, _item_errors, cover=_cover)
                    ack["explored"] = mv
                    if "error" in mv:
                        return _plan_err(ack, str(mv["error"]))
                    if _segs > 1:
                        mv["segments"] = _segs
                    # A TRUNCATED PASS READS AS TRUNCATED. ``_cov_note`` is
                    # built from the stops the pass PLANNED, so a pass a
                    # backstop cut short would report the whole walk's reach
                    # and read as "that is all there was" -- the same defect
                    # ``_cover_items`` already names for its own stop bound.
                    # ``_cover_ack`` counts the stops actually flown, so the
                    # short case says what happened instead of over-claiming.
                    if mv.get("stops", len(_accepted)) < len(_accepted):
                        _cov_note = (f"the pass ended after "
                                     f"{mv.get('stops', 0)} of its stops.")
                    if _cov_note:
                        mv["note"] = (f"{_cov_note} {mv['note']}"
                                      if mv.get("note") else _cov_note)
                    # THE PASS SHOWS ITS CATCH. You swept a street for a
                    # class; the sweep hands you the pictures of what it
                    # found, in the pass's own (spatial) order, one badge
                    # per crop -- choosing among contenders becomes a
                    # reaction to what is already in front of you rather
                    # than a discipline to summon. This is the image
                    # budget spent on "the rows the coverage found, which
                    # is what vision is actually needed for" -- capped so
                    # the model's own later looks keep room.
                    _h = _flight_minted[0]
                    if _h:
                        _catch = [r for r in (carry_results.get(_h) or [])
                                  if isinstance(r, dict)
                                  and r.get("fid")][:_CATCH_CROPS]
                        if _catch:
                            _cfids = [
                                _crop_token_of(r, str(r["fid"]) in _ns_rows)
                                for r in _catch]
                            _cack = _crops_of(
                                _cfids, cid,
                                preamble=("The pass's catch, one crop per "
                                          "row it added, in pass order: "))
                            if not _cack.get("error"):
                                mv["catch"] = {
                                    k: _cack[k] for k in
                                    ("delivered", "not_drawn")
                                    if k in _cack}
                                if len(_catch) < len(
                                        carry_results.get(_h) or []):
                                    mv["catch"]["shown"] = (
                                        f"first {len(_catch)} of "
                                        f"{len(carry_results[_h])}" + (
                                            "" if withheld("look") else
                                            f" -- look({_h}[a:b]) pages "
                                            "the rest"))
                                if withheld("images"):
                                    mv["catch"] = {"note": _PICTURE_WITHHELD}
                    continue
                tok = _refval(step["to"])
                if isinstance(tok, str):
                    # a flown stop is a place: fly its recorded point
                    # (the xy→fid snap keeps a row's identity if one
                    # stands there)
                    tok, _stop_err = _flown_stop(tok)
                    if _stop_err:
                        return _plan_err(ack, _stop_err)
                _accepted, _item_errors = _mission_sanitize_moves(
                    [tok], staged_viewpoints)
                mv = _fly_moves(_accepted, _item_errors)
                ack["flown"] = mv
                _ok = [r for r in mv.get("flown") or []
                       if isinstance(r, dict) and "error" not in r]
                if not (_ok and _accepted):
                    # nothing flew: the failure is THE result, top-level
                    # (pilot-3: 'go to stop_003.' returned a
                    # flight-shaped envelope that never flew, invisible
                    # to the error count)
                    _msg = next(
                        (str(r.get("error"))
                         for r in mv.get("flown") or []
                         if isinstance(r, dict) and r.get("error")),
                        None) or next(
                        (str(e.get("error"))
                         for e in mv.get("rejected") or []
                         if isinstance(e, dict) and e.get("error")),
                        None) or str(mv.get("error") or "nothing flew")
                    return _plan_err(ack, _msg)
            elif do == "watch":
                ack["watching"] = _watch_apply(step)
                if ack["watching"].get("error"):
                    return _plan_err(ack, str(ack["watching"]["error"]))
            elif do == "boost":
                _where = [_refval(w) for w in step.get("where") or []]
                ack["boost"] = _boost_apply(step, _where)
                if ack["boost"].get("error"):
                    return _plan_err(ack, str(ack["boost"]["error"]))
            elif do == "crops":
                _cf = step.get("fid")
                _cf = ([_refval(f) for f in _cf] if isinstance(_cf, list)
                       else _refval(_cf)) if not step.get("of") else None
                # RESOLVE THE MARK SLOTS. The compiler writes each mark as
                # a grounding slot -- ('fid', 'g1') -- exactly as it writes
                # the crop's own subject, and _look_strs downstream keeps
                # only strings, so an unresolved slot vanished silently.
                _mk = [_refval(m) for m in (step.get("mark_fids") or [])]
                _mk = [m for m in _mk if isinstance(m, str) and m.strip()]
                ack["crop"] = (
                    _crops_set(str(step["of"]), cid, marks=_mk or None)
                    if step.get("of")
                    else _crops_of(_cf, cid, marks=_mk or None))
                if (ack["crop"].get("error")
                        and not ack["crop"].get("delivered")):
                    return _plan_err(ack, str(ack["crop"]["error"]))
            elif do == "map":
                # THE MAP BADGES THE SET IN SCOPE. Unmarked, every detection
                # paints as one white class (map_render's _DETECTION_RGB), so
                # the picture cannot answer "which of these rows is it" -- the
                # one question a narrowed set is asking. The marks path
                # already exists and resolves off the same rows crops use;
                # nothing passed any. Measured: `show me the map.` runs in
                # 94.5% of episodes at 1.5 per episode.
                _h = discourse.get("those")
                _mrows = [r for r in (_carry_read(_h) or [])
                          if isinstance(r, dict) and r.get("fid")] if _h else []
                _mfids = [str(r["fid"]) for r in _mrows[:_MAP_MAX_MARKS]]
                ack["map"] = _map_of(
                    None if step.get("around") is None
                    else _refval(step["around"]), cid, mark_fids=_mfids)
                if _mfids:
                    ack["map"]["badged_from"] = _h
                    if len(_mrows) > len(_mfids):
                        ack["map"].setdefault("ref_notes", []).append(
                            f"{len(_mrows)} rows in {_h}; this map badges the "
                            f"first {len(_mfids)}")
                if (ack["map"].get("error")
                        and not ack["map"].get("delivered")):
                    return _plan_err(ack, str(ack["map"]["error"]))
        # AN ORDERING THAT DID NOT ORDER ANYTHING. A big prior is a big
        # relation: Neuralia Road is 64 merged ways and 8,849 cells, so
        # every pool along four kilometres of it is 0 m away and
        # find("pool", near=<that road>) hands back a sort in which rank 0
        # means nothing. The model read rank 0 and committed 2.8 km from
        # the answer, and nothing in the result said the sort was empty.
        # It is a FACT, not advice: how many rows tie, and on what column.
        # Narrowing further -- another anchor, a containment, or looking at
        # the candidates -- is the model's to choose.
        _ok_key = (plan.get("_code_order") or [None, 1])[0]
        if _ok_key:
            _rows = [r for r in (final_rows or []) if isinstance(r, dict)]
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
        if ground_notes:
            ack["note"] = (f"{ack['note']}\n" if ack.get("note")
                           else "") + "; ".join(ground_notes)
        # a sentence that executed ends the consecutive-rejection streak
        # (mirror of the analyze path's reset; the next same-class error
        # serves its full text again)
        _err_streak["sig"] = None
        _err_streak["n"] = 0
        _update_discourse(plan, final_rows, discourse)
        if _flight_minted[0] is not None:
            # AFTER the plan's own discourse writes: a sentence that both
            # builds a set and flies ("go to the intersection of A and B.")
            # ends standing over what the flight revealed, so the flight's
            # set is the one in scope. The set the sentence built keeps its
            # own handle and stays reachable by name.
            discourse["those"] = _flight_minted[0]
            _flight_minted[0] = None
        return _echo(plan, ack)


    _mission_py: list = [None]

    def _dispatch_run(args, cid: str) -> dict:
        """One code block: real Python, executed against the shared plan
        machinery.

        The block runs in ONE persistent namespace -- a variable an
        earlier turn bound is still bound -- and every mission call
        builds the same parsed step the retired parser produced, so what
        executes underneath (grounding, programs, flights, serving) is
        the measured path. An error costs the line it is on: everything
        the block ran before it stands, receipts included.
        """
        code = (args or {}).get("code")
        if not isinstance(code, str) or not code.strip():
            return {"error": "run takes one block of code in code"}
        if _mission_py[0] is None:
            _mission_py[0] = _py_surface.MissionPython(_py_surface.MissionEnv(
                run_plan=_run_plan,
                # THE CODE ARM READS THE FLOWN WORLD TOO. Python
                # evaluates go_to(red_cars[0])'s subscript BEFORE go_to
                # dispatches, so without this the commit target comes out
                # of the pre-flight world -- and len(), print() and
                # iteration read it the same way.
                # compute_namespace is updated IN PLACE after every flight
                # (_surface_from_ctx), so it is always the flown world.
                rows_of_ref=lambda ref: (
                    _ensure_fresh([ref]),
                    [r for r in (carry_results.get(ref) or [])
                     if isinstance(r, dict)])[1],
                tied_of_ref=lambda ref: bool(
                    getattr(_carry_read(ref), "name_set", False)),
                plan_rn=plan_rn,
                discourse=discourse,
                minted=_code_minted,
                say_err=_plan_err,
                row_of_fid=_ns_rows.get))
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
        # Terminal carry, no dispatch: re-scanning a known-infeasible thread
        # costs a full-transcript tokenize per stop and re-raises every time
        # (measured: 17 s x every remaining tour stop). The tour keeps
        # flying; the commit binds the candidates accumulated so far.
        return _mission_end("compaction_infeasible")

    for turn_idx in itertools.count():
        # Failure-mode diagnostics: stamp turns-used each iteration so it survives
        # whichever return point fires (terminal / carry / cap / transient).
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
        # The bench passes nothing. ``retask is None`` appends not one
        # byte, which is what keeps "the operator instruction is the single
        # user message" literally true on the measured path.
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
            with net_meter_stage("mission"):
                r = reasoner.step_tool_call(stop_tools)
        except (CompactionInfeasible, CompactionInvariantError) as exc:
            # CompactionInvariantError shares this handler rather than
            # escaping uncaught. It is a sibling of Infeasible, not a
            # subclass, so it had none -- a compacted list that broke tool
            # closure killed the PROCESS where an uncompactable one ends
            # the episode. Both mean the same thing to a caller: this
            # conversation cannot be served, so end it the same way.
            # Terminal for the EPISODE, not just this round: nothing removes
            # the residue the compactor cannot touch, so flag the thread
            # (later rounds carry instantly, above) and drop this round's
            # packet — dead weight no dispatch will ever read.
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
            # NEVER carry on it.
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
                # ending carries text (520/520 across two fleets), and
                # 28/28 no-commit episodes were THIS shape -- the serve's
                # tool-grammar patch dropping a header/payload name
                # mismatch, the think span ending mid-plan ("let's fly to
                # prior_293") with nothing on either channel. Repair once
                # (measured 84% next-turn recovery where the round path's
                # nudge fires); a second consecutive empty still ends the
                # mission -- the ending stays model-owned.
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
                # dropped message, not an ending (51% of a fleet's
                # episodes ended this way on the say surface,
                # mid-search, most of them on a `go to`; a block
                # written here ends the mission the same silent way,
                # short, with the flights it was about to make never
                # made). Repair once; the streak is shared with the
                # empty turn, so a repeated not-a-reply still ends and
                # the ending stays model-owned.
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
            # A REPLY BEFORE ANY FLIGHT answers with the start pose, so
            # say so ONCE and hand the turn back (NO_FLIGHT_REPLY_NUDGE).
            # Its own streak, because this is not a channel miss: the
            # reply arrived on the right channel and is a legal ending,
            # it just has nothing behind it. A second one ends the
            # mission unchanged, which is what keeps the ending
            # model-owned rather than harness-owned.
            #
            # ONLY A CLEAN PROSE REPLY. The empty turn and the
            # code-shaped reply fall through to this same ending after
            # their own repair, and neither is the model saying "this is
            # my answer" -- one is a dropped message and the other is a
            # block on the wrong channel. Nudging those about not having
            # flown answers a question they did not ask, and it costs the
            # turn their own repair already spent.
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
            # PROSE reply. (Envelope residue and a reply that IS
            # a reply that IS a block of the surface are channel
            # misses, repaired above -- a dropped message is not an
            # ending.)
            log.info("%s mission ended by reply at turn %d (%d flights)",
                     rid_tag, turn_idx, mission_flights)
            return _mission_end("reply", reply_text=content,
                                turns=turn_idx + 1)
        no_call_streak = 0
        # THE MODEL DID SOMETHING, so the abstain streak restarts. A find
        # that turns up nothing is still work; only replies with nothing
        # between them accumulate toward the abstain.
        no_flight_streak = 0

        # Tool-call turn: dispatch each call IN ORDER, answering EVERY call
        # id (the compactor's id-closure sweep requires a result per id).
        for tc in r["tool_calls"]:
            name = (tc.get("function") or {}).get("name") or ""
            cid = tc.get("id") or ""
            args = _tool_call_args(tc)
            # analyze is a ROUND-ERA verb: the mission surface declares
            # [say] only, so a mission-era analyze call is an undeclared
            # name and falls through to the id-closure answer below --
            # never into _dispatch_analyze_scene, whose errors and hints
            # render round-era op-JSON. Only an unmasked serve can emit
            # one (the tool tag pins declared names), but that route was
            # the one way round-era text could land in a bench transcript.
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
                        # the code it decided on -- round-tripped on `r`
                        # from step_tool_call, already computed for the
                        # envelope-residue check above; a viewer reads
                        # "why" a turn's code looks the way it does.
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
                # (id-closure) with an actionable error; never executed.
                #
                # IT NAMES THE TOOL THAT IS SERVED. The one message sent
                # when the model has already lost the tool name must not
                # hand it one the surface refuses: a surface that refuses a form must
                # not still teach it.
                reasoner.append_tool_result(cid, name or "unknown", {
                    "error": f"unknown tool {name!r}; available: run"})
        # Two-message look delivery: append the crop images as a USER turn AFTER
        # the turn's tool results (keeps tool-result id-closure contiguous). Look
        # is non-terminal, so the crops sit in context for the next turn / stop.
        for _vtext, _vjpegs in pending_vision:
            reasoner.append_vision_user(_vtext, _vjpegs)
        pending_vision.clear()
