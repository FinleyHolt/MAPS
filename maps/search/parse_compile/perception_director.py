"""Perception direction state for the three-verb tool surface (C4).

One :class:`PerceptionDirector` per episode (owned by ``search_loop._State``),
holding the agent-owned perception-direction channels the ``view`` tool
mutates and the ``act`` tool coordinates with:

  * the WATCHLIST — the persistent passive SAM class set segmented at every
    subsequently-flown stop. Initial value = the committed
    ``sam_concepts`` union; ``view{watchlist:{add,remove}}`` mutates it
    mid-flight and the mutation is pushed to the live concept provider via
    ``push`` (``ReasonerUnionConceptProvider.apply_watchlist`` — the
    freeze-exempt agent-owned channel), so the per-stop ``/segment`` calls
    segment the LIVE watchlist.
  * BOOSTS — targeted vocabulary from ``view{boost:{concepts,where}}``.
    RETROSPECTIVE PERCEPTION: the ortho render is deterministic
    over the static raster, so a boost whose ``where`` targets an
    ALREADY-FLOWN stop/pose executes IMMEDIATELY through
    ``retro_perceiver`` (re-render at the recorded pose → SAM with the
    boost vocabulary → normal ingest) and the DELTA returns synchronously
    in the same view result — zero flight; vocabulary alone NEVER re-flies.
    A boost on a PENDING (unflown) stop / future area queues: the runtime
    ACTIVATES pending boosts for the next perception round
    (``activate_pending`` pushes ``watchlist + boost concepts``) and
    RETIRES them after it (``retire_after_round``); an unmatched targeted
    boost returns to pending so it stays attached to its queued stop until
    that stop flies. ``where`` entries address a pending stop by its
    ``qstop_N``/``stop_NNN`` id, a fid, or an ``[x, y]`` literal.
  * the PENDING-STOP registry — every ``act{viewpoints}`` item is registered
    at accept time with a minted ``qstop_N`` id (full intent tuple kept),
    making the queue ADDRESSABLE: the act result / stop packet
    name the pending stops, later ``view{boost}`` calls target them before
    they fly, and ``act{cancel}`` removes them. (Registering a BAKED tour's
    own stops was a round-era runtime hook and retired with it: the mission
    flies each viewpoint the moment it lands, so nothing is ever pending.)
  * CANCELLATION — ``cancel_stops`` removes
    PENDING (unflown) stops: agent-queued qstops AND unflown baked tour
    stops. Only pending stops are cancellable; a flown/unknown id is
    rejected with a factual error while the valid ids in the same call
    still apply. Boosts attached ONLY to a canceled stop retire with it
    (stated in the result); the runtime filters canceled intents out of
    the tour extension and skips canceled baked stops.

ROUND-ERA ARM, left in place: ``register_tour_stops`` / ``is_stop_canceled`` /
``canceled_intent_filter`` / ``mark_intents_absorbed`` / ``dedup_exempt`` were
consulted only by the retired Phase A-D executor and now have no caller. They
are NOT deleted because ``cancel_stops`` is still reached from the round path
and its result text is served to the model verbatim; re-shaping its branches to
drop the baked-tour arm would edit bytes the quarantine freezes. Agent-queued
qstop cancellation is live; the baked-tour half of it never fires.

The dedup-bypass contract (C4, NARROWED by retrospective perception): a
boost never re-flies anything on its own; an EXPLICIT ``act{viewpoints}``
re-request of an already-flown pose bypasses the dedup only while a
genuinely-pending boost exists for it (``dedup_exempt`` is the predicate
``tour_synth.synthesize_extension`` consults).

Pure state + arithmetic (no LLM, no perception imports; the retro executor
is an injected callable) so the whole surface is unit-testable without the
engine.
"""

from __future__ import annotations

import logging

from maps.util.xy_token import parse_xy_token
import math
from dataclasses import dataclass, field
from typing import Callable

#: The two roles a watched class can carry. `target` is what the cue asks
#: for; `anchor` is a landmark that helps find it. Both are segmented the
#: same way -- the role orders nothing and gates nothing, it only says why
#: the class was asked for.
WATCH_ROLES = ("target", "anchor")

log = logging.getLogger(__name__)

# World-meter radius within which an [x, y] (or fid-resolved) boost ``where``
# matches a flown pose / a re-requested viewpoint's pose. Sized to the stop
# view square (fov ~100 m): a boost aimed at an area matches the stop that
# actually images that area, without leaking to neighboring stops.
BOOST_MATCH_RADIUS_M: float = 50.0


def _canon(s: str) -> str:
    return " ".join(str(s or "").split()).strip().lower()


def _dedup_concepts(seq) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq or []:
        raw = str(s or "").strip()
        key = _canon(raw)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out


def _xy_of(v) -> tuple[float, float] | None:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            return (float(v[0]), float(v[1]))
        except (TypeError, ValueError):
            return None
    return None


def _intent_key(it: dict) -> tuple:
    return (
        str(it.get("target_fid") or "").strip() or None,
        it.get("view_intent"),
        it.get("view_face"),
        it.get("view_from_fid"),
    )


@dataclass
class _Boost:
    concepts: list[str]
    # Resolved where entries: {"kind": "stop"|"fid"|"xy", "stop": stop id?,
    # "fid": str?, "xy": (x, y)?}. Empty = where-less (next flown stops).
    where: list[dict] = field(default_factory=list)
    active: bool = False

    def label(self) -> str:
        return "/".join(self.concepts)


@dataclass
class PerceptionDirector:
    """Agent-owned perception direction (watchlist / boosts / stop queue)."""

    # The live channel to the SAM concept provider (freeze-exempt). None keeps
    # every mutation state-only (offline tests / non-SAM adapters); the delta
    # text stays honest either way.
    push: Callable[[list[str]], None] | None = None
    # RETROSPECTIVE executor: injected by the runtime.
    # Called with (flown_entry, concepts) where flown_entry is
    # {"pose_xy": (x, y), "alt_m": float|None, "stop_id": str|None,
    #  "target_fids": [...]}; returns {"n_dets": int, "new_fids": [...]} or
    # {"error": str}. None -> a flown-targeted boost returns a LOUD factual
    # unavailability error (never a silent degrade to attach-only).
    retro_perceiver: Callable[[dict, list], dict] | None = None

    watchlist: list[str] = field(default_factory=list)
    boosts: list[_Boost] = field(default_factory=list)
    # stop id -> {"target_fid": str, "xy": (x, y) | None, "flown": bool,
    #             "canceled": bool, "intent": tuple | None,
    #             "source": "agent" | "tour", "note": str}
    # Agent-queued stops mint ``qstop_N`` ids; baked/synthesized tour stops
    # register under their own ``stop_NNN`` ids (the whole
    # remaining tour is addressable + cancellable while pending).
    pending_stops: dict[str, dict] = field(default_factory=dict)
    # Canceled intent tuples — the runtime filters the round's aggregated
    # viewpoint requests through these before synthesis.
    canceled_intents: set = field(default_factory=set)
    # Canceled baked/synthesized tour stop ids — the runtime skips them.
    canceled_tour_stops: set = field(default_factory=set)
    # Flown-observation log (appended by retire_after_round): each entry
    # {"pose_xy": (x, y), "alt_m": float|None, "stop_id": str|None,
    #  "target_fids": [str, ...]} — the RETROSPECTIVE address space (an
    # already-visited pose is permanently re-analyzable).
    flown_log: list[dict] = field(default_factory=list)
    _qstop_seq: int = 0
    # Fired-subset telemetry (folded into react_stats by the caller).
    counters: dict = field(default_factory=lambda: {
        "watchlist_mutations": 0,
        "boost_queued": 0,
        "boost_retro_fired": 0,
        "boost_activated_rounds": 0,
        "boost_dedup_bypass": 0,
        "canceled_stops": 0,
    })

    # ------------------------------------------------------------------
    # Watchlist
    # ------------------------------------------------------------------

    def init_watchlist(self, concepts) -> None:
        """Set the initial watchlist (the iteration-0 sam_concepts union) and
        push it live. Does NOT count as an agent mutation."""
        self.watchlist = _dedup_concepts(concepts)
        self._push_live()

    def watchlist_apply(self, add, remove) -> dict:
        """Apply one ``view{watchlist:{add,remove}}`` mutation; returns the
        delta facts the tool result states — including whether the mutation
        actually reached the LIVE segment channel (``live``), so the ack can
        never claim a live effect the push did not deliver."""
        add = _dedup_concepts(add)
        remove_keys = {_canon(s) for s in (remove or []) if _canon(s)}
        removed = [w for w in self.watchlist if _canon(w) in remove_keys]
        unknown_removes = sorted(
            remove_keys - {_canon(w) for w in self.watchlist})
        kept = [w for w in self.watchlist if _canon(w) not in remove_keys]
        existing = {_canon(w) for w in kept}
        added = [a for a in add if _canon(a) not in existing]
        self.watchlist = kept + added
        self.counters["watchlist_mutations"] += 1
        push_status = self._push_live()
        out = {
            "watchlist": list(self.watchlist),
            "added": added,
            "removed": removed,
            "unknown_removes": unknown_removes,
            "pending_stops": self.pending_unflown_count(),
            "live": push_status == "ok",
        }
        if push_status == "failed":
            out["live_push_failed"] = True
        return out

    def _push_live(self) -> str:
        """Push watchlist + active-boost concepts to the live provider.
        Returns ``"ok"`` | ``"no_channel"`` | ``"failed"`` so callers can be
        HONEST about whether the live segment set actually changed (a
        swallowed push failure must never produce a success-shaped ack)."""
        if self.push is None:
            return "no_channel"
        try:
            active = [c for b in self.boosts if b.active for c in b.concepts]
            self.push(_dedup_concepts(list(self.watchlist) + active))
            return "ok"
        except Exception:  # noqa: BLE001 — a push failure must not kill the round
            log.exception("PerceptionDirector: live concept push failed")
            return "failed"

    # ------------------------------------------------------------------
    # Pending-stop registry (the addressable queue)
    # ------------------------------------------------------------------

    def register_viewpoints(self, items, rows_by_fid=None) -> list[dict]:
        """Register accepted ``act{viewpoints}`` items; mint ``qstop_N`` ids.

        Returns one ``{"stop", "target_fid", "xy"}`` record per item, in
        order — the names the act result states so later view/cancel calls
        can target the queue before it flies."""
        out: list[dict] = []
        rows_by_fid = rows_by_fid or {}
        for it in items or []:
            if not isinstance(it, dict):
                continue
            fid = str(it.get("target_fid") or "").strip()
            if not fid:
                continue
            self._qstop_seq += 1
            qid = f"qstop_{self._qstop_seq}"
            row = rows_by_fid.get(fid)
            xy = _xy_of(row.get("xy")) if isinstance(row, dict) else None
            if xy is None:
                # an "[x, y]"-literal target: the coordinate IS the stop's
                # position — register it so boost/cancel addressing and the
                # queued ack read the real point (a None here made a point
                # target read "queued" with no location).
                xy = parse_xy_token(fid)
            self.pending_stops[qid] = {
                "target_fid": fid, "xy": xy, "flown": False,
                "canceled": False, "intent": _intent_key(it),
                "source": "agent", "note": "",
            }
            rec: dict = {"stop": qid, "target_fid": fid}
            if xy is not None:
                rec["xy"] = [round(xy[0], 1), round(xy[1], 1)]
            out.append(rec)
        return out

    def register_tour_stops(self, stops) -> None:
        """Register the baked/synthesized tour's own stops under their
        ``stop_NNN`` ids (the remaining tour is addressable
        + cancellable while pending). ``stops``: iterable of
        ``{"stop_id", "target_fid"?, "xy"?}``; re-registration of a known id
        is a no-op (keeps flown/canceled state)."""
        for st in stops or []:
            if not isinstance(st, dict):
                continue
            sid = str(st.get("stop_id") or "").strip()
            if not sid or sid in self.pending_stops:
                continue
            self.pending_stops[sid] = {
                "target_fid": str(st.get("target_fid") or "").strip(),
                "xy": _xy_of(st.get("xy")),
                "flown": False, "canceled": False, "intent": None,
                "source": "tour", "note": "",
            }

    def pending_unflown_count(self) -> int:
        return sum(1 for p in self.pending_stops.values()
                   if not p["flown"] and not p["canceled"])

    def pending_stop_ids(self) -> list[str]:
        return [sid for sid, p in self.pending_stops.items()
                if not p["flown"] and not p["canceled"]]

    def remaining_tour_line(self) -> str:
        """One fact line for the stop decision packet: the remaining
        (pending, uncanceled) stops with their targets + attached boosts —
        the addressability surface. '' when nothing pends."""
        parts: list[str] = []
        for sid, p in self.pending_stops.items():
            if p["flown"] or p["canceled"]:
                continue
            tag = p["target_fid"] or (
                f"({p['xy'][0]:.0f}, {p['xy'][1]:.0f})" if p["xy"] else "?")
            battached = [b.label() for b in self.boosts
                         if any(w.get("stop") == sid for w in b.where)]
            if battached:
                tag += f", {'+'.join(battached)} boost"
            parts.append(f"{sid} ({tag})")
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel_stops(self, ids) -> dict:
        """``act{cancel}``: cancel PENDING stops (agent qstops + unflown
        baked tour stops). Valid ids apply even when others are rejected
        (never partial-silent); each rejected id gets its factual reason;
        boosts attached ONLY to canceled stops retire with them."""
        canceled: list[str] = []
        canceled_intents: list[tuple] = []
        rejected: dict[str, str] = {}
        for raw in ids or []:
            sid = str(raw or "").strip()
            if not sid:
                continue
            p = self.pending_stops.get(sid)
            if p is None:
                rejected[sid] = "unknown stop id"
                continue
            if p["canceled"]:
                rejected[sid] = "already canceled"
                continue
            if p["flown"]:
                rejected[sid] = "already flown (only pending stops cancel)"
                continue
            p["canceled"] = True
            canceled.append(sid)
            if p["intent"] is not None:
                self.canceled_intents.add(p["intent"])
                canceled_intents.append(p["intent"])
            if p["source"] == "tour":
                self.canceled_tour_stops.add(sid)
        # Retire boosts attached to the canceled stops: drop the canceled
        # where entries; a boost whose wheres ALL canceled retires outright
        # (it must NOT degrade to a where-less next-flown boost).
        retired_boosts: list[str] = []
        if canceled:
            survivors: list[_Boost] = []
            gone = set(canceled)
            for b in self.boosts:
                if not b.where:
                    survivors.append(b)
                    continue
                keep_where = [w for w in b.where if w.get("stop") not in gone]
                if keep_where:
                    b.where = keep_where
                    survivors.append(b)
                else:
                    retired_boosts.append(b.label())
            self.boosts = survivors
            self.counters["canceled_stops"] += len(canceled)
        return {
            "canceled": canceled,
            "retired_boosts": retired_boosts,
            "rejected": rejected,
            "pending": self.pending_stop_ids(),
            # For the round's LOCAL requested-viewpoints filter (a remove of
            # a stop queued earlier in the same round must also strip it
            # from the reply the runtime consumes). Tuples, not model text.
            "canceled_intents": canceled_intents,
        }

    def mark_intents_absorbed(self, vps) -> int:
        """Reconcile the registry after synthesis: a queued agent stop whose request was ABSORBED (deduped
        against an already-flown pose with no boost pending, or its pose
        never resolved) will never fly — mark it so it leaves the pending
        counts / remaining-tour line instead of stranding forever.
        Deliberately does NOT touch ``canceled_intents``: absorption is a
        runtime fact, not an agent cancellation — an identical future
        request must still be schedulable. Returns the number marked."""
        keys = {_intent_key(vp) for vp in vps or [] if isinstance(vp, dict)}
        n = 0
        for p in self.pending_stops.values():
            if (p["source"] == "agent" and not p["flown"]
                    and not p["canceled"] and p["intent"] in keys):
                p["canceled"] = True
                p["note"] = "absorbed at synthesis (deduped/unresolvable)"
                n += 1
        if n:
            log.info("pending-stop registry: %d absorbed request(s) retired",
                     n)
        return n

    def canceled_intent_filter(self, vps) -> list[dict]:
        """Filter a viewpoint-request list down to the non-canceled ones
        (the runtime applies this to the round's aggregated requests before
        ``synthesize_extension``)."""
        return [
            vp for vp in (vps or [])
            if isinstance(vp, dict)
            and _intent_key(vp) not in self.canceled_intents
        ]

    def is_stop_canceled(self, stop_id: str) -> bool:
        return str(stop_id or "").strip() in self.canceled_tour_stops

    # ------------------------------------------------------------------
    # Boosts (queue + RETROSPECTIVE execution)
    # ------------------------------------------------------------------

    def _find_flown(self, w: dict) -> dict | None:
        """The flown observation a resolved where entry addresses, or None.

        Matches (in order): a registered stop id marked flown — resolved to
        its RECORDED flown-log observation by stop id, target fid, or
        xy-radius (amendment verifier A2: the log's pose+altitude is the
        view the drone actually captured); only when no log entry matches
        does the target-centered synthetic fallback fire (a nadir ortho
        render centered on the target — deterministic and legitimate, but
        not the recorded frame; alt None -> the executor's default). Then:
        a fid any flown stop targeted; an [x, y] within radius of a flown
        pose."""
        sid = w.get("stop")
        if sid:
            p = self.pending_stops.get(sid)
            if p is not None and p["flown"]:
                for f in self.flown_log:
                    if sid == f.get("stop_id") or (
                        p["target_fid"]
                        and p["target_fid"] in (f.get("target_fids") or [])
                    ):
                        return f
                pxy = _xy_of(p.get("xy"))
                if pxy is not None:
                    for f in self.flown_log:
                        fxy = _xy_of(f.get("pose_xy"))
                        if fxy is not None and math.hypot(
                                pxy[0] - fxy[0], pxy[1] - fxy[1]
                        ) <= BOOST_MATCH_RADIUS_M:
                            return f
                return {"pose_xy": p["xy"], "alt_m": None,
                        "stop_id": sid, "target_fids": [p["target_fid"]]}
            # An unregistered stop token may still name a flown tour stop
            # present only in the flown log.
            for f in self.flown_log:
                if sid == f.get("stop_id"):
                    return f
            return None
        fid = w.get("fid")
        if fid:
            for f in self.flown_log:
                if fid in (f.get("target_fids") or []):
                    return f
            return None
        wxy = w.get("xy")
        if wxy is not None:
            for f in self.flown_log:
                fxy = _xy_of(f.get("pose_xy"))
                if fxy is not None and math.hypot(
                        wxy[0] - fxy[0], wxy[1] - fxy[1]) <= BOOST_MATCH_RADIUS_M:
                    return f
        return None

    def boost_queue(self, concepts, where) -> dict:
        """One ``view{boost}``: RETROSPECTIVE targets (already-flown
        stop/pose/fid) execute IMMEDIATELY through ``retro_perceiver`` with
        the delta returned in this ack; pending/future targets (and a
        where-less boost) QUEUE for fly-time attachment."""
        concepts = _dedup_concepts(concepts)
        if not concepts:
            return {"error": "boost needs a non-empty 'concepts' list"}
        resolved: list[dict] = []
        unresolved: list = []
        for w in where or []:
            xy = _xy_of(w)
            if xy is not None:
                resolved.append({"kind": "xy", "xy": xy})
                continue
            token = str(w or "").strip()
            if not token:
                continue
            if token in self.pending_stops:
                p = self.pending_stops[token]
                if p["canceled"]:
                    unresolved.append(token)
                    continue
                resolved.append({
                    "kind": "stop", "stop": token,
                    "fid": p["target_fid"], "xy": p["xy"],
                })
            elif token.startswith(("qstop_", "stop_")):
                # A stop token: maybe a flown tour stop known only to the
                # flown log; else unknown.
                resolved.append({"kind": "stop", "stop": token,
                                 "fid": "", "xy": None})
            else:
                resolved.append({"kind": "fid", "fid": token})
        # Split: retro (already flown — execute now) vs queue (pending).
        retro_results: list[dict] = []
        queue_where: list[dict] = []
        for w in resolved:
            flown = self._find_flown(w)
            token = (w.get("stop") or w.get("fid")
                     or [round(w["xy"][0], 1), round(w["xy"][1], 1)])
            if flown is None:
                if w["kind"] == "stop" and w.get("stop") not in self.pending_stops:
                    unresolved.append(w["stop"])
                    continue
                queue_where.append(w)
                continue
            # RETROSPECTIVE PERCEPTION: zero flight — re-analysis of the
            # legitimately captured observation at the recorded pose.
            if self.retro_perceiver is None:
                retro_results.append({
                    "where": token,
                    "error": ("retrospective re-segmentation is not wired "
                              "on this run; the flown observation cannot "
                              "be re-analyzed"),
                })
                continue
            try:
                res = self.retro_perceiver(dict(flown), list(concepts))
            except Exception as exc:  # noqa: BLE001 — surfaced, never a round crash
                log.exception("retro perceiver raised")
                res = {"error": f"retrospective re-segmentation failed: "
                                f"{type(exc).__name__}: {exc}"}
            rec = {"where": token}
            if isinstance(res, dict):
                rec.update(res)
            else:
                rec["error"] = "retrospective re-segmentation returned nothing"
            if "error" not in rec:
                self.counters["boost_retro_fired"] += 1
            retro_results.append(rec)
        ack: dict = {"queued": [], "where": []}
        if queue_where or not (where or []):
            # Queue when targets remain pending OR the boost is where-less
            # (next flown stops). A purely-retro boost queues NOTHING.
            self.boosts.append(_Boost(concepts=concepts, where=queue_where))
            self.counters["boost_queued"] += 1
            ack["queued"] = concepts
            ack["where"] = [
                (w["stop"] if w["kind"] == "stop"
                 else w["fid"] if w["kind"] == "fid"
                 else [round(w["xy"][0], 1), round(w["xy"][1], 1)])
                for w in queue_where
            ]
        if retro_results:
            ack["retro"] = retro_results
        if unresolved:
            ack["unknown_where"] = unresolved
        if not ack["queued"]:
            del ack["queued"], ack["where"]
        if "queued" not in ack and "retro" not in ack:
            # A TARGETED boost none of whose where entries resolved must NOT
            # silently degrade to a where-less next-round boost: say so,
            # and name the addressable set.
            return {
                "error": ("no boost target resolved; nothing was queued or "
                          "re-segmented"),
                "unknown_where": unresolved,
                "pending": self.pending_stop_ids(),
            }
        return ack

    def activate_pending(self) -> list[str]:
        """Activate every pending boost for the NEXT perception round;
        push ``watchlist + boost concepts`` live. Returns the activated
        concept union ([] when nothing was pending). A FAILED live push
        REVERTS the activation (the boosts stay pending and retry next
        round) — otherwise retire would consume boosts that never actually
        rode a segment call."""
        pending = [b for b in self.boosts if not b.active]
        if not pending:
            return []
        for b in pending:
            b.active = True
        if self._push_live() == "failed":
            for b in pending:
                b.active = False
            log.warning(
                "boost activation reverted: live concept push failed; "
                "%d boost(s) stay pending for the next round", len(pending),
            )
            return []
        self.counters["boost_activated_rounds"] += 1
        return _dedup_concepts([c for b in pending for c in b.concepts])

    def _where_matches_flown(self, w: dict, flown: list[dict]) -> bool:
        for f in flown:
            fids = {str(x) for x in (f.get("target_fids") or []) if x}
            fxy = _xy_of(f.get("pose_xy"))
            sid = f.get("stop_id")
            if w.get("stop") and sid and w["stop"] == sid:
                return True
            if w["kind"] in ("fid", "stop") and w.get("fid") and w["fid"] in fids:
                return True
            wxy = w.get("xy")
            if wxy is not None and fxy is not None and (
                math.hypot(wxy[0] - fxy[0], wxy[1] - fxy[1])
                <= BOOST_MATCH_RADIUS_M
            ):
                return True
        return False

    def retire_after_round(self, flown: list[dict]) -> None:
        """Consume boosts + mark pending stops after one perception round.

        ``flown`` entries: ``{"pose_xy": (x, y), "alt_m"?: float,
        "stop_id"?: str, "target_fids": [str, ...]}`` — one per stop the
        round actually perceived; each is APPENDED to the flown log (the
        retrospective address space). A where-less ACTIVE boost is
        consumed; a where-targeted active boost is consumed iff any of its
        where entries matched a flown stop, else it returns to PENDING (it
        stays attached to its queued stop until that stop flies). Pending
        stops whose id / target fid / xy matched are marked flown. Pushes
        the post-retire concept set live."""
        flown = [f for f in (flown or []) if isinstance(f, dict)]
        for f in flown:
            self.flown_log.append({
                "pose_xy": _xy_of(f.get("pose_xy")),
                "alt_m": f.get("alt_m"),
                "stop_id": (str(f.get("stop_id")).strip()
                            if f.get("stop_id") else None),
                "target_fids": [str(x) for x in (f.get("target_fids") or [])
                                if x],
            })
        flown_fids = {
            str(x) for f in flown for x in (f.get("target_fids") or []) if x}
        flown_sids = {str(f.get("stop_id")).strip() for f in flown
                      if f.get("stop_id")}
        flown_xys = [xy for xy in (_xy_of(f.get("pose_xy")) for f in flown)
                     if xy is not None]
        for qid, p in self.pending_stops.items():
            if p["flown"] or p["canceled"]:
                continue
            if qid in flown_sids or (
                    p["target_fid"] and p["target_fid"] in flown_fids):
                p["flown"] = True
                continue
            # xy-radius fallback ONLY for AGENT-queued stops (their eventual
            # synthesized pose may differ from the request's target xy). A
            # registered TOUR stop's own stop_id IS its identity — a radius
            # fallback would blanket-mark neighboring tour stops flown.
            if p["source"] != "agent":
                continue
            pxy = p.get("xy")
            if pxy is not None and any(
                math.hypot(pxy[0] - fx, pxy[1] - fy) <= BOOST_MATCH_RADIUS_M
                for fx, fy in flown_xys
            ):
                p["flown"] = True
        survivors: list[_Boost] = []
        for b in self.boosts:
            if not b.active:
                survivors.append(b)
                continue
            if not b.where:
                continue  # where-less: consumed by the round it rode
            if any(self._where_matches_flown(w, flown) for w in b.where):
                continue  # its target flew: consumed
            b.active = False  # target did not fly: back to pending
            survivors.append(b)
        self.boosts = survivors
        self._push_live()

    # ------------------------------------------------------------------
    # Dedup bypass (C4, narrowed: explicit re-request + pending boost only)
    # ------------------------------------------------------------------

    def dedup_exempt(self, target_fid: str, pose_xy=None) -> bool:
        """True when an EXPLICIT ``act{viewpoints}`` re-request over
        ``target_fid`` (resolved pose ``pose_xy`` when known) has a
        genuinely-PENDING boost for that stop/area — the re-flight is a new
        perception act and bypasses the already-flown dedup. Vocabulary
        alone never re-flies (flown-targeted boosts execute retrospectively
        and never linger here); this predicate only widens what an explicit
        re-request may do."""
        fid = str(target_fid or "").strip()
        pxy = _xy_of(pose_xy)
        for b in self.boosts:
            if not b.where:
                # A where-less outstanding boost applies to every
                # subsequently-flown stop: an explicit re-flight perceives
                # with new vocabulary.
                self.counters["boost_dedup_bypass"] += 1
                return True
            for w in b.where:
                if fid and w.get("fid") == fid:
                    self.counters["boost_dedup_bypass"] += 1
                    return True
                wxy = w.get("xy")
                if wxy is not None and pxy is not None and (
                    math.hypot(wxy[0] - pxy[0], wxy[1] - pxy[1])
                    <= BOOST_MATCH_RADIUS_M
                ):
                    self.counters["boost_dedup_bypass"] += 1
                    return True
        return False
