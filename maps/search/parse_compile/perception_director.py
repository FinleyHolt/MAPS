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
    boost returns to pending so it stays attached to its target until that
    target flies. ``where`` entries address a flown stop by its
    ``qstop_N``/``stop_NNN`` id, a fid, or an ``[x, y]`` literal. The
    mission flies each viewpoint the moment it lands, so no stop is ever
    queued unflown: the watchlist and boost results report
    ``pending_stops: 0`` and ``pending: []``.

Pure state + arithmetic (no LLM, no perception imports; the retro executor
is an injected callable) so the whole surface is unit-testable without the
engine.
"""

from __future__ import annotations

import logging
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


def canon_concept(s) -> str:
    """A concept phrase's one canonical form: case-folded, whitespace
    collapsed; empty input returns ``""``."""
    return " ".join(str(s or "").lower().split())


def _dedup_concepts(seq) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq or []:
        raw = str(s or "").strip()
        key = canon_concept(raw)
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


@dataclass
class _Boost:
    concepts: list[str]
    # Resolved where entries: {"kind": "stop"|"fid"|"xy", "stop": stop id?,
    # "fid": str?, "xy": (x, y)?}. Empty = where-less (next flown stops).
    where: list[dict] = field(default_factory=list)
    active: bool = False


@dataclass
class PerceptionDirector:
    """Agent-owned perception direction (watchlist / boosts)."""

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
    # Flown-observation log (appended by retire_after_round): each entry
    # {"pose_xy": (x, y), "alt_m": float|None, "stop_id": str|None,
    #  "target_fids": [str, ...]} — the RETROSPECTIVE address space (an
    # already-visited pose is permanently re-analyzable).
    flown_log: list[dict] = field(default_factory=list)

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
        remove_keys = {canon_concept(s) for s in (remove or []) if canon_concept(s)}
        removed = [w for w in self.watchlist if canon_concept(w) in remove_keys]
        unknown_removes = sorted(
            remove_keys - {canon_concept(w) for w in self.watchlist})
        kept = [w for w in self.watchlist if canon_concept(w) not in remove_keys]
        existing = {canon_concept(w) for w in kept}
        added = [a for a in add if canon_concept(a) not in existing]
        self.watchlist = kept + added
        push_status = self._push_live()
        out = {
            "watchlist": list(self.watchlist),
            "added": added,
            "removed": removed,
            "unknown_removes": unknown_removes,
            "pending_stops": 0,
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
    # Boosts (queue + RETROSPECTIVE execution)
    # ------------------------------------------------------------------

    def _find_flown(self, w: dict) -> dict | None:
        """The flown observation a resolved where entry addresses, or None:
        the flown-log entry with that stop id; a fid any flown stop
        targeted; an [x, y] within radius of a flown pose."""
        sid = w.get("stop")
        if sid:
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
            if token.startswith(("qstop_", "stop_")):
                # A stop token: a flown stop in the flown log, else unknown.
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
                if w["kind"] == "stop":
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
            retro_results.append(rec)
        ack: dict = {"queued": [], "where": []}
        if queue_where or not (where or []):
            # Queue when targets remain pending OR the boost is where-less
            # (next flown stops). A purely-retro boost queues NOTHING.
            self.boosts.append(_Boost(concepts=concepts, where=queue_where))
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
                "pending": [],
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
        """Consume boosts after one perception round.

        ``flown`` entries: ``{"pose_xy": (x, y), "alt_m"?: float,
        "stop_id"?: str, "target_fids": [str, ...]}`` — one per stop the
        round actually perceived; each is APPENDED to the flown log (the
        retrospective address space). A where-less ACTIVE boost is
        consumed; a where-targeted active boost is consumed iff any of its
        where entries matched a flown stop, else it returns to PENDING (it
        stays attached to its target until that target flies). Pushes the
        post-retire concept set live."""
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
