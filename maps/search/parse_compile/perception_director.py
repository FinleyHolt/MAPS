"""Per-episode perception direction: the segmenter watchlist and the boost.

The WATCHLIST is the class set every later flight segments; each change is
pushed whole to the live concept provider. A BOOST (``look_for(..., at=)``)
whose target is a flown stop, a fid a flown stop targeted, or a point near a
flown pose re-segments that stop now through ``retro_perceiver`` and returns
the delta. A boost on a target not yet flown (or on none) queues: the next
flight activates it, pushing its classes beside the watchlist, and retires
it after; a targeted boost whose target did not fly returns to pending.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

from maps.util.xy_token import parse_xy_token

log = logging.getLogger(__name__)

# How near a flown pose a point boost must fall to address that stop: about
# half the ~100 m stop view square.
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
    """The watchlist, the boosts, the flown stops and the retrospective
    executor."""

    # The live concept provider's setter; None keeps changes state-only.
    push: Callable[[list[str]], None] | None = None
    # Called with (flown entry, concepts); returns {"n_dets", "new_fids", ...}
    # or {"error": str}. None: a boost at a flown stop says re-segmentation
    # is not wired.
    retro_perceiver: Callable[[dict, list], dict] | None = None

    watchlist: list[str] = field(default_factory=list)
    boosts: list[_Boost] = field(default_factory=list)
    # Every flown stop: {"pose_xy", "alt_m", "stop_id", "target_fids"}.
    flown_log: list[dict] = field(default_factory=list)

    def init_watchlist(self, concepts) -> None:
        """Set the first declared watchlist and push it."""
        self.watchlist = _dedup_concepts(concepts)
        self._push_live()

    def watchlist_apply(self, add, remove) -> None:
        """Remove then add classes (canonical-deduped) and push the result."""
        remove_keys = {canon_concept(s) for s in (remove or []) if canon_concept(s)}
        kept = [w for w in self.watchlist if canon_concept(w) not in remove_keys]
        existing = {canon_concept(w) for w in kept}
        self.watchlist = kept + [a for a in _dedup_concepts(add)
                                 if canon_concept(a) not in existing]
        self._push_live()

    def _push_live(self) -> str:
        """Push the watchlist plus active boost classes to the provider:
        ``"ok"``, ``"no_channel"`` or ``"failed"``."""
        if self.push is None:
            return "no_channel"
        try:
            active = [c for b in self.boosts if b.active for c in b.concepts]
            self.push(_dedup_concepts(list(self.watchlist) + active))
            return "ok"
        except Exception:  # noqa: BLE001 — a push failure must not kill the round
            log.exception("PerceptionDirector: live concept push failed")
            return "failed"

    def _find_flown(self, w: dict) -> dict | None:
        """The flown stop a where entry addresses: by stop id, by a fid it
        targeted, or by a pose within the match radius of a point."""
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
        """One boost: each flown target re-segments now; an unflown fid or
        point (or no target at all) queues; an unknown stop id is named."""
        concepts = _dedup_concepts(concepts)
        if not concepts:
            return {"error": "boost needs a non-empty 'concepts' list"}
        resolved: list[dict] = []
        unresolved: list = []
        for w in where or []:
            xy = _xy_of(w) or parse_xy_token(w)
            if xy is not None:
                resolved.append({"kind": "xy", "xy": xy})
                continue
            token = str(w or "").strip()
            if not token:
                continue
            if token.startswith(("qstop_", "stop_")):
                resolved.append({"kind": "stop", "stop": token,
                                 "fid": "", "xy": None})
            else:
                resolved.append({"kind": "fid", "fid": token})
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
            # A targeted boost none of whose targets resolved says so rather
            # than becoming a where-less one.
            return {
                "error": ("no boost target resolved; nothing was queued or "
                          "re-segmented"),
                "unknown_where": unresolved,
                "pending": [],
            }
        return ack

    def activate_pending(self) -> list[str]:
        """Activate every pending boost for the next flight and push; returns
        their classes. A failed push leaves them pending."""
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
        """Log the flown stops, then consume each active boost that was
        where-less or whose target flew (the rest return to pending), and
        push."""
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
