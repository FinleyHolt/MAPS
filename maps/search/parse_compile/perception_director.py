"""The segmenter watchlist (the classes every later flight segments, pushed
whole to the concept provider) and the boost: ``look_for(..., at=)`` on a
flown stop, a fid one targeted or a point near a flown pose re-segments that
stop now; on anything else it queues beside the watchlist until a flight
reaches its target (a where-less boost rides one flight)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

from maps.util.xy_token import parse_xy_token

log = logging.getLogger(__name__)

# How near a flown pose a point boost must fall to address that stop.
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


def _find_flown(w: dict, stops: list[dict]) -> dict | None:
    """The stop a where entry addresses: by stop id, by a fid it targeted, or
    by a pose within the match radius of a point."""
    sid = w.get("stop")
    if sid:
        return next((f for f in stops if sid == f["stop_id"]), None)
    fid = w.get("fid")
    if fid:
        return next((f for f in stops if fid in f["target_fids"]), None)
    wxy = w["xy"]
    return next((f for f in stops if f["pose_xy"] is not None and math.hypot(
        wxy[0] - f["pose_xy"][0], wxy[1] - f["pose_xy"][1]) <= BOOST_MATCH_RADIUS_M), None)


@dataclass
class _Boost:
    concepts: list[str]
    # {"fid": str} or {"xy": (x, y)} entries; empty = where-less.
    where: list[dict]
    # Set from the flight's start until its stop is recorded.
    riding: bool = False


@dataclass
class PerceptionDirector:
    """The watchlist, the queued boosts, the flown stops and the
    retrospective executor."""

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

    def watchlist_apply(self, add, remove) -> None:
        """Remove then add classes (canonical-deduped) and push the result."""
        remove_keys = {canon_concept(s) for s in (remove or []) if canon_concept(s)}
        kept = [w for w in self.watchlist if canon_concept(w) not in remove_keys]
        existing = {canon_concept(w) for w in kept}
        self.watchlist = kept + [a for a in _dedup_concepts(add)
                                 if canon_concept(a) not in existing]
        self._push_live()

    def _push_live(self) -> None:
        if self.push is None:
            return
        try:
            self.push(_dedup_concepts(
                self.watchlist + [c for b in self.boosts if b.riding for c in b.concepts]))
        except Exception:  # noqa: BLE001 -- a push failure never crashes the turn
            log.exception("PerceptionDirector: live concept push failed")

    def _retro(self, flown: dict, concepts: list[str], token) -> dict:
        if self.retro_perceiver is None:
            return {"where": token,
                    "error": ("retrospective re-segmentation is not wired "
                              "on this run; the flown observation cannot "
                              "be re-analyzed")}
        try:
            res = self.retro_perceiver(dict(flown), list(concepts))
        except Exception as exc:  # noqa: BLE001 -- served to the model; never crash the turn
            log.exception("retro perceiver raised")
            res = {"error": f"retrospective re-segmentation failed: "
                            f"{type(exc).__name__}: {exc}"}
        rec = {"where": token}
        if isinstance(res, dict):
            rec.update(res)
        else:
            rec["error"] = "retrospective re-segmentation returned nothing"
        return rec

    def boost_queue(self, concepts, where) -> dict:
        """One boost: each flown target re-segments now; an unflown fid or
        point (or no target at all) is acknowledged as queued; an unknown
        stop id is named."""
        concepts = _dedup_concepts(concepts)
        if not concepts:
            return {"error": "boost needs a non-empty 'concepts' list"}
        queued: list = []
        entries: list[dict] = []
        retro: list[dict] = []
        unknown: list[str] = []
        for w in where or []:
            xy = _xy_of(w) or parse_xy_token(w)
            token = str(w or "").strip()
            if xy is not None:
                entry, token = {"xy": xy}, [round(xy[0], 1), round(xy[1], 1)]
            elif not token:
                continue
            elif token.startswith("stop_"):
                entry = {"stop": token}
            else:
                entry = {"fid": token}
            flown = _find_flown(entry, self.flown_log)
            if flown is not None:
                retro.append(self._retro(flown, concepts, token))
            elif "stop" in entry:
                unknown.append(token)
            else:
                queued.append(token)
                entries.append(entry)
        ack: dict = {}
        if queued or not (where or []):
            self.boosts.append(_Boost(concepts, entries))
            ack = {"queued": concepts, "where": queued}
        elif not retro:
            # A targeted boost none of whose targets resolved says so rather
            # than becoming a where-less one.
            return {
                "error": ("no boost target resolved; nothing was queued or "
                          "re-segmented"),
                "unknown_where": unknown,
            }
        if retro:
            ack["retro"] = retro
        if unknown:
            ack["unknown_where"] = unknown
        return ack

    def start_flight(self) -> None:
        """Every queued boost rides the coming flight: push its classes
        beside the watchlist when one has not ridden yet."""
        if any(not b.riding for b in self.boosts):
            for b in self.boosts:
                b.riding = True
            self._push_live()

    def record_flown(self, flown: dict) -> None:
        """Log one flown stop for a boost to address, drop the riding boosts
        it satisfied (where-less, or its target is this stop), and push the
        watchlist."""
        stop = {
            "pose_xy": _xy_of(flown.get("pose_xy")),
            "alt_m": flown.get("alt_m"),
            "stop_id": (str(flown.get("stop_id")).strip()
                        if flown.get("stop_id") else None),
            "target_fids": [str(x) for x in (flown.get("target_fids") or [])
                            if x],
        }
        self.flown_log.append(stop)
        self.boosts = [b for b in self.boosts if not b.riding or (
            b.where and not any(_find_flown(w, [stop]) for w in b.where))]
        for b in self.boosts:
            b.riding = False
        self._push_live()
