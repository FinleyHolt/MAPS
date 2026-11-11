"""The verbs behind ``look_for``: arming the segmenter watchlist and the
targeted boost."""

from __future__ import annotations

import logging


log = logging.getLogger(__name__)


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


class _WatchVerbs:
    """The verbs that arm the segmenter: the watchlist (``look_for``) and the
    targeted boost (``look_for(..., at=)``)."""

    def _watching_trailer(self) -> list:
        """The armed watchlist with each class's live detection count."""
        counts: dict[str, int] = {}
        for r in self.ns.get("cands") or []:
            if str(r["fid"]).startswith("detected_"):
                cls = str(r.get("class") or "?")
                counts[cls] = counts.get(cls, 0) + 1
        return [f"{c} ({counts.get(c, 0)} detected)" for c in self.mission_sam]

    def _watch_apply(self, adds: list[str]) -> dict:
        """look_for(): add classes to the watchlist (a class declared again
        moves to the end; the list only grows) and arm it with the engine."""
        for add in adds:
            self.mission_sam[:] = [c for c in self.mission_sam if c != add]
            self.mission_sam.append(add)
        err = self.mission_spec.declare_concepts(list(self.mission_sam))
        ack: dict = {"now": [f"{c} (target)" for c in self.mission_sam]}
        if err:
            ack["error"] = err
        names = {s for s in (
            str(n.name or "").strip().lower()
            for n in self.ctx.graph.nodes.copy().values()) if s}
        note = _named_feature_note(list(adds), names)
        if note:
            ack["note"] = note
        log.info("%s SAY watch: +%s -> %s", self.rid_tag, adds, self.mission_sam)
        return ack

    def _boost(self, where: str, concepts: list) -> dict:
        """look_for(..., at=) -- the targeted boost."""
        res = self.boost_queue(list(concepts), [where]) or {}
        if "error" not in res:
            # The placed form also arms the class it names.
            adds = [str(c) for c in concepts if c]
            if adds:
                w = self._watch_apply(adds)
                res["watching"] = w.get("error") or w.get("now")
            if res.get("queued"):
                res["note"] = ("boost queued -- extra segmentation at the "
                               "asked spot fires on your next flight; the "
                               "class is also on the standing watchlist for "
                               "every later flight")
            # Any accepted boost refreshes the surface and bumps its version,
            # as a flight does (a retro boost has ingested detections).
            self._world_moved()
        return self._acked("boost", res)

    def _watch(self, concepts: list) -> dict:
        return self._acked("watching", self._watch_apply(list(concepts)))
