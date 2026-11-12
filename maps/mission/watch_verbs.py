"""The verbs behind ``look_for``: arming the segmenter watchlist and the
targeted boost."""

from __future__ import annotations

import logging
from collections.abc import Callable

from maps.mission.acks import Rejections
from maps.mission.spec import MissionSpec
from maps.mission.surface_state import Progress, Surface

log = logging.getLogger(__name__)


def _named_feature_note(classes: list[str], names: set[str]) -> str | None:
    """A note when a watch class IS a world-state name (the watchlist takes
    visual classes); the class is watched as said."""
    named = [c for c in classes if c.strip().lower() in names]
    if not named:
        return None
    q = ", ".join(f"'{c}'" for c in named)
    return (
        f"{q}: a NAMED feature on this map, already a world-state row "
        f"-- the segmenter takes visual classes like 'house' or "
        f"'pool'. Watched as said."
    )


class Watchlist:
    """The verbs that arm the segmenter: the watchlist (``look_for``) and the
    targeted boost (``look_for(..., at=)``)."""

    def __init__(
        self,
        spec: MissionSpec,
        surface: Surface,
        rejections: Rejections,
        progress: Progress,
        *,
        boost_queue: Callable[[list, list], dict],
    ):
        self.spec = spec
        self.surface = surface
        self.rejections = rejections
        self.progress = progress
        self.boost_queue = boost_queue
        #: The watched classes, in declaration order.
        self.classes: list[str] = []

    def trailer(self) -> list:
        """The armed watchlist with each class's live detection count."""
        counts: dict[str, int] = {}
        for r in self.surface.ns.get("cands") or []:
            if str(r["fid"]).startswith("detected_"):
                cls = str(r.get("class") or "?")
                counts[cls] = counts.get(cls, 0) + 1
        return [f"{c} ({counts.get(c, 0)} detected)" for c in self.classes]

    def _arm(self, adds: list[str]) -> dict:
        """look_for(): add classes to the watchlist (a class declared again
        moves to the end; the list only grows) and arm it with the engine."""
        for add in adds:
            self.classes[:] = [c for c in self.classes if c != add]
            self.classes.append(add)
        err = self.spec.declare_concepts(list(self.classes))
        ack: dict = {"now": [f"{c} (target)" for c in self.classes]}
        if err:
            ack["error"] = err
        names = {
            s
            for s in (
                str(n.name or "").strip().lower()
                for n in self.surface.ctx.graph.nodes.copy().values()
            )
            if s
        }
        note = _named_feature_note(list(adds), names)
        if note:
            ack["note"] = note
        log.info("%s SAY watch: +%s -> %s", self.progress.rid_tag, adds, self.classes)
        return ack

    def boost(self, where: str, concepts: list) -> dict:
        """look_for(..., at=) -- the targeted boost."""
        res = self.boost_queue(list(concepts), [where]) or {}
        if "error" not in res:
            # The placed form also arms the class it names.
            adds = [str(c) for c in concepts if c]
            if adds:
                w = self._arm(adds)
                res["watching"] = w.get("error") or w.get("now")
            if res.get("queued"):
                res["note"] = (
                    "boost queued -- extra segmentation at the "
                    "asked spot fires on your next flight; the "
                    "class is also on the standing watchlist for "
                    "every later flight"
                )
            # Any accepted boost refreshes the surface and bumps its version,
            # as a flight does (a retro boost has ingested detections).
            self.surface.world_moved()
        return self.rejections.acked("boost", res)

    def watch(self, concepts: list) -> dict:
        """look_for(): the classes join the watchlist."""
        return self.rejections.acked("watching", self._arm(list(concepts)))
