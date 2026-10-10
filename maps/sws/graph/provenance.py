"""Provenance classification over the one enriched world-state graph.

The runtime carries a single ``ObjectGraph`` that starts as the baked prebake
graph and accumulates SAM-detected nodes in flight (enrichment is ephemeral —
never serialized, so every run starts at the prebake seed). A node's
provenance is derived, not stored, from two fields that already exist on
``MapObject``:

  - ``provenance``          — ``"prebaked"`` (map prior) or ``"detected"``
                              (SAM, in flight).
  - ``matched_prebake_id``  — written on every detected-node upsert by the
                              geometry-only mask-overlap matcher in
                              ``maps.sws.graph.runtime``. The link points
                              from a detection to the prebake footprint it
                              sits inside.

Inverting those links yields three states the agent reasons over:

  PRIOR            a prebake node no detection has matched.
  PRIOR_CONFIRMED  a prebake node >= 1 detection has matched (the drone has
                   now seen the map feature).
  RUNTIME          a detected node with no prebake match (a fresh in-flight
                   object the map did not have).

A detected node that *did* match a prebake node is folded into that prebake
node: the surface yields the prebake row tagged ``PRIOR_CONFIRMED`` (it
carries the precise footprint geometry the matcher rewards detections for
sitting inside), with the detection's perception evidence attached read-only,
and suppresses the detected twin so the same physical object is one row.

The matcher is geometry-only by audited decision (see
``world_state/graph/runtime.py`` ``_find_prebake_overlap_match_batch``: SAM
open-vocab labels and CityRefer categories describe the same physical object
at different granularities, so a class-string gate produced false negatives).
This module classifies on top of the matcher's stored output; it never adds
a second association rule and must not reintroduce class gating.
"""

from __future__ import annotations

from maps.sws.graph.model import ObjectGraph

PRIOR = "prior"
PRIOR_CONFIRMED = "prior_confirmed"
RUNTIME = "runtime"


def build_confirmed_set(graph: ObjectGraph | None) -> set[str]:
    """Prebake ids that at least one detected node has matched.

    One O(N_detected) pass. Recompute once per ``ToolContext`` build.
    """
    if graph is None:
        return set()
    confirmed: set[str] = set()
    for node in graph.nodes.values():
        if getattr(node, "provenance", "prebaked") != "detected":
            continue
        pid = getattr(node, "matched_prebake_id", None)
        if pid:
            confirmed.add(pid)
    return confirmed
