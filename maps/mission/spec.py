"""The mission contract a benchmark or the flight harness fills: the engine
half the loop calls (:class:`MissionSpec`) and how the conversation ended
(:class:`MissionReply`)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class MissionSpec:
    """The engine half of the mission contract: ``fly`` flies one move item
    (``{"target_fid": <fid or "[x, y]">}``) and returns its report or
    ``{"error": ...}``; ``declare_concepts`` arms the watchlist and returns
    the error a failed arming serves, or None."""

    fly: Callable[[dict], dict]
    declare_concepts: Callable[[list[str]], str | None]
    # ``coverage_stops(fid, spacing_m)``: an explore's stops over that fid's
    # footprint, or None when it carries no cells.
    coverage_stops: Callable[[str, float], list | None] | None = None
    #: Which stops of a pass the recording already holds
    #: (``[(token, xy or None), ...] -> [bool, ...]``); a held stop flies
    #: first. None on the bench.
    held_now: Callable[[list], list[bool]] | None = None
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
    "request_refused" | "oversize_rejection" | "content_filter" |
    "compaction_infeasible"."""

    ended_by: str
    turns_used: int = 0
