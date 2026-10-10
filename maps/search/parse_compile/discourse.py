"""The mission's discourse state, and the register echo on a result.

ONE discourse dict lives on the loop for the whole mission, carrying what an
earlier block bound and a later one may refer to:

  ``those``    the last non-empty result handle, so a later narrowing has a
               set in scope. An EMPTY tail must never take the slot, or the
               rows it failed to reach become unreachable.
  ``widened``  how far past a handle's head the agent has asked, so a second
               ``more_of`` on the same handle serves the next block instead
               of the same one. Written only by a plan that EXECUTED: a
               refused widening must not advance the window.

``update_discourse`` applies a SUCCESSFULLY executed plan's writes; a plan
that raised must never reach it. ``echo`` orders one executed call's served
result -- what ran, then the result envelopes byte-identical. It never states
a count of turns or flights, because budgets are liveness backstops the model
must not hear about.
"""

from __future__ import annotations


def new_discourse() -> dict:
    return {"those": None, "widened": {}}


def update_discourse(plan: dict, result_rows, discourse: dict) -> None:
    """Apply an executed plan's discourse writes."""
    if plan.get("widened"):
        discourse["widened"] = {**(discourse.get("widened") or {}),
                                **plan["widened"]}
    if plan.get("last_ref") and result_rows:
        # NON-EMPTY only: an empty filter tail must not become "the set in
        # scope"; the last set with rows stays live for a later narrowing.
        discourse["those"] = plan["last_ref"]


def echo(plan: dict, ack: dict) -> dict:
    """Order the served result: what ran, then the envelopes."""
    out: dict = {"said": plan["said"]}
    for k, v in ack.items():
        if k != "said":
            out[k] = v
    return out
