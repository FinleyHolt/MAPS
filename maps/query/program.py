"""The op-program executor: runs a compiled op program over the ``(cands, anchors)`` surface.

:func:`run_program` executes the ``{"ops": [{ref, op, args}, ...]}`` the op
compiler emits, with :mod:`maps.query.spatial_math` primitives; every op
returns rows, and :func:`maps.query.result_tables.compact_results` renders
them for the model. An ``items`` value is a handle -- ``rN`` (a result of this
program or an earlier call) or a window ``rN[a:b]`` -- or a fid list; a ref is
a fid, an ``[x, y]``, or ``{ref, as}`` naming a fid, a point, or a set (a
union)."""

from __future__ import annotations

from maps.query.op_compare import run_compare
from maps.query.op_cut_and_sort import run_filter_items, run_score_by
from maps.query.op_find import run_find, run_stratum
from maps.query.op_io import OpArgs, ProgramError, ProgramRun, as_rows
from maps.query.op_roads import run_crossings, run_ends

OPS = {
    "find": run_find,
    "crossings": run_crossings,
    "ends": run_ends,
    "compare": run_compare,
    "filter_items": run_filter_items,
    "score_by": run_score_by,
    "stratum": run_stratum,
}


def run_program(cands, anchors, program, seed_results=None, staged=None, standing=None, watched=()):
    """Run ``{"ops": [{ref, op, args}, ...]}`` (dependency order) over the
    surface rows; returns ``{"results": {ref: rows}}``. A handle defined here
    shadows its ``seed_results`` entry; ``staged`` maps stand() handles to
    ``{"handle", "from_xy", "up_unit"}`` and ``standing`` names the one whose
    side columns ride every find; ``watched`` is the armed watchlist."""
    cx = ProgramRun(cands, anchors, seed_results, staged, standing, watched)
    ops = program["ops"]
    for op in ops:
        cx.op = op["op"]
        # A null arg is an absent one: a handle bound to a row without a fid
        # compiles to items=None.
        try:
            res = cx.results[op["ref"]] = OPS[op["op"]](
                cx, OpArgs({k: v for k, v in op["args"].items() if v is not None}, cx)
            )
        except ProgramError as exc:
            exc.op = cx.op  # the caller's repeated-error class
            exc.ref = op["ref"]  # the handle whose op refused
            raise
        if not res and op["args"].get("items") is not None:
            # An op over an empty input answers with the stats of the cut
            # that emptied it, so every later answer still says why.
            src = as_rows(cx, op["args"]["items"])
            if not src and getattr(src, "cut_stats", None):
                res.cut_stats = src.cut_stats
    return {"results": cx.results}
