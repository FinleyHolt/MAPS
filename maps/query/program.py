"""The op-program executor: runs a compiled op program over the ``(cands, anchors)`` surface.

:func:`run_program` executes the ``{"ops": [{ref, op, args}, ...]}`` the op
compiler emits, with :mod:`maps.query.spatial_math` primitives; every op
returns rows, and :func:`maps.query.result_tables.compact_results` renders
them for the model. An ``items`` value is a handle -- ``rN`` (a result of this
program or an earlier call) or a window ``rN[a:b]`` -- or a fid list; a ref is
a fid, an ``[x, y]``, or ``{ref, as}`` naming a fid, a point, or a set (a
union)."""

from __future__ import annotations

from maps.query.op_compare import _compare
from maps.query.op_cut_and_sort import _filter_items, _score_by
from maps.query.op_find import _find, _stratum
from maps.query.op_io import ProgramError, _Args, _Run
from maps.query.op_roads import _crossings, _ends


OPS = {
    "find": _find,
    "crossings": _crossings,
    "ends": _ends,
    "compare": _compare,
    "filter_items": _filter_items,
    "score_by": _score_by,
    "stratum": _stratum,
}


def run_program(cands, anchors, program, seed_results=None,
                staged=None, standing=None):
    """Run ``{"ops": [{ref, op, args}, ...]}`` (dependency order) over the
    surface rows; returns ``{"results": {ref: rows}}``. A handle defined here
    shadows its ``seed_results`` entry; ``staged`` maps stand() handles to
    ``{"handle", "from_xy", "up_unit"}`` and ``standing`` names the one whose
    side columns ride every find."""
    cx = _Run(cands, anchors, seed_results, staged, standing)
    ops = program["ops"]
    for op in ops:
        cx.op = op["op"]
        # A null arg is an absent one: a handle bound to a row without a fid
        # compiles to items=None.
        try:
            cx.results[op["ref"]] = OPS[op["op"]](cx, _Args(
                {k: v for k, v in op["args"].items() if v is not None}, cx))
        except ProgramError as exc:
            exc.op = cx.op  # the caller's repeated-error class
            exc.ref = op["ref"]   # the handle whose op refused
            raise
    return {"results": cx.results}
