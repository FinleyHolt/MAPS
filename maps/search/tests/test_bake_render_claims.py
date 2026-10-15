"""Every claim the render makes about a REAL-bake surface, re-derived."""
from __future__ import annotations

import glob
import json
import os
import re

import pytest

from maps.search.analyze_scene import geom_compute as G
from maps.search.analyze_scene.geom_program import (
    BROAD_DUMP_SOFT_ROWS,
    _COMPARE_COL_PREFIXES,
    compact_results,
    run_program,
)
from maps.search.analyze_scene.surface import build_surface_from_graph
from maps.sws.graph.builder import build_object_graph

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# THE bake the run would use: ``SEARCH_CITYNAV_TILES`` wins, else the shipped
# default -- the rule ``benchmarks.citynav.paths`` applies, read off the env
# directly because ``maps`` never imports ``benchmarks``.
_TILES = (os.environ.get("SEARCH_CITYNAV_TILES")
          or os.path.join(_REPO, "tiles", "citynav_2m_osmv3"))

pytestmark = pytest.mark.skipif(not os.path.isdir(_TILES),
                                reason="tile bake absent")


def _blocks():
    return sorted(p[:-len("_meta.json")]
                  for p in glob.glob(os.path.join(_TILES, "*_meta.json")))


def _surface(stem):
    meta = json.loads(open(stem + "_meta.json").read())
    b = meta["bounds"]
    graph = build_object_graph(json.loads(open(stem + "_features.json").read()),
                               cell_size_m=float(meta["cell_size_m"]))
    return build_surface_from_graph(graph.nodes.items(),
                                    (b["x_min"], b["y_min"]),
                                    float(meta["cell_size_m"]))


def test_a_cue_sorted_head_is_not_told_its_own_order_is_arbitrary():
    armed = 0
    for stem in _blocks():
        cands, anchors = _surface(stem)
        lines = [a for a in anchors if a.get("centerline")]
        if not lines:
            continue
        ref = max(lines, key=lambda a: G.arc_lengths(a["centerline"])[-1])
        out = run_program(cands, anchors, {"ops": [
            {"ref": "r0", "op": "find", "args": {"query": "building"}},
            {"ref": "r1", "op": "compare",
             "args": {"items": "r0",
                      "refs": [{"ref": ref["fid"], "as": "road"}]}},
            {"ref": "r2", "op": "score_by",
             "args": {"items": "r1",
                      "emphasis": [{"key": "distance to road", "w": -1}]}},
        ]})["results"]
        pool = out["r0"]
        nm = {r.get("name_match") for r in pool}
        # BRANCH FLOOR: the clause only exists past the soft-dump knee AND
        # on a single-valued name_match — without both, neither assertion
        # below is testing anything.
        if len(pool) <= BROAD_DUMP_SOFT_ROWS or len(nm) != 1 or None in nm:
            continue
        assert getattr(out["r2"], "ordered", False), stem
        env = compact_results({"r0": pool, "r2": out["r2"]})
        assert "arbitrary" in (env["r0"].get("hint") or ""), (stem, env["r0"])
        assert "name_match" not in (env["r0"].get("hint") or ""), env["r0"]
        assert "arbitrary" not in (env["r2"].get("hint") or ""), \
            (stem, env["r2"].get("hint"))
        armed += 1
    assert armed >= 3, armed          # vacuity floor: the branch ran


_OP_REGISTER = re.compile(r"score_by|filter_items|find\s*\{")
# A served hint quotes what it wants read or said. Unquoted prose ("contact
# anywhere along the line") is ordinary English and out of scope.
_QUOTED = re.compile(r"'([^']{2,80})'")


def test_no_rendered_hint_teaches_the_op_json_register():
    """A remedy names the FACT, never a channel's spelling."""
    bad: list[tuple[str, str, str]] = []
    seen = claims = 0
    for stem in _blocks():
        cands, anchors = _surface(stem)
        lines = [a for a in anchors if a.get("centerline")]
        if not lines:
            continue
        ref = max(lines, key=lambda a: G.arc_lengths(a["centerline"])[-1])
        out = run_program(cands, anchors, {"ops": [
            {"ref": "r0", "op": "find", "args": {"query": "building"}},
            {"ref": "r1", "op": "compare",
             "args": {"items": "r0",
                      "refs": [{"ref": ref["fid"], "as": "road"}]}},
            # A palette colour word matches no row lexically, which is the
            # branch the colour remedy rides.
            {"ref": "r2", "op": "find", "args": {"query": "red"}},
        ]})["results"]
        for key, env in compact_results(dict(out)).items():
            hint = (env or {}).get("hint") or ""
            if not hint:
                continue
            seen += 1
            if _OP_REGISTER.search(hint):
                bad.append((os.path.basename(stem), key, hint[:200]))
            claims += sum(q.startswith(_COMPARE_COL_PREFIXES) for q in _QUOTED.findall(hint))
    assert not bad, (
        "served hints teach a spelling the sentence channel cannot emit: "
        + "; ".join(f"{s}/{k}: {h!r}" for s, k, h in bad[:4]))
    # A walk that rendered no hint asserts nothing.
    assert seen >= 60, f"only {seen} rendered hints inspected"
    # A road measures its gap like any other shape, so no hint names a column.
    assert claims == 0, f"{claims} hint(s) quote a compare column"
