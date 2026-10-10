"""Every claim the render makes about a REAL-bake surface, re-derived.

Two served claims are asserted here against the tile bake rather than a
fixture, because both were wrong in a way only real geometry shows:

* HEAD-ORDER — the truncated-head tie clause ("all N rows tie at
  name_match X, so this head's order is arbitrary") gated on name_match
  UNIFORMITY and not on whether an ordering exists. Every class find gives
  every row the same name_match, so it fired on the ordered output of every
  ``find <class> near <road>`` and told the model its own score_by was
  arbitrary. ``_CueSortedRows`` states the opposite in its own docstring
  and the index guard honours it; the render note now does too.

* SIDE-COMPASS — a ROW ref's ``left of``/``right of`` columns followed its
  end0->end1 travel, an order the bake picks arbitrarily, and the served
  note never said which way it ran. The columns went with every other
  engine-inferred axis, and the claim went with them: no row ref names a
  side any more. That absence is what is asserted below — an unmakeable
  claim is a stronger guarantee than a re-derivable one.

Reads the bake ``SEARCH_CITYNAV_TILES`` names, else ``tiles/citynav_2m_osmv3``
(the shipped default), and skips without it.
``pytest maps/search/tests/test_bake_render_claims.py``
"""
from __future__ import annotations

import glob
import json
import os
import re

import pytest

from maps.search.analyze_scene import geom_compute as G
from maps.search.analyze_scene.geom_program import (
    _BROAD_DUMP_SOFT_ROWS,
    _is_compare_column,
    build_surface_from_graph,
    compact_results,
    run_program,
)
from maps.sws.graph.io import load_graph

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# THE bake the run would use: ``SEARCH_CITYNAV_TILES`` wins, else the shipped
# default -- the rule ``benchmarks.citynav.paths`` applies, read off the env
# directly because ``maps`` never imports ``benchmarks``.
_TILES = (os.environ.get("SEARCH_CITYNAV_TILES")
          or os.path.join(_REPO, "tiles", "citynav_2m_osmv3"))
_GEOM_SRC = os.path.join(_REPO, "maps", "search", "analyze_scene",
                         "geom_program.py")

pytestmark = pytest.mark.skipif(not os.path.isdir(_TILES),
                                reason="tile bake absent")


def _blocks():
    return sorted(p[:-len("_meta.json")]
                  for p in glob.glob(os.path.join(_TILES, "*_meta.json")))


def _surface(stem):
    meta = json.loads(open(stem + "_meta.json").read())
    b = meta["bounds"]
    return build_surface_from_graph(load_graph(stem).nodes.items(),
                                    (b["x_min"], b["y_min"]),
                                    float(meta["cell_size_m"]))


def test_a_cue_sorted_head_is_not_told_its_own_order_is_arbitrary():
    """The ONE branch: the same rows, same name_match tie, same truncation
    — the tie clause fires on the retrieval-ordered find and is gone on the
    score_by output that resolved it."""
    armed = 0
    for stem in _blocks():
        cands, anchors = _surface(stem)
        lines = [a for a in anchors if a.get("centerline")]
        if not lines:
            continue
        ref = max(lines, key=lambda a: G._arc_lengths(a["centerline"])[-1])
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
        if len(pool) <= _BROAD_DUMP_SOFT_ROWS or len(nm) != 1 or None in nm:
            continue
        assert getattr(out["r2"], "cue_sorted", False), stem
        env = compact_results({"r0": pool, "r2": out["r2"]})
        assert "arbitrary" in (env["r0"].get("hint") or ""), (stem, env["r0"])
        # The clause states the fact columnlessly now: name_match no longer
        # renders as a grid column (audit: 7/8 not-observed, zero
        # reads-to-pick-a-row), so the hint must not name it or quote the
        # value of an invisible column.
        assert "name_match" not in (env["r0"].get("hint") or ""), env["r0"]
        assert "arbitrary" not in (env["r2"].get("hint") or ""), \
            (stem, env["r2"].get("hint"))
        armed += 1
    assert armed >= 3, armed          # vacuity floor: the branch ran


def test_the_tie_clause_still_exists_to_be_gated():
    """A derived absence test passes when the thing it looks for was
    deleted. This pins the clause itself, so removing the wording fails
    here instead of greening the test above.

    The clause is the RETRIEVAL's, not the render's: `find` compares the
    scores it computed and says either "best-first" or "equally good
    matches, so this order is arbitrary". The render used to re-derive the
    same fact from a uniform name_match column and glue it on, which is how
    one hint came to carry both claims (23 results in the audited corpus).
    """
    src = open(_GEOM_SRC).read()
    assert "so this order is arbitrary" in src
    assert 'if _m_vals[0] == _m_vals[-1]:' in src
    # ...and the render no longer has a second opinion about it.
    assert "so this head's order is " not in src


# The op-JSON register: the vocabulary of the frozen round-era surface.
_OP_REGISTER = re.compile(r"score_by|filter_items|find\s*\{")
# A served hint quotes what it wants read or said. Unquoted prose ("contact
# anywhere along the line") is ordinary English and out of scope.
_QUOTED = re.compile(r"'([^']{2,80})'")


def test_no_rendered_hint_teaches_the_op_json_register():
    """A remedy names the FACT, never a channel's spelling.

    geom_program renders for BOTH surfaces -- the sentence channel the bench
    flies and the frozen round-era one robots' field path serves -- so a hint
    written in op-JSON tells a model whose only move is a sentence to emit a
    call it cannot form. Two producers were rewritten and two survived: the
    truncated-head tie clause ("chain score_by/filter_items on a cue column")
    and the colour-miss remedy ("attached by find{color:'red'}"). On
    saydialect_1000 they reached 525 of 1,000 episodes, 676 B/episode.

    Rendered, not grepped from source: the source carries these identifiers
    legitimately (they are op names), and only the SERVED string matters.

    SECOND HALF -- a hint may not QUOTE a column that did not attach. The
    line-ref note said a line's "'left of road'/'right of road' ... attach";
    ``_side_pair`` runs ``sides=False`` on that path, so across 9,822 served
    grids of saydialect_1000 zero side columns rendered for any line ref,
    against 348 notes over 227 of the 1,000 episodes claiming the pair. The
    subset check catches that WITHOUT a removed-production ban list: a quoted
    name either names a column of this envelope or it is a claim the engine
    did not honour. 'along road' passes because it genuinely attaches; the
    model reads it as a table header whether or not the note names it.
    """
    bad: list[tuple[str, str, str]] = []
    seen = claims = 0
    for stem in _blocks():
        cands, anchors = _surface(stem)
        lines = [a for a in anchors if a.get("centerline")]
        if not lines:
            continue
        ref = max(lines, key=lambda a: G._arc_lengths(a["centerline"])[-1])
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
            cols = set((env or {}).get("columns") or [])
            named = {q for q in _QUOTED.findall(hint) if _is_compare_column(q)}
            claims += len(named)
            for miss in sorted(named - cols):
                bad.append((os.path.basename(stem), key,
                            f"claims {miss!r}, which did not attach"))
    assert not bad, (
        "served hints teach a spelling the sentence channel cannot emit, or "
        "claim a column that did not attach: "
        + "; ".join(f"{s}/{k}: {h!r}" for s, k, h in bad[:4]))
    # VACUITY FLOOR, measured on tiles/citynav_2m_osmv3 (34 blocks): 102
    # rendered hints. A walk that rendered no hint asserts nothing.
    assert seen >= 60, f"only {seen} rendered hints inspected"
    # NO hint quotes a compare column any more, and that IS the contract now.
    # The only note that did was the line-ref note, promising 'along <ref>'
    # plus a lateral 'distance to <ref>' in place of the footprint gap; a
    # road measures its gap like any other shape, so the note has no subject.
    # This replaces the old `claims >= 30` floor, which counted exactly that
    # note: keeping it would demand the deleted behaviour, and dropping it
    # silently would leave the claims-vs-columns check above passing over an
    # empty set forever.
    assert claims == 0, (
        f"{claims} hint(s) quote a compare column; the served hints stopped "
        "naming columns when the line-ref chart was deleted")


# --------------------------------------------------------------------------- #
# SIDE-COMPASS                                                                #
# --------------------------------------------------------------------------- #

def test_no_row_ref_carries_sides():
    """A side needs a travel direction, and no ROW ref has one: the bake
    picks a way's end0->end1 order arbitrarily, so a left/right column off it
    names a side of nothing. Only a DECLARED chart (frame:{along: ...}) has a
    direction its caller chose. Asserted over the shapes a ref can take — a
    footprint, a bare point, a union — because none of them is exempt."""
    checked = 0
    for stem in _blocks():
        cands, anchors = _surface(stem)
        areal = [a for a in anchors if a.get("mask")]
        if len(areal) < 2:
            continue
        refs = [{"ref": areal[0]["fid"], "as": "lot"},
                {"ref": [float(areal[1]["xy"][0]), float(areal[1]["xy"][1])],
                 "as": "pt"},
                {"ref": [a["fid"] for a in areal[:2]], "as": "both"}]
        r1 = run_program(cands, anchors, {"ops": [
            {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
            {"ref": "r1", "op": "compare",
             "args": {"items": "r0", "refs": refs}},
        ]})["results"]["r1"]
        assert len(r1.refs) == 3, r1.refs
        for hdr in r1.refs:
            assert "sides" not in hdr, (stem, hdr)
        checked += 1
        if checked >= 3:
            break
    assert checked >= 3, checked      # vacuity floor
