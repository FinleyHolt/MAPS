"""The fuzz band weights coverage by IDF, over SATISFIABLE words only."""

from __future__ import annotations

import json
import os

import pytest

from maps.search.analyze_scene import text_match as T
from maps.search.analyze_scene.geom_program import run_program
from maps.search.analyze_scene.surface import build_surface_from_graph
from maps.sws.graph.builder import build_object_graph

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_TILES = (os.environ.get("SEARCH_CALCITY_TILES")
          or os.path.join(_REPO, "benchmarks", "calcity", "tiles"))


def _find(rows, query):
    """The served find, over a prepared surface."""
    out = run_program([], [dict(r) for r in rows],
                      {"ops": [{"ref": "r0", "op": "find",
                                "args": {"query": query}}]})
    return out["results"]["r0"]


# ---------------------------------------------------------------------------
# The mechanism, on a surface small enough to read. No bake needed.
# ---------------------------------------------------------------------------


def _row(fid, x, text, name=""):
    return {"fid": fid, "name": name, "xy": [float(x), 0.0],
            "tags": {}, "search_text": text}


# 60 rows spelling `way` and one spelling `hugo way`: the `hugo way` shape,
# shrunk. Equal-weighted coverage gives every one of the 60 a ratio of 1/2,
# clear of the 0.45 floor, and buries the row the query named.
_STOPWORD_SURFACE = (
    [_row(f"p_way_{i}", i * 10, "service way") for i in range(60)]
    + [_row("p_hugo", 999, "hugo way", name="Hugo Way")])


def test_a_common_word_cannot_carry_a_match_by_itself():
    got = [r["fid"] for r in _find(_STOPWORD_SURFACE, "hugo way")]
    # The vacuity floor for this pin: the surface really does spell `way` on
    # 60 rows, so a scorer that had already stopped matching them, and a
    # surface that never carried the word, both read as this test passing.
    assert sum(1 for r in _STOPWORD_SURFACE
               if "way" in r["search_text"]) == 61
    assert got == ["p_hugo"], got
    # ... and it is the WEIGHTING that dropped them, not a coverage floor:
    # unweighted, one of two words is exactly the 0.5 the threshold admits.
    idf = T.token_idf(_STOPWORD_SURFACE)
    assert idf["way"] < idf["hugo"], (idf["way"], idf["hugo"])
    clause = T.fuzzy_prepare_query("hugo way")[0]
    assert T.fuzzy_match_score(clause, "service way", idf) < T.FUZZY_THRESHOLD


# The same surface with a word NO row can cover: `lot`.
_UNSATISFIABLE_SURFACE = [
    _row("p_park_a", 0, "amenity parking"),
    _row("p_park_b", 10, "amenity parking"),
    _row("p_house", 20, "building house"),
]


def test_a_word_no_row_can_cover_leaves_the_denominator():
    # Vacuity floor: `lot` is genuinely absent from this surface, which is
    # the whole premise -- if some row spelled it, the test would pass while
    # asserting nothing about unsatisfiable words.
    assert "lot" not in T.token_idf(_UNSATISFIABLE_SURFACE)
    got = {r["fid"] for r in _find(_UNSATISFIABLE_SURFACE, "parking lot")}
    assert got == {"p_park_a", "p_park_b"}, got
    assert not _find(_UNSATISFIABLE_SURFACE, "helipad lot")


# ---------------------------------------------------------------------------
# The corpus, derived from the bake.
# ---------------------------------------------------------------------------

# Gated on the META FILE, not the directory: the tiles dir is committed with
# a `.gitkeep` and no bake, so `isdir` is true on every checkout and the gate
# would let the fixture raise instead of skipping.
_META = os.path.join(_TILES, "calcity_meta.json")
needs_bake = pytest.mark.skipif(not os.path.isfile(_META),
                                reason="calcity tile bake absent")


@pytest.fixture(scope="module")
def calcity_rows():
    meta = json.loads(open(_META).read())
    b = meta["bounds"]
    graph = build_object_graph(json.loads(open(os.path.join(_TILES, "calcity_features.json")).read()),
                               cell_size_m=float(meta["cell_size_m"]))
    _cands, anchors = build_surface_from_graph(
        graph.nodes.items(), (float(b["x_min"]), float(b["y_min"])),
        float(meta["cell_size_m"]))
    return anchors


@needs_bake
def test_the_surface_vocabulary_is_real_and_prices_common_words_low(
        calcity_rows):
    idf = T.token_idf(calcity_rows)
    # VACUITY FLOOR. A df scan that found nothing would satisfy every ordering below by
    # vacuous truth.
    assert len(idf) >= 500, len(idf)
    assert len(calcity_rows) >= 5000, len(calcity_rows)
    # A street's generic head noun is common; the name that picks the street out is
    # rare.
    assert idf["way"] < idf["hugo"], (idf["way"], idf["hugo"])
    assert idf["building"] < idf["avenue"], (idf["building"], idf["avenue"])


@needs_bake
def test_find_parking_lot_still_reaches_the_parking_rows(calcity_rows):
    got = _find(calcity_rows, "parking lot")
    assert len(got) == 75, len(got)
    assert all("parking" in (r.get("search_text") or "") for r in got)


@needs_bake
def test_find_swimming_pool_still_returns_its_thirty_four(calcity_rows):
    assert len(_find(calcity_rows, "swimming pool")) == 34
    # The bare noun stays at the three rows that actually spell it: a single
    # satisfiable word is its own whole denominator, weighting or not.
    pool = _find(calcity_rows, "pool")
    assert len(pool) == 3, len(pool)
    assert all(r["tags"].get("leisure") == "swimming_pool" for r in pool)
