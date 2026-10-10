"""The fuzz band weights coverage by IDF, over SATISFIABLE words only.

Counting query words equally let the corpus's commonest word carry a match by
itself. Eleven of the calcity surface's top thirteen near-stopword tokens were
BAKE PROVENANCE rather than map vocabulary -- ``way`` is a token of every
``osm_id: way/1136853905`` once the normalizer splits on '/', and
``worldkit_layer=roads`` donated ``road`` to all 810 of its rows -- so
``find("hugo way")`` covered one of two words on 4,733 of 5,257 rows and
``find("neuralia road")`` on 810.

TWO HALVES, AND NEITHER WORKS ALONE (measured as a 2x2):

* ``geom_program._SEARCH_PROVENANCE_DROP`` keeps the four provenance keys out
  of ``search_text``, which is what makes ``way`` and ``road`` ordinary words
  again. The node keeps every tag, so the exact tag tiers and
  ``tag_family_keys`` are untouched.
* :func:`text_match.token_idf` prices what is left, and the fuzz band divides
  by the IDF mass of the words SOME row could cover.

Dropping unsatisfiable words is not a refinement, it is what makes the
weighting usable at all: plain IDF takes ``find("parking lot")`` to ZERO rows
because ``lot`` has zero df, therefore maximum idf, and it vetoes the word
that does match. But dropping them ON ITS OWN re-opens the flood, because any
absent rare word leaves the common one as the whole denominator.

The two shapes are pinned SYNTHETICALLY below (they run on any checkout), and
the corpus assertions are derived from the calcity bake when it is staged.
A derived scan fails open, so the derivation carries a vacuity floor on the
vocabulary it found, and the two contracts the scan could not notice
disappearing are pinned BY NAME: ``find("parking lot")`` still reaches the
parking rows, and ``find("swimming pool")`` still returns its 34.
"""

from __future__ import annotations

import json
import os

import pytest

from maps.search.analyze_scene import text_match as T
from maps.search.analyze_scene.geom_program import (
    _SEARCH_PROVENANCE_DROP,
    _TAG_RENDER_DROP,
    build_surface_from_graph,
    run_program,
)
from maps.sws.graph.io import load_graph

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
# THE bake a Cal City run would use: ``SEARCH_CALCITY_TILES`` wins, else the
# shipped default -- the rule ``benchmarks.calcity.paths`` applies, read off
# the env directly because ``maps`` never imports ``benchmarks``.
_TILES = (os.environ.get("SEARCH_CALCITY_TILES")
          or os.path.join(_REPO, "benchmarks", "calcity", "tiles"))


def test_the_searchable_drop_is_a_subset_of_what_the_model_never_reads():
    """Withheld from the HAYSTACK only where it is already withheld from the
    EYES. The four keys were in the fuzzy haystack purely by omission -- the
    tree had already decided the model should not read them -- so this pins
    the two sets in the one direction that matters. Widening the search drop
    past the render drop would hide from search something the model can see
    quoted in its own table, which is a different and much worse change.
    """
    assert _SEARCH_PROVENANCE_DROP <= _TAG_RENDER_DROP, (
        _SEARCH_PROVENANCE_DROP - _TAG_RENDER_DROP)
    # ...and it is the FOUR, not the whole render drop: the full 16-key set
    # was measured at 80.3% precision against 79.8% for these four, which
    # does not pay for the blast radius.
    assert len(_SEARCH_PROVENANCE_DROP) == 4, _SEARCH_PROVENANCE_DROP


def _find(rows, query):
    """The served find, over a prepared surface."""
    out = run_program([], [dict(r) for r in rows],
                      {"ops": [{"ref": "r0", "op": "find",
                                "args": {"provenance": "prior",
                                         "query": query}}]})
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
    # Keeping it would have vetoed the word that matched: `lot` is absent,
    # so its df is 0 and its IDF is the maximum the corpus can express.
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
    graph = load_graph(os.path.join(_TILES, "calcity"))
    _cands, anchors = build_surface_from_graph(
        graph.nodes.items(), (float(b["x_min"]), float(b["y_min"])),
        float(meta["cell_size_m"]))
    return anchors


@needs_bake
def test_the_surface_vocabulary_is_real_and_prices_common_words_low(
        calcity_rows):
    idf = T.token_idf(calcity_rows)
    # VACUITY FLOOR. A df scan that found nothing would satisfy every
    # ordering below by vacuous truth. The calcity surface reads 979 distinct
    # tokens once the provenance keys are out of `search_text` (6,269 with
    # them in -- `osm_id` alone contributes one unique numeric token per row,
    # which is precisely the vocabulary that was drowning the real one).
    assert len(idf) >= 500, len(idf)
    assert len(calcity_rows) >= 5000, len(calcity_rows)
    # A street's generic head noun is common; the name that picks the street
    # out is rare. This is the whole ordering the fuzz band now turns on.
    assert idf["way"] < idf["hugo"], (idf["way"], idf["hugo"])
    assert idf["building"] < idf["avenue"], (idf["building"], idf["avenue"])


@needs_bake
def test_find_parking_lot_still_reaches_the_parking_rows(calcity_rows):
    """PINNED BY NAME: the contract the df scan above cannot notice going.

    ``lot`` is a zero-df word on this bake, so a plain IDF weighting scores
    every parking row 0 and this returns nothing at all.
    """
    got = _find(calcity_rows, "parking lot")
    assert len(got) == 75, len(got)
    assert all("parking" in (r.get("search_text") or "") for r in got)


@needs_bake
def test_find_swimming_pool_still_returns_its_thirty_four(calcity_rows):
    """PINNED BY NAME: the second contract the scan cannot notice going.

    31 of the 34 arrive through ``tag_family_keys`` -- ``swimming pool`` is
    an exact ``leisure`` VALUE -- so this also pins that the IDF weighting
    left the corpus widening alone.
    """
    assert len(_find(calcity_rows, "swimming pool")) == 34
    # The bare noun stays at the three rows that actually spell it: a single
    # satisfiable word is its own whole denominator, weighting or not.
    pool = _find(calcity_rows, "pool")
    assert len(pool) == 3, len(pool)
    assert all(r["tags"].get("leisure") == "swimming_pool" for r in pool)
