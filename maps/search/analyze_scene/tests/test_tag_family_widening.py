"""``find <noun>`` reaches the noun's own tag FAMILY, exact hits first."""
from __future__ import annotations

from maps.search.analyze_scene import text_match as T
from maps.search.analyze_scene.geom_program import run_program

# `house` is a value of `building` here and of nothing else; `residential`
# is a value of `highway`. A row keyed `highway` must therefore stay OUT of
# the `house` family -- the widening reaches one key, not the surface.
_ANCHORS = [
    {"fid": "p_house", "name": "", "xy": [0.0, 0.0],
     "tags": {"building": "house"}, "search_text": "building house"},
    {"fid": "p_yes_a", "name": "", "xy": [30.0, 0.0],
     "tags": {"building": "yes"}, "search_text": "building yes"},
    {"fid": "p_yes_b", "name": "", "xy": [60.0, 0.0],
     "tags": {"building": "yes"}, "search_text": "building yes"},
    {"fid": "p_road", "name": "Earl Street", "xy": [0.0, 40.0],
     "tags": {"highway": "residential"},
     "search_text": "earl street highway residential"},
]


def _find(query, anchors=None):
    out = run_program([], [dict(a) for a in (anchors or _ANCHORS)],
                      {"ops": [{"ref": "r0", "op": "find",
                                "args": {"query": query}}]})
    return out["results"]["r0"]


def _narrow(query, anchors=None):
    """The match set the per-row scorer alone would return -- the BEFORE."""
    rows = anchors or _ANCHORS
    q = T.fuzzy_prepare_query(query)
    idf = T.token_idf(rows)
    return {r["fid"] for r in rows
            if T.find_score(q, r.get("search_text") or "", r.get("name"),
                            r.get("tags"), idf=idf) >= T.FUZZY_THRESHOLD}


def test_tag_family_keys_reads_values_and_not_keys():
    assert T.tag_family_keys(T.fuzzy_prepare_query("house"),
                             _ANCHORS) == frozenset({"building"})
    assert T.tag_family_keys(T.fuzzy_prepare_query("residential"),
                             _ANCHORS) == frozenset({"highway"})
    # A tag KEY is not a value of anything -- `find building.` already
    # retrieves the family through the exact-tag tier and must not widen
    # twice.
    assert T.tag_family_keys(T.fuzzy_prepare_query("building"),
                             _ANCHORS) == frozenset()
    # A word the surface does not spell reaches nothing.
    assert T.tag_family_keys(T.fuzzy_prepare_query("helipad"),
                             _ANCHORS) == frozenset()
    assert T.tag_family_keys((), _ANCHORS) == frozenset()


def test_find_widens_to_the_family_with_the_exact_hits_still_first():
    rows = _find("house")
    got = [r["fid"] for r in rows]
    # The BEFORE is non-empty and strictly smaller -- the vacuity floor: a
    # widening that fired on nothing, and a scorer that had already matched
    # everything, both read as this test passing.
    before = _narrow("house")
    assert before == {"p_house"}, before
    assert set(got) > before, (got, before)
    # WIDENED, not dumped: the highway row is not in the building family.
    assert set(got) == {"p_house", "p_yes_a", "p_yes_b"}, got
    # The exact `building=house` hit keeps the head of the find order.
    assert got[0] == "p_house", got
    scores = [r["name_match"] for r in rows]
    assert scores == sorted(scores, reverse=True), scores
    assert scores[0] >= 0.80, scores
    assert scores[1:] == [T.TAG_FAMILY_SCORE] * 2, scores


def test_the_family_score_is_retrieved_and_ranked_under_every_exact_tier():
    # The only two constraints on the constant. Below it the family is not
    # retrieved at all; at or above the exact tiers it outranks the rows the
    # query actually names.
    assert T.FUZZY_THRESHOLD < T.TAG_FAMILY_SCORE < T.KV_TIER_BASE


def test_a_name_query_is_untouched_by_the_widening():
    # A proper-name query is a value of no tag key, so the family is empty
    # and the retrieval is the scorer's alone.
    rows = _find("Earl Street")
    assert [r["fid"] for r in rows] == ["p_road"], rows
    assert _narrow("Earl Street") == {"p_road"}
