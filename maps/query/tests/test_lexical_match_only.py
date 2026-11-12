"""``find``'s text matching is strictly LEXICAL — no semantic lookup table."""

from __future__ import annotations

import pytest

from maps.query import text_match as T
from maps.query.op_io import ProgramError
from maps.query.program import run_program


def _flat(*texts: str) -> dict:
    return T.token_idf([{"search_text": " ".join(texts), "name": ""}])


def _score(query: str, candidate: str) -> float:
    idf = _flat(query, candidate)
    return max(
        T.fuzzy_match_score(clause, candidate, idf) for clause in T.fuzzy_prepare_query(query)
    )


def _rows(*texts: str, colour: str | None = None) -> list[dict]:
    """Matched rows as ``unmatched_query_tokens`` reads them: the searched text, plus
    perception's ``color`` mixture when the row carries one."""
    return [
        dict({"search_text": t}, **({"color": colour} if colour is not None else {})) for t in texts
    ]


def test_semantic_gap_scores_zero_census_is_the_remedy():
    # 'lake' shares no token / no >=0.7-similarity token with natural=water
    # search_text: the honest score is 0.0 (below FUZZY_THRESHOLD), which
    # routes the agent to the served tag census — not a silent hit.
    assert _score("lake", "natural water") == 0.0
    assert _score("trail", "highway footway") < T.FUZZY_THRESHOLD
    assert _score("gas station", "amenity fuel") == 0.0


def test_lexical_token_overlap_reaches_tag_vocabulary():
    # 'parking lot' reaches the parking tag family LEXICALLY: the token
    # 'parking' overlaps the tag key/value tokens, no table needed.
    assert _score("parking lot", "amenity parking") >= T.FUZZY_THRESHOLD
    assert _score("parking lot", "parking surface") >= T.FUZZY_THRESHOLD
    assert _score("lot", "amenity parking") == 0.0  # bare 'lot' has no token

    # 'parking' is then the whole of the denominator and the row saturates the band.
    corpus = [
        {"search_text": "amenity parking", "name": ""},
        {"search_text": "building yes", "name": ""},
    ]
    idf = T.token_idf(corpus)
    assert "lot" not in idf
    (clause,) = T.fuzzy_prepare_query("parking lot")
    assert T.fuzzy_match_score(clause, "amenity parking", idf) == T.TOKEN_BAG_CEIL
    # ...and dropping it is not "ignore the word": a query of ONLY
    # unsatisfiable words still scores nothing.
    (none_of_it,) = T.fuzzy_prepare_query("lot")
    assert T.fuzzy_match_score(none_of_it, "amenity parking", idf) == 0.0


def test_abbreviation_expansion_is_kept():
    # st <-> street and friends are LEXICAL normalization, not semantics.
    assert _score("main st", "Main Street") == 1.0
    assert _score("elm ave", "Elm Avenue") == 1.0
    assert _score("city hosp", "City Hospital") == 1.0


def test_plural_fuzz_is_kept_and_typo_fuzz_is_GONE():
    assert _score("buildings", "building yes") >= T.FUZZY_THRESHOLD
    # ...and the -ses ambiguity does not eat the stem: houses -> house, not "hous".
    assert _score("houses", "building house") >= T.FUZZY_THRESHOLD
    assert _score("house", "building house") >= T.FUZZY_THRESHOLD
    assert _score("librari", "library") == 0.0


def test_containment_fuzz_never_enters_the_exact_tier_bands():
    """A row NAME contained in the query (reverse containment — the admin-boundary
    'Cambridge' vs a 'Museum of Cambridge' query) is a fuzz hit and must stay at or
    below the fuzz ceiling, never inside the k=v (0.80+) or name-phrase (0.90+) bands
    the served docs declare."""
    from maps.query.text_match import TOKEN_BAG_CEIL, find_score, fuzzy_prepare_query

    q = fuzzy_prepare_query("Museum of Cambridge")
    idf = _flat("boundary administrative cambridge", "tourism museum the museum of cambridge")
    squatter = find_score(
        q, "boundary=administrative", name="Cambridge", tags={"boundary": "administrative"}, idf=idf
    )
    assert 0.0 < squatter <= TOKEN_BAG_CEIL
    named = find_score(
        q, "tourism=museum", name="The Museum of Cambridge", tags={"tourism": "museum"}, idf=idf
    )
    assert named > 0.9
    assert named > squatter


def test_a_query_is_scored_by_what_it_found_not_by_the_haystack_length():
    """The partial paths are alternatives; the score is the MAX over them."""
    # `pool` is a single satisfiable word, so it is its own whole denominator and the
    # IDF MASS FOUND over IDF MASS SATISFIABLE saturates the band whatever else the row
    # happens to spell.
    corpus = [
        {"search_text": "leisure swimming pool", "name": ""},
        {"search_text": "leisure swimming pool building yes access private", "name": ""},
        {"search_text": "highway residential", "name": ""},
    ]
    idf = T.token_idf(corpus)
    q = T.fuzzy_prepare_query("pool")[0]
    bare = T.fuzzy_match_score(q, "leisure=swimming_pool", idf)
    with_more_tags = T.fuzzy_match_score(
        q, "leisure=swimming_pool building=yes access=private", idf
    )
    assert bare == with_more_tags, (bare, with_more_tags)
    assert with_more_tags >= T.FUZZY_THRESHOLD, with_more_tags

    # VACUITY FLOOR: a query whose every word IS found still saturates the
    # band. The weighting decides which PARTIAL hits survive; it never
    # discounts a complete one.
    ms_idf = T.token_idf([{"search_text": "old main street", "name": ""}])
    assert (
        T.fuzzy_match_score(T.fuzzy_prepare_query("main street")[0], "old main street", ms_idf)
        > 0.6
    )
    # ...the fuzz ceiling still holds, so no fuzz path reaches an exact tier.
    assert (
        max(
            T.fuzzy_match_score(q, c, idf)
            for c in ("leisure=swimming_pool", "pool house", "swimming pool")
        )
        <= T.TOKEN_BAG_CEIL
    )
    # ...and a query with nothing in common still scores nothing.
    assert T.fuzzy_match_score(q, "highway=residential", idf) == 0.0


def test_single_token_generic_query_never_enters_name_phrase_tier():
    """Query 'building' stays out of the name-phrase band (>=0.90) for 'Baker Building'."""
    from maps.query.text_match import KV_TIER_BASE, TOKEN_BAG_CEIL, find_score, fuzzy_prepare_query

    q = fuzzy_prepare_query("building")
    idf = _flat("baker building", "building terrace")
    name_mention = find_score(
        q, "baker building", name="Baker Building", tags={"amenity": "university"}, idf=idf
    )
    assert name_mention <= TOKEN_BAG_CEIL, name_mention
    tag_hit = find_score(q, "building terrace", name=None, tags={"building": "terrace"}, idf=idf)
    assert KV_TIER_BASE <= tag_hit < 0.90, tag_hit
    assert tag_hit > name_mention


def test_multi_token_name_phrase_tier_kept():
    """'Downing Street' keeps its 1.0 exact / 0.9x phrase-inside-name scores."""
    from maps.query.text_match import find_score, fuzzy_prepare_query

    q = fuzzy_prepare_query("Downing Street")
    idf = _flat("downing street", "little downing street")
    exact = find_score(q, "downing street", name="Downing Street", tags={}, idf=idf)
    assert exact == 1.0
    phrase = find_score(q, "little downing street", name="Little Downing Street", tags={}, idf=idf)
    assert 0.90 <= phrase < 1.0, phrase


def test_single_token_exact_name_equality_still_perfect():
    # Equality is tier 1.0 regardless of token count — only the
    # phrase-INSIDE-a-longer-name tier requires a multi-token query.
    from maps.query.text_match import find_score, fuzzy_prepare_query

    q = fuzzy_prepare_query("grass")
    assert find_score(q, "grass", name="Grass", tags={}, idf=_flat("grass")) == 1.0


def test_7872_multiword_value_coverage_separates_compound_query():
    from maps.query.text_match import find_score, fuzzy_prepare_query

    q = fuzzy_prepare_query("addr_street=Castle Street building=yes")
    idf = _flat("castle street building yes 12", "building yes")
    both = find_score(
        q,
        "castle street building yes",
        name=None,
        tags={"addr_street": "Castle Street", "building": "yes", "addr_housenumber": "12"},
        idf=idf,
    )
    bldg_only = find_score(q, "building yes", name=None, tags={"building": "yes"}, idf=idf)
    assert both > bldg_only, (both, bldg_only)
    # full coverage tops the k=v band; half coverage sits mid-band
    assert abs(both - 0.89) < 1e-9, both
    assert abs(bldg_only - 0.845) < 1e-9, bldg_only


def test_kv_value_extension_stops_at_the_next_pair():
    # 'a=b c=d' keeps two independent single-token pairs — the extension
    # consumes only BARE tokens, so pair grammar is unchanged.
    from maps.query.text_match import fuzzy_prepare_query

    (clause,) = fuzzy_prepare_query("landuse=grass building=yes")
    groups = clause[3]
    assert groups == ((("landuse", "grass"),), (("building", "yes"),))


def test_a_query_token_no_matched_row_carries_is_named():
    # A junk token dilutes the token-bag ratio and never disqualifies, so a glued
    # spelling still binds.
    from maps.query.text_match import unmatched_query_tokens

    rows = _rows("Ford Focus car", "Vauxhall car", "car amenity parking")
    assert unmatched_query_tokens("car withcolor", rows) == ["withcolor"]
    # the remaining glued spellings
    assert unmatched_query_tokens("house onBeche Road", _rows("Beche Road house")) == ["onbeche"]
    # a clean query, and an attribute the rows DO carry, stay silent
    assert unmatched_query_tokens("car", rows) == []
    assert unmatched_query_tokens("car parking", rows) == []
    # short tokens are never named (a/of/the carry no narrowing)
    assert unmatched_query_tokens("a car", rows) == []
    # A PLURAL of a word the rows plainly carry is NOT a miss.
    assert unmatched_query_tokens("cars", rows) == []
    assert unmatched_query_tokens("houses", _rows("terraced house")) == []
    assert unmatched_query_tokens("buildings", _rows("building yes")) == []
    # ONE word per prefix family: 'lots'/'lot' is one problem, not two
    assert unmatched_query_tokens("parking lots", _rows("amenity parking")) == ["lot"]
    # and every real glue catch survives the loosening
    assert unmatched_query_tokens("S andG Cycles", _rows("S G Cycles")) == ["andg"]
    assert unmatched_query_tokens("house onCastle Street", _rows("Castle Street house")) == [
        "oncastle"
    ]


def test_a_palette_word_the_rows_were_perceived_as_is_not_flagged():
    from maps.query.text_match import COLORS, _colors_read, unmatched_query_tokens as U

    coloured = _rows("car", "car", colour="gray 0.55 / black 0.30")

    # The read is the closed PALETTE, word by word: a shade prefix and an
    # uncertainty mark fall out, and an off-palette synonym is not
    # vocabulary the note reasons over (a colour word the compiler cannot
    # emit is a NAME word).
    assert _colors_read(_rows("car", colour="dk blue/black")) == {"blue", "black"}
    assert _colors_read(_rows("car", colour="gray?")) == {"gray"}
    assert _colors_read(_rows("car", colour="lavender 0.60")) == set()
    assert _colors_read(_rows("car")) == set()

    # THREE DIRECTIONS over the WHOLE enumerated palette -- the vacuity
    # floor. A guard that suppressed nothing fails the first count; one
    # that suppressed on the mere PRESENCE of colour fails the third; one
    # that suppressed everything fails the second and the third.
    assert len(COLORS) == 15, sorted(COLORS)
    _other = {w: ("purple" if w != "purple" else "green") for w in COLORS}
    read = [w for w in COLORS if U(f"{w} car", _rows("car", colour=f"{w} 0.60")) == []]
    colourless = [w for w in COLORS if U(f"{w} car", _rows("car")) == [w]]
    other = [w for w in COLORS if U(f"{w} car", _rows("car", colour=f"{_other[w]} 0.60")) == [w]]
    for got, why in ((read, "read"), (colourless, "colourless"), (other, "read as another colour")):
        assert sorted(got) == sorted(COLORS), (why, sorted(set(COLORS) - set(got)))

    # A colour the rows were NOT perceived as is the note's most useful
    # claim, and it stays: true in the text channel AND in the colour one.
    assert U("purple car", _rows("car", "car", colour="red 0.80")) == ["purple"]
    # Every component of the mixture counts, shade prefix and uncertainty
    # mark included -- that is what the model is shown.
    assert U("black car", coloured) == []
    assert U("blue car", _rows("car", colour="dk blue/black")) == []
    assert U("gray car", _rows("car", colour="gray?")) == []
    # The prior map is the TRUE case the note exists for: a prior row
    # carries no perceived colour at all (the legend says so), so the
    # word really did narrow nothing.
    assert U("brown building", _rows("Green Building building yes")) == ["brown"]
    # An UNCERTAIN/absent read is not a colour read either.
    assert U("gray car", _rows("car", colour="?")) == ["gray"]
    assert U("gray car", _rows("car", colour="")) == ["gray"]
    # A find is unscoped over priors AND detections, so a mixed set is the
    # common case: one red detection does not vouch for 'gray'.
    assert U("gray car", _rows("car", "car") + _rows("car", colour="red 0.60")) == ["gray"]

    # Suppression is EXACT-WORD, never the prefix relation is_carried
    # uses: real glue is untouched, and a compound that merely STARTS
    # with a palette word is still glue.
    assert U("car withcolor", coloured) == ["withcolor"]
    assert U("house onCastle Street", _rows("Castle Street house", colour="gray 0.5")) == [
        "oncastle"
    ]
    assert U("S andG Cycles", _rows("S G Cycles", colour="red 0.6")) == ["andg"]
    assert U("greenhouse car", _rows("car", colour="green 0.6")) == ["greenhouse"]
    # A PLURAL of the palette word itself is one word, not an escape hatch
    # (the noun's plural is the separate is_carried case).
    assert U("grays car", coloured) == []
    assert U("gray cars", coloured) == []
    assert U("Green Street", _rows("Green Street highway residential")) == []
    assert U("Green Street", _rows("Castle Street highway", colour="gray 0.55")) == ["green"]


def test_find_query_returns_the_match_set_best_first():
    # A block of streets, so the head noun is COMMON and the proper name is RARE -- the
    # shape of every real map, and the one the fuzz band's IDF weighting reads. (Three
    # rows with `earl` on two of them inverted that, and made a proper name the LESS
    # discriminating half of the query.)
    anchors = [
        {"fid": "b_a", "name": "Earl Street", "xy": [0.0, 0.0], "class": "building"},
        {"fid": "b_b", "name": "Earl Court", "xy": [10.0, 0.0], "class": "building"},
        {"fid": "b_c", "name": "King Road", "xy": [20.0, 0.0], "class": "building"},
        {"fid": "b_d", "name": "Main Street", "xy": [30.0, 0.0], "class": "building"},
        {"fid": "b_e", "name": "Oak Street", "xy": [40.0, 0.0], "class": "building"},
        {"fid": "b_f", "name": "Pine Street", "xy": [50.0, 0.0], "class": "building"},
        {"fid": "b_g", "name": "Elm Street", "xy": [60.0, 0.0], "class": "building"},
    ]
    out = run_program(
        [],
        [dict(a) for a in anchors],
        {
            "ops": [
                {"ref": "r0", "op": "find", "args": {"query": "Earl Street"}},
            ]
        },
    )
    rows = out["results"]["r0"]
    # The result IS the match set (len = match count), best-first, ordered;
    # the sub-threshold row stays out (a query-less find browses it).
    assert rows.kind == "broad", rows.kind
    assert getattr(rows, "ordered", False), "a query-matched find is ordered"
    # b_b is "Earl Court" -- it shares the word `earl` with an exactly-named
    # "Earl Street" and is a DIFFERENT street, so once the query has named
    # something the band leaves the match set.
    assert {r["fid"] for r in rows} == {"b_a"}, rows
    assert all(isinstance(r.get("name_match"), (int, float)) for r in rows), rows
    scores = [r["name_match"] for r in rows]
    assert scores == sorted(scores, reverse=True), scores  # best-first
    assert rows[0]["fid"] == "b_a", rows  # the exact name match heads the set


def test_find_query_no_match_returns_empty_miss_affordance():
    out = run_program(
        [],
        [{"fid": "b_a", "name": "Earl Street", "xy": [9.0, 9.0], "class": "building"}],
        {
            "ops": [
                {"ref": "r0", "op": "find", "args": {"query": "zxqv wqzzk"}},
            ]
        },
    )
    rows = out["results"]["r0"]
    assert rows.kind == "miss", rows.kind
    assert len(rows) == 0, rows


def test_a_find_refusal_advertises_only_the_args_find_reads():
    with pytest.raises(ProgramError) as e:
        run_program(
            [],
            [{"fid": "a0", "xy": [0.0, 0.0], "name": "x"}],
            {"ops": [{"ref": "r0", "op": "find", "args": {"query": "a;b"}}]},
        )
    assert "color=" not in str(e.value) and "provenance" not in str(e.value)
