from __future__ import annotations

import re

from maps.search.analyze_scene import text_match as T
from maps.search.analyze_scene.geom_program import (
    _BROAD_DUMP_SOFT_ROWS,
    _BROAD_HEAD_ROWS,
    compact_results,
    run_program,
)


def _prior(fid, x, y, name=None, tags=None):
    return {"fid": fid, "name": name, "xy": [float(x), float(y)],
            "centerline": None, "tags": tags or {}, "color": "",
            "search_text": " ".join(
                [name or ""] + [f"{str(k).replace('_', ' ')} "
                                f"{str(v).replace('_', ' ')}"
                                for k, v in (tags or {}).items()]).lower()}


def _grass_world(n_pad=0):
    rows = [
        _prior("prior_1", 0, 0, tags={"landuse": "grass"}),
        _prior("prior_2", 10, 0, tags={"landuse": "grass"}),
        _prior("prior_3", 20, 0, tags={"landuse": "meadow"}),
        _prior("prior_4", 30, 0, tags={"building": "yes"}),
    ]
    for i in range(n_pad):
        rows.append(_prior(f"prior_p{i}", 40 + i, 0,
                           tags={"building": "yes"}))
    return rows


# ---------------------------------------------------------------------------
# B1: key=value + OR consume the taught vocabulary; truthful empty hint
# ---------------------------------------------------------------------------

def test_key_value_query_retrieves_the_tagged_rows():
    out = run_program([], _grass_world(), {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"query": "landuse=grass"}},
    ]})
    env = compact_results(out["results"])["r0"]
    assert env["len"] == 3  # the MATCH SET: len IS the match count
    assert env["fids"][:2] == ["prior_1", "prior_2"]  # exact k=v tier first
    rows_by_fid = {r["fid"]: r for r in out["results"]["r0"]}
    assert rows_by_fid["prior_1"]["name_match"] >= 0.8
    # the sub-threshold row stays OUT of the rN; the note states the tail
    assert "prior_4" not in rows_by_fid
    assert "3 of 4 rows matched" in env["hint"]
    assert "1 sub-threshold row" in env["hint"]
    assert "NOT in this rN" in env["hint"]
    assert f"match {max(rows_by_fid[f]['name_match'] for f in rows_by_fid)}" in env["hint"]


def test_or_query_is_max_over_clauses():
    # 'park OR natural OR garden OR lawn OR grass OR hedge' diluted the
    # coverage ratio to 0.143 (served empty) before; the canonical form is
    # max over the OR-clauses.
    out = run_program([], _grass_world(), {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"query": "park OR natural OR garden OR lawn OR grass "
                           "OR hedge"}},
    ]})
    rows = out["results"]["r0"]
    best = max(r["name_match"] for r in rows)
    assert best >= T.FUZZY_THRESHOLD
    top = max(rows, key=lambda r: r["name_match"])
    assert top["tags"] == {"landuse": "grass"}


def test_query_miss_serves_truthful_condition_not_world_claim():
    # A real miss states the scoring condition — never "nothing met its
    # condition" as a claim about the world.
    out = run_program([], _grass_world(), {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"query": "zzqx"}},
    ]})
    env = compact_results(out["results"])["r0"]
    assert env["len"] == 0
    assert f"no row scored >= {T.FUZZY_THRESHOLD}" in env["hint"]
    assert "'zzqx'" in env["hint"]
    assert "nothing met its condition" not in env["hint"]


def test_miss_census_is_query_conditioned_first():
    # The long-tail tag the global top-12 hides must surface when the query
    # shares its tokens (landuse=grass sat at 17 below the cut in 8267).
    rows = _grass_world(n_pad=40)  # 41 building=yes rows dominate the top
    out = run_program([], rows, {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"query": "grassy terrain zone area"}},  # a real miss
    ]})
    env = compact_results(out["results"])["r0"]
    hint = env["hint"]
    assert "tags sharing tokens with your query" in hint
    qpart = hint.split("tags sharing tokens with your query:")[1]
    qpart = qpart.split("top tags overall:")[0]
    assert "landuse=grass(2)" in qpart
    assert "building=yes" not in qpart  # query-conditioned, not the global top
    assert "top tags overall:" in hint  # the global top still follows


# ---------------------------------------------------------------------------
# B2: tiered lexical precision
# ---------------------------------------------------------------------------

def _downing_world():
    return [
        _prior("prior_way", 0, 0, name="Downing Street",
               tags={"highway": "unclassified"}),
        _prior("prior_addr1", 5, 0,
               tags={"addr:street": "Downing Place", "building": "yes"}),
        _prior("prior_addr2", 6, 0,
               tags={"addr:street": "Downing Place", "building": "yes"}),
        _prior("prior_naptan", 7, 0, name="Downing College",
               tags={"naptan:street": "Downing Street",
                     "highway": "bus_stop"}),
        # Spells BOTH query words -- `street` off the tag KEY, `downing` off
        # a value that is NOT the phrase -- so it saturates the token bag
        # without any exact tier being available to it.
        _prior("prior_bag", 8, 0,
               tags={"addr_street": "Downing Place",
                     "highway": "residential"}),
    ]


def test_exact_name_outranks_token_bag_saturation():
    # Exact name > everything else.
    out = run_program([], _downing_world(), {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"query": "Downing Street"}},
    ]})
    rows = out["results"]["r0"]
    assert rows[0]["fid"] == "prior_way"
    assert rows[0]["name_match"] == 1.0
    by_fid = {r["fid"]: r["name_match"] for r in rows}
    # token-bag saturation is capped below every exact tier: prior_bag covers
    # the WHOLE query and still cannot leave the fuzz band -- and once the
    # query has NAMED something, the band is a different feature and leaves
    # the match set entirely. The cap is still what puts it there, so it is
    # asserted on the SCORER, which is where that fact lives.
    from maps.search.analyze_scene import text_match as _T
    _bag = next(n for n in _downing_world() if n["fid"] == "prior_bag")
    assert _T.find_score(_T.fuzzy_prepare_query("Downing Street"),
                         _bag["search_text"], _bag.get("name"),
                         _bag.get("tags"), idf={}) <= 0.75
    assert "prior_bag" not in by_fid, by_fid
    # prior_naptan carries the phrase as a whole tag VALUE, which is the tag
    # band -- above the bag, below the exact name. A NAME that hits answers
    # alone, so the band goes with the bag: it is a different KIND of row
    # (the thing AT the street, not the street), and it is reachable on its
    # own name. The scorer still ranks it where it always did.
    from maps.search.analyze_scene import text_match as _T2
    _nap = next(n for n in _downing_world() if n["fid"] == "prior_naptan")
    assert 0.80 <= _T2.find_score(
        _T2.fuzzy_prepare_query("Downing Street"), _nap["search_text"],
        _nap.get("name"), _nap.get("tags"), idf={}) < 1.0
    assert "prior_naptan" not in by_fid, by_fid
    assert "prior_addr1" not in by_fid, by_fid
    assert "prior_addr2" not in by_fid, by_fid


def test_exact_kv_outranks_token_bag_variant():
    # 'amenity parking' scored a bicycle_parking node 1.0 ABOVE
    # amenity=parking (0.767) via bag saturation; the k=v form is exact.
    world = [
        _prior("prior_park", 0, 0, tags={"amenity": "parking"}),
        _prior("prior_bike", 5, 0, tags={"amenity": "bicycle_parking"}),
    ]
    out = run_program([], world, {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"query": "amenity=parking"}},
    ]})
    rows = out["results"]["r0"]
    assert rows[0]["fid"] == "prior_park"
    assert rows[0]["name_match"] >= 0.8
    assert rows[1]["name_match"] <= 0.75


def test_name_phrase_tier_beats_fuzz_but_not_exact():
    # 'The Museum of Cambridge' (name-phrase tier) must outrank a fixme-tag
    # token-bag row.
    world = [
        _prior("prior_museum", 0, 0, name="The Museum of Cambridge",
               tags={"tourism": "museum"}),
        _prior("prior_fixme", 5, 0,
               tags={"fixme": "museum of cambridge probably"}),
    ]
    out = run_program([], world, {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"query": "Museum of Cambridge"}},
    ]})
    rows = out["results"]["r0"]
    assert rows[0]["fid"] == "prior_museum"
    assert rows[0]["name_match"] >= 0.9
    # The fixme row (phrase inside a tag VALUE, not a name) stays BELOW the
    # exact-named row and below the name-phrase tier -- and because the query
    # NAMED something, the band it sits in is a different feature and leaves
    # the match set. The ranking fact is asserted on the scorer, which is
    # where it lives; the set fact is asserted on the result.
    assert [r["fid"] for r in rows] == ["prior_museum"], rows
    _fixme = next(n for n in world if n["fid"] == "prior_fixme")
    _s = T.find_score(T.fuzzy_prepare_query("Museum of Cambridge"),
                      _fixme["search_text"], _fixme.get("name"),
                      _fixme.get("tags"), idf={})
    assert _s < 0.9 and _s < rows[0]["name_match"], _s
    # a two-word query with one unmatched word still clears the threshold on the other
    q = T.fuzzy_prepare_query("Con Exchange")
    assert T.find_score(
        q, "corn exchange", "Corn Exchange", {},
        idf=T.token_idf([{"search_text": "corn exchange",
                          "name": "Corn Exchange"}])) >= T.FUZZY_THRESHOLD


# ---------------------------------------------------------------------------
# B3: the ONE broad-result envelope law (census + head), floods capped
# ---------------------------------------------------------------------------

def _wide_world(n=120):
    return [_prior(f"prior_{i:03d}", i, 0, tags={"building": "yes"})
            for i in range(n)]


def test_broad_score_by_serves_census_plus_head_not_full_dump():
    # The 10099-class flood: a broad score_by served the FULL ranked fid
    # array + full grid (113-148KB). The envelope is census + head; the
    # full set stays live in the rN.
    n = 120
    assert n > _BROAD_DUMP_SOFT_ROWS  # past the knee, so the law applies
    out = run_program([], _wide_world(n), {"ops": [
        {"ref": "r0", "op": "find", "args": {}},
        {"ref": "r1", "op": "compare", "args": {"items": "r0",
                                                "refs": ["prior_000"]}},
        {"ref": "r2", "op": "score_by",
         "args": {"items": "r1",
                  "emphasis": [{"key": "distance to prior_000", "w": -1}]}},
    ]})
    env = compact_results(out["results"])["r2"]
    assert env["len"] == n
    assert len(env["fids"]) == _BROAD_HEAD_ROWS
    # ranked head IS the ranking's top: rank-0 nearest prior_000
    assert env["fids"][0] == "prior_000"
    assert len(out["results"]["r2"]) == n


def test_broad_filter_items_capped_same_law():
    # A filter over an ORDERED input preserves the order, so it renders —
    # under the same census+head law past the knee.
    n = 120
    out = run_program([], _wide_world(n), {"ops": [
        {"ref": "r0", "op": "find", "args": {}},
        {"ref": "rc", "op": "compare",
         "args": {"items": "r0", "refs": ["prior_000"]}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "rc",
                  "emphasis": [{"key": "distance to prior_000", "w": -1}]}},
        {"ref": "r2", "op": "filter_items",
         "args": {"items": "r1", "key": "distance to prior_000", "op": "le",
                  "value": 500}},
    ]})
    env = compact_results(out["results"])["r2"]
    assert env["len"] == n
    assert len(env["fids"]) == _BROAD_HEAD_ROWS
    assert env["rows"].count("\n") <= _BROAD_HEAD_ROWS + 4
    assert len(out["results"]["r2"]) == n  # raw rows all live


def test_windowed_ordered_result_serves_the_continuation_rows():
    # A continuation is filter_items over rN[12:24]: a window of an ordered
    # result stays ordered and serves the ranking's own next rows.
    n = 120
    out = run_program([], _wide_world(n), {"ops": [
        {"ref": "r0", "op": "find", "args": {}},
        {"ref": "rc", "op": "compare",
         "args": {"items": "r0", "refs": ["prior_000"]}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "rc",
                  "emphasis": [{"key": "distance to prior_000", "w": -1}]}},
        {"ref": "r2", "op": "filter_items",
         "args": {"items": "r1[12:24]", "key": "xy", "op": "truthy"}},
    ]})
    win = out["results"]["r2"]
    assert getattr(win, "ordered", False), "the window lost the order bit"
    env = compact_results(out["results"])["r2"]
    assert env["len"] == 12
    assert "rows" in env and env["fids"], "a continuation is rows, not census"
    ranked = [r["fid"] for r in out["results"]["r1"]]
    assert env["fids"] == ranked[12:24], "the window is the ranking's own 12:24"


def test_unordered_filter_serves_census_not_a_row_head():
    # A filter over a query-less BROWSE has no real ordering: census + len
    # only, no row head (a fid-lexicographic head reads as a candidate
    # list). The full set stays live in the rN.
    n = 120
    out = run_program([], _wide_world(n), {"ops": [
        {"ref": "r0", "op": "find", "args": {}},
        {"ref": "rc", "op": "compare",
         "args": {"items": "r0", "refs": ["prior_000"]}},
        {"ref": "r1", "op": "filter_items",
         "args": {"items": "rc", "key": "distance to prior_000", "op": "le",
                  "value": 500}},
    ]})
    env = compact_results(out["results"])["r1"]
    assert env["len"] == n
    assert "rows" not in env and "fids" not in env
    assert "census" in env or "census_ref" in env
    assert "cut_stats" in env  # the cut's own facts still ride
    assert len(out["results"]["r1"]) == n  # raw rows all live


def test_compare_grid_joins_identity_and_relational_columns():
    # The relational block IS the attribute block: one joined table, identity +
    # attached columns.
    out = run_program([], _wide_world(60), {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"query": "building"}},
        {"ref": "r1", "op": "compare", "args": {"items": "r0",
                                                "refs": ["prior_000"]}},
        {"ref": "r2", "op": "score_by", "args": {"items": "r1", "emphasis": [
            {"key": "distance to prior_000", "w": -1}]}},
    ]})
    envs = compact_results(out["results"], answer_ref="r2")
    r2_header = envs["r2"]["rows"].splitlines()[0]
    assert "tags" in r2_header and "xy" in r2_header
    assert "distance to prior_000" in r2_header
    # raw compare rows keep every column (render shape only, never a cut)
    assert all("tags" in r for r in out["results"]["r1"]
               if isinstance(r, dict))


def test_identical_census_served_once_per_response():
    out = run_program([], _wide_world(120), {"ops": [
        {"ref": "r0", "op": "find", "args": {}},
        {"ref": "rc", "op": "compare",
         "args": {"items": "r0", "refs": ["prior_000"]}},
        {"ref": "r1", "op": "filter_items",
         "args": {"items": "rc", "key": "distance to prior_000", "op": "le",
                  "value": 500}},
    ]})
    envs = compact_results(out["results"])
    n_census = sum(1 for e in envs.values()
                   if isinstance(e, dict) and "census" in e)
    n_refs = sum(1 for e in envs.values()
                 if isinstance(e, dict) and "census_ref" in e)
    assert n_census == 1
    assert n_refs == 1
    assert "r0" in [e.get("census_ref", "").split("'")[0].split()[3]
                    for e in envs.values() if isinstance(e, dict)
                    and "census_ref" in e][0]


def test_render_budget_elides_column_tail_never_rows():
    # ONE render budget: a result's rendered grid fits _RENDER_CHAR_BUDGET.
    # Program-named columns render first; the elided tail is counted in the
    # hint. Render shape only — the raw rows keep every column (the F485
    # bound case: one uncapped compare rendered ~97k chars).
    from maps.search.analyze_scene.geom_program import _RENDER_CHAR_BUDGET
    n_refs = 40
    world = _wide_world(60)
    refs = [f"prior_{i:03d}" for i in range(1, n_refs + 1)]
    out = run_program([], world, {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"query": "building"}},
        {"ref": "r1", "op": "compare", "args": {"items": "r0", "refs": refs}},
        {"ref": "r2", "op": "score_by", "args": {"items": "r1", "emphasis": [
            {"key": "distance to prior_001", "w": -1}]}},
    ]})
    env = compact_results(out["results"], answer_ref="r2")["r2"]
    assert len(env["rows"]) <= _RENDER_CHAR_BUDGET
    # a program-named column survives the cut; the raw rows keep every one
    assert "distance to prior_001" in env["rows"].splitlines()[0]
    raw = out["results"]["r1"]
    assert all("distance to prior_040" in r for r in raw if isinstance(r, dict))
    # the elided tail is COUNTED in the hint — the render says what it dropped
    assert "elided from this render" in env.get("hint", ""), env.get("hint")


def test_budget_walk_skips_an_oversized_column_not_the_columns_behind_it():
    # One oversized column must not wall off the cheaper columns after it
    # in priority order: the walk skips it and keeps walking.
    from maps.search.analyze_scene.geom_program import _budget_columns
    rows = [{"fid": f"f{i}", "a": 1, "big": "x" * 4000, "b": 2}
            for i in range(9)]
    cols, n_cut = _budget_columns(rows, ["a", "big", "b"], ["a", "big", "b"])
    assert cols == ["a", "b"], cols
    assert n_cut == 1


def test_full_set_stays_chainable_past_the_head():
    # A row past the rendered head is reachable via rN slicing/chaining —
    # the cap is response shape, never a candidate cap.
    out = run_program([], _wide_world(120), {"ops": [
        {"ref": "r0", "op": "find", "args": {}},
        {"ref": "rc", "op": "compare",
         "args": {"items": "r0[100:110]", "refs": ["prior_119"]}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "rc",
                  "emphasis": [{"key": "distance to prior_119", "w": -1}]}},
    ]})
    env = compact_results(out["results"])["r1"]
    assert env["len"] == 10
    assert env["fids"][0] == "prior_109"


# ---------------------------------------------------------------------------
# B4 served strings
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dedup: an identical re-issue gets a one-line REFERENCE, never a payload
# ---------------------------------------------------------------------------


def test_dedup_note_is_a_reference_not_a_payload():
    # The hit serves one line naming the original turn; the rN handles stay live in
    # carry_results, so nothing is locked away — a new program slices them.
    from maps.search.parse_compile.reasoner_loop import _DEDUP_NOTE
    note = _DEDUP_NOTE.format(turn=3, no_look_for=", no look_for",
                              look_for=", look_for")
    assert "no new information since turn 3" in note
    assert "not re-rendered" in note
    assert "no flight, no look_for" in note and "(fly, look_for)" in note
    assert "{args}" not in _DEDUP_NOTE and "arrived" not in note


def test_a_calls_own_steps_ride_its_answer_unprefixed_and_once():
    from maps.search.analyze_scene.geom_program import Result, compact_results

    hint = compact_results({"r3": Result(kind="miss"), "r4": Result(kind="miss")},
                           answer_ref="r4")["r4"]["hint"]
    assert "r3" not in hint, hint
    assert hint.count("this op returned NO items") == 1, hint


def test_a_map_names_the_set_it_badges_as_the_model_does():
    from maps.search.tests.fakes import map_mission

    res, _vision = map_mission(['a = find("building")', "show_map(around=prior_7)",
                                'find("building")\nshow_map(around=prior_7)'])
    assert res[1]["map"]["badged_from"] == "a", res[1]
    assert res[2]["map"]["badged_from"] == "find('building')", res[2]


def test_a_folded_result_rides_under_a_name_the_model_can_read():
    from maps.search.tests.fakes import drive, run, tool_results

    _reply, fr, _f, _d = drive([[run('n = len(find("zeppelin"))\nb = find("car")\n')],
                                "done"])
    hint = tool_results(fr)[0]["result"]["results"]["b"]["hint"]
    assert "find('zeppelin'): find matched nothing" in hint, hint
    assert not re.search(r"(?m)^r\d+: ", hint), hint


def test_a_keyword_only_find_serves_its_hint_once_and_never_as_find_of_nothing():
    from maps.search.tests.fakes import drive, run, tool_results

    _reply, fr, _f, _d = drive([[run('find(near="prior_1", dwithin=60)\n')], "done"])
    (env,) = tool_results(fr)[0]["result"]["results"].values()
    lines = env["hint"].splitlines()
    assert len(lines) == len(set(lines)), lines
    assert "find('')" not in env["hint"], env["hint"]


def test_a_folded_hint_keeps_the_lines_the_answer_does_not_say():
    from maps.search.tests.fakes import drive, run, tool_results

    _reply, fr, _f, _d = drive([[run(
        'a = find("zeppelin").filter(near="prior_1", dwithin=10)\n'
        'b = find("car").filter(near="prior_1", dwithin=1)\n')], "done"])
    hint = tool_results(fr)[0]["result"]["results"]["b"]["hint"]
    assert "\na: this op returned NO items: nothing met its condition.\n" in hint, hint


def test_a_looked_slice_rides_under_the_models_name_for_it():
    from maps.search.tests.fakes import drive, run, tool_results

    _reply, fr, _f, _d = drive([[run('b = find("depot")\na = find("car")\n')],
                                [run('look(a[0:2])\n')], "done"])
    assert list(tool_results(fr)[1]["result"]["results"]) == ["a[0:2]"]


def test_refusals_over_an_unnamed_result_quote_the_models_own_text():
    from maps.search.tests.fakes import drive, run, tool_results

    _reply, fr, _f, _d = drive([[run('look(find("zeppelin"))\n')],
                                [run('more_of(find("car").order(nearest="prior_1"))\n')],
                                "done"])
    look_err, page_err = (t["error"] for t in tool_results(fr)[:2])
    assert look_err.startswith("find('zeppelin') holds no rows"), look_err
    assert "row(s) of find('car').order(nearest='prior_1') are" in page_err, page_err
    assert not re.search(r"\br\d+\b", look_err + page_err), (look_err, page_err)


def test_an_emptied_sibling_keeps_its_own_status_line():
    from maps.search.tests.fakes import drive, run, tool_results

    _reply, fr, _f, _d = drive([[run(
        'a = find("car").filter(near="prior_1", dwithin=1)\n'
        'b = find("depot").filter(near="prior_1", dwithin=1)\n')], "done"])
    hint = tool_results(fr)[0]["result"]["results"]["b"]["hint"]
    assert "a: this op returned NO items" in hint, hint
    lines = hint.splitlines()
    assert len(lines) == len(set(lines)), lines


def test_refusals_over_a_keyword_only_find_quote_the_models_own_text():
    from maps.search.tests.fakes import drive, run, tool_results

    _reply, fr, _f, _d = drive([[run('more_of(find(nearest="prior_1"))\n')],
                                [run('find(nearest="prior_1")[999]\n')], "done"])
    page_err, index_err = (t["error"] for t in tool_results(fr)[:2])
    assert "find(nearest='prior_1')" in page_err, page_err
    assert "find(nearest='prior_1')[999] did not bind" in index_err, index_err


def test_a_cut_steps_hints_do_not_ride_the_answer():
    # find -> compare -> filter -> sort: the find's count line and head name
    # the 40 cars it matched, not the 6 the answer holds.
    from maps.search.tests.fakes import drive, run, tool_results

    cands = [{"fid": "prior_7", "xy": [0.0, 0.0], "class": "depot", "search_text": "depot"},
             {"fid": "prior_1", "xy": [100.0, 0.0], "class": "gate", "search_text": "gate"}]
    cands += [{"fid": f"car_{i}", "xy": [float(i * 3), 1.0], "class": "car", "search_text": "car"}
              for i in range(40)]
    _r, fr, _f, _d = drive([[run('nc = find("car", nearest=prior_1, dwithin=10)')], "done"],
                           cands=cands)
    env = tool_results(fr)[0]["result"]["results"]["nc"]
    hint = env.get("hint", "")
    assert set(re.findall(r"car_\d+", hint)) <= set(env["fids"]), hint
    assert "rows matched" not in hint, hint
