"""Every set is named to the model by its own spelling, never by an executor
ref it did not bind: one rule (MissionPython.spelling) for every served
label."""
from __future__ import annotations

import json
import re

import pytest

from maps.search.parse_compile.perception_director import PerceptionDirector
from maps.search.tests.fakes import CANDS, drive, run, tool_results

_CANDS = CANDS + [
    {"fid": f"car_{i}", "xy": [5.0 * i, 7.0], "class": "car",
     "search_text": "car"} for i in range(3, 9)]

_FILTERED = 'find("car").filter(near="prior_7", dwithin=5000)'

# Slices, tier views, filters, bare lines, show_map, look_for(at=), the
# go_to/explore refusals, print and the slots, each over a bound, an unbound
# and a nameless set.
SWEEP = [
    'cars = find("car")\nprint(cars[0:2])\n',
    'print(find("car")[0:3])\n',
    f'print({_FILTERED})\n',
    'cars = find("car")\ncars[:2].exact\n',
    'find("car")[0:2].exact\n',
    'cars = find("car")\ncars.exact\n',
    'cars = find("car")\ncars.filter(near=cars[0], dwithin=50).exact\n',
    'cars = find("car")\nnear = cars[:2].filter(near="prior_7", dwithin=500)\n',
    'find("car")[0:2].filter(near="prior_7", dwithin=5000)\n',
    'find("depot", near=find("car")[0:3])\n',
    'find("car").order(nearest=find("depot")[0:1])\n',
    'find("car").filter(near=find("depot").exact, dwithin=100)\n',
    f'x = find("depot").order(nearest={_FILTERED})\nprint(x[0])\n',
    'find("car")[1:3]\n',
    'cars = find("car")\ncars[1:3]\n',
    f'{_FILTERED}\n',
    'show_map(around=find("car")[0:2])\n',
    'cars = find("car")\nshow_map(around=cars[0:2])\n',
    f'show_map(around={_FILTERED})\n',
    'go_to(find("car")[0:2])\n',
    'cars = find("car")\ngo_to(cars[0:2])\n',
    f'go_to({_FILTERED})\n',
    'explore(find("car")[5:5])\n',
    f'explore({_FILTERED}.filter(within="prior_7"))\n',
    'explore(find("car")[0:2])\n',
    f'explore({_FILTERED})\n',
    f'look({_FILTERED}[0:2])\n',
    'ends_of(find("qqqzzz"))\n',
    f'ends_of({_FILTERED}.filter(within="prior_7"))\n',
    f'intersection({_FILTERED}, "prior_7")\n',
    'post = find("car")[0]\npost\n',
    f'{_FILTERED}.filter(within="prior_7")\nexplore({_FILTERED})\n',
    'cars = find("car")\nlook_for("car", at=cars)\n',
    f'look_for("car", at={_FILTERED})\n',
    'explore([find("car")[0], find("qqqzzz")])\n',
    f'explore([{_FILTERED}.filter(within="qqqzzz")])\n',
]


def _served(code: str) -> tuple[str, dict]:
    _, fr, _, _ = drive([[run(code)], "done"], cands=_CANDS)
    out = tool_results(fr)[0]
    return json.dumps(out), out


@pytest.mark.parametrize("code", SWEEP)
def test_no_served_text_names_an_rn_the_model_did_not_bind(code):
    text, out = _served(code)
    own = set(re.findall(r"\br\d+\b", code)) | set(out.get("handles") or ())
    assert not set(re.findall(r"\br(?:\d+|N)\b", text)) - own, text


def _keys(out: dict) -> list:
    return list(((out.get("result") or {}).get("results") or {}))


def test_a_slice_prints_under_its_spelling():
    _, out = _served('cars = find("car")\nprint(cars[0:2])\n')
    assert out["printed"].startswith("<cars[0:2]: 2 row(s)"), out
    _, out = _served(f"print({_FILTERED})\n")
    assert out["printed"].startswith("<(unnamed): 8 row(s)"), out


def test_a_tier_view_is_shown_once_under_its_spelling():
    _, out = _served('cars = find("car")\ncars.exact\n')
    assert _keys(out) == ["cars.exact"], out
    _, out = _served('cars = find("car")\ncars[:2].exact\n')
    assert "cars[0:2].exact" in _keys(out), out


def test_a_nameless_set_is_refused_with_bind_it_first():
    _, out = _served(f"go_to({_FILTERED})\n")
    assert "Bind it first (s = ...)" in out["error"], out
    assert "go_to(s[0])" in out["error"], out


def test_a_slot_and_an_explore_say_the_slice_as_spelled():
    _, out = _served('explore(find("car")[0:2])\n')
    assert "pass over find('car')[0:2]" in json.dumps(out), out


_DEPOT = 'find("depot").filter(near="prior_7", dwithin=50)'
_GATE = 'find("gate").filter(near="prior_1", dwithin=50)'


@pytest.mark.parametrize("call", [
    "find('car', north_of={a}, west_of={b})",
    "find('car', nearest=[{a}, {b}])",
    "find('car').order(nearest={a}).order(nearest={b})",
])
def test_two_nameless_anchors_stay_two_places(call):
    # Each nameless set gets its own label, so its columns and cuts never
    # merge with another's: the block serves what binding them serves.
    _, bare = _served(call.format(a=_DEPOT, b=_GATE) + "\n")
    _, bound = _served(f"a = {_DEPOT}\nb = {_GATE}\n"
                       + call.format(a="a", b="b") + "\n")
    got, want = (list(o["result"]["results"].values())[-1] for o in (bare, bound))
    assert got["fids"] == want["fids"], (got, want)
    header = lambda e: re.sub(r"\b(unnamed_\d|a|b)\b", "X", e["rows"].split("\n")[0])
    assert header(got) == header(want), (got["rows"], want["rows"])


def test_an_unbound_find_is_keyed_by_its_call_unless_a_key_says_it():
    _, out = _served('post = find("car")[0]\npost\n')
    assert _keys(out)[0] == "find('car')", out
    _, out = _served('find("car")\n'
                     'post = find("car", nearest="prior_7")[0]\n')
    assert _keys(out) == ["find('car')", "(unnamed 1)"], out


@pytest.mark.parametrize("code", [
    f"explore({_FILTERED}[0:2])\n",
])
def test_a_nameless_set_has_one_number_in_keys_and_acks(code):
    _, out = _served(code)
    keys = {tuple(v["fids"]): k for k, v in out["result"]["results"].items()}
    car = next(k for f, k in keys.items() if f and f[0].startswith("car_"))
    assert out["explored"]["explored"] == car, out


@pytest.mark.parametrize("code,said", [
    ('cars = find("car")\nlook_for("car", at=cars)\n', "Index it -- at=cars[0]"),
    (f'look_for("car", at={_FILTERED})\n', "Bind it first (s = ...)"),
])
def test_look_for_at_a_set_is_refused_and_queues_nothing(code, said):
    pd = PerceptionDirector()
    _, fr, _, _ = drive([[run(code)], "done"], cands=_CANDS,
                        boost_queue=pd.boost_queue)
    out = tool_results(fr)[0]
    assert said in out.get("error", ""), out
    assert "boost" not in out and not pd.boosts, out


@pytest.mark.parametrize("code", [
    'none = find("qqqzzz")\nexplore(none)\n',
    'none = find("qqqzzz")\ncars = find("car")\nexplore([cars[0], none])\n',
])
def test_an_empty_set_to_explore_is_refused_under_its_spelling(code):
    _, out = _served(code)
    assert "none holds 0 row(s) here; there is nothing to cover." \
        in out.get("error", ""), out


def test_stand_at_a_set_is_refused_under_its_spelling():
    _, out = _served('cars = find("car")\nstand(at=cars)\n')
    assert "at=cars[0]" in out["error"], out


def test_a_set_passed_as_a_predicate_or_key_gets_the_lambda_remedy():
    # A set is callable (calling one says so); as filter's or
    # order's first argument it still gets the remedy that names a lambda.
    _, out = _served('cars = find("car")\nroads = find("depot")\ncars.filter(roads)\n')
    assert "rows.filter(lambda r:" in out["error"], out
    _, out = _served('cars = find("car")\nroads = find("depot")\ncars.order(roads)\n')
    assert "rows.order(lambda r:" in out["error"], out


def test_a_move_name_the_block_binds_reads_as_its_variable():
    _, out = _served('explore = find("car")\nprint(len(explore))\n')
    assert "error" not in out, out


# Ten ways of one street: find serves them as a tied view.
_TIED = CANDS + [
    {"fid": f"prior_{900 + i}", "xy": [10.0 * i + 5, 0.0], "name": "Tied Way",
     "search_text": "tied way", "tags": {"highway": "residential"},
     "centerline": [[10.0 * i, 0.0], [10.0 * i + 10, 0.0]]}
    for i in range(10)]


def _first_error(code: str) -> str:
    _, fr, _, _ = drive([[run(code)], "done"], cands=_TIED)
    return next(str(o["error"]) for o in tool_results(fr) if "error" in o)


def test_the_fixture_way_is_a_tied_view():
    assert "tied view" in _first_error('w = find("Tied Way")\nw[0]')


@pytest.mark.parametrize("code", [
    'pools = find("zzqq pool")\nshow_map(around=pools)',
    'pools = find("zzqq pool")\nprint(pools.xy)',
    'pools = find("zzqq pool")\nprint(pools["xy"])',
    'pools = find("zzqq pool")\nstand(at=pools)',
    'w = find("Tied Way")\nshow_map(around=w)',
    'w = find("Tied Way")\nprint(w.xy)',
    'w = find("Tied Way")\ngo_to(w)',
    'w = find("Tied Way")\nlook_for("car", at=w)',
])
def test_a_set_remedy_never_names_an_index_that_refuses(code):
    err = _first_error(code)
    assert f"{code.split(' = ')[0]}[0]" not in err, err


def test_a_tied_views_remedy_runs():
    err = _first_error('w = find("Tied Way")\nprint(w.xy)')
    assert "w.order(nearest=<place>)[0].xy" in err, err
    _, fr, _, _ = drive([[run('w = find("Tied Way")\n'
                              'print(w.order(nearest="prior_7")[0].xy)')],
                         "done"], cands=_TIED)
    assert not any("error" in o for o in tool_results(fr))


def test_an_empty_set_centres_no_map_and_is_offered_no_set_rule():
    err = _first_error('pools = find("zzqq pool")\nshow_map(around=pools)')
    assert err.endswith("pools holds 0 row(s) here."), err


def test_a_set_read_for_a_key_no_row_carries_names_the_columns():
    err = _first_error('cars = find("car")\nprint(cars.census)')
    assert "cars[0].census" not in err and "its columns are" in err, err


@pytest.mark.parametrize("read", ["x.foo", "x.keys()", 'x.get("foo")'])
def test_a_read_off_an_empty_view_never_names_its_first_row(read):
    err = _first_error(f'r1 = find("Tied")\nx = r1.exact\n{read}')
    assert "0 row(s)" in err and "x[0]" not in err, err


def test_calling_a_set_names_the_call_it_was_bound_over():
    err = _first_error('b = find("car")\nintersection = b\nintersection("a", "b")')
    assert err.startswith("line 3: TypeError: intersection is a set"), err
    assert "del intersection gives the call back" in err, err


def test_an_empty_list_the_model_bound_is_named_by_its_name():
    err = _first_error('x = find("car").filter(lambda r: False)\nx.nearest')
    assert "bind it first" not in err and "x.order(nearest=...)" in err, err


@pytest.mark.parametrize("code, want", [
    ('x = find("car").filter(lambda r: False)\nx[0]',
     "line 2: x[0] did not bind: x holds 0 row(s). There is no row to index."),
    ('x = find("car").filter(lambda r: True)\nx[5]',
     "line 2: x[5] did not bind: x holds 2 row(s)."),
    ('go_to(find("car").filter(lambda r: False)[0])',
     "line 1: find('car').filter(lambda r: False)[0] did not bind: "
     "find('car').filter(lambda r: False) holds 0 row(s). There is no row to index."),
    ('x = find("car").filter(lambda r: True)\nx["fid"]',
     "line 2: x['fid'] did not bind: x is a set of rows, indexed by an int "
     "position; a column is read off one row: x[0]['fid']."),
    ('x = find("car").filter(lambda r: False)\nx["fid"]',
     "line 2: x['fid'] did not bind: x is a set of rows, indexed by an int position."),
])
def test_a_list_of_rows_indexed_past_its_rows_or_by_a_non_int_is_refused(code, want):
    assert _first_error(code) == want


def test_a_list_of_rows_indexes_and_slices_as_a_list():
    _, fr, _, _ = drive([[run('x = find("car").filter(lambda r: True)\n'
                              'print(x[-1].fid, len(x[0:1]))')], "done"], cands=_TIED)
    assert [o.get("error") for o in tool_results(fr)] == [None]


@pytest.mark.parametrize("slot", ["show_map(around=x)", 'look_for("car", at=x)', "stand(at=x)"])
def test_an_empty_list_of_rows_in_a_one_place_slot_holds_0_rows(slot):
    err = _first_error(f'x = find("car").filter(lambda r: False)\n{slot}')
    assert err.endswith("takes ONE place and that list holds 0 row(s)."), err


@pytest.mark.parametrize("code", ['cars = find("car")\nsorted([1, 2], key=cars)',
                                  'cars = find("car")\nmax([1, 2], key=cars)'])
def test_a_set_called_by_a_builtin_is_named_by_its_own_name(code):
    assert _first_error(code).startswith("line 2: TypeError: cars is a set of 2 row(s)")


def test_a_comprehension_variable_named_after_a_move_is_the_models_variable():
    _, fr, _, _ = drive([[run('xs = ["prior_7"]\nprint([go_to for go_to in xs])')], "done"],
                        cands=_TIED)
    assert [o.get("error") for o in tool_results(fr)] == [None]


@pytest.mark.parametrize("code", ['cars = find("car")\nprint(cars["census"])',
                                  'cars = find("car").filter(lambda r: True)\nprint(cars["census"])'])
def test_a_set_indexed_by_a_key_no_row_carries_names_the_columns(code):
    err = _first_error(code)
    assert "cars[0]['census']" not in err and "its columns are" in err, err


@pytest.mark.parametrize("held", ['find("car")', 'find("car").filter(lambda r: True)',
                                  'find("car")[0]'])
def test_an_unknown_order_keyword_is_refused_naming_orders_keywords(held):
    assert _first_error(f'cars = {held}\ncars.order(distance="prior_7")') == (
        "line 2: order takes key=/nearest=/color=/direction=/reverse=; "
        "distance= is none of them.")


@pytest.mark.parametrize("held, what", [
    ('find("car").filter(lambda r: True)', "a set of 2 row(s)"),
    ('find("car").filter(lambda r: False)', "a set of 0 row(s)"),
    ('find("car")[0]', "a row")])
def test_a_list_or_a_row_called_says_what_it_is(held, what):
    assert _first_error(f'intersection = {held}\nintersection("a", "b")') == (
        f"line 2: TypeError: intersection is {what}, not a call: binding "
        "intersection shadows the intersection() call; del intersection "
        "gives the call back.")
    assert _first_error(f'x = {held}\nsorted([1, 2], key=x)') == (
        f"line 2: TypeError: x is {what}, not a call.")


@pytest.mark.parametrize("read", ['x["name"]', 'x.name'])
def test_a_list_led_by_a_non_row_refuses_a_column_read_by_name(read):
    err = _first_error(f'x = ["a"] + find("car")\nprint({read})')
    assert err.startswith("line 2: ") and "row car_1 carries no 'name'" in err, err


def test_a_slice_of_a_list_of_rows_still_cuts_and_orders():
    _, fr, _, _ = drive([[run('x = find("car").filter(lambda r: True)[0:1]\n'
                              'print(len(x.order(lambda r: r.fid)))')], "done"], cands=_TIED)
    assert [o.get("printed") for o in tool_results(fr)] == ["1"]


def test_a_row_called_is_named_as_the_model_wrote_it():
    assert _first_error('cars = find("car")\ncars[0]()') == (
        "line 2: TypeError: cars[0] is a row, not a call.")


def test_a_comprehension_variable_named_after_a_move_ends_with_its_comprehension():
    err = _first_error('xs = ["prior_7"]\nz = [go_to for go_to in xs]\ny = go_to\n')
    assert "line 3: go_to is a move; call it once" in err, err


def test_a_list_holding_a_non_row_finds_its_mission_on_its_rows():
    err = _first_error('x = ["a"] + find("car")\nx.filter(near="prior_7", dwithin=5)')
    assert "_mp" not in err, err


@pytest.mark.parametrize("code, want", [
    ('cars.distance("a", "b")', "distance() takes 1 positional argument. distance keywords: none"),
    ('cars[0].distance("a", "b")', "distance() takes 1 positional argument. distance keywords: none"),
    ('cars.distance()', "distance() missing 1 required positional argument: 'other'. "
                        "distance keywords: none"),
    ('cars.order(1, 2)', "order() takes 1 positional argument. order keywords: key=, "
                         "nearest=, color=, direction=, reverse="),
    ('cars.filter(lambda r: True).filter(1, 2)',
     "filter() takes 1 positional argument. filter keywords: pred=, near=, within=, "
     "dwithin=, north_of=, south_of=, east_of=, west_of=, facing=, direction="),
    ('cars[0].filter(1, 2)',
     "filter() takes 1 positional argument. filter keywords: pred=, near=, within=, "
     "dwithin=, north_of=, south_of=, east_of=, west_of=, facing=, direction="),
])
def test_a_set_methods_signature_error_counts_no_self_and_names_no_class(code, want):
    assert _first_error(f'cars = find("car")\n{code}') == f"line 2: {want}"


def test_a_type_error_inside_a_predicate_is_served_as_its_own():
    # The model's own TypeError is not a signature error: no keyword tail.
    err = _first_error('s = find("car")\ns.filter(lambda r: r + 1)\n')
    assert "unsupported operand type(s) for +" in err and "keywords" not in err, err


def test_a_sliced_tags_column_names_the_reads_it_takes():
    err = _first_error('r = find("Tied Way").order(nearest="prior_900")[0]\nr.get("tags")[:20]')
    assert err == ('line 2: tags is a dict of OSM tags, not text: r.tags["building"], '
                   'r.tags.get("amenity"), or str(r.tags) for its text.'), err


def test_a_tags_column_reads_as_the_dict_it_is():
    _, fr, _, _ = drive([[run(
        'r = find("Tied Way").order(nearest="prior_900")[0]\n'
        'print(r.tags, r["tags"]["highway"], r.tags.get("x", 1), list(r.tags), str(r.tags),'
        ' r.tags == {"highway": "residential"}, isinstance(r.tags, dict), type(r.tags))\n'
        'r.tags.lower()')], "done"], cands=_TIED)
    out = tool_results(fr)[0]
    assert out["printed"] == ("{'highway': 'residential'} residential 1 ['highway'] "
                              "{'highway': 'residential'} True True <class 'dict'>"), out
    assert out["error"] == "line 3: AttributeError: 'dict' object has no attribute 'lower'", out

