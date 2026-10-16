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

# Slices, tier views, filters, show, show_map, the go_to/explore refusals,
# print and the slots, each over a bound, an unbound and a nameless set.
SWEEP = [
    'cars = find("car")\nprint(cars[0:2])\n',
    'print(find("car")[0:3])\n',
    f'print({_FILTERED})\n',
    'cars = find("car")\ncars[:2].exact\n',
    'find("car")[0:2].exact\n',
    'cars = find("car")\nshow(cars.exact)\n',
    'cars = find("car")\ncars.filter(near=cars[0], dwithin=50).exact\n',
    'cars = find("car")\nnear = cars[:2].filter(near="prior_7", dwithin=500)\n',
    'find("car")[0:2].filter(near="prior_7", dwithin=5000)\n',
    'find("depot", near=find("car")[0:3])\n',
    'find("car").order(nearest=find("depot")[0:1])\n',
    'find("car").filter(near=find("depot").exact, dwithin=100)\n',
    f'x = find("depot").order(nearest={_FILTERED})\nprint(x[0])\n',
    'show(find("car")[1:3])\n',
    'cars = find("car")\nshow(cars[1:3])\n',
    f'show({_FILTERED})\n',
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
]


def _served(code: str) -> tuple[str, dict]:
    _, fr, _, _ = drive([[run(code)], "done"], cands=_CANDS)
    out = tool_results(fr)[0]
    out.pop("said", None)
    return json.dumps(out), out


@pytest.mark.parametrize("code", SWEEP)
def test_no_served_text_names_an_rn_the_model_did_not_bind(code):
    text, out = _served(code)
    own = set(re.findall(r"\br\d+\b", code)) | set(out.get("handles") or ())
    assert not set(re.findall(r"\br\d+\b", text)) - own, text


def _keys(out: dict) -> list:
    return list(((out.get("result") or {}).get("results") or {}))


def test_a_slice_prints_under_its_spelling():
    _, out = _served('cars = find("car")\nprint(cars[0:2])\n')
    assert out["printed"].startswith("<cars[0:2]: 2 row(s)"), out
    _, out = _served(f"print({_FILTERED})\n")
    assert out["printed"].startswith("<(unnamed): 8 row(s)"), out


def test_a_tier_view_is_shown_once_under_its_spelling():
    _, out = _served('cars = find("car")\nshow(cars.exact)\n')
    assert _keys(out) == ["cars.exact"], out
    _, out = _served('cars = find("car")\ncars[:2].exact\n')
    assert "cars[0:2].exact" in _keys(out), out


def test_a_nameless_set_is_shown_unnamed_and_refused_with_bind_it_first():
    _, out = _served(f"show({_FILTERED})\n")
    assert _keys(out) == ["(unnamed 1)"], out
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
    _, out = _served('show(find("car"))\n'
                     'post = find("car", nearest="prior_7")[0]\n')
    assert _keys(out) == ["find('car')", "(unnamed 1)"], out


@pytest.mark.parametrize("code", [
    f"show({_GATE})\nexplore({_FILTERED})\n",
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
