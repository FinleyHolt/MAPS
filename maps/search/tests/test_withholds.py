"""Bench-only withholds: each takes exactly its part away, and none serves the flown
text byte for byte."""

from __future__ import annotations

import difflib
import hashlib
import json

import pytest

from maps.llm.endpoint_spec import EndpointSpec
from maps.search import withhold as W
from maps.search.parse_compile import contract, prompts
from maps.search.parse_compile import python_surface as ps
from maps.search.parse_compile import reasoner_agent as ra
from maps.search.parse_compile.code_surface import act_needs_a_thing
from maps.search.parse_compile.reasoner_loop import _PICTURE_WITHHELD
from maps.search.tests.fakes import (
    quiet_env,
    cars,
    drive,
    look_drive,
    map_mission,
    run,
    tool_results,
)

FLOWN_SHA256 = {
    "oryx_orx32_25mm_nadir": "32c45644985a346c96f4a6748d5b74c0c6c48120369404e04030233765e48115",
    "nadir_1008_90": "ae04a35e9450365d0bf3fe16e137285cc72a11b8adc0d01ac10e204e16884dec",
    None: "761e32b3eebe88c7433e939a1f06426c4950da755a5c566d2d97dbb7041dd1fc",
}


def _dump(monkeypatch, names: str, sensor: str | None = "oryx_orx32_25mm_nadir") -> str:
    monkeypatch.setenv(W.ENV, names)
    return contract.dump_text(sensor=sensor, agl_m=914.4)


def _sections(text: str) -> dict[str, str]:
    out, key = {}, None
    for ln in text.splitlines(keepends=True):
        if ln.startswith("=== ") and ln.split()[1] in ("SYSTEM", "TOOLS", "SURFACE", "USER"):
            key = ln.split()[1]
            out[key] = ""
        out[key] += ln
    return out


def _diff(a: str, b: str) -> tuple[int, int]:
    lines = list(difflib.ndiff(a.splitlines(), b.splitlines()))
    return (sum(ln.startswith("- ") for ln in lines),
            sum(ln.startswith("+ ") for ln in lines))


@pytest.mark.parametrize("sensor", list(FLOWN_SHA256))
def test_no_withhold_serves_the_flown_contract(monkeypatch, sensor):
    text = _dump(monkeypatch, "", sensor)
    assert hashlib.sha256(text.encode()).hexdigest() == FLOWN_SHA256[sensor]
    assert prompts.served_doc() == prompts.CODE_DOC


# withhold -> (sections that change, marker strings it takes away,
# (removed, added) lines of the turn, of the description, of the surface)
CASES = {
    "prior": (("SYSTEM", "TOOLS"),
              ("never a coordinate", "do not move you", "holds only PRIOR",
               "not make it movable", "the prior map alone",
               "prior (map features)", "prior footprints", "only what OSM recorded",
               "prior_3123"),
              (13, 12), (6, 6), (0, 0)),
    "images": ((), (), (0, 0), (0, 0), (0, 0)),
    "look_for": (("SYSTEM", "TOOLS", "SURFACE"),
                 ("look_for", "watched classes", "segmenting your", "ARM EVERY CLASS",
                  "WHAT YOU SEE BECOMES ROWS", "The flight folds", "SAM segments roads",
                  "pools = find(", "empty SAM response", "SAM covers"),
                 (23, 10), (21, 3), (1, 0)),
    "look": (("SYSTEM", "TOOLS", "SURFACE"),
             ("look(", "a CLOSE look", "known only from a look",
              "WHAT NO COLUMN CARRIES IS IN THE PIXELS", "look at them",
              "RELATE and LOOK", "look and relate", "loop to LOOK"),
             (13, 6), (8, 5), (1, 0)),
    "mission_map": (("SYSTEM",), ("=== MISSION MAP ===", "You are the reasoner"),
                    (40, 0), (0, 0), (0, 0)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_each_withhold_takes_exactly_its_part(monkeypatch, name):
    changes, markers, turn, doc, surface = CASES[name]
    base = _sections(_dump(monkeypatch, ""))
    got = _sections(_dump(monkeypatch, name))
    for key in base:
        if key in changes:
            assert got[key] != base[key], key
        else:
            # an unchanged section keeps its bytes, length header included
            assert got[key] == base[key], key
    assert _diff(prompts.CODE_SYSTEM_PROMPT, prompts.served_system_prompt()) == turn
    assert _diff(prompts.CODE_DOC, prompts.served_doc()) == doc
    base_surface = base["SURFACE"].split("\n", 1)[1]
    got_surface = got["SURFACE"].split("\n", 1)[1]
    assert _diff(base_surface, got_surface) == surface
    served = (prompts.served_system_prompt() + prompts.served_doc()
              + "\n".join(contract.surface_lines()) + got["SYSTEM"])
    for m in markers:
        assert m in base["SYSTEM"] + base["TOOLS"] + base_surface, m
        assert m not in served, m


def test_a_look_for_withhold_serves_no_segmenter_in_the_sensor_block(monkeypatch):
    base = _sections(_dump(monkeypatch, "", "nadir_1008_90"))["SYSTEM"]
    got = _sections(_dump(monkeypatch, "look_for", "nadir_1008_90"))["SYSTEM"]
    assert "SAM may merge" in base and "small entity counts ±1" in got
    assert "SAM" not in got, got


def test_the_system_turn_withhold_serves_no_system_message(monkeypatch):
    monkeypatch.setenv(W.ENV, "system_turn")
    assert prompts.served_system_prompt("RGB, nadir.") == ""
    spec = EndpointSpec(base_url="http://unused", model_id="inkling-small", max_model_len=262144,
                        tokenizer_name="qwen", max_output_tokens=40000)
    r = ra.Reasoner(reasoner_url="http://unused", endpoint_spec=spec,
                    sensor_description="RGB, nadir.")
    assert r.messages == []
    got = _sections(_dump(monkeypatch, "system_turn"))
    base = _sections(_dump(monkeypatch, ""))
    assert got["SYSTEM"] == "=== SYSTEM (0 chars) ===\n\n\n"
    assert got["TOOLS"] == base["TOOLS"] and got["SURFACE"] == base["SURFACE"]


def test_withheld_verbs_are_unbound(monkeypatch):
    monkeypatch.setenv(W.ENV, "look_for,look")
    assert "look_for" not in ps.served_calls() and "look" not in ps.served_calls()
    mp = ps.MissionPython(quiet_env())
    assert "look_for" not in mp.ns and "look" not in mp.ns and "go_to" in mp.ns
    out = json.dumps(mp.run_block('look_for("pool")'))
    assert "did you mean look_for" not in out


def test_a_look_for_withhold_takes_its_remedies(monkeypatch):
    assert "look_for" in act_needs_a_thing(name="go_to", got="1, 2")
    monkeypatch.setenv(W.ENV, "look_for")
    assert "look_for" not in act_needs_a_thing(name="go_to", got="1, 2")


def test_an_images_withhold_delivers_the_looks_caption(ortho_dir, monkeypatch):
    monkeypatch.setenv(W.ENV, "images")
    results, reqs, _recs, vision = look_drive(
        [[run('a = find("car")')], [run("look(a)")], "over"], cars(3), ortho_dir)
    assert reqs, "the look rendered"
    assert len(vision) == 1
    assert "slice the handle" not in json.dumps(results), results
    assert _PICTURE_WITHHELD in json.dumps(results)


def test_an_unknown_withhold_is_refused(monkeypatch):
    monkeypatch.setenv(W.ENV, "pictures")
    with pytest.raises(ValueError, match="unknown withholds"):
        W.active()


def _map_mission(script, monkeypatch, names="", renderer=None, finds=1):
    pytest.importorskip("cv2")
    monkeypatch.setenv(W.ENV, names)
    return map_mission(script, renderer=renderer, finds=finds)


def _crops(reqs):
    return [{"jpeg": b"x", "center_fid": r.get("center_fid") or "", "legend": [],
             "context_m": 30.0, "world_window": [0, 0, 60, 60], "marks_dropped": 0,
             "marks_out_of_view": [], "marks_unbadged_in_frame": []} for r in reqs]


def test_a_look_withhold_map_caption_says_fly_alone(monkeypatch):
    _, vision = _map_mission(["show_map(around=prior_7)"], monkeypatch)
    assert "fly and look: the camera" in vision[0][0]
    _, vision = _map_mission(["show_map(around=prior_7)"], monkeypatch, "look")
    assert "fly: the camera" in vision[0][0] and "fly and look" not in vision[0][0]


_PICTURE_CALLS = {"show_map(around=prior_7)": "World-state maps",
                  "go_to(prior_7)": "Overhead crops",
                  "a = explore(prior_1)": "The pass's catch",
                  "look(prior_7)": "Overhead crops"}


@pytest.mark.parametrize("call", sorted(_PICTURE_CALLS))
def test_an_images_withhold_serves_each_picture_as_its_caption(call, monkeypatch):
    _, vision = _map_mission([call], monkeypatch, renderer=_crops)
    assert [len(j) for _, j in vision] == [1]
    _, vision = _map_mission([call], monkeypatch, "images", renderer=_crops)
    assert [len(j) for _, j in vision] == [0], vision
    assert vision[0][0].startswith(_PICTURE_CALLS[call])


@pytest.mark.parametrize("call", ["show_map(around=prior_7)", "look(prior_7)"])
def test_an_images_withhold_counts_no_picture_delivered(call, monkeypatch):
    results, _ = _map_mission([call], monkeypatch, "images", renderer=_crops)
    served = json.dumps(results)
    assert '"delivered": 0' in served and '"delivered": 1' not in served, served


def test_a_look_for_withhold_serves_no_watchlist(monkeypatch):
    script = ["go_to(prior_7)", "a = explore(prior_1)", "show_map(around=prior_7)"]
    for finds in (1, 0):
        results, _ = _map_mission(script, monkeypatch, renderer=_crops, finds=finds)
        assert "watching" in json.dumps(results)
        results, _ = _map_mission(script, monkeypatch, "look_for", renderer=_crops,
                                  finds=finds)
        served = json.dumps(results)
        assert all("error" not in r for r in results), served
        assert "watching" not in served and "look_for" not in served, served


def test_a_prior_withhold_flies_to_a_point_and_serves_no_prior_row(monkeypatch):
    script = [[run('go_to([30, 30])\nexplore([40, 40])\ngo_to("[50, 50]")\n')], "done"]
    _reply, fr, flights, _d = drive(script, cands=[])
    assert "takes a THING" in (tool_results(fr)[0].get("error") or "")
    assert not flights
    monkeypatch.setenv(W.ENV, "prior")
    _reply, fr, flights, _d = drive(script, cands=[])
    served = json.dumps(tool_results(fr))
    assert "error" not in served and "prior_" not in served, served
    assert [f["target_fid"] for f in flights] == ["[30, 30]", "[40, 40]", "[50, 50]"], flights


def test_the_served_literals_hold_their_worked_block_and_every_edit_target():
    """The worked block in CODE_DOC scans on the real surface and runs the
    chain it teaches; every withhold edit names text its literal carries once."""
    import ast

    worked = "\n".join(
        ln[2:] for ln in prompts.CODE_DOC.partition("This is one block:\n")[2]
        .partition("\nEach line runs")[0].splitlines() if ln.startswith("  "))
    assert worked.count("\n") >= 5, worked
    tree = ps.scan_block(worked)
    calls = [n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert {"look_for", "find", "look", "explore", "go_to"} <= set(calls), calls
    assert calls.index("explore") < calls.index("go_to"), calls
    assert "find" in calls[calls.index("explore"):], calls
    find_kwargs = {k.arg for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "find" for k in n.keywords}
    assert "color" in find_kwargs, find_kwargs
    assert find_kwargs & {"nearest", "near", "within", "east_of", "west_of",
                          "north_of", "south_of"}, find_kwargs

    from maps.perception.geometry.sensor_spec import resolve_sensor

    for edits, text, turn in ((prompts._DOC_EDITS, prompts.CODE_DOC, False),
                              (prompts._TURN_EDITS, prompts.CODE_SYSTEM_PROMPT, True),
                              (prompts._SENSOR_EDITS, resolve_sensor("nadir_1008_90").description, False)):
        for name, items in edits.items():
            assert name in W.WITHHOLDS, name
            for e in items:
                assert prompts._edit(text, e, turn) not in (None, text), (name, e)
                if e[0] == "drop":
                    n = sum(u.startswith(e[1]) for u in prompts._units(text, turn))
                elif e[0] == "sub" and turn:
                    n = sum(e[1] in prompts._flat(u) for u in prompts._units(text, turn))
                else:
                    n = 1 if e[0] == "cut" else text.count(e[1])
                assert n == 1, (name, e)


def test_a_prior_withhold_empty_find_names_no_prior(monkeypatch):
    from maps.search.analyze_scene.geom_program import Result, compact_results

    def hint():
        return compact_results({"r0": Result(kind="miss")})["r0"]["hint"]

    assert "searched every surface row (prior + runtime) and" in hint()
    monkeypatch.setenv(W.ENV, "prior")
    assert "searched every surface row and matched nothing" in hint()
    assert "prior" not in hint()
