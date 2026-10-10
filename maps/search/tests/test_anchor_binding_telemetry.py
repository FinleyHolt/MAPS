"""Anchor-binding telemetry extraction: every prior-``find``-with-query gets
its top name-matched rows recorded (the model's own landmark resolution),
so offline analysis classifies bindings directly instead of inferring them
from viewpoint fids."""
from __future__ import annotations

from maps.search.parse_compile.reasoner_loop import (
    _extract_anchor_bindings,
)

_ROWS = [
    {"fid": "road_9", "name": "Earl Street", "name_match": 0.71, "xy": [5, 5]},
    {"fid": "road_62", "name": "Earl Street", "name_match": 0.93, "xy": [1, 2]},
    {"fid": "building_3", "name": "Earl House", "name_match": 0.41, "xy": [9, 9]},
]


def test_prior_query_find_is_recorded_name_match_sorted():
    program = {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"provenance": "prior", "query": "Earl Street"}},
    ]}
    out = _extract_anchor_bindings(program, {"r0": list(_ROWS)})
    assert len(out) == 1
    b = out[0]
    assert b["query"] == "Earl Street"
    assert b["provenance"] == "prior"
    assert b["n_results"] == 3
    assert [r["fid"] for r in b["results"]] == ["road_62", "road_9", "building_3"]


def test_runtime_and_queryless_finds_are_not_bindings():
    program = {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"provenance": "runtime", "query": "silver car"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "prior"}},
    ]}
    assert _extract_anchor_bindings(program, {"r0": list(_ROWS), "r1": list(_ROWS)}) == []


def test_sigiled_ref_reads_bare_results_and_caps_rows():
    rows = [{"fid": f"building_{i}", "name": "Mill Rd", "name_match": 0.5}
            for i in range(10)]
    program = {"ops": [
        {"ref": "$r2", "op": "find",
         "args": {"provenance": "any", "query": "Mill Rd"}},
    ]}
    out = _extract_anchor_bindings(program, {"r2": rows})
    assert len(out) == 1
    assert out[0]["ref"] == "r2"
    assert out[0]["n_results"] == 10
    assert len(out[0]["results"]) == 6


def test_malformed_inputs_yield_nothing():
    assert _extract_anchor_bindings(None, {}) == []
    assert _extract_anchor_bindings({"ops": "x"}, {}) == []
    program = {"ops": [{"ref": "r0", "op": "find",
                        "args": {"provenance": "prior", "query": "X"}}]}
    assert _extract_anchor_bindings(program, {})[0]["results"] == []
