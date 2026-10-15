"""A preflight sort whose colour term dropped says why once: the score_by
note, not a second preflight note restating it."""
from __future__ import annotations

from maps.search.parse_compile.carry import _preflight_color_note, run_ops


def test_a_dropped_colour_term_is_explained_once():
    anchors = [{"fid": f"prior_{i}", "name": f"House {i}", "xy": [10.0 * i, 0.0],
                "tags": {"building": "house"},
                "search_text": f"house {i} building house", "cells": [],
                "color": ""} for i in range(1, 4)]
    ops = [{"ref": "r1", "op": "find", "args": {"query": "house"}},
           {"ref": "r2", "op": "compare",
            "args": {"items": "r1", "refs": ["prior_1"]}},
           {"ref": "r3", "op": "score_by", "args": {"items": "r2", "emphasis": [
               {"key": "distance to prior_1", "w": -1},
               {"key": "color_red", "w": 1}]}}]
    out = run_ops(ops, namespace={"cands": [], "anchors": anchors},
                  preflight=True, answer_ref="r3")
    hint = out["results"]["r3"]["hint"]
    assert "'color_red' ranked nothing here" in hint, hint
    assert _preflight_color_note() not in hint, hint
