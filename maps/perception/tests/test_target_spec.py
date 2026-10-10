#!/usr/bin/env python
"""``TargetSpec.target_match_classes`` pins.

There is no filed target noun: the color-expansion head derives from the
FIRST target-role class of the union, and the returned classes are the
target-role classes (color-expanded and color-filtered when the raw
instruction names a color). Run:
``pytest maps/perception/tests/test_target_spec.py``.
"""
from __future__ import annotations

from maps.perception.target_spec import TargetSpec


def _spec(instruction: str, union: list[dict]) -> TargetSpec:
    return TargetSpec.from_union(
        raw_instruction=instruction, union_concepts=union
    )


def test_target_role_classes_lead():
    spec = _spec("find the car by the hedge", [
        {"class": "car", "roles": ["target"]},
        {"class": "hedge", "roles": ["anchor"]},
    ])
    assert spec.target_match_classes() == ["car"]


def test_color_expansion_head_is_first_target_class():
    # "gray" from the instruction + target class "car" => "gray car",
    # exactly what the filed-noun head used to produce; the color filter
    # then keeps the color-bearing entry.
    spec = _spec("find the gray car parked near the shed", [
        {"class": "car", "roles": ["target"]},
        {"class": "shed", "roles": ["anchor"]},
    ])
    assert spec.target_match_classes() == ["gray car"]


def test_anchor_only_union_yields_no_target_classes():
    spec = _spec("find the building with the blue roof", [
        {"class": "building", "roles": ["anchor"]},
    ])
    assert spec.target_match_classes() == []


def test_multi_target_classes_kept_and_specific_preferred():
    spec = _spec("find the white van", [
        {"class": "van", "roles": ["target"]},
        {"class": "delivery van", "roles": ["target"]},
    ])
    # Color expansion off the first target class ("white van"), color
    # filter keeps color-bearing entries, specific (multi-word) preferred.
    assert spec.target_match_classes() == ["white van"]


def test_no_color_returns_bare_target_classes():
    spec = _spec("find the truck", [
        {"class": "truck", "roles": ["target"]},
        {"class": "box truck", "roles": ["target"]},
    ])
    # specific (multi-word) classes win when present.
    assert spec.target_match_classes() == ["box truck"]


def test_empty_union_returns_empty():
    spec = _spec("find the gray car", [])
    assert spec.target_match_classes() == []
