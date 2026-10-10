"""Tests for resolve_class_hint."""

from __future__ import annotations

import pytest

from maps.perception.sam.class_hint import _load_map, resolve_class_hint


def test_exact_match():
    assert resolve_class_hint("building") == "buildings"
    assert resolve_class_hint("road") == "roads"
    assert resolve_class_hint("river") == "water"
    assert resolve_class_hint("parking lot") == "parking"
    assert resolve_class_hint("tree") == "natural"


def test_case_and_whitespace_insensitive():
    assert resolve_class_hint("BUILDING") == "buildings"
    assert resolve_class_hint("  road  ") == "roads"


def test_close_match_handles_minor_typos():
    # Single-char typo / plural form lands in the right bucket thanks to
    # difflib close-match.
    assert resolve_class_hint("buildinq") == "buildings"


def test_unknown_label_falls_back_to_object():
    assert resolve_class_hint("random_unseen_thing_xyz") == "object"
    assert resolve_class_hint("") == "object"
    assert resolve_class_hint("   ") == "object"


def test_discrete_object_labels_collapse_to_object_bucket():
    # Cars / people / street furniture no longer carry a coarse identity
    # bucket — the fine label lives on the graph node; class_hint is just
    # the dedup/re-id bucket, so they all resolve to the generic "object".
    for label in ("car", "white car", "person", "bench", "sign"):
        assert resolve_class_hint(label) == "object"


def test_underscore_key_ignored():
    # The "_doc" comment key must not leak into the mapping.
    mapping = _load_map()
    assert "_doc" not in mapping
    assert all(not k.startswith("_") for k in mapping.keys())


def test_mapping_covers_all_open_vocab_buckets():
    mapping = _load_map()
    buckets = set(mapping.values())
    # The class_hint contract requires all these buckets to be representable.
    expected = {"object", "buildings", "water", "roads", "parking", "natural", "landuse"}
    assert expected.issubset(buckets)
