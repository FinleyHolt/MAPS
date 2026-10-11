"""Unit tests for ReasonerUnionConceptProvider."""

from __future__ import annotations

from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider


def test_set_concepts_dedups_and_replaces_wholesale() -> None:
    provider = ReasonerUnionConceptProvider()
    assert provider.concepts == []
    provider.set_concepts(["red car", "tree", "Red Car", "  ", "tree"])
    # Case-insensitive dedup; first-occurrence casing kept, stripped; blanks dropped.
    assert provider.concepts == ["red car", "tree"]
    provider.set_concepts(["blue truck"])
    assert provider.concepts == ["blue truck"]
