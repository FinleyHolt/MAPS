"""Unit tests for ReasonerUnionConceptProvider."""

from __future__ import annotations

from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider


def test_empty_until_set_concepts_called() -> None:
    provider = ReasonerUnionConceptProvider()
    assert provider.concepts_for(obs=None) == []


def test_set_concepts_dedups_and_preserves_order() -> None:
    provider = ReasonerUnionConceptProvider()
    provider.set_concepts(["red car", "tree", "Red Car", "  ", "tree"])
    # Case-insensitive dedup; original first-occurrence casing preserved
    # (with surrounding whitespace stripped); blank entries discarded.
    assert provider.concepts_for(obs=None) == ["red car", "tree"]


def test_set_concepts_replaces_the_prior_call() -> None:
    """``set_concepts`` replaces the held union wholesale: the union builder
    already produced a deduped, canonical set, and carrying entries from a
    prior call would defeat that canonicalization."""
    provider = ReasonerUnionConceptProvider()
    provider.set_concepts(["red car", "tree"])
    provider.set_concepts(["blue truck"])
    assert provider.concepts_for(obs=None) == ["blue truck"]


def test_concepts_for_ignores_instruction_hint() -> None:
    provider = ReasonerUnionConceptProvider()
    provider.set_concepts(["building"])
    a = provider.concepts_for(obs=None, instruction_hint="anything")
    b = provider.concepts_for(obs=None, instruction_hint=None)
    assert a == b == ["building"]


def test_set_concepts_returns_fresh_list_each_call() -> None:
    provider = ReasonerUnionConceptProvider()
    provider.set_concepts(["building"])
    out1 = provider.concepts_for(obs=None)
    out1.append("mutation")
    out2 = provider.concepts_for(obs=None)
    assert out2 == ["building"]
