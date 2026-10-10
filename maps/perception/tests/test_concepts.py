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


def test_set_concepts_replaces_prior_turn_until_frozen() -> None:
    """Pre-freeze, ``set_concepts`` replaces the held union wholesale.

    The compiler's union builder already produced a deduped, canonical
    set across every active reasoner thread; carrying entries from a
    prior call would defeat that canonicalization. (Post-freeze,
    subsequent calls are no-ops — see the freeze tests below.)
    """
    provider = ReasonerUnionConceptProvider()
    provider.set_concepts(["red car", "tree"])
    provider.set_concepts(["blue truck"])
    assert provider.concepts_for(obs=None) == ["blue truck"]


def test_freeze_locks_concept_list() -> None:
    """After ``freeze``, ``set_concepts`` calls are silently ignored.

    The runner calls this once after the pre-flight compile so the
    SAM-video tracker keeps the turn-0 union for the rest of the
    episode, even when in-flight compile turns produce a different
    union (e.g. operator intel changes the target).
    """
    provider = ReasonerUnionConceptProvider()
    provider.set_concepts(["building", "car"])
    provider.freeze()
    provider.set_concepts(["unrelated"])
    assert provider.concepts_for(obs=None) == ["building", "car"]
    assert provider.frozen is True


def test_freeze_is_idempotent() -> None:
    provider = ReasonerUnionConceptProvider()
    provider.set_concepts(["a"])
    provider.freeze()
    provider.freeze()
    assert provider.concepts_for(obs=None) == ["a"]
    assert provider.frozen is True


def test_freeze_with_empty_concepts_is_valid() -> None:
    """Freezing before any concepts are set leaves the provider empty.

    Caller-side: a crashed pre-flight compile produces no union; the
    runner still freezes so a stray in-flight turn can't retroactively
    populate the catalog."""
    provider = ReasonerUnionConceptProvider()
    provider.freeze()
    provider.set_concepts(["building"])
    assert provider.concepts_for(obs=None) == []


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
