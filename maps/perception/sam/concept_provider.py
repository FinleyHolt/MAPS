"""Concept providers — what to look for on each frame.

``ReasonerUnionConceptProvider`` returns a ``list[str]`` of short noun phrases for
SAM 3.1's concept-prompt segmenter. The source of truth is the declared
class union (``maps.search.parse_compile.compiler.concept_union``: case-folded,
deduplicated, first-seen order), which the search loop pushes into the
provider via :meth:`ReasonerUnionConceptProvider.set_concepts`. The provider
serves that list verbatim on every per-frame ``detect`` call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

from maps.perception.observation import Observation

log = logging.getLogger(__name__)


@dataclass
class ReasonerUnionConceptProvider:
    """Holds the concept list the mission declared last.

    TWO CHANNELS WRITE HERE, both wholesale replacements.

    :meth:`set_concepts` is the union channel: the deduped, canonicalized
    list ``search_loop.declare_mission_concepts`` builds from the mission's
    declarations.

    :meth:`apply_watchlist` is the agent-owned watchlist channel.
    ``perceive``'s watchlist channel pushes the maintained full set on every
    add or remove, latest-wins by class, at ANY turn: the model owns what
    gets segmented and can retarget it mid-flight. ``mission_engine`` arms
    this empty on turn one and the model fills it.

    ``concepts_for`` returns whichever list was written last. Empty is a valid
    state and the common one before the model has declared anything:
    ``SamDetectClient.detect`` short-circuits to ``[]`` rather than segmenting
    for nobody.
    """

    _concepts: list[str] = field(default_factory=list)

    def set_concepts(self, concepts: Iterable[str]) -> None:
        # Wholesale replacement: the union builder already produced the
        # deduped, canonicalized set, so carrying stale entries from a prior
        # call would defeat the canonicalization that just happened.
        self._concepts = _dedup_preserve_order(concepts)

    def apply_watchlist(self, concepts: Iterable[str]) -> None:
        """The AGENT-OWNED watchlist channel: an explicit agent decision to
        retarget SAM mid-flight, plus the runtime's boost
        activation/retirement pushes. The SearchLoop's PerceptionDirector
        is the single caller; it always pushes the FULL current set
        (watchlist + active boost concepts), so this is a wholesale
        replacement exactly like ``set_concepts``."""
        self._concepts = _dedup_preserve_order(concepts)
        log.info(
            "concept provider watchlist applied (%d concepts): %s",
            len(self._concepts),
            self._concepts,
        )

    def concepts_for(
        self, obs: Observation | None = None, instruction_hint: str | None = None
    ) -> list[str]:
        del obs, instruction_hint
        return list(self._concepts)


def _dedup_preserve_order(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq or []:
        key = (s or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s.strip())
    return out
