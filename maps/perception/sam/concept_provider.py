"""Concept providers — what to look for on each frame.

``ReasonerUnionConceptProvider`` returns a ``list[str]`` of short noun phrases for
SAM 3.1's concept-prompt segmenter. The source of truth in the v1
parse-first stack is the canonicalized union of every active reasoner
thread's ``sam_concepts``: the reasoning engine's ``ConceptUnionBuilder``
deduplicates and rewrites synonyms to a single canonical form, capped at
32 entries, and the runner pushes that union into the provider via
:meth:`ReasonerUnionConceptProvider.set_concepts` after the pre-flight
compile turn. The provider then serves the same list verbatim on every
per-frame ``detect`` call for the rest of the episode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

from maps.perception.observation import Observation

log = logging.getLogger(__name__)


@dataclass
class ReasonerUnionConceptProvider:
    """Holds the LLM-canonicalized concept union from the pre-flight compile.

    TWO CHANNELS WRITE HERE, and the mission path uses only the second.

    :meth:`set_concepts` + :meth:`freeze` are the union channel: a deduped,
    canonicalized list built across reasoner threads, pushed once and then
    locked. On the bench that fires ONLY on resume-restore
    (``episode_runner`` re-arms a finished catalog); ``robots`` uses it too.
    A mission conversation never touches it.

    :meth:`apply_watchlist` is what a mission actually drives, and it is
    freeze-exempt on purpose. ``perceive``'s watchlist channel calls
    ``search_loop.declare_mission_concepts`` with the maintained full set on
    every add or remove, latest-wins by class, at ANY turn — the model owns
    what gets segmented and can retarget it mid-flight. ``mission_engine``
    arms this empty on turn one and the model fills it.

    ``concepts_for`` returns whichever list was written last. Empty is a valid
    state and the common one before the model has declared anything:
    ``SamDetectClient.detect`` short-circuits to ``[]`` rather than segmenting
    for nobody.

    Why freeze exists at all, given the above: it stops a mid-flight COMPILE
    turn from silently replacing a restored catalog. It was never a statement
    that concepts are pre-flight-only, and reading it that way is how the
    engine's text-encoder memo came to be justified by an episode-long freeze
    that does not happen. What makes that memo pay is measured reuse (936
    ``/segment`` calls against 19 distinct lists in job 1806), not a lock.
    """

    _concepts: list[str] = field(default_factory=list)
    _frozen: bool = False

    def set_concepts(self, concepts: Iterable[str]) -> None:
        # Wholesale replacement until frozen. The union builder already
        # produced the deduped, canonicalized set across every active
        # thread, so carrying stale entries from a prior call would
        # defeat the canonicalization that just happened. After freeze,
        # in-flight compile turns may still call this (the SearchLoop
        # is unconditionally wired) — those calls are no-ops. Any future
        # scout / orchestrator that wants to grow the SAM class set
        # MUST do so by writing additions back into reasoner threads'
        # sam_concepts and re-running the union — never by patching
        # this field directly.
        if self._frozen:
            return
        self._concepts = _dedup_preserve_order(concepts)

    def freeze(self) -> None:
        """Lock the current concept list. Subsequent :meth:`set_concepts`
        calls are silently ignored."""
        if self._frozen:
            return
        self._frozen = True
        log.info(
            "concept provider frozen with %d concepts: %s",
            len(self._concepts),
            self._concepts,
        )

    def apply_watchlist(self, concepts: Iterable[str]) -> None:
        """The AGENT-OWNED watchlist channel (three-verb surface) —
        deliberately EXEMPT from :meth:`freeze`.

        ``freeze`` exists to stop mid-flight COMPILE turns from silently
        replacing the catalog (the old session rationale;
        per-frame perception is stateless image-mode ``/segment`` today).
        The C4 ``view{watchlist}``/boost channel is the opposite of that
        drift: an explicit agent decision to retarget SAM mid-flight, plus
        the runtime's boost activation/retirement pushes. The SearchLoop's
        PerceptionDirector is the single caller; it always pushes the FULL
        current set (watchlist + active boost concepts), so this is a
        wholesale replacement exactly like ``set_concepts``."""
        self._concepts = _dedup_preserve_order(concepts)
        log.info(
            "concept provider watchlist applied (%d concepts): %s",
            len(self._concepts),
            self._concepts,
        )

    @property
    def frozen(self) -> bool:
        return self._frozen

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
