"""The concept list SAM segments on every frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class ReasonerUnionConceptProvider:
    """Holds the concept list written last; ``SamDetectClient.detect`` sends it
    verbatim on every frame.

    Two channels write it through :meth:`set_concepts`, each a wholesale
    replacement: the mission's declared class union
    (``search_loop.declare_mission_concepts``) and the agent-owned watchlist
    (the ``PerceptionDirector`` pushes the full current set, watchlist plus
    the classes of the boosts riding a flight, on every change). Empty is
    valid and the common state before the model declares anything:
    ``detect`` then returns ``[]``.
    """

    concepts: list[str] = field(default_factory=list)

    def set_concepts(self, concepts: Iterable[str]) -> None:
        """Replace the list: case-insensitive dedup in first-seen order,
        entries stripped, blanks dropped."""
        seen: set[str] = set()
        out: list[str] = []
        for s in concepts or []:
            key = (s or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(s.strip())
        self.concepts = out
