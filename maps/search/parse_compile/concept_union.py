"""Manager-level concept-union builder.

Each persistent reasoner thread carries its own list of SAM 3.1 noun
phrases (``ReasonerThread.sam_concepts``) seeded at parse time. When a
turn dispatches SAM, we want a single segmentation pass over the union
of every live thread's concepts rather than one pass per thread. That
union must collapse trivial synonyms (case and whitespace variants of
the same phrase) so SAM is not asked the same question twice.

The collapse is a deterministic case-fold + whitespace normalization
(case and whitespace variants of the same phrase fold to one canonical
form). SAM 3.1 image-mode batches across concepts in one image-encoder
pass and the per-concept text encode + decode is cheap, so heavier
dedup is not worth a round trip.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from maps.search.parse_compile.perception_director import WATCH_ROLES

log = logging.getLogger(__name__)


@dataclass
class ConceptUnionResult:
    """Output of one ConceptUnionBuilder.build() call.

    ``union`` is the canonical class-string set fed to SAM 3.1 this
    turn (SAM does not consume roles). ``union_specs`` carries the
    same classes with their merged role tags
    (``{"class": str, "roles": list[str]}``). ``mutations[thread_id]
    [raw_concept] == canonical_concept`` for every raw concept the
    thread proposed; entries where raw == canonical mean the
    proposal was accepted as-is. ``raw_to_canonical`` is a flat
    accumulation over every thread's proposals.
    """

    union: list[str] = field(default_factory=list)
    union_specs: list[dict] = field(default_factory=list)
    mutations: dict[str, dict[str, str]] = field(default_factory=dict)
    raw_to_canonical: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-safe view for run-checkpointing."""
        return {
            "union": list(self.union),
            "union_specs": [
                {
                    "class": str(s.get("class") or ""),
                    "roles": list(s.get("roles") or []),
                }
                for s in self.union_specs
            ],
            "mutations": {
                str(tid): {str(k): str(v) for k, v in (per or {}).items()}
                for tid, per in self.mutations.items()
            },
            "raw_to_canonical": {
                str(k): str(v) for k, v in self.raw_to_canonical.items()
            },
        }


def _normalize_concept(raw: str) -> str:
    """Case-fold and collapse whitespace; empty input returns ``""``."""
    return " ".join((raw or "").strip().lower().split())


class ConceptUnionBuilder:
    """Deterministic canonicalizer for per-thread SAM concept proposals.

    Accepts either flat string proposals (legacy) or role-tagged
    ``(class, role)`` tuples. The ``union`` returned to SAM is always
    flat class strings (SAM does not consume roles); ``union_specs``
    carries the same classes with their merged roles for downstream
    triage.
    """

    def build(
        self,
        proposals: dict[str, list],
    ) -> ConceptUnionResult:
        """Canonicalize per-thread concepts into a single union.

        Empty thread proposals are silently dropped. Empty (or
        whitespace-only) class strings are also dropped. When the
        same class appears under multiple roles across threads, the
        ``union_specs`` entry merges both roles.
        """
        union: list[str] = []
        seen: set[str] = set()
        mutations: dict[str, dict[str, str]] = {}
        raw_to_canonical: dict[str, str] = {}
        roles_by_class: dict[str, list[str]] = {}
        for thread_id, raws in proposals.items():
            per: dict[str, str] = {}
            for raw in raws or []:
                if isinstance(raw, tuple) and len(raw) == 2:
                    raw_class, raw_role = raw
                else:
                    raw_class, raw_role = raw, "target"
                if not isinstance(raw_class, str):
                    continue
                normalized = _normalize_concept(raw_class)
                if not normalized:
                    continue
                role = str(raw_role).strip().lower() if raw_role else "target"
                if role not in WATCH_ROLES:
                    role = "target"
                per[raw_class] = normalized
                raw_to_canonical[raw_class] = normalized
                if normalized not in seen:
                    seen.add(normalized)
                    union.append(normalized)
                bucket = roles_by_class.setdefault(normalized, [])
                if role not in bucket:
                    bucket.append(role)
            mutations[thread_id] = per
        union_specs = [
            {"class": c, "roles": list(roles_by_class.get(c) or ["target"])}
            for c in union
        ]
        return ConceptUnionResult(
            union=union,
            union_specs=union_specs,
            mutations=mutations,
            raw_to_canonical=raw_to_canonical,
        )
