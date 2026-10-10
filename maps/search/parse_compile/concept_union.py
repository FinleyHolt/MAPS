"""``ConceptUnionBuilder`` over ``compiler.concept_union``, for the search
loop's declaration call."""

from __future__ import annotations

from types import SimpleNamespace

from maps.search.parse_compile.compiler import concept_union


class ConceptUnionBuilder:
    def build(self, proposals: dict[str, list]) -> SimpleNamespace:
        union, union_specs = concept_union(
            [spec for specs in proposals.values() for spec in specs])
        return SimpleNamespace(union=union, union_specs=union_specs)
