"""Structured representation of the committed target intel.

The reasoner's committed ``sam_concepts`` union (visual classes with
target/anchor roles) plus the color attributes extracted from the raw
instruction become a ``TargetSpec``; the runtime reads its
``target_match_classes`` to scope the commit-time observed-candidate
pool. This module owns ``TargetSpec`` only — no heuristic detection
scoring lives here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_COLOR_TOKENS = frozenset(
    {
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "purple",
        "violet",
        "pink",
        "black",
        "white",
        "gray",
        "gray",
        "brown",
        "tan",
        "beige",
        "cream",
        "gold",
        "silver",
        "bronze",
        "copper",
        "navy",
        "teal",
        "maroon",
        "olive",
    }
)


def _extract_visual_attrs(text: str | None) -> dict[str, set[str]]:
    """Pull color tokens out of a free-form prose string."""
    out: dict[str, set[str]] = {}
    if not text:
        return out
    s = text.lower()
    colors = {m for m in re.findall(r"[a-z]+", s) if m in _COLOR_TOKENS}
    if colors:
        out["color"] = colors
    return out


@dataclass
class TargetSpec:
    """Structured representation of the committed target intel.

    The visual concepts come from the Stage-1.5 concept union (the
    model's ``sam_concepts`` declarations), and the visual attributes
    are extracted from the raw instruction. There is no separate filed
    target noun: the target-role classes ARE the model's naming of the
    target.
    """

    visual_concepts: list[str] = field(default_factory=list)
    visual_concept_specs: list[dict] = field(default_factory=list)
    visual_attributes: dict[str, set[str]] = field(default_factory=dict)
    raw_instruction: str = ""

    def target_match_classes(self) -> list[str]:
        classes: list[str] = []
        seen: set[str] = set()
        required_colors = {
            " ".join(str(color).strip().casefold().split())
            for color in (self.visual_attributes.get("color") or set())
        }
        required_colors.discard("")

        for entry in self.visual_concept_specs:
            if not isinstance(entry, dict):
                continue
            roles = {
                str(r).strip().casefold()
                for r in (entry.get("roles") or [])
                if isinstance(r, str)
            }
            role = str(entry.get("role") or "").strip().casefold()
            if role != "target" and "target" not in roles:
                continue
            cls = " ".join(str(entry.get("class") or "").strip().casefold().split())
            if cls and cls not in seen:
                seen.add(cls)
                classes.append(cls)

        # The color-expansion head: the FIRST target-role class (there is
        # no separate filed noun; the target classes are the model's own
        # naming, so the head rides the same vocabulary the segmenter
        # runs).
        noun = classes[0] if classes else ""
        noun_tokens = set(noun.split())
        if noun and required_colors:
            for color in sorted(required_colors):
                if color in noun_tokens:
                    continue
                cls = f"{color} {noun}"
                if cls not in seen:
                    seen.add(cls)
                    classes.append(cls)

        if required_colors:
            color_filtered = [
                cls for cls in classes if set(cls.split()) & required_colors
            ]
            if color_filtered:
                classes = color_filtered
        specific = [cls for cls in classes if len(cls.split()) > 1]
        if specific:
            return specific
        return classes

    @classmethod
    def from_union(
        cls,
        raw_instruction: str = "",
        union_concepts: list[dict] | list[str] | None = None,
    ) -> "TargetSpec":
        """Build a TargetSpec from the Stage-1.5 SAM concept union.

        ``union_concepts`` is the union (preferred shape: list of
        ``{class, roles}`` dicts; flat list of strings also accepted);
        it becomes the canonical ``visual_concepts`` and
        ``visual_concept_specs``, keeping the spec and runtime SAM in
        lockstep.
        """
        visual_concepts: list[str] = []
        visual_concept_specs: list[dict] = []
        visual_attrs: dict[str, set[str]] = {}

        if union_concepts:
            seen: set[str] = set()
            for entry in union_concepts:
                if isinstance(entry, dict):
                    cls_str = str(entry.get("class") or "").strip().lower()
                    roles = [str(r) for r in (entry.get("roles") or ["target"])]
                else:
                    cls_str = str(entry or "").strip().lower()
                    roles = ["target"]
                if not cls_str or cls_str in seen:
                    continue
                seen.add(cls_str)
                visual_concepts.append(cls_str)
                visual_concept_specs.append({"class": cls_str, "roles": roles})

        for k, v in _extract_visual_attrs(raw_instruction).items():
            visual_attrs.setdefault(k, set()).update(v)

        return cls(
            visual_concepts=visual_concepts,
            visual_concept_specs=visual_concept_specs,
            visual_attributes=visual_attrs,
            raw_instruction=raw_instruction,
        )


__all__ = ["TargetSpec"]
