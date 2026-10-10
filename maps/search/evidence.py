"""Evidence tuple e_i = (k_i, τ_i, y_i, m_i) per heater_notes.tex §"Dynamic Inputs".

Each evidence item carries its modality, the reference time it describes, its
raw content, and a metadata dict (source identity, stated confidence, sensor
pose, FoV, detection score, track id, provenance).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

EvidenceKind = Literal["text", "sensor_update", "anchor_grounded"]
_VALID_KINDS = {"text", "sensor_update", "anchor_grounded"}


@dataclass(frozen=True)
class EvidenceItem:
    """A single evidence observation e_i = (k_i, τ_i, y_i, m_i).

    Immutable after construction. ``SearchLoop.snapshot`` shallow-casts
    an evidence record into a tuple; producers must build a new
    ``EvidenceItem`` rather than mutate fields in place. ``metadata`` is a
    dict by necessity (heterogeneous m_i payload), so callers must not
    write into it post-construction either.

    ``kind`` values:
      - ``text``: operator intel / chat messages (y_i = string).
      - ``sensor_update``: per-step detection (y_i = detection payload).
    """

    kind: EvidenceKind
    reference_time: float  # τ_i: wall-clock seconds the evidence refers to
    raw: Any  # y_i: raw content (text, detection payload, sensor frame handle)
    metadata: dict[str, Any] = field(default_factory=dict)  # m_i
    id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )  # stable cross-log reference

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"EvidenceItem.kind must be one of {sorted(_VALID_KINDS)}, "
                f"got {self.kind!r}"
            )
