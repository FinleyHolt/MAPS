"""The mission's conversation, on disk as it happens.

A mission's transcript is a list the loop appends to: one row per turn
(``run``), plus the reasons a turn was not a block and the end-of-mission
``summary``. ``ConversationLog`` IS that list, and every ``append`` also
lands the row in ``<workdir>/conversation.jsonl`` before returning, so a
mission killed between turns still leaves every turn it flew. A dump at
mission end left nothing for exactly the missions worth reading.

Open-per-append rather than a held handle: a row is a few hundred bytes a
turn, and a handle nothing closes is what a SIGKILL leaves half-flushed.
``path=None`` is a plain list -- a unit test, not a mission.
"""

from __future__ import annotations

import json
from pathlib import Path

CONVERSATION_JSONL = "conversation.jsonl"


class ConversationLog(list):
    """A transcript list whose every append also lands on disk."""

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self.path = Path(path) if path is not None else None

    def append(self, row: dict) -> None:  # type: ignore[override]
        super().append(row)
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            fh.flush()
