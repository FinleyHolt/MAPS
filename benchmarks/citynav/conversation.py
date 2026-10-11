"""The mission's conversation, on disk as it happens.

``ConversationLog`` is the transcript list the loop appends to (a row per
turn, the reasons a turn was not a block, the closing ``summary``); every
``append`` also lands the row in ``<workdir>/conversation.jsonl``, opened per
append, so a mission killed between turns keeps every turn it flew.
``path=None`` is a plain list.
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
