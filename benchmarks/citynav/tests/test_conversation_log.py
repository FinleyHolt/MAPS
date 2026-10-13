"""``conversation.jsonl`` holds every row the loop appended, as it happened."""

from __future__ import annotations

import json

from benchmarks.citynav.conversation import (
    CONVERSATION_JSONL,
    ConversationLog,
)


def _lines(path):
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_every_append_is_on_disk_before_the_next(tmp_path):
    path = tmp_path / "ep" / CONVERSATION_JSONL
    log = ConversationLog(path)
    log.append({"kind": "run", "turn_idx": 0, "said": "x = 1"})
    assert _lines(path) == [{"kind": "run", "turn_idx": 0, "said": "x = 1"}]
    log.append({"kind": "retask", "said": "look north", "when": object()})
    rows = _lines(path)
    assert len(rows) == 2 and rows[1]["kind"] == "retask"
    assert "object" in rows[1]["when"], "an unserializable value lands as its string"
    assert log[0] == rows[0]


def test_a_log_with_no_path_is_a_plain_list(tmp_path):
    log = ConversationLog(None)
    log.append({"kind": "run"})
    assert log == [{"kind": "run"}]
    assert not list(tmp_path.iterdir())
