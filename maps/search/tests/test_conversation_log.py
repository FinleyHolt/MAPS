"""``conversation.jsonl`` holds every row the loop appended, as it happened."""

from __future__ import annotations

import json

from maps.search.parse_compile.conversation import (
    CONVERSATION_JSONL,
    ConversationLog,
)


def _lines(path):
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_every_append_is_on_disk_before_the_next(tmp_path):
    path = tmp_path / "ep" / CONVERSATION_JSONL
    log = ConversationLog(path)
    log.append({"stage": "mission", "kind": "run", "turn_idx": 0, "said": "x = 1"})
    assert _lines(path) == [{"stage": "mission", "kind": "run", "turn_idx": 0, "said": "x = 1"}]
    log.append({"stage": "mission", "kind": "summary", "ended_by": "reply"})
    rows = _lines(path)
    assert len(rows) == 2 and rows[1]["kind"] == "summary"
    assert list(log) == rows


def test_a_log_with_no_path_is_a_plain_list(tmp_path):
    log = ConversationLog(None)
    log.append({"kind": "run"})
    assert log == [{"kind": "run"}]
    assert not list(tmp_path.iterdir())


def test_unserializable_values_land_as_strings(tmp_path):
    path = tmp_path / CONVERSATION_JSONL
    log = ConversationLog(path)
    log.append({"kind": "run", "when": object()})
    assert "object" in _lines(path)[0]["when"]
