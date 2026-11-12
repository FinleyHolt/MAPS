"""The thinking span decides the reasoning-token count: a response with no span reads zero whatever
``usage.reasoning_tokens`` claims (a serve reports completion - 1 on a null span)."""

from __future__ import annotations

import pytest

from maps.llm.flight_summary import reasoning_tokens_of


def _response(message: dict, *, completion: int, reported: int) -> dict:
    return {
        "model": "inkling-small",
        "choices": [{"message": {"content": "ok", **message}}],
        "usage": {"completion_tokens": completion, "reasoning_tokens": reported},
    }


@pytest.mark.parametrize(
    "message",
    [{"reasoning_content": None}, {}, {"reasoning_content": ""}],
    ids=["null_span", "absent_key", "empty_span"],
)
def test_no_span_reads_zero_however_loudly_usage_disagrees(message) -> None:
    assert reasoning_tokens_of(_response(message, completion=26, reported=25)) == 0


def test_a_real_span_still_takes_the_serve_s_own_count() -> None:
    assert (
        reasoning_tokens_of(
            _response(
                {"reasoning_content": "thought about it at length"}, completion=318, reported=250
            )
        )
        == 250
    )
