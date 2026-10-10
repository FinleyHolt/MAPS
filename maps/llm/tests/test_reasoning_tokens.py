"""The thinking span decides the reasoning-token count, not ``usage``.

sglang 0.6.17 serving Inkling reports ``reasoning_tokens =
completion_tokens - 1`` on a response whose ``reasoning_content`` is
``null``. Reading that field first made an arm served
``reasoning_effort=0.0`` -- 341 responses, ZERO carrying a span -- report
339 of them as almost entirely thinking, so the least-thinking rung of an
effort ladder read as the most-thinking one, monotonically, which is the
shape a reader believes.
"""

from __future__ import annotations

from maps.llm.net_meter import reasoning_tokens_of


def _response(reasoning: str | None, *, completion: int, reported: int | None) -> dict:
    usage: dict = {"completion_tokens": completion}
    if reported is not None:
        usage["reasoning_tokens"] = reported
    return {
        "model": "inkling-small",
        "choices": [{"message": {"content": "ok", "reasoning_content": reasoning}}],
        "usage": usage,
    }


def test_no_span_reads_zero_however_loudly_usage_disagrees() -> None:
    """The measured shape: null span, and usage claiming completion - 1."""
    assert reasoning_tokens_of(_response(None, completion=26, reported=25)) == 0


def test_an_absent_reasoning_key_also_reads_zero() -> None:
    assert reasoning_tokens_of(
        {"model": "m", "choices": [{"message": {"content": "ok"}}],
         "usage": {"completion_tokens": 61, "reasoning_tokens": 60}}
    ) == 0


def test_an_empty_span_is_not_a_span() -> None:
    assert reasoning_tokens_of(_response("", completion=61, reported=60)) == 0


def test_a_real_span_still_takes_the_serve_s_own_count() -> None:
    """The field is exact where a span exists -- the gate must not cost that."""
    assert reasoning_tokens_of(
        _response("thought about it at length", completion=318, reported=250)
    ) == 250


def test_the_gate_is_what_separates_the_ladder_s_ends() -> None:
    """Floor on the whole point: an effort ladder's rungs must not invert.

    Without the gate this reads 25 for the thinking-free rung against 250
    for the thinking one -- ordered, plausible, and backwards.
    """
    none_rung = reasoning_tokens_of(_response(None, completion=26, reported=25))
    full_rung = reasoning_tokens_of(_response("t" * 400, completion=318, reported=250))
    assert none_rung == 0
    assert full_rung > 0
    assert none_rung < full_rung
