"""A request that does not FIT is terminal; a rejected generation is not.

Both arrive as an httpx 400 and the loop retried both, six times, with the
same bytes each time. On the reasoning-effort ladder that landed on 7 of 32
episodes of the max-effort arm and ZERO of the other two -- a defect
attacking one arm of a three-arm comparison, and the arm with the largest
effect.

The discriminator is whether a retry changes the request. A grammar
rejection turns on what the model GENERATED, so a fresh generation differs.
An oversize rejection turns on what we SENT, and the retry re-sends it.
"""

from __future__ import annotations

import httpx
import pytest

from maps.search.parse_compile.reasoner_loop import _is_oversize_rejection


def _status_error(status: int, body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://127.0.0.1:30000/v1/chat/completions")
    response = httpx.Response(status, text=body, request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


# The literal sglang body, from e0.99 of job 1902.
SGLANG_OVERSIZE = (
    "Requested token count exceeds the model's maximum context length of "
    "65536 tokens. You requested a total of 66644 tokens: 50260 tokens from "
    "the input messages and 16384 tokens for the completion."
)


def test_the_served_window_rejection_is_permanent() -> None:
    assert _is_oversize_rejection(_status_error(400, SGLANG_OVERSIZE))


@pytest.mark.parametrize(
    "body",
    [
        "This model's maximum context length is 65536 tokens",
        "prompt is longer than the maximum model length",
        "Please reduce the length of the messages",
    ],
)
def test_other_spellings_of_the_same_refusal(body: str) -> None:
    assert _is_oversize_rejection(_status_error(400, body))


def test_a_rejected_generation_stays_retriable() -> None:
    """xgrammar refusing a malformed tool-args envelope. A fresh generation
    produces a different envelope, so the retry is what fixes it."""
    assert not _is_oversize_rejection(
        _status_error(400, "Failed to parse tool call arguments against schema")
    )


def test_an_unrecognised_400_stays_retriable() -> None:
    """Fail-safe: a spelling this misses degrades to today's retry path.
    It can cost a wasted retry; it must not turn a retriable error
    permanent."""
    assert not _is_oversize_rejection(_status_error(400, "something else entirely"))


def test_a_5xx_is_not_this_and_never_was() -> None:
    """Capacity, and the transport layer already retries it. Matching the
    prose in a 503 body would steal an error that a retry does fix."""
    assert not _is_oversize_rejection(
        _status_error(503, "maximum context length exceeded downstream")
    )


def test_a_non_http_error_is_not_this() -> None:
    assert not _is_oversize_rejection(httpx.ConnectError("tunnel flap"))
    assert not _is_oversize_rejection(ValueError("empty body"))


def test_the_matcher_is_case_insensitive() -> None:
    assert _is_oversize_rejection(_status_error(400, SGLANG_OVERSIZE.upper()))
