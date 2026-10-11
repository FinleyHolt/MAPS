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


@pytest.mark.parametrize("body", [
    SGLANG_OVERSIZE,
    SGLANG_OVERSIZE.upper(),
    "This model's maximum context length is 65536 tokens",
    "prompt is longer than the maximum model length",
    "Please reduce the length of the messages",
])
def test_a_400_saying_the_request_does_not_fit_is_permanent(body: str) -> None:
    assert _is_oversize_rejection(_status_error(400, body))


@pytest.mark.parametrize("err", [
    # a grammar refusal: a fresh generation differs, so a retry fixes it
    _status_error(400, "Failed to parse tool call arguments against schema"),
    # fail-safe: an unrecognised 400 keeps the retry path
    _status_error(400, "something else entirely"),
    # capacity, which the transport already retries
    _status_error(503, "maximum context length exceeded downstream"),
    httpx.ConnectError("tunnel flap"),
    ValueError("empty body"),
])
def test_everything_else_stays_retriable(err) -> None:
    assert not _is_oversize_rejection(err)
