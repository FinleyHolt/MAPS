"""What the loop does with a failed reasoner request: the errors a retry may
cure, and the rejections a retry would only repeat (a request that does not
fit the window, a content filter, a refused request), each ending the
mission under its own name."""

from __future__ import annotations

import httpx

from maps.llm.endpoint_spec import TRANSIENT_CLIENT_STATUS


# Transient LLM-side errors: a retry, never the end of the reasoner (a
# non-2xx, a link fault, or a ValueError: an undecodable body or a collapsed
# completion).
_RETRIABLE_LLM_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPStatusError,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    ValueError,
)

#: What a serve says when it rejects a request for its SIZE (lowercased
#: body substrings). A spelling this misses falls through to the retry path.
_OVERSIZE_REJECTION = (
    "maximum context length",
    "context length of",
    "longer than the maximum",
    "reduce the length",
)


def _rejection_body(exc: BaseException, statuses) -> str:
    """The lowercased body of an HTTP rejection whose status is in
    ``statuses``; empty otherwise."""
    if not (isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code in statuses):
        return ""
    try:
        return exc.response.text.lower()
    except Exception:
        return ""


def _is_oversize_rejection(exc: BaseException) -> bool:
    """Did the serve reject this request because it does not FIT? A retry
    would re-send the same bytes for the same answer."""
    body = _rejection_body(exc, range(400, 500))
    return any(s in body for s in _OVERSIZE_REJECTION)


#: What a hosted endpoint says when its content filter refuses the prompt
#: (Azure: 400 with ``"code": "content_filter"``, inner code
#: ``ResponsibleAIPolicyViolation``), lowercased.
_CONTENT_FILTER_REJECTION = ("content_filter", "responsibleaipolicyviolation")


def _is_content_filter_rejection(exc: BaseException) -> bool:
    """Did a content filter refuse this prompt? The same prompt is refused the
    same way, so the mission ends rather than re-send it."""
    body = _rejection_body(exc, (400,))
    return any(s in body for s in _CONTENT_FILTER_REJECTION)


def _is_refused_request(exc: BaseException) -> bool:
    """Did the serve refuse what was sent (a 4xx outside
    ``TRANSIENT_CLIENT_STATUS``, which re-sending fails the same way)?"""
    return (isinstance(exc, httpx.HTTPStatusError)
            and 400 <= exc.response.status_code < 500
            and exc.response.status_code not in TRANSIENT_CLIENT_STATUS)


# The rejections a retry would only repeat, in the order they are tested:
# ``(test, ended_by, the log's why)``.
_REFUSAL_ENDINGS = (
    (_is_oversize_rejection, "oversize_rejection",
     "request does not fit the served window -> mission ends (retry would "
     "re-send the same bytes)"),
    (_is_content_filter_rejection, "content_filter",
     "the endpoint's content filter refused the prompt -> mission ends (a "
     "retry re-sends the same prompt)"),
    (_is_refused_request, "request_refused",
     "the serve refused the request -> mission ends (a retry re-sends the "
     "same request)"),
)
