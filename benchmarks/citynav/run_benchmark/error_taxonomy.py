"""Exception classifiers that separate infrastructure casualties (excluded from
the SR denominator and re-queued) from genuine algorithm failures (scored misses).

Used by ``dispatch._dispatch_one``. Isinstance-, module- and status-based,
walking the full ``__cause__`` / ``__context__`` chain.
"""

from __future__ import annotations

from maps.perception.sam.detect_client import SamServerOutage


def _chain(exc: BaseException):
    """``exc`` and every exception below it, each once."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def _exc_is_sam_infra(exc: BaseException) -> bool:
    """True if any link in the chain is a ``SamServerOutage``.

    A flight-time SAM ``/segment`` outage (server unreachable / 5xx /
    terminal None after retries) is an infrastructure casualty, not a
    grounding miss — the frame produced NO usable detections for a reason
    the algorithm never controlled. Isinstance-based so it never catches a
    real empty-result frame, which never raises.
    """
    return any(isinstance(cur, SamServerOutage) for cur in _chain(exc))


# HTTP-client modules that ONLY talk to the shared inference services (vLLM /
# SAM). An exception whose type lives in one of these is a transport fault to
# a shared server. Constrained to these modules on purpose: a bare builtin
# ``TimeoutError`` / ``ConnectionError`` / ``OSError`` can be raised by
# algorithm or file code, and misclassifying one as infra would drop a real
# CityNav failure from SR.
_HTTP_CLIENT_MODULES = ("httpx", "httpcore", "requests", "urllib3", "aiohttp")
# 4xx statuses a serve may clear on its own (request timeout, rate limit).
# Every other status below 500 is the request's own fault.
_TRANSIENT_CLIENT_STATUSES = (408, 429)


def _exc_is_transient_infra(exc: BaseException) -> bool:
    """True iff a link in the chain is a transport fault to a shared
    inference server: its type is from an HTTP-client module, and it carries
    no HTTP status below 500 other than ``_TRANSIENT_CLIENT_STATUSES``."""
    for cur in _chain(exc):
        if (type(cur).__module__ or "").split(".", 1)[0] not in _HTTP_CLIENT_MODULES:
            continue
        status = getattr(getattr(cur, "response", None), "status_code", None)
        return not (
            isinstance(status, int)
            and status < 500
            and status not in _TRANSIENT_CLIENT_STATUSES
        )
    return False
