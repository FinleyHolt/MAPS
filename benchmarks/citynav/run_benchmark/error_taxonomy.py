"""Exception classifiers that separate infrastructure casualties (excluded from
the SR denominator and re-queued) from genuine algorithm failures (scored misses).

Used by the episode dispatch / error-handling paths in ``engine``. Isinstance-
and marker-string-based, walking the full ``__cause__`` / ``__context__`` chain.
"""

from __future__ import annotations

from maps.perception.sam.detect_client import SamServerOutage


# Substrings that identify a CUDA/CuPy device-memory-exhaustion failure
# anywhere in an exception chain. Mirrors bench_fullset._OOM_MARKERS so a
# mid-flight contention OOM is a recognized infra abort, not a silent
# completed-zero.
_CUDA_OOM_MARKERS = (
    "OutOfMemoryError",
    "CUDA out of memory",
    "CUDA error: out of memory",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUDNN_STATUS_ALLOC_FAILED",
)


def _exc_is_cuda_oom(exc: BaseException) -> bool:
    """True if any link in the exception chain is a CUDA/CuPy OOM."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        qualname = f"{type(cur).__module__}.{type(cur).__qualname__}"
        if "OutOfMemoryError" in qualname or any(
            m in str(cur) for m in _CUDA_OOM_MARKERS
        ):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _exc_is_sam_infra(exc: BaseException) -> bool:
    """True if any link in the chain is a ``SamServerOutage``.

    A flight-time SAM ``/segment`` outage (server unreachable / 5xx /
    terminal None after retries) is an infrastructure casualty, not a
    grounding miss — the frame produced NO usable detections for a reason
    the algorithm never controlled. Without this the episode would record a
    completed scored-zero indistinguishable from a genuine no-detections
    flight, silently flooring the fleet SR. Isinstance-based so it never
    catches a real empty-result frame, which never raises.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, SamServerOutage):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


# Transient infrastructure faults that are NOT the algorithm's doing: a
# shared server (vLLM / SAM / FlightSim render) momentarily unreachable,
# 5xx, connection reset, or a read/connect timeout under contention. An
# episode that dies on one of these never got a fair run, so — like the
# OOM path — it is excluded from the SR denominator and fully
# re-queued, not recorded as a completed grounding zero. A genuine
# algorithm bug (KeyError/assert/etc. in the packages) is deliberately
# NOT matched here: that is the algorithm's failure and stays visible.
# HTTP-client modules that ONLY talk to the shared inference services
# (vLLM / SAM). An exception whose type lives in
# one of these is a transport fault to a shared server — infra, never
# the algorithm. Constrained to these modules on purpose: a bare
# builtin ``TimeoutError`` / ``ConnectionError`` / ``OSError`` can be
# raised by algorithm or file code for non-infra reasons, and
# misclassifying one as infra would silently drop a real CityNav
# failure and inflate SR. CityNav-compliant rule: algorithm fault =
# scored miss; only an external shared-server transport fault is
# excluded.
_HTTP_CLIENT_MODULES = ("httpx", "httpcore", "requests", "urllib3", "aiohttp")
_TRANSIENT_INFRA_MARKERS = (
    "Server disconnected",
    "Connection reset by peer",
    "Connection refused",
    "Remote end closed connection",
    "peer closed connection",
    "503 Service Unavailable",
    "502 Bad Gateway",
    "504 Gateway Timeout",
)


def _exc_is_transient_infra(exc: BaseException) -> bool:
    """True iff a link in the chain is a transport fault to a shared
    inference server (its exception type is from an HTTP-client module,
    or the message is an explicit server-transport phrase). CUDA-OOM
    faults are classified by their own predicate first.
    Deliberately strict: a generic builtin error is NOT infra, so a
    real algorithm failure is never excluded from SR."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        mod = (type(cur).__module__ or "").split(".", 1)[0]
        if mod in _HTTP_CLIENT_MODULES or any(
            m in str(cur) for m in _TRANSIENT_INFRA_MARKERS
        ):
            return True
        cur = cur.__cause__ or cur.__context__
    return False
