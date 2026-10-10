"""Launcher preflight: the exit-2 endpoint validation both launchers run
before spawning a runner.

``launch.py`` (single episode) and ``bench_fullset.py`` (full split)
validate the same endpoints the same way: probe the reasoner, SELECT the
live SAM replica subset of ``SAM_SERVER_URL`` and write it back to
``os.environ``, then VALIDATE the selected set with
``maps.llm.endpoint_spec.discover_from_env`` -- the same fail-loud probe
the runner re-runs at its own startup.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maps.llm.endpoint_spec import EndpointSpec

DEFAULT_REASONER_URL = "http://127.0.0.1:8001/v1"
DEFAULT_SAM_URL = "http://127.0.0.1:8772"


@dataclass(frozen=True)
class ProbeResult:
    """One HTTP probe's outcome. ``status`` carries the HTTP code whenever
    the server ANSWERED -- a 401 is a live-but-misconfigured endpoint, not
    an unreachable one. ``detail`` holds the connection-level error when
    nothing answered."""

    ok: bool
    status: int | None = None
    detail: str = ""


def probe_http(url: str, timeout_s: float = 3.0, bearer: str | None = None) -> ProbeResult:
    """GET ``url``; 2xx/3xx is ok. Never raises."""
    try:
        req = urllib.request.Request(url)
        if bearer:
            req.add_header("Authorization", f"Bearer {bearer}")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return ProbeResult(200 <= resp.status < 400, resp.status)
    except urllib.error.HTTPError as exc:
        return ProbeResult(False, exc.code, f"HTTP {exc.code}")
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        reason = getattr(exc, "reason", None)
        return ProbeResult(False, None, str(reason or exc))


def _reasoner_url() -> str:
    raw = os.environ.get("SEARCH_REASONER_URL", DEFAULT_REASONER_URL) or DEFAULT_REASONER_URL
    return raw.strip().rstrip("/")


def _reasoner_probe(timeout_s: float = 10.0) -> ProbeResult:
    """``/models`` on ``SEARCH_REASONER_URL``, with the chat path's bearer
    (hosted endpoints auth ``/v1/models``) and room for a cold TLS handshake."""
    return probe_http(
        f"{_reasoner_url()}/models",
        timeout_s=timeout_s,
        bearer=os.environ.get("SEARCH_REASONER_API_KEY") or None,
    )


def reasoner_alive() -> bool:
    """True when the reasoner endpoint answers ``/models``."""
    return _reasoner_probe().ok


def _fail(msg: str) -> None:
    print(f"[preflight] FAIL: {msg}", file=sys.stderr)
    sys.exit(2)


def preflight_services() -> EndpointSpec:
    """Endpoint discovery preflight. Exit 2 with a reason on any failure;
    on success return the validated ``EndpointSpec`` (served model, window,
    service URLs) so the caller records the measured facts instead of
    probing again.

    The reasoner probe is readiness-only: ONE URL (the engine load-balances
    its own replicas behind it). SELECTION applies to SAM: the replicas of
    ``SAM_SERVER_URL`` that answer ``/health`` are written back to
    ``os.environ`` so the runner subprocess inherits exactly the set that
    was validated. VALIDATION is ``maps.llm.endpoint_spec.discover`` --
    served model + window facts off ``/v1/models``, tokenizer load, every
    service ``/health`` with its identity check."""
    print("[preflight] discovering endpoints...", file=sys.stderr)

    r = _reasoner_probe()
    if not r.ok:
        if r.status in (401, 403):
            _fail(
                f"the reasoner rejected the bearer key at {_reasoner_url()} "
                f"(HTTP {r.status} on /models) -- the endpoint is up; fix "
                "SEARCH_REASONER_API_KEY"
            )
        _fail(f"reasoner unreachable at {_reasoner_url()}: {r.detail or r.status}")

    raw = os.environ.get("SAM_SERVER_URL", DEFAULT_SAM_URL)
    urls = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    alive = [u for u in urls if probe_http(f"{u}/health").ok]
    if not alive:
        _fail(f"SAM unreachable at {raw}")
    os.environ["SAM_SERVER_URL"] = ",".join(alive)
    print(f"[preflight] SAM at {os.environ['SAM_SERVER_URL']}", file=sys.stderr)

    from maps.llm.endpoint_spec import EndpointDiscoveryError, discover_from_env

    try:
        spec = discover_from_env()
    except EndpointDiscoveryError as exc:
        _fail(f"endpoint discovery: {exc}")
    print(
        f"[preflight] reasoner serves {spec.model_id} "
        f"(max_model_len={spec.max_model_len}) at {spec.base_url}",
        file=sys.stderr,
    )
    print("[preflight] all services ready", file=sys.stderr)
    return spec
