"""The pre-flight reasoner reachability probe."""

from __future__ import annotations

import time
import logging

log = logging.getLogger("run_benchmark")


class ReasonerUnreachableError(RuntimeError):
    """Raised when the reasoner warmup probe exhausts its retry budget."""


def _reasoner_warmup_ping(
    *,
    base_url: str,
    api_key: str,
    timeout_s: float = 30.0,
    poll_every_s: float = 15.0,
    total_wait_s: float = 0.0,
) -> None:
    """Probe the endpoint's ``/models`` until it answers, so the mission's
    first turn never races an unreachable reasoner.

    ``total_wait_s <= 0`` means wait indefinitely (operator can SIGINT
    when they give up). A positive value caps the wait and raises
    ``ReasonerUnreachableError`` on exhaustion.
    """
    import httpx

    url = base_url.rstrip("/") + "/models"
    headers: dict[str, str] = {}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    bounded = float(total_wait_s) > 0.0
    t0 = time.perf_counter()
    attempts = 0
    last_heartbeat_at = 0.0
    while True:
        attempts += 1
        try:
            r = httpx.get(url, headers=headers, timeout=timeout_s)
            r.raise_for_status()
        except (httpx.HTTPError, OSError) as exc:
            elapsed = time.perf_counter() - t0
            if bounded and elapsed + poll_every_s > total_wait_s:
                log.error(
                    "reasoner warmup unreachable after %.0fs / %d attempts; "
                    "aborting pre-flight (last_error=%s)",
                    elapsed,
                    attempts,
                    exc,
                )
                raise ReasonerUnreachableError(
                    f"reasoner warmup failed after {elapsed:.0f}s "
                    f"({attempts} attempts); last_error={type(exc).__name__}: {exc}"
                ) from exc
            heartbeat_due = elapsed - last_heartbeat_at >= 60.0
            if attempts <= 3 or heartbeat_due:
                log.warning(
                    "reasoner warmup attempt %d failed (%s); retrying in %.0fs "
                    "(elapsed=%.0fs%s; standing by for tunnel)",
                    attempts,
                    exc,
                    poll_every_s,
                    elapsed,
                    f"/{total_wait_s:.0f}s" if bounded else "",
                )
                last_heartbeat_at = elapsed
            time.sleep(poll_every_s)
            continue
        elapsed = time.perf_counter() - t0
        log.info("LLM API probe ok in %.2fs (attempts=%d)", elapsed, attempts)
        return
