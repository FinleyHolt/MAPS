"""Cold-start reasoner warmup (grammar/CUDA-graph pre-pay) and the
pre-flight health sidecar.
"""

from __future__ import annotations

import json
import os
import time
import logging

log = logging.getLogger("run_benchmark")


class ReasonerUnreachableError(RuntimeError):
    """Raised when the reasoner warmup probe exhausts its retry budget."""


def _reasoner_warmup_grammar(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_s: float | None = None,
) -> None:
    """Pre-pay stage-1 cold-start costs with representative requests.

    The API probe only proves the HTTP app is reachable. The first real
    compile pays xgrammar setup, thinking-mode decode, and CUDA graph
    capture at the reasoner's request shape. Warm that shape here, without
    disabling thinking or adding a tiny ``max_tokens`` cap that would
    exercise a different path from the benchmark calls.
    """
    import concurrent.futures

    import httpx

    try:
        from maps.llm.sampling import thinking_sampling_kwargs
        from maps.search.parse_compile.prompts import served_tools
    except Exception as exc:
        log.warning("grammar/cudagraph warmup skipped (import failed: %s)", exc)
        return

    if timeout_s is None:
        try:
            timeout_s = float(os.environ.get("SEARCH_REASONER_WARMUP_TIMEOUT_S", "120"))
        except ValueError:
            timeout_s = 120.0
    timeout_s = max(1.0, float(timeout_s))
    io_timeout_s = min(15.0, timeout_s)
    request_timeout = httpx.Timeout(
        connect=io_timeout_s,
        read=timeout_s,
        write=io_timeout_s,
        pool=io_timeout_s,
    )
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    thinking_samp = thinking_sampling_kwargs(temperature=1.0)
    # WARM THE SHAPE THE BENCH ACTUALLY SENDS. That is CODE_TOOLS ([run])
    # with tool_choice "auto" and NO decoder mask -- reasoner_loop states
    # why the mission path carries none: a say mask built over CODE_TOOL
    # names the wrong tool, the wrong argument and a grammar this surface
    # does not speak. Warming any other shape pre-pays an xgrammar setup no
    # request uses and leaves the real one to the first live call -- which
    # is the entire cost this function exists to avoid.
    _warm_msg = {
        "role": "user",
        "content": (
            "Warmup only. In <think>, write exactly one short "
            "sentence. Then call one available tool with minimal "
            "valid arguments (empty arrays where allowed)."
        ),
    }
    warm_body = {
        "model": model,
        "messages": [_warm_msg],
        **thinking_samp,
        "tools": served_tools(),
        "tool_choice": "auto",
    }

    def _post_warmup(request_body: dict) -> str:
        deadline = time.perf_counter() + timeout_s
        raw = bytearray()
        try:
            with httpx.stream(
                "POST",
                url,
                headers=headers,
                json=request_body,
                timeout=request_timeout,
            ) as r:
                r.raise_for_status()
                for chunk in r.iter_bytes():
                    if chunk:
                        raw.extend(chunk)
                    if time.perf_counter() >= deadline:
                        return f"wall_timeout>{timeout_s:.0f}s"
        except (httpx.HTTPError, OSError) as exc:
            return f"{type(exc).__name__}: {exc}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return f"bad_json_response: {exc}"
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return "bad_response: no choices"
        choice = choices[0] if isinstance(choices[0], dict) else {}
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            return "length_finish"
        if finish_reason not in (None, "stop", "tool_calls"):
            return f"finish_reason={finish_reason!r}"
        message = choice.get("message")
        if not isinstance(message, dict):
            return "bad_response: no message"
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        has_text = isinstance(content, str) and bool(content.strip())
        has_calls = isinstance(tool_calls, list) and bool(tool_calls)
        if not has_text and not has_calls:
            return "empty_content"
        return "ok"

    warm_bodies = [warm_body]
    n_total = len(warm_bodies)
    for attempt in (1, 2):
        tt0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_total) as ex:
            thinking_results = list(ex.map(_post_warmup, warm_bodies))
        thinking_elapsed = time.perf_counter() - tt0
        thinking_ok = sum(1 for r in thinking_results if r == "ok")
        if thinking_ok == n_total:
            log.info(
                "thinking grammar/cudagraph warmup ok: %d/%d concurrent "
                "tool-call-shaped calls in %.1fs (attempt=%d)",
                thinking_ok,
                n_total,
                thinking_elapsed,
                attempt,
            )
            break
        log.warning(
            "thinking grammar/cudagraph warmup partial: %d/%d ok in %.1fs; "
            "attempt=%d/2 errors=%s%s",
            thinking_ok,
            n_total,
            thinking_elapsed,
            attempt,
            [r for r in thinking_results if r != "ok"][:3],
            "; retrying once" if attempt == 1 else "; run continues",
        )
        if attempt == 2:
            break

    # Vision-path warm: shaped like a ``view{crops}`` delivery so the vision
    # encoder and the image-bearing decode CUDA graph are paid before the
    # reasoner's first crop arrives. That turn emits free-text prose (no
    # response_format), so this warmup sends plain text too — no
    # schema-grammar warm is needed.
    try:
        import base64 as _b64
        import io as _io

        from PIL import Image as _Image
    except Exception as exc:
        log.warning("vision warmup skipped (import failed: %s)", exc)
        return
    _buf = _io.BytesIO()
    _Image.new("RGB", (1024, 1024), (127, 127, 127)).save(_buf, "JPEG", quality=85)
    _img_b64 = _b64.b64encode(_buf.getvalue()).decode("ascii")
    vision_samp = thinking_sampling_kwargs(temperature=1.0)
    vbody = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Warmup only. In <think>, write exactly one short "
                            "sentence. After </think>, emit one short sentence "
                            "in prose."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_img_b64}"},
                    },
                ],
            }
        ],
        **vision_samp,
    }
    vt0 = time.perf_counter()
    vresult = _post_warmup(vbody)
    if vresult == "ok":
        log.info(
            "vision/cudagraph warmup ok: 1 per-stop-vision-shaped image call "
            "in %.1fs (free-text prose, no schema)",
            time.perf_counter() - vt0,
        )
    else:
        log.warning(
            "vision warmup failed in %.1fs (%s); run continues, first "
            "the reasoner's first crop may pay the cold vision cost",
            time.perf_counter() - vt0,
            vresult,
        )


def _reasoner_warmup_ping(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_s: float = 30.0,
    poll_every_s: float = 15.0,
    total_wait_s: float = 0.0,
    pre_pay_cold_start: bool = True,
) -> None:
    """Probe the endpoint until it answers, then run representative warmups.

    The probe uses ``/models`` for reachability only. Once the endpoint is
    reachable, ``_reasoner_warmup_grammar`` pre-pays the cold model/grammar/
    CUDA-graph path with thinking-mode requests shaped like benchmark calls.
    Pre-flight compiles before this succeeds have no chance of finalizing,
    so the warmup blocks rather than letting the drone race past it.

    ``pre_pay_cold_start=False`` keeps the reachability probe but skips the
    grammar/vision warmup. A hosted multi-tenant endpoint has no engine
    cold start to pre-pay, so those tool-schema completions and the
    1024x1024 image call buy nothing and are billed per token.

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
        if pre_pay_cold_start:
            _reasoner_warmup_grammar(base_url=base_url, model=model, api_key=api_key)
        else:
            log.info(
                "grammar/vision warmup skipped: %s serves a hosted endpoint "
                "with no engine cold start to pre-pay",
                base_url,
            )
        return
