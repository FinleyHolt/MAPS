"""The chat transport's retries split by cause: 5xx and connection faults burn ``_MAX_RETRIES``, 429/402
wait on ``_RATE_LIMIT_BUDGET_S``, any other 4xx raises at once."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from maps.llm import http as http_mod
from tests.serves import ChatServe

_MESSAGES = [{"role": "user", "content": "hi"}]


def _err(status: int, headers: dict | None = None, payload: dict | None = None):
    return status, payload or {"error": "boom"}, headers or {}


@pytest.fixture
def transport(monkeypatch):
    """Run ``_chat_completions_raw`` against a ChatServe scripted with ``answers``; returns
    ``(result_or_exc, bodies, sleeps)`` with the backoff sleeps captured, not taken."""
    monkeypatch.delenv("SEARCH_COMPILE_WORKDIR", raising=False)
    sleeps: list[float] = []
    monkeypatch.setattr(http_mod.time, "sleep", sleeps.append)

    def _run(answers, **kwargs):
        with ChatServe(answers) as serve:
            try:
                result = http_mod._chat_completions_raw(
                    f"{serve.url}/v1", "key", model="m", messages=_MESSAGES, timeout=60.0, **kwargs)
            except Exception as exc:  # noqa: BLE001 - the test inspects it
                result = exc
        return result, serve.bodies, sleeps

    return _run


def test_rate_limit_is_retried_and_honours_retry_after(transport) -> None:
    result, calls, sleeps = transport([_err(429, {"retry-after": "3"})])
    assert not isinstance(result, Exception)
    assert len(calls) == 2
    assert sleeps == [3.0]


def test_rate_limit_wait_is_capped(transport) -> None:
    _result, _calls, sleeps = transport([_err(429, {"retry-after": "9999"})])
    assert sleeps == [http_mod._RATE_LIMIT_MAX_BACKOFF_S]


def test_rate_limit_budget_is_separate_from_transient_budget(transport) -> None:
    """A permanent 429 waits until ``_RATE_LIMIT_BUDGET_S`` is spent, far past
    ``_MAX_RETRIES`` (2), then raises: a saturated limit costs wall clock, not the episode."""
    result, calls, sleeps = transport([_err(429)] * 200)
    assert isinstance(result, httpx.HTTPStatusError)
    assert result.response.status_code == 429
    assert len(calls) == len(sleeps) + 1 > 20
    assert http_mod._RATE_LIMIT_BUDGET_S - http_mod._RATE_LIMIT_MAX_BACKOFF_S < sum(sleeps)
    assert sum(sleeps) <= http_mod._RATE_LIMIT_BUDGET_S


def test_azure_s_millisecond_retry_after_wins(transport) -> None:
    _result, calls, sleeps = transport([_err(429, {"retry-after-ms": "2500", "retry-after": "3"})])
    assert (len(calls), sleeps) == (2, [2.5])


def test_a_retry_now_answer_still_spends_the_budget(transport) -> None:
    """``retry-after: 0`` forever would loop on a time budget; each wait is floored."""
    result, _calls, sleeps = transport([_err(429, {"retry-after": "0"})] * 2000)
    assert isinstance(result, httpx.HTTPStatusError)
    assert min(sleeps) == 1.0


def test_quota_status_is_retried_too(transport) -> None:
    result, calls, _sleeps = transport([_err(402)])
    assert not isinstance(result, Exception)
    assert len(calls) == 2


def test_malformed_request_raises_without_retry(transport) -> None:
    result, calls, sleeps = transport([_err(400)])
    assert isinstance(result, httpx.HTTPStatusError)
    assert result.response.status_code == 400
    assert len(calls) == 1
    assert sleeps == []


def _posted_body(transport, **kwargs) -> dict:
    """The body ``_chat_completions_raw`` actually puts on the wire."""
    return transport([], **kwargs)[1][0]


def test_an_arm_naming_no_effort_puts_no_effort_key_on_the_wire(monkeypatch, transport) -> None:
    """Unset carries no key at all (not a null), and the named effort is the only difference on the wire."""
    from maps.llm.model_cards import sampling_kwargs

    monkeypatch.delenv("SEARCH_SAMPLING_REASONING_EFFORT", raising=False)
    quiet = _posted_body(transport, **sampling_kwargs("inkling-small"))
    assert "reasoning_effort" not in quiet
    assert "reasoning_effort" not in json.dumps(quiet)

    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", "0.99")
    named = _posted_body(transport, **sampling_kwargs("inkling-small"))
    assert named["reasoning_effort"] == "0.99"
    assert '"reasoning_effort": "0.99"' in json.dumps(named)
    assert {k: v for k, v in named.items() if k != "reasoning_effort"} == quiet


def test_rejected_reasoning_effort_is_named_in_the_refusal(transport, caplog) -> None:
    """A 400 on an effort field raises at once, and the request summary names the key and value."""
    with caplog.at_level(logging.ERROR, logger="maps.llm.http"):
        result, calls, sleeps = transport(
            [_err(400, payload={"error": "unrecognized field reasoning_effort"})], reasoning_effort="0.99")

    assert isinstance(result, httpx.HTTPStatusError)
    assert len(calls) == 1
    assert sleeps == []
    summaries = [r.getMessage() for r in caplog.records if "request summary" in r.getMessage()]
    assert len(summaries) == 1
    assert "'reasoning_effort': '0.99'" in summaries[0]


def test_transient_5xx_still_uses_the_fault_budget(transport) -> None:
    result, calls, sleeps = transport([_err(503)])

    assert not isinstance(result, Exception)
    assert len(calls) == 2
    assert len(sleeps) == 1


def test_permanent_5xx_exhausts_the_fault_budget(transport) -> None:
    result, calls, _sleeps = transport([_err(503)] * 5)

    assert isinstance(result, httpx.HTTPStatusError)
    assert len(calls) == 3


class _RaisingClient:
    """Raises a scripted exception per ``post``, recording the call count."""

    def __init__(self, exc: Exception, calls: list[int]) -> None:
        self._exc = exc
        self._calls = calls
        self.is_closed = False

    def __enter__(self) -> "_RaisingClient":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def post(self, _url: str, headers=None, json=None):
        self._calls.append(len(self._calls))
        raise self._exc

    def close(self) -> None:
        self.is_closed = True


def _run_raising(monkeypatch, exc: Exception) -> list[int]:
    monkeypatch.delenv("SEARCH_COMPILE_WORKDIR", raising=False)
    monkeypatch.setattr(http_mod.time, "sleep", lambda _s: None)
    calls: list[int] = []
    monkeypatch.setattr(
        http_mod.httpx, "Client", lambda *a, **kw: _RaisingClient(exc, calls)
    )
    with pytest.raises(httpx.TimeoutException):
        http_mod._chat_completions_raw(
            "http://endpoint/v1",
            "key",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            timeout=60.0,
        )
    return calls


def test_read_timeout_is_dispatched_once_and_never_re_sent(monkeypatch) -> None:
    """The serve still holds a timed-out request, so a re-send would queue a duplicate behind it."""
    req = httpx.Request("POST", "http://endpoint/v1/chat/completions")
    for exc in (
        httpx.ReadTimeout("read", request=req),
        httpx.WriteTimeout("write", request=req),
        httpx.PoolTimeout("pool", request=req),
    ):
        assert _run_raising(monkeypatch, exc) == [0], type(exc).__name__


def test_connect_timeout_still_burns_the_fault_budget(monkeypatch) -> None:
    """Nothing was accepted, so a retry duplicates no work."""
    req = httpx.Request("POST", "http://endpoint/v1/chat/completions")
    calls = _run_raising(monkeypatch, httpx.ConnectTimeout("connect", request=req))
    assert calls == [0, 1, 2]


def test_every_call_names_its_timeout() -> None:
    with pytest.raises(TypeError, match="timeout"):
        http_mod._chat_completions_raw(
            "http://endpoint/v1", "key", model="m", messages=[{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
def test_a_404_for_a_model_no_longer_served_is_named_on_both_paths(transport, monkeypatch, stream) -> None:
    """A serve restarted behind the URL with another model answers 404; the re-probe names the change."""
    import threading

    from maps.llm import endpoint_spec

    monkeypatch.setattr(endpoint_spec, "_probe_models", lambda _url: [{"id": "another-model"}])
    result, calls, _sleeps = transport([_err(404)], **({"cancel_event": threading.Event()} if stream else {}))
    assert isinstance(result, RuntimeError), result
    assert "served-model mismatch" in str(result) and "another-model" in str(result)
    assert len(calls) == 1


@pytest.mark.parametrize("tail,finish", [
    (b"", None),
    (b"data: [DONE]\n\n", "stop"),
    (b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n', "length"),
], ids=["cut", "done", "finish"])
def test_a_stream_cut_before_its_end_is_a_transport_fault(monkeypatch, tail, finish) -> None:
    """A stream that closes with neither a finish_reason nor ``[DONE]`` was cut short: it raises
    (and is retried) instead of reading as a whole answer that stopped."""
    sse = b'data: {"choices":[{"delta":{"content":"half"}}]}\n\n' + tail
    real = httpx.Client
    mock = httpx.MockTransport(lambda _r: httpx.Response(200, content=sse))
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=mock, **kw))
    if finish is None:
        with pytest.raises(httpx.RemoteProtocolError, match="without a finish_reason"):
            http_mod._stream_chat_completions("http://e/v1/chat/completions", {}, {}, timeout=5.0)
        return
    out = http_mod._stream_chat_completions("http://e/v1/chat/completions", {}, {}, timeout=5.0)
    assert out["choices"][0]["finish_reason"] == finish
    assert out["choices"][0]["message"]["content"] == "half"
