"""Retry behaviour of the chat-completions transport.

The transport splits retries by cause: transient faults (5xx, connection
errors) burn ``max_retries``; rate-limit and quota responses (429/402)
draw on the separate ``max_rate_limit_retries`` budget; every other 4xx
is a malformed request and raises at once. These tests pin that split,
because a hosted endpoint rate-limits routinely and a single shared
budget would end an episode mid-compile.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from maps.llm import http as http_mod


_OK_BODY = {
    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}


class _FakeResponse:
    """Duck-typed stand-in for ``httpx.Response``.

    ``HTTPStatusError`` stores whatever ``response`` it is handed, so the
    transport's status/header/body reads work against this without a live
    server.
    """

    def __init__(
        self,
        status_code: int,
        body: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body if body is not None else {"error": "boom"}
        self.headers = headers or {}
        self.text = json.dumps(self._body)

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("POST", "http://endpoint/v1/chat/completions"),
                response=self,  # type: ignore[arg-type]
            )


class _FakeClient:
    """Yields a scripted response per ``post``, recording the call count."""

    def __init__(self, script: list[_FakeResponse], calls: list[int]) -> None:
        self._script = script
        self._calls = calls
        self.is_closed = False

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def post(self, _url: str, headers=None, json=None) -> _FakeResponse:
        idx = len(self._calls)
        self._calls.append(idx)
        return self._script[min(idx, len(self._script) - 1)]

    def close(self) -> None:
        self.is_closed = True


@pytest.fixture
def transport(monkeypatch):
    """Drive ``_chat_completions_raw`` against a scripted response list.

    Returns a callable taking the script and returning
    ``(result_or_exc, calls, sleeps)``. Sleeps are captured rather than
    taken so the rate-limit backoff is asserted without real waiting.
    """
    monkeypatch.delenv("SEARCH_LLM_CACHE_MODE", raising=False)
    # No workdir means no mission to record -- the flight log is derived
    # from it and there is no separate switch to unset.
    monkeypatch.delenv("SEARCH_COMPILE_WORKDIR", raising=False)
    monkeypatch.delenv("SEARCH_LLM_DUMP_4XX_DIR", raising=False)
    monkeypatch.delenv("SEARCH_STREAM_TOKENS", raising=False)

    sleeps: list[float] = []
    monkeypatch.setattr(http_mod.time, "sleep", lambda s: sleeps.append(s))

    def _run(script: list[_FakeResponse], **kwargs):
        calls: list[int] = []
        monkeypatch.setattr(
            http_mod.httpx,
            "Client",
            lambda *a, **kw: _FakeClient(script, calls),
        )
        try:
            result = http_mod._chat_completions_raw(
                "http://endpoint/v1",
                "key",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            result = exc
        return result, calls, sleeps

    return _run


def test_rate_limit_is_retried_and_honours_retry_after(transport) -> None:
    script = [
        _FakeResponse(429, headers={"retry-after": "3"}),
        _FakeResponse(200, _OK_BODY),
    ]
    result, calls, sleeps = transport(script)

    assert not isinstance(result, Exception)
    assert len(calls) == 2
    assert sleeps == [3.0]


def test_rate_limit_wait_is_capped(transport) -> None:
    script = [
        _FakeResponse(429, headers={"retry-after": "9999"}),
        _FakeResponse(200, _OK_BODY),
    ]
    _result, _calls, sleeps = transport(script)

    assert sleeps == [http_mod._RATE_LIMIT_MAX_BACKOFF_S]


def test_rate_limit_budget_is_separate_from_transient_budget(transport) -> None:
    """A 429 storm must not be capped by ``max_retries``.

    With ``max_retries=2`` and ``max_rate_limit_retries=5`` a permanent
    429 is sent 6 times (initial + 5 waits) — proof the two budgets do
    not share a counter.
    """
    result, calls, sleeps = transport(
        [_FakeResponse(429)], max_retries=2, max_rate_limit_retries=5
    )

    assert isinstance(result, httpx.HTTPStatusError)
    assert result.response.status_code == 429
    assert len(calls) == 6
    assert len(sleeps) == 5


def test_quota_status_is_retried_too(transport) -> None:
    result, calls, _sleeps = transport(
        [_FakeResponse(402), _FakeResponse(200, _OK_BODY)]
    )

    assert not isinstance(result, Exception)
    assert len(calls) == 2


def test_malformed_request_raises_without_retry(transport) -> None:
    result, calls, sleeps = transport([_FakeResponse(400)])

    assert isinstance(result, httpx.HTTPStatusError)
    assert result.response.status_code == 400
    assert len(calls) == 1
    assert sleeps == []


class _RecordingClient(_FakeClient):
    """A ``_FakeClient`` that keeps the bodies it was handed."""

    def __init__(self, script: list[_FakeResponse], calls: list[int]) -> None:
        super().__init__(script, calls)
        self.bodies: list[dict] = []

    def post(self, _url: str, headers=None, json=None) -> _FakeResponse:
        self.bodies.append(json or {})
        return super().post(_url, headers=headers, json=json)


def _posted_body(monkeypatch, **kwargs) -> dict:
    """The body ``_chat_completions_raw`` actually puts on the wire."""
    monkeypatch.delenv("SEARCH_COMPILE_WORKDIR", raising=False)
    client = _RecordingClient([_FakeResponse(200, _OK_BODY)], [])
    monkeypatch.setattr(http_mod.httpx, "Client", lambda *a, **kw: client)
    http_mod._chat_completions_raw(
        "http://endpoint/v1",
        "key",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        **kwargs,
    )
    return client.bodies[0]


def test_an_arm_naming_no_effort_puts_no_effort_key_on_the_wire(
    monkeypatch,
) -> None:
    """Unset carries NO key — not the key at a null, the key absent.

    Every arm flying today names no effort, and their requests have to
    stay exactly what they were: a body carrying ``reasoning_effort:
    null`` would be a new field on the wire for a serve that never saw
    one, and a served default would be overridden by a value no arm
    asked for. The two bodies below differ by that single key and by
    nothing else, which is the whole footprint of the knob.
    """
    from maps.llm.sampling import thinking_sampling_kwargs

    monkeypatch.delenv("SEARCH_SAMPLING_REASONING_EFFORT", raising=False)
    quiet = _posted_body(monkeypatch, **thinking_sampling_kwargs(temperature=1.0, model_id="qwen36-35b-a3b"))
    assert "reasoning_effort" not in quiet
    assert "reasoning_effort" not in json.dumps(quiet)

    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", "0.99")
    named = _posted_body(monkeypatch, **thinking_sampling_kwargs(temperature=1.0, model_id="qwen36-35b-a3b"))
    assert named["reasoning_effort"] == "0.99"
    assert '"reasoning_effort": "0.99"' in json.dumps(named)
    assert {k: v for k, v in named.items() if k != "reasoning_effort"} == quiet


def test_rejected_reasoning_effort_is_named_in_the_refusal(
    transport, caplog
) -> None:
    """A serve whose template carries no effort field 400s on one, and
    the log has to say which field.

    The arm names an effort the card does not take: the transport raises
    at once (a malformed request, not a capacity event — no retry, and no
    quiet re-send without the field), and the request summary carries the
    key and the value that caused it. Absent the key on that line the
    operator reads a bare 400 beside a summary of five sampling fields,
    none of them the one they changed.
    """
    with caplog.at_level(logging.ERROR, logger="maps.llm.http"):
        result, calls, sleeps = transport(
            [_FakeResponse(400, {"error": "unrecognized field reasoning_effort"})],
            reasoning_effort="0.99",
        )

    assert isinstance(result, httpx.HTTPStatusError)
    assert len(calls) == 1
    assert sleeps == []
    summaries = [
        r.getMessage() for r in caplog.records if "request summary" in r.getMessage()
    ]
    assert len(summaries) == 1
    assert "'reasoning_effort': '0.99'" in summaries[0]


def test_transient_5xx_still_uses_the_fault_budget(transport) -> None:
    result, calls, sleeps = transport(
        [_FakeResponse(503), _FakeResponse(200, _OK_BODY)]
    )

    assert not isinstance(result, Exception)
    assert len(calls) == 2
    assert len(sleeps) == 1


def test_permanent_5xx_exhausts_the_fault_budget(transport) -> None:
    result, calls, _sleeps = transport([_FakeResponse(503)], max_retries=2)

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
            max_retries=2,
        )
    return calls


def test_read_timeout_is_dispatched_once_and_never_re_sent(monkeypatch) -> None:
    """A read timeout must NOT be retried.

    The server still holds the request (the blocking POST cannot abort it),
    so a re-send is a duplicate queued behind the original -- and it fires
    exactly when the server is saturated. That feedback loop is what took
    one reasoner serve to 0% GPU at workers=128.
    """
    req = httpx.Request("POST", "http://endpoint/v1/chat/completions")
    for exc in (
        httpx.ReadTimeout("read", request=req),
        httpx.WriteTimeout("write", request=req),
        httpx.PoolTimeout("pool", request=req),
    ):
        assert _run_raising(monkeypatch, exc) == [0], type(exc).__name__


def test_connect_timeout_still_burns_the_fault_budget(monkeypatch) -> None:
    """The opposite case: nothing was accepted, so retrying duplicates no work."""
    req = httpx.Request("POST", "http://endpoint/v1/chat/completions")
    calls = _run_raising(monkeypatch, httpx.ConnectTimeout("connect", request=req))
    assert calls == [0, 1, 2]
