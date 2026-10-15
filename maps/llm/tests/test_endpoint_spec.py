#!/usr/bin/env python
"""Fail-loud taxonomy + budget arithmetic of ``maps.llm.endpoint_spec``.

Offline + hermetic: ``_get_json`` (the one HTTP seam) is monkeypatched with a
url->payload table, and ``_require_tokenizer`` is stubbed except in the
tokenizer-failure test (which drives the REAL ``_require_tokenizer`` against a
stubbed ``maps.llm.tokenizer._tokenizer``). Every abort case a-i is a
distinct ``EndpointDiscoveryError`` naming the fact; the install-once authority
is exercised on an isolated module global.

Run: ``pytest maps/llm/tests/test_endpoint_spec.py``.
"""
from __future__ import annotations

import http.client
import urllib.error

import pytest

from maps.llm import endpoint_spec as E
from maps.llm.endpoint_spec import (
    EndpointDiscoveryError,
    EndpointOverride,
    EndpointSpec,
    discover,
    parse_override,
)

_URL_A = "http://a:8001/v1"
_URL_B = "http://b:8001/v1"

# The real tokenizer check, captured before the autouse stub patches it.
_REAL_REQUIRE_TOKENIZER = E._require_tokenizer


def _models_payload(model_id="gemma4-e4b", max_model_len=262144, extra=None):
    data = [{"id": model_id, "max_model_len": max_model_len}]
    if extra:
        data.extend(extra)
    return {"data": data}


def _fake_http(monkeypatch, table: dict):
    """``_get_json`` fake: an absent url raises like an unreachable host."""

    def _get(url, *, bearer=None):
        if url not in table:
            raise EndpointDiscoveryError(f"endpoint unreachable: GET {url}")
        return table[url]

    monkeypatch.setattr(E, "_get_json", _get)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Fresh install slot + a stub tokenizer check per test (the
    tokenizer-failure test overrides the stub with the real path)."""
    monkeypatch.setattr(E, "_SPEC", None)
    monkeypatch.setattr(E, "_require_tokenizer", lambda m: f"tok:{m}")
    yield


# --------------------------------------------------------------------------- #
# Happy path + derivation arithmetic                                          #
# --------------------------------------------------------------------------- #


def test_discover_single_endpoint(monkeypatch):
    _fake_http(monkeypatch, {f"{_URL_A}/models": _models_payload()})
    s = discover(_URL_A)
    assert s.base_url == _URL_A
    assert s.model_id == "gemma4-e4b"
    assert s.max_model_len == 262144
    assert s.input_budget == 262144 - 40000 - 1024
    assert s.max_output_tokens == 40000


def test_small_window_derivation(monkeypatch):
    # 32768 -> reserve 8192 (min(40000, window//4)), input_budget 24576 - margin.
    _fake_http(
        monkeypatch,
        {f"{_URL_A}/models": _models_payload("gemma4-26b-a4b-nvfp4", 32768)},
    )
    s = discover(_URL_A)
    assert s.input_budget == 32768 - 8192 - 1024
    assert s.max_output_tokens == 8192


def test_service_urls_frozen_and_probed(monkeypatch):
    _fake_http(
        monkeypatch,
        {
            f"{_URL_A}/models": _models_payload(),
            "http://sam:8772/health": {"service": "sam"},
        },
    )
    s = discover(_URL_A, ("http://sam:8772",))
    assert s.sam_urls == ("http://sam:8772",)


# --------------------------------------------------------------------------- #
# Abort taxonomy a-f                                                          #
# --------------------------------------------------------------------------- #


def test_case_a_unreachable_endpoint(monkeypatch):
    _fake_http(monkeypatch, {})
    with pytest.raises(EndpointDiscoveryError, match="unreachable"):
        discover(_URL_A)


def test_case_b_empty_model_list(monkeypatch):
    _fake_http(monkeypatch, {f"{_URL_A}/models": {"data": []}})
    with pytest.raises(EndpointDiscoveryError, match="no served models"):
        discover(_URL_A)


def test_case_c_more_than_one_model(monkeypatch):
    _fake_http(
        monkeypatch,
        {
            f"{_URL_A}/models": _models_payload(
                extra=[{"id": "other", "max_model_len": 8192}]
            )
        },
    )
    with pytest.raises(EndpointDiscoveryError, match="2 models"):
        discover(_URL_A)


def test_case_d_missing_max_model_len(monkeypatch):
    _fake_http(monkeypatch, {f"{_URL_A}/models": {"data": [{"id": "m"}]}})
    with pytest.raises(EndpointDiscoveryError, match="max_model_len"):
        discover(_URL_A)


def test_case_e_comma_list_refused():
    # A comma-list is refused before any probe.
    with pytest.raises(EndpointDiscoveryError, match="comma-list"):
        discover(f"{_URL_A},{_URL_B}")


def test_case_f_window_too_small(monkeypatch):
    _fake_http(monkeypatch, {f"{_URL_A}/models": _models_payload("gemma4-e2b", 8192)})
    with pytest.raises(EndpointDiscoveryError, match="input_budget"):
        discover(_URL_A)


# --------------------------------------------------------------------------- #
# Case g: tokenizer is a discovered fact                                      #
# --------------------------------------------------------------------------- #


def test_case_g_tokenizer_load_failure(monkeypatch):
    _fake_http(monkeypatch, {f"{_URL_A}/models": _models_payload("gemma4-26b-a4b-nvfp4", 32768)})
    # Re-enable the REAL _require_tokenizer over a stubbed loader.
    monkeypatch.setattr(E, "_require_tokenizer", _REAL_REQUIRE_TOKENIZER)
    import maps.llm.tokenizer as tok

    def _unloadable(m):
        raise RuntimeError(f"tokenizer for served model {m!r} failed to load")

    monkeypatch.setattr(tok, "_tokenizer", _unloadable)
    with pytest.raises(EndpointDiscoveryError, match="tokenizer"):
        discover(_URL_A)


class _NoToolTurnTemplate:
    """A chat template that renders plain turns and fails on a tool turn."""

    def apply_chat_template(self, messages, tools=None, **_kw):
        if tools or any(m.get("tool_calls") for m in messages):
            raise TypeError("template cannot render tool calls")
        return "rendered"


def test_case_g_discovery_refuses_a_template_that_cannot_render_a_tool_turn(monkeypatch):
    _fake_http(monkeypatch, {f"{_URL_A}/models": _models_payload("gemma4-26b-a4b-nvfp4", 32768)})
    monkeypatch.setattr(E, "_require_tokenizer", _REAL_REQUIRE_TOKENIZER)
    import maps.llm.tokenizer as tok

    monkeypatch.setattr(tok, "_tokenizer", lambda m: _NoToolTurnTemplate())
    with pytest.raises(EndpointDiscoveryError, match="chat template"):
        discover(_URL_A)


# --------------------------------------------------------------------------- #
# Cases h/i: service URLs are first-class in the fail-loud engine             #
# --------------------------------------------------------------------------- #


def test_case_h_service_unreachable(monkeypatch):
    _fake_http(monkeypatch, {f"{_URL_A}/models": _models_payload()})
    with pytest.raises(EndpointDiscoveryError, match="'sam'"):
        discover(_URL_A, ("http://sam:8772",))


def test_case_i_service_identity_mismatch(monkeypatch):
    # A reachable-but-WRONG server on the right port: /health answers 200
    # with a foreign identity -> abort instead of a silently-wrong green run.
    _fake_http(
        monkeypatch,
        {
            f"{_URL_A}/models": _models_payload(),
            "http://sam:8772/health": {"service": "flight_sim"},
        },
    )
    with pytest.raises(EndpointDiscoveryError, match="WRONG"):
        discover(_URL_A, ("http://sam:8772",))


def test_service_health_without_identity_field_is_reachability_only(monkeypatch):
    _fake_http(
        monkeypatch,
        {
            f"{_URL_A}/models": _models_payload(),
            "http://sam:8772/health": {"status": "ok"},
        },
    )
    s = discover(_URL_A, ("http://sam:8772",))
    assert s.sam_urls == ("http://sam:8772",)


# --------------------------------------------------------------------------- #
# Offline + override escape hatches                                           #
# --------------------------------------------------------------------------- #


def test_offline_without_pin_raises():
    with pytest.raises(EndpointDiscoveryError, match="OFFLINE"):
        discover(_URL_A, offline=True)


def test_offline_with_pin_skips_probes(monkeypatch):
    _fake_http(monkeypatch, {})  # every URL unreachable — must not matter
    s = discover(
        _URL_A,
        ("http://sam:8772",),
        offline=True,
        override=EndpointOverride("gemma4-e4b", 131072),
    )
    assert s.model_id == "gemma4-e4b"
    assert s.max_model_len == 131072
    assert s.sam_urls == ("http://sam:8772",)  # frozen unprobed offline


def test_override_cross_checked_against_live_ids(monkeypatch):
    # The override never bypasses the live check: pin absent from /v1/models
    # -> abort (a stale override).
    _fake_http(monkeypatch, {f"{_URL_A}/models": _models_payload("served-model")})
    with pytest.raises(EndpointDiscoveryError, match="does not match"):
        discover(_URL_A, override=EndpointOverride("pinned-model", 131072))


def test_override_selects_among_many_models(monkeypatch):
    # Non-vLLM shape: many served models -> the pin picks one, live-checked.
    _fake_http(
        monkeypatch,
        {
            f"{_URL_A}/models": {
                "data": [{"id": "gemma4-e4b"}, {"id": "gemma4-26b-a4b-nvfp4"}, {"id": "inkling-small"}]
            }
        },
    )
    s = discover(_URL_A, override=EndpointOverride("gemma4-26b-a4b-nvfp4", 131072))
    assert s.model_id == "gemma4-26b-a4b-nvfp4"
    assert s.max_model_len == 131072


def test_parse_override():
    o = parse_override("gemma4-e4b:262144")
    assert o == EndpointOverride("gemma4-e4b", 262144)
    with pytest.raises(EndpointDiscoveryError, match="malformed"):
        parse_override("no-window")
    with pytest.raises(EndpointDiscoveryError, match="malformed"):
        parse_override(":123")


# --------------------------------------------------------------------------- #
# install_spec / spec: one authority, install-once-or-abort                   #
# --------------------------------------------------------------------------- #


def _spec(**kw) -> EndpointSpec:
    base = dict(
        base_url=_URL_A,
        model_id="gemma4-e4b",
        max_model_len=262144,
        tokenizer_name="Qwen/Qwen3.6-35B-A3B",
        max_output_tokens=40000,
    )
    base.update(kw)
    return EndpointSpec(**base)


def test_spec_before_install_raises():
    with pytest.raises(EndpointDiscoveryError, match="no EndpointSpec installed"):
        E.spec()
    assert E.spec_installed() is False


def test_install_then_read_back():
    s = _spec()
    E.install_spec(s)
    assert E.spec() is s
    assert E.spec_installed() is True
    E.install_spec(_spec())  # equal fields: idempotent
    assert E.spec() is s


def test_divergent_second_install_raises():
    E.install_spec(_spec())
    with pytest.raises(EndpointDiscoveryError, match="divergent"):
        E.install_spec(_spec(max_model_len=131072))


def test_a_transport_failure_is_asked_again_before_it_is_a_verdict(monkeypatch):
    """One closed connection while the arm's relay re-forms is not an
    unreachable endpoint: the probe asks again, and only a link that stays
    down is a verdict."""
    import io
    import urllib.error

    monkeypatch.setattr(E, "_PROBE_RETRY_S", 0.0)
    calls: list[int] = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def flaky(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.URLError("Remote end closed connection without response")
        return _Resp(b'{"data": []}')

    monkeypatch.setattr(E.urllib.request, "urlopen", flaky)
    assert E._get_json("http://arm:18001/v1/models") == {"data": []}
    assert len(calls) == 3

    calls.clear()

    def dead(req, timeout):
        calls.append(1)
        raise urllib.error.URLError("No route to host")

    monkeypatch.setattr(E.urllib.request, "urlopen", dead)
    with pytest.raises(EndpointDiscoveryError, match="after 3 tries"):
        E._get_json("http://arm:18001/v1/models")
    assert len(calls) == E._PROBE_TRIES


@pytest.mark.parametrize("fault,unreachable", [
    (lambda: urllib.error.HTTPError("u", 429, "Too Many Requests", None, None), True),
    (lambda: urllib.error.HTTPError("u", 401, "Unauthorized", None, None), False),
    (lambda: ValueError("unknown url type: 'arm'"), False),
    (lambda: http.client.IncompleteRead(b'{"data":'), True),
], ids=["429", "401", "malformed url", "truncated body"])
def test_a_serve_that_is_busy_or_drops_its_body_is_unreachable_and_a_refusal_is_not(monkeypatch, fault, unreachable):
    """A 4xx the serve clears on its own and a body cut short are the serve's
    fault, asked again; any other 4xx is the config's, refused at once."""
    monkeypatch.setattr(E, "_PROBE_RETRY_S", 0.0)
    calls: list[int] = []

    def failing(req, timeout):
        calls.append(1)
        raise fault()

    monkeypatch.setattr(E.urllib.request, "urlopen", failing)
    with pytest.raises(EndpointDiscoveryError) as err:
        E._get_json("http://arm:18001/v1/models")
    assert isinstance(err.value, E.EndpointUnreachable) is unreachable
    assert len(calls) == (E._PROBE_TRIES if unreachable else 1)


# --------------------------------------------------------------------------- #
# The arm's reasoning effort is one the served card takes                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id,effort", [
    ("inkling-small", "0.99"), ("qwen38-flash-next-nvfp4", "xhigh"), ("gemma4-e4b", "0.9"),
])
def test_discovery_takes_an_effort_the_card_takes(monkeypatch, model_id, effort):
    _fake_http(monkeypatch, {f"{_URL_A}/models": _models_payload(model_id, 262144)})
    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", effort)
    assert discover(_URL_A).model_id == model_id


@pytest.mark.parametrize("model_id,effort,named", [
    ("inkling-small", "nan", "nan"), ("inkling-small", "1.0", "1.0"),
    ("inkling-small", "hgih", "hgih"), ("inkling-small", "xhigh", "xhigh"),
    ("inkling-small", "-0.0", "'0.0'"), ("qwen38-flash-next-nvfp4", "0.9", "0.9"),
])
def test_discovery_refuses_an_effort_the_card_does_not_take(monkeypatch, model_id, effort, named):
    _fake_http(monkeypatch, {f"{_URL_A}/models": _models_payload(model_id, 262144)})
    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", effort)
    with pytest.raises(EndpointDiscoveryError, match=f"SEARCH_SAMPLING_REASONING_EFFORT.*{named}"):
        discover(_URL_A)
