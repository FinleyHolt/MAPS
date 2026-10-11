"""A model behind OpenAI's own chat-completions contract (GPT-5.6 Sol on Bedrock)
is sent only what that contract takes, and is metered by what it reports."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from maps.llm import endpoint_spec as E
from maps.llm import openai_wire
from maps.llm.flight_summary import reasoning_tokens_of, summarize
from maps.llm.model_cards import CARDS, check_effort
from maps.llm.sampling import thinking_sampling_kwargs
from maps.search.parse_compile import reasoner_agent
from maps.search.parse_compile.reasoner_agent import Reasoner
from maps.search.tests.fakes import TEST_SPEC, jpeg

GPT = "openai.gpt-5.6-sol"
GPT_SPEC = replace(TEST_SPEC, model_id=GPT, tokenizer_name="openai/gpt-oss-120b")


@pytest.fixture(autouse=True)
def _no_ambient_effort(monkeypatch) -> None:
    monkeypatch.delenv("SEARCH_SAMPLING_REASONING_EFFORT", raising=False)


def test_the_card_is_the_one_an_openai_wire_reader_keys_on() -> None:
    """VACUITY FLOOR: every branch below keys on this one row; without it they test nothing."""
    assert [a for a, c in CARDS.items() if c.get("wire") == "openai"] == [GPT]


def test_no_sampling_envelope_and_no_template_kwargs_ride_the_request(monkeypatch) -> None:
    assert thinking_sampling_kwargs(temperature=1.0, model_id=GPT) == {}
    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", "high")
    assert thinking_sampling_kwargs(temperature=1.0, model_id=GPT) == {"reasoning_effort": "high"}


def test_a_temperature_the_api_refuses_raises_here() -> None:
    with pytest.raises(ValueError, match="1.0 only"):
        thinking_sampling_kwargs(temperature=0.6, model_id=GPT)


@pytest.mark.parametrize("rung", ["none", "low", "medium", "high", "xhigh", "max"])
def test_the_test_plan_s_efforts_pass_discovery(rung) -> None:
    check_effort(GPT, rung)


@pytest.mark.parametrize("raw", ["0.9", "minimal", "Instant"])
def test_an_effort_the_model_does_not_take_is_refused(raw) -> None:
    with pytest.raises(ValueError, match="is not one"):
        check_effort(GPT, raw)


def test_each_message_carries_only_its_role_s_fields() -> None:
    history = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": [{"type": "text", "text": "u"}]},
        {"role": "assistant", "content": "", "reasoning_content": "",
         "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c0", "name": "run", "content": "t"},
    ]
    sent = openai_wire.messages(history)
    assert [sorted(m) for m in sent] == [
        ["content", "role"], ["content", "role"], ["content", "role", "tool_calls"],
        ["content", "role", "tool_call_id"]]
    assert "reasoning_content" in history[2] and "name" in history[3], "the history itself is untouched"
    with pytest.raises(ValueError, match="developer"):
        openai_wire.messages([{"role": "developer", "content": "x"}])


def _serve(monkeypatch) -> list[dict]:
    sent: list[dict] = []

    def _fake(url, api_key, **kwargs):
        sent.append(kwargs)
        return {"choices": [{"finish_reason": "tool_calls", "message": {
            "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{\"code\": \"x\"}"}}]}}]}

    monkeypatch.setattr(reasoner_agent, "_chat_completions_raw", _fake)
    return sent


def test_the_reasoner_s_request_is_the_openai_shape(monkeypatch) -> None:
    sent = _serve(monkeypatch)
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=GPT_SPEC, seed=7, reasoning_effort="medium")
    r.append_user("find the truck")
    r.step_tool_call([{"type": "function", "function": {"name": "run"}}])
    r.append_tool_result("c1", "run", "ok")
    r.step_tool_call([{"type": "function", "function": {"name": "run"}}])

    body = sent[1]
    assert body["max_completion_tokens"] == GPT_SPEC.max_output_tokens and "max_tokens" not in body
    for key in ("temperature", "top_p", "top_k", "chat_template_kwargs", "seed"):
        assert key not in body
    assert body["reasoning_effort"] == "medium"
    assert [sorted(m) for m in body["messages"][-2:]] == [
        ["content", "role", "tool_calls"], ["content", "role", "tool_call_id"]]
    # The reasoner's own history keeps the fields the wire dropped.
    assert "reasoning_content" in r.messages[-3] and r.messages[-2]["name"] == "run"


def test_the_run_tool_s_documentation_rides_the_system_turn(monkeypatch) -> None:
    """OpenAI refuses a function description over 1,024 characters; the run tool's is far
    longer, so it moves verbatim into the system turn and the tool points at it."""
    from maps.search.parse_compile.prompts import served_tools

    tools = served_tools()
    doc = tools[0]["function"]["description"]
    assert len(doc) > openai_wire.DESCRIPTION_MAX_CHARS, "VACUITY FLOOR: nothing to move"
    sent = _serve(monkeypatch)
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=GPT_SPEC)
    r.append_user("find the truck")
    r.step_tool_call(tools)
    body = sent[0]
    assert all(len(t["function"]["description"]) <= openai_wire.DESCRIPTION_MAX_CHARS for t in body["tools"])
    assert body["tools"][0]["function"]["parameters"] == tools[0]["function"]["parameters"]
    system = body["messages"][0]["content"]
    assert system.startswith(r.messages[0]["content"]) and system.endswith(doc)
    assert tools[0]["function"]["description"] == doc and r.messages[0]["content"] != system, "an input was mutated"


def test_descriptions_that_fit_pass_untouched() -> None:
    tools = [{"type": "function", "function": {"name": "run", "description": "short"}}]
    sent = [{"role": "user", "content": "u"}]
    assert openai_wire.fit_tool_descriptions(tools, sent) == (tools, sent)
    long = [{"type": "function", "function": {"name": "run", "description": "x" * 2000}}]
    fitted, with_system = openai_wire.fit_tool_descriptions(long, sent)
    assert with_system[0]["role"] == "system" and with_system[1:] == sent, "no system turn: one is added"


def test_a_self_hosted_card_keeps_its_shape(monkeypatch) -> None:
    """The narrowing is the OpenAI card's alone: Inkling's body is what it always was."""
    sent = _serve(monkeypatch)
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=TEST_SPEC, seed=7)
    r.append_user("find the truck")
    r.step_tool_call([])
    body = sent[0]
    assert body["max_tokens"] == TEST_SPEC.max_output_tokens and "max_completion_tokens" not in body
    assert body["seed"] == 7
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["top_k"] == 64 and body["messages"] is r.messages


def test_the_request_never_carries_more_images_than_bedrock_takes(monkeypatch) -> None:
    sent = _serve(monkeypatch)
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=GPT_SPEC)
    picture = jpeg(64)
    for k in range(3):
        r.append_vision_user(f"look {k}", [picture] * 40)
    r.step_tool_call([])
    images = [p for m in sent[0]["messages"] if isinstance(m["content"], list)
              for p in m["content"] if p.get("type") == "image_url"]
    # 120 is over Bedrock's 100, so the oldest go down to the low-water mark.
    assert CARDS[GPT]["max_images"] == 100 and len(images) == 80
    assert sum(p.get("type") == "image_url" for p in r.messages[-2]["content"]) == 40, "the newest look lost a crop"
    assert sum(p.get("type") == "image_url" for p in r.messages[1]["content"]) == 0


def test_reasoning_is_metered_from_usage_when_the_span_never_comes_back() -> None:
    response = {"model": "gpt-5.6-sol-2026-07-09", "choices": [{"message": {"content": "ok"}}],
                "usage": {"completion_tokens": 900, "completion_tokens_details": {"reasoning_tokens": 812}}}
    assert reasoning_tokens_of(response, model=GPT) == 812
    # Unkeyed, the span rule reads it as no thinking at all: the reason the model is passed.
    assert reasoning_tokens_of(response) == 0


def test_the_episode_summary_meters_an_openai_turn(tmp_path: Path) -> None:
    llm = tmp_path / "flight" / "llm"
    llm.mkdir(parents=True)
    records = [
        {"kind": "params", "sha": "p", "params": {"model": GPT, "tools": []}},
        {"kind": "request", "request_id": "r1", "created_at": 1.0},
        {"kind": "response", "request_id": "r1", "ts": 2.0, "response": {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 5000, "completion_tokens": 900, "total_tokens": 5900,
                      "completion_tokens_details": {"reasoning_tokens": 812}}}},
    ]
    (llm / "transcript.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    out = summarize(tmp_path / "flight")
    assert (out["reasoning_tokens"], out["completion_tokens"], out["turns"]) == (812, 900, 1)


def test_the_model_list_is_read_where_the_endpoint_keeps_it(monkeypatch) -> None:
    """bedrock-mantle serves GPT-5.6 under /openai/v1 and lists its models under /v1."""
    base = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
    listing = "https://bedrock-mantle.us-east-1.api.aws/v1/models"
    seen: list[str] = []

    def _get(url, *, bearer=None):
        seen.append(url)
        if url != listing:
            raise E.EndpointDiscoveryError(f"endpoint unreachable: GET {url}")
        return {"data": [{"id": "openai.gpt-oss-120b"}, {"id": GPT}]}

    monkeypatch.setattr(E, "_get_json", _get)
    monkeypatch.setattr(E, "_require_tokenizer", lambda m: f"tok:{m}")
    monkeypatch.setenv("SEARCH_REASONER_MODELS_URL", listing)
    spec = E.discover(base, override=E.EndpointOverride(GPT, 262144))
    assert (spec.base_url, spec.model_id, spec.max_model_len, seen) == (base, GPT, 262144, [listing])
    assert spec.server_declared_window is False
