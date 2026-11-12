"""A model behind OpenAI's own chat-completions contract (GPT-5.6 Sol on an Azure
deployment) is sent only what that contract takes, and is metered by what it reports."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from maps.llm import endpoint_spec as E
from maps.llm import openai_wire
from maps.llm.flight_summary import reasoning_tokens_of, summarize
from maps.llm.model_cards import CARDS, check_effort, sampling_kwargs
from maps.mission import compactor, reasoner
from maps.mission.reasoner import Reasoner
from maps.mission.tests.fakes import TEST_SPEC, jpeg
from maps.settings import from_environ

GPT = "gpt-5.6-sol"
GPT_SPEC = replace(TEST_SPEC, model_id=GPT, tokenizer_name="openai/gpt-oss-120b")


class _Tok:
    """Four characters a token over the rendered transcript."""

    def apply_chat_template(self, messages, tools=None, **_kw):
        return json.dumps([messages, tools], default=str)

    def encode(self, text, add_special_tokens=False):
        return range(len(text) // 4)


@pytest.fixture(autouse=True)
def _offline_tokenizer(monkeypatch) -> None:
    """The compactor sizes the transcript with a stub, never a hub download."""
    monkeypatch.setattr(compactor, "load_tokenizer", lambda _model, _tokenizer_name="": _Tok())


def test_the_card_is_the_one_an_openai_wire_reader_keys_on() -> None:
    """VACUITY FLOOR: every branch below keys on this one row; without it they test nothing."""
    assert [a for a, c in CARDS.items() if c.get("wire") == "openai"] == [GPT]


def test_no_sampling_envelope_and_no_template_kwargs_ride_the_request() -> None:
    assert sampling_kwargs(GPT) == {}
    arm = from_environ({"SEARCH_SAMPLING_REASONING_EFFORT": "high"}).sampling
    assert sampling_kwargs(GPT, sampling=arm) == {"reasoning_effort": "high"}


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
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "",
            "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "run", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c0", "name": "run", "content": "t"},
    ]
    sent = openai_wire.messages(history)
    assert [sorted(m) for m in sent] == [
        ["content", "role"],
        ["content", "role"],
        ["content", "role", "tool_calls"],
        ["content", "role", "tool_call_id"],
    ]
    assert "reasoning_content" in history[2] and "name" in history[3], (
        "the history itself is untouched"
    )
    with pytest.raises(ValueError, match="developer"):
        openai_wire.messages([{"role": "developer", "content": "x"}])


def _serve(monkeypatch) -> list[dict]:
    sent: list[dict] = []

    def _fake(url, api_key, *, workdir=None, routing_key=None, endpoint=None, **kwargs):
        sent.append(kwargs)
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "run", "arguments": '{"code": "x"}'},
                            }
                        ],
                    },
                }
            ]
        }

    monkeypatch.setattr(reasoner, "chat_completions_raw", _fake)
    return sent


def test_the_reasoner_s_request_is_the_openai_shape(monkeypatch) -> None:
    sent = _serve(monkeypatch)
    r = Reasoner(endpoint_spec=GPT_SPEC, seed=7, reasoning_effort="medium")
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
        ["content", "role", "tool_calls"],
        ["content", "role", "tool_call_id"],
    ]
    # The reasoner's own history keeps the fields the wire dropped.
    assert "reasoning_content" in r.messages[-3] and r.messages[-2]["name"] == "run"


def test_the_episode_s_routing_key_is_its_prompt_cache_key(monkeypatch) -> None:
    """Azure matches a GPT-5.6 prefix cache within a ``prompt_cache_key``; the episode's
    routing key carries its turns on one. A self-hosted card's body gains no such key."""
    sent = _serve(monkeypatch)
    for spec in (GPT_SPEC, TEST_SPEC):
        r = Reasoner(endpoint_spec=spec, routing_key="calcity::ep-7")
        r.append_user("find the truck")
        r.step_tool_call([])
    assert sent[0]["prompt_cache_key"] == "calcity::ep-7"
    assert "prompt_cache_key" not in sent[1]
    r = Reasoner(endpoint_spec=GPT_SPEC)
    r.append_user("find the truck")
    r.step_tool_call([])
    assert "prompt_cache_key" not in sent[2]


def test_each_request_s_bytes_extend_the_last_one_s(monkeypatch) -> None:
    """A provider caches the longest prefix it has seen byte for byte: under the image
    cap and the budget, each turn's tools serialize as the last turn's did and its
    messages array opens with the last turn's bytes."""
    from maps.mission.served_text import SERVED_TOOLS

    sent: list[dict] = []

    def _fake(url, api_key, *, workdir=None, routing_key=None, endpoint=None, **kwargs):
        sent.append(json.loads(json.dumps(kwargs)))  # the bytes as sent, not a live alias
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"c{len(sent)}",
                                "type": "function",
                                "function": {"name": "run", "arguments": '{ "code":  "look()" }'},
                            }
                        ],
                    },
                }
            ]
        }

    monkeypatch.setattr(reasoner, "chat_completions_raw", _fake)
    r = Reasoner(endpoint_spec=GPT_SPEC, reasoning_effort="xhigh")
    r.append_vision_user("look 0", [jpeg(64)] * 4)
    for k in range(1, 5):
        r.step_tool_call(SERVED_TOOLS)
        r.append_tool_result(f"c{k}", "run", {"printed": f"turn {k}"})
        r.append_vision_user(f"look {k}", [jpeg(64)] * 4)
    assert len(sent) == 4, "VACUITY FLOOR"
    for prev, cur in zip(sent, sent[1:]):
        assert json.dumps(cur["tools"]) == json.dumps(prev["tools"])
        assert json.dumps(cur["messages"]).startswith(json.dumps(prev["messages"])[:-1])


def test_the_run_tool_s_documentation_rides_the_system_turn(monkeypatch) -> None:
    """OpenAI refuses a function description over 1,024 characters; the run tool's is far
    longer, so it moves verbatim into the system turn and the tool points at it."""
    from maps.mission.served_text import SERVED_TOOLS

    tools = SERVED_TOOLS
    doc = tools[0]["function"]["description"]
    assert len(doc) > openai_wire.DESCRIPTION_MAX_CHARS, "VACUITY FLOOR: nothing to move"
    sent = _serve(monkeypatch)
    r = Reasoner(endpoint_spec=GPT_SPEC)
    r.append_user("find the truck")
    r.step_tool_call(tools)
    body = sent[0]
    assert all(
        len(t["function"]["description"]) <= openai_wire.DESCRIPTION_MAX_CHARS
        for t in body["tools"]
    )
    assert body["tools"][0]["function"]["parameters"] == tools[0]["function"]["parameters"]
    system = body["messages"][0]["content"]
    assert system.startswith(r.messages[0]["content"]) and system.endswith(doc)
    assert tools[0]["function"]["description"] == doc and r.messages[0]["content"] != system, (
        "an input was mutated"
    )


def test_descriptions_that_fit_pass_untouched() -> None:
    tools = [{"type": "function", "function": {"name": "run", "description": "short"}}]
    sent = [{"role": "user", "content": "u"}]
    assert openai_wire.fit_tool_descriptions(tools, sent) == (tools, sent)
    long = [{"type": "function", "function": {"name": "run", "description": "x" * 2000}}]
    fitted, with_system = openai_wire.fit_tool_descriptions(long, sent)
    assert with_system[0]["role"] == "system" and with_system[1:] == sent, (
        "no system turn: one is added"
    )


def test_a_self_hosted_card_keeps_its_shape(monkeypatch) -> None:
    """The narrowing is the OpenAI card's alone: Inkling's body is what it always was."""
    sent = _serve(monkeypatch)
    r = Reasoner(endpoint_spec=TEST_SPEC, seed=7)
    r.append_user("find the truck")
    r.step_tool_call([])
    body = sent[0]
    assert body["max_tokens"] == TEST_SPEC.max_output_tokens and "max_completion_tokens" not in body
    assert body["seed"] == 7
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["top_k"] == 64 and body["messages"] is r.messages


def test_the_request_never_carries_more_images_than_azure_takes(monkeypatch) -> None:
    from maps.mission.picture_requests import LOOK_MAX_CROPS

    sent = _serve(monkeypatch)
    r = Reasoner(endpoint_spec=GPT_SPEC)
    picture = jpeg(64)
    for k in range(4):
        r.append_vision_user(f"look {k}", [picture] * LOOK_MAX_CROPS)
    r.step_tool_call([])
    images = [
        p
        for m in sent[0]["messages"]
        if isinstance(m["content"], list)
        for p in m["content"]
        if p.get("type") == "image_url"
    ]
    # Four full looks (64) are over Azure's 50, so the oldest go down to the low-water mark.
    assert CARDS[GPT]["max_images"] == 50 and len(images) == 30
    crops = [sum(p.get("type") == "image_url" for p in m["content"]) for m in r.messages[1:5]]
    assert crops == [0, 0, 14, LOOK_MAX_CROPS], "the newest look lost a crop"


def test_reasoning_is_metered_from_usage_when_the_span_never_comes_back() -> None:
    response = {
        "model": "gpt-5.6-sol-2026-07-09",
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"completion_tokens": 900, "completion_tokens_details": {"reasoning_tokens": 812}},
    }
    assert reasoning_tokens_of(response, model=GPT) == 812
    # Unkeyed, the span rule reads it as no thinking at all: the reason the model is passed.
    assert reasoning_tokens_of(response) == 0


def test_the_episode_summary_meters_an_openai_turn(tmp_path: Path) -> None:
    llm = tmp_path / "flight" / "llm"
    llm.mkdir(parents=True)
    records = [
        {"kind": "params", "sha": "p", "params": {"model": GPT, "tools": []}},
        {"kind": "request", "request_id": "r1", "created_at": 1.0},
        {
            "kind": "response",
            "request_id": "r1",
            "ts": 2.0,
            "response": {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": 5000,
                    "completion_tokens": 900,
                    "total_tokens": 5900,
                    "completion_tokens_details": {"reasoning_tokens": 812},
                },
            },
        },
    ]
    (llm / "transcript.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    out = summarize(tmp_path / "flight")
    assert (out["reasoning_tokens"], out["completion_tokens"], out["turns"]) == (812, 900, 1)


def test_the_model_list_is_read_where_the_endpoint_keeps_it(monkeypatch) -> None:
    """SEARCH_REASONER_MODELS_URL names a listing kept away from SEARCH_REASONER_URL/models."""
    base = "https://reasoner.example/serve/v1"
    listing = "https://reasoner.example/catalog/models"
    seen: list[str] = []

    def _get(url, *, bearer=None):
        seen.append(url)
        if url != listing:
            raise E.EndpointDiscoveryError(f"endpoint unreachable: GET {url}")
        return {"data": [{"id": "gemma4-e4b"}, {"id": "inkling-small"}]}

    monkeypatch.setattr(E, "_get_json", _get)
    monkeypatch.setattr(E, "_require_tokenizer", lambda m, override="": f"tok:{m}")
    spec = E.discover(
        base,
        override=E.EndpointOverride("inkling-small", 262144),
        settings=from_environ({"SEARCH_REASONER_MODELS_URL": listing}),
    )
    assert (spec.base_url, spec.model_id, spec.max_model_len, seen) == (
        base,
        "inkling-small",
        262144,
        [listing],
    )


def test_a_deployment_name_the_listing_does_not_carry_passes_discovery(monkeypatch) -> None:
    """Azure's /openai/v1/models lists the resource's base models, not its deployments: the
    deployment-named alias passes on the listing answering the key, and a self-hosted pin the
    listing does not carry is still refused."""
    base = "https://example.openai.azure.com/openai/v1"
    seen: list[tuple[str, str | None]] = []

    def _get(url, *, bearer=None):
        seen.append((url, bearer))
        return {"data": [{"id": "gpt-5.6-sol-2026-07-09"}, {"id": "gpt-5.4-2026-03-05"}]}

    monkeypatch.setattr(E, "_get_json", _get)
    monkeypatch.setattr(E, "_require_tokenizer", lambda m, override="": f"tok:{m}")
    keyed = from_environ({"SEARCH_REASONER_API_KEY": "k"})
    spec = E.discover(base, override=E.EndpointOverride(GPT, 262144), settings=keyed)
    assert (spec.model_id, spec.max_model_len) == (GPT, 262144)
    assert seen == [(f"{base}/models", "k")], "the listing is asked, with the key"
    with pytest.raises(E.EndpointDiscoveryError, match="does not match the live endpoint"):
        E.discover(base, override=E.EndpointOverride("inkling-small", 262144), settings=keyed)


def test_a_chat_body_becomes_the_responses_request_carrying_the_same_conversation() -> None:
    picture = "data:image/jpeg;base64,AAAA"
    body = {
        "model": GPT,
        "max_completion_tokens": 40000,
        "reasoning_effort": "xhigh",
        "prompt_cache_key": "ep",
        "tool_choice": "auto",
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {"name": "run", "description": "d", "parameters": {"type": "object"}},
            }
        ],
        "messages": [
            {"role": "system", "content": "s"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "u"},
                    {"type": "image_url", "image_url": {"url": picture, "detail": "high"}},
                ],
            },
            {
                "role": "assistant",
                "content": "thinking aloud",
                "reasoning_items": [
                    {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "E1"}
                ],
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ],
    }
    out = openai_wire.to_responses(body)
    assert {k for k in out} == {
        "model",
        "input",
        "tools",
        "tool_choice",
        "max_output_tokens",
        "reasoning",
        "prompt_cache_key",
        "store",
        "include",
        "stream",
    }
    assert (out["max_output_tokens"], out["reasoning"], out["store"]) == (
        40000,
        {"effort": "xhigh"},
        False,
    )
    assert out["include"] == ["reasoning.encrypted_content"]
    assert out["tools"] == [
        {"type": "function", "name": "run", "description": "d", "parameters": {"type": "object"}}
    ]
    assert [i.get("type") for i in out["input"]] == [
        "message",
        "message",
        "reasoning",
        "message",
        "function_call",
        "function_call_output",
    ]
    assert out["input"][1]["content"][1] == {
        "type": "input_image",
        "image_url": picture,
        "detail": "high",
    }
    assert out["input"][2]["encrypted_content"] == "E1"
    assert out["input"][4] == {
        "type": "function_call",
        "call_id": "c1",
        "name": "run",
        "arguments": "{}",
    }
    assert out["input"][5] == {"type": "function_call_output", "call_id": "c1", "output": "ok"}
    assert "messages" in body and "reasoning_items" in body["messages"][2], "the input was mutated"


@pytest.mark.parametrize(
    "status,reason,finish",
    [
        ("completed", None, "tool_calls"),
        ("incomplete", "max_output_tokens", "length"),
        ("incomplete", "content_filter", "content_filter"),
    ],
)
def test_a_responses_answer_reads_as_the_chat_answer_the_engine_takes(
    status, reason, finish
) -> None:
    resp = {
        "id": "r",
        "model": "gpt-5.6-sol-2026-07-09",
        "status": status,
        **({"incomplete_details": {"reason": reason}} if reason else {}),
        "output": [
            {"type": "reasoning", "id": "rs", "summary": [], "encrypted_content": "E"},
            {"type": "function_call", "call_id": "c9", "name": "run", "arguments": '{"code": "x"}'},
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 30,
            "total_tokens": 130,
            "input_tokens_details": {"cached_tokens": 64},
            "output_tokens_details": {"reasoning_tokens": 20},
        },
    }
    out = openai_wire.from_responses(resp)
    msg = out["choices"][0]["message"]
    assert out["choices"][0]["finish_reason"] == finish
    assert msg["tool_calls"] == [
        {"id": "c9", "type": "function", "function": {"name": "run", "arguments": '{"code": "x"}'}}
    ]
    assert msg["reasoning_items"] == [
        {"type": "reasoning", "id": "rs", "summary": [], "encrypted_content": "E"}
    ]
    assert out["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 30,
        "total_tokens": 130,
        "prompt_tokens_details": {"cached_tokens": 64},
        "completion_tokens_details": {"reasoning_tokens": 20},
    }
    assert reasoning_tokens_of(out, model=GPT) == 20


def test_the_turn_s_encrypted_reasoning_rides_the_next_request(monkeypatch) -> None:
    """The reasoner keeps a Responses turn's reasoning items on its assistant turn and the
    wire narrowing passes them on; the compactor prices them by the tokens they took."""
    from maps.mission import compactor

    sent: list[dict] = []

    def _fake(url, api_key, *, workdir=None, routing_key=None, endpoint=None, **kwargs):
        sent.append(json.loads(json.dumps(kwargs)))
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "reasoning_items": [
                            {"type": "reasoning", "id": f"rs{len(sent)}", "encrypted_content": "E"}
                        ],
                        "tool_calls": [
                            {
                                "id": f"c{len(sent)}",
                                "type": "function",
                                "function": {"name": "run", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ],
            "usage": {"completion_tokens_details": {"reasoning_tokens": 5000}},
        }

    monkeypatch.setattr(reasoner, "chat_completions_raw", _fake)
    r = Reasoner(endpoint_spec=GPT_SPEC, reasoning_effort="medium")
    r.append_user("find the truck")
    r.step_tool_call([])
    r.append_tool_result("c1", "run", "ok")
    r.step_tool_call([])
    turn = sent[1]["messages"][-2]
    assert turn["reasoning_items"] == [{"type": "reasoning", "id": "rs1", "encrypted_content": "E"}]
    assert "reasoning_tokens" not in turn, "the price is the compactor's, never sent"
    with_items = compactor._estimate_input_tokens(r.messages, model=GPT)
    bare = [
        {k: v for k, v in m.items() if k not in ("reasoning_items", "reasoning_tokens")}
        for m in r.messages
    ]
    assert with_items - compactor._estimate_input_tokens(bare, model=GPT) == 2 * 5000


def test_a_bench_worker_s_cache_key_outranks_the_episode_s(monkeypatch) -> None:
    """The bench names one key per worker, shared by the episodes it flies in turn, so each
    after its first reads the common head from the cache."""
    sent = _serve(monkeypatch)
    r = Reasoner(
        endpoint_spec=GPT_SPEC,
        routing_key="calcity::ep-7",
        settings=from_environ({"SEARCH_LLM_CACHE_KEY": "gto2_CG-HI_M_89_abc/w3"}),
    )
    r.append_user("find the truck")
    r.step_tool_call([])
    assert sent[0]["prompt_cache_key"] == "gto2_CG-HI_M_89_abc/w3"
