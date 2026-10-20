"""The request shape OpenAI's own chat-completions contract takes.

A self-hosted serve renders the reasoner's history through a chat template,
which reads the extra fields the history carries (``reasoning_content`` on an
assistant turn, ``name`` on a tool result) and ignores the rest. OpenAI's API
validates each message against its own schema instead, so a model card marked
``wire: "openai"`` (``maps.llm.model_cards``) is sent each message with only
the fields that schema names, its output cap under the name its reasoning
models take, and no seed (``reasoner_agent.Reasoner.step_tool_call``). The
reasoner's own history keeps every field; only the request body is narrowed.

The API also refuses a function description over 1,024 characters, and the
``run`` tool's documentation is far longer. A chat template renders tool
descriptions into the system turn, so that is where an over-long one goes
here: the system turn gains a section holding it verbatim, and the tool
keeps a one-line pointer to that section.
"""

from __future__ import annotations

import json
from typing import Any

#: The fields each role's message takes on OpenAI's chat-completions API, and
#: an assistant turn's encrypted reasoning, which only :func:`to_responses` sends.
_MESSAGE_FIELDS: dict[str, frozenset[str]] = {
    "system": frozenset({"role", "content"}),
    "user": frozenset({"role", "content"}),
    "assistant": frozenset({"role", "content", "tool_calls", "reasoning_items"}),
    "tool": frozenset({"role", "content", "tool_call_id"}),
}

#: The output cap's name on a reasoning model, where ``max_tokens`` is refused.
OUTPUT_CAP_KEY = "max_completion_tokens"

#: The longest ``tools[].function.description`` the API takes.
DESCRIPTION_MAX_CHARS = 1024


def messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``history`` with each message narrowed to its role's fields. A role
    the contract does not name raises: sending it would be refused anyway."""
    out = []
    for message in history:
        fields = _MESSAGE_FIELDS.get(message.get("role"))
        if fields is None:
            raise ValueError(f"no OpenAI chat-completions message role {message.get('role')!r}")
        out.append({k: v for k, v in message.items() if k in fields})
    return out


def fit_tool_descriptions(
    tools: list[dict[str, Any]], sent: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(tools, sent)`` with every description over the limit moved into the
    system turn under its own heading, the tool keeping a pointer to it. A
    transcript with no system turn gains one. Inputs are never mutated, and
    both come back as given when every description fits."""
    sections: list[str] = []
    fitted: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool.get("function") or {}
        doc = fn.get("description") or ""
        if len(doc) <= DESCRIPTION_MAX_CHARS:
            fitted.append(tool)
            continue
        heading = f"# The `{fn['name']}` tool"
        sections.append(f"{heading}\n\n{doc}")
        fitted.append({**tool, "function": {**fn, "description": f"Documented in the system message under \"{heading}\"."}})
    if not sections:
        return tools, sent
    docs = "\n\n".join(sections)
    if sent and sent[0].get("role") == "system":
        system = sent[0]["content"]
        if not isinstance(system, str):
            raise ValueError("the system turn is not text, so a tool's documentation cannot join it")
        return fitted, [{**sent[0], "content": f"{system}\n\n{docs}"}, *sent[1:]]
    return fitted, [{"role": "system", "content": docs}, *sent]


# ---------------------------------------------------------------------------
# The Responses API
# ---------------------------------------------------------------------------
#
# A reasoning model that refuses function tools beside a reasoning effort on
# chat completions (GPT-5.6 Sol on Azure: "use /v1/responses or set
# reasoning_effort to 'none'") takes the same request as Responses input
# items. The engine keeps building, recording and reading the chat shape;
# :func:`to_responses` and :func:`from_responses` translate at the wire. Its
# reasoning never comes back in the clear: each assistant turn carries the
# encrypted reasoning items its response returned (``reasoning_items``), and
# the next request sends them back before that turn's calls, so the model
# reasons with its earlier reasoning in view, as a self-hosted template
# renders ``reasoning_content``. Nothing is stored server-side.

#: The path a Responses request posts to, beside ``/chat/completions``.
RESPONSES_PATH = "/responses"

#: The chat body keys the Responses body carries under another name, or not at all.
_CHAT_ONLY_KEYS = frozenset({"messages", "max_completion_tokens", "reasoning_effort", "stream_options", "tools"})


def _input_content(content: Any, role: str) -> Any:
    """A chat message's content as Responses content parts."""
    if not isinstance(content, list):
        return content if isinstance(content, str) else ""
    text_type = "output_text" if role == "assistant" else "input_text"
    out = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text":
            out.append({"type": text_type, "text": part.get("text") or ""})
        elif kind == "image_url":
            image = part.get("image_url") or {}
            out.append({"type": "input_image", "image_url": image.get("url"),
                        **({"detail": image["detail"]} if image.get("detail") else {})})
        else:
            raise ValueError(f"no Responses content part for {kind!r}")
    return out


def _input_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The chat history as Responses input items: each message as one, an
    assistant turn as its reasoning items, its text and its calls."""
    items: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"],
                          "output": m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))})
            continue
        if role == "assistant":
            items.extend(m.get("reasoning_items") or [])
            text = m.get("content") or ""
            if text:
                items.append({"type": "message", "role": "assistant",
                              "content": [{"type": "output_text", "text": text}]})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                items.append({"type": "function_call", "call_id": tc["id"], "name": fn.get("name", ""),
                              "arguments": fn.get("arguments") if isinstance(fn.get("arguments"), str)
                              else json.dumps(fn.get("arguments") or {})})
            continue
        if role not in ("system", "user", "developer"):
            raise ValueError(f"no Responses input item for role {role!r}")
        items.append({"type": "message", "role": role, "content": _input_content(m.get("content"), role)})
    return items


def to_responses(body: dict[str, Any]) -> dict[str, Any]:
    """A chat-completions body as the Responses request carrying the same
    conversation, tools, output cap, effort and cache key; stateless
    (``store: false``), the reasoning returned encrypted for the next turn."""
    out = {k: v for k, v in body.items() if k not in _CHAT_ONLY_KEYS}
    out["input"] = _input_items(body.get("messages") or [])
    if "tools" in body:
        out["tools"] = [{"type": "function", "name": t["function"]["name"],
                         "description": t["function"].get("description", ""),
                         "parameters": t["function"].get("parameters") or {"type": "object", "properties": {}}}
                        for t in body["tools"]]
    if "max_completion_tokens" in body:
        out["max_output_tokens"] = body["max_completion_tokens"]
    if body.get("reasoning_effort"):
        out["reasoning"] = {"effort": body["reasoning_effort"]}
    out["store"] = False
    out["include"] = ["reasoning.encrypted_content"]
    return out


def _usage(usage: dict[str, Any]) -> dict[str, Any]:
    """Responses usage under the chat-completions names the meters read."""
    inp = usage.get("input_tokens_details") or {}
    outp = usage.get("output_tokens_details") or {}
    return {"prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0)),
            "prompt_tokens_details": {k: inp[k] for k in ("cached_tokens", "cache_write_tokens") if k in inp},
            "completion_tokens_details": {k: outp[k] for k in ("reasoning_tokens",) if k in outp}}


def from_responses(resp: dict[str, Any]) -> dict[str, Any]:
    """A Responses response as the chat-completions response the engine
    reads: the output text, the calls, the finish reason, the usage, and the
    encrypted reasoning items the assistant turn carries forward."""
    text: list[str] = []
    calls: list[dict[str, Any]] = []
    reasoning: list[dict[str, Any]] = []
    for item in resp.get("output") or []:
        kind = item.get("type")
        if kind == "reasoning":
            reasoning.append({k: item[k] for k in ("type", "id", "summary", "encrypted_content") if k in item})
        elif kind == "message":
            text += [p.get("text") or "" for p in item.get("content") or [] if p.get("type") == "output_text"]
        elif kind == "function_call":
            calls.append({"id": item["call_id"], "type": "function",
                          "function": {"name": item.get("name", ""), "arguments": item.get("arguments") or ""}})
    status = resp.get("status")
    reason = (resp.get("incomplete_details") or {}).get("reason")
    finish = ("length" if status == "incomplete" and reason == "max_output_tokens"
              else "content_filter" if status == "incomplete" and reason == "content_filter"
              else "tool_calls" if calls else "stop")
    message: dict[str, Any] = {"role": "assistant", "content": "".join(text)}
    if calls:
        message["tool_calls"] = calls
    if reasoning:
        message["reasoning_items"] = reasoning
    out = {"id": resp.get("id"), "model": resp.get("model"), "status": status,
           "choices": [{"index": 0, "message": message, "finish_reason": finish}]}
    if isinstance(resp.get("usage"), dict):
        out["usage"] = _usage(resp["usage"])
    return out
