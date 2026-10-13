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

from typing import Any

#: The fields each role's message takes on OpenAI's chat-completions API.
_MESSAGE_FIELDS: dict[str, frozenset[str]] = {
    "system": frozenset({"role", "content"}),
    "user": frozenset({"role", "content"}),
    "assistant": frozenset({"role", "content", "tool_calls"}),
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
