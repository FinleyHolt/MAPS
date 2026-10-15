"""The reasoner: one persistent message history for the mission, advanced
one native tool-call turn at a time (``Reasoner.step_tool_call``)."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from maps.llm import openai_wire
from maps.llm.endpoint_spec import EndpointSpec
from maps.llm.http import (
    CallCanceled,
    DegenerateCompletion,
    _chat_completions_raw,
)
from maps.llm.model_cards import card_for, sampling_kwargs, speaks_openai
from maps.llm.vision_content import build_vision_user_content
from maps.search.parse_compile.compactor import bound_image_count, compact_transcript
from maps.search.parse_compile.prompts import served_system_prompt

log = logging.getLogger(__name__)


_REASONER_TIMEOUT_DEFAULT_S = 240.0

# One character repeated to the end of a channel: degenerate decode output.
# A collapse runs to thousands of characters; 64 stays clear of markdown
# rules and whitespace runs, which matters because a false positive repeats
# deterministically until the mission ends on transient failures.
_COLLAPSE_RUN = re.compile(r"(.)\1{63,}\s*$")


def _degenerate_reason(*, text: str, reasoning: str, calls: list) -> str:
    """Why this completion is a serve fault, or ``""``: a callless
    completion whose content, or (with no content) whose reasoning, ends in
    a repeated-character run. Content is checked first because a prose reply
    ends the mission."""
    if calls:
        return ""
    if _COLLAPSE_RUN.search(text):
        return "repeated-token collapse in the content channel"
    if text.strip():
        return ""
    if _COLLAPSE_RUN.search(reasoning):
        return "repeated-token collapse in the reasoning channel"
    return ""


def _reasoner_timeout_s() -> float:
    """Per-call transport timeout from ``SEARCH_REASONER_TIMEOUT_S`` (unset:
    the default); anything but a positive number of seconds up to
    ``threading.TIMEOUT_MAX`` raises."""
    raw = os.environ.get("SEARCH_REASONER_TIMEOUT_S", "").strip()
    if not raw:
        return _REASONER_TIMEOUT_DEFAULT_S
    value = float(raw)   # loud on garbage, deliberately
    if not 0 < value <= threading.TIMEOUT_MAX:
        raise ValueError(
            f"SEARCH_REASONER_TIMEOUT_S={raw!r} is not a positive number of "
            f"seconds at most threading.TIMEOUT_MAX; httpx reads 0 as 'time out "
            f"immediately', never 'no limit', and a socket refuses a longer wait"
        )
    return value


def _normalize_tool_call_arguments(tc: dict) -> dict:
    """A copy of ``tc`` whose ``function.arguments`` is a JSON object, so
    every later chat-template render of the history is safe."""
    fn = dict(tc.get("function") or {})
    raw = fn.get("arguments")
    if not isinstance(raw, str) or not raw.strip():
        fn["arguments"] = "{}"
    else:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            # Complete a truncated object, else carry the real parse error
            # in the __args_parse_error marker.
            repaired = _complete_json_brackets(raw)
            parsed2 = None
            if repaired is not None:
                try:
                    parsed2 = json.loads(repaired)
                except json.JSONDecodeError:
                    parsed2 = None
            if isinstance(parsed2, dict):
                fn["arguments"] = json.dumps(parsed2)
            else:
                fn["arguments"] = json.dumps({
                    "__args_parse_error": (
                        f"{exc.msg} at pos {exc.pos} "
                        f"(tail: ...{raw[-80:]!r})"
                    )
                })
        else:
            fn["arguments"] = json.dumps(parsed) if isinstance(parsed, dict) else "{}"
    out = dict(tc)
    out["function"] = fn
    return out


def _complete_json_brackets(raw: str) -> str | None:
    """``raw`` with the closers a truncated JSON object is missing, or None
    when it is broken another way (a mismatched closer, a dangling escape)."""
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in raw:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_str:
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            o = stack.pop()
            if (o, ch) not in (("{", "}"), ("[", "]")):
                return None
    if esc:
        return None
    out = raw
    if in_str:
        out += '"'
    for o in reversed(stack):
        out += "}" if o == "{" else "]"
    return out


@dataclass
class Reasoner:
    """The mission's message history and its request settings.
    ``endpoint_spec`` (server-discovered) supplies the base URL, the model
    name, ``max_tokens`` and the compaction budget, so prompt + completion
    fits the served window."""

    endpoint_spec: EndpointSpec
    # "EMPTY" is the no-auth sentinel a local vLLM accepts.
    api_key: str = field(
        default_factory=lambda: os.environ.get("SEARCH_REASONER_API_KEY", "EMPTY"))
    timeout: float = field(default_factory=_reasoner_timeout_s)
    seed: int | None = None
    messages: list[dict] = field(default_factory=list)
    sensor_description: str = ""
    cancel_event: threading.Event | None = None
    # the requests' reasoning effort (``model_cards.sampling_kwargs``); None: the default
    reasoning_effort: str | None = None
    # Set when compaction cannot fit the thread: no later turn grows it.
    compaction_infeasible: bool = False

    @property
    def model(self) -> str:
        """The served model id."""
        return self.endpoint_spec.model_id

    def __post_init__(self) -> None:
        system = served_system_prompt(self.sensor_description)
        if not self.messages and system:
            self.messages.append({"role": "system", "content": system})

    def append_user(self, content: str) -> None:
        """Append a text user turn."""
        self.messages.append({"role": "user", "content": content})

    def append_vision_user(self, text: str, jpegs: list[bytes]) -> None:
        """Append a user turn of text plus one image part per JPEG, the only
        way images reach the context (no jpegs: a text turn)."""
        content = build_vision_user_content(text, list(jpegs or []))
        self.messages.append({"role": "user", "content": content})

    def append_repair(self, message: str) -> None:
        """Append a ``[Repair]`` user turn (a protocol nudge)."""
        body = f"[Repair] {message.strip()}"
        self.append_user(body)

    def append_tool_result(self, tool_call_id: str, name: str, result: dict) -> None:
        """Append one tool-role result, ``json.dumps(result, default=str)``.
        Every tool_call id the assistant emitted gets exactly one (the
        compactor refuses a call without a result)."""
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id,
                              "name": name,
                              "content": json.dumps(result, default=str)})

    def step_tool_call(self, tools: list[dict]) -> dict:
        """One turn under ``tool_choice="auto"`` (a plain reply must stay
        possible: it ends the mission); it streams when ``cancel_event`` is
        set, so the event can abort it. Compacts the history,
        appends the assistant turn and returns ``{"kind": "tool_call" |
        "text", "tool_calls"?, "content", "reasoning", "finish_reason"}``.
        A degenerate completion raises ``DegenerateCompletion`` unappended.
        """
        # A card's ``max_images`` (a hosted endpoint's per-request limit)
        # bounds the crop COUNT by evicting the oldest; it runs BEFORE the
        # compactor, which would otherwise price crops about to go.
        max_images = card_for(self.model).get("max_images")
        if max_images:
            self.messages = bound_image_count(self.messages, max_images=max_images)
        self.messages = compact_transcript(
            self.messages,
            target_input_tokens=self.endpoint_spec.input_budget,
            model=self.model,
            tools=tools,
        )
        wire = speaks_openai(self.model)
        sent = self.messages
        if wire:
            tools, sent = openai_wire.fit_tool_descriptions(tools, openai_wire.messages(self.messages))
        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "model": self.model,
            "messages": sent,
            openai_wire.OUTPUT_CAP_KEY if wire else "max_tokens": self.endpoint_spec.max_output_tokens,
            # No `stop` sequence: engines trim the matched text, so a stopped
            # collapse would arrive as an empty turn the guard cannot see, and
            # a hosted endpoint 400s on the engine-specific no-trim flags.
            **sampling_kwargs(self.model, self.reasoning_effort),
            "tools": tools,
            "tool_choice": "auto",
        }
        # The OpenAI wire sends no seed: there it is a deprecated best-effort
        # hint that replays nothing a reasoning model does.
        if self.seed is not None and not wire:
            kwargs["seed"] = self.seed
        # A hosted provider's prefix cache is matched within a key: the
        # episode's routing key (one request a turn) keeps its turns on one.
        routing_key = os.environ.get("SEARCH_LLM_ROUTING_KEY")
        if wire and routing_key:
            kwargs["prompt_cache_key"] = routing_key
        if self.cancel_event is not None:
            if self.cancel_event.is_set():
                raise CallCanceled(
                    "reasoner.step_tool_call: cancel_event set before dispatch"
                )
            kwargs["cancel_event"] = self.cancel_event

        resp = _chat_completions_raw(self.endpoint_spec.base_url, self.api_key, **kwargs)
        choice = resp["choices"][0]
        msg = choice.get("message", {})
        # The reasoning rides the assistant turn back to the next request (the
        # serve names it ``reasoning`` or ``reasoning_content``).
        reasoning_text = msg.get("reasoning") or msg.get("reasoning_content") or ""
        text = msg.get("content") or ""
        finish_reason = str(choice.get("finish_reason") or "")
        # An unknown tool name stays on history: the loop answers its id.
        calls = [_normalize_tool_call_arguments({
            "id": tc.get("id"),
            "type": tc.get("type") or "function",
            "function": {
                "name": tc.get("function", {}).get("name", ""),
                "arguments": tc.get("function", {}).get("arguments", ""),
            },
        }) for tc in (msg.get("tool_calls") or [])]
        # raised before the append, so a collapse never reaches the prefix
        degenerate = _degenerate_reason(
            text=text, reasoning=reasoning_text, calls=calls,
        )
        if degenerate:
            log.warning(
                "reasoner.step_tool_call: degenerate completion (%s); "
                "finish_reason=%s reasoning_chars=%d -- retriable serve fault",
                degenerate, finish_reason or "?", len(reasoning_text),
            )
            raise DegenerateCompletion(
                f"reasoner returned a degenerate completion: {degenerate} "
                f"(finish_reason={finish_reason!r}, "
                f"reasoning_chars={len(reasoning_text)})"
            )
        self.messages.append({
            "role": "assistant", "content": text,
            "reasoning_content": reasoning_text,
            **({"tool_calls": calls} if calls else {})})
        return {"kind": "tool_call" if calls else "text",
                **({"tool_calls": calls} if calls else {}),
                "content": text, "reasoning": reasoning_text,
                "finish_reason": finish_reason}
