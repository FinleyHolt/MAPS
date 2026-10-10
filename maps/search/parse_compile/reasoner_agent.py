"""Reasoner LLM context for the single persistent reasoner.

The reasoner is one persistent message history. The orchestrator
appends a per-iteration packet (see ``prompts.py``) and advances the
reasoner across the mission. The only step mode is:

  - ``step_tool_call(tools)``: one NATIVE tool-call turn under
    ``tool_choice="auto"``. Every model act is a tool call and everything
    else is thinking; the tool call IS the output, and there is no second
    emission after it.

Serving rails (load-bearing): per-tool ``strict: true`` (the xgrammar
constraint makes a malformed tool-args envelope an impossible token);
``tool_choice`` is ALWAYS ``"auto"`` — never ``required``/named — because
auto IS the abstain path.

The reasoner only knows its own context.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from maps.search.analyze_scene.compactor import compact_transcript
from maps.llm.endpoint_spec import EndpointSpec
from maps.llm.http import (
    CallCanceled,
    DegenerateCompletion,
    _chat_completions_raw,
)
from maps.llm.sampling import thinking_sampling_kwargs
from maps.llm.vision_content import build_vision_user_content

log = logging.getLogger(__name__)


# Every model-facing surface carries the minimum simplest set that is
# SUFFICIENT and NECESSARY -- mission structure, capability facts,
# composition grammar,
# world conventions. No steering, no method recipes, no worked examples, no
# imperative restatements: a fact in imperative dress is stated as its
# mechanical consequence instead. The per-tool capability legend rides the
# tool descriptions (``prompts.py``); this constant carries the rest.
def served_system_text(*, sensor_description: str = "") -> str:
    """The single system message the model is served.

    One edit to the one constant prompt: the run's sensor-rig block fuses
    onto the tail. The Qwen3 template rejects a second ``role=system`` turn,
    so the rig block is fused here rather than appended as its own message.
    The ``Reasoner``, the contract dump and the dashboard lab all serve this
    exact string, so what the lab shows on screen is the byte stream a
    flight sends. A blank ``sensor_description`` returns the constant prompt
    alone, no trailing separator — what a run with no rig configured serves.

    There is no crops variant. The say surface swapped a no-crops fact for a
    look-capability legend when the renderer was wired; this surface states
    its own look capability inline and ``run_mission`` asserts the renderer
    IS wired, so there was never a second byte stream to choose between.

    The ``system_turn`` withhold serves no system turn: an empty string,
    which the ``Reasoner`` sends as no message.
    """
    from maps.search.parse_compile.prompts import (  # noqa: PLC0415
        served_system_prompt)
    from maps.search.withhold import withheld  # noqa: PLC0415
    if withheld("system_turn"):
        return ""
    text = served_system_prompt()
    if sensor_description.strip():
        text += "\n\n" + _format_sensor_system_message(sensor_description)
    return text


def _env_look_enabled() -> bool:
    return os.environ.get("SEARCH_LOOK_TOOL", "").strip().lower() in (
        "1", "true", "yes",
    )


# There is no in-context image cap. Crops accumulate and the compactor bounds
# them in TOKENS, which is the resource the served window actually spends.
#
# A count cap stood here, evicting the oldest crops after every delivery to keep
# sixteen. It guarded a per-request limit no serve declared: none passed
# --limit-mm-data-per-request, and sglang defaults it to None. The 400 it defended against could not happen, and the
# eviction it performed to prevent it rewrote already-sent history, which is the
# one thing a radix prefix cannot survive. Measured on 1,525 pulled Cal City
# episodes: rollbacks on 8.13% of requests discarding a mean of 35.2 messages,
# and all 763 classified rollbacks were this eviction losing exactly one image.
#
# A count is also the wrong proxy: sixteen thumbnails are ~3.6k tokens and
# sixteen full-size crops ~17k. If a serve ever does advertise
# limit_mm_data_per_request, enforce it at DELIVERY by declining the crop that
# would exceed it. Never by rewriting the first one.

_REASONER_TIMEOUT_DEFAULT_S = 240.0

#: A run of one character repeated to the end of a channel. Degenerate decode
#: output, never prose -- one serve returned `!` (token id 0) to max_tokens on a
#: fifth of xhigh requests, after a median of 3 clean characters.
#:
#: 64 characters, not 16. A collapse runs to thousands (40,000 measured), so
#: the threshold costs nothing in detection and buys the whole false-positive
#: margin: a markdown rule, a run of trailing spaces or tabs, an ellipsis
#: block -- all reach 16 and none reach 64. That margin is load-bearing
#: because a false positive here is NOT one wasted retry: it is deterministic
#: per prompt, so it repeats, and six repeats end the mission
#: ``transient_failures``, which analysis drops from SR and the driver
#: re-queues forever.
_COLLAPSE_RUN = re.compile(r"(.)\1{63,}\s*$")


def _degenerate_reason(*, text: str, reasoning: str, calls: list) -> str:
    """Why this completion is a SERVE fault, or ``""`` if it is not.

    ONE test: a callless completion whose output ends in a repeated-token
    run. BOTH channels are checked, because nothing about a NaN logit row is
    channel-specific -- a collapse after ``</think>`` puts the run in
    ``content``, and the loop reads a non-empty prose reply as the mission's
    ONE model-owned ending, so an uncaught one stops the mission on a reply
    of pure garbage with no repair and no retry. The content check therefore
    runs BEFORE the has-a-real-reply short circuit, not after it.

    ``finish_reason`` is deliberately NOT a test, and that is a correction.
    An earlier version treated every ``finish_reason == "length"`` with no
    content as degenerate -- which is exactly the CLIP ("the think span ran
    long") that ``probes/probe_run.py`` was changed to stop conflating with
    a collapse. That arm was deterministic per prompt, so it did not cost one
    retry: six consecutive raises reach ``MAX_ROUND_TRANSIENT`` and end the
    mission ``transient_failures``, which ``analysis/rows.py`` drops from SR
    as an infra abort and ``bench_fullset`` re-queues for a full redo -- a
    deterministic failure re-queues the same episode forever. A clip has to
    keep taking the empty-turn repair, whose measured next-turn recovery is
    84%.
    """
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
    """Per-call transport timeout, from ``SEARCH_REASONER_TIMEOUT_S``.

    Unset or blank takes the default. Anything else must be a positive
    number: a malformed or zero value raises rather than falling back,
    because a silent fallback is exactly what this read was written to end
    (the env was exported by two run scripts and documented in
    ``maps/search/README.md`` while nothing read it, so a slow call died
    at the default with every caller convinced it had 900).
    """
    raw = os.environ.get("SEARCH_REASONER_TIMEOUT_S", "").strip()
    if not raw:
        return _REASONER_TIMEOUT_DEFAULT_S
    value = float(raw)   # loud on garbage, deliberately
    if value <= 0:
        raise ValueError(
            f"SEARCH_REASONER_TIMEOUT_S={raw!r} is not a positive number of "
            f"seconds; httpx reads 0 as 'time out immediately', never 'no limit'"
        )
    return value


def _format_sensor_system_message(description: str) -> str:
    """Return the per-run sensor-rig block appended to the constant reasoner
    system prompt. vLLM's Qwen3 chat-template wrapper rejects requests
    with more than one ``role=system`` message ("System message must be
    at the beginning"), so this block is fused into the single system
    message at ``Reasoner`` construction time rather than appended as
    a second turn. Facts only (radical arm): what the rig resolves and
    what SAM does; no segmentation directives."""
    return (
        "Sensor rig (active for this run):\n"
        f"{description.strip()}\n\n"
        "A concept this rig cannot see produces an empty SAM response. SAM "
        "covers un-cataloged / movable objects; static map features are "
        "already in the graph as prior footprints with "
        "geometry and raw OSM tags. SAM segments roads as merged blobs and over-splits "
        "buildings."
    )


def _normalize_tool_call_arguments(tc: dict) -> dict:
    """Return a copy of ``tc`` whose ``function.arguments`` parses as a JSON
    mapping.

    The Qwen3 chat template renders ``tool_call.arguments|items``; vLLM's
    ``chat_utils._postprocess_messages`` ``json.loads``-es a non-empty
    arguments string into a dict before templating, then raises if the
    JSON is malformed and 400s when the decoded value is not a mapping.
    Per-tool ``strict: true`` enforces the schema on the SERVER, but the
    parser can still hand back oddities on edge paths (and offline test
    doubles inject arbitrary strings). Normalizing at the point of history
    append keeps every subsequent turn's chat-template render safe.
    """
    fn = dict(tc.get("function") or {})
    raw = fn.get("arguments")
    if not isinstance(raw, str) or not raw.strip():
        fn["arguments"] = "{}"
    else:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            # A malformed args string is NOT silently replaced with {} —
            # that would make the tool layer error "program must be an
            # object..." and mislead the model (it HAD written a program;
            # its JSON broke). First try a mechanical brace/bracket
            # completion (truncation is the common break); else carry the
            # REAL parse error to the tool layer via the __args_parse_error
            # marker so the model sees the truth.
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
    """Append the closing brackets a TRUNCATED JSON object is missing.
    Walks the string outside string-literals tracking an opener stack;
    returns the completed string, or None when the text is broken in a
    non-truncation way (mismatched closers, dangling escapes)."""
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
    """Persistent message history for one reasoner's reasoner LLM.

    ``endpoint_spec`` is the server-discovered endpoint authority
    (``maps.llm.endpoint_spec``), threaded explicitly from the
    compiler: the request's ``model``, ``max_tokens`` and the compaction
    ``target_input_tokens`` all come from it, so the sent name IS the
    served name and ``prompt + completion`` fits the served window."""

    reasoner_url: str
    endpoint_spec: EndpointSpec
    api_key: str = "EMPTY"
    # Per-call transport timeout. The default fits local-serve turn latencies;
    # a slower endpoint (a frontier-model shim whose fresh first turn processes
    # the whole packet, or any serve reached across a field link) raises it via
    # SEARCH_REASONER_TIMEOUT_S without touching the default arm. The env read
    # is load-bearing, not decoration: two run scripts and fsg_env.sh export
    # 900 believing it applies, maps/search/README.md documents it as the
    # per-round deadline, and while nothing read it a slow call died at 240 s
    # with every caller convinced it had 900.
    # A garbage value raises here rather than reverting to the default: a
    # typo'd export that silently fell back to 240 while every caller believed
    # 900 is the failure this read exists to end. 0 is rejected for the same
    # reason -- httpx reads it as "time out immediately", never "no limit".
    timeout: float = field(
        default_factory=lambda: _reasoner_timeout_s())
    seed: int | None = None
    reasoner_index: int | None = None
    messages: list[dict] = field(default_factory=list)
    sensor_description: str = ""
    cancel_event: threading.Event | None = None
    #: The reasoning effort this reasoner's requests carry, a string the
    #: serve accepts (``maps.llm.sampling``); None sends the arm's own.
    reasoning_effort: str | None = None

    @property
    def model(self) -> str:
        """The served model id — discovered, never a client-side guess."""
        return self.endpoint_spec.model_id

    def __post_init__(self) -> None:
        system = served_system_text(sensor_description=self.sensor_description)
        if not self.messages and system:
            self.messages.append({"role": "system", "content": system})

    # ------------------------------------------------------------------
    # Message-history helpers
    # ------------------------------------------------------------------

    def append_user(self, content: str) -> None:
        """Append a TEXT user-role turn (homework packet).

        The per-stop reasoner round reasons over the structured world-state
        (SAM detections + perceived attributes on the runtime graph, read in
        TEXT via ``analyze``). This method only ever appends text; the
        opt-in ``view{crops}`` path is the SOLE one that attaches image
        crops, via the separate :meth:`append_vision_user`.
        """
        self.messages.append({"role": "user", "content": content})

    def append_vision_user(self, text: str, jpegs: list[bytes]) -> None:
        """Append a MULTI-PART user turn (text + image crops) — the ONLY path
        that puts non-text on the reasoner's context, driven exclusively by the
        opt-in ``look`` tool (``reasoner_loop`` two-message delivery). ``content``
        is the OpenAI multi-part list (``vision_content.build_vision_user_content``: text
        part first, then one image part per JPEG) the vLLM Qwen3.6 endpoint
        forwards unchanged. With ``look`` unexposed this is never called, so the
        history stays 100% text (parity). An empty ``jpegs`` degrades to a plain
        text turn.

        Append-only: no earlier message is touched, so a crop delivery extends
        the served prefix rather than invalidating it."""
        content = build_vision_user_content(text, list(jpegs or []))
        self.messages.append({"role": "user", "content": content})

    def append_repair(self, message: str) -> None:
        """Append a user-level repair turn (a protocol nudge: the no-call
        repair, a malformed-envelope re-ask). The orchestrator counts repair
        rounds separately so repair traffic is tracked apart from
        substantive reasoning turns.
        """
        body = f"[Repair] {message.strip()}"
        self.append_user(body)

    def append_tool_result(self, tool_call_id: str, name: str, result: Any) -> None:
        """Append a tool-result message after dispatching a tool call.

        EVERY tool_call id the assistant emitted gets exactly one result
        (the compactor's id-closure sweep requires a result per id). Dict
        results serialize as ``json.dumps(result, default=str)`` —
        byte-matching ``distill/render/chat_render.tool_result_msg`` so the
        student trains on the exact string the harness serves.
        """
        if isinstance(result, str):
            content = result
        else:
            try:
                content = json.dumps(result, default=str)
            except (TypeError, ValueError):
                content = str(result)
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": content,
            }
        )


    # ------------------------------------------------------------------
    # THE step mode (three-verb native tool-call surface)
    # ------------------------------------------------------------------

    def step_tool_call(self, tools: list[dict]) -> dict:
        """One native tool-call turn under ``tool_choice="auto"``.

        Appends the assistant turn (content + ``reasoning_content`` +
        normalized ``tool_calls``) to the persistent history — the exact
        shape ``distill/render/chat_render.trace_to_messages`` renders, so
        training and serving share one message surface — and returns:

            {"kind": "tool_call", "tool_calls": [...], "content": str,
             "reasoning": str, "finish_reason": str}
            {"kind": "text", "content": str, "reasoning": str,
             "finish_reason": str}

        On ``"tool_call"`` the orchestrator dispatches each call IN ORDER
        and appends one tool-role result PER call id, then decides whether
        a terminal fired. On ``"text"`` the orchestrator applies the
        no-call rule (nudge -> carry) — a prose turn is never an error by
        itself.

        Rails: ``tool_choice`` is hard-pinned to ``"auto"`` (never
        required/named — auto is the abstain path); NON-STREAMING only;
        the tool call IS the output — no free-standing structured-output
        emission. No client-side grammar rides the request: the surface is
        one tool with one string argument, which ``strict: true`` on the
        envelope already pins, and the block inside it is held to the
        language by the parser's own refusals.
        """
        # Sampling lives in `maps.llm.sampling.thinking_sampling_kwargs`
        # so every reasoner turn stays in lock-step.
        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "model": self.model,
            "messages": self.messages,
            # Pin the completion to the spec's output reserve so
            # prompt(<=input_budget) + completion(<=max_output_tokens) fits
            # the served window — the server-side length check cannot trip.
            "max_tokens": self.endpoint_spec.max_output_tokens,
            # NO `stop` sequence to cut a collapsed sampler short, and that
            # is deliberate. Trimming a collapse to ~1 s of decode instead of
            # ~9 min is a THROUGHPUT win, and it cost correctness everywhere
            # but one engine: sglang deletes matched stop text unless
            # `no_stop_trim` is set, vLLM spells the same idea
            # `include_stop_str_in_output`, and llama.cpp has neither. On any
            # of those the stop fires, the run is deleted, and the completion
            # arrives as a bare few characters -- contentless and callless,
            # which is the empty turn again. The stop sequence would DEFEAT
            # the guard it exists to serve. `no_stop_trim` is also an unknown
            # top-level field to a hosted OpenAI-compatible endpoint, which
            # `.env.example` documents as a supported target and which 400s
            # on unrecognized params: six retries, then every episode ends
            # `transient_failures` and is dropped from SR as infra.
            **thinking_sampling_kwargs(temperature=1.0, reasoning_effort=self.reasoning_effort),
            "tools": tools,
            "tool_choice": "auto",
            "transport_kind": "reasoner",
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        sub_id = (
            str(self.reasoner_index) if self.reasoner_index is not None else None
        )
        if sub_id is not None:
            kwargs["transport_subagent_id"] = sub_id
        compacted_messages = compact_transcript(
            self.messages,
            target_input_tokens=self.endpoint_spec.input_budget,
            model=self.model,
            tools=tools,
            site="reasoner.step_tool_call",
        )
        if compacted_messages is not self.messages:
            self.messages = compacted_messages
        kwargs["messages"] = self.messages
        if self.cancel_event is not None:
            if self.cancel_event.is_set():
                raise CallCanceled(
                    "reasoner.step_tool_call: cancel_event set before dispatch"
                )
            kwargs["cancel_event"] = self.cancel_event

        resp = _chat_completions_raw(self.reasoner_url, self.api_key, **kwargs)
        choice = resp.choices[0]
        msg = choice.message
        # Round-trip the reasoning channel back to the model on the next
        # request (OpenCode's `interleaved` pattern): prior thinking is
        # preserved on the assistant turn so vLLM's Qwen3 chat template
        # renders it as `<think>...</think>` in the next prompt. Empty
        # string (rather than absent key) keeps the field shape stable.
        reasoning_text = msg.reasoning or ""
        text = msg.content or ""
        finish_reason = str(choice.finish_reason or "")
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": text,
            "reasoning_content": reasoning_text,
        }
        raw_calls = [
            {
                "id": tc.id,
                "type": tc.type or "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in (msg.tool_calls or [])
        ]
        # An unregistered function name is a parser mangle (fragments of the
        # XML envelope leaking into the name). Keep the call ON history with
        # its id: the round loop's unknown-tool branch answers it with an
        # actionable error result (the id-closure invariant), never executes.
        calls = [_normalize_tool_call_arguments(tc) for tc in raw_calls]
        # A collapsed completion never reaches history. Raising HERE rather
        # than in the loop is what keeps the garbage out of the prefix: the
        # append below is what put 40,000 characters of `!` on the next
        # request, the compactor, and every later prefill.
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
        if calls:
            assistant_msg["tool_calls"] = calls
            self.messages.append(assistant_msg)
            return {
                "kind": "tool_call",
                "tool_calls": calls,
                "content": text,
                "reasoning": reasoning_text,
                "finish_reason": finish_reason,
            }
        self.messages.append(assistant_msg)
        return {
            "kind": "text",
            "content": text,
            "reasoning": reasoning_text,
            "finish_reason": finish_reason,
        }
