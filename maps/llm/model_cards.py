"""Client-side model cards, keyed by the exact discovered served alias: the
tokenizer repo, the thinking-mode sampling envelope and the
``reasoning_effort`` values each chat template takes. Any other name is
refused."""

from __future__ import annotations

import os
from typing import Any

#: alias -> {tokenizer, top_p, top_k[, presence_penalty][, efforts][, effort_numbers]}, or for a
#: hosted OpenAI model {tokenizer, wire: "openai", efforts[, max_images]}.
#: ``efforts`` are the rungs the chat template takes and ``effort_numbers``
#: whether it also takes a number in [0, 0.99]; a card naming no rungs takes
#: any rung of ``_EFFORT_RUNGS`` or any such number. The tokenizer repo is the
#: served weights' (a quantized repo ships its own). ``wire: "openai"`` marks
#: OpenAI's own chat-completions contract (``maps.llm.openai_wire``);
#: ``max_images`` is a per-request image limit the reasoner evicts crops to.
CARDS: dict[str, dict[str, Any]] = {
    # Thinking Machines publishes no sampling recommendation for Inkling (no
    # generation_config.json in the BF16 or the NVFP4 repo, nothing in the
    # model card): a permissive envelope, not a card-derived one. The
    # template's `reasoning_effort` ({none:0.0, minimal:0.1, low:0.2,
    # medium:0.7, high:0.9, max:0.99}, or a raw float in [0.0, 0.99])
    # DEFAULTS TO 0.9 when the request omits it — the expensive end.
    "inkling-small": {"tokenizer": "thinkingmachines/Inkling-Small-NVFP4", "top_p": 0.95, "top_k": 64,
                      "efforts": ("none", "minimal", "low", "medium", "high", "max"), "effort_numbers": True},
    # Qwen3.8-Flash-Next NVFP4, the Qwen4 preview architecture (hybrid
    # GatedDeltaNet + Qwen Sparse Attention, n-gram PLE, one MTP layer). Its
    # generation_config.json states temperature 1.0, top_p 0.95, top_k 20 and
    # nothing else; temperature is the caller's argument. The template's
    # `reasoning_effort` DEFAULTS TO xhigh — the expensive end. The serve's
    # --tool-call-parser qwen3_coder owns enforcement.
    "qwen38-flash-next-nvfp4": {"tokenizer": "RadixArk/Qwen3.8-Flash-Next-NVFP4", "top_p": 0.95, "top_k": 20,
                                "efforts": ("xhigh", "medium", "low")},
    # Gemma 4, the family envelope (top_p 0.95, top_k 64) on every size,
    # `-it` throughout: the base checkpoints carry no chat template and no
    # tool-call surface. The GGUF alias is the same weights served by
    # llama.cpp (flown aboard); a serve is not a model, and a run has to be
    # able to name which one it got. The bench NVFP4 serve adds presence
    # penalty 0.5, the setting its CityNav numbers were measured at: without
    # it the 26B loops in its thinking, and over 1,000 episodes 0.25-1.5 left
    # SR flat (.37-.39) while 0.5 cut p95 completion tokens 4,554 -> 3,337.
    "gemma4-26b-a4b-nvfp4": {"tokenizer": "nvidia/Gemma-4-26B-A4B-NVFP4", "top_p": 0.95, "top_k": 64,
                             "presence_penalty": 0.5},
    "gemma4-26b-a4b-gguf": {"tokenizer": "google/gemma-4-26B-A4B-it", "top_p": 0.95, "top_k": 64},
    "gemma4-e4b": {"tokenizer": "google/gemma-4-E4B-it", "top_p": 0.95, "top_k": 64},
    "gemma4-e2b": {"tokenizer": "google/gemma-4-E2B-it", "top_p": 0.95, "top_k": 64},
    # GPT-5.6 Sol on an Azure (Microsoft Foundry) deployment, keyed by the
    # deployment's name, which is the alias the request carries. OpenAI
    # publishes no tokenizer for it; gpt-oss-120b's o200k_harmony vocabulary
    # is the nearest published one, so prompt sizing is an estimate, and the
    # window a run pins (SEARCH_ENDPOINT_SPEC_OVERRIDE) sits far enough under
    # the served 1,050,000 that an undercount never reaches the server.
    # ponytail: Azure refuses a GPT-5.x chat-completions request carrying more
    # than 50 images ("Exceeded maximum number of images (50)"), a limit its
    # quota page does not list and no quota request raises; raise this when
    # Azure does.
    "gpt-5.6-sol": {"tokenizer": "openai/gpt-oss-120b", "wire": "openai",
                    "efforts": ("none", "low", "medium", "high", "xhigh", "max"), "max_images": 50},
}


def card_for(model_id: str | None) -> dict[str, Any]:
    """The card for a served alias, by exact name. A name the table does not
    carry is refused: a near name would draw another model's envelope."""
    alias = (model_id or "").strip().lower()
    hit = CARDS.get(alias)
    if hit is None:
        raise ValueError(f"no model card for served model {model_id!r}; serve it under one of: "
                         + ", ".join(sorted(CARDS)))
    return hit


_EFFORT_RUNGS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_EFFORT_MAX = 0.99


def check_effort(model_id: str | None, raw: str) -> None:
    """Refuse a ``reasoning_effort`` the served model's card does not take.

    A rung the card names passes; so does a number in [0, 0.99] where the
    card takes numbers, spelled as Python writes the float (``"0.90"`` is
    refused naming ``"0.9"``, ``"-0.0"`` naming ``"0.0"``), because the
    value rides the wire and the row as the string it was given. NaN, inf,
    negatives and typos are refused.
    """
    card = card_for(model_id)
    rungs = card.get("efforts", _EFFORT_RUNGS)
    if raw in rungs:
        return
    accepted = "|".join(rungs)
    if "efforts" not in card or card.get("effort_numbers"):
        accepted += " or a number in [0, 0.99]"
        try:
            value = float(raw)
        except ValueError:
            value = None
        if value is not None and 0.0 <= value <= _EFFORT_MAX:  # NaN fails the range
            canonical = repr(value + 0.0)
            if raw == canonical:
                return
            raise ValueError(f"reasoning effort {raw!r} is spelled {canonical!r}")
    raise ValueError(f"reasoning effort {raw!r} is not one {model_id!r} takes: {accepted}")


def speaks_openai(model_id: str | None) -> bool:
    """Whether the served model takes OpenAI's chat-completions contract
    rather than a self-hosted chat template's. A name with no card reads
    False, so a reader of an old record never raises on it."""
    return CARDS.get((model_id or "").strip().lower(), {}).get("wire") == "openai"


def effort(reasoning_effort: str | None = None) -> str | None:
    """The reasoning effort a request carries: the mission's own word, else
    ``SEARCH_SAMPLING_REASONING_EFFORT``, stripped; empty is none."""
    raw = reasoning_effort if reasoning_effort is not None else os.environ.get(
        "SEARCH_SAMPLING_REASONING_EFFORT", "")
    return raw.strip() or None


def sampling_kwargs(model_id: str, reasoning_effort: str | None = None) -> dict[str, Any]:
    """The request's sampling keys, flat at the top level (a hosted endpoint
    drops an ``extra_body`` object whole). A chat-template card sends
    temperature 1.0, its ``top_p``/``top_k`` (and ``presence_penalty`` where it
    names one) and ``enable_thinking``; an
    OpenAI-wire card sends none of them (1.0 is that API's default and the
    only value its reasoning models take, and it thinks unasked).

    The effort is :func:`effort`'s; it rides verbatim as a string (a serve
    refuses a JSON float for it), and none sends no key, leaving the
    template's own default."""
    card = card_for(model_id)
    if card.get("wire") == "openai":
        kwargs: dict[str, Any] = {}
    else:
        kwargs = {"temperature": 1.0, "top_p": card["top_p"], "top_k": card["top_k"],
                  "chat_template_kwargs": {"enable_thinking": True}}
        if "presence_penalty" in card:
            kwargs["presence_penalty"] = card["presence_penalty"]
    chosen = effort(reasoning_effort)
    if chosen:
        # ponytail: the request key is spelled `reasoning_effort` for every
        # card; a card that names no effort values (the gemma rows) takes any
        # well-formed value at discovery, and a template without the field
        # ignores it while the row stamps an effort the template never saw.
        # The upgrade path is declaring those cards' values once their serves
        # are checked.
        kwargs["reasoning_effort"] = chosen
    return kwargs
