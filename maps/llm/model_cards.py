"""Client-side model cards, keyed by the discovered served alias.

One row per served model: the HF tokenizer repo (prompt sizing and
response metering), the card-recommended thinking-mode sampling envelope,
and the ``reasoning_effort`` values its chat template takes. The
serve-side half of a model's facts (model path, parsers, TP, spec-decode
composition) belongs to whoever runs the serve — serve and client run on
different boxes, so each layer carries only the fields it consumes.

Resolution: exact served-alias match; any other name is refused. No card
declares a presence penalty, so no request carries one.
"""

from __future__ import annotations

from typing import Any

#: alias -> {tokenizer, top_p, top_k[, efforts][, effort_numbers]}.
#: ``efforts`` are the ``reasoning_effort`` rungs the chat template takes and
#: ``effort_numbers`` whether it also takes a number in [0, 0.99]; a card that
#: names no rungs takes any rung of ``_EFFORT_RUNGS`` or any such number.
#: Each served alias needs its own row (card_for and tokenizer.py take an
#: EXACT alias hit only). The tokenizer repo is the SERVED weights', because a
#: quantized repo ships its own tokenizer and prompt sizing has to be measured
#: against what is serving.
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
    # Gemma 4, the family envelope (top_p 0.95, top_k 64, no penalty term) on
    # every size, `-it` throughout: the base checkpoints carry no chat
    # template and no tool-call surface. The GGUF alias is the same weights
    # served by llama.cpp (flown aboard); a serve is not a model, and a run
    # has to be able to name which one it got.
    "gemma4-26b-a4b-nvfp4": {"tokenizer": "nvidia/Gemma-4-26B-A4B-NVFP4", "top_p": 0.95, "top_k": 64},
    "gemma4-26b-a4b-gguf": {"tokenizer": "google/gemma-4-26B-A4B-it", "top_p": 0.95, "top_k": 64},
    "gemma4-e4b": {"tokenizer": "google/gemma-4-E4B-it", "top_p": 0.95, "top_k": 64},
    "gemma4-e2b": {"tokenizer": "google/gemma-4-E2B-it", "top_p": 0.95, "top_k": 64},
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


def envelope_kwargs(model_id: str | None) -> dict[str, Any]:
    """The card's sampling fields as flat request kwargs (no temperature)."""
    card = card_for(model_id)
    return {"top_p": card["top_p"], "top_k": card["top_k"]}
