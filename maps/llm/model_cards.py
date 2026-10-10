"""Client-side model cards, keyed by the discovered served alias.

One row per served model: the HF tokenizer repo (prompt sizing and
response metering) and the card-recommended thinking-mode sampling
envelope. The serve-side half of a model's facts (model path, parsers,
TP, spec-decode composition) belongs to whoever runs the serve — serve
and client run on different boxes, so each layer carries only the fields
it consumes.

Resolution: exact served-alias match (a LoRA ``base:adapter`` takes its
base's card); any other name is refused. ``presence_penalty: None`` means
the card declares no penalty term and the request carries no key.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class ModelCard(TypedDict):
    tokenizer: str
    top_p: float
    top_k: int
    presence_penalty: float | None
    # Optional envelope terms, emitted only by a card that names them. A
    # card whose publisher states them as identity values (min_p 0.0,
    # repetition_penalty 1.0) still says so rather than omitting them: the
    # request then pins the envelope instead of inheriting whatever the
    # serve's --sampling-defaults resolves from generation_config.json,
    # which is a serve fact and can move under us.
    min_p: NotRequired[float]
    repetition_penalty: NotRequired[float]
    # The ``reasoning_effort`` values the chat template takes: its named
    # rungs, and whether it also takes a number in [0, 0.99]. A card that
    # names no rungs takes any rung below or any such number.
    efforts: NotRequired[tuple[str, ...]]
    effort_numbers: NotRequired[bool]


_QWEN = ModelCard(
    tokenizer="Qwen/Qwen3.6-35B-A3B",
    top_p=0.95,
    top_k=20,
    # Breaks Qwen3 thinking-loop repetition that drove single calls to
    # 300 s tail latency; the model card warns >1.5 causes language
    # mixing — do not raise.
    presence_penalty=1.5,
)

_GEMMA = ModelCard(
    tokenizer="google/gemma-4-26B-A4B-it",
    top_p=0.95,
    top_k=64,
    presence_penalty=None,
)

# The same weights served from GGUF by llama.cpp, on its own alias. Same
# envelope and tokenizer as the SGLang row above; the alias exists because a
# serve is not a model, and a run has to be able to name which one it got.
_GEMMA_GGUF = ModelCard(
    tokenizer="google/gemma-4-26B-A4B-it",
    top_p=0.95,
    top_k=64,
    presence_penalty=None,
)

# The other four sizes. Each needs its own row: card_for() and tokenizer.py
# both take an EXACT alias hit only.
#
# The envelope is one family fact (top_p 0.95, top_k 64, no penalty term) and
# the say dialect is another, so only the tokenizer repo varies — and it
# varies to the SERVED weights, because a quantized repo ships its own
# tokenizer and prompt sizing has to be measured against what is serving.
#
# `-it` throughout. The base checkpoints (`google/gemma-4-E4B`, and its
# siblings) carry no chat template and no tool-call surface: a reasoner
# pointed at one cannot hold the mission conversation at all.
_GEMMA_E2B = ModelCard(
    tokenizer="google/gemma-4-E2B-it",
    top_p=0.95,
    top_k=64,
    presence_penalty=None,
)

_GEMMA_E4B = ModelCard(
    tokenizer="google/gemma-4-E4B-it",
    top_p=0.95,
    top_k=64,
    presence_penalty=None,
)

_GEMMA_26B_A4B_NVFP4 = ModelCard(
    tokenizer="nvidia/Gemma-4-26B-A4B-NVFP4",
    top_p=0.95,
    top_k=64,
    presence_penalty=None,
)

_GEMMA_31B_NVFP4 = ModelCard(
    tokenizer="nvidia/Gemma-4-31B-IT-NVFP4",
    top_p=0.95,
    top_k=64,
    presence_penalty=None,
)

# Thinking Machines publishes no sampling recommendation for Inkling — no
# generation_config.json in the BF16 or the NVFP4 repo, nothing in the
# model card. This is a permissive envelope, not a card-derived one. What it
# must NOT inherit is Qwen's presence_penalty=1.5, a Qwen3 thinking-loop fix
# that would be a blind guess here.
#
# Response length is governed separately by the chat template's
# `reasoning_effort` request field ({none:0.0, minimal:0.1, low:0.2,
# medium:0.7, high:0.9, max:0.99}, or a raw float in [0.0, 0.99]), which
# DEFAULTS TO 0.9 when the request omits it — the expensive end.
_INKLING = ModelCard(
    tokenizer="thinkingmachines/Inkling-Small-NVFP4",
    top_p=0.95,
    top_k=64,
    presence_penalty=None,
    efforts=("none", "minimal", "low", "medium", "high", "max"),
    effort_numbers=True,
)

# Qwen3-VL-235B-A22B-Instruct, the open same-generation counterpart to the
# hosted Qwen3-VL-Plus tier. The envelope is the checkpoint's own
# generation_config.json (top_p 0.8, top_k 20, temperature 0.7,
# repetition_penalty 1.0, no presence term) — NOT the Qwen row, whose
# presence_penalty=1.5 is a Qwen3 thinking-loop fix that would be a blind
# import into a non-thinking Instruct model. The tokenizer is the served
# NVFP4 repo, which ships its own tokenizer and chat template.
_QWEN3VL = ModelCard(
    tokenizer="nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4",
    top_p=0.8,
    top_k=20,
    presence_penalty=None,
)

# The Thinking sibling, and the one the CityNav backbone arm flies. Its own
# envelope: generation_config.json gives top_p 0.95 / top_k 20 (temperature
# 0.8), wider than the Instruct row's 0.8 / 20, and a different tokenizer
# repo because a different quantization of the weights is what is served.
# Its own row: every reader takes an EXACT alias hit.
_QWEN3VL_THINKING = ModelCard(
    tokenizer="QuantTrio/Qwen3-VL-235B-A22B-Thinking-AWQ",
    top_p=0.95,
    top_k=20,
    presence_penalty=None,
)

# The NVFP4 build of the fleet default. Same weights, so the same envelope —
# a quantizer does not move a model's sampling recommendation — but its own
# tokenizer repo, because the served repo ships one and tokenizer.py takes an
# EXACT alias hit. What the quantization DID drop is the native MTP head; that
# is a serve-side fact and lives in the card.env, not here.
_QWEN_NVFP4 = ModelCard(
    tokenizer="nvidia/Qwen3.6-35B-A3B-NVFP4",
    top_p=0.95,
    top_k=20,
    presence_penalty=1.5,
)

# The SAME base model at a different 4-bit scheme, and it takes its own row
# because it is a different CHECKPOINT: nvidia/ ships W4A16_NVFP4, which is
# weight-only and never fires the FP4 tensor cores, while RedHatAI/ ships
# W4A4 with activations quantized too. Same base weights and the same
# thinking-loop presence term, so only the tokenizer repo differs.
#
# A SEPARATE ALIAS rather than a flag, for the reason the fp8/nvfp4 split
# already has one: the alias is what run provenance records, and two arms
# both reporting `qwen36-35b-a3b-nvfp4` could not be told apart in the ledger
# afterwards. The two repos additionally share a BASENAME, which is how a
# probe served nvidia's weights under the W4A4 card and returned a perfectly
# plausible 249.76 tok/s.
_QWEN_NVFP4_W4A4 = ModelCard(
    tokenizer="RedHatAI/Qwen3.6-35B-A3B-NVFP4",
    top_p=0.95,
    top_k=20,
    presence_penalty=1.5,
)

# Qwen3.8-27B NVFP4 — a DIFFERENT MODEL from the Qwen3.6 rows above, not a
# requantization of one, so it takes its publisher's envelope rather than
# inheriting theirs: no presence_penalty, which on _QWEN is a Qwen3.6
# thinking-loop fix and NOT part of this model's recommendation. The full published thinking-mode set is
# temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0,
# repetition_penalty 1.0; temperature is the caller's argument
# (sampling.thinking_sampling_kwargs already passes 1.0 for reasoner turns)
# and the rest are stated here. presence_penalty 0.0 is spelled as None —
# "declare no penalty term, send no key" — which is the same request the
# server sees and the shape the schema already has for it.
_QWEN38_NVFP4 = ModelCard(
    tokenizer="RadixArk/Qwen3.8-27B-NVFP4",
    top_p=0.95,
    top_k=20,
    presence_penalty=None,
    min_p=0.0,
    repetition_penalty=1.0,
)

# Qwen3.8-Flash-Next NVFP4 — the Qwen4 preview architecture (hybrid
# GatedDeltaNet + Qwen Sparse Attention, n-gram PLE, one MTP layer), and again
# a DIFFERENT MODEL rather than a requantization of any row above, so it takes
# its own publisher's envelope. Its generation_config.json states exactly three
# terms — temperature 1.0, top_p 0.95, top_k 20 — and nothing else; temperature
# is the caller's argument, so the two stated here are the whole envelope. No
# min_p or repetition_penalty: the publisher declares neither, and inventing
# identity values would pin an envelope the model card never claimed.
#
# presence_penalty stays OFF: _QWEN's 1.5 is a Qwen3.6 thinking-loop fix,
# not part of this model's recommendation. The tokenizer is the served NVFP4 repo, which ships its own tokenizer and chat
# template, and tokenizer.py takes an EXACT alias hit only.
#
# Response length here is governed by the chat template's `reasoning_effort`
# field (xhigh | medium | low), which DEFAULTS TO xhigh — the expensive end.
# Nothing in this envelope reaches it, and the serve deliberately does not pin
# it; see the card.env.
_QWEN38_FLASH_NEXT_NVFP4 = ModelCard(
    tokenizer="RadixArk/Qwen3.8-Flash-Next-NVFP4",
    top_p=0.95,
    top_k=20,
    presence_penalty=None,
    efforts=("xhigh", "medium", "low"),
    # The serve's --tool-call-parser qwen3_coder owns enforcement.
)

# Muse Glimmer 30B (Meta Superintelligence Lab). The envelope is the model
# card's own recommendation — temperature 1.0, top_p 0.95, top_k 64 — and NOT
# the Qwen row's, whose presence_penalty=1.5 is a Qwen3 thinking-loop fix that
# would be a blind import into a different family.
#
# This model's reasoning control is a SYSTEM-PROMPT line
# (``Reasoning strength: low|medium|high|xhigh``), not a request field, so
# nothing in this envelope reaches it. See the card's SERVE.md.
# sglang's muse detector hard-returns supports_structural_tag() False and
# xgrammar ships no muse builtin tag, so `strict: true` compiles no
# constraint on this family. It does not need to: the surface is one tool
# with one string argument, and the parser refuses what the envelope cannot.
_MUSE = ModelCard(
    tokenizer="meta-models/Muse-Glimmer-30B",
    top_p=0.95,
    top_k=64,
    presence_penalty=None,
)

_MUSE_NVFP4 = ModelCard(
    tokenizer="RadixArk/Muse-Glimmer-NVFP4",
    top_p=0.95,
    top_k=64,
    presence_penalty=None,
)

# Qwen3.5, four served sizes. Each needs its own row: every reader takes an
# EXACT alias hit, and the tokenizer repo is the SERVED one because every quantized repo here ships
# its own.
#
# presence_penalty is None on all four, and that is a declaration rather than
# an oversight. The 1.5 on _QWEN was MEASURED against Qwen3.6 thinking loops;
# Qwen3.5 is a different generation, publishes no penalty recommendation, and
# has not been measured here. Carrying a number across on a family-name
# argument is how a card grows a term nobody can defend. The envelope below is
# the one the checkpoints' own generation_config.json declares.
_QWEN35_4B = ModelCard(
    tokenizer="AxionML/Qwen3.5-4B-NVFP4",
    top_p=0.95,
    top_k=20,
    presence_penalty=None,
)
_QWEN35_9B = ModelCard(
    tokenizer="ig1/Qwen3.5-9B-NVFP4",
    top_p=0.95,
    top_k=20,
    presence_penalty=None,
)
_QWEN35_27B = ModelCard(
    tokenizer="apolo13x/Qwen3.5-27B-NVFP4",
    top_p=0.95,
    top_k=20,
    presence_penalty=None,
)
_QWEN35_122B_A10B = ModelCard(
    tokenizer="nvidia/Qwen3.5-122B-A10B-NVFP4",
    top_p=0.95,
    top_k=20,
    presence_penalty=None,
)

CARDS: dict[str, ModelCard] = {
    "qwen36-35b-a3b": _QWEN,
    "qwen36-35b-a3b-nvfp4": _QWEN_NVFP4,
    "qwen36-35b-a3b-nvfp4-w4a4": _QWEN_NVFP4_W4A4,
    "qwen35-4b-nvfp4": _QWEN35_4B,
    "qwen35-9b-nvfp4": _QWEN35_9B,
    "qwen35-27b-nvfp4": _QWEN35_27B,
    "qwen35-122b-a10b-nvfp4": _QWEN35_122B_A10B,
    "qwen38-27b-nvfp4": _QWEN38_NVFP4,
    "qwen38-flash-next-nvfp4": _QWEN38_FLASH_NEXT_NVFP4,
    "qwen3vl-235b-a22b": _QWEN3VL,
    "qwen3vl-235b-a22b-thinking": _QWEN3VL_THINKING,
    "gemma4-26b-a4b": _GEMMA,
    "gemma4-26b-a4b-gguf": _GEMMA_GGUF,
    "gemma4-26b-a4b-nvfp4": _GEMMA_26B_A4B_NVFP4,
    "gemma4-31b-nvfp4": _GEMMA_31B_NVFP4,
    "gemma4-e4b": _GEMMA_E4B,
    "gemma4-e2b": _GEMMA_E2B,
    "inkling-small": _INKLING,
    "muse-glimmer-30b": _MUSE,
    "muse-glimmer-30b-nvfp4": _MUSE_NVFP4,
}

def card_for(model_id: str | None) -> ModelCard:
    """The card for a served alias, by exact name (a LoRA ``base:adapter``
    rides its base's card). A name the table does not carry is refused: a
    near name would draw another model's envelope."""
    alias = (model_id or "").strip().lower().partition(":")[0]
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
    kwargs: dict[str, Any] = {"top_p": card["top_p"], "top_k": card["top_k"]}
    if card["presence_penalty"] is not None:
        kwargs["presence_penalty"] = card["presence_penalty"]
    for opt in ("min_p", "repetition_penalty"):
        if opt in card:
            kwargs[opt] = card[opt]  # type: ignore[literal-required]
    return kwargs
