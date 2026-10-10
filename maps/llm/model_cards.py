"""Client-side model cards, keyed by the discovered served alias.

One row per served model: the HF tokenizer repo (prompt sizing and
response metering) and the card-recommended thinking-mode sampling
envelope. The serve-side half of a model's facts (model path, parsers,
TP, spec-decode composition) belongs to whoever runs the serve — serve
and client run on different boxes, so each layer carries only the fields
it consumes.

Resolution: exact served-alias match, else family substring, else the
Qwen card (the fleet default serve). ``presence_penalty: None`` means
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

# The other four sizes. Each needs its own row rather than riding the family
# substring, because the two readers of this dict disagree — card_for() falls
# back through _FAMILY_SUBSTRING, but tokenizer.py takes an EXACT alias hit
# only. Absent these rows 'gemma4-e4b' would draw the gemma envelope and then
# try to load a tokenizer named 'gemma4-e4b'.
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
# Its own row rather than the family substring, because tokenizer.py takes an
# EXACT alias hit only — the trap the _GEMMA_E4B comment above records.
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
# inheriting theirs. It needs its own row for one concrete reason: without it
# the alias falls through _FAMILY_SUBSTRING's "qwen" to _QWEN and picks up
# presence_penalty=1.5, which is a Qwen3.6 thinking-loop fix and NOT part of
# this model's recommendation. The full published thinking-mode set is
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
# The row exists to keep presence_penalty OFF. Without it the alias falls
# through _FAMILY_SUBSTRING's "qwen" to _QWEN and picks up presence_penalty=1.5
# — a Qwen3.6 thinking-loop fix, not part of this model's recommendation. The
# tokenizer is the served NVFP4 repo, which ships its own tokenizer and chat
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

# Qwen3.5, four served sizes. Each needs its own row rather
# than riding the family substring: tokenizer.py takes an EXACT alias hit, and
# the tokenizer repo is the SERVED one because every quantized repo here ships
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

# Order matters: "qwen3vl" precedes "qwen", or every Qwen3-VL alias draws the
# Qwen3.6 tokenizer and prompt sizing goes silently wrong — the same trap the
# _GEMMA_E4B row above exists to close.
_FAMILY_SUBSTRING: list[tuple[str, ModelCard]] = [
    ("gemma", _GEMMA),
    ("inkling", _INKLING),
    ("muse", _MUSE),
    ("qwen3vl", _QWEN3VL),
    ("qwen", _QWEN),
]


def card_for(model_id: str | None) -> ModelCard:
    """The card for a served alias; family substring, then Qwen, as fallback."""
    alias = (model_id or "").strip().lower()
    hit = CARDS.get(alias)
    if hit is not None:
        return hit
    for needle, card in _FAMILY_SUBSTRING:
        if needle in alias:
            return card
    return _QWEN


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
