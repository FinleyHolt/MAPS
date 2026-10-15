"""The served model's HuggingFace tokenizer, cached per process, for prompt
sizing (the compactor) and reasoning-token counts (``flight_summary``)."""

from __future__ import annotations

import os
from typing import Any

# Imported at module load: transformers' lazy module races when several
# threads first import AutoTokenizer at once.
from transformers import AutoTokenizer

from maps.llm.model_cards import CARDS


_TOKENIZER_CACHE: dict[str, Any] = {}


def hf_tokenizer_name(model: str) -> str:
    """The tokenizer repo for a served alias: ``SEARCH_COMPACT_TOKENIZER_NAME``,
    else the model card's, else ``model`` itself."""
    env = os.environ.get("SEARCH_COMPACT_TOKENIZER_NAME", "").strip()
    if env:
        return env
    card = CARDS.get((model or "").strip().lower())
    if card is not None:
        return card["tokenizer"]
    return model


def load_tokenizer(model: str):
    """The cached tokenizer for ``model``, loaded on first call; a repo that
    cannot load raises ``RuntimeError`` naming it."""
    hf_name = hf_tokenizer_name(model)
    cached = _TOKENIZER_CACHE.get(hf_name)
    if cached is not None:
        return cached
    try:
        tok = AutoTokenizer.from_pretrained(hf_name)
    except Exception as exc:  # noqa: BLE001 — tokenizer load fails for many reasons
        raise RuntimeError(
            f"tokenizer {hf_name!r} for served model {model!r} failed to load: {exc}. "
            f"Set SEARCH_COMPACT_TOKENIZER_NAME to the HF repo or local directory that carries it."
        ) from exc
    _TOKENIZER_CACHE[hf_name] = tok
    return tok
