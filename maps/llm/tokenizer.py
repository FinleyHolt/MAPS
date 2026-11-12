"""The served model's HuggingFace tokenizer, cached per process, for prompt
sizing (the compactor) and reasoning-token counts (``flight_summary``)."""

from __future__ import annotations

import threading
from typing import Any

from maps.llm.model_cards import CARDS

# transformers' lazy module races when several threads first import
# AutoTokenizer at once, so the deferred import runs under one lock.
_IMPORT_LOCK = threading.Lock()

# A pure cache: a tokenizer is immutable once loaded, so one per repo serves
# every mission in the process.
_TOKENIZER_CACHE: dict[str, Any] = {}


def hf_tokenizer_name(model: str, tokenizer_name: str = "") -> str:
    """The tokenizer repo for a served alias: ``tokenizer_name`` (the run's
    ``SEARCH_COMPACT_TOKENIZER_NAME``, ``Endpoint.tokenizer_name``), else the
    model card's, else ``model`` itself."""
    if tokenizer_name:
        return tokenizer_name
    card = CARDS.get((model or "").strip().lower())
    if card is not None:
        return card["tokenizer"]
    return model


def load_tokenizer(model: str, tokenizer_name: str = ""):
    """The cached tokenizer for ``model`` (:func:`hf_tokenizer_name`), loaded
    on first call; a repo that cannot load raises ``RuntimeError`` naming it."""
    hf_name = hf_tokenizer_name(model, tokenizer_name)
    cached = _TOKENIZER_CACHE.get(hf_name)
    if cached is not None:
        return cached
    with _IMPORT_LOCK:
        from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(hf_name)
    except Exception as exc:
        raise RuntimeError(
            f"tokenizer {hf_name!r} for served model {model!r} failed to load: {exc}. "
            f"Set SEARCH_COMPACT_TOKENIZER_NAME to the HF repo or local directory that carries it."
        ) from exc
    _TOKENIZER_CACHE[hf_name] = tok
    return tok
