"""HuggingFace tokenizer factory keyed by vLLM model alias.

Process-wide cache, lazy load, override via
``SEARCH_COMPACT_TOKENIZER_NAME`` env var. Consumed by the autocompact
prompt sizer (``maps.search.analyze_scene.compactor``) and by the
transcript's reasoning-token count (``maps.llm.flight_summary``).
"""

from __future__ import annotations

import os
from typing import Any

# Eager import to dodge a thread race in transformers v5.x's ``_LazyModule``.
# ``_tokenizer()`` is called from the concurrent per-stop round threads.
# When several
# threads enter ``from transformers import AutoTokenizer`` concurrently the
# lazy module hook races and some threads raise
# ``ImportError: cannot import name 'AutoTokenizer' from 'transformers'``.
# Importing eagerly at module load resolves the lazy attribute once, before
# any worker thread exists.
from transformers import AutoTokenizer

from maps.llm.model_cards import CARDS


# Process-wide tokenizer cache, keyed by HF name.
_TOKENIZER_CACHE: dict[str, Any] = {}


def _hf_tokenizer_name(model: str) -> str:
    """Resolve the HF tokenizer repo for a served model alias.

    Order: ``SEARCH_COMPACT_TOKENIZER_NAME`` env var > the model card
    (``maps.llm.model_cards``, exact-alias hit only) > ``model`` itself.
    """
    env = os.environ.get("SEARCH_COMPACT_TOKENIZER_NAME", "").strip()
    if env:
        return env
    card = CARDS.get((model or "").strip().lower())
    if card is not None:
        return card["tokenizer"]
    return model


def _tokenizer(model: str):
    """Return the cached HuggingFace tokenizer for ``model``.

    Lazy-loaded on first call. A repo that cannot be loaded raises
    ``RuntimeError`` naming it: discovery (``endpoint_spec.discover``)
    refuses to start a run on that, and nothing sizes a prompt without a
    tokenizer.
    """
    hf_name = _hf_tokenizer_name(model)
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
