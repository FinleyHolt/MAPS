"""HuggingFace tokenizer factory keyed by vLLM model alias.

Process-wide cache, lazy load, override via
``SEARCH_COMPACT_TOKENIZER_NAME`` env var. Consumed by the autocompact
prompt sizer (``maps.search.analyze_scene.compactor``) and by the response-side
metering (``maps.llm.net_meter``); placing the helpers here breaks
the otherwise-cyclic dep between those two modules and keeps the
infrastructure → robotics arrow pointing the right way.
"""

from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)


# Process-wide tokenizer cache, keyed by HF name.
_TOKENIZER_CACHE: dict[str, Any] = {}

_TOKENIZER_LOAD_FAILURES: set[str] = set()


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
    """Return a cached HuggingFace tokenizer, or ``None`` if load fails.

    Lazy-loaded on first call. Returns ``None`` (cached) when the HF repo
    cannot be resolved (offline runs, unknown model alias, missing
    ``SEARCH_COMPACT_TOKENIZER_NAME``). ``None`` is tolerated only under a
    declared offline run (``SEARCH_ENDPOINT_OFFLINE=1``): on a live run
    ``endpoint_spec.discover()`` verifies the tokenizer loads at startup
    (its case g), and the compactor raises rather than sending an
    unmeasured prompt. The first failure logs a WARNING; subsequent calls
    are silent.
    """
    hf_name = _hf_tokenizer_name(model)
    cached = _TOKENIZER_CACHE.get(hf_name)
    if cached is not None:
        return cached
    if hf_name in _TOKENIZER_LOAD_FAILURES:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(hf_name)
    except Exception as exc:  # noqa: BLE001 — tokenizer load fails for many reasons
        _TOKENIZER_LOAD_FAILURES.add(hf_name)
        log.warning(
            "tokenizer load failed for %r: %s; "
            "set SEARCH_COMPACT_TOKENIZER_NAME to override.",
            hf_name,
            exc,
        )
        return None
    _TOKENIZER_CACHE[hf_name] = tok
    return tok
