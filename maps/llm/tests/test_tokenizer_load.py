"""A tokenizer that cannot load raises naming its repo; nothing sizes a prompt without one."""

from __future__ import annotations

import pytest

from maps.llm.tokenizer import load_tokenizer


def test_a_tokenizer_that_cannot_load_raises(monkeypatch, tmp_path):
    absent = str(tmp_path / "no-tokenizer-here")
    monkeypatch.setenv("SEARCH_COMPACT_TOKENIZER_NAME", absent)
    with pytest.raises(RuntimeError, match="no-tokenizer-here"):
        load_tokenizer("inkling-small")
