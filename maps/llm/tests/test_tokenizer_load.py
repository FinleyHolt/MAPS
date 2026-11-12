"""A tokenizer that cannot load raises naming its repo; nothing sizes a prompt without one."""

from __future__ import annotations

import pytest

from maps.llm.tokenizer import load_tokenizer
from maps.settings import from_environ


def test_a_tokenizer_that_cannot_load_raises(tmp_path):
    """The run's tokenizer override (SEARCH_COMPACT_TOKENIZER_NAME) names the repo that is loaded."""
    absent = from_environ(
        {"SEARCH_COMPACT_TOKENIZER_NAME": str(tmp_path / "no-tokenizer-here")}
    ).endpoint.tokenizer_name
    with pytest.raises(RuntimeError, match="no-tokenizer-here"):
        load_tokenizer("inkling-small", absent)
