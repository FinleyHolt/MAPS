"""A tokenizer that cannot load raises naming its repo; nothing sizes a prompt without one."""

from __future__ import annotations

import subprocess
import sys

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


def test_importing_flight_summary_leaves_transformers_unloaded():
    """Importing the flight summary leaves transformers unloaded: a CLI that reads run records starts
    without its import cost or banner."""
    probe = "import sys, maps.llm.flight_summary; print('transformers' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
