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


@pytest.mark.parametrize("module", ["maps.llm.flight_summary", "maps.mission.endpoint_preflight"])
def test_importing_a_cli_module_leaves_transformers_unloaded(module):
    """A CLI that reads run records or preflights an endpoint starts without transformers' import
    cost or banner."""
    probe = f"import sys, {module}; print('transformers' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
