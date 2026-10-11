"""``SEARCH_REASONER_TIMEOUT_S`` reaches the per-call transport timeout.

A reader that quietly stops reading has no caller for the interpreter to
break: a slow call simply dies at the default while every caller believes it
has the exported deadline.
"""

from __future__ import annotations

import pytest

from maps.search.parse_compile.reasoner_agent import (
    _REASONER_TIMEOUT_DEFAULT_S,
    _reasoner_timeout_s,
)

ENV = "SEARCH_REASONER_TIMEOUT_S"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unset_or_blank_takes_the_default(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert _reasoner_timeout_s() == _REASONER_TIMEOUT_DEFAULT_S


@pytest.mark.parametrize("value", ["900", " 900 "])
def test_the_exported_value_applies(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    assert _reasoner_timeout_s() == 900.0 != _REASONER_TIMEOUT_DEFAULT_S


@pytest.mark.parametrize("bad", ["nine hundred", "900s", "0", "-1", "nan", "inf", "1e12"])
def test_a_value_that_cannot_apply_is_loud(monkeypatch, bad):
    """A typo must not revert to the default -- that is the original bug.

    0 is rejected with the rest: httpx reads it as "time out immediately",
    never as "no limit", so the plausible-looking spelling of "unlimited" is
    the one that would break every call.
    """
    monkeypatch.setenv(ENV, bad)
    with pytest.raises(ValueError):
        _reasoner_timeout_s()


def test_a_reasoner_takes_its_timeout_from_the_env(monkeypatch):
    from maps.search.parse_compile.reasoner_agent import Reasoner
    from maps.search.tests.fakes import TEST_SPEC

    monkeypatch.setenv(ENV, "777")
    assert Reasoner(reasoner_url="http://unused", endpoint_spec=TEST_SPEC).timeout == 777.0
