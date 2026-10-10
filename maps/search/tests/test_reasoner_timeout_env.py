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


def test_unset_takes_the_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert _reasoner_timeout_s() == _REASONER_TIMEOUT_DEFAULT_S


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_takes_the_default(monkeypatch, blank):
    monkeypatch.setenv(ENV, blank)
    assert _reasoner_timeout_s() == _REASONER_TIMEOUT_DEFAULT_S


def test_the_value_the_run_scripts_export_actually_applies(monkeypatch):
    monkeypatch.setenv(ENV, "900")
    assert _reasoner_timeout_s() == 900.0
    assert _reasoner_timeout_s() != _REASONER_TIMEOUT_DEFAULT_S


def test_surrounding_whitespace_survives_the_shell(monkeypatch):
    monkeypatch.setenv(ENV, " 900 ")
    assert _reasoner_timeout_s() == 900.0


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


def test_the_reasoner_field_is_wired_to_the_env(monkeypatch):
    """The helper being right is not enough; the dataclass must call it."""
    from maps.search.parse_compile import reasoner_agent

    monkeypatch.setenv(ENV, "777")
    field = reasoner_agent.Reasoner.__dataclass_fields__["timeout"]
    assert field.default_factory() == 777.0, (
        "Reasoner.timeout no longer reads SEARCH_REASONER_TIMEOUT_S"
    )
