"""``SEARCH_REASONER_TIMEOUT_S`` reaches the per-call transport timeout."""

from __future__ import annotations

import pytest

from maps.llm.endpoint_spec import REASONER_TIMEOUT_DEFAULT_S, reasoner_timeout_s
from maps.mission.reasoner import Reasoner
from maps.mission.tests.fakes import TEST_SPEC

ENV = "SEARCH_REASONER_TIMEOUT_S"


def test_the_env_sets_the_timeout_and_a_value_that_cannot_apply_is_loud(monkeypatch):
    monkeypatch.setenv(ENV, "   ")
    assert reasoner_timeout_s() == REASONER_TIMEOUT_DEFAULT_S
    monkeypatch.setenv(ENV, " 900 ")
    assert Reasoner(endpoint_spec=TEST_SPEC).timeout == 900.0
    for bad in ("900s", "0", "nan", "1e12"):
        monkeypatch.setenv(ENV, bad)
        with pytest.raises(ValueError):
            reasoner_timeout_s()


def test_the_default_wait_outlasts_one_full_turn_on_a_loaded_serve():
    # 40,000 output tokens at ~20 tok/s per stream under a saturated batch.
    assert REASONER_TIMEOUT_DEFAULT_S >= 40_000 / 20
