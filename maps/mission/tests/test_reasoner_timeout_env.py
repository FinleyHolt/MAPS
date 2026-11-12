"""``SEARCH_REASONER_TIMEOUT_S`` reaches the per-call transport timeout."""

from __future__ import annotations

import pytest

from maps.llm.endpoint_spec import REASONER_TIMEOUT_DEFAULT_S, reasoner_timeout_s
from maps.mission.reasoner import Reasoner
from maps.mission.tests.fakes import TEST_SPEC
from maps.settings import from_environ

ENV = "SEARCH_REASONER_TIMEOUT_S"


def _endpoint(value: str):
    """The endpoint settings of an environment whose SEARCH_REASONER_TIMEOUT_S is ``value``."""
    return from_environ({ENV: value}).endpoint


def test_the_env_sets_the_timeout_and_a_value_that_cannot_apply_is_loud():
    assert reasoner_timeout_s(_endpoint("   ").reasoner_timeout_s) == REASONER_TIMEOUT_DEFAULT_S
    assert Reasoner(endpoint_spec=TEST_SPEC, settings=from_environ({ENV: " 900 "})).timeout == 900.0
    assert (
        Reasoner(endpoint_spec=TEST_SPEC, settings=from_environ({})).timeout
        == REASONER_TIMEOUT_DEFAULT_S
    )
    for bad in ("900s", "0", "nan", "1e12"):
        with pytest.raises(ValueError):
            reasoner_timeout_s(_endpoint(bad).reasoner_timeout_s)
        with pytest.raises(ValueError):
            Reasoner(endpoint_spec=TEST_SPEC, settings=from_environ({ENV: bad}))


def test_the_default_wait_outlasts_one_full_turn_on_a_loaded_serve():
    # 40,000 output tokens at ~20 tok/s per stream under a saturated batch.
    assert REASONER_TIMEOUT_DEFAULT_S >= 40_000 / 20
