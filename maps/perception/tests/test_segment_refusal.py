"""The ``/segment`` poster's one retry layer. A 4xx other than 408/429 is the client's own request,
refused: it raises ``SamRequestRefused`` at once, which is not a ``SamServerOutage``. A connect error,
408, 429 or 5xx is asked again of the next replica after a 3, 6, 12 s backoff; a read timeout is never
resent to the replica that timed out; one per-tile deadline bounds the whole."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

import maps.perception.sam.detect_client as dc
from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider
from tests.serves import SegmentServe

_REFUSED = {"detail": "concepts: too many items"}


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    """Backoff sleeps are recorded, not slept, and the clock the deadline reads advances by them."""
    slept: list[float] = []
    monkeypatch.setattr(dc, "time", SimpleNamespace(sleep=slept.append, monotonic=lambda: sum(slept)))
    monkeypatch.setattr(dc, "_rr", itertools.count())
    return slept


def _post(*serves_or_urls):
    urls = [getattr(s, "url", s) for s in serves_or_urls]
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SAM_SERVER_URL", ",".join(urls))
        mp.delenv("SAM_VIDEO_SERVER_URL", raising=False)
        return dc.SamDetectClient(concept_provider=ReasonerUnionConceptProvider())._post_tile(
            np.zeros((8, 8, 3), np.uint8), ["car"])


def test_a_refused_request_raises_at_once_and_is_not_an_outage():
    with SegmentServe([(422, _REFUSED, {})]) as first, SegmentServe() as second:
        with pytest.raises(dc.SamRequestRefused, match="422") as caught:
            _post(first, second)
    assert (len(first.bodies), len(second.bodies)) == (1, 0)
    assert not isinstance(caught.value, dc.SamServerOutage)


@pytest.mark.parametrize("status", [408, 429, 503])
def test_the_serves_weather_moves_the_call_to_another_replica_after_the_first_backoff(status, clock):
    with SegmentServe([(status, _REFUSED, {})]) as first, SegmentServe() as second:
        assert _post(first, second) == SegmentServe.default[1]
    assert (len(first.bodies), len(second.bodies)) == (1, 1)
    assert clock == [3.0]


def test_a_serve_that_keeps_failing_is_asked_four_times_over_a_3_6_12_s_backoff(clock):
    with SegmentServe([(503, {}, {})] * 4) as only:
        assert _post(only) is None
    assert len(only.bodies) == 4 and clock == [3.0, 6.0, 12.0]


def test_a_read_timeout_is_never_resent_to_the_replica_that_timed_out(monkeypatch):
    monkeypatch.setattr(dc, "SEGMENT_READ_TIMEOUT_S", 0.3)
    with SegmentServe(hold="before_headers", hold_s=1.0) as slow:
        assert _post(slow) is None
    assert len(slow.bodies) == 1


def test_a_read_timeout_moves_the_call_to_the_next_replica(monkeypatch):
    monkeypatch.setattr(dc, "SEGMENT_READ_TIMEOUT_S", 0.3)
    with SegmentServe(hold="before_headers", hold_s=1.0) as slow, SegmentServe() as second:
        assert _post(slow, second) == SegmentServe.default[1]
    assert (len(slow.bodies), len(second.bodies)) == (1, 1)


def test_the_per_tile_deadline_ends_the_backoff_ladder(monkeypatch, clock):
    monkeypatch.setattr(dc, "SEGMENT_DEADLINE_S", 10.0)
    assert _post("http://127.0.0.1:9") is None      # nothing listens: every connect is refused
    assert clock == [3.0, 6.0]
