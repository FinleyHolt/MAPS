"""A 4xx ``/segment`` refusal other than 408/429 is the client's own request, refused: it raises
``SamRequestRefused`` at once, which is not a ``SamServerOutage``. A 408 or 429 moves to another replica."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

import maps.perception.sam.detect_client as dc
from tests.serves import SegmentServe

_REFUSED = {"detail": "concepts: too many items"}


def _post(first, second):
    turn = iter(range(100))
    urls = [first.url, second.url]
    return dc._post_segment_avoid_stale_pool(picker=lambda: urls[next(turn) % 2], json_body={"image_b64": ""})


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(dc, "time", SimpleNamespace(sleep=lambda _s: None, perf_counter=time.perf_counter))


def test_a_refused_request_raises_at_once_and_is_not_an_outage():
    with SegmentServe([(422, _REFUSED, {})]) as first, SegmentServe() as second:
        with pytest.raises(dc.SamRequestRefused, match="422") as caught:
            _post(first, second)
    assert (len(first.bodies), len(second.bodies)) == (1, 0)
    assert not isinstance(caught.value, dc.SamServerOutage)


@pytest.mark.parametrize("status", [408, 429])
def test_the_serves_weather_moves_the_call_to_another_replica(status):
    with SegmentServe([(status, _REFUSED, {})]) as first, SegmentServe() as second:
        assert _post(first, second) == SegmentServe.default[1]
    assert (len(first.bodies), len(second.bodies)) == (1, 1)
