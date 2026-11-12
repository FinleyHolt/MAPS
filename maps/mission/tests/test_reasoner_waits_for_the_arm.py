from __future__ import annotations

import threading

import httpx
import pytest

from maps.llm.http import CallCanceled
from maps.mission.loop import _Mission
from maps.mission.reasoner import DegenerateCompletion, Reasoner
from maps.mission.serve_refusals import RETRIABLE_LLM_ERRORS
from maps.mission.spec import MissionReply, MissionSpec
from maps.mission.tests.fakes import TEST_SPEC, bare_ctx, unarmed_boost

_REPLY = {"kind": "text", "content": "the goal is the flown object", "reasoning": ""}


class _Stop(threading.Event):
    """A cancel event that records each wait and sets itself on the nth."""

    def __init__(self, set_on: int | None = None) -> None:
        super().__init__()
        self.waits: list[float] = []
        self.set_on = set_on

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.set_on is not None and len(self.waits) >= self.set_on:
            self.set()
        return self.is_set()


def _round(
    monkeypatch,
    faults: int,
    *,
    cancel_event=None,
    transcript=None,
    fault: BaseException = httpx.ConnectError("relay down"),
):
    monkeypatch.setattr("maps.mission.loop.time.sleep", lambda _s: None)
    turns = iter([fault] * faults + [_REPLY] * 5)

    def _step(tools):
        nxt = next(turns)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    r = Reasoner(endpoint_spec=TEST_SPEC, cancel_event=cancel_event)
    r.step_tool_call = _step

    return _Mission(
        boost_queue=unarmed_boost,
        reasoner=r,
        ctx=bare_ctx(),
        compute_namespace={"cands": [], "anchors": []},
        rid_tag="h0",
        transcript=transcript,
        mission_spec=MissionSpec(
            fly=lambda item: {
                "stop_id": "stop_000",
                "target": item["target_fid"],
                "pose_xy": [0.0, 0.0],
                "new_detections": 0,
            },
            declare_concepts=lambda items: None,
            flight_budget=4,
            turn_cap=16,
        ),
    ).converse("find the car")


def test_a_live_mission_outlasts_the_bench_budget_and_resumes(monkeypatch):
    stop = _Stop()
    rows: list[dict] = []
    reply = _round(monkeypatch, faults=16, cancel_event=stop, transcript=rows)
    assert isinstance(reply, MissionReply)
    assert reply.ended_by != "transient_failures"
    assert stop.waits == [min(30.0, 0.5 + 2.0 * n) for n in range(1, 17)]
    assert max(stop.waits) == 30.0
    faults = [r for r in rows if r["kind"] == "transient_error"]
    assert [r["attempt"] for r in faults] == list(range(1, 17))


def test_stop_is_the_one_way_out_of_the_wait(monkeypatch):
    with pytest.raises(CallCanceled):
        _round(monkeypatch, faults=9, cancel_event=_Stop(set_on=3))


def test_a_collapsed_completion_is_retried_not_terminal(monkeypatch):
    reply = _round(monkeypatch, faults=1, fault=DegenerateCompletion("collapse"))
    assert reply.ended_by != "transient_failures"


def test_the_bench_keeps_its_budget(monkeypatch):
    reply = _round(monkeypatch, faults=6)
    assert reply.ended_by == "transient_failures"


def _status(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://serve.invalid/v1/chat/completions")
    return httpx.HTTPStatusError(f"{code}", request=req, response=httpx.Response(code, request=req))


@pytest.mark.parametrize(
    "code, ended_by",
    [
        (400, "request_refused"),
        (422, "request_refused"),
        (402, "transient_failures"),
        (408, "transient_failures"),
        (429, "transient_failures"),
        (503, "transient_failures"),
    ],
)
def test_a_refused_request_is_the_methods_failure(monkeypatch, code, ended_by):
    """A 4xx other than 402/408/429 ends the mission request_refused, which is scored."""
    reply = _round(monkeypatch, faults=6, fault=_status(code))
    assert reply.ended_by == ended_by


def test_a_refused_request_is_sent_once_on_the_bench_and_live(monkeypatch):
    """A 400 ends the mission at its first send: re-sending fails the same way."""
    for stop in (None, _Stop()):
        rows: list[dict] = []
        reply = _round(
            monkeypatch, faults=6, fault=_status(400), cancel_event=stop, transcript=rows
        )
        assert reply.ended_by == "request_refused"
        assert reply.turns_used == 1
        assert [r for r in rows if r["kind"] == "transient_error"] == []
        assert stop is None or stop.waits == []


def test_a_dead_relay_port_is_a_link_fault():
    assert issubclass(httpx.ConnectTimeout, RETRIABLE_LLM_ERRORS)
    assert issubclass(httpx.PoolTimeout, RETRIABLE_LLM_ERRORS)
