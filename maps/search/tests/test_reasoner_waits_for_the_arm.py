"""A live mission holds its turn while the arm is down.

The bench has a transient budget: six link faults in one round end the
episode as ``transient_failures``. A LIVE mission (the reasoner carries a
cancel_event) has none. The arm going down mid-flight is a VPN cookie or a
relay to bring back, not a verdict on the run, so the loop waits on the
cancel event between attempts, capped at thirty seconds, and resumes the
moment the reasoner answers. Stop is the one way out: the wait raises
``CallCanceled``, which the station reads as ``stopped``.
"""

from __future__ import annotations

import threading

import httpx
import pytest

from maps.llm.endpoint_spec import EndpointSpec
from maps.llm.http import CallCanceled
from maps.search.parse_compile.reasoner_agent import Reasoner
from maps.search.parse_compile.reasoner_loop import (
    _RETRIABLE_LLM_ERRORS,
    MissionReply,
    MissionSpec,
    advance_reasoner_one_round,
)

_TEST_SPEC = EndpointSpec(
    base_url="http://unused",
    model_id="inkling-small",
    max_model_len=262144,
    tokenizer_name="qwen",
    output_reserve=40000,
)

_REPLY = {"kind": "text", "content": "the goal is the flown object",
          "reasoning": ""}


class _Stop(threading.Event):
    """A cancel event that records each wait and sets itself on the nth."""

    def __init__(self, set_on: int | None = None) -> None:
        super().__init__()
        self.waits: list[float] = []
        self.set_on = set_on

    def wait(self, timeout=None):  # noqa: D102
        self.waits.append(timeout)
        if self.set_on is not None and len(self.waits) >= self.set_on:
            self.set()
        return self.is_set()


def _round(monkeypatch, faults: int, *, cancel_event=None, transcript=None,
           fault: BaseException = httpx.ConnectError("relay down")):
    monkeypatch.setattr(
        "maps.search.parse_compile.reasoner_loop.time.sleep", lambda _s: None)
    turns = iter([fault] * faults + [_REPLY] * 5)

    def _step(tools):
        nxt = next(turns)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    r = Reasoner(reasoner_url="http://unused", endpoint_spec=_TEST_SPEC,
                 cancel_event=cancel_event)
    r.step_tool_call = _step

    class _Ctx:
        pass

    return advance_reasoner_one_round(
        reasoner=r, user_message="find the car", ctx=_Ctx(),
        compute_namespace={"cands": [], "anchors": []},
        rid_tag="h0", transcript=transcript,
        mission_spec=MissionSpec(
            fly=lambda item: {"stop_id": "stop_000", "target": item["target_fid"],
                              "pose_xy": [0.0, 0.0], "new_detections": 0},
            declare_concepts=lambda items: {"declared": []},
            flight_budget=4, turn_cap=16))


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
    assert len({r["since"] for r in faults}) == 1


def test_stop_is_the_one_way_out_of_the_wait(monkeypatch):
    with pytest.raises(CallCanceled):
        _round(monkeypatch, faults=9, cancel_event=_Stop(set_on=3))


def test_the_bench_keeps_its_budget(monkeypatch):
    reply = _round(monkeypatch, faults=6)
    assert reply.ended_by == "transient_failures"


def _status(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://serve.invalid/v1/chat/completions")
    return httpx.HTTPStatusError(f"{code}", request=req,
                                 response=httpx.Response(code, request=req))


@pytest.mark.parametrize("code, ended_by", [
    (400, "request_refused"), (422, "request_refused"),
    (408, "transient_failures"), (429, "transient_failures"),
    (503, "transient_failures")])
def test_a_refused_request_is_the_methods_failure(monkeypatch, code, ended_by):
    """A 4xx other than 408/429 is the serve refusing what was sent: the
    budget still retries it, and when it runs out the mission ends
    ``request_refused``, which is scored, not ``transient_failures``."""
    reply = _round(monkeypatch, faults=6, fault=_status(code))
    assert reply.ended_by == ended_by


def test_a_dead_relay_port_is_a_link_fault():
    assert issubclass(httpx.ConnectTimeout, _RETRIABLE_LLM_ERRORS)
    assert issubclass(httpx.PoolTimeout, _RETRIABLE_LLM_ERRORS)
