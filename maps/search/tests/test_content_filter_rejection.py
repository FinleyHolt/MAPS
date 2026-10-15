"""A prompt a hosted endpoint's content filter refuses ends the mission at once:
the same prompt is refused the same way, so a retry is a wasted call."""

from __future__ import annotations

import httpx
import pytest

from maps.search.parse_compile.reasoner_loop import _is_content_filter_rejection
from maps.search.tests import fakes


def _status_error(status: int, body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.openai.azure.com/openai/v1/chat/completions")
    response = httpx.Response(status, text=body, request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


# Azure's documented shape for a filtered prompt.
AZURE_FILTERED = (
    '{"error": {"message": "The response was filtered due to the prompt triggering Azure '
    'OpenAI\'s content management policy.", "type": null, "param": "prompt", "code": '
    '"content_filter", "status": 400, "innererror": {"code": "ResponsibleAIPolicyViolation"}}}'
)


@pytest.mark.parametrize("body", [AZURE_FILTERED, '{"error": {"code": "content_filter"}}',
                                  '{"error": {"innererror": {"code": "ResponsibleAIPolicyViolation"}}}'])
def test_azure_s_filtered_prompt_is_recognised(body: str) -> None:
    assert _is_content_filter_rejection(_status_error(400, body))


@pytest.mark.parametrize("err", [
    _status_error(400, "Failed to parse tool call arguments against schema"),
    _status_error(429, AZURE_FILTERED),
    _status_error(500, "content_filter service unavailable"),
    httpx.ConnectError("tunnel flap"),
    ValueError("empty body"),
])
def test_everything_else_is_not_a_filter(err) -> None:
    assert not _is_content_filter_rejection(err)


def test_the_mission_ends_on_the_first_filtered_request(monkeypatch) -> None:
    calls: list[int] = []

    def _refused(self, tools):
        calls.append(1)
        raise _status_error(400, AZURE_FILTERED)

    monkeypatch.setattr(fakes.FakeReasoner, "step_tool_call", _refused)
    reply, *_ = fakes.drive([])
    assert reply.ended_by == "content_filter"
    assert calls == [1], "the refused prompt was sent again"
