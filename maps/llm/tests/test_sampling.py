from __future__ import annotations

import pytest

from maps.llm.model_cards import card_for, sampling_kwargs
from maps.llm.vision_content import build_image_part, build_vision_user_content


@pytest.fixture(autouse=True)
def _no_ambient_sampling_env(monkeypatch) -> None:
    """The envelope pins measure the code, not a shell that sourced an arm file."""
    monkeypatch.delenv("SEARCH_SAMPLING_REASONING_EFFORT", raising=False)


@pytest.mark.parametrize("model, top_k, extra", [
    ("qwen38-flash-next-nvfp4", 20, {}),
    ("gemma4-26b-a4b-nvfp4", 64, {"presence_penalty": 1.0}),
    ("gemma4-26b-a4b-gguf", 64, {}),
])
def test_the_envelope_is_the_cards_flat_top_level_keys(model, top_k, extra) -> None:
    # A key added or removed here changes every reasoner call.
    assert sampling_kwargs(model) == {
        "temperature": 1.0, "top_p": 0.95, "top_k": top_k, "chat_template_kwargs": {"enable_thinking": True},
        **extra}


def test_reasoning_effort_rides_as_the_string_given(monkeypatch) -> None:
    """The arm's effort is sent verbatim, never parsed into a number; unset or
    empty sends no key; a mission's own effort wins, and None reads the arm's."""
    assert "reasoning_effort" not in sampling_kwargs("inkling-small")
    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", "")
    assert "reasoning_effort" not in sampling_kwargs("inkling-small")
    for arm in ("high", "none", "0.99", "0.0"):
        monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", arm)
        assert sampling_kwargs("inkling-small")["reasoning_effort"] == arm
    assert sampling_kwargs("inkling-small", "0.3")["reasoning_effort"] == "0.3"
    assert sampling_kwargs("inkling-small", None)["reasoning_effort"] == "0.0"
    assert "reasoning_effort" not in sampling_kwargs("inkling-small", "")


def test_vision_content_helpers_build_openai_parts() -> None:
    image = build_image_part(b"jpeg")
    assert image["type"] == "image_url"
    assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert image["image_url"]["detail"] == "high"
    assert build_vision_user_content("look", [b"jpeg", b""]) == [{"type": "text", "text": "look"}, image]


def test_a_served_name_with_no_card_is_refused_with_the_names_there_are() -> None:
    for near in ("qwen38-flash-next", "gemma-4-e2b", "qwen"):
        with pytest.raises(ValueError, match="inkling-small"):
            card_for(near)
