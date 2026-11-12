from __future__ import annotations

import pytest

from maps.llm.model_cards import card_for, sampling_kwargs
from maps.llm.vision_content import build_image_part, build_vision_user_content
from maps.settings import Sampling, from_environ


def _run_sampling(effort: str) -> Sampling:
    """The sampling of an environment whose SEARCH_SAMPLING_REASONING_EFFORT is ``effort``. A call
    that names no sampling reads an empty environment's, never the shell's."""
    return from_environ({"SEARCH_SAMPLING_REASONING_EFFORT": effort}).sampling


@pytest.mark.parametrize(
    "model, top_k, extra",
    [
        ("qwen38-flash-next-nvfp4", 20, {}),
        ("gemma4-26b-a4b-nvfp4", 64, {"presence_penalty": 1.0}),
        ("gemma4-26b-a4b-gguf", 64, {}),
    ],
)
def test_the_envelope_is_the_cards_flat_top_level_keys(model, top_k, extra) -> None:
    # A key added or removed here changes every reasoner call.
    assert sampling_kwargs(model) == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": top_k,
        "chat_template_kwargs": {"enable_thinking": True},
        **extra,
    }


def test_reasoning_effort_rides_as_the_string_given() -> None:
    """The run's effort is sent verbatim, never parsed into a number; unset or
    empty sends no key; a mission's own effort wins, and None reads the run's."""
    assert "reasoning_effort" not in sampling_kwargs("inkling-small")
    assert "reasoning_effort" not in sampling_kwargs("inkling-small", sampling=_run_sampling(""))
    for effort in ("high", "none", "0.99", "0.0"):
        assert (
            sampling_kwargs("inkling-small", sampling=_run_sampling(effort))["reasoning_effort"]
            == effort
        )
    assert (
        sampling_kwargs("inkling-small", "0.3", _run_sampling("0.0"))["reasoning_effort"] == "0.3"
    )
    assert sampling_kwargs("inkling-small", None, _run_sampling("0.0"))["reasoning_effort"] == "0.0"
    assert "reasoning_effort" not in sampling_kwargs("inkling-small", "", _run_sampling("0.0"))


def test_none_on_the_gemma_26b_card_sends_its_template_switch_off() -> None:
    """Gemma's template reads enable_thinking and never reasoning_effort: none
    turns the switch off and sends no effort key; another rung still rides."""
    kw = sampling_kwargs("gemma4-26b-a4b-nvfp4", "none")
    assert kw["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in kw and kw["presence_penalty"] == 1.0
    assert sampling_kwargs("gemma4-26b-a4b-nvfp4", sampling=_run_sampling("none"))[
        "chat_template_kwargs"
    ] == {"enable_thinking": False}
    assert sampling_kwargs("gemma4-26b-a4b-nvfp4", "high", _run_sampling("none"))[
        "chat_template_kwargs"
    ] == {"enable_thinking": True}


def test_vision_content_helpers_build_openai_parts() -> None:
    image = build_image_part(b"jpeg")
    assert image["type"] == "image_url"
    assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert image["image_url"]["detail"] == "high"
    assert build_vision_user_content("look", [b"jpeg", b""]) == [
        {"type": "text", "text": "look"},
        image,
    ]


def test_a_served_name_with_no_card_is_refused_with_the_names_there_are() -> None:
    for near in ("qwen38-flash-next", "gemma-4-e2b", "qwen"):
        with pytest.raises(ValueError, match="inkling-small"):
            card_for(near)


def test_the_deepseek_card_sends_its_recommended_envelope() -> None:
    assert sampling_kwargs("deepseek-v4.1-flash-1017") == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "chat_template_kwargs": {"enable_thinking": True},
    }
