from __future__ import annotations

import json

import pytest

from maps.llm.sampling import thinking_sampling_kwargs
from maps.llm.vision_content import build_image_part, build_vision_user_content


@pytest.fixture(autouse=True)
def _no_ambient_sampling_env(monkeypatch) -> None:
    """The envelope pins measure the code, not a shell that sourced an arm file."""
    monkeypatch.delenv("SEARCH_SAMPLING_REASONING_EFFORT", raising=False)


def test_thinking_sampling_kwargs_are_flat_top_level_keys() -> None:
    # The envelope, pinned exactly — a key added or removed here changes
    # every reasoner call. A nested envelope (extra_body) is dropped
    # wholesale by hosted OpenAI-compatible endpoints; flat top-level keys
    # are the contract.
    assert thinking_sampling_kwargs(temperature=0.6, model_id="qwen38-flash-next-nvfp4") == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_gemma_envelope_keys_off_model_id() -> None:
    # The gemma family envelope is the gemma-4 card's recommended set:
    # top_k 64, no penalty term at all — a presence_penalty key would be
    # an off-card term the serve never saw validated.
    assert thinking_sampling_kwargs(temperature=1.0, model_id="gemma4-26b-a4b-nvfp4") == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_reasoning_effort_is_never_parsed_into_a_number(monkeypatch) -> None:
    # Every rung stays the string the arm wrote: a named level is not
    # mapped to its float, and a numeric one is not parsed and re-emitted.
    # Either translation would send the type the serve rejects.
    for named in ("high", "max", "none"):
        monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", named)
        assert thinking_sampling_kwargs(temperature=1.0, model_id="inkling-small")["reasoning_effort"] == named
    for numeric in ("0.99", "0.0", "0.7"):
        monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", numeric)
        got = thinking_sampling_kwargs(temperature=1.0, model_id="inkling-small")["reasoning_effort"]
        assert got == numeric and not isinstance(got, float)


def test_unset_reasoning_effort_sends_no_key_at_all(monkeypatch) -> None:
    # Unset means the request omits the field and the served chat
    # template's own default applies. A default invented here would become
    # the arm's identity without an arm having asked for it, and a card
    # with no effort field would 400 on every request.
    monkeypatch.delenv("SEARCH_SAMPLING_REASONING_EFFORT", raising=False)
    assert "reasoning_effort" not in thinking_sampling_kwargs(temperature=1.0, model_id="inkling-small")
    # An empty export (the arm file's "declare it, leave it off" spelling)
    # reads as unset, not as an empty-string effort the serve would refuse.
    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", "")
    assert "reasoning_effort" not in thinking_sampling_kwargs(temperature=1.0, model_id="inkling-small")


def test_vision_content_helpers_build_openai_parts() -> None:
    image = build_image_part(b"jpeg", detail="low")

    assert image["type"] == "image_url"
    assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert image["image_url"]["detail"] == "low"

    content = build_vision_user_content("look", [b"jpeg", b""], detail="low")
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1] == image
    assert len(content) == 2


def test_a_missions_own_effort_wins_over_the_arms(monkeypatch) -> None:
    """The field station names an effort per mission; given here it is the
    one sent, as a string, whatever the arm's environment says, and None
    reads the arm's."""
    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", "0.99")
    assert thinking_sampling_kwargs(temperature=1.0, model_id="inkling-small", reasoning_effort="0.3")["reasoning_effort"] == "0.3"
    assert thinking_sampling_kwargs(temperature=1.0, model_id="inkling-small", reasoning_effort="0.0")["reasoning_effort"] == "0.0"
    assert thinking_sampling_kwargs(temperature=1.0, model_id="inkling-small", reasoning_effort=None)["reasoning_effort"] == "0.99"
    monkeypatch.delenv("SEARCH_SAMPLING_REASONING_EFFORT")
    assert "reasoning_effort" not in thinking_sampling_kwargs(temperature=1.0, model_id="inkling-small", reasoning_effort="")


def test_a_served_name_with_no_card_is_refused_with_the_names_there_are() -> None:
    import pytest

    from maps.llm.model_cards import card_for

    for near in ("qwen38-flash-next", "gemma-4-e2b", "qwen"):
        with pytest.raises(ValueError, match="inkling-small"):
            card_for(near)
