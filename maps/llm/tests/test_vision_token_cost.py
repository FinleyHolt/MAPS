"""Image crops cost tokens, and the client has to count them.

They were counted as ZERO. Every ``image_url`` part collapsed to the string
``"[image crop]"`` -- about 3 tokens -- before tokenization, on the stated
theory that the server's vision budget was governed separately by a
per-request image cap. It is not, twice over. A count cap does not bill
tokens -- they land in the same context window as the text -- and no cap is
declared anyway: the sglang flag is ``--limit-mm-data-per-request``
(``--limit-mm-per-prompt`` is vLLM's spelling), and no card or script in
this tree passes either.

Measured consequence on a reasoning-effort ladder: the max-effort arm sent
twelve requests 1,049-4,582 tokens past its own declared budget of 48,128,
the server rejected every one, and 130,095 uncounted tokens over 136 images
was the entire gap. The two lower-effort arms never took enough turns to
reach the window, so it read as an effort effect rather than a harness bug.
"""

from __future__ import annotations

import io

import pytest

from maps.llm.vision_content import (
    build_image_part,
    build_vision_user_content,
    image_tokens_in,
    image_tokens_of,
)


def _jpeg(width: int, height: int) -> bytes:
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (128, 128, 128)).save(buf, format="JPEG")
    return buf.getvalue()


# The two single-image points measured against the served build.
@pytest.mark.parametrize("side,server_billed", [(400, 427), (1024, 2632)])
def test_the_estimate_brackets_the_server_from_above(side, server_billed) -> None:
    """Never under, because under-counting is what kills the episode; and
    within a third over, because over-counting evicts a crop early."""
    got = image_tokens_of(build_image_part(_jpeg(side, side)))
    assert got >= server_billed, f"{side}px under-counted: {got} < {server_billed}"
    assert got <= server_billed * 1.33, f"{side}px over-counted: {got}"


def test_a_crop_is_not_three_tokens() -> None:
    """The floor on the whole fix. The placeholder it replaces is ~3 tokens,
    so any figure in that neighbourhood means the bug is back."""
    assert image_tokens_of(build_image_part(_jpeg(1024, 1024))) > 1000


def test_sixteen_maps_are_most_of_an_input_budget() -> None:
    """The measured shape of the failure: the per-request image cap is 16,
    denominated in COUNT, against a 48,128-token budget. Sixteen 1024px maps
    have to read as the great majority of it, not as an empty context."""
    content = build_vision_user_content("look", [_jpeg(1024, 1024)] * 16)
    assert image_tokens_in([{"role": "user", "content": content}]) > 40000


def test_non_square_crops_price_by_area() -> None:
    wide = image_tokens_of(build_image_part(_jpeg(800, 400)))
    tall = image_tokens_of(build_image_part(_jpeg(400, 800)))
    small = image_tokens_of(build_image_part(_jpeg(400, 400)))
    assert wide == tall
    assert wide > small


def test_text_only_messages_cost_nothing() -> None:
    """Parity: a text-only transcript must be untouched by any of this."""
    assert image_tokens_in([
        {"role": "user", "content": "plain text"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": [{"type": "text", "text": "a result"}]},
    ]) == 0


def test_a_malformed_part_reads_zero_rather_than_raising() -> None:
    """An estimate that dies on a bad crop is worse than one that misses a
    crop the server is going to reject anyway."""
    assert image_tokens_of({"type": "image_url", "image_url": {"url": "nonsense"}}) == 0
    assert image_tokens_of({
        "type": "image_url", "image_url": {"url": "data:image/jpeg;base64,!!!"}
    }) == 0
    assert image_tokens_of({"type": "text", "text": "not an image"}) == 0
    assert image_tokens_of("not even a part") == 0
