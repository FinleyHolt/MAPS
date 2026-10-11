"""Image crops are counted in the client's token estimate: they share the context window with the text,
and an uncounted crop sends a request past its budget."""

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


def test_sixteen_maps_are_most_of_an_input_budget() -> None:
    """Sixteen 1024 px maps (the per-request image cap) are most of a 48,128-token budget."""
    content = build_vision_user_content("look", [_jpeg(1024, 1024)] * 16)
    assert image_tokens_in([{"role": "user", "content": content}]) > 40000


def test_non_square_crops_price_by_area() -> None:
    wide = image_tokens_of(build_image_part(_jpeg(800, 400)))
    tall = image_tokens_of(build_image_part(_jpeg(400, 800)))
    small = image_tokens_of(build_image_part(_jpeg(400, 400)))
    assert wide == tall
    assert wide > small


def test_text_only_messages_cost_nothing() -> None:
    assert image_tokens_in([
        {"role": "user", "content": "plain text"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": [{"type": "text", "text": "a result"}]},
    ]) == 0


def test_a_malformed_part_reads_zero_rather_than_raising() -> None:
    assert image_tokens_of({"type": "image_url", "image_url": {"url": "nonsense"}}) == 0
    assert image_tokens_of({
        "type": "image_url", "image_url": {"url": "data:image/jpeg;base64,!!!"}
    }) == 0
    assert image_tokens_of({"type": "text", "text": "not an image"}) == 0
    assert image_tokens_of("not even a part") == 0
