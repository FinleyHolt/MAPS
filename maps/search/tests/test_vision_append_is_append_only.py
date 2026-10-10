#!/usr/bin/env python
"""Delivering a crop never rewrites an earlier message.

The reasoner history is the served prompt, and a served prompt is a radix
prefix. Rewriting message ``i`` re-hashes every block from ``i`` on, so the
server re-prefills the whole tail; the transcript records that as ``n_prefix``
rolling back. A count cap used to sit in ``append_vision_user`` evicting the
oldest crop past sixteen, which did exactly this on 8.13% of requests,
discarding a mean of 35.2 messages each time.

The cap is gone and the compactor bounds tokens instead, which is the resource
the window actually spends. This pins the property the cap violated rather than
the cap's absence, so it also fails if a cap returns in another shape: every
message but the newly appended one is byte-identical across a delivery.

No network. Run:
``pytest maps/search/tests/test_vision_append_is_append_only.py``.
"""
from __future__ import annotations

import hashlib
import json

from maps.llm.endpoint_spec import EndpointSpec
from maps.search.parse_compile.reasoner_agent import Reasoner

_TEST_SPEC = EndpointSpec(
    base_url="http://unused",
    model_id="inkling-small",
    max_model_len=262144,
    tokenizer_name="qwen",
    output_reserve=40000,
)

# Enough deliveries to pass any cap anyone would plausibly reintroduce. The
# retired one was 16, and it first bit on the 17th image.
_DELIVERIES = 40


def _digests(messages: list[dict]) -> list[str]:
    """The same per-message digest ``maps.llm.http`` takes to compute
    ``n_prefix``. Comparing these is comparing what the server sees."""
    return [
        hashlib.sha256(
            json.dumps(m, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        for m in messages
    ]


def _jpeg(n: int) -> bytes:
    """Distinct bytes per crop, so an eviction cannot hide behind two crops
    happening to serialize the same."""
    return b"\xff\xd8\xff" + bytes([n % 251]) * 32


def test_appending_a_crop_leaves_every_earlier_message_byte_identical():
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=_TEST_SPEC)
    for i in range(_DELIVERIES):
        before = _digests(r.messages)
        r.append_vision_user(f"look {i}", [_jpeg(i)])
        after = _digests(r.messages)
        assert after[: len(before)] == before, (
            f"delivery {i} rewrote an earlier message: the prefix diverges at "
            f"index {next(k for k, (a, b) in enumerate(zip(before, after)) if a != b)}"
        )
        assert len(after) == len(before) + 1


def test_every_delivered_crop_survives_the_whole_episode():
    """No image part is ever replaced by a placeholder, at any depth."""
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=_TEST_SPEC)
    for i in range(_DELIVERIES):
        r.append_vision_user(f"look {i}", [_jpeg(i)])
    images = [
        part
        for m in r.messages
        if isinstance(m.get("content"), list)
        for part in m["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    assert len(images) == _DELIVERIES


def test_a_text_turn_between_crops_does_not_disturb_them():
    """Interleaving is the real shape: analyze results and repairs land
    between deliveries, and the digest check has to hold across them."""
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=_TEST_SPEC)
    for i in range(_DELIVERIES):
        r.append_vision_user(f"look {i}", [_jpeg(i)])
        before = _digests(r.messages)
        r.append_user(f"packet {i}")
        assert _digests(r.messages)[: len(before)] == before
