#!/usr/bin/env python
"""Delivering a crop never rewrites an earlier message."""
from __future__ import annotations

import hashlib
import json

from maps.search.parse_compile.reasoner_agent import Reasoner
from maps.search.tests.fakes import TEST_SPEC

# Enough deliveries to pass any cap anyone would plausibly reintroduce.
_DELIVERIES = 40


def _digests(messages: list[dict]) -> list[str]:
    """The same per-message digest ``maps.llm.http`` takes to compute ``n_prefix``."""
    return [
        hashlib.sha256(
            json.dumps(m, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        for m in messages
    ]


def _jpeg(n: int) -> bytes:
    """Distinct bytes per crop, so an eviction cannot hide behind two crops happening to
    serialize the same."""
    return b"\xff\xd8\xff" + bytes([n % 251]) * 32


def test_appending_a_crop_leaves_every_earlier_message_byte_identical():
    r = Reasoner(endpoint_spec=TEST_SPEC)
    for i in range(_DELIVERIES):
        before = _digests(r.messages)
        r.append_vision_user(f"look {i}", [_jpeg(i)])
        after = _digests(r.messages)
        assert after[: len(before)] == before, (
            f"delivery {i} rewrote an earlier message: the prefix diverges at "
            f"index {next(k for k, (a, b) in enumerate(zip(before, after)) if a != b)}"
        )
        assert len(after) == len(before) + 1
