"""The instruction parse keeps only what a checkpoint can use."""

from __future__ import annotations

import pytest

from baselines.released.describe_instructions import parse_reply


def test_landmarks_outside_the_block_are_dropped():
    reply = ('Sure.\n{"target": "white house", "landmarks": ["Main Street", "Nowhere Rd"],'
             ' "surroundings": ["parked car", " "]}')
    assert parse_reply(reply, ["Main Street", "Elm Street"]) == {
        "target": "white house", "landmarks": ["Main Street"], "surroundings": ["parked car"],
    }


def test_a_reply_without_a_target_refuses():
    with pytest.raises(ValueError):
        parse_reply('{"landmarks": []}', [])
    with pytest.raises(ValueError):
        parse_reply("no json here", [])
