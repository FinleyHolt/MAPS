"""The turns that are not a reply, and the nudge each draws: an empty turn, a
turn the output cap cut off, a block of code sent as text, tool-call markup
written as text, and a reply with no flight behind it."""

from __future__ import annotations

import re

from maps.library.block_scan import looks_like_code

# Consecutive zero-flight replies that end the mission: the first two draw
# NO_FLIGHT_REPLY_NUDGE, the third is an abstain. Anything the model
# executes between two replies resets the streak.
NO_FLIGHT_ABSTAIN_STREAK = 3


def _envelope_residue(text: str) -> bool:
    """True when a prose turn carries tool-call markup: a malformed
    envelope, repaired, never an ending."""
    t = str(text or "")
    return any(m in t for m in ("<tool_call", "</tool_call", "<function"))


EMPTY_TURN_NUDGE = (
    "Your last turn arrived empty -- no tool call and no text, so nothing "
    "executed. Re-emit it: the tool call you intended, or your plain-text "
    "reply."
)

TRUNCATED_TURN_NUDGE = (
    "Your last turn ran out of output tokens while thinking, so nothing "
    "arrived and nothing executed. Act on what you have: send the one tool "
    "call you settled on, or your plain-text reply."
)

#: A turn the output cap cut off empty is the serve's truncation, not a
#: dropped reply: it gets its own nudge and spends no repair, up to this many
#: in a row, after which it reads as an empty turn.
TRUNCATED_TURN_STREAK = 3

CODE_REPLY_NUDGE = (
    "Your reply is code, but it arrived as plain text, not through the run "
    "tool, so nothing was executed. Send the block with the tool to run "
    "it; a plain reply in your own words ends the mission."
)

MALFORMED_ENVELOPE_NUDGE = (
    "Your reply contains tool-call markup that did not parse as a native "
    "tool call; an envelope written as text is not executed. Only a call on "
    "the API's tool-call channel executes."
)

NO_FLIGHT_REPLY_NUDGE = (
    "Your reply ends the mission, and your answer is the row of your LAST "
    "flight -- but you have flown nowhere, so it answers nothing. Naming "
    "the object commits nothing; only flying to it does. "
    "If you know which one it is, go_to() it first, then reply."
)

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|\n```\s*$")


def _code_reply(text: str) -> bool:
    """True when the whole reply is a block of the surface: it parses whole
    (prose narrating a call does not) and carries a library call (plain
    prose that happens to parse does not)."""
    t = _CODE_FENCE_RE.sub("", str(text or "").strip()).strip()
    return bool(t) and looks_like_code(t)


def _not_a_reply(content: str) -> tuple[str, str, str] | None:
    """A text turn that is a dropped message rather than a reply -- empty,
    or a block of code -- as ``(transcript kind, log label, nudge)``."""
    if not content.strip():
        return "empty_turn", "EMPTY turn", EMPTY_TURN_NUDGE
    if _code_reply(content):
        return "code_shaped_reply", "CODE-SHAPED reply turn", CODE_REPLY_NUDGE
    return None
