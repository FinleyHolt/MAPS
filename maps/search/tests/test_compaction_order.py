#!/usr/bin/env python
"""Text goes before pixels, and no existing test said so.

Every compactor test pins ONE step in isolation, with a fixture that can only
trigger that step. Reordering the ladder therefore changed real behaviour while
all 50 of them stayed green -- the same shape of gap as ``n_prefix`` having no
reader. These pin the ORDER.

The order matters for two reasons that outlive the current numbers. A stubbed
analyze dump costs a re-query, because the analyze surface re-derives from the
world after every flight; an evicted crop costs the view of a place at a moment,
and its pixels live only in the on-disk look audit the model cannot read. And
crops-first would undo, one step down, exactly what removing the in-context
image cap was for.
"""
from __future__ import annotations

import json

from maps.llm.vision_content import build_image_part
from maps.search.analyze_scene import compactor as C
from maps.search.analyze_scene.compactor import (
    _EVICTED_CROP_TEXT,
    compact_transcript,
)
from maps.search.tests.fakes import jpeg


def _content_str(m: dict) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c, default=str)


def _charcount(messages, *, model, tools=None):
    return sum(len(_content_str(m)) // 4 for m in messages)


def _vision_turn(n_images: int) -> dict:
    parts: list[dict] = [{"type": "text", "text": "crops"}]
    parts += [build_image_part(jpeg()) for _ in range(n_images)]
    return {"role": "user", "content": parts}


def _assistant_call(tcid: str) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": tcid, "type": "function",
                            "function": {"name": "analyze", "arguments": "{}"}}]}


def _analyze_dump(tcid: str, *, filler: int = 0) -> dict:
    head = {
        "len": 3,
        "fids": ["d0", "d1", "d2"],
        "units": "d_/dc_/along_/off_/vx_/vy_ in meters",
        "rows": "| idx | fid | class | color | color_conf | xy |\n"
                "| --- | --- | --- | --- | --- | --- |\n"
                "| 0 | d0 | car | white | 0.8 | 0.0, 0.0 |"
                + ("\n" + "z" * filler if filler else ""),
    }
    return {"role": "tool", "tool_call_id": tcid, "name": "analyze",
            "content": json.dumps({"results": {"r0": head}})}


def _live_crops(messages) -> int:
    return sum(1 for m in messages if isinstance(m.get("content"), list)
               for p in m["content"]
               if isinstance(p, dict) and p.get("type") == "image_url")


def _evicted_markers(messages) -> int:
    return sum(1 for m in messages if isinstance(m.get("content"), list)
               for p in m["content"]
               if isinstance(p, dict) and p.get("text") == _EVICTED_CROP_TEXT)


def _transcript() -> list[dict]:
    """A vision surface with a bulky STALE dump, two vision turns, a live dump.

    TWO vision turns, not one: ``_evict_oldest_crops`` never touches the last
    message carrying images, so a single-vision-turn fixture has no stale crop
    to take and the step is a no-op by construction rather than by order.
    """
    return [
        {"role": "system", "content": "SYS"},
        _assistant_call("t0"),
        _analyze_dump("t0", filler=40000),   # stale, and the bulk
        _vision_turn(1),                     # stale crop
        _vision_turn(1),                     # the current turn's crop
        _assistant_call("tL"),
        _analyze_dump("tL"),                 # live
        {"role": "user", "content": "u1"},
    ]


def test_a_clearable_dump_is_taken_and_the_crops_are_left(monkeypatch) -> None:
    """The whole change, in one assertion. Over budget by less than the stale
    dump is worth, so stubbing it suffices -- and the crops must survive
    untouched. Under the old order the crop step ran first and would have
    evicted pixels to free tokens the text was about to give up anyway."""
    monkeypatch.setattr(C, "_estimate_input_tokens", _charcount)
    msgs = _transcript()
    before = _live_crops(msgs)
    assert before == 2, "fixture must carry crops for this to mean anything"

    out = compact_transcript(
        msgs, target_input_tokens=2000, model="m", site="test-order")

    assert out is not msgs, "the fixture must actually be over budget"
    assert _live_crops(out) == 2, (
        "crops were evicted while a re-derivable text dump was still verbatim"
    )
    assert _evicted_markers(out) == 0
    assert '"rows"' not in out[2]["content"], "the stale dump should be stubbed"


def test_crops_still_go_when_the_text_is_exhausted(monkeypatch) -> None:
    """Order, not exemption. Once every stale dump is stubbed and the
    transcript is still over, the crop step runs -- five max-effort episodes on
    the effort ladder ran out of crops and died 1,822 tokens over, so neither
    half of the ladder is sufficient alone."""
    monkeypatch.setattr(C, "_estimate_input_tokens", _charcount)
    msgs = _transcript()

    # A target the text steps alone cannot reach but one stale crop can, so
    # the ladder must walk past them and into the crops -- and then stop,
    # leaving the current turn's own crop alone.
    out = compact_transcript(
        msgs, target_input_tokens=900, model="m", site="test-order")

    assert _evicted_markers(out) == 1, "an evicted crop must say so"
    assert _live_crops(out) == 1, (
        "the STALE crop goes and the current turn's stays: evicting the crops "
        "the turn was just handed makes the turn meaningless"
    )


def test_a_text_only_transcript_is_unchanged_by_the_reorder(monkeypatch) -> None:
    """Byte-parity guard: the crop step is a no-op without list content, so a
    text-only episode must compact exactly as it did before."""
    monkeypatch.setattr(C, "_estimate_input_tokens", _charcount)
    msgs = [
        {"role": "system", "content": "SYS"},
        _assistant_call("t0"),
        _analyze_dump("t0", filler=40000),
        _assistant_call("tL"),
        _analyze_dump("tL"),
        {"role": "user", "content": "u1"},
    ]
    out = compact_transcript(
        msgs, target_input_tokens=2000, model="m", site="test-order")
    assert _evicted_markers(out) == 0
    assert all(isinstance(m.get("content"), str) for m in out)
