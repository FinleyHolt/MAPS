"""Over budget on a vision surface, the oldest STALE crops go, newest kept.

Counting images correctly is only half the fix: on the mission surface every
other step keys on ``name == "analyze"`` and a mission declares one tool
named ``run``, so on 859 measured mission tool-results they match nothing and
report "cleared 0 ... degraded 0" while looking like they worked. Without a
step that sheds image mass, a true estimate converts a silent server 400 into
a loud CompactionInfeasible and the episode still dies.

Crops are the LAST thing given up, not the first. They led the ladder while the
budget was 48,128 and sixteen 1024px maps were 42,112 tokens of it. At 221,120,
with a delivered crop measuring a median 225 tokens across 11,635 look
manifests, images are ~9% of a large prompt and the re-derivable text dumps are
the rest. ``maps/search/tests/test_compaction_order.py`` pins the ordering;
these pin this step's own behaviour once it is reached.
"""

from __future__ import annotations

from maps.llm.vision_content import (
    build_image_part,
    image_tokens_in,
    image_tokens_of,
)
from maps.search.analyze_scene.compactor import (
    _EVICTED_CROP_TEXT,
    _evict_current_turn_crops,
    _evict_oldest_crops,
)
from maps.search.tests.fakes import jpeg


def _vision_turn(text: str, n_images: int, side: int = 512) -> dict:
    parts: list[dict] = [{"type": "text", "text": text}]
    parts += [build_image_part(jpeg(side)) for _ in range(n_images)]
    return {"role": "user", "content": parts}


def _n_live_crops(messages: list[dict]) -> int:
    return sum(
        1
        for m in messages
        if isinstance(m.get("content"), list)
        for p in m["content"]
        if isinstance(p, dict) and p.get("type") == "image_url"
    )


def test_it_shes_enough_and_stops() -> None:
    msgs = [_vision_turn(f"turn {i}", 2) for i in range(5)]
    one = image_tokens_of(build_image_part(jpeg(512)))

    out, evicted, freed = _evict_oldest_crops(msgs, shed_tokens=one * 3)

    # Vacuity floor: a no-op would satisfy "did not over-evict" trivially.
    assert evicted >= 3, "nothing was shed"
    assert freed >= one * 3
    assert evicted <= 4, "shed far past what was asked"
    assert _n_live_crops(out) == 10 - evicted


def test_the_current_turn_s_crops_are_never_evicted() -> None:
    """They are what the turn is FOR. Evicting them saves a window the turn
    then has no reason to occupy."""
    msgs = [_vision_turn("old", 2), _vision_turn("newest", 2)]
    out, evicted, _ = _evict_oldest_crops(msgs, shed_tokens=10**9)
    assert evicted == 2, "it reached into the newest vision message"
    assert _n_live_crops([out[1]]) == 2
    assert _n_live_crops([out[0]]) == 0


def test_oldest_first() -> None:
    msgs = [_vision_turn("a", 1), _vision_turn("b", 1), _vision_turn("c", 1)]
    one = image_tokens_of(build_image_part(jpeg(512)))
    out, evicted, _ = _evict_oldest_crops(msgs, shed_tokens=one)
    assert evicted == 1
    assert _n_live_crops([out[0]]) == 0, "it took a newer crop before an older"
    assert _n_live_crops([out[1]]) == 1


def test_an_evicted_crop_says_so() -> None:
    """A hole would read to the model as an image it failed to see."""
    msgs = [_vision_turn("old", 1), _vision_turn("new", 1)]
    out, _, _ = _evict_oldest_crops(msgs, shed_tokens=10**9)
    assert any(
        p.get("text") == _EVICTED_CROP_TEXT
        for p in out[0]["content"]
        if isinstance(p, dict)
    )


def test_a_text_only_transcript_is_returned_untouched() -> None:
    """Parity: the identity, not a copy, so text-only compaction is
    byte-identical to what it was before any of this existed."""
    msgs = [
        {"role": "user", "content": "plain"},
        {"role": "assistant", "content": "reply"},
    ]
    out, evicted, freed = _evict_oldest_crops(msgs, shed_tokens=10**9)
    assert out is msgs
    assert (evicted, freed) == (0, 0)


def test_nothing_to_shed_returns_the_identity() -> None:
    msgs = [_vision_turn("only", 2)]
    out, evicted, freed = _evict_oldest_crops(msgs, shed_tokens=10**9)
    assert out is msgs, "the sole vision turn is the current one"
    assert (evicted, freed) == (0, 0)


def test_the_input_list_is_never_mutated() -> None:
    msgs = [_vision_turn("old", 1), _vision_turn("new", 1)]
    before = image_tokens_in(msgs)
    _evict_oldest_crops(msgs, shed_tokens=10**9)
    assert image_tokens_in(msgs) == before, "the caller's messages were mutated"


def test_the_current_turn_is_thinned_only_as_a_last_resort() -> None:
    """One turn may carry the whole per-request image cap. When THAT is what
    will not fit, every tier above has already run and the choice is a
    thinner view or a dead episode -- measured: two max-effort ladder
    episodes died 2,500 tokens over with one 314 KB user message holding
    every image in the transcript."""
    msgs = [_vision_turn("only turn", 6)]
    one = image_tokens_of(build_image_part(jpeg(512)))

    out, evicted, freed = _evict_current_turn_crops(msgs, shed_tokens=one * 2)
    assert evicted >= 2, "the turn that could not fit was left untouched"
    assert freed >= one * 2
    assert _n_live_crops(out) == 6 - evicted


def test_thinning_never_takes_the_newest_crop() -> None:
    """The turn keeps the view it most recently asked for; a text-only
    prompt is not a cheaper answer, it is a different question."""
    msgs = [_vision_turn("only turn", 4)]
    out, evicted, _ = _evict_current_turn_crops(msgs, shed_tokens=10**9)
    assert evicted == 3
    assert _n_live_crops(out) == 1
    assert out[0]["content"][-1]["type"] == "image_url", "it took the newest"


def test_thinning_leaves_earlier_turns_alone() -> None:
    """Tier 0 already swept those; reaching back here would double-count."""
    msgs = [_vision_turn("old", 2), _vision_turn("current", 3)]
    out, evicted, _ = _evict_current_turn_crops(msgs, shed_tokens=10**9)
    assert evicted == 2
    assert _n_live_crops([out[0]]) == 2, "an earlier turn was thinned"
    assert _n_live_crops([out[1]]) == 1


def test_thinning_a_text_only_transcript_is_the_identity() -> None:
    msgs = [{"role": "user", "content": "plain"}]
    out, evicted, freed = _evict_current_turn_crops(msgs, shed_tokens=10**9)
    assert out is msgs
    assert (evicted, freed) == (0, 0)


def test_a_single_crop_turn_cannot_be_thinned() -> None:
    """Nothing to give up: the one crop IS the newest."""
    msgs = [_vision_turn("only", 1)]
    out, evicted, freed = _evict_current_turn_crops(msgs, shed_tokens=10**9)
    assert out is msgs
    assert (evicted, freed) == (0, 0)
