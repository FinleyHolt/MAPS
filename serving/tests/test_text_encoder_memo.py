"""The SAM text-encoder memo must be a cache, never a trade.

``engine._memoize_text_encoder`` wraps ``VETextEncoder.forward`` because that
call is a pure function of the concept phrase list and costs ~16 ms of a
~119 ms SAM call at every concept count. A cache over a pure function is
bit-identical by construction, so the whole risk is in the wrapper: caching
the branch that is NOT pure, keying too loosely, or handing two callers the
same tensor so one can write into the other's copy.

No sam3 here. The wrapper is exercised against a stand-in class installed at
the import path the engine names, which is also what pins that path: a sam3
that moves ``VETextEncoder`` makes this test's fake and the engine's import
disagree.
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")


class _FakeTextEncoder:
    """Counts real invocations and returns tensors derived from the input."""

    def __init__(self) -> None:
        self.calls = 0

    def forward(self, text, input_boxes=None, device=None):
        self.calls += 1
        n = len(text) if text else 0
        base = float(sum(len(t) for t in text if isinstance(t, str)))
        return (
            torch.full((n or 1,), base),
            torch.full((n or 1, 2), base + 1.0),
            torch.full((n or 1, 3), base + 2.0),
        )


def _install(monkeypatch):
    """Put the fake at the exact import path ``engine`` reaches for."""
    mod = types.ModuleType("sam3.model.text_encoder_ve")
    mod.VETextEncoder = _FakeTextEncoder
    pkg = types.ModuleType("sam3")
    model_pkg = types.ModuleType("sam3.model")
    monkeypatch.setitem(sys.modules, "sam3", pkg)
    monkeypatch.setitem(sys.modules, "sam3.model", model_pkg)
    monkeypatch.setitem(sys.modules, "sam3.model.text_encoder_ve", mod)
    return mod


class _Model:
    def __init__(self, *mods):
        self._mods = mods

    def modules(self):
        return iter(self._mods)


def _wrapped(monkeypatch):
    from serving import sam_engine as engine

    _install(monkeypatch)
    enc = _FakeTextEncoder()
    engine._memoize_text_encoder(_Model(enc))
    return engine, enc


def test_same_phrases_are_encoded_once(monkeypatch):
    _, enc = _wrapped(monkeypatch)
    first = enc.forward(["car", "building"], None, "cuda")
    second = enc.forward(["car", "building"], None, "cuda")

    assert enc.calls == 1, "the second identical call re-ran the encoder"
    for a, b in zip(first, second):
        assert torch.equal(a, b), "a cache hit returned different values"


def test_the_key_separates_what_must_be_separate(monkeypatch):
    _, enc = _wrapped(monkeypatch)
    enc.forward(["car"], None, "cuda")
    enc.forward(["building"], None, "cuda")  # different phrase
    enc.forward(["building", "car"], None, "cuda")  # different length
    enc.forward(["car", "building"], None, "cuda")  # different ORDER
    enc.forward(["car"], None, "cpu")  # different device
    assert enc.calls == 5, f"keys collided: {enc.calls} real calls, expected 5"


def test_a_caller_cannot_write_into_another_callers_copy(monkeypatch):
    """The load-bearing one. The encoder's output feeds straight into the
    transformer; if the cache handed out the stored tensor, one in-place op
    anywhere downstream would silently corrupt every later call."""
    _, enc = _wrapped(monkeypatch)
    first = enc.forward(["car"], None, "cuda")
    first[1].add_(999.0)

    second = enc.forward(["car"], None, "cuda")
    assert enc.calls == 1, "the mutation should not have invalidated anything"
    assert not torch.equal(first[1], second[1]), "the cache handed out its own tensor"
    assert torch.allclose(second[1], torch.full_like(second[1], 4.0)), (
        "the stored value was mutated through a handed-out reference"
    )


def test_the_already_encoded_branch_is_never_cached(monkeypatch):
    """``VETextEncoder.forward`` also accepts a 3-tuple of tensors meaning
    'already encoded', which is the shape this cache PRODUCES. Caching that
    branch would let the two feed each other."""
    _, enc = _wrapped(monkeypatch)
    pre = (torch.zeros(1), torch.zeros(1, 2), torch.zeros(1, 3))
    enc.forward(pre, None, "cuda")
    enc.forward(pre, None, "cuda")
    assert enc.calls == 2, "the already-encoded branch was cached"


def test_input_boxes_bypass_the_cache(monkeypatch):
    _, enc = _wrapped(monkeypatch)
    enc.forward(["car"], ["a box"], "cuda")
    enc.forward(["car"], ["a box"], "cuda")
    assert enc.calls == 2, "a call carrying input_boxes was cached"


def test_the_bound_holds(monkeypatch):
    from serving import sam_engine as engine

    _, enc = _wrapped(monkeypatch)
    for i in range(engine._TEXT_MEMO_MAX + 4):
        enc.forward([f"concept_{i}"], None, "cuda")
    assert len(enc._maps_memo) <= engine._TEXT_MEMO_MAX, "the cache grew unbounded"


def test_wrapping_twice_does_not_stack(monkeypatch):
    """``_ensure_sam`` builds once, but a caller that rebuilt would otherwise
    wrap the wrapper and count every hit as a miss one layer down."""
    engine, enc = _wrapped(monkeypatch)
    engine._memoize_text_encoder(_Model(enc))
    enc.forward(["car"], None, "cuda")
    enc.forward(["car"], None, "cuda")
    assert enc.calls == 1, "a second wrap defeated the cache"
