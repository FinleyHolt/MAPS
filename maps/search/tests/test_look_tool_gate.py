"""The look tool is on, and a look tool that cannot be wired is an error.

``view{crops}`` is the only path that puts an image in front of the reasoner.
When its renderer is unwired the reasoner goes text-only, and until 25 Jul 2026
that happened silently: the tool was off from ~13 Jul and every eval in the
window flew a blind reasoner with nothing on the row saying so.

So asking for look and not getting it raises at construction. The other half
— provenance recording the resolved answer rather than an env flag that may
never have been set — is pinned in ``benchmarks/tests``, since ``maps`` is
the base layer and may not import upward.
"""

from __future__ import annotations

import pytest

from maps.search.search_loop import SearchLoopConfig


def test_the_look_tool_is_on_by_default():
    assert SearchLoopConfig().look_tool is True


def test_the_env_flag_is_the_off_switch(monkeypatch):
    monkeypatch.setenv("SEARCH_LOOK_TOOL", "0")
    assert SearchLoopConfig.from_env().look_tool is False
    monkeypatch.setenv("SEARCH_LOOK_TOOL", "1")
    assert SearchLoopConfig.from_env().look_tool is True


def test_an_unwirable_look_tool_raises(monkeypatch):
    """A degrade to text-only must never be silent."""
    from maps.search import search_loop as sl

    monkeypatch.delenv("SEARCH_LOOK_TOOL", raising=False)
    loop = object.__new__(sl.SearchLoop)
    loop._rgbd_dir = None  # the renderer has nothing to render from
    cfg = SearchLoopConfig()

    with pytest.raises(ValueError) as exc:
        sl.SearchLoop._resolve_look_gate(loop, cfg)
    assert "look tool requested" in str(exc.value)
    assert "SEARCH_LOOK_TOOL=0" in str(exc.value), "the message must name the opt-out"


def test_an_explicit_opt_out_never_raises(monkeypatch):
    """The field replay has no ortho raster dir and says so on purpose."""
    from maps.search import search_loop as sl

    loop = object.__new__(sl.SearchLoop)
    loop._rgbd_dir = None
    sl.SearchLoop._resolve_look_gate(loop, SearchLoopConfig(look_tool=False))
    assert loop._look_tool_enabled is False
