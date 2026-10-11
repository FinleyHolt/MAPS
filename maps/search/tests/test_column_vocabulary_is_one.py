"""Column keys group into per-relation families on the phrase spelling the executor
emits."""

from __future__ import annotations

from maps.search.analyze_scene.geom_program import _grouped_keys


def test_families_collapse_on_the_shipped_spelling():
    """A fan-out compare's per-ref columns summarise as one family."""
    keys = [f"distance to building_{i}" for i in range(100)] + ["along road_5"]
    out = _grouped_keys(keys)
    assert "distance to building_* (100 instances" in out
    assert "along road_5" in out
    assert out.count("distance to building_") == 2  # the family + its example
    assert len(out) < 200, f"the 400-name dump is back: {out[:200]}"


def test_longest_prefix_wins_so_no_key_prints_twice():
    keys = ([f"facing b_{i}" for i in range(5)]
            + [f"bearing from b_{i}" for i in range(5)])
    out = _grouped_keys(keys)
    assert "facing b_* (5 instances" in out
    assert "bearing from b_* (5 instances" in out
    # Ten keys, two families, each named once.
    assert out.count("instances") == 2
