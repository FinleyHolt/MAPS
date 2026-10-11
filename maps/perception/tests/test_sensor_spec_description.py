"""Every registered sensor carries the description the reasoner's system prompt quotes, and an
override refuses a key it does not know."""

from __future__ import annotations

from maps.perception.geometry.sensor_spec import (
    _REGISTRY,
    resolve_sensor,
)


def test_every_registered_sensor_has_a_description() -> None:
    missing = [name for name, spec in _REGISTRY.items() if not spec.description.strip()]
    assert not missing, (
        f"Registered sensors with empty description: {missing}. The description "
        "is patched into agent system prompts; an empty string silently re-enables "
        "the 'agents request impossible concepts' bug."
    )


def test_resolve_sensor_refuses_an_unknown_override_key() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown sensor override key"):
        resolve_sensor({"base": "nadir_1008_90", "hfov": 40.0})


def test_nadir_description_warns_about_vertical_surfaces() -> None:
    """The nadir description tells the reasoner a straight-down camera sees no walls."""
    nadir = _REGISTRY["nadir_1008_90"]
    text = nadir.description.lower()
    assert "nadir" in text or "straight down" in text or "90" in text
    assert "wall" in text or "facade" in text or "vertical" in text
