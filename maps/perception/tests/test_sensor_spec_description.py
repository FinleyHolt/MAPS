"""SensorSpec.description plumbing.

The description is patched into the reasoner's system prompt, so
every registered sensor must carry a non-empty description; YAML
overrides must inherit from their base unless overridden, and refuse a key
they do not know.
"""

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


def test_resolve_sensor_inline_dict_inherits_base_description() -> None:
    base = _REGISTRY["nadir_1008_90"]
    spec = resolve_sensor({"base": "nadir_1008_90", "max_range_m": 50.0})
    assert spec is not None
    assert spec.description == base.description


def test_resolve_sensor_inline_dict_overrides_description() -> None:
    spec = resolve_sensor(
        {"base": "nadir_1008_90", "description": "custom override"}
    )
    assert spec is not None
    assert spec.description == "custom override"


def test_resolve_sensor_refuses_an_unknown_override_key() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown sensor override key"):
        resolve_sensor({"base": "nadir_1008_90", "hfov": 40.0})


def test_nadir_description_warns_about_vertical_surfaces() -> None:
    """Smoke check that the nadir description actually constrains the agent.

    The whole point of this field is to stop reasoners from asking SAM
    for facade / wall segmentations on a straight-down camera. If the
    nadir description ever loses that warning, the reasoner prompt will
    quietly stop carrying it and the bug returns.
    """
    nadir = _REGISTRY["nadir_1008_90"]
    text = nadir.description.lower()
    assert "nadir" in text or "straight down" in text or "90" in text
    assert "wall" in text or "facade" in text or "vertical" in text
