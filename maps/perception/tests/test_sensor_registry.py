"""A caller registers its own rig by name; a clashing name is refused."""

from __future__ import annotations

import dataclasses

import pytest

from maps.perception.geometry import sensor_spec


def test_a_registered_rig_resolves_and_a_clash_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(sensor_spec, "_REGISTRY", dict(sensor_spec._REGISTRY))
    rig = dataclasses.replace(sensor_spec.resolve_sensor("nadir_1008_90"), name="field_rig")
    sensor_spec.register_sensor(rig)
    sensor_spec.register_sensor(rig)  # the same spec again is a no-op
    assert sensor_spec.resolve_sensor("field_rig") == rig
    with pytest.raises(ValueError, match="already registered"):
        sensor_spec.register_sensor(dataclasses.replace(rig, mount_yaw_deg=90.0))
