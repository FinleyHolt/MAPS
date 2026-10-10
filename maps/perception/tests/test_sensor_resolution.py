"""``resolve_sensor`` contract: registry names, inline overrides, unknowns."""

from __future__ import annotations

import pytest

from maps.perception.geometry.sensor_spec import list_sensors, resolve_sensor


def test_registry_name_resolves():
    spec = resolve_sensor("nadir_1008_90")
    assert spec is not None
    assert spec.name == "nadir_1008_90"


def test_the_shipped_citynav_sensor_keeps_its_geometry():
    """Every config resolves this one. The ``90`` in the name is the field of
    view; the observation is an orthographic nadir crop by construction."""
    spec = resolve_sensor("nadir_1008_90")
    assert spec is not None
    assert spec.intrinsics.image_width == 1008
    assert spec.intrinsics.image_height == 1008
    assert spec.intrinsics.hfov_deg == pytest.approx(90.0)


def test_empty_spec_resolves_to_none():
    """``None`` means the caller falls back to orthographic raster rendering."""
    assert resolve_sensor(None) is None
    assert resolve_sensor("") is None


def test_inline_dict_overrides_a_base():
    spec = resolve_sensor(
        {"base": "nadir_1008_90", "name": "wide", "hfov_deg": 120.0}
    )
    assert spec is not None
    assert spec.name == "wide"
    assert spec.intrinsics.hfov_deg == pytest.approx(120.0)
    # Unstated fields keep the base's values.
    base = resolve_sensor("nadir_1008_90")
    assert base is not None
    assert spec.max_range_m == pytest.approx(base.max_range_m)


def test_unknown_name_names_what_is_registered():
    with pytest.raises(ValueError) as exc:
        resolve_sensor("no_such_sensor")
    assert "no_such_sensor" in str(exc.value)
    for name in list_sensors():
        assert name in str(exc.value)
