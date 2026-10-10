"""The served sensor catalog is valid JSON, and carries what a readout needs.

Both map surfaces (the dashboard canvas and the cockpit nav console) fetch this
list and derive "the altitude that sees this much ground" from a picked
sensor's HFOV. Two ways that goes wrong quietly: a non-finite float, which
``json.dumps`` writes as a bare ``Infinity`` that no browser will parse (so the
frontend's catch hands back an empty list and the altitude just never appears);
and a missing HFOV, which leaves nothing to compute from.
"""

from __future__ import annotations

import json

from maps.perception.geometry.sensor_spec import (
    list_sensors,
    resolve_sensor,
    sensor_catalog,
)


def test_catalog_is_strict_json():
    # allow_nan=False is the browser's rule: Infinity/NaN are not JSON, and a
    # document containing one fails to parse in its entirety.
    text = json.dumps(sensor_catalog(), allow_nan=False)
    assert json.loads(text) == sensor_catalog()


def test_catalog_covers_every_registered_sensor():
    names = [s["name"] for s in sensor_catalog()]
    assert names == list_sensors()
    # Vacuity floor: the registry is not empty, and the two rigs the map
    # readouts default to are both in it.
    assert len(names) >= 2  # the CityNav crop and the C-12's Oryx, at the least
    assert {"nadir_1008_90", "oryx_orx32_25mm_nadir"} <= set(names)


def test_every_entry_carries_a_usable_hfov():
    for entry in sensor_catalog():
        hfov = entry["hfov_deg"]
        assert isinstance(hfov, float)
        assert 0.0 < hfov < 180.0, entry["name"]


def test_infinite_range_becomes_null_not_infinity():
    # nadir_1008_90 has no range envelope (max_range_m = inf). The catalog
    # says so as null; to_dict still says inf, which is what the episode
    # record wants.
    entry = next(s for s in sensor_catalog() if s["name"] == "nadir_1008_90")
    assert entry["max_range_m"] is None
    assert resolve_sensor("nadir_1008_90").to_dict()["max_range_m"] == float("inf")
