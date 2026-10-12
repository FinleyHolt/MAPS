from __future__ import annotations

import pytest

from benchmarks.flight.tests import world


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """The synthetic world, built once: ``(tiles_dir, raster_dir, ortho, sensor)``."""
    from benchmarks.flight import map_render
    from maps.perception.geometry.sensor_spec import resolve_sensor

    tiles, raster = world.build(tmp_path_factory.mktemp("world"))
    return tiles, raster, map_render.load_raster(raster, "calcity"), resolve_sensor(world.SENSOR)


def _recording_legs(vehicle):
    """``vehicle`` with every leg posted to it kept in ``legs``, for a test to read."""
    vehicle.legs, post = [], vehicle.post
    vehicle.post = lambda leg: (vehicle.legs.append(leg), post(leg))[-1]
    return vehicle


@pytest.fixture
def make_twin(built):
    def make(start_xy=(600.0, 100.0)):
        from benchmarks.flight.twin import Twin

        tiles, _raster, ortho, sensor = built
        twin = Twin(tiles_dir=tiles, map_name="calcity", ortho=ortho, sensor=sensor, start_xy=start_xy,
                    agl_m=world.AGL_M)
        return _recording_legs(twin)
    return make


@pytest.fixture
def make_live(built):
    def make(vehicle, register=False):
        from benchmarks.flight.live_source import open_live

        tiles, raster, _ortho, sensor = built
        return open_live(vehicle=vehicle, tiles_dir=tiles, map_name="calcity", sensor=sensor,
                         register_raster=raster if register else None, corrector=False)
    return make
