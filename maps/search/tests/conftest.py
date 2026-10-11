"""Search-test fixtures: a synthetic ortho per module (MAP, GROUND_Z, ORTHO_N, ORTHO_RAMP
module attributes override the fakes defaults)."""
from __future__ import annotations

import pytest

from maps.search.tests.fakes import ORTHO_GROUND_Z, ORTHO_MAP
from maps.search.tests.ortho_fixture import write_ortho


def _ortho_params(module):
    return (getattr(module, "MAP", ORTHO_MAP), getattr(module, "GROUND_Z", ORTHO_GROUND_Z))


@pytest.fixture(scope="module")
def _ortho_root(request, tmp_path_factory):
    name, ground_z = _ortho_params(request.module)
    return write_ortho(tmp_path_factory.mktemp("ortho"), name, ground_z=ground_z,
                       n=getattr(request.module, "ORTHO_N", 2000),
                       ramp=getattr(request.module, "ORTHO_RAMP", True))


@pytest.fixture
def ortho_dir(_ortho_root, request, monkeypatch):
    from maps.perception.ortho import camera_model
    from maps.perception.ortho.ground_level import GROUND_LEVEL

    monkeypatch.setitem(GROUND_LEVEL, *_ortho_params(request.module))
    camera_model._ORTHO_RASTER_CACHE.clear()
    return _ortho_root
