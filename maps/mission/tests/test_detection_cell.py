from types import SimpleNamespace

from maps.mission.episode_world import _detection_to_mapobject


def test_a_point_south_west_of_the_tile_origin_lands_in_the_cell_it_is_in():
    det = SimpleNamespace(world_xy=(99.0, 99.0), label="car", attributes={}, metadata={})
    assert _detection_to_mapobject(det, (100.0, 100.0), 2.0).centroid == (-1, -1)
