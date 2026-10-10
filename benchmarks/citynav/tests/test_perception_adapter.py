"""CityRefObject → GTObject mapping for the gt_perfect diagnostic sensor."""

from __future__ import annotations

from benchmarks.citynav.perception_adapter import citynav_to_gt_perfect_objects
from benchmarks.citynav.adapter import CityRefObject


def _cityref(
    id_: int,
    name: str,
    object_type: str,
    pos: tuple[float, float, float],
    contour: list[tuple[float, float]] | None = None,
) -> CityRefObject:
    return CityRefObject(
        map_name="test_block",
        id=id_,
        name=name,
        object_type=object_type,
        position=pos,
        dimension=(1.0, 1.0, 1.0),
        contour=contour or [],
    )


def test_label_is_class_noun_never_the_cityrefer_name():
    # Anti-leak wall: the CityRefer name must NEVER reach the GTObject
    # label (the runtime surfaces node names to the reasoner).
    cityref = [_cityref(1, "Old Hall", "Building", (10.0, 20.0, 0.0))]
    gts = citynav_to_gt_perfect_objects(cityref)
    assert len(gts) == 1
    assert gts[0].label == "building"
    assert gts[0].class_hint == "buildings"
    assert gts[0].object_type == "Building"
    assert gts[0].world_xy == (10.0, 20.0)
    assert gts[0].id == 1


def test_unknown_object_type_falls_back_to_lowercased_type():
    cityref = [_cityref(2, "Weird Thing", "UnknownCategory", (0.0, 0.0, 5.0))]
    gts = citynav_to_gt_perfect_objects(cityref)
    assert gts[0].label == "unknowncategory"
    assert gts[0].class_hint == "unknowncategory"


def test_traffic_road_maps_to_roads_bucket_and_z_drops():
    cityref = [_cityref(3, "", "TrafficRoad", (5.0, 6.0, 100.0))]
    gts = citynav_to_gt_perfect_objects(cityref)
    assert gts[0].label == "road"
    assert gts[0].class_hint == "roads"
    assert gts[0].world_xy == (5.0, 6.0)


def test_contour_and_gt_color_row_carried():
    contour = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    cityref = [_cityref(4, "Red Car", "Car", (1.0, 2.0, 0.0), contour=contour)]
    table = {"4": {"color": "red", "color_conf": 0.9}}
    gts = citynav_to_gt_perfect_objects(cityref, color_table=table)
    assert gts[0].contour == contour
    assert gts[0].attributes == {"color": "red", "color_conf": 0.9}
    # Empty contour collapses to None; missing table row to {}.
    bare = citynav_to_gt_perfect_objects([_cityref(5, "", "Car", (0.0, 0.0, 0.0))])
    assert bare[0].contour is None
    assert bare[0].attributes == {}
