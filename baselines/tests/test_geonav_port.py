"""The GeoNav port reproduces the released code's arithmetic, offline.

No LLM, no SAM: the direction table, the opposite-move guard, the merge rule,
the query ladder's relaxation, the validator's coercion and the cognitive-map
render are the pieces a wrong port would get wrong silently.
"""

from __future__ import annotations

import math

import numpy as np

from baselines.geonav.cognitive_map import CognitiveMap
from baselines.geonav.policy import direction_map, next_position
from baselines.geonav.scene_graph import GeoNode, KnowledgeGraph, QueryEngine
from baselines.seam import Landmark


def _square(cx, cy, s):
    return [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)]


def _lm(name, cx, cy, s=20.0):
    return Landmark(name=name, contour=_square(cx, cy, s), centroid=(cx, cy), tags={"name": name})


def test_navigate_step_is_view_width_over_six_times_one_and_a_half():
    # 125 m over ground 20 -> view width 210 m -> base 35 -> Navigate 52.5 m.
    view_width = 2 * (125.0 - 20.0)
    dmap = direction_map(view_width / 6, 1.5)
    assert dmap["north"] == (0, 52.5)
    assert math.isclose(math.hypot(*dmap["northwest"]), 52.5 * math.sqrt(2))
    hist: list[str] = []
    nxt = next_position((0.0, 0.0), "Navigate", {"movement": "Move northwest"}, view_width, "", hist)
    assert nxt == (-52.5, 52.5) and hist == ["northwest"]
    # Search divides the unscaled base by three.
    assert next_position((0.0, 0.0), "Search", {"movement": "Move east"}, view_width, "", []) == (35.0 / 3, 0.0)


def test_opposite_move_guard_fires_only_on_a_bare_direction_word():
    hist: list[str] = []
    # released: the guard keys OPPOSITE_DIRS by the whole movement string.
    assert next_position((0.0, 0.0), "Navigate", {"movement": "south"}, 60.0, "north", hist) == (0.0, 0.0)
    assert hist == ["stop"]
    assert next_position((0.0, 0.0), "Navigate", {"movement": "Move south"}, 60.0, "north", []) == (0.0, -15.0)


def test_merge_rule_merges_under_a_metre_and_not_at_three():
    g = KnowledgeGraph()
    a = g.add_object_node_with_attrs((0.0, 0.0), {"obj_class": "vehicle", "confidence": 1.0, "timestamp": 1})
    b = g.add_object_node_with_attrs((0.5, 0.0), {"obj_class": "vehicle", "confidence": 1.0, "timestamp": 1})
    c = g.add_object_node_with_attrs((3.0, 0.0), {"obj_class": "vehicle", "confidence": 1.0, "timestamp": 1})
    assert a == b == "vehicle_0" and c == "vehicle_1" and len(g.nodes) == 2


def test_query_ladder_keeps_the_previous_result_on_an_empty_filter():
    g = KnowledgeGraph()
    g.add_geo_node(_lm("Davey Road", 0.0, 0.0))
    g.add_object_node_with_attrs((5.0, 0.0), {"obj_class": "vehicle", "confidence": 1.0, "timestamp": 1, "color": "white"})
    g.add_edge("landmark_Davey Road", "vehicle_0", "contains")
    e = QueryEngine(g)
    chain = [
        {"method": "get_geonode_by_name", "args": ["Davey Road"]},
        {"method": "get_child_nodes", "kwargs": {"relation_type": "contains"}},
        {"method": "filter_by_class", "args": ["vehicle"]},
        {"method": "filter_by_attribute", "args": ["color", "purple"]},
    ]
    out = e.robust_subgraph_query(chain)
    assert [n.id for n in out] == ["vehicle_0"]


def test_get_child_nodes_coerces_adjacent_to_into_contains():
    g = KnowledgeGraph()
    g.add_geo_node(_lm("Park", 0.0, 0.0))
    g.add_object_node_with_attrs((30.0, 0.0), {"obj_class": "vehicle", "confidence": 1.0, "timestamp": 1})
    g.add_object_node_with_attrs((5.0, 0.0), {"obj_class": "building", "confidence": 1.0, "timestamp": 1})
    g.add_edge("landmark_Park", "vehicle_0", "adjacent_to")
    g.add_edge("landmark_Park", "building_0", "contains")
    e = QueryEngine(g)
    parent = next(n for n in g.nodes.values() if isinstance(n, GeoNode))
    # released: adjacent_to is not in the validator's list, so it reads as contains.
    assert [n.id for n in e.get_child_nodes(parent, "adjacent_to")] == ["building_0"]


def test_cognitive_map_renders_the_released_canvas():
    lms = [_lm("A", 50.0, 50.0), _lm("B", 150.0, 60.0), _lm("C", 90.0, 160.0)]
    cmap = CognitiveMap((0.0, 0.0, 200.0, 200.0), 0.0, lms, "red car")
    assert len({tuple(c) for c in np.unique(cmap.landmark_color.reshape(-1, 4), axis=0) if c[3]}) >= 3
    cmap.add_pose((100.0, 100.0), 50.0)
    cmap.paint_detection("tree", _square(120.0, 120.0, 3.0))
    for kind in ("landmark", "semantic"):
        rgb = cmap.render(kind, [("tree_0", (120.0, 120.0), False)])
        assert rgb.shape == (1000, 1000, 3) and rgb.dtype == np.uint8
        assert rgb.std() > 1.0
    assert cmap.semantic_map_array().max() > 0


def test_run_episode_over_a_stub_harness():
    """The loop end to end with canned model replies: it plans, navigates,
    searches, locates through the graph, and commits with a policy tier."""
    from types import SimpleNamespace

    from PIL import Image
    import io

    from baselines.geonav.policy import run_episode
    from baselines.seam import Budget
    from benchmarks.citynav.action_snap import Pose

    canned = {
        "geonav_extract": {"target": "red car", "landmarks": ["Davey Road"], "surroundings": ["tree"]},
        "geonav_plan": {"plan": "p", "sub_goals": [
            {"goal": "go", "desired_state": "s", "strategy": "Navigate"},
            {"goal": "look", "desired_state": "s", "strategy": "Search"},
            {"goal": "red car parked on Davey Road", "desired_state": "s", "strategy": "Locate"},
        ]},
        "geonav_action": {"reason": "r", "movement": "Move north"},
        "geonav_understand": {"nodes": [{"id": "Red01", "object_type": "vehicle", "bbox": [100, 100, 140, 120], "color": "red"}],
                              "edges": []},
        "geonav_chain": {"operations": [
            {"method": "get_geonode_by_name", "args": ["Davey Road"], "kwargs": {"relation_type": ""}},
            {"method": "get_child_nodes", "args": [], "kwargs": {"relation_type": "contains"}},
            {"method": "filter_by_class", "args": ["vehicle"], "kwargs": {"relation_type": ""}},
        ]},
        "geonav_locate": {"reason": "r", "selected_pos": [120.0, 130.0]},
    }
    calls: list[str] = []

    class Stub:
        episode = SimpleNamespace(description="the red car parked on Davey Road", start_xy=(100.0, 100.0), map_name="m")
        assets = SimpleNamespace(bounds=SimpleNamespace(x_min=0.0, y_min=0.0, x_max=400.0, y_max=400.0))
        start_alt_m = 125.0
        ground_z_m = 20.0
        budget = Budget(200, Pose(100.0, 100.0, 125.0, 0.0))

        def observe(self, xy, alt):
            half = alt - self.ground_z_m
            corners = ((xy[0] - half, xy[1] + half), (xy[0] + half, xy[1] + half),
                       (xy[0] + half, xy[1] - half), (xy[0] - half, xy[1] - half))
            return SimpleNamespace(rgb=np.zeros((256, 256, 3), np.uint8), ortho_view_corners_xy=corners, pose_xy=xy)

        def view_half_m(self, alt=None):
            return (alt or self.start_alt_m) - self.ground_z_m

        view_half_short_m = view_half_m

        def search_window(self, landmarks):
            return self.assets.bounds

        def pixel_to_world(self, obs, row, col):
            hh, ww = obs.rgb.shape[:2]
            u, v = col / (ww - 1), row / (hh - 1)
            (flx, fly), (frx, fry), _b, (blx, bly) = obs.ortho_view_corners_xy
            return (flx + u * (frx - flx) + v * (blx - flx), fly + u * (fry - fly) + v * (bly - fly))

        def detect(self, obs, concepts):
            return []

        def landmarks(self):
            return [_lm("Davey Road", 140.0, 140.0, 60.0), _lm("Far", 380.0, 380.0)]

        def jpeg(self, rgb):
            buf = io.BytesIO()
            Image.fromarray(np.asarray(rgb, np.uint8)).save(buf, format="JPEG")
            return buf.getvalue()

        def user_message(self, text, images=None):
            return {"role": "user", "content": text}

        def llm_json(self, messages, schema, name, **kw):
            calls.append(name)
            return dict(canned[name])

        def record(self, name, payload):
            return None

        def save_image(self, name, rgb):
            return None

    res = run_episode(Stub())
    assert calls[:2] == ["geonav_extract", "geonav_plan"]
    assert "geonav_action" in calls and "geonav_chain" in calls
    assert res.commit_tier in ("geonav:graph", "geonav:vlm_pos")
    assert res.stops and res.extra_metrics["landmarks_matched"] == 1
    assert res.lock_step is not None and res.commit_xy is not None
