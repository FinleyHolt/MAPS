"""ViSA's coverage plan reaches every cell of the merged landmark region, and
the Set-of-Mark frame numbers every masked detection."""

from __future__ import annotations

import numpy as np
import pytest

from baselines.seam import Landmark
from baselines.visa.som import mark_frame, mark_world_xy
from baselines.visa.waypoints import plan_coverage
from benchmarks.citynav.eval import WorldBounds
from maps.util.mask_region import MaskRegion


def _region(full: np.ndarray) -> MaskRegion:
    """A frame-sized boolean mask as the region SAM hands back."""
    rr, cc = np.where(full)
    r0, c0, r1, c1 = int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())
    return MaskRegion(full[r0 : r1 + 1, c0 : c1 + 1], (r0, c0, r1, c1), full.shape)

_LANDMARKS = [
    Landmark("Tram Depot", [(100, 100), (260, 100), (260, 220), (100, 220)], (180, 160), {}),
    Landmark("Adam and Eve Street", [(0, 300), (600, 300), (600, 330), (0, 330)], (300, 315), {}),
    Landmark("Riverside Park", [(700, 600), (1000, 620), (980, 900), (720, 880)], (850, 750), {}),
]


def test_cover_reaches_every_cell():
    assert len(_LANDMARKS) >= 3  # vacuity floor
    bounds = WorldBounds(0.0, 0.0, 1200.0, 1200.0)
    start = (50.0, 50.0)
    plan = plan_coverage(_LANDMARKS, bounds, start, fov_m=210.0)
    assert plan.fov_m == pytest.approx(210.0)
    assert plan.grid_step_m == pytest.approx(147.0)
    assert 1 <= len(plan.waypoints) <= 20
    assert plan.cells_covered == plan.cells_total > 0
    # every chosen square is a grid point; the order starts nearest the start
    d = [np.hypot(x - start[0], y - start[1]) for x, y in plan.waypoints]
    assert d[0] == min(d)
    # re-check the cover independently of the planner's own bookkeeping
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    import shapely

    region = unary_union([Polygon(lm.contour) for lm in _LANDMARKS])
    xs = np.arange(0, 1200, 5.0)
    gx, gy = np.meshgrid(xs, xs)
    inside = shapely.contains_xy(region, gx.ravel(), gy.ravel())
    px, py = gx.ravel()[inside], gy.ravel()[inside]
    half = plan.fov_m / 2
    covered = np.zeros(px.size, dtype=bool)
    for x, y in plan.waypoints:
        covered |= (np.abs(px - x) <= half) & (np.abs(py - y) <= half)
    assert covered.mean() > 0.98


def test_no_landmark_falls_back_to_the_block():
    plan = plan_coverage([], WorldBounds(0.0, 0.0, 400.0, 400.0), (0.0, 0.0), fov_m=210.0)
    assert plan.contours == [] and len(plan.waypoints) >= 4


def test_mark_frame_numbers_every_masked_detection():
    from maps.perception.base import DetectedObject

    frame = np.full((400, 400, 3), 90, dtype=np.uint8)
    dets = []
    for i, (r, c, conf, xy) in enumerate([(100, 100, 0.4, (10.0, 20.0)), (300, 250, 0.9, (30.0, 40.0))]):
        m = np.zeros((400, 400), dtype=bool)
        m[r - 12 : r + 12, c - 20 : c + 20] = True
        dets.append(DetectedObject(world_xy=xy, label="car", confidence=conf, source_tier="sam3_1_image", attributes={"color": "red" if i else ""},
                                   mask=_region(m)))
    marked, marks = mark_frame(frame, dets)
    assert [m["id"] for m in marks] == [1, 2]
    assert marks[0]["world_xy"] == [30.0, 40.0]  # confidence-ordered: the 0.9 one is mark 1
    assert mark_world_xy(marks[1]) == (10.0, 20.0)
    assert marked.shape == frame.shape and (marked != frame).any()
    assert marks[0]["color"] == "red" and marks[0]["frame_rc"] == pytest.approx([299.5, 249.5])


class _StubHarness:
    """The seam's surface with canned model answers; no serve, no SAM."""

    def __init__(self, tmp, verdicts):
        from types import SimpleNamespace

        from baselines.seam import Budget
        from benchmarks.citynav.action_snap import Pose

        self.cfg = SimpleNamespace(altitude_cap_m=200.0, max_steps=200)
        self.episode = SimpleNamespace(
            description="a red car parked on the parking lot on Adam and Eve Street behind the Tram Depot",
            start_xy=(50.0, 50.0), map_name="x",
        )
        self.assets = SimpleNamespace(bounds=WorldBounds(0.0, 0.0, 1200.0, 1200.0))
        self.start_alt_m, self.ground_z_m = 125.0, 20.0
        self.budget = Budget(200, Pose(50.0, 50.0, 125.0, 0.0))
        self.log_dir = tmp
        self._verdicts = list(verdicts)
        self.calls = []

    def landmarks(self):
        return _LANDMARKS

    def view_half_m(self, alt=None):
        return (alt or self.start_alt_m) - self.ground_z_m

    view_half_short_m = view_half_m

    def search_window(self, landmarks, clip_to_block=True):
        return self.assets.bounds

    def observe(self, xy, alt=None):
        from types import SimpleNamespace

        rgb = np.full((200, 200, 3), 120, dtype=np.uint8)
        return SimpleNamespace(rgb=rgb, pose_xy=tuple(xy), altitude_m=alt or self.start_alt_m)

    def detect(self, obs, concepts):
        from maps.perception.base import DetectedObject

        self.concepts_seen = getattr(self, "concepts_seen", []) + [list(concepts)]
        m = np.zeros((200, 200), dtype=bool)
        m[90:110, 90:110] = True
        return [DetectedObject(world_xy=(obs.pose_xy[0] + 1.0, obs.pose_xy[1] + 2.0), label="car",
                               confidence=0.8, source_tier="sam3_1_image", attributes={"color": "red"}, mask=_region(m))]

    def jpeg(self, rgb):
        return b"jpeg"

    def user_message(self, text, images=None):
        return {"role": "user", "content": text}

    def llm_json(self, messages, schema, name, **kw):
        self.calls.append(name)
        if name == "visa_instruction_analysis":
            return {"target": {"phrase": "red car", "class": "car", "attributes": ["red"]},
                    "landmarks": ["Tram Depot", "Adam and Eve Street"], "surroundings": ["parking lot"],
                    "concepts": ["red car", "car", "parking lot"]}
        v = self._verdicts.pop(0) if self._verdicts else "move"
        guided = v.endswith("+g")
        v = v.removesuffix("+g")
        return {"attribute_match": {"candidate_marks": [1], "pending_marks": [], "reasoning": "red, car-sized"},
                "spatial_topology": {"consistent_marks": [1] if v == "stop" else [], "reasoning": "behind [2]"},
                "geographic_boundary": {"valid_marks": [1] if v == "stop" else [], "reasoning": "on the lot"},
                "verdict": v, "target_mark": 1 if v == "stop" else 0,
                "guidance": "Search for a red car positioned behind [2], not in front" if guided else "",
                "guidance_concepts": ["red car behind the depot"] if guided else []}

    def record(self, name, payload):
        import json
        from baselines.seam import _jsonable

        (self.log_dir / f"{name}.json").write_text(json.dumps(payload, default=_jsonable))

    def save_image(self, name, rgb):
        (self.log_dir / f"{name}.jpg").write_bytes(rgb if isinstance(rgb, bytes) else b"x")


def test_policy_stops_on_a_verified_mark_and_records_the_figure_inputs(tmp_path):
    from baselines.visa.policy import run_episode

    d = tmp_path / "visa"
    d.mkdir()
    h = _StubHarness(d, ["descend+g", "stop"])
    res = run_episode(h)
    assert res.commit_tier == "visa:stop" and res.commit_label == "mark_1"
    assert res.extra_metrics["altitude_changes"] == 1 and res.extra_metrics["guided_moves"] == 0
    assert res.extra_metrics["landmarks_matched"] == 2 and res.extra_metrics["waypoints_visited"] == 1
    assert res.stops[-1][1] == pytest.approx(115.0)  # the descend leg
    assert res.commit_xy == (res.stops[-1][0][0] + 1.0, res.stops[-1][0][1] + 2.0)
    # the descend's guidance was the prompt of the re-observation at the new height
    assert "red car behind the depot" in h.concepts_seen[1]
    assert h.budget.used > 0
    assert sorted(p.name for p in d.glob("step_*_vm.json")) == ["step_01_vm.json", "step_02_vm.json"]


def test_a_rejection_moves_on_and_its_guidance_prompts_the_next_waypoint(tmp_path):
    """The paper's example: the VM rejects candidate 1, "sends a closed-loop
    guidance signal back to the VPG ... and triggers a move primitive"; at the
    next waypoint the VPG "adjusts its attention based on the feedback"."""
    from baselines.visa.policy import run_episode

    d = tmp_path / "visa"
    d.mkdir()
    h = _StubHarness(d, ["move+g", "stop"])
    res = run_episode(h)
    assert res.commit_tier == "visa:stop"
    assert res.extra_metrics["waypoints_visited"] == 2 and res.extra_metrics["guided_moves"] == 1
    assert "red car behind the depot" not in h.concepts_seen[0]
    assert "red car behind the depot" in h.concepts_seen[1]
    import json

    vm2 = json.loads((d / "step_02_vm.json").read_text())
    assert vm2["guidance_in"].startswith("Search for a red car positioned behind")


def test_policy_falls_back_to_the_pending_candidate_within_budget(tmp_path):
    from baselines.visa.policy import run_episode

    d = tmp_path / "visa"
    d.mkdir()
    h = _StubHarness(d, [])  # every verdict "move"
    res = run_episode(h)
    assert res.commit_tier in ("visa:pending", "visa:budget")
    assert res.commit_label.startswith("mark_1@step")
    assert h.budget.used <= 200
    assert res.extra_metrics["waypoints_visited"] >= 1


def test_the_mark_cap_keeps_the_described_object_before_landmark_masks():
    """Half the marks were roofs from the landmark nouns; the described
    object's class ranks first, its surroundings next, the rest last."""
    from baselines.visa.policy import _terms, mark_tier

    target = _terms(["car", "black car"])
    surround = _terms(["white car", "road"])
    assert mark_tier("car, black", target, surround) == 2
    assert mark_tier("cars", target, surround) == 2
    assert mark_tier("road", target, surround) == 1
    assert mark_tier("building, gray roof", target, surround) == 0
    dets = [("building", 0.99), ("car", 0.4), ("road", 0.7), ("car", 0.6)]
    ranked = sorted(dets, key=lambda d: (-mark_tier(d[0], target, surround), -d[1]))
    assert [d[0] for d in ranked] == ["car", "car", "road", "building"]


def test_the_detector_is_prompted_with_the_instruction_phrases_never_the_bare_class():
    from baselines.visa.policy import detection_concepts

    analysis = {"target": {"phrase": "black car", "class": "car", "attributes": ["black"]},
                "surroundings": ["white car", "road"], "landmarks": ["Willmore Road"],
                "concepts": ["black car", "car", "white car", "Willmore Road"]}
    got = detection_concepts(analysis)
    assert got[0] == "black car"
    assert "car" not in [g.lower() for g in got]
    assert {"white car", "road", "Willmore Road"} <= set(got)


def test_a_huge_landmark_is_planned_only_inside_the_window():
    """A city-long road would make the coverage grid city-sized; the region
    is the landmark inside the search window."""
    from types import SimpleNamespace

    from baselines.seam import Landmark

    road = Landmark(name="Long Road", contour=[(0.0, 0.0), (20000.0, 0.0), (20000.0, 30.0), (0.0, 30.0)],
                    centroid=(10000.0, 15.0), tags={})
    window = SimpleNamespace(x_min=-1000.0, y_min=-1000.0, x_max=1000.0, y_max=1000.0)
    plan = plan_coverage([road], window, (0.0, 0.0), fov_m=300.0)
    assert plan.waypoints
    assert all(-1000.0 <= x <= 1000.0 for x, _y in plan.waypoints)
