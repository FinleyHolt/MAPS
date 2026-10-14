"""GeoNav's episode loop, as released, over the harness.

``scenegraphnav/agent.py::GeonavAgent.run`` (github.com/Xhtshr/geonav-official):
one extraction call stands in for ``processed_descriptions.json``; the
planner call names the sub-goals; then at most 20 perceive -> MLLM ->
teleport iterations under Navigate / Search / Locate with the released
step sizes, stage-switch rules and query ladder. Departures from the release
are marked ``# substitution:``; release behaviours the paper does not
describe are kept and marked ``# released:``.
"""

from __future__ import annotations

import math

import numpy as np

from baselines.geonav import prompts as P
from baselines.geonav.cognitive_map import CognitiveMap, rgb_entropy
from baselines.geonav.scene_graph import KnowledgeGraph, QueryEngine, ingest_subgraph, recursive_query
from baselines.seam import BaselineHarness, BaselineResult, landmark_distance, match_landmarks, pixel_bbox

# gsamllavanav/parser.py: --eval_max_timestep 20 (one perceive + one decision each).
MAX_TIMESTEPS = 20
# gsam_params: max_box_size 50 m, max_box_area 3000 m2.
MAX_BOX_SIZE_M = 50.0
MAX_BOX_AREA_M2 = 3000.0
# agent.py: the VLM extractor fires after 20 steps or 20 m under Search.
VLM_DETECTION_INTERVAL = 20
VLM_DETECTION_DISTANCE_M = 20.0
# agent.py::call_response max_tokens; understand() sends none.
CALL_MAX_TOKENS = 800
# substitution: understand() sends no max_tokens, so on the released client
# the DashScope default output cap (2,048 tokens for the qwen3-vl-plus API)
# bounded it; the same bound, stated.
UNDERSTAND_MAX_TOKENS = 2048
# check_subgoals: Navigate ends past t=12 or with every landmark within 40 m;
# Search ends 5 steps after the last Navigate check or on a flat entropy.
NAVIGATE_MAX_T = 12
LANDMARK_REACHED_M = 40.0
SEARCH_STEPS = 5
ENTROPY_THRESHOLD = 1e-8

OPPOSITE_DIRS = {
    "north": "south", "south": "north", "east": "west", "west": "east",
    "northeast": "southwest", "southwest": "northeast", "northwest": "southeast", "southeast": "northwest",
}
ACTION_SCALE = {"Navigate": 1, "Search": 3}
SYS_PROMPT = {"Navigate": P.GOAL_DESCRIPTION_NAV, "Search": P.GOAL_DESCRIPTION_SEA, "Locate": P.GOAL_DESCRIPTION_LOC}
STRATEGIES = ("Navigate", "Search", "Locate")
JSON_SUFFIX = (
    "\n\nPlease strictly follow the requirements below when responding:\n"
    "1. Use standard JSON format\n2. Include all required fields\n3. Wrap the response with ```json"
)

_STR = {"type": "string"}
_STRS = {"type": "array", "items": _STR}


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


EXTRACT_SCHEMA = _obj({"target": _STR, "landmarks": _STRS, "surroundings": _STRS})
PLAN_SCHEMA = _obj({
    "plan": _STR,
    "sub_goals": {"type": "array", "items": _obj({
        "goal": _STR, "desired_state": _STR, "strategy": {"type": "string", "enum": list(STRATEGIES)},
    })},
})
ACTION_SCHEMA = _obj({"reason": _STR, "movement": _STR})
LOCATE_SCHEMA = _obj({
    "reason": _STR,
    "selected_pos": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
})
GRAPH_SCHEMA = _obj({
    "nodes": {"type": "array", "items": _obj({
        "id": _STR, "object_type": _STR,
        "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
        "color": _STR,
    })},
    "edges": {"type": "array", "items": _obj({"source": _STR, "target": _STR, "relationship": _STR})},
})
CHAIN_SCHEMA = _obj({
    "operations": {"type": "array", "items": _obj({
        "method": _STR, "args": _STRS, "kwargs": _obj({"relation_type": _STR}),
    })},
})


# --- pure pieces (tested offline) --------------------------------------------

def direction_map(base_distance: float, multiplier: float) -> dict[str, tuple[float, float]]:
    """``get_next_position``'s table, in its insertion order: diagonals first
    so ``"northwest"`` is not matched by ``"north"``."""
    b = base_distance * multiplier
    return {
        "northwest": (-b, b), "northeast": (b, b), "southwest": (-b, -b), "southeast": (b, -b),
        "north": (0, b), "south": (0, -b), "west": (-b, 0), "east": (b, 0), "stop": (0, 0),
    }


def next_position(pose, mode: str, target_json, view_width: float, last_movement: str, history_actions: list[str]):
    """``GeonavAgent.get_next_position``, as released."""
    base_distance = view_width / 6
    multiplier = 1.5 if mode == "Navigate" else 1
    if not isinstance(target_json, dict):
        history_actions.append("stop")
        return pose
    if mode in ("Navigate", "Search"):
        dmap = direction_map(base_distance, multiplier)
        direction = str(target_json.get("movement", "stop")).lower()
        # released: the guard keys the opposite map by the whole lowercased
        # movement string ("move north"), so it only fires on a bare word.
        if OPPOSITE_DIRS.get(direction) == last_movement:
            history_actions.append("stop")
            return pose
        dx, dy = dmap["stop"]
        for key in dmap:
            if key in direction:
                dx, dy = dmap[key]
                history_actions.append(key)
                break
        else:
            history_actions.append("stop")
        scale = ACTION_SCALE.get(mode, 1)
        return (pose[0] + dx / scale, pose[1] + dy / scale)
    if mode == "Locate":
        if "selected_pos" in target_json:
            sp = target_json["selected_pos"]
            history_actions.append(str(sp))
            return (float(sp[0]), float(sp[1]))
        history_actions.append("stop")
    return pose


def is_complex_query(query: str) -> bool:
    if "." in query and query.count(".") > 1:
        return True
    keywords = ["in front of", "behind", "next to", "beside", "between", "across from", "opposite",
                "facing", "same direction", "opposite direction", "parked", "in", "on"]
    if sum(k in query.lower() for k in keywords) >= 2:
        return True
    return query.lower().count(" and ") > 1 or query.lower().count(",") > 1


def clamp_to_bounds(xy, bounds) -> tuple[float, float]:
    # substitution: a teleport outside the block has no ortho under it.
    return (min(max(float(xy[0]), bounds.x_min), bounds.x_max), min(max(float(xy[1]), bounds.y_min), bounds.y_max))


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# --- the episode ---------------------------------------------------------------

def _bbox_corners_world(h: BaselineHarness, obs, r0, c0, r1, c1):
    hh, ww = obs.rgb.shape[:2]
    # substitution: the released extractor read Qwen3-VL-PLUS boxes as pixels
    # of its 1024 px crop; a model that answers in unit fractions of the frame
    # would land every node in its top-left corner, so fractions scale to pixels.
    if all(0.0 <= float(v) <= 1.0 for v in (r0, c0, r1, c1)):
        r0, r1, c0, c1 = r0 * (hh - 1), r1 * (hh - 1), c0 * (ww - 1), c1 * (ww - 1)
    cl = lambda v, hi: min(max(float(v), 0.0), float(hi - 1))  # noqa: E731
    r0, r1, c0, c1 = cl(r0, hh), cl(r1, hh), cl(c0, ww), cl(c1, ww)
    return [h.pixel_to_world(obs, r0, c0), h.pixel_to_world(obs, r0, c1),
            h.pixel_to_world(obs, r1, c1), h.pixel_to_world(obs, r1, c0)]


def _default_plan(instruction: str) -> dict:
    return {"plan": instruction, "sub_goals": [
        {"goal": instruction, "desired_state": "near the landmark", "strategy": "Navigate"},
        {"goal": instruction, "desired_state": "target observed", "strategy": "Search"},
        {"goal": instruction, "desired_state": "target located", "strategy": "Locate"},
    ]}


def run_episode(h: BaselineHarness, released: bool = False) -> BaselineResult:
    """The episode. ``released`` selects the as-shipped arm: the released
    token caps under a thinking dial the seam sets to ``none``."""
    # substitution: the released caps assumed a backbone with no reasoning
    # span. Thinking tokens count against max_tokens here, so an 800-token call
    # returns an empty object (measured on a recorded run). The headline arm
    # takes the seam's free-call cap (None: the served output reserve).
    call_cap = CALL_MAX_TOKENS if released else None
    understand_cap = UNDERSTAND_MAX_TOKENS if released else None
    ep = h.episode
    instruction = ep.description

    # substitution: one extraction call in place of processed_descriptions.json.
    ext = h.llm_json(
        [{"role": "user", "content": P.EXTRACTION_PROMPT.format(instruction=instruction)}],
        EXTRACT_SCHEMA, "geonav_extract", max_tokens=call_cap,
    )
    target_name = (str(ext.get("target") or "").strip() or instruction)
    surroundings = [str(s).strip() for s in ext.get("surroundings") or [] if str(s).strip()]
    matched = match_landmarks([str(s) for s in ext.get("landmarks") or []], h.landmarks())
    h.record("00_extraction", {"instruction": instruction, "response": ext,
                               "matched_landmarks": [lm.name for lm in matched]})

    # substitution: the map and every teleport stay inside the search window
    # (the block on CityNav, where blocks are narrower than its minimum).
    bounds = h.search_window(matched)
    graph = KnowledgeGraph()
    for lm in matched:
        graph.add_geo_node(lm)
    engine = QueryEngine(graph)
    cmap = CognitiveMap((bounds.x_min, bounds.y_min, bounds.x_max, bounds.y_max), h.ground_z_m,
                        matched, target_name)

    pose = (float(ep.start_xy[0]), float(ep.start_xy[1]))
    alt = h.start_alt_m  # released: altitude never changes
    t = 0
    plan = None
    task_idx = 0
    sys_prompt = None
    subtask_status = "pending"
    history_actions: list[str] = []
    last_movement = ""
    switch_time = 0
    previous_sem_map = np.ones((*cmap.shape, 3), dtype=np.float32)
    last_vlm_t = 0
    last_vlm_pos = pose
    strategy = ""
    stops: list[tuple[tuple[float, float], float]] = []
    commit_xy = None
    commit_label = "geonav:none"
    commit_tier = "geonav:timeout"
    lock_step = None
    understand_calls = 0
    locate_path = None
    complex_query = None

    while t < MAX_TIMESTEPS:
        obs = h.observe(pose, alt)
        t += 1
        frame = h.jpeg(obs.rgb)
        cmap.add_pose(pose, alt, half_m=h.view_half_short_m(alt))

        # The detector paints the maps (released: GroundingDINO + MobileSAM,
        # one pass per caption set; substitution: one SAM 3.1 pass over the
        # union, split by label).
        half = h.view_half_m(alt)
        ppm_frame = obs.rgb.shape[1] / (2.0 * half)
        painted = 0
        for det in h.detect(obs, [target_name] + surroundings):
            bb = pixel_bbox(det)
            if bb is None:
                continue
            r0, c0, r1, c1 = bb
            w_m, h_m = (c1 - c0) / ppm_frame, (r1 - r0) / ppm_frame
            if not (w_m < MAX_BOX_SIZE_M and h_m < MAX_BOX_SIZE_M and w_m * h_m < MAX_BOX_AREA_M2):
                continue
            cmap.paint_detection(str(det.label), _bbox_corners_world(h, obs, r0, c0, r1, c1))
            painted += 1

        # released: build_geo_nodes every step (idempotent), then the geo text.
        for lm in matched:
            graph.add_geo_node(lm)
        geo_text = engine.is_within_geo_node(pose) + engine.get_enhanced_geo_relation(pose)

        if strategy == "Search" and (
            t - last_vlm_t >= VLM_DETECTION_INTERVAL or _dist(pose, last_vlm_pos) >= VLM_DETECTION_DISTANCE_M
        ):
            sub = _understand(h, frame, instruction, [target_name] + surroundings, frame_px=obs.rgb.shape[:2], max_tokens=understand_cap)
            landed = ingest_subgraph(
                graph, sub,
                lambda bbox, _o=obs: _bbox_corners_world(h, _o, bbox[1], bbox[0], bbox[3], bbox[2]),
            )
            engine = QueryEngine(graph)
            understand_calls += 1
            last_vlm_t, last_vlm_pos = t, pose
            h.record(f"{t:02d}_understand", {"response": sub, "nodes_landed": landed, "graph_nodes": len(graph.nodes)})

        geoinstruct = P.LANDMARK_DESCRIPTION.format(landmark=geo_text, recent_objects="", surroundings="")
        view_width = 2.0 * half
        xyxy = [(pose[0] - view_width / 2, pose[1] - view_width / 2), (pose[0] + view_width / 2, pose[1] + view_width / 2)]

        if plan is None:
            plan = h.llm_json(
                [{"role": "user", "content": P.PLANNER_PROMPTV2.format(instruction=instruction, geoinstruct=geoinstruct)}],
                PLAN_SCHEMA, "geonav_plan", max_tokens=call_cap,
            )
            if not plan.get("sub_goals"):
                plan = _default_plan(instruction)
            h.record("01_plan", plan)
        task = plan["sub_goals"][task_idx]
        if subtask_status == "pending":
            # released: subtask_status never returns to "pending", so this
            # system prompt (the Navigate one) serves the whole episode.
            sys_prompt = SYS_PROMPT[task["strategy"]]
            subtask_status = "running"

        mode = task["strategy"]
        action_json = None
        target_pos = None
        if mode == "Navigate":
            img = cmap.render("landmark")
            text = P.LANDMARK_NAVIGATION_PROMPT.format(geoinstruct=geoinstruct, goal=task["goal"], state=task["desired_state"])
        elif mode == "Search":
            objs = [(c["id"], graph.nodes[c["id"]].position, c["target"])
                    for c in engine.get_context(pose, radius=500.0) if c["type"] == "object"]
            img = cmap.render("semantic", objs)
            text = P.OBJECT_SEARCH_PROMPT.format(geoinstruct=geoinstruct, goal=task["goal"], state=task["desired_state"])
        else:  # Locate
            complex_query = is_complex_query(task["goal"])
            # released: the complex path raises on a return-shape mismatch
            # and falls through to this chain path, so the chain path is
            # the only one that ever resolves.
            chain = _operation_chain(h, task["goal"], max_tokens=call_cap)
            nodes = recursive_query(engine, chain)
            h.record(f"{t:02d}_locate_query", {"goal": task["goal"], "complex_query": complex_query,
                                               "operation_chain": chain,
                                               "result_ids": [n.id for n in nodes] if nodes else []})
            if nodes:
                best = max(nodes, key=lambda n: getattr(n, "confidence", 0))
                target_pos = best.position
                commit_label, commit_tier, locate_path = f"geonav:{best.id}", "geonav:graph", "graph"
                img, text = None, None
            else:
                img = obs.rgb
                text = P.TARGET_LOCATE_PROMPT.format(pos=pose, area=xyxy, goal=instruction)

        if text is not None:
            schema, name = (LOCATE_SCHEMA, "geonav_locate") if mode == "Locate" else (ACTION_SCHEMA, "geonav_action")
            # substitution: the released client sends the image part first; the harness builds text first.
            messages = [{"role": "system", "content": sys_prompt}, h.user_message(text + JSON_SUFFIX, [img])]
            try:
                action_json = h.llm_json(messages, schema, name, max_tokens=call_cap)
            except RuntimeError:
                action_json = None  # released: three failed tries stay put
            if mode != "Locate":
                h.save_image(f"{t:02d}_{mode.lower()}_map", h.jpeg(img))
            h.record(f"{t:02d}_{mode.lower()}", {"prompt": text, "response": action_json, "pose": pose,
                                                  "geoinstruct": geo_text, "painted": painted})
            target_pos = next_position(pose, mode, action_json, view_width, last_movement, history_actions)
            if mode == "Locate" and isinstance(action_json, dict) and "selected_pos" in action_json:
                commit_label, commit_tier, locate_path = "geonav:selected_pos", "geonav:vlm_pos", "vlm_pos"
            elif mode == "Locate":
                commit_label, commit_tier, locate_path = "geonav:pose", "geonav:locate_pose", "pose"

        next_pos = clamp_to_bounds(target_pos if target_pos is not None else pose, bounds)
        if mode == "Locate":
            commit_xy = next_pos
            lock_step = t
        if next_pos != pose:
            if not h.budget.affords(next_pos, alt):
                if mode != "Locate":
                    commit_xy, commit_label, commit_tier, lock_step = pose, "geonav:budget", "geonav:budget", t
                break
            h.budget.move(next_pos, alt)
            pose = next_pos
            stops.append((pose, alt))
        strategy = mode

        # check_subgoals, as released.
        done = False
        if mode == "Navigate":
            switch_time = t
            last_movement = history_actions[-1] if history_actions else "stop"
            # substitution: released measures to a CityRefer object's centre; an
            # OSM street unioned and buffered has a centroid far from every
            # point of it, so the test reads distance to the landmark's area.
            done = t > NAVIGATE_MAX_T or all(landmark_distance(lm, pose) < LANDMARK_REACHED_M for lm in matched)
        elif mode == "Search":
            if t > switch_time + SEARCH_STEPS:
                done = True
            else:
                sem = cmap.semantic_map_array()
                if abs(rgb_entropy(sem) - rgb_entropy(previous_sem_map)) > ENTROPY_THRESHOLD:
                    previous_sem_map = sem.copy()
                else:
                    done = True
        else:
            done = True
        if done:
            task_idx += 1
            if task_idx >= len(plan["sub_goals"]):
                break

    if commit_xy is None:
        commit_xy = pose
    return BaselineResult(
        stops=stops,
        commit_xy=commit_xy,
        commit_label=commit_label,
        commit_tier=commit_tier,
        lock_step=lock_step,
        extra_metrics={
            "stage_at_end": strategy or "none",
            "timesteps": t,
            "understand_calls": understand_calls,
            "graph_nodes": len(graph.nodes),
            "landmarks_matched": len(matched),
            "locate_path": locate_path or "none",
            "complex_query": bool(complex_query),
        },
    )


def _understand(h: BaselineHarness, frame: bytes, instruction: str, objects: list[str], *, frame_px, max_tokens) -> dict:
    """``LLMController.understand``: the scene-graph extractor over the raw frame."""
    text = P.LOCAL_GRAPH_PROMPT.format(instruction=instruction, objects=objects)
    # substitution: the frame's pixel size, stated ahead of the verbatim prompt,
    # because the released 1024 px crop is the run's sensor frame here and the
    # prompt's own example boxes are in pixels of the image it sees.
    text = f"The image is {frame_px[1]}x{frame_px[0]} pixels; bbox values are pixel coordinates of this image.\n" + text
    messages = [
        {"role": "system", "content": "You are a geospatial scene graph extractor analyzing a north-aligned satellite image."},
        h.user_message(text, [frame]),
    ]
    try:
        return h.llm_json(messages, GRAPH_SCHEMA, "geonav_understand", max_tokens=max_tokens)
    except RuntimeError:
        return {"nodes": [], "edges": []}  # released: two failed tries yield an empty graph


def _operation_chain(h: BaselineHarness, goal: str, *, max_tokens) -> list[dict]:
    """``GeonavAgent.generate_operation_chain``: text-only, through call_response."""
    try:
        out = h.llm_json(
            [{"role": "user", "content": P.QUERY_OPERATION_CHAIN_PROMPT.format(instruction=goal)}],
            CHAIN_SCHEMA, "geonav_chain", max_tokens=max_tokens,
        )
    except RuntimeError:
        return []
    chain = []
    for op in out.get("operations") or []:
        step: dict = {"method": str(op.get("method", ""))}
        if op.get("args"):
            step["args"] = list(op["args"])
        rel = (op.get("kwargs") or {}).get("relation_type")
        if rel:
            step["kwargs"] = {"relation_type": rel}
        chain.append(step)
    return chain
