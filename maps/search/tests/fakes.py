"""Shared fakes for the search tests: a scripted reasoner, mission drivers, block compilers."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np

from maps.llm.endpoint_spec import EndpointSpec
from maps.world_state.graph import MapObject

TEST_SPEC = EndpointSpec(
    base_url="http://unused",
    model_id="inkling-small",
    max_model_len=262144,
    tokenizer_name="qwen",
    max_output_tokens=40000,
)


class FakeReasoner:
    """Scripted turn source: a turn is a plain string (a text reply) or a list
    of (tool_name, args_dict) calls emitted together. Vision turns land in
    ``vision`` as (text, jpegs)."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.exhausted = 0
        self.messages = [{"role": "system", "content": "sys"}]
        self.vision: list[tuple[str, list]] = []
        self.compaction_infeasible = False
        self.cancel_event = None
        self.model = "test"

    def append_user(self, content):
        self.messages.append({"role": "user", "content": content})

    def append_repair(self, message):
        self.messages.append(
            {"role": "user", "content": f"[Repair] {message}"})

    def append_tool_result(self, cid, name, result):
        content = result if isinstance(result, str) else json.dumps(result)
        self.messages.append({"role": "tool", "tool_call_id": cid,
                              "name": name, "content": content})

    def append_vision_user(self, text, jpegs):
        self.vision.append((text, list(jpegs)))
        self.messages.append({"role": "user", "content": text})

    def step_tool_call(self, tools):
        # On exhaustion reply (the mission's ending) at most twice, so a
        # script need not answer the loop's own repair nudge.
        if not self.turns:
            self.exhausted += 1
            assert self.exhausted <= 2, "loop did not return after two replies"
            self.messages.append({"role": "assistant", "content": ""})
            return {"kind": "text", "content": "", "reasoning": "",
                    "finish_reason": "stop"}
        t = self.turns.pop(0)
        if isinstance(t, dict):  # a text turn with its own finish_reason
            self.messages.append({"role": "assistant", "content": t["content"]})
            return {"kind": "text", "content": t["content"], "reasoning": "",
                    "finish_reason": t["finish_reason"]}
        if isinstance(t, str):
            self.messages.append({"role": "assistant", "content": t})
            return {"kind": "text", "content": t, "reasoning": "",
                    "finish_reason": "stop"}
        calls = [{"id": f"c{len(self.messages)}_{i}", "type": "function",
                  "function": {"name": n, "arguments": json.dumps(a)}}
                 for i, (n, a) in enumerate(t)]
        self.messages.append({"role": "assistant", "content": "",
                              "tool_calls": calls})
        return {"kind": "tool_call", "tool_calls": calls, "content": "",
                "reasoning": "", "finish_reason": "tool_calls"}


def bare_ctx(graph=None, grid_shape=(10, 10), pose_xy=None):
    return SimpleNamespace(graph=SimpleNamespace(nodes={}) if graph is None else graph, cell_size_m=1.0,
                           tile_origin_xy=(0.0, 0.0), grid_shape=grid_shape,
                           pose_xy=pose_xy)


def search_loop(nodes=None, *, cell_size_m=2.0, sam_setter=None):
    """A started SearchLoop over a graph holding ``nodes`` (``{fid: node}``),
    tile origin (0, 0)."""
    from maps.search.search_loop import SearchLoop
    from maps.world_state.graph import GridDomain

    loop = SearchLoop(GridDomain(height=10, width=10, cell_size_m=cell_size_m),
                      tile_origin_xy=(0.0, 0.0), map_name="test", sam_setter=sam_setter,
                      rgbd_dir="")
    loop.start(SimpleNamespace(nodes=dict(nodes or {})))
    return loop


def run(code: str):
    """One ``run`` tool call."""
    return ("run", {"code": code})


def tool_results(fr: FakeReasoner) -> list[dict]:
    out = []
    for m in fr.messages:
        if m.get("role") == "tool":
            try:
                out.append(json.loads(m["content"]))
            except Exception:
                out.append({"raw": m["content"]})
    return out


CANDS = [
    {"fid": "prior_7", "xy": [3.0, 4.0], "class": "depot",
     "search_text": "depot"},
    {"fid": "prior_1", "xy": [10.0, 20.0], "class": "gate",
     "search_text": "gate"},
    {"fid": "prior_2", "xy": [10.0, 40.0], "class": "tower",
     "search_text": "tower"},
    {"fid": "car_1", "xy": [20.0, 20.0], "class": "car",
     "search_text": "car"},
    {"fid": "car_2", "xy": [0.0, 30.0], "class": "car",
     "search_text": "car"},
]


def unarmed_boost(concepts, where):
    """A boost channel that refuses every boost."""
    return {"error": "boost queueing is not armed this episode"}


def drive(script, *, fly=None, flight_budget=8, turn_cap=48,
          transcript=None, cands=None, retask=None, look_renderer=None,
          reasoner=None, **loop_kw):
    """One operator turn over a still surface. Returns (reply, reasoner,
    flights, declared); ``reasoner`` (a returned one) continues its mission
    with ``script`` as the next operator turn (its flights and declarations
    keep landing in the first turn's lists); ``loop_kw`` reaches
    ``_Mission``."""
    from maps.search.parse_compile.reasoner_loop import (
        MissionReply,
        MissionSpec,
        _Mission,
        continue_mission,
    )

    if reasoner is not None:
        reasoner.turns += list(script)
        mission, flights, declared = reasoner.mission
        reply = continue_mission(mission, "Find the goal object.", rid_tag="m1")
        assert isinstance(reply, MissionReply)
        return reply, reasoner, flights, declared

    flights: list[dict] = []

    def _fly(item):
        flights.append(item)
        if fly is not None:
            return fly(item)
        return {"stop_id": f"stop_{len(flights) - 1:03d}",
                "target": item["target_fid"], "pose_xy": [1.0, 2.0],
                "new_detections": 1, "cue": "the cue"}

    declared: list[list[dict]] = []

    def _declare(items):
        declared.append(list(items))

    fr = FakeReasoner(script)
    loop_kw.setdefault("boost_queue", unarmed_boost)
    mission = _Mission(
        reasoner=fr,
        ctx=bare_ctx(),
        rid_tag="m0",
        transcript=transcript,
        compute_namespace={"cands": list(CANDS if cands is None else cands),
                           "anchors": [], "cell_m": 1.0},
        diag={},
        mission_spec=MissionSpec(fly=_fly, declare_concepts=_declare,
                                 flight_budget=flight_budget,
                                 turn_cap=turn_cap),
        retask=retask,
        look_renderer=look_renderer,
        **loop_kw,
    )
    fr.mission = (mission, flights, declared)
    reply = mission.converse("Find the goal object.")
    assert isinstance(reply, MissionReply)
    return reply, fr, flights, declared


# --- explore: an engine that reports a footprint's stops -------------------

STOPS = [[100.0, 200.0], [200.0, 200.0], [300.0, 200.0]]
START = (7.0, 9.0)


def explore_drive(script, *, stops=STOPS, obs_span_m=200.0, cands=None,
                  obs_minor_m=None, with_renderer=False,
                  deadline_epoch=None, on_fly=None, held_now=None):
    """The mission with coverage stops; each flight reports its own target
    point as its pose. ``stops`` is one list every fid answers or a dict by
    fid. Returns (reply, tool results, flights, stops asked, rendered)."""
    from maps.search.parse_compile.reasoner_loop import (
        MissionSpec,
        _Mission,
    )
    from maps.xy_token import parse_xy_token

    flights: list[dict] = []

    def _fly(item):
        flights.append(item)
        if on_fly is not None:
            on_fly(item)
        _pt = parse_xy_token(item["target_fid"]) or (1.0, 2.0)
        return {"stop_id": f"stop_{len(flights) - 1:03d}",
                "target": item["target_fid"],
                "pose_xy": [float(_pt[0]), float(_pt[1])],
                "new_detections": 1, "cue": "the cue"}

    asked: list[tuple] = []

    def _stops(fid, spacing_m):
        asked.append((fid, spacing_m))
        got = stops.get(fid) if isinstance(stops, dict) else stops
        return [list(p) for p in got] if got else None

    rendered: list = []

    def _renderer(reqs):
        rendered.append(list(reqs))
        return [{"jpeg": b"x", "center_fid": r["center_fid"],
                 "legend": [{"digit": 1, "label": "car", "role": "center"}],
                 "context_m": 30.0, "world_window": [0, 0, 200, 200]}
                for r in reqs]

    fr = FakeReasoner(script)
    reply = _Mission(
        boost_queue=unarmed_boost,
        reasoner=fr, ctx=bare_ctx(pose_xy=(START[0], START[1])),
        rid_tag="m0",
        compute_namespace={"cands": list(CANDS if cands is None else cands),
                           "anchors": [], "cell_m": 1.0},
        diag={}, obs_span_m=obs_span_m,
        obs_minor_m=obs_minor_m,
        look_renderer=_renderer if with_renderer else None,
        mission_spec=MissionSpec(held_now=held_now,
            fly=_fly, declare_concepts=lambda i: None,
            coverage_stops=_stops, flight_budget=32, turn_cap=48,
            deadline_epoch=deadline_epoch),
    ).converse("Find the goal object.")
    return reply, tool_results(fr), flights, asked, rendered


# --- a graph the flight grows ----------------------------------------------

class GraphNode(MapObject):
    """A graph node at world point ``xy`` (cell_m 1, origin 0)."""

    def __init__(self, xy, class_name, provenance="detected"):
        super().__init__(class_name=class_name, name=None, tags={},
                         centroid=(int(xy[1]), int(xy[0])), provenance=provenance,
                         world_xy=[float(xy[0]), float(xy[1])])


class Road(GraphNode):
    """A prior with a baked centerline (cell -> world is the cell centre at
    cell_m 1), so ``crossings`` can construct a place from two of them."""

    def __init__(self, cells, name):
        (r0, c0) = cells[len(cells) // 2]
        super().__init__((c0 + 0.5, r0 + 0.5), "road", provenance="prebaked")
        self.centerline = list(cells)
        self.name = name
        self.axis_length_m = float(len(cells))


def priors():
    return {
        "prior_7": GraphNode((30.0, 30.0), "depot", provenance="prebaked"),
        "prior_1": GraphNode((80.0, 80.0), "gate", provenance="prebaked"),
    }


# Two roads that meet at one place.
ROADS = {
    "road_1": Road([(100, c) for c in range(80, 121)], "Long Road"),
    "road_2": Road([(r, 100) for r in range(80, 121)], "Cross Street"),
}


def grow_drive(script, *, per_flight=3, flights_add=True, evict_on_flight=None,
               extra_priors=None):
    """One mission over a graph each flight grows by ``per_flight`` cars (one
    black, the rest white, each flight on its own line). ``evict_on_flight``
    = {flight_index: [fid, ...]} deletes nodes on that flight. Returns (tool
    results, the graph, the fids each flight added)."""
    from maps.query.surface import build_surface_from_graph
    from maps.search.parse_compile.reasoner_loop import (
        MissionSpec,
        _Mission,
    )

    graph = SimpleNamespace(nodes={**priors(), **(extra_priors or {})})
    added: list[list[str]] = []

    def _fly(item):
        for _dead in (evict_on_flight or {}).get(len(added)) or []:
            graph.nodes.pop(_dead, None)
        new: list[str] = []
        if flights_add:
            base = len(graph.nodes)
            for i in range(per_flight):
                fid = f"detected_{base + i}"
                graph.nodes[fid] = GraphNode(
                    (100.0 + 6.0 * i, 100.0 + 20.0 * len(added)),
                    "black car" if i == 0 else "white car")
                new.append(fid)
        added.append(new)
        return {"stop_id": f"stop_{len(added) - 1:03d}",
                "target": item["target_fid"], "pose_xy": [100.0, 100.0],
                "alt_m": 144.8, "new_detections": len(new), "cue": "the cue"}

    cands, anchors = build_surface_from_graph(
        graph.nodes.items(), (0.0, 0.0), 1.0)
    fr = FakeReasoner(script)
    _Mission(
        boost_queue=unarmed_boost,
        reasoner=fr,
        ctx=bare_ctx(graph=graph),
        rid_tag="m0",
        transcript=[],
        compute_namespace={"cands": cands, "anchors": anchors,
                           "cell_m": 1.0},
        diag={},
        look_renderer=lambda reqs: [
            {"jpeg": b"x", "center_fid": r["center_fid"], "legend": [
                {"digit": 1, "label": "car", "role": "center"}] + [
                {"digit": 3 + i, "label": "car", "role": "context"}
                for i, m in enumerate(r["marks"])
                if m["fid"] != r["center_fid"]],
             "context_m": 30.0, "world_window": [0, 0, 200, 200],
             "marks_dropped": 0, "marks_out_of_view": [],
             "marks_unbadged_in_frame": []}
            for r in reqs],
        mission_spec=MissionSpec(fly=_fly, declare_concepts=lambda i: None),
    ).converse("Find the goal object.")
    return tool_results(fr), graph, added


def map_mission(script, renderer=None, finds=1, extra_priors=None, spread=False):
    """``script`` (one block per turn) over two surveyed buildings (plus
    ``extra_priors``), each flight adding ``finds`` cars at one point, or
    along a row with ``spread``; returns (tool results, vision turns)."""
    from maps.query.surface import build_surface_from_graph
    from maps.search.parse_compile.reasoner_loop import (
        MissionSpec,
        _Mission,
    )

    graph = SimpleNamespace(nodes={**priors(), **(extra_priors or {})})
    for n in graph.nodes.values():
        n.cells = np.asarray([[int(n.world_xy[1]), int(n.world_xy[0])]], dtype=np.int32)
        n.tags = {"building": "yes"}

    cars_added = [0]

    def _fly(item):
        for _ in range(finds):
            x = 30 + 4 * cars_added[0] if spread else 34
            cars_added[0] += 1
            node = GraphNode((float(x), 34.0), "car")
            node.cells = np.asarray([[34, x]], dtype=np.int32)
            node.tags = {}
            graph.nodes[f"detected_{len(graph.nodes)}"] = node
        return {"stop_id": "stop_001", "target": item["target_fid"], "pose_xy": [30.0, 30.0],
                "alt_m": 100.0, "new_detections": finds, "cue": "the cue"}

    cands, anchors = build_surface_from_graph(graph.nodes.items(), (0.0, 0.0), 1.0)
    fr = FakeReasoner([[run(c)] for c in script] + ["done"])
    _Mission(
        boost_queue=unarmed_boost,
        reasoner=fr, ctx=bare_ctx(graph=graph, grid_shape=(100, 100)),
        rid_tag="m0", transcript=[],
        compute_namespace={"cands": cands, "anchors": anchors, "cell_m": 1.0},
        diag={}, look_renderer=renderer or (lambda reqs: []),
        mission_spec=MissionSpec(fly=_fly, declare_concepts=lambda i: None),
    ).converse("Find the car.")
    return tool_results(fr), fr.vision


# --- the real look renderer over a synthetic ortho -------------------------

ORTHO_MAP = "unit_test_ortho"
ORTHO_GROUND_Z = 10.0


def cars(n: int, step_m: float = 6.0) -> list[dict]:
    """``n`` cars on a square lattice centred near (100, 100)."""
    side = int(n ** 0.5) + 1
    return [{"fid": f"detected_{i}",
             "xy": [100.0 + step_m * (i % side - side / 2.0),
                    100.0 + step_m * (i // side - side / 2.0)],
             "class": "car", "search_text": "car"} for i in range(n)]


def look_drive(script, cands, ortho_dir, map_name=ORTHO_MAP):
    """One mission with the REAL look renderer; returns (tool results, the
    requests the renderer saw, the rendered records, the badge-legend
    vision texts)."""
    from maps.search.parse_compile.reasoner_loop import (
        MissionSpec,
        _Mission,
    )
    from maps.render.look_render import render_candidate_crops

    seen_reqs: list[list[dict]] = []
    seen_recs: list[list[dict]] = []

    def renderer(reqs):
        seen_reqs.append(reqs)
        recs = render_candidate_crops(reqs, map_name=map_name, rgbd_dir=ortho_dir)
        seen_recs.append(recs)
        return recs

    fr = FakeReasoner(script)
    _Mission(
        boost_queue=unarmed_boost,
        reasoner=fr,
        ctx=bare_ctx(),
        rid_tag="m0",
        transcript=[],
        compute_namespace={"cands": list(cands), "anchors": [],
                           "cell_m": 1.0},
        diag={},
        look_renderer=renderer,
        mission_spec=MissionSpec(fly=lambda i: {"error": "unused"},
                                 declare_concepts=lambda i: None),
    ).converse("find the car")
    vision = [str(m.get("content")) for m in fr.messages
              if m.get("role") == "user" and "badge legend" in
              str(m.get("content"))]
    return tool_results(fr), seen_reqs, seen_recs, vision


# --- one block run against a recording env --------------------------------

def quiet_env(**kw):
    """A MissionEnv whose verbs do nothing and refuse nothing; ``kw``
    overrides any field."""
    from maps.search.parse_compile import python_surface as ps

    def nop(*_a):
        return {}

    fields = dict(
        rows_of_ref=lambda ref: [], plan_rn=[0], widened={}, minted=[],
        say_err=lambda ack, msg: {**ack, "error": msg},
        ground=lambda names: {"ties": {n: [n] for n in names}},
        analyze=nop, stand=nop, move=nop, cover=nop, crops=nop, map=nop,
        watch=nop, boost=nop, finish=lambda ack: ack,
        row_of_fid=lambda fid: None, tied_of_ref=lambda ref: False,
        version=lambda: 0)
    fields.update(kw)
    return ps.MissionEnv(**fields)


def record_block(code: str, bound: dict | None = None, rn: int = 0,
                 widened: dict | None = None) -> tuple[list[tuple], dict]:
    """Run ONE block against an env whose verbs record: (calls, ack).

    Each call is ``(verb, *args)``; a grounding is recorded only when it
    names something, and a NAME grounds to the one fid ``"<name>"``.
    ``bound`` seeds the namespace as an earlier block would: ``rN`` binds a
    result handle, any other string a row's fid, anything else as is."""
    import re

    from maps.search.parse_compile import python_surface as ps

    calls: list[tuple] = []
    counter = [rn]

    def verb(name):
        def f(*args):
            calls.append((name, *args))
            return {}
        return f

    def ground(names):
        if names:
            calls.append(("ground", dict(names)))
        return {"ties": {n: [f"<{n}>"] for n in names}}

    def analyze(ops, rn_end, answer, page):
        calls.append(("analyze", ops, answer, page))
        counter[0] = max(counter[0], rn_end)
        return {}

    mp = ps.MissionPython(quiet_env(
        plan_rn=counter, widened=widened if widened is not None else {},
        ground=ground, analyze=analyze,
        **{v: verb(v) for v in ("stand", "move", "cover", "crops", "map",
                                "watch", "boost")}))
    for name, held in (bound or {}).items():
        if isinstance(held, str) and re.fullmatch(r"r\d+", held):
            mp.ns[name] = ps.Rows(held, mp)
        elif isinstance(held, str):
            mp.ns[name] = ps.Row({"fid": held}, mp=mp)
        else:
            mp.ns[name] = held
    return calls, mp.run_block(code)


def calls_of_block(code: str, bound: dict | None = None, rn: int = 0,
                   widened: dict | None = None) -> list[tuple]:
    """The verb calls ONE block makes, raising its refusal when it made
    none."""
    from maps.search.parse_compile import python_surface as ps

    calls, out = record_block(code, bound, rn, widened)
    if "error" in out and not calls:
        raise ps.CodeError(str(out["error"]))
    return calls


def acts_of_block(code: str, bound: dict | None = None) -> list[tuple]:
    """The act calls ONE block makes: no groundings, no programs."""
    return [c for c in calls_of_block(code, bound)
            if c[0] not in ("ground", "analyze")]


def raising_ops_of_block(code: str, bound: dict | None = None,
                         rn: int = 0) -> list[dict]:
    """Every op ONE block compiles, raising the block's own refusal even when
    an earlier line already ran."""
    from maps.search.parse_compile import python_surface as ps

    calls, out = record_block(code, bound, rn)
    if "error" in out:
        raise ps.CodeError(str(out["error"]))
    return [op for c in calls if c[0] == "analyze" for op in c[1]]


def ops_of_block(code: str, bound: dict | None = None,
                 rn: int = 0) -> list[dict]:
    """Every op ONE block compiles, in order."""
    return [op for c in calls_of_block(code, bound, rn) if c[0] == "analyze"
            for op in c[1]]


def grounded_in_block(code: str, bound: dict | None = None) -> list[str]:
    """The NAMES one block grounds, in order."""
    return [n for c in calls_of_block(code, bound) if c[0] == "ground"
            for n in c[1]]


# --- small builders --------------------------------------------------------

def ring(x, y, w, h):
    """An axis-aligned w x h rectangle as one closed ring."""
    return [[(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]]


def jpeg(side: int = 256) -> bytes:
    import pytest

    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new("RGB", (side, side), (90, 90, 90)).save(buf, format="JPEG")
    return buf.getvalue()
