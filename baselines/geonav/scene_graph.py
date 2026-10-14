"""GeoNav's hierarchical scene graph and query engine, as released.

``scenegraphnav/city_scene_graph.py`` and the graph half of
``scenegraphnav/llm_controller.py`` (github.com/Xhtshr/geonav-official),
with the R-tree, networkx and sentence-transformers dependencies replaced by
lists and dicts over at most a few hundred nodes. Every rule, threshold and
release behaviour is kept; each departure is marked ``# substitution:`` and
each replicated defect ``# released:``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from difflib import SequenceMatcher

from shapely.geometry import Point, Polygon
from shapely.validation import make_valid

from baselines.seam import Landmark

# utils/tools.py, verbatim.
DIRECTION_NAMES = ["East", "Northeast", "North", "Northwest", "West", "Southwest", "South", "Southeast"]


def get_direction(src: tuple[float, float], target: tuple[float, float]) -> str:
    dx = target[0] - src[0]
    dy = target[1] - src[1]
    angle = math.degrees(math.atan2(dy, dx)) % 360
    return DIRECTION_NAMES[round(angle / 45) % 8]


def _dist(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class KnowledgeNode:
    def __init__(self, node_id: str, node_type: str, position: tuple[float, float]) -> None:
        self.id = node_id
        self.type = node_type
        self.position = (float(position[0]), float(position[1]))


class GeoNode(KnowledgeNode):
    """A landmark: ``{object_type}_{name}``."""

    def __init__(self, landmark: Landmark) -> None:
        # substitution: CityRefer's object_type is the OSM row's category here.
        object_type = str(landmark.tags.get("category") or "landmark")
        super().__init__(f"{object_type}_{landmark.name}", "geo", landmark.centroid)
        self.name = landmark.name
        self.object_type = object_type
        ring = list(landmark.contour)
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        poly = Polygon(ring)
        self.contour_polygon = poly if poly.is_valid else make_valid(poly)


class ObjectNode(KnowledgeNode):
    def __init__(self, detection_id, position, obj_class, confidence, target=False, **attrs):
        super().__init__(detection_id, "object", position)
        self.obj_class = obj_class
        self.confidence = confidence
        self.target = target
        for key, value in attrs.items():
            setattr(self, key, value)


class KnowledgeGraph:
    """Nodes, directed edges carrying relation lists, and the merge rules."""

    def __init__(self) -> None:
        self.nodes: dict[str, KnowledgeNode] = {}
        self.class_counters: defaultdict[str, int] = defaultdict(int)
        self.edges: dict[tuple[str, str], list[str]] = {}
        self.alpha = 0.95
        self.spatial_threshold = 4.0

    # released: rtree ``intersection`` over a (c-r, c-r, c+r, c+r) box, so
    # the radius query is a square, not a disc.
    def query_radius(self, center, radius: float) -> list[KnowledgeNode]:
        return [
            n for n in self.nodes.values()
            if abs(n.position[0] - center[0]) <= radius and abs(n.position[1] - center[1]) <= radius
        ]

    def add_edge(self, source_id: str, target_id: str, relation_type: str) -> None:
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Cannot create edge between non-existing nodes")
        rels = self.edges.setdefault((source_id, target_id), [])
        if relation_type not in rels:
            rels.append(relation_type)

    def out_edges(self, node_id: str):
        return [(s, t, r) for (s, t), r in self.edges.items() if s == node_id]

    def in_edges(self, node_id: str):
        return [(s, t, r) for (s, t), r in self.edges.items() if t == node_id]

    def add_geo_node(self, landmark: Landmark) -> None:
        self._add_node(GeoNode(landmark), "geo")

    def add_object_node_with_attrs(self, position, attrs: dict) -> str:
        attrs = dict(attrs)
        obj_class = attrs.pop("obj_class")
        confidence = attrs.pop("confidence")
        target = attrs.pop("target", False)
        node_id = f"{obj_class}_{self.class_counters[obj_class]}"
        node = ObjectNode(node_id, position, obj_class, confidence, target, **attrs)
        merged = self._add_node_v2(node, obj_class)
        return merged if merged else node_id

    def _add_node(self, node: KnowledgeNode, obj_class: str) -> str | None:
        for n in self.query_radius(node.position, 20.0):
            if n.type == node.type and _dist(n.position, node.position) < 5.0:
                return n.id
        self.class_counters[obj_class] += 1
        self.nodes[node.id] = node
        return None

    def _add_node_v2(self, node: KnowledgeNode, obj_class: str) -> str | None:
        merge_candidate = None
        max_score = -1.0
        for n in self.query_radius(node.position, 20.0):
            # released: the semantic term compares node KIND ('geo'/'object'),
            # not class, so two object nodes score 1.0 there and merge iff the
            # spatial term alone clears 0.8, i.e. within ~0.95 m.
            score = self.alpha * self._spatial_similarity(n.position, node.position) \
                + (1 - self.alpha) * self._semantic_similarity(n.type, node.type)
            if score > max_score and score >= 0.8:
                max_score = score
                merge_candidate = n
        if merge_candidate:
            return merge_candidate.id
        self.class_counters[obj_class] += 1
        self.nodes[node.id] = node
        return None

    def _spatial_similarity(self, p1, p2) -> float:
        return math.exp(-_dist(p1, p2) / self.spatial_threshold)

    @staticmethod
    def _semantic_similarity(type1: str, type2: str) -> float:
        # substitution: upstream scores unequal kinds with a MiniLM cosine
        # (a small positive number); 0.0 here moves the cross-kind merge
        # radius from ~0.76 m to ~0.69 m.
        return 1.0 if type1 == type2 else 0.0


# --- rule edges (llm_controller.get_object_geo_relations, the active block) --

def polygon_to_polygon_distance(poly1: Polygon, poly2: Polygon) -> float:
    """Minimum VERTEX-pair distance, 0 when the polygons intersect."""
    if poly1.intersects(poly2):
        return 0.0
    # released: a vertex-to-vertex scan, not the edge distance shapely gives.
    min_dist = float("inf")
    for e1 in poly1.exterior.coords[:-1]:
        for e2 in poly2.exterior.coords[:-1]:
            d = math.hypot(e1[0] - e2[0], e1[1] - e2[1])
            if d < min_dist:
                min_dist = d
    return min_dist


def object_geo_relations(object_pos, corners, geo_node: GeoNode) -> list[str]:
    polygon = geo_node.contour_polygon
    obj_polygon = Polygon(corners)
    if not obj_polygon.is_valid:
        obj_polygon = make_valid(obj_polygon)
    min_distance = polygon_to_polygon_distance(obj_polygon, polygon)
    if polygon.contains(obj_polygon):
        return ["contains"]
    if min_distance < 20.0:
        return ["adjacent_to"]
    if min_distance < 35.0:
        return ["near_corner"]
    return [f"{get_direction(geo_node.position, object_pos).lower()}_of"]


def ingest_subgraph(graph: KnowledgeGraph, subgraph: dict, bbox_to_corners) -> int:
    """``llm_controller.build_scene_graph``: the extractor's nodes and edges
    into the graph. ``bbox_to_corners(bbox) -> [(x, y)] * 4`` is the frame
    geometry. Returns the number of nodes that landed."""
    if not isinstance(subgraph, dict):
        return 0
    nodes = [
        n for n in subgraph.get("nodes", [])
        if isinstance(n, dict) and n.get("bbox") and isinstance(n.get("object_type"), str)
    ]
    id_mapping: dict = {}
    landed = 0
    for node in nodes:
        try:
            corners = bbox_to_corners(node["bbox"])
            pos = (sum(c[0] for c in corners) / 4, sum(c[1] for c in corners) / 4)
            attrs = {"obj_class": node.get("object_type", "unknown"), "confidence": 1.0}
            attrs.update({k: v for k, v in node.items() if k not in ("id", "object_type", "bbox")})
            global_id = graph.add_object_node_with_attrs(pos, attrs)
            if global_id:
                id_mapping[node.get("id")] = global_id
                landed += 1
                for gn in list(graph.nodes.values()):
                    if isinstance(gn, GeoNode):
                        for rel in object_geo_relations(pos, corners, gn):
                            graph.add_edge(gn.id, global_id, rel)
        except Exception:  # noqa: BLE001 -- released: a bad node is skipped, never fatal
            continue
    for edge in subgraph.get("edges", []):
        if not isinstance(edge, dict) or "source" not in edge or "target" not in edge:
            continue
        source = id_mapping.get(edge["source"])
        target = id_mapping.get(edge["target"])
        if not (source and target):
            continue
        graph.add_edge(source, target, edge.get("relationship") or "related_to")
    return landed


# --- the query engine --------------------------------------------------------

class QueryEngine:
    DIRECTION_REVERSE_MAP = {
        "north_of": "south_of", "south_of": "north_of", "east_of": "west_of", "west_of": "east_of",
        "northeast_of": "southwest_of", "southwest_of": "northeast_of",
        "southeast_of": "northwest_of", "northwest_of": "southeast_of",
        "overlaps": "overlaps", "separates": "separates",
    }

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def get_context(self, position, radius=10.0) -> list[dict]:
        out = []
        for n in self.graph.query_radius(position, radius):
            d = {"type": n.type, "distance": f"{_dist(n.position, position):.1f} meters",
                 "direction": get_direction(position, n.position)}
            if isinstance(n, GeoNode):
                d.update(name=n.name, category=n.object_type)
            elif isinstance(n, ObjectNode):
                d.update(id=n.id, **{"class": n.obj_class}, confidence=n.confidence, target=n.target)
            out.append(d)
        return out

    def get_geonode_by_name(self, name_pattern: str, candidates=None, threshold=0.8):
        nodes = candidates if candidates else self.graph.nodes.values()
        results = []
        for n in nodes:
            if isinstance(n, GeoNode):
                if name_pattern.lower() in n.name.lower():
                    results.append((n, 1.0))
                else:
                    sim = SequenceMatcher(None, name_pattern.lower(), n.name.lower()).ratio()
                    if sim >= threshold:
                        results.append((n, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in results]

    def get_child_nodes(self, parent_node, relation_type, candidates=None, bidirectional=True):
        parent_nodes = parent_node if isinstance(parent_node, list) else [parent_node]
        # released: this validator speaks overlaps/separates while the graph is
        # built with adjacent_to/near_corner, so those two are coerced to
        # ``contains`` here.
        valid_relations = [
            "contains", "overlaps", "separates",
            "north_of", "south_of", "east_of", "west_of",
            "northeast_of", "northwest_of", "southeast_of", "southwest_of",
        ]
        if isinstance(relation_type, list):
            target_relations = relation_type
        elif isinstance(relation_type, str) and "," in relation_type:
            target_relations = [r.strip() for r in relation_type.split(",")]
        else:
            target_relations = [relation_type]
        if [r for r in target_relations if r not in valid_relations]:
            target_relations = ["contains"]
        directional = [r for r in target_relations if r in self.DIRECTION_REVERSE_MAP]
        reverse_relations = [self.DIRECTION_REVERSE_MAP[r] for r in directional]
        all_children = []
        seen: set[str] = set()
        for parent in parent_nodes:
            for s, t, rels in self.graph.out_edges(parent.id):
                if set(rels) & set(target_relations) and t not in seen:
                    seen.add(t)
                    all_children.append(self.graph.nodes[t])
            if bidirectional and reverse_relations:
                for s, t, rels in self.graph.in_edges(parent.id):
                    if set(rels) & set(reverse_relations) and s not in seen:
                        seen.add(s)
                        all_children.append(self.graph.nodes[s])
        return all_children

    def filter_by_class(self, obj_class: str, candidates):
        return [n for n in candidates if isinstance(n, ObjectNode) and n.obj_class == obj_class]

    def filter_by_attribute(self, key: str, value: str, candidates, threshold=0.7):
        results = []
        for n in candidates:
            if isinstance(n, ObjectNode) and hasattr(n, key):
                attr_value = getattr(n, key)
                if str(attr_value).lower() == str(value).lower():
                    results.append((n, 1.0))
                else:
                    sim = SequenceMatcher(None, str(attr_value).lower(), str(value).lower()).ratio()
                    if sim >= threshold:
                        results.append((n, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in results]

    def _geo_nodes(self) -> list[KnowledgeNode]:
        return [n for n in self.graph.nodes.values() if isinstance(n, GeoNode)]

    def robust_subgraph_query(self, operation_chain: list, fallback=True, min_results=1) -> list:
        """The released relaxation ladder, step for step."""
        if not operation_chain:
            return []
        current_nodes = None
        all_results: list = []
        fallback_modes: list[bool] = []
        for i, op in enumerate(operation_chain):
            method_name = op["method"]
            try:
                if not hasattr(self, method_name):
                    fallback_modes.append(True)
                    continue
                func = getattr(self, method_name)
                fallback_step = False
                if method_name == "get_child_nodes":
                    if not current_nodes:
                        if not fallback:
                            break
                        current_nodes = self._geo_nodes()
                        fallback_step = True
                    if not fallback_step:
                        # released: only the FIRST current node is the parent.
                        temp = func(current_nodes[0], **op.get("kwargs", {}), candidates=current_nodes)
                        if not temp or len(temp) < min_results:
                            if fallback and i > 0:
                                fallback_step = True
                            else:
                                current_nodes = temp
                        else:
                            current_nodes = temp
                elif method_name == "get_geonode_by_name":
                    args = op.get("args", [])
                    if not args or (len(args) == 1 and not args[0]):
                        current_nodes = self._geo_nodes()
                    else:
                        temp = func(*args, candidates=current_nodes)
                        if not temp or len(temp) < min_results:
                            if fallback:
                                current_nodes = self._geo_nodes()
                                fallback_step = True
                            else:
                                current_nodes = temp
                        else:
                            current_nodes = temp
                elif method_name in ("filter_by_class", "filter_by_attribute"):
                    if current_nodes is None:
                        current_nodes = list(self.graph.nodes.values())
                    temp = func(*op.get("args", []), **op.get("kwargs", {}), candidates=current_nodes)
                    if not temp or len(temp) < min_results:
                        if fallback:
                            fallback_step = True
                        else:
                            current_nodes = temp
                    else:
                        current_nodes = temp
                else:
                    temp = func(*op.get("args", []), **op.get("kwargs", {}), candidates=current_nodes)
                    if not temp or len(temp) < min_results:
                        if fallback and i > 0:
                            fallback_step = True
                        else:
                            current_nodes = temp
                    else:
                        current_nodes = temp
                all_results.append(current_nodes)
                fallback_modes.append(fallback_step)
            except Exception:  # noqa: BLE001 -- released: an error keeps the previous step's result
                if not fallback:
                    break
                if i > 0 and all_results:
                    fallback_step = True
                else:
                    current_nodes = list(self.graph.nodes.values())
                    fallback_step = True
                fallback_modes.append(fallback_step)
                all_results.append(current_nodes)
        if (not current_nodes or len(current_nodes) < min_results) and fallback and all_results:
            for k in range(len(all_results) - 1, -1, -1):
                if not fallback_modes[k] and all_results[k] and len(all_results[k]) >= min_results:
                    current_nodes = all_results[k]
                    break
        return current_nodes or []

    def is_within_geo_node(self, position) -> str:
        point = Point(position[0], position[1])
        names = [n.name for n in self._geo_nodes() if n.contour_polygon.contains(point)]
        if names:
            return f" in {', '.join(names)}"
        return " not in any landmarks. "

    def get_enhanced_geo_relation(self, position) -> str:
        point = Point(position[0], position[1])
        descriptions = []
        for node in self._geo_nodes():
            polygon = node.contour_polygon
            # substitution: released measures to the object's centre; a street's
            # area is what "you are N m from it" means here (see policy.py).
            distance = float(polygon.distance(point))
            direction = get_direction(position, node.position)
            if polygon.contains(point):
                c = polygon.centroid
                intra = get_direction((c.x, c.y), position)
                descriptions.append(f"You are inside {node.name} ({intra} area)")
                continue
            closest, min_corner = None, float("inf")
            for coord in polygon.exterior.coords[:-1]:
                d = math.hypot(position[0] - coord[0], position[1] - coord[1])
                if d < min_corner:
                    min_corner, closest = d, (coord[0], coord[1])
            if min_corner < 5.0:
                descriptions.append(f"near {node.name}'s {get_direction(closest, position)} corner ({min_corner:.1f}m)")
                continue
            descriptions.append(f"{node.name} is {distance:.1f}m {direction} of you")
        return " | ".join(sorted(descriptions, key=len))


# scenegraphnav/agent.py::recursive_query, as released.
_CHAIN_RELATIONS = [
    "contains", "adjacent_to", "near_corner",
    "north_of", "south_of", "east_of", "west_of",
    "northeast_of", "northwest_of", "southeast_of", "southwest_of",
]


def recursive_query(engine: QueryEngine, operation_chain):
    if not operation_chain:
        return []
    try:
        for op in operation_chain:
            if not isinstance(op, dict):
                continue
            # released: this coercion speaks adjacent_to/near_corner, the
            # engine's validator speaks overlaps/separates; both are kept.
            if op.get("method") == "get_child_nodes" and "relation_type" in op.get("kwargs", {}):
                if op["kwargs"]["relation_type"] not in _CHAIN_RELATIONS:
                    op["kwargs"]["relation_type"] = "contains"
        current_nodes = engine.robust_subgraph_query(operation_chain, fallback=True, min_results=1)
        return current_nodes or None
    except Exception:  # noqa: BLE001
        return None
