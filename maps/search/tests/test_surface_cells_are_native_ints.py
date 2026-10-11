"""``cells`` on a surface row is a plain nested list of native ints."""

from __future__ import annotations

import json

from maps.search.analyze_scene import cell_mesh
from maps.search.analyze_scene.geom_program import build_surface_from_graph


class _Node:
    def __init__(self, cells, provenance="prebaked"):
        import numpy as np

        self.cells = np.asarray(cells, dtype=np.int32)
        self.world_xy = [float(cells[0][1]) * 2.0, float(cells[0][0]) * 2.0]
        self.centroid = (int(cells[0][0]), int(cells[0][1]))
        self.centerline = None
        self.class_name = "building"
        self.name = None
        self.provenance = provenance


def _graph():
    return {
        f"building_{i}": _Node([[r, c] for r in range(i, i + 4)
                                for c in range(i, i + 5)])
        for i in range(1, 12)
    }


def _rows():
    nodes = _graph()
    cands, anchors = build_surface_from_graph(nodes.items(), (0.0, 0.0), 2.0)
    return nodes, list(cands) + list(anchors)


def test_cells_match_the_nodes_they_came_from():
    nodes, rows = _rows()
    by_fid = {r["fid"]: r for r in rows if r.get("fid")}
    # vacuity floor: a scan that found nothing would pass every assert below
    assert len(by_fid) == len(nodes) == 11, (len(by_fid), len(nodes))
    checked = 0
    for fid, node in nodes.items():
        got = by_fid[fid].get("cells")
        assert got == cell_mesh.cells_of(node).tolist(), fid
        assert got == [[int(r), int(k)] for r, k in cell_mesh.cells_of(node)], fid
        assert len(got) == 20, (fid, len(got))
        checked += 1
    assert checked == 11, checked


def test_cells_are_json_native_ints_not_numpy_scalars():
    _nodes, rows = _rows()
    cells = [c for r in rows for c in (r.get("cells") or [])]
    assert len(cells) == 11 * 20, len(cells)
    for pair in cells:
        assert type(pair) is list, type(pair)
        for v in pair:
            assert type(v) is int, type(v)
    json.dumps(cells)  # raises on any numpy scalar that slipped through


def test_a_node_with_no_cells_stays_bare():
    import numpy as np

    node = _Node([[1, 1]])
    node.cells = np.zeros((0, 2), dtype=np.int32)
    cands, anchors = build_surface_from_graph(
        {"building_1": node}.items(), (0.0, 0.0), 2.0)
    (row,) = list(cands) + list(anchors)
    assert not row.get("cells"), row.get("cells")
