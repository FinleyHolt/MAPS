"""Tags-only world model: the retrieval surface speaks raw OSM tags."""
from __future__ import annotations


from maps.search.analyze_scene import text_match as T
from maps.search.analyze_scene.geom_program import (
    build_surface,
    compact_results,
    run_program,
)


class _Node:
    """Minimal MapObject-shaped prior/detection node."""

    def __init__(self, *, name=None, class_name="", tags=None,
                 centroid=(0, 0), provenance="prebaked"):
        self.name = name
        self.class_name = class_name
        self.tags = tags or {}
        self.centroid = centroid
        self.centerline = None
        self.provenance = provenance
        self.cells = None
        self.world_contour = None


def _prior_surface():
    priors = [
        ("prior_0", _Node(tags={"amenity": "parking"}, centroid=(0, 0))),
        ("prior_1", _Node(tags={"service": "parking_aisle"}, centroid=(0, 5))),
        ("prior_2", _Node(tags={"parking": "surface"}, centroid=(0, 10))),
        ("prior_3", _Node(name="Elm Street",
                          tags={"highway": "residential"}, centroid=(0, 15))),
    ]
    dets = [("detected_0", _Node(class_name="car", centroid=(5, 5),
                                 provenance="detected"))]
    return build_surface(dets, priors, (0.0, 0.0), 2.0)


def test_prior_rows_carry_tags_and_no_class():
    cands, anchors = _prior_surface()
    assert all("class" not in a for a in anchors), anchors
    assert anchors[0]["tags"] == {"amenity": "parking"}, anchors[0]
    # detections keep the detector-class measurement
    assert cands[0]["class"] == "car"


def test_query_over_tags_reaches_key_and_value_matches():
    # ONE idiom: 'parking' reaches amenity=parking (value),
    # service=parking_aisle (value token) AND parking=surface (KEY).
    cands, anchors = _prior_surface()
    out = run_program(cands, anchors, {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"query": "parking"}},
    ]})
    rows = {r["fid"]: r for r in out["results"]["r0"]}
    for fid in ("prior_0", "prior_1", "prior_2"):
        assert rows[fid]["name_match"] >= T.FUZZY_THRESHOLD, (fid, rows[fid])
    # the non-parking row is sub-threshold: NOT in the match set (a
    # query-less find browses it)
    assert "prior_3" not in rows


def test_broad_dump_census_renders_tag_counts():
    # 60 tagged priors (past the census knee): the summarize envelope carries
    # tag key=value counts, not a class taxonomy.
    priors = [(f"prior_{i}",
               _Node(tags={"building": "yes"} if i % 2 else
                     {"highway": "residential"}, centroid=(i, 0)))
              for i in range(60)]
    cands, anchors = build_surface([], priors, (0.0, 0.0), 2.0)
    out = run_program(cands, anchors, {"ops": [
        {"ref": "r0", "op": "find", "args": {}},
    ]})
    env = compact_results(out["results"])["r0"]
    assert env["len"] == 60
    assert "census" in env, env.keys()
    assert env["census"]["building=yes"] == 30, env["census"]
    assert env["census"]["highway=residential"] == 30, env["census"]
    assert "by_class" not in env


def test_no_match_query_ships_surface_census_with_name_samples():
    cands, anchors = _prior_surface()
    out = run_program(cands, anchors, {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"query": "zzzqx"}},
    ]})
    env = compact_results(out["results"])["r0"]
    assert env["len"] == 0
    hint = env.get("hint") or ""
    assert "surface census" in hint, hint
    assert "amenity=parking(1)" in hint, hint
    assert "Elm Street" in hint, hint


def test_grid_renders_prior_tags_compact_kv():
    cands, anchors = _prior_surface()
    out = run_program(cands, anchors, {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"query": "parking"}},
    ]})
    env = compact_results(out["results"])["r0"]
    grid = str(env.get("legend")) + "\n" + str(env.get("rows"))
    assert "tags" in grid, grid
    assert "amenity=parking" in grid, grid
    assert "service=parking_aisle" in grid, grid
