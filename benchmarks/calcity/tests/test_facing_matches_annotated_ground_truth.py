"""The served orientation, checked against hand-annotated facing.

A facing clause is the strongest single predictor of failure inside the
conjunction family: on the 93-episode subset it reads 8/52 with a clause
against 15/41 without (Fisher p = 0.029), and the gap survives inside every
candidate-density bucket. That makes "is the orientation math right?" the
first question, because serving a wrong bearing more loudly would make the
family worse, not better.

It is right, and this pins it. Twenty-five Cal City episodes state a house's
facing in words AND carry the house's own footprint polygon, so each one is a
labelled instance: the direction a house faces is the direction from the house
to the road it fronts. The check runs the SHIPPED chain end to end --
``mask_geom`` -> ``gap_witness`` -> ``bearing``, then the surface's own
``+180`` rule that turns "facing <road>" into "which way this faces"
-- and compares the 8-point compass word to the annotator's.

The two episodes that disagree are corner lots whose nearest road is not the
road they front (``278#276`` says "at the corner of 88th Street" and fronts
Hickory Drive by 15.0 m; ``572#570`` fronts Margery Avenue at 19.8 m while
Ethel Street sits 16.3 m off its side). That is this test's nearest-road
heuristic, not the geometry: both resolve when the fronting road is named, so
the floor is 22/25 rather than 25/25.

Vacuity floor: the derivation is a scan over the corpus, and a scan that finds
nothing passes. ``test_the_ground_truth_population_is_still_there`` asserts the
case count, and two episodes are pinned by name with their expected words, so a
derivation that silently stops matching cannot read as agreement.
"""

import collections
import json
import os
import re

import pytest

from benchmarks.calcity.paths import calcity_bench
from maps.query import spatial_math as G

# The bench as this process's environment places it.
_BENCH = calcity_bench(os.environ)
_PRIOR = _BENCH.tiles_dir / "calcity_prior_layer.json"
_META = _BENCH.tiles_dir / "calcity_meta.json"
_SPLIT = _BENCH.split_path("calcity")

_ROAD_TYPES = {"residential", "tertiary", "living_street", "service", "track"}
_COMPASS = "north|south|east|west|northeast|northwest|southeast|southwest"
_FACES = re.compile(rf"\b({_COMPASS})[-\s]?facing\b", re.I)
_FACING = re.compile(rf"\bfacing\s+({_COMPASS})\b", re.I)
# The TARGET is the house itself, so ``target_mask`` is the footprint whose
# orientation the words describe. A pool-in-the-backyard row states the
# house's facing too, but annotates the pool.
_IS_HOUSE = re.compile(
    r"^\s*(a\s+|the\s+)?(small\s+|large\s+|little\s+|red[- ]roofed\s+|white\s+)*"
    r"(house|home|building)\b",
    re.I,
)

_needs_bake = pytest.mark.skipif(
    not _PRIOR.is_file() or not _SPLIT.is_file(),
    reason="the Cal City bake / dataset is not present in this tree",
)


def _road_chains():
    """``{name: [chain, ...]}`` for every named road in the prior layer."""
    chains = collections.defaultdict(list)
    for f in json.loads(_PRIOR.read_text())["layer"]["features"]:
        if f.get("name") and f.get("object_type") in _ROAD_TYPES:
            chains[f["name"]].extend(f.get("chains_world") or [])
    return chains


def _road_geoms():
    """``{name: (mask_geom, bbox)}`` for every named road in the prior layer."""
    chains = collections.defaultdict(list)
    for f in json.loads(_PRIOR.read_text())["layer"]["features"]:
        if f.get("name") and f.get("object_type") in _ROAD_TYPES:
            chains[f["name"]].extend(f.get("chains_world") or [])
    out = {}
    for name, chs in chains.items():
        if chs:
            g = G.mask_geom(chs)
            out[name] = (g, g["BB"])
    return out


def _cases():
    """``(episode_id, stated_word, footprint)`` per labelled episode."""
    found = []
    for i, row in enumerate(json.loads(_SPLIT.read_text())):
        desc = (row.get("descriptions") or [""])[0]
        m = _FACES.search(desc) or _FACING.search(desc)
        if not m or not _IS_HOUSE.search(desc):
            continue
        poly = row.get("target_mask")
        if poly and len(poly) >= 3:
            ann = (row.get("ann_ids") or [0])[0]
            found.append((f"calcity::{ann}#{i}", m.group(1).lower(), poly))
    return found


def _faces(footprint, roads):
    """The shipped chain: nearest road, its facing, the +180 rule."""
    house = G.mask_geom([list(footprint) + [footprint[0]]])
    hbox = house["BB"]
    best = (float("inf"), None, None)
    for name, (rg, rbox) in roads.items():
        if G.bbox_gap(hbox, rbox) >= best[0]:
            continue
        gap, on_house, on_road = G.gap_witness(house, rg)
        if gap < best[0]:
            best = (gap, name, G.bearing(on_road, on_house))
    gap, name, edge_bearing = best
    return name, gap, G._compass8((edge_bearing + 180.0) % 360.0)


@_needs_bake
def test_the_ground_truth_population_is_still_there():
    """The vacuity floor: a scan that matches nothing would agree perfectly."""
    cases = _cases()
    assert len(cases) >= 20, f"only {len(cases)} labelled facing episodes found"
    assert {e for e, _, _ in cases} >= {"calcity::110#109", "calcity::195#193", "calcity::567#565"}


@_needs_bake
def test_the_world_frame_is_east_north():
    """Every bearing rests on this. ``bearing`` is ``atan2(dx, dy)``, which is
    compass only while +y is north -- a bake that flipped y would leave the
    function unchanged and every served direction mirrored."""
    frame = json.loads(_META.read_text())["coordinate_frame"]
    assert frame["y_axis"] == "north"
    assert G.bearing([0, 0], [0, 1]) == 0.0  # +y reads north
    assert G.bearing([0, 0], [1, 0]) == 90.0  # +x reads east


@_needs_bake
def test_the_served_facing_matches_the_annotators():
    roads = _road_geoms()
    cases = _cases()
    wrong = []
    for episode, stated, footprint in cases:
        road, gap, got = _faces(footprint, roads)
        if got != stated:
            wrong.append((episode, stated, got, road, round(gap, 1)))
    assert len(cases) - len(wrong) >= 22, (
        f"{len(cases) - len(wrong)}/{len(cases)} agree; disagreements: {wrong}"
    )


@_needs_bake
@pytest.mark.parametrize(
    "episode,expected",
    [
        ("calcity::110#109", "west"),
        ("calcity::195#193", "northwest"),
        ("calcity::567#565", "north"),
    ],
)
def test_pinned_episodes_face_the_way_they_are_labelled(episode, expected):
    """Named instances, because a ratio floor cannot notice a derivation that
    quietly stops producing the hard cases (``195#193`` is the one diagonal)."""
    roads = _road_geoms()
    footprint = next(f for e, _, f in _cases() if e == episode)
    _road, _gap, got = _faces(footprint, roads)
    assert got == expected


@_needs_bake
def test_the_served_facing_column_matches_the_annotators():
    """The SURFACE half of the same fact, through the shipped compare.

    The checks above pin the measurement. This one runs ``run_program``'s
    ``compare`` on real Cal City geometry and reads the column the model
    actually receives -- ``facing <road>`` -- asserting the compass WORD is
    the annotator's.
    """
    from maps.query.program import run_program

    roads = _road_geoms()
    chains = _road_chains()
    checked = 0
    for episode, stated, footprint in _cases():
        road, _gap, got = _faces(footprint, roads)
        if got != stated:  # a corner lot; another pin owns it
            continue
        house = {
            "fid": "detected_1",
            "name": "h",
            "class": "house",
            "xy": [
                sum(p[0] for p in footprint) / len(footprint),
                sum(p[1] for p in footprint) / len(footprint),
            ],
            "mask": [list(footprint) + [footprint[0]]],
            "tags": {},
            "search_text": "house",
        }
        ref = {
            "fid": "prior_1",
            "name": road,
            "xy": chains[road][0][0],
            "mask": chains[road],
            "tags": {"highway": "residential"},
            "search_text": "",
        }
        out = run_program(
            [house],
            [ref],
            {
                "ops": [
                    {"ref": "r1", "op": "find", "args": {"query": "house"}},
                    {
                        "ref": "r2",
                        "op": "compare",
                        "args": {"items": "r1", "refs": [{"ref": ["prior_1"], "as": road}]},
                    },
                ]
            },
        )
        row = out["results"]["r2"][0]
        key = next(k for k in row if k.startswith("facing "))
        assert row[key] == stated, (episode, road, row[key], stated)
        checked += 1
        if checked >= 8:
            break
    # Vacuity floor: a loop that checked nothing would pass.
    assert checked >= 8, checked
