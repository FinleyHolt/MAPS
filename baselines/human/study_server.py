"""The capacity study's operator console, served from the study download alone (``STUDY.md``).

One participant, one condition, one browser tab: no reasoner, no segmenter, no network. The
session is :class:`baselines.human.capacity.Capacity` over the plan's list; the map is the
bench's ortho as tiles (``study_package tiles``), the briefed AO and the bench's OSM prior;
a verified report reads MAPS's own records from the download's MAPS root. Nothing served
carries an episode's target or a score.

    python -m baselines.human.study_server p01 practice|first|second [--kit ~/maps-study] [--port 8788]

``first`` and ``second`` are the participant's two conditions in the order the plan
counterbalanced (``plan/plan.json``); ``solo`` and ``team`` name one directly.

The answers land in ``<kit>/results/<participant>/<condition>/`` (``answers.jsonl`` and the
session's clock, ``session.json``). Starting a condition again resumes its clock; it never
starts a new session. Practice starts empty each time.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("study_server")

#: The download's folder, unless ``--kit`` names another.
KIT = Path(os.environ.get("MAPS_STUDY_KIT", Path.home() / "maps-study"))
#: The OSM prior's outlines are drawn simplified to this, metres: a pixel at the closest zoom.
PRIOR_SIMPLIFY_M = 0.3
#: The map's zoom-out floor: the whole block in a laptop's window.
MIN_ZOOM = 12
_SHA = re.compile(r"^[0-9a-f]{64}$")


def use_kit(kit: Path) -> None:
    """Point the bench at the download, before anything imports its paths."""
    os.environ.setdefault("SEARCH_CALCITY_DATA", str(kit))
    os.environ.setdefault("SEARCH_CALCITY_TILES", str(kit / "tiles"))
    import benchmarks.calcity

    benchmarks.calcity.use()


def _kind(row: dict) -> str:
    """What an OSM row is, in a word or two: its layer, and its class where it has one (a road's
    highway, a POI's amenity, a landuse's own tag: farmland, meadow...)."""
    layer = str(row.get("worldkit_layer") or row.get("category") or "").rstrip("s")
    building = row.get("building") if row.get("building") not in (None, "yes") else None
    sub = (row.get("highway") or row.get("amenity") or row.get("shop") or row.get("leisure") or row.get("aeroway")
           or row.get("landuse") or row.get("natural") or building)
    return f"{layer} · {sub}" if sub and sub != layer else layer


def prior_doc(map_name: str, ll) -> dict[str, Any]:
    """The bench's OSM prior in WGS84: every row's outline (a road its centreline), and the
    named places, for the toggle and the search."""
    from baselines.seam import _row_area, block_landmarks
    from benchmarks.citynav.run_benchmark.assets import load_episode_assets

    features = load_episode_assets(map_name).base_features or {}
    shapes = []
    for category, rs in features.items():
        for r in rs if isinstance(rs, list) else []:
            if not isinstance(r, dict):
                continue
            pts = [(float(p[0]), float(p[1])) for p in r.get("world_contour") or [] if len(p) >= 2]
            if (category == "roads" or r.get("highway")) and len(pts) >= 2:
                line = pts
            else:
                geom = _row_area({**r, "category": category})
                if geom is None or not hasattr(geom, "exterior"):
                    continue
                line = list(geom.simplify(PRIOR_SIMPLIFY_M).exterior.coords)
            shapes.append({"type": "Feature", "properties": {"name": str(r.get("name") or ""), "kind": _kind(r)},
                           "geometry": {"type": "LineString", "coordinates": ll(line)}})
    names = sorted(({"name": lm.name, "kind": _kind(lm.tags), "lonlat": ll([lm.centroid])[0]}
                    for lm in block_landmarks(features)), key=lambda d: d["name"])
    return {"type": "FeatureCollection", "features": shapes, "names": names}


def review_doc(cap: Any, ll) -> dict[str, Any]:
    """A verified report's review with every point and outline in WGS84."""
    session = cap.review
    state = session.state()
    r = session.review
    if r is None:
        return state
    by_id = {c.id: c for c in r.candidates}
    for c in state["candidates"]:
        cand = by_id[c["id"]]
        c["lonlat"] = ll([cand.xy])[0]
        c["ring"] = ll(cand.contour) if len(cand.contour) >= 3 else None
    sel = state.get("selected")
    if sel:
        sel["lonlat"] = ll([sel["xy"]])[0]
        del sel["xy"]
    state.pop("landmarks", None)
    state["maps_lonlat"] = ll([r.maps_xy])[0]
    state["maps_prior_ring"] = ll(r.maps_prior[0]) if r.maps_prior else None
    state["stops"] = ll(r.stops)
    return state


def build_app(cap: Any, kit: Path, console: Path):
    """The console's routes over one capacity session. One request at a time: a session
    is a ledger, and a double click must not answer twice."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, Response
    from fastapi.staticfiles import StaticFiles

    from benchmarks.calcity.data_prep.aoi import AO_BBOX_WGS84
    from benchmarks.flight.tile_georef import local_to_lonlat, lonlat_to_local

    map_name = "calcity"
    tiles_dir = kit / "tiles"
    lock, tile_lock = threading.Lock(), threading.Lock()
    app = FastAPI()

    def ll(pts) -> list[list[float]]:
        return [list(local_to_lonlat(tiles_dir, map_name, float(x), float(y))) for x, y in pts]

    def doc() -> dict[str, Any]:
        state = cap.state()
        cur = state.get("current")
        if cur:
            if cur.get("review") is not None:
                cur["review"] = review_doc(cap, ll)
            if cur.get("point"):
                cur["point"] = ll([cur["point"]])[0]
        return state

    built: dict[str, Any] = {}

    def build_prior() -> None:
        try:
            built["prior"] = prior_doc(map_name, ll)
        except Exception:
            # The route answers "building" for what never built; the log says why.
            log.exception("the OSM prior for %s did not build", map_name)

    threading.Thread(target=build_prior, daemon=True).start()
    ortho = sqlite3.connect(f"file:{kit / 'calcity_ortho.mbtiles'}?mode=ro", uri=True, check_same_thread=False)

    @app.get("/capacity/state")
    def capacity_state() -> dict[str, Any]:
        with lock:
            return doc()

    @app.post("/capacity/act")
    def capacity_act(req: dict[str, Any]) -> dict[str, Any]:
        with lock:
            if req.get("action") == "select" and "lon" in req:
                xy = lonlat_to_local(tiles_dir, map_name, float(req["lon"]), float(req["lat"]))
                if xy is None:
                    raise HTTPException(400, "that point is off the map")
                req = {"action": "select", "x": xy[0], "y": xy[1]}
            out = cap.act(req)
            return {**out, "state": doc()}

    @app.get("/judge/frames/{sha}.jpg")
    def frame(sha: str) -> Response:
        review = cap.review.review if cap.review is not None else None
        path = review.attempt / "frames" / f"{sha}.jpg" if review is not None and _SHA.match(sha) else None
        if path is None or not path.is_file():
            raise HTTPException(404)
        return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    @app.get("/study/prior.json")
    def study_prior() -> dict[str, Any]:
        # Still building is an answer, not a fault: the console asks again.
        return built.get("prior") or {"building": True}

    @app.get("/study/tiles/{z}/{x}/{y}.webp")
    def tile(z: int, x: int, y: int) -> Response:
        with tile_lock:
            row = ortho.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                                (z, x, 2**z - 1 - y)).fetchone()
        # No tile is ground the raster holds no imagery for: the console's paper tone.
        if row is None:
            return Response(status_code=204)
        return Response(row[0], media_type="image/webp", headers={"Cache-Control": "max-age=86400"})

    @app.get("/area")
    def area() -> dict[str, Any]:
        return {"map": map_name, "bbox": list(AO_BBOX_WGS84)}

    @app.get("/location")
    def location() -> dict[str, Any]:
        w, s, e, n = AO_BBOX_WGS84
        return {"center": {"lon": (w + e) / 2, "lat": (s + n) / 2}, "minzoom": MIN_ZOOM}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(console / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/", StaticFiles(directory=console), name="console")
    return app


def open_session(kit: Path, participant: str, condition: str, out: Path | None = None) -> Any:
    """The participant's capacity session in one condition, off the download's plan."""
    from baselines.human.audit_console import bench_labels
    from baselines.human.capacity import Capacity

    if condition == "practice":
        root, order_file, arm = kit / "plan" / "practice", kit / "plan" / "practice" / "practice.txt", "solo"
    else:
        root, order_file, arm = kit / "maps_root", kit / "plan" / participant / f"{condition}.txt", condition
    if not order_file.is_file():
        raise SystemExit(f"no list for {participant} {condition} at {order_file}: is {participant} on the plan?")
    out = out or kit / "results" / participant / condition
    if condition == "practice":
        shutil.rmtree(out, ignore_errors=True)
    elif (out / "session.json").is_file():
        print(f"{participant} {condition} has a session already: resuming its clock")
    order = [ln.strip() for ln in order_file.read_text().splitlines() if ln.strip()]
    return Capacity(root, order, out, arm, labels=bench_labels())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("participant", help="the participant's label, p01 and on, never a name")
    ap.add_argument("condition", choices=("practice", "first", "second", "solo", "team"))
    ap.add_argument("--kit", type=Path, default=KIT, help="the study download's folder (default %(default)s)")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"p\d{2,}", args.participant):
        ap.error("a participant is a label like p01, never a name")
    kit = args.kit.expanduser()
    console = kit / "console"
    if not (console / "index.html").is_file():
        raise SystemExit(f"no console at {console}: is {kit} the study download?")
    condition = args.condition
    if condition in ("first", "second"):
        plan = json.loads((kit / "plan" / "plan.json").read_text())["participants"].get(args.participant)
        if plan is None:
            raise SystemExit(f"{args.participant} is not on the plan: ask the study lead for your label")
        condition = plan["first"] if condition == "first" else ({"solo", "team"} - {plan["first"]}).pop()
    use_kit(kit)
    cap = open_session(kit, args.participant, condition)
    import uvicorn

    print(f"{args.participant} {condition}: open http://{args.host}:{args.port}/ in a browser. Ctrl-C stops the console.")
    uvicorn.run(build_app(cap, kit, console), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
