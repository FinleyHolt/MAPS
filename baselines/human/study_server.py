"""The capacity study's operator console, served from the study download alone (``STUDY.md``).

One participant, one condition, one browser tab: no reasoner, no segmenter, no network. The
session is :class:`baselines.human.capacity.Capacity` over the plan's list; the map is the
bench's ortho as tiles (``study_package tiles``), the briefed AO and the bench's OSM prior;
a verified report reads MAPS's own records from the download's MAPS root. Nothing served
carries an episode's target or a score.

    python -m baselines.human.study_server [p01 [practice|first|second|solo|team]] [--kit ~/maps-study]

With a label alone (or none, and the page asks for it), the browser opens the visit page,
which walks the briefing, practice, the first condition, a break and the second, and offers the
results file at the end. ``first`` and ``second`` are the participant's two conditions in the
order the plan counterbalanced (``plan/plan.json``); naming a condition opens it at once.

The answers land in ``<kit>/results/<participant>/<condition>/`` (``answers.jsonl`` and the
session's clock, ``session.json``). Starting a condition again resumes its clock; it never
starts a new session. Practice starts empty each time.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import logging
import os
import re
import shutil
import sqlite3
import tarfile
import threading
import time
import webbrowser
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


def build_app(cap: Any, kit: Path, console: Path, visit: Visit | None = None):
    """The console's routes over one capacity session. One request at a time: a session
    is a ledger, and a double click must not answer twice. With a ``visit``, the visit page
    opens each session in turn, and the console's session is whichever it opened last."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
    from fastapi.staticfiles import StaticFiles

    from benchmarks.calcity.data_prep.aoi import AO_BBOX_WGS84
    from benchmarks.flight.tile_georef import local_to_lonlat, lonlat_to_local

    map_name = "calcity"
    tiles_dir = kit / "tiles"
    lock, tile_lock = threading.Lock(), threading.Lock()
    app = FastAPI()

    def ll(pts) -> list[list[float]]:
        return [list(local_to_lonlat(tiles_dir, map_name, float(x), float(y))) for x, y in pts]

    box = {"cap": cap, "practice": False}

    def doc() -> dict[str, Any]:
        cap = box["cap"]
        state = cap.state()
        state["practice"] = box["practice"]
        cur = state.get("current")
        if cur:
            if cur.get("review") is not None:
                cur["review"] = review_doc(box["cap"], ll)
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
            out = box["cap"].act(req)
            return {**out, "state": doc()}

    @app.get("/judge/frames/{sha}.jpg")
    def frame(sha: str) -> Response:
        cap = box["cap"]
        review = cap.review.review if cap is not None and cap.review is not None else None
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

    if visit is not None:
        @app.get("/visit")
        def visit_page() -> HTMLResponse:
            return HTMLResponse(visit.page(), headers={"Cache-Control": "no-store"})

        @app.get("/visit/label")
        def visit_label(label: str = "") -> Response:
            visit.set_label(label.strip().lower())
            return RedirectResponse("/visit", status_code=303)

        @app.post("/visit/start/{step}")
        def visit_start(step: str) -> Response:
            with lock:
                cap = visit.start(step)
                if cap is None:
                    return RedirectResponse("/visit", status_code=303)
                box["cap"], box["practice"] = cap, step == "practice"
            return RedirectResponse("/", status_code=303)

        @app.get("/visit/results.tgz")
        def visit_results() -> Response:
            return Response(visit.results_tgz(), media_type="application/gzip", headers={
                "Content-Disposition": f'attachment; filename="maps-study-{visit.participant}.tgz"'})

    @app.get("/")
    def index() -> Response:
        if box["cap"] is None and visit is not None:
            return RedirectResponse("/visit")
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


BRIEFING = Path(__file__).with_name("briefing.md")
STEPS = ("practice", "first", "second")


def md_html(text: str) -> str:
    """The briefing's Markdown as HTML: headings, bullets (one level of nesting), bold, code, paragraphs."""
    inline = lambda t: re.sub(r"`(.+?)`", r"<code>\1</code>", re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(t)))
    out, para, depth = [], [], 0

    def flush() -> None:
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    for line in text.splitlines() + [""]:
        bullet = re.match(r"( *)- (.*)", line)
        want = (len(bullet.group(1)) // 2 + 1) if bullet else (depth if line.startswith("  ") and depth else 0)
        if not bullet and depth and line.startswith("  "):
            out[-1] = out[-1][:-5] + " " + inline(line.strip()) + "</li>"
            continue
        flush() if bullet or not line.strip() or line.startswith("#") else None
        while depth < want:
            out.append("<ul>")
            depth += 1
        while depth > want:
            out.append("</ul>")
            depth -= 1
        if bullet:
            out.append(f"<li>{inline(bullet.group(2))}</li>")
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level + 1}>{inline(line.lstrip('#').strip())}</h{level + 1}>")
        elif line.strip():
            para.append(line.strip())
    return "\n".join(out)


class Visit:
    """One participant's visit, as the visit page walks it: the briefing, practice, the first
    condition, a break, the second, done. Where they are is read off the results on disk, so a
    restarted console picks up where the visit left off."""

    def __init__(self, kit: Path, participant: str | None) -> None:
        self.kit = kit
        self.plan = json.loads((kit / "plan" / "plan.json").read_text())["participants"]
        self.participant, self.error = None, ""
        if participant:
            self.set_label(participant)
            if self.error:
                raise SystemExit(self.error)

    def set_label(self, label: str) -> None:
        if label in self.plan:
            self.participant, self.error = label, ""
        else:
            self.error = f"{label or 'That'} is not a participant label on this study. Check it with the study lead."

    def condition(self, step: str) -> str:
        first = self.plan[self.participant]["first"]
        return {"practice": "practice", "first": first, "second": ({"solo", "team"} - {first}).pop()}[step]

    def status(self, step: str) -> str:
        """``todo``, ``running`` or ``done``."""
        out = self.kit / "results" / self.participant / self.condition(step)
        if step == "practice":
            return "done" if out.is_dir() else "todo"
        meta = out / "session.json"
        if not meta.is_file():
            return "todo"
        session = json.loads(meta.read_text())
        answered = sum(1 for _ in (out / "answers.jsonl").open()) if (out / "answers.jsonl").is_file() else 0
        over = time.time() - float(session["started_epoch"]) >= float(session["session_s"]) or answered >= session["episodes_n"]
        return "done" if over else "running"

    def start(self, step: str) -> Any:
        """The step's session, or None when it cannot open now: no label yet, the second
        before the first is done, or a timed session that is over."""
        if self.participant is None or step not in STEPS or (step == "second" and self.status("first") != "done"):
            return None
        if step != "practice" and self.status(step) == "done":
            return None
        return open_session(self.kit, self.participant, self.condition(step))

    def results_tgz(self) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(self.kit / "results" / self.participant, arcname=f"results/{self.participant}")
        return buf.getvalue()

    def page(self) -> str:
        if self.participant is None:
            body = (f"<h1>The search study</h1><p class=err>{html.escape(self.error)}</p>"
                    "<form action='/visit/label'><p>Your participant label, from the study lead:</p>"
                    "<input name=label placeholder='p01' autofocus> <button>Continue</button></form>")
            return PAGE.format(body=body)
        st = {s: self.status(s) for s in STEPS}

        def button(step: str, label: str, note: str = "") -> str:
            return (f"<form method=post action='/visit/start/{step}'><button>{label}</button></form>"
                    + (f"<p class=note>{note}</p>" if note else ""))

        clock = "The 40 minutes start when you press the button, so be seated and ready."
        if st["second"] == "done":
            now = ("<h2>Done. Thank you!</h2><p>Your answers are saved. Let the study lead know you have finished.</p>"
                   "<p class=note>Taking part from your own computer? <a href='/visit/results.tgz'>Save your results "
                   "file</a> and email it to the study lead.</p>")
        elif st["second"] == "running":
            now = "<h2>Session 2 is under way</h2>" + button("second", "Return to session 2")
        elif st["first"] == "done":
            now = ("<h2>Session 1 is done. Take a five-minute break.</h2>"
                   "<p>Then start session 2. Read the console's top bar: it says whether you work alone or with MAPS.</p>"
                   + button("second", "Start session 2", clock))
        elif st["first"] == "running":
            now = "<h2>Session 1 is under way</h2>" + button("first", "Return to session 1")
        else:
            practiced = st["practice"] == "done"
            now = ("<h2>Step 1: read the briefing below</h2><h2>Step 2: practice</h2>"
                   "<p>Four reports that never count. Try every tool the briefing lists, then press "
                   "<b>Done practicing</b> at the top of the console.</p>"
                   + button("practice", "Practice again" if practiced else "Start practice")
                   + "<h2>Step 3: session 1</h2>" + (button("first", "Start session 1", clock) if practiced
                                                      else "<p class=note>Opens after practice.</p>"))
        brief = md_html(BRIEFING.read_text()) if BRIEFING.is_file() else ""
        return PAGE.format(body=f"<p class=who>Participant {self.participant}</p>{now}<hr><div class=brief>{brief}</div>")


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8><title>The search study</title><style>
:root {{ --ink: #141310; --paper: #fbfaf5; --navy: #1f3a5f; --grey: #6b675d; }}
body {{ margin: 0 auto; max-width: 46rem; padding: 32px 16px 64px; font: 17px/1.5 system-ui, sans-serif;
       color: var(--ink); background: var(--paper); }}
h1 {{ font-size: 28px; }} h2 {{ font-size: 21px; margin-top: 28px; }} h3 {{ font-size: 18px; }}
button {{ font: inherit; font-weight: 700; padding: 10px 22px; color: #fff; background: var(--navy);
         border: 0; cursor: pointer; }}
input {{ font: inherit; padding: 8px; width: 8rem; }}
.note, .who {{ color: var(--grey); }} .err {{ color: #a3341f; }}
hr {{ margin: 36px 0; border: 0; border-top: 1px solid #d9d4c6; }}
</style></head><body>{body}</body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("participant", nargs="?", help="the participant's label, p01 and on, never a name; "
                                                   "without it the visit page asks for it")
    ap.add_argument("condition", nargs="?", choices=("practice", "first", "second", "solo", "team"),
                    help="open this session at once instead of at the visit page")
    ap.add_argument("--kit", type=Path, default=KIT, help="the study download's folder (default %(default)s)")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = ap.parse_args(argv)
    if args.participant and not re.fullmatch(r"p\d{2,}", args.participant):
        ap.error("a participant is a label like p01, never a name")
    kit = args.kit.expanduser()
    console = kit / "console"
    if not (console / "index.html").is_file():
        raise SystemExit(f"no console at {console}: is {kit} the study download?")
    use_kit(kit)
    visit = Visit(kit, args.participant)
    cap = None
    if args.condition:
        if not args.participant:
            ap.error("a condition needs a participant")
        condition = args.condition if args.condition in ("practice", "solo", "team") else visit.condition(args.condition)
        cap = open_session(kit, args.participant, condition)
    import uvicorn

    url = f"http://{args.host}:{args.port}/"
    print(f"The study is at {url} in a browser. Keep this window open; Ctrl-C stops the console.")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, (url,)).start()
    uvicorn.run(build_app(cap, kit, console, visit), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
