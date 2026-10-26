"""The capacity study's visit, served from the study download alone (``STUDY.md``).

    python -m baselines.human.study_server [--kit ~/maps-study]

The download's launchers run this, and the browser opens the visit page. It asks for the
participant's label, walks the briefing, practice, the first condition, a break and the second,
and offers the results file at the end; ``first`` and ``second`` are the participant's two
conditions in the order the plan counterbalanced (``plan/plan.json``). Where the visit is, is
read off the results on disk, so a relaunch picks up where it left off.

The console (``console/``) is one browser tab over the session the visit page opened last:
:class:`baselines.human.capacity.Capacity` over the plan's list, the bench's ortho as tiles
(``study_package tiles``), the briefed AO and the bench's OSM prior; a verified report reads
MAPS's own records from the download's MAPS root. No reasoner, no segmenter, no network, and
nothing served carries an episode's target or a score.

The answers land in ``<kit>/results/<participant>/<condition>/`` (``answers.jsonl`` and the
session's clock, ``session.json``). Starting a condition again resumes its clock; practice
starts empty each time.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import logging
import math
import os
import re
import shutil
import socket
import sqlite3
import tarfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

log = logging.getLogger("study_server")

#: The download's folder, unless ``--kit`` names another.
KIT = Path.home() / "maps-study"
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


def build_app(kit: Path, console: Path, visit: Visit):
    """The visit page's routes, and the console's over the session the visit page opened last.
    One request at a time: a session is a ledger, and a double click must not answer twice."""
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

    box: dict[str, Any] = {"cap": None, "practice": False}

    def session() -> Any:
        if box["cap"] is None:
            # A console tab left open across a relaunch: the visit page opens the session again.
            raise HTTPException(409, "No session is open. Start it again from the study page.")
        return box["cap"]

    def doc() -> dict[str, Any]:
        state = session().state()
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
    ortho = sqlite3.connect((kit / "calcity_ortho.mbtiles").resolve().as_uri() + "?mode=ro", uri=True,
                            check_same_thread=False)

    @app.get("/capacity/state")
    def capacity_state() -> dict[str, Any]:
        with lock:
            return doc()

    @app.post("/capacity/act")
    def capacity_act(req: dict[str, Any]) -> dict[str, Any]:
        with lock:
            cap = session()
            if req.get("action") == "select" and "lon" in req:
                xy = lonlat_to_local(tiles_dir, map_name, float(req["lon"]), float(req["lat"]))
                if xy is None:
                    raise HTTPException(400, "that point is off the map")
                req = {"action": "select", "x": xy[0], "y": xy[1]}
            out = cap.act(req)
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

    @app.get("/visit")
    def visit_page() -> HTMLResponse:
        return HTMLResponse(visit.page(), headers={"Cache-Control": "no-store"})

    @app.get("/visit/label")
    def visit_label(label: str = "") -> Response:
        visit.set_label(label)
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
        data = visit.results_tgz()
        if data is None:
            return HTMLResponse(PAGE.format(body="<h1>No results yet</h1><p>Nothing is saved yet. "
                                                 "<a href='/visit'>Back to the study page</a></p>"), status_code=404)
        return Response(data, media_type="application/gzip", headers={
            "Content-Disposition": f'attachment; filename="maps-study-{visit.participant}.tgz"'})

    # The static mount below would serve index.html by name too, around the redirect.
    @app.get("/")
    @app.get("/index.html")
    def index() -> Response:
        if box["cap"] is None:
            return RedirectResponse("/visit")
        return FileResponse(console / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/", StaticFiles(directory=console), name="console")
    return app


def session_s() -> float:
    """A session's length, seconds. A rehearsal of the visit shortens it; a real visit never sets it."""
    from baselines.human.capacity import SESSION_S

    return float(os.environ.get("MAPS_STUDY_SESSION_S") or SESSION_S)


def open_session(kit: Path, participant: str, condition: str) -> Any:
    """The participant's capacity session in one condition, off the download's plan."""
    from baselines.human.audit_console import bench_labels
    from baselines.human.capacity import Capacity

    if condition == "practice":
        root, order_file, arm = kit / "plan" / "practice", kit / "plan" / "practice" / "practice.txt", "solo"
    else:
        root, order_file, arm = kit / "maps_root", kit / "plan" / participant / f"{condition}.txt", condition
    if not order_file.is_file():
        raise SystemExit(f"no list for {participant} {condition} at {order_file}: is {participant} on the plan?")
    out = kit / "results" / participant / condition
    if condition == "practice":
        shutil.rmtree(out, ignore_errors=True)
    elif (out / "session.json").is_file():
        print(f"{participant} {condition} has a session already: resuming its clock")
    order = [ln.strip() for ln in order_file.read_text().splitlines() if ln.strip()]
    return Capacity(root, order, out, arm, labels=bench_labels(), session_s=session_s())


BRIEFING = Path(__file__).with_name("briefing.md")
STEPS = ("practice", "first", "second")
#: Each condition as the visit page names it.
CONDITIONS = {"solo": "alone", "team": "with MAPS"}


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
    relaunch picks up where the visit left off, under the one label it finds there."""

    def __init__(self, kit: Path) -> None:
        self.kit = kit
        self.plan = json.loads((kit / "plan" / "plan.json").read_text())["participants"]
        contact = kit / "contact.txt"
        self.contact = contact.read_text().strip() if contact.is_file() else "the study lead"
        results = kit / "results"
        found = [p.name for p in results.iterdir() if p.is_dir() and p.name in self.plan] if results.is_dir() else []
        self.participant, self.error = (found[0] if len(found) == 1 else None), ""

    def set_label(self, label: str) -> None:
        label = label.strip().lower()
        if self.participant is not None and label != self.participant:
            self.error = (f"This copy of the study is set up for {self.participant}. "
                          "Another label needs its own copy: ask the study lead.")
        elif label in self.plan:
            self.participant, self.error = label, ""
        else:
            self.error = f"{label or 'That'} is not a participant label on this study. Check it with the study lead."

    def condition(self, step: str) -> str:
        first = self.plan[self.participant]["first"]
        return {"practice": "practice", "first": first, "second": ({"solo", "team"} - {first}).pop()}[step]

    def session(self, step: str) -> dict | None:
        """The step's ``session.json``, once its session has started."""
        meta = self.kit / "results" / self.participant / self.condition(step) / "session.json"
        return json.loads(meta.read_text()) if meta.is_file() else None

    def status(self, step: str) -> str:
        """``todo``, ``running`` or ``done``: a session is over when its clock runs out or
        every listed report has an answer."""
        from baselines.human.audit_console import _jsonl

        out = self.kit / "results" / self.participant / self.condition(step)
        if step == "practice":
            return "done" if out.is_dir() else "todo"
        session = self.session(step)
        if session is None:
            return "todo"
        answered = {a.get("episode_id") for a in _jsonl(out / "answers.jsonl")}
        over = (time.time() - float(session["started_epoch"]) >= float(session["session_s"])
                or len(answered) >= session["episodes_n"])
        return "done" if over else "running"

    def start(self, step: str) -> Any:
        """The step's session, or None when it cannot open now: no label yet, a step out of
        turn (practice once session 1 has started, a session before the one ahead of it), or a
        timed session that is over."""
        if self.participant is None or step not in STEPS:
            return None
        ahead = {"practice": self.status("first") == "todo", "first": self.status("practice") == "done",
                 "second": self.status("first") == "done"}[step]
        if not ahead or (step != "practice" and self.status(step) == "done"):
            return None
        return open_session(self.kit, self.participant, self.condition(step))

    def results_tgz(self) -> bytes | None:
        """The participant's results folder, with the computer's account names left out."""
        folder = self.kit / "results" / self.participant if self.participant else None
        if folder is None or not folder.is_dir():
            return None

        def anonymous(info: tarfile.TarInfo) -> tarfile.TarInfo:
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            return info

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(folder, arcname=f"results/{self.participant}", filter=anonymous)
        return buf.getvalue()

    def page(self) -> str:
        err = f"<p class=err>{html.escape(self.error)}</p>" if self.error else ""
        if self.participant is None:
            body = (f"<h1>The search study</h1>{err}"
                    "<form action='/visit/label'><p>Your participant label, from the study lead:</p>"
                    "<input name=label placeholder='p01' autofocus> <button>Continue</button></form>")
            return PAGE.format(body=body)
        st = {s: self.status(s) for s in STEPS}
        minutes = round(session_s() / 60)

        def name(step: str) -> str:
            return f"Session {STEPS.index(step)}: {CONDITIONS[self.condition(step)]}, {minutes} minutes"

        def button(step: str, label: str, note: str = "") -> str:
            return (f"<form method=post action='/visit/start/{step}'><button>{label}</button></form>"
                    + (f"<p class=note>{note}</p>" if note else ""))

        def running(step: str) -> str:
            session = self.session(step)
            left = float(session["session_s"]) - (time.time() - float(session["started_epoch"]))
            n = STEPS.index(step)
            return (f"<h2>{name(step)}, under way</h2><p>About {math.ceil(left / 60)} minutes left. "
                    f"Its clock kept running while you were away.</p>" + button(step, f"Return to session {n}"))

        clock = f"The {minutes} minutes start when you press the button, so be seated and ready."
        if st["second"] == "done":
            now = ("<h2>Done. One last step: send your results</h2>"
                   "<form action='/visit/results.tgz'><button>Save my results file</button></form>"
                   f"<p>Your browser saves maps-study-{self.participant}.tgz, usually in your Downloads folder. "
                   f"Attach it to an email to {html.escape(self.contact)}. Then you can close this page and the "
                   "text window.</p>")
        elif st["second"] == "running":
            now = running("second")
        elif st["first"] == "done":
            now = (f"<h2>Session 1 is done. Take a five-minute break.</h2><p>Then start {name('second')}.</p>"
                   + button("second", "Start session 2", clock))
        elif st["first"] == "running":
            now = running("first")
        else:
            practiced = st["practice"] == "done"
            now = ("<h2>Step 1: read the briefing below</h2><h2>Step 2: practice</h2>"
                   "<p>Four reports that never count. You work them alone. Try the search box, the street map "
                   "and Measure, then press <b>Done practicing</b> at the top of the console.</p>"
                   + button("practice", "Practice again" if practiced else "Start practice")
                   + f"<h2>Step 3: {name('first')}</h2>" + (button("first", "Start session 1", clock) if practiced
                                                           else "<p class=note>Opens after practice.</p>"))
        brief = md_html(BRIEFING.read_text()) if BRIEFING.is_file() else ""
        return PAGE.format(body=f"<p class=who>Participant {self.participant}</p>{err}{now}<hr>"
                                f"<div class=brief>{brief}</div>")


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


def _answers(host: str, port: int) -> bool:
    """Whether a server answers at the address."""
    try:
        socket.create_connection((host, port), timeout=1).close()
        return True
    except OSError:
        return False


def _open_when_ready(host: str, port: int, url: str) -> None:
    while not _answers(host, port):
        time.sleep(0.5)
    webbrowser.open(url)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", type=Path, default=KIT, help="the study download's folder (default %(default)s)")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = ap.parse_args(argv)
    url = f"http://{args.host}:{args.port}/"
    if _answers(args.host, args.port):
        print("The study is already running in another window. Your browser will show it.")
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    kit = args.kit.expanduser()
    console = kit / "console"
    if not (console / "index.html").is_file():
        raise SystemExit(f"no console at {console}: is {kit} the study download?")
    use_kit(kit)
    import uvicorn

    print(f"The study is running at {url}. Keep this window open until you have saved your results file.")
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(args.host, args.port, url), daemon=True).start()
    uvicorn.run(build_app(kit, console, Visit(kit)), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
