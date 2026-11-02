"""The capacity study's visit, served from the study download alone (``STUDY.md``).

    python -m baselines.human.study_server [--kit ~/maps-study]

The download's launchers run this, and the browser opens the visit page. It asks for the
participant's label, walks the briefing, practice and the one timed session, in the condition
the plan dealt that participant (``plan/plan.json``), and offers the results file at the end.
Where the visit is, is read off the results on disk, so a relaunch picks up where it left off.

The console (``console/``) is one browser tab over the session the visit page opened last:
:class:`baselines.human.capacity.Capacity` over the plan's list, the bench's ortho as tiles
(``study_package tiles``), the briefed AO and the bench's OSM prior; a verified report reads
MAPS's own records from the download's MAPS root. No reasoner, no segmenter, no network, and
nothing served carries an episode's target or a score.

The first label the participant gives mints the visit's run id, ``<label>-<8 hex>``, so no
two visits share a results folder. The answers land in ``<kit>/results/<run id>/<condition>/``
(``answers.jsonl`` and the session's clock, ``session.json``). Starting a condition again resumes its clock; practice
starts empty each time.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
import re
import shutil
import socket
import sqlite3
import threading
import time
import uuid
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


def label_of(run_id: str) -> str:
    """A run id's plan label: the text before its first dash (a bare label is its own)."""
    return run_id.split("-")[0]


def use_kit(kit: Path) -> None:
    """Point the bench at the download, before anything imports its paths."""
    os.environ.setdefault("SEARCH_CALCITY_DATA", str(kit))
    os.environ.setdefault("SEARCH_CALCITY_TILES", str(kit / "tiles"))
    import benchmarks.calcity

    benchmarks.calcity.use()


#: OSM classes whose tag is not a plain word: by (layer, class) where the word depends on the
#: layer (a residential road or area), else by class alone (``WORDS``); any other reads as its tag.
PLAIN = {("road", "residential"): "residential street", ("road", "living_street"): "street",
         ("road", "service"): "service road", ("road", "track"): "dirt road", ("road", "primary"): "main road",
         ("road", "secondary"): "main road", ("road", "tertiary"): "road", ("road", "tertiary_link"): "road",
         ("road", "unclassified"): "road", ("road", "footway"): "footpath", ("road", "cycleway"): "bike path",
         ("road", "stop"): "stop sign", ("road", "traffic_signals"): "traffic lights", ("road", "crossing"): "crosswalk",
         ("road", "turning_circle"): "cul-de-sac", ("landuse", "residential"): "residential area",
         ("landuse", "tree"): "trees", ("landuse", "pitch"): "sports field", ("landuse", "track"): "running track",
         ("landuse", "peak"): "hill", ("landuse", "religious"): "church grounds",
         ("landuse", "commercial"): "commercial area", ("landuse", "industrial"): "industrial area",
         ("landuse", "retail"): "shopping area", ("poi", None): "place", ("power", None): "power line",
         ("barrier", None): "fence or wall", ("building", "commercial"): "commercial building",
         ("building", "retail"): "store"}
WORDS = {"aerodrome": "airport", "apron": "airport apron", "doityourself": "hardware store", "fuel": "gas station",
         "tyres": "tire shop", "alcohol": "liquor store", "convenience": "convenience store",
         "mobile_phone": "phone shop", "tobacco": "tobacco shop", "veterinary": "vet", "parking": "parking lot",
         "greenfield": "building land", "townhall": "town hall", "roof": "shelter roof"}


def _layer(row: dict) -> str:
    """An OSM row's layer, singular: road, building, landuse, poi..."""
    return str(row.get("worldkit_layer") or row.get("category") or "").rstrip("s")


def _kind(row: dict) -> str:
    """What an OSM row is, in plain words: its class where it has one (a road's highway, a POI's
    amenity, a landuse's own tag: farmland, meadow...), else its layer."""
    layer = _layer(row)
    building = row.get("building") if row.get("building") not in (None, "yes") else None
    sub = (row.get("highway") or row.get("amenity") or row.get("shop") or row.get("leisure") or row.get("aeroway")
           or row.get("landuse") or row.get("natural") or building)
    sub = sub if sub != layer else None
    return PLAIN.get((layer, sub)) or WORDS.get(sub) or (sub or layer).replace("_", " ")


def prior_doc(map_name: str, ll) -> dict[str, Any]:
    """The bench's OSM prior in WGS84: every row's outline (a road its centreline), each with the
    id MAPS knew it by (``<category>_<index>``, ``maps.sws.graph.builder``: ``prior_2777``), and
    the named places, for the toggle and the search."""
    from baselines.human.audit_console import block_prior
    from baselines.seam import _row_area

    features, landmarks = block_prior(map_name)
    shapes = []
    for category, rs in features.items():
        for i, r in enumerate(rs if isinstance(rs, list) else []):
            if not isinstance(r, dict):
                continue
            r = {**r, "category": category}
            pts = [(float(p[0]), float(p[1])) for p in r.get("world_contour") or [] if len(p) >= 2]
            if (category == "roads" or r.get("highway")) and len(pts) >= 2:
                line = pts
            else:
                geom = _row_area(r)
                if geom is None or not hasattr(geom, "exterior"):
                    continue
                line = list(geom.simplify(PRIOR_SIMPLIFY_M).exterior.coords)
            shapes.append({"type": "Feature", "properties": {"name": str(r.get("name") or ""), "kind": _kind(r),
                                                          "layer": _layer(r), "fid": f"{category}_{i}"},
                           "geometry": {"type": "LineString", "coordinates": ll(line)}})
    names = sorted(({"name": lm.name, "kind": _kind(lm.tags), "lonlat": ll([lm.centroid])[0]}
                    for lm in landmarks), key=lambda d: d["name"])
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
    # What each step and the reply name, their points and detections in WGS84 too.
    state["turns"] = [{**t, "refs": {**t["refs"], "xy": ll(t["refs"]["xy"])}} for t in state["turns"]]
    state["reply_refs"] = {**state["reply_refs"], "xy": ll(state["reply_refs"]["xy"])}
    state["detected"] = {fid: {"label": d["label"], "lonlat": ll([d["xy"]])[0],
                               "ring": ll(d["contour"]) if len(d["contour"]) >= 3 else None}
                         for fid, d in state["detected"].items()}
    return state


def build_app(kit: Path, console: Path, visit: Visit):
    """The visit page's routes, and the console's over the session the visit page opened last.
    One request at a time: a session is a ledger, and a double click must not answer twice."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
    from fastapi.staticfiles import StaticFiles

    from baselines.human.session_plan import scored_rows
    from benchmarks.calcity.data_prep.aoi import AO_BBOX_WGS84
    from benchmarks.flight.tile_georef import local_to_lonlat, lonlat_to_local

    map_name = "calcity"
    tiles_dir = kit / "tiles"
    lock, tile_lock = threading.Lock(), threading.Lock()
    app = FastAPI()

    def ll(pts) -> list[list[float]]:
        return [list(local_to_lonlat(tiles_dir, map_name, float(x), float(y))) for x, y in pts]

    box: dict[str, Any] = {"cap": None, "practice": False}
    running = [step for step in ("session",) if visit.participant and visit.status(step) == "running"]
    if running:
        # A relaunch mid-session: a console tab left open picks the session up on its next poll.
        box["cap"] = open_session(kit, visit.run, visit.condition(running[0]))

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
    # MAPS's ledger, decoded before the first session opens on it.
    threading.Thread(target=scored_rows, args=(kit / "maps_root",), daemon=True).start()
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
                req = {**{k: v for k, v in req.items() if k not in ("lon", "lat")}, "x": xy[0], "y": xy[1]}
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
    def visit_page(change: bool = False) -> HTMLResponse:
        return HTMLResponse(visit.page(change), headers={"Cache-Control": "no-store"})

    @app.get("/visit/tutorial.mp4")
    def visit_tutorial() -> Response:
        if not visit.video.is_file():
            raise HTTPException(404)
        return FileResponse(visit.video, media_type="video/mp4")

    @app.get("/visit/label")
    def visit_label(label: str = "") -> Response:
        with lock:
            before = visit.run
            visit.set_label(label)
            if visit.run != before:
                # The old label's practice is gone: its session with it.
                box["cap"] = None
        return RedirectResponse("/visit", status_code=303)

    @app.post("/visit/start/{step}")
    def visit_start(step: str) -> Response:
        with lock:
            cap = visit.start(step)
            if cap is None:
                return RedirectResponse("/visit", status_code=303)
            box["cap"], box["practice"] = cap, step == "practice"
        return RedirectResponse("/", status_code=303)

    @app.get("/visit/results.json")
    def visit_results() -> Response:
        data = visit.results_json()
        if data is None:
            return HTMLResponse(PAGE.format(body="<h1>No results yet</h1><p>Nothing is saved yet. "
                                                 "<a href='/visit'>Back to the study page</a></p>"), status_code=404)
        return Response(data, media_type="application/json", headers={
            "Content-Disposition": f'attachment; filename="maps-study-{visit.run}.json"'})

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


def open_session(kit: Path, run: str, condition: str) -> Any:
    """The visit's capacity session in one condition, off the download's plan for its label."""
    from baselines.human.audit_console import bench_labels
    from baselines.human.capacity import Capacity

    participant = label_of(run)
    if condition == "practice":
        root, order_file, arm = kit / "plan" / "practice", kit / "plan" / "practice" / "practice.txt", "solo"
    else:
        root, order_file, arm = kit / "maps_root", kit / "plan" / participant / f"{condition}.txt", condition
    if not order_file.is_file():
        raise SystemExit(f"no list for {participant} {condition} at {order_file}: is {participant} on the plan?")
    out = kit / "results" / run / condition
    if condition == "practice":
        shutil.rmtree(out, ignore_errors=True)
    order = [ln.strip() for ln in order_file.read_text().splitlines() if ln.strip()]
    return Capacity(root, order, out, arm, labels=bench_labels(), session_s=session_s(),
                    ids={"run_id": run, "participant": participant})


BRIEFING = Path(__file__).with_name("briefing.md")
#: The tutorial video's name at the bundle's top, beside the READ ME.
VIDEO = "Watch first - tutorial.mp4"
STEPS = ("practice", "session")
#: Each condition as the visit page names it.
CONDITIONS = {"solo": "alone", "team": "with MAPS"}
#: The results file's ``format`` and ``version``, as ``study_collect`` checks them.
RESULTS_FORMAT, RESULTS_VERSION = "maps-capacity-results", 1


class Visit:
    """One participant's visit, as the visit page walks it: the briefing, practice, the session
    in the participant's condition, done. Where they are is read off the results on disk, so a
    relaunch picks up where the visit left off, under the one run id it finds there.
    ``participant`` is the plan label; ``run`` the visit's run id, its results folder."""

    def __init__(self, kit: Path) -> None:
        self.kit = kit
        #: The tutorial, where the bundle ships it: beside the READ ME, a folder above the kit.
        self.video = kit.parent / VIDEO
        self.plan = json.loads((kit / "plan" / "plan.json").read_text())["participants"]
        contact, roster = kit / "contact.txt", kit / "roster.json"
        self.contact = contact.read_text().strip() if contact.is_file() else "the study lead"
        # The participants' first names by label: the label form is then a list of names. The label
        # alone goes into the results.
        self.roster = {k: v for k, v in json.loads(roster.read_text()).items() if k in self.plan} if roster.is_file() else {}
        results = kit / "results"
        found = [p.name for p in results.iterdir() if p.is_dir() and label_of(p.name) in self.plan] if results.is_dir() else []
        self.run = found[0] if len(found) == 1 else None
        self.participant, self.error = (label_of(self.run) if self.run else None), ""
        self.typed = ""
        self.returning = self.participant is not None
        # A session whose clock ran on while the study was closed, until the participant returns to it.
        self.away = self.returning and self.status("session") == "running"

    def locked(self) -> bool:
        """Whether the label is fixed: once the session has started, its answers are under it."""
        return self.participant is not None and self.session("session") is not None

    def set_label(self, label: str) -> None:
        """The participant's label; the first mints the visit's run id, and a change before
        the session discards the old run's practice and mints another."""
        label = label.strip().lower()
        self.typed = label
        if not label:
            self.error = "Choose your first name." if self.roster else "Type your label, for example p07."
        elif label not in self.plan:
            self.error = ("That is not a participant label in this study. Labels look like p07; "
                          "check the one the study lead sent you.")
        elif label != self.participant and self.locked():
            what = "name" if self.roster else "label"
            self.error = f"You started as {self.who()}. A different {what} needs a fresh copy of the study: ask the study lead."
        else:
            if label != self.participant:
                if self.run is not None:
                    shutil.rmtree(self.kit / "results" / self.run, ignore_errors=True)
                self.run, self.returning = f"{label}-{uuid.uuid4().hex[:8]}", False
            self.participant, self.error, self.typed = label, "", ""

    def who(self) -> str:
        """The participant as the page names them: their first name from the roster, else their label."""
        return self.roster.get(self.participant, self.participant or "")

    def condition(self, step: str) -> str:
        """The step's condition: practice, or the participant's one condition from the plan."""
        return "practice" if step == "practice" else self.plan[self.participant]["arm"]

    def session(self, step: str) -> dict | None:
        """The step's ``session.json``, once its session has started."""
        meta = self.kit / "results" / self.run / self.condition(step) / "session.json"
        return json.loads(meta.read_text()) if meta.is_file() else None

    def status(self, step: str) -> str:
        """``todo``, ``running`` or ``done``: a session is over when its clock runs out or
        every listed report has an answer."""
        from baselines.human.audit_console import _jsonl

        out = self.kit / "results" / self.run / self.condition(step)
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
        turn (practice once the session has started, the session before practice), or a timed
        session that is over."""
        if self.participant is None or step not in STEPS:
            return None
        ahead = {"practice": self.status("session") == "todo", "session": self.status("practice") == "done"}[step]
        if not ahead or (step != "practice" and self.status(step) == "done"):
            return None
        if step != "practice":
            self.away = False
        return open_session(self.kit, self.run, self.condition(step))

    def results_json(self) -> bytes | None:
        """The visit's results file: each session's clock and answers in one JSON document. Its
        own keys are sorted and the ledgers' rows keep the order they were written in, so two
        saves of the same results differ only in ``saved_at`` and ``study_collect`` writes the
        ledgers back byte for byte. It holds the ledgers' contents alone, so the computer's
        account names and paths stay out of it."""
        from baselines.human.audit_console import _jsonl

        folder = self.kit / "results" / self.run if self.run else None
        if folder is None or not folder.is_dir():
            return None
        sessions = {}
        for name in ("practice", "solo", "team"):
            meta, ledger = folder / name / "session.json", folder / name / "answers.jsonl"
            if not meta.is_file():
                continue
            answers = _jsonl(ledger)
            sessions[name] = {"answers": answers, "session": json.loads(meta.read_text())}
            lines = sum(1 for ln in ledger.read_text().splitlines() if ln.strip()) if ledger.is_file() else 0
            if lines > len(answers):
                sessions[name]["unreadable_lines"] = lines - len(answers)
        doc = {"format": RESULTS_FORMAT, "participant": self.participant, "run_id": self.run,
               "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sessions": sessions,
               "version": RESULTS_VERSION}
        return (json.dumps(doc, indent=1) + "\n").encode()

    def name(self, step: str) -> str:
        """A timed session as the visit page names it, at the length its own clock holds once it has one."""
        session = self.session(step)
        minutes = round((float(session["session_s"]) if session else session_s()) / 60)
        return f"your session ({CONDITIONS[self.condition(step)]}, {minutes} minutes)"

    def page(self, change: bool = False) -> str:
        """The visit page; ``change`` asks for the label form while the label can still change.
        An error shows on the page that follows it, once."""
        err = f"<p class=err>{html.escape(self.error)}</p>" if self.error else ""
        typed, self.error, self.typed = self.typed, "", ""
        if self.participant is None or (change and not self.locked()):
            if self.roster:
                # By name, in the alphabet's order: the list says nothing of labels or conditions.
                names = sorted(self.roster.items(), key=lambda kv: kv[1].lower())
                options = "".join(f"<option value='{html.escape(k, quote=True)}'{' selected' if k == typed else ''}>"
                                  f"{html.escape(v)}</option>" for k, v in names)
                field = ("<p>Select who you are from the list, then press Continue.</p><select name=label autofocus>"
                         f"<option value=''>Choose...</option>{options}</select> ")
            else:
                field = ("<p>Your participant label, from the study lead:</p>"
                         f"<input name=label placeholder='like p07' value='{html.escape(typed, quote=True)}' autofocus> ")
            body = f"<h1>The search study</h1>{err}<form action='/visit/label'>{field}<button>Continue</button></form>"
            return PAGE.format(body=body)
        st = {s: self.status(s) for s in STEPS}
        minutes = round(session_s() / 60)

        def button(step: str, label: str, note: str = "") -> str:
            # Opening a session takes a few seconds: the button says so, and a second click cannot repeat it.
            return (f"<form method=post action='/visit/start/{step}' onsubmit=\"this.firstChild.disabled=true;"
                    f"this.firstChild.textContent='Opening...'\"><button>{label}</button></form>"
                    + (f"<p class=note>{note}</p>" if note else ""))

        def running(step: str) -> str:
            session = self.session(step)
            left = math.ceil((float(session["session_s"]) - (time.time() - float(session["started_epoch"]))) / 60)
            away = " Its clock kept running while you were away." if self.away else ""
            return (f"<h2>Your session ({CONDITIONS[self.condition(step)]}) is under way</h2>"
                    f"<p>About {left} minute{'' if left == 1 else 's'} left.{away}</p>"
                    + button(step, "Return to the session"))

        clock = f"The {minutes} minutes start when you press the button, so be seated and ready."
        contact, filename = html.escape(self.contact), f"maps-study-{self.run}.json"
        if st["session"] == "done":
            now = ("<h2>Done. One last step: send your results</h2>"
                   "<form action='/visit/results.json' onsubmit=\"document.getElementById('saved').hidden=false\">"
                   "<button>Save my results file</button></form>"
                   "<p id=saved hidden>Your browser is saving the file.</p>"
                   f"<p>Your browser saves {filename}, usually in your Downloads folder. "
                   f"Attach it to an email to {contact}. Then you can close this page and the "
                   "text window (on a Mac, press Terminate if asked).</p>")
        elif st["session"] == "running":
            now = running("session")
        else:
            practiced = st["practice"] == "done"
            first = int(self.video.is_file())
            # Back from practice, the next step leads the page: the video and briefing stay below it.
            ready = (f"<div class=next><h2>Practice done. Next: {self.name('session')}</h2>"
                     + button("session", "Start your session", clock) + "</div>") if practiced else ""
            now = ready + (f"<p class=note><a href='/visit?change=1'>Not {'you' if self.roster else 'your label'}? Change it</a></p>"
                   + ("<h2>Step 1: watch the tutorial</h2><video controls preload=metadata "
                      "src='/visit/tutorial.mp4'></video>" if self.video.is_file() else "")
                   + f"<h2>Step {first + 1}: read the briefing</h2><div class=brief>{self.briefing()}</div>"
                   f"<h2>Step {first + 2}: practice</h2>"
                   "<p>Four reports that never count, worked alone. Try the search box, the street map and "
                   "shift-click measuring. Practice ends after the fourth report; <b>Done practicing</b> at the top "
                   "stops it early.</p>"
                   + button("practice", "Practice again" if practiced else "Start practice")
                   + f"<h2>Step {first + 3}: {self.name('session')}</h2>" + (button("session", "Start your session", clock)
                                                                   if practiced else "<p class=note>Opens after practice.</p>"))
        who = (f"Welcome back, {html.escape(self.who())}. Your answers are saved." if self.returning
               else f"Welcome, {html.escape(self.who())}" if self.roster else f"Participant {self.participant}")
        # The run id, for the participant to quote in an email about this visit.
        # Under way or done, the briefing stays below as the reference it is.
        below = "" if st["session"] == "todo" else f"<hr><div class=brief>{self.briefing()}</div>"
        return PAGE.format(body=f"<p class=who>{who} <small>(results id {self.run})</small></p>{err}{now}{below}")

    def briefing(self) -> str:
        """The briefing as HTML: the participant's own session section, at the session's length."""
        from markdown_it import MarkdownIt

        if not BRIEFING.is_file():
            return ""
        own = f"Your session: {CONDITIONS[self.condition('session')]}"
        parts = BRIEFING.read_text().split("\n## ")
        kept = [parts[0]] + [t for t in parts[1:] if not t.startswith("Your session: ") or t.startswith(own + "\n")]
        session = self.session("session")
        minutes = round((float(session["session_s"]) if session else session_s()) / 60)
        text = "\n## ".join(kept).replace("{MINUTES}", str(minutes))
        return MarkdownIt("commonmark", {"html": False}).enable("table").render(text)


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8><title>The search study</title><style>
:root {{ --ink: #141310; --paper: #fbfaf5; --navy: #1f3a5f; --grey: #6b675d; }}
body {{ margin: 0 auto; max-width: 46rem; padding: 32px 16px 64px; font: 17px/1.5 system-ui, sans-serif;
       color: var(--ink); background: var(--paper); }}
h1 {{ font-size: 28px; }} h2 {{ font-size: 21px; margin-top: 28px; }}
.brief h1 {{ font-size: 24px; }} .brief h2 {{ font-size: 19px; }}
button {{ font: inherit; font-weight: 700; padding: 10px 22px; color: #fff; background: var(--navy);
         border: 0; cursor: pointer; }}
input, select {{ font: inherit; padding: 8px; }} input {{ width: 8rem; }} select {{ min-width: 12rem; }}
.next {{ background: #e3e9f1; padding: 4px 20px 20px; margin-bottom: 8px; }}
.note, .who {{ color: var(--grey); }} .err {{ color: #a3341f; }}
video {{ width: 100%; background: #000; }}
.brief table {{ border-collapse: collapse; font-size: 15px; }}
.brief th, .brief td {{ border-top: 1px solid #d9d4c6; padding: 6px 10px 6px 0; text-align: left; vertical-align: top; }}
hr {{ margin: 36px 0; border: 0; border-top: 1px solid #d9d4c6; }}
</style></head><body>{body}</body></html>"""


def _answers(host: str, port: int) -> bool:
    """Whether a server answers at the address."""
    try:
        socket.create_connection((host, port), timeout=1).close()
        return True
    except OSError:
        return False


def _is_study(url: str) -> bool:
    """Whether the server at ``url`` is this study, by its visit page."""
    import urllib.request

    # Straight to the computer itself, past any system proxy.
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url + "visit", timeout=3) as r:
            return b"The search study" in r.read()
    except (OSError, ValueError):
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

    def taken() -> int:
        if not _is_study(url):
            print(f"Port {args.port} is taken by another program. Restart the computer and start the study again.")
            return 1
        print("The study is already running in another window. Your browser will show it.")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    if _answers(args.host, args.port):
        return taken()
    kit = args.kit.expanduser()
    console = kit / "console"
    if not (console / "index.html").is_file():
        raise SystemExit(f"no console at {console}: is {kit} the study download?")
    use_kit(kit)
    import uvicorn

    app = build_app(kit, console, Visit(kit))
    # The port, bound and listening before anything says the study runs: a second start during
    # this one's startup finds it here rather than failing to bind after its own.
    sock = socket.socket()
    if os.name != "nt":
        # Windows lets a second program bind a port held with this option; elsewhere it only
        # skips the wait after a closed server.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.host, args.port))
        sock.listen()
    except OSError:
        sock.close()
        return taken()
    print(f"The study is running at {url}. Keep this window open until you have saved your results file.")
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(args.host, args.port, url), daemon=True).start()
    uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")).run(sockets=[sock])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
