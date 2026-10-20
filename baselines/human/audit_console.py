"""The audit console: a person judges each finished MAPS episode, then confirms its target or picks the true one.

MAPS flies alone first (the paired root). The console shows the auditor the
instruction, the imagery around MAPS's commit with every candidate MAPS
detected outlined and its pick marked, MAPS's own reasoning for the pick, the
full trace behind it and a search over the block's named prior. The auditor
confirms, or selects another candidate or point and commits that; a
replacement the budget MAPS left cannot fly is refused. Nothing the auditor
sees carries the target or a score: the trace is built from MAPS's records
alone (``blind_review``), and no verdict is shown before, during or after.

    python -m baselines.human.audit_console \\
        --maps-root output/e2/<root> --episode-allowlist <study>/p01/team.txt --out <study>/p01

reads the imagery and the tile bake of the bench the root flew (the
``SEARCH_CITYNAV_*`` roots, or ``SEARCH_CALCITY_*`` for a Cal City root),
serves ``http://127.0.0.1:8078/`` and appends one line per decision to
``<out>/decisions.jsonl``; a restarted session resumes after the last one.
``team_rows`` scores the decisions.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from baselines.human.policy import LABEL, world_to_px
from baselines.human.session_plan import scored_rows
from baselines.human.team_rows import affordable, root_config, select_bench
from baselines.seam import Landmark, block_landmarks
from benchmarks.citynav.analysis.rows import iter_json_rows

DEFAULT_PORT = 8078
#: The view's half-widths the auditor steps through, metres.
ZOOM_HALF_M = (30.0, 60.0, 120.0, 240.0)
#: The canvas's longest side, pixels: the 0.1 m/px raster is resampled to it.
CANVAS_PX = 1100
#: A detection within this of MAPS's commit is the candidate MAPS picked.
PICK_TOL_M = 2.0
COLOURS = {"prior": (255, 214, 0), "candidate": (0, 220, 255), "maps": (255, 0, 255),
           "selected": (60, 255, 60), "stop": (255, 255, 255)}
_SHA = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class Candidate:
    """One object MAPS detected: what the auditor may pick instead."""

    id: int
    label: str
    color: str
    xy: tuple[float, float]
    contour: list[tuple[float, float]]


@dataclass
class Review:
    """One finished MAPS episode as an auditor may see it: no target, no score."""

    episode_id: str
    map_name: str
    instruction: str
    attempt: Path
    maps_xy: tuple[float, float]
    turns: list[dict]
    ended_by: str
    candidates: list[Candidate]
    maps_pick: int | None
    stops: list[tuple[float, float]]
    landmarks: list[Landmark] = field(default_factory=list)


def _jsonl(path: Path) -> list[dict]:
    return list(iter_json_rows(path)) if path.is_file() else []


def candidates_of(attempt: Path) -> list[Candidate]:
    """Every object MAPS detected, once per track (its most confident sighting)."""
    best: dict[str, dict] = {}
    for i, d in enumerate(_jsonl(attempt / "detections.jsonl")):
        key = str((d.get("metadata") or {}).get("track_id") or f"row{i}")
        if key not in best or float(d.get("confidence") or 0) > float(best[key].get("confidence") or 0):
            best[key] = d
    out = []
    for d in best.values():
        contour = [(float(x), float(y)) for x, y in (d.get("metadata") or {}).get("world_contour") or []]
        out.append(Candidate(id=len(out), label=str(d.get("label") or "object"),
                             color=str((d.get("attributes") or {}).get("color") or "").split(" / ")[0],
                             xy=(float(d["world_xy"][0]), float(d["world_xy"][1])), contour=contour))
    return out


def blind_review(maps_root: Path, row: dict, landmarks: list[Landmark]) -> Review:
    """The review of one MAPS row. Reads the row's instruction and commit, never its
    target or metrics, and the attempt's own records."""
    attempt = maps_root / row["workdir"]
    turns = [{"reasoning": str(r.get("reasoning") or ""), "said": str(r.get("said") or ""),
              "printed": str(r.get("printed") or ""), "frames": [f for f in r.get("frames") or [] if _SHA.match(str(f))]}
             for r in _jsonl(attempt / "conversation.jsonl") if r.get("kind") == "run"]
    ended = next((str(r.get("ended_by")) for r in _jsonl(attempt / "conversation.jsonl") if r.get("kind") == "mission_end"), "")
    maps_xy = (float(row["commit_xy"][0]), float(row["commit_xy"][1]))
    cands = candidates_of(attempt)
    near = [c for c in cands if np.hypot(c.xy[0] - maps_xy[0], c.xy[1] - maps_xy[1]) <= PICK_TOL_M]
    try:
        cursor = json.loads((attempt / "flight" / "tour_cursor.json").read_text())
        stops = [(float(s["xy"][0]), float(s["xy"][1])) for s in cursor.get("stops_visited") or [] if s.get("xy")]
    except (OSError, ValueError):
        stops = []
    return Review(episode_id=str(row["episode_id"]), map_name=str(row["map_name"]), instruction=str(row["description"]),
                  attempt=attempt, maps_xy=maps_xy, turns=turns, ended_by=ended, candidates=cands,
                  maps_pick=min(near, key=lambda c: np.hypot(c.xy[0] - maps_xy[0], c.xy[1] - maps_xy[1])).id if near else None,
                  stops=stops, landmarks=landmarks)


def _inside(xy, contour) -> bool:
    if len(contour) < 3:
        return False
    from shapely.geometry import Point, Polygon

    return Polygon(contour).buffer(0).contains(Point(xy))


class Session:
    """An auditor's pass over an allowlist, one review at a time."""

    def __init__(self, maps_root: Path, episode_ids: list[str], out: Path, crop) -> None:
        self.maps_root, self.out, self.crop = maps_root, out, crop
        self.rows = scored_rows(maps_root)
        missing = [e for e in episode_ids if e not in self.rows]
        if missing:
            raise SystemExit(f"FATAL: {len(missing)} allowlisted episodes have no scored MAPS row, e.g. {missing[:3]}")
        self.max_steps = int(root_config(maps_root)["max_steps"])
        self.decisions = out / "decisions.jsonl"
        done = {d["episode_id"] for d in _jsonl(self.decisions)}
        self.queue = [e for e in episode_ids if e not in done]
        self.total, self.done_n = len(episode_ids), len(episode_ids) - len(self.queue)
        self._landmarks: dict[str, list[Landmark]] = {}
        self.review: Review | None = None
        self._next()

    def _next(self) -> None:
        self.review = None
        if not self.queue:
            return
        from benchmarks.citynav.run_benchmark.assets import load_episode_assets

        row = self.rows[self.queue[0]]
        if row["map_name"] not in self._landmarks:
            self._landmarks[row["map_name"]] = block_landmarks(load_episode_assets(row["map_name"]).base_features)
        self.review = blind_review(self.maps_root, row, self._landmarks[row["map_name"]])
        self.centre, self.zoom = self.review.maps_xy, 1
        self.selected: dict | None = None
        self.highlight: str | None = None
        self.t0 = time.monotonic()
        self.zooms = 0
        self._render()

    def _render(self) -> None:
        r = self.review
        self.obs = self.crop(r.map_name, self.centre, ZOOM_HALF_M[self.zoom])
        img = Image.fromarray(np.asarray(self.obs.rgb, dtype=np.uint8))
        scale = min(1.0, CANVAS_PX / max(img.size))
        if scale < 1.0:
            img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.BILINEAR)
        draw, shape = ImageDraw.Draw(img), (img.size[1], img.size[0])
        x0, y0, x1, y1 = self.obs.world_window

        def px(p):
            return world_to_px(self.obs.ortho_view_corners_xy, shape, p)

        def visible(pts) -> bool:
            xs, ys = zip(*pts)
            return not (max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1)

        for lm in r.landmarks:
            if visible(lm.contour):
                heavy = lm.name == self.highlight
                draw.line([px(p) for p in lm.contour + lm.contour[:1]], fill=COLOURS["prior"], width=4 if heavy else 1)
                draw.text(px(lm.centroid), lm.name, fill=COLOURS["prior"], **LABEL)
        sel = (self.selected or {}).get("candidate")
        for c in r.candidates:
            pts = c.contour or [c.xy]
            if not visible(pts):
                continue
            kind = "selected" if c.id == sel else "maps" if c.id == r.maps_pick else "candidate"
            if len(pts) >= 2:
                draw.line([px(p) for p in pts + pts[:1]], fill=COLOURS[kind], width=4 if kind != "candidate" else 2)
            cx, cy = px(c.xy)
            draw.text((cx + 4, cy - 12), f"{c.id}" + (" MAPS" if c.id == r.maps_pick else ""), fill=COLOURS[kind], **LABEL)
        for s in r.stops:
            sx, sy = px(s)
            draw.ellipse((sx - 3, sy - 3, sx + 3, sy + 3), outline=COLOURS["stop"])
        for xy, colour in ((r.maps_xy, COLOURS["maps"]), ((self.selected or {}).get("xy"), COLOURS["selected"])):
            if xy is not None:
                mx, my = px(xy)
                draw.line((mx - 9, my, mx + 9, my), fill=colour, width=3)
                draw.line((mx, my - 9, mx, my + 9), fill=colour, width=3)
        self.shape = shape
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        self.frame = buf.getvalue()

    def state(self) -> dict:
        r = self.review
        if r is None:
            return {"done": True, "total": self.total}
        sel = self.selected
        return {
            "done": False, "index": self.done_n + 1, "total": self.total, "instruction": r.instruction,
            "ended_by": r.ended_by, "maps_pick": r.maps_pick,
            "maps_reason": r.turns[-1]["reasoning"] if r.turns else "",
            "turns": r.turns, "zoom": self.zoom, "view_m": 2 * ZOOM_HALF_M[self.zoom],
            "candidates": [{"id": c.id, "label": c.label, "color": c.color,
                            "dist_m": round(float(np.hypot(c.xy[0] - r.maps_xy[0], c.xy[1] - r.maps_xy[1])))}
                           for c in r.candidates],
            "selected": None if sel is None else {**sel, "affordable": affordable(self.rows[r.episode_id], sel["xy"], self.max_steps)},
            "landmarks": [{"name": lm.name, "kind": str(lm.tags.get("category") or ""),
                           "dist_m": round(float(np.hypot(lm.centroid[0] - r.maps_xy[0], lm.centroid[1] - r.maps_xy[1])))}
                          for lm in sorted(r.landmarks, key=lambda lm: np.hypot(lm.centroid[0] - r.maps_xy[0], lm.centroid[1] - r.maps_xy[1]))],
        }

    def act(self, req: dict) -> dict:
        r = self.review
        if r is None:
            return {"ok": False, "error": "the session is complete"}
        kind = req.get("action")
        if kind == "pick":
            rows, cols = self.shape
            from maps.perception.geometry.geolocate import frame_world_xy

            _u, _v, x, y = frame_world_xy(self.obs.ortho_view_corners_xy, np.array([float(req["v"]) * (rows - 1)]),
                                          np.array([float(req["u"]) * (cols - 1)]), self.shape)
            xy = (float(x[0]), float(y[0]))
            hit = next((c for c in r.candidates if _inside(xy, c.contour)), None)
            self.selected = {"candidate": hit.id, "xy": hit.xy} if hit else {"candidate": None, "xy": xy}
        elif kind == "candidate":
            c = r.candidates[int(req["id"])]
            self.selected = {"candidate": c.id, "xy": c.xy}
            self.centre = c.xy
        elif kind == "clear":
            self.selected = None
        elif kind == "zoom":
            self.zoom = min(max(self.zoom + int(req["step"]), 0), len(ZOOM_HALF_M) - 1)
            self.zooms += 1
        elif kind == "centre":
            target = req.get("on")
            if target == "maps":
                self.centre = r.maps_xy
            elif target == "selected" and self.selected:
                self.centre = self.selected["xy"]
            elif target == "landmark":
                lm = next(lm for lm in r.landmarks if lm.name == req["name"])
                self.centre, self.highlight = lm.centroid, lm.name
        elif kind in ("confirm", "replace"):
            return self._decide(kind, req)
        else:
            return {"ok": False, "error": f"unknown action {kind!r}"}
        self._render()
        return {"ok": True}

    def _decide(self, kind: str, req: dict) -> dict:
        r = self.review
        if kind == "replace":
            if not self.selected:
                return {"ok": False, "error": "select a candidate or a point first"}
            if not affordable(self.rows[r.episode_id], self.selected["xy"], self.max_steps):
                return {"ok": False, "error": "MAPS's remaining budget cannot fly to that point"}
        decision = {
            "episode_id": r.episode_id, "decision": kind,
            "xy": list(self.selected["xy"]) if kind == "replace" else list(r.maps_xy),
            "candidate": (self.selected or {}).get("candidate") if kind == "replace" else r.maps_pick,
            "maps_xy": list(r.maps_xy), "review_s": round(time.monotonic() - self.t0, 1),
            "zooms": self.zooms, "opened": sorted({int(i) for i in req.get("opened") or []}),
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.out.mkdir(parents=True, exist_ok=True)
        with self.decisions.open("a") as fh:
            fh.write(json.dumps(decision) + "\n")
        self.queue.pop(0)
        self.done_n += 1
        self._next()
        return {"ok": True}


def ortho_crop(rgbd_dir: Path):
    """The block's raster around a point, north-up, ``2 * half`` metres a side."""
    from maps.perception.observation import crop_observation
    from maps.perception.ortho.ground_level import GROUND_LEVEL

    def crop(map_name: str, xy: tuple[float, float], half: float):
        return crop_observation(xy, altitude_m=float(GROUND_LEVEL[map_name]) + half, rgbd_dir=rgbd_dir,
                                map_name=map_name, sensor=None)
    return crop


def _handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/state":
                self._send(json.dumps(session.state()).encode(), "application/json")
            elif path == "/frame.jpg" and session.review is not None:
                self._send(session.frame, "image/jpeg")
            elif path.startswith("/maps_frame/") and session.review is not None:
                sha = path.rsplit("/", 1)[1].removesuffix(".jpg")
                f = session.review.attempt / "frames" / f"{sha}.jpg"
                if _SHA.match(sha) and f.is_file():
                    self._send(f.read_bytes(), "image/jpeg")
                else:
                    self._send(b"not found", "text/plain", 404)
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                out = session.act(req)
            except (ValueError, KeyError, TypeError, IndexError, StopIteration) as exc:
                out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self._send(json.dumps(out).encode(), "application/json", 200 if out["ok"] else 400)

        def log_message(self, *_args) -> None:
            pass

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--maps-root", type=Path, required=True)
    ap.add_argument("--episode-allowlist", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="the participant's folder: decisions.jsonl lands here")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args(argv)
    select_bench(args.maps_root)
    from benchmarks.citynav.paths import CITYNAV_NATIVE_RGBD_DIR

    ids = [ln.strip() for ln in args.episode_allowlist.read_text().splitlines() if ln.strip()]
    session = Session(args.maps_root, ids, args.out, ortho_crop(CITYNAV_NATIVE_RGBD_DIR))
    with HTTPServer((args.host, args.port), _handler(session)) as server:
        print(f"audit console: http://{args.host}:{args.port}/  {session.total - session.done_n} of {session.total} to review", flush=True)
        while session.review is not None:
            server.handle_request()
        # The page's last poll learns the session is complete.
        server.timeout = 5
        server.handle_request()
    print(f"session complete: {session.decisions}")
    return 0


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Audit Console</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f5f5f3;--panel:#fff;--fg:#1a1a1a;--muted:#6b6b6b;--line:#d9d9d6;--accent:#0a5cc2;--ok:#1d7a3a;--warn:#b3261e;--maps:#c000c0}
@media (prefers-color-scheme:dark){:root{--bg:#121212;--panel:#1c1c1c;--fg:#e9e9e9;--muted:#9a9a9a;--line:#333;--accent:#6aa8ff;--ok:#5fcf80;--warn:#ff8a80;--maps:#ff6bff}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{display:flex;gap:16px;align-items:baseline;padding:10px 16px;border-bottom:1px solid var(--line)}
header .k{color:var(--muted)}
#instr{font-size:19px;font-weight:600;flex:1}
main{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:16px;padding:16px;max-width:1600px;margin:auto}
@media (max-width:900px){main{grid-template-columns:1fr}}
#wrap{line-height:0;text-align:center}
#frame{max-width:100%;max-height:calc(100vh - 170px);cursor:crosshair;border:1px solid var(--line);background:#000}
.tools{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0;align-items:center}
button{font:inherit;padding:6px 11px;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:5px;cursor:pointer}
button.big{padding:10px 14px;font-weight:600;width:100%}
button.confirm{background:var(--maps);border-color:var(--maps);color:#fff}
button.replace{background:var(--ok);border-color:var(--ok);color:#fff}
button:disabled{opacity:.45;cursor:not-allowed}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:3px;padding:0 4px;margin-left:6px;opacity:.85}
aside{position:sticky;top:8px;align-self:start;max-height:calc(100vh - 16px);overflow:auto}
@media (max-width:900px){aside{position:static;max-height:none}}
aside section{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px 12px;margin-bottom:12px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 6px}
#reason{white-space:pre-wrap;max-height:180px;overflow:auto}
.list{max-height:200px;overflow:auto}
.item{padding:3px 4px;cursor:pointer;border-radius:3px;display:flex;justify-content:space-between;gap:8px}
.item:hover,.item.on{background:var(--line)}
.item .k,.k{color:var(--muted)}
.maps{color:var(--maps);font-weight:600}
#msg{color:var(--warn);min-height:1.3em;margin:6px 0}
details{border-top:1px solid var(--line);padding:4px 0}
summary{cursor:pointer}
pre{white-space:pre-wrap;font:12px/1.4 ui-monospace,monospace;background:var(--bg);padding:6px;border-radius:4px;max-height:220px;overflow:auto}
.thumbs img{width:31%;margin:1%;cursor:zoom-in;border:1px solid var(--line)}
#big{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:9}
#big img{max-width:94vw;max-height:94vh}
input{font:inherit;width:100%;padding:5px;border:1px solid var(--line);border-radius:4px;background:var(--panel);color:var(--fg)}
</style></head><body>
<header><div id="instr"></div><div class="k" id="prog"></div><div class="k" id="clock"></div></header>
<main>
<div>
  <div id="wrap"><img id="frame" alt="imagery around MAPS's commit"></div>
  <div class="tools">
    <button data-a="zin">Zoom in<kbd>+</kbd></button><button data-a="zout">Zoom out<kbd>-</kbd></button>
    <button data-a="cmaps">Centre on MAPS<kbd>M</kbd></button><button data-a="csel">Centre on selection<kbd>C</kbd></button>
    <button data-a="clear">Clear selection<kbd>Esc</kbd></button><span class="k" id="view"></span>
  </div>
  <div class="k">Magenta: MAPS's pick. Cyan: other objects MAPS detected, numbered. Yellow: named places in the map prior. White rings: where MAPS stopped. Click an outline to select that object, or anywhere to select a point. North is up.</div>
</div>
<aside>
  <section>
    <button class="big confirm" id="confirm">Confirm MAPS's target<kbd>Enter</kbd></button>
    <div style="height:8px"></div>
    <button class="big replace" id="replace" disabled>Select a different target first<kbd>S</kbd></button>
    <div id="msg"></div>
  </section>
  <section><h2>Why MAPS chose its target</h2><div id="reason"></div></section>
  <section><h2>Objects MAPS detected</h2><div class="list" id="cands"></div></section>
  <section><h2>Search the map prior</h2><input id="q" type="search" placeholder="a named place"><div class="list" id="lms"></div></section>
  <section><h2>Full MAPS trace</h2><div id="trace"></div></section>
</aside>
</main>
<div id="big"><img alt="frame MAPS saw"></div>
<script>
const $=id=>document.getElementById(id);
const esc=t=>String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;');
let st=null,opened=new Set(),t0=Date.now(),busy=false;
const tick=setInterval(()=>{const s=Math.round((Date.now()-t0)/1000);$('clock').textContent=Math.floor(s/60)+':'+String(s%60).padStart(2,'0')},1000);
async function post(b){
  if(busy)return false;busy=true;$('msg').textContent='';
  try{const r=await(await fetch('/',{method:'POST',body:JSON.stringify(b)})).json();if(!r.ok){$('msg').textContent=r.error;return false}return true}
  finally{busy=false}
}
async function refresh(full){
  const prev=st&&st.index;
  st=await(await fetch('/state')).json();
  if(st.done){clearInterval(tick);document.body.innerHTML='<main><section><h2>Session complete</h2><p>'+st.total+' episodes reviewed. You can close this page.</p></section></main>';return}
  if(st.index!==prev){opened=new Set();t0=Date.now();$('q').value='';full=true}
  $('frame').src='/frame.jpg?'+Date.now();
  $('view').textContent=`view ${st.view_m} m across`;
  if(full){
    $('instr').textContent=st.instruction;$('prog').textContent=`episode ${st.index} of ${st.total}`;
    $('reason').textContent=st.maps_reason||'(MAPS gave no reasoning on its last turn)';
    $('trace').innerHTML=st.turns.map((t,i)=>`<details data-i="${i}"><summary>Turn ${i+1}: ${esc(t.reasoning.slice(0,90))}…</summary>
      <div>${esc(t.reasoning)}</div><pre>${esc(t.said)}</pre>${t.printed?`<pre>${esc(t.printed)}</pre>`:''}
      ${t.frames.length?`<div class="thumbs">${t.frames.map(f=>`<img loading="lazy" src="/maps_frame/${f}.jpg" alt="frame">`).join('')}</div>`:''}</details>`).join('')
      +`<div class="k">ended by: ${esc(st.ended_by)}</div>`;
    lms();
  }
  const sel=st.selected;
  $('cands').innerHTML=st.candidates.map(c=>`<div class="item ${sel&&sel.candidate===c.id?'on':''}" data-c="${c.id}">
    <span class="${c.id===st.maps_pick?'maps':''}">${c.id} ${esc(c.label)} ${esc(c.color)}${c.id===st.maps_pick?' (MAPS)':''}</span><span class="k">${c.dist_m} m</span></div>`).join('')||'<div class="k">none recorded</div>';
  const rb=$('replace');
  if(!sel){rb.disabled=true;rb.firstChild.textContent='Select a different target first'}
  else if(!sel.affordable){rb.disabled=true;rb.firstChild.textContent='Out of MAPS\'s remaining budget'}
  else{rb.disabled=false;rb.firstChild.textContent=sel.candidate===null?'Commit the selected point':`Commit object ${sel.candidate}`}
}
function lms(){
  const w=$('q').value.toLowerCase().split(/\s+/).filter(Boolean);
  $('lms').innerHTML=st.landmarks.filter(l=>w.every(x=>l.name.toLowerCase().includes(x))).slice(0,150)
    .map(l=>`<div class="item" data-n="${esc(l.name)}"><span>${esc(l.name)} <span class="k">${esc(l.kind)}</span></span><span class="k">${l.dist_m} m</span></div>`).join('');
}
$('q').oninput=lms;
$('lms').onclick=async e=>{const d=e.target.closest('[data-n]');if(d&&await post({action:'centre',on:'landmark',name:d.dataset.n}))refresh()};
$('cands').onclick=async e=>{const d=e.target.closest('[data-c]');if(d&&await post({action:'candidate',id:+d.dataset.c}))refresh()};
$('trace').addEventListener('toggle',e=>{if(e.target.open)opened.add(+e.target.dataset.i)},true);
$('trace').onclick=e=>{if(e.target.tagName==='IMG'){$('big').firstElementChild.src=e.target.src;$('big').style.display='flex'}};
$('big').onclick=()=>{$('big').style.display='none'};
$('frame').onclick=async e=>{const r=e.target.getBoundingClientRect();if(await post({action:'pick',u:(e.clientX-r.left)/r.width,v:(e.clientY-r.top)/r.height}))refresh()};
const acts={zin:{action:'zoom',step:-1},zout:{action:'zoom',step:1},cmaps:{action:'centre',on:'maps'},csel:{action:'centre',on:'selected'},clear:{action:'clear'}};
document.querySelectorAll('[data-a]').forEach(b=>b.onclick=async()=>{if(await post(acts[b.dataset.a]))refresh()});
async function decide(kind){if(await post({action:kind,opened:[...opened]}))refresh(true)}
$('confirm').onclick=()=>decide('confirm');
$('replace').onclick=()=>decide('replace');
document.addEventListener('keydown',async e=>{
  if(e.target.tagName==='INPUT')return;
  const k={'+':'zin','=':'zin','-':'zout',m:'cmaps',c:'csel',Escape:'clear'}[e.key];
  if(k){e.preventDefault();if(await post(acts[k]))refresh()}
  else if(e.key==='Enter'){e.preventDefault();decide('confirm')}
  else if(e.key==='s'&&!$('replace').disabled){e.preventDefault();decide('replace')}
});
refresh(true);
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
