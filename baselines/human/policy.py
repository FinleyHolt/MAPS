"""The operator console: one local web page per episode, served until commit.

The bench flies each episode in a child process with no stdin, so the
console is a page at ``http://<HUMAN_CONSOLE_HOST>:<HUMAN_CONSOLE_PORT>/``
(default ``127.0.0.1:8077``). It is served one request at a time on the
policy's own thread, so the harness is never touched concurrently. No
detector runs: the operator reads the pixels.
"""

from __future__ import annotations

import io
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from baselines.seam import BaselineHarness, BaselineResult

DEFAULT_PORT = 8077
#: The lowest stop above ground, as ViSA flies it.
MIN_AGL_M = 20.0
#: The outline and label colour of the landmark prior on the frame, and of
#: the one landmark the operator selected from the search.
PRIOR_RGB = (255, 214, 0)
SELECTED_RGB = (255, 0, 255)
#: Labels on the frame: large enough to read at a glance, outlined so they read on any roof.
LABEL = {"font": ImageFont.load_default(size=16), "stroke_width": 2, "stroke_fill": (0, 0, 0)}


def world_to_px(corners, shape, xy) -> tuple[float, float]:
    """``(col, row)`` of a world point: the inverse of
    ``geolocate.frame_world_xy``'s affine map through ``[FL, FR, BR, BL]``."""
    h, w = shape
    fl, fr, bl = (np.asarray(corners[i], dtype=float) for i in (0, 1, 3))
    u, v = np.linalg.solve(np.column_stack([fr - fl, bl - fl]), np.asarray(xy, dtype=float) - fl)
    return float(u * (w - 1)), float(v * (h - 1))


def prior_frame(h: BaselineHarness, obs, selected: str | None = None) -> bytes:
    """The frame as JPEG with every named landmark that reaches into the
    view outlined and labelled, the ``selected`` one heavier."""
    img = Image.fromarray(np.asarray(obs.rgb, dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = obs.world_window
    shape = obs.rgb.shape[:2]
    for lm in h.landmarks():
        xs, ys = zip(*lm.contour)
        if max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1:
            continue
        pts = [world_to_px(obs.ortho_view_corners_xy, shape, p) for p in lm.contour]
        rgb, width = (SELECTED_RGB, 5) if lm.name == selected else (PRIOR_RGB, 2)
        draw.line(pts + pts[:1], fill=rgb, width=width)
        c, r = world_to_px(obs.ortho_view_corners_xy, shape, lm.centroid)
        draw.text((c, r), lm.name, fill=rgb, **LABEL)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class Console:
    """The episode as the operator flies it."""

    def __init__(self, h: BaselineHarness) -> None:
        self.h = h
        self.xy = (float(h.episode.start_xy[0]), float(h.episode.start_xy[1]))
        self.alt = float(h.start_alt_m)
        self.alt_lo = float(h.ground_z_m) + MIN_AGL_M
        self.alt_hi = float(h.cfg.altitude_cap_m)
        self.stops: list[tuple[tuple[float, float], float]] = []
        self.log: list[dict] = []
        self.refused = 0
        self.commit_xy: tuple[float, float] | None = None
        self.done = False
        self.t0 = time.monotonic()
        self._look()

    def _look(self) -> None:
        self.obs = self.h.observe(self.xy, self.alt)
        self.frame = prior_frame(self.h, self.obs)

    def _point(self, u: float, v: float) -> tuple[float, float]:
        """The ground point under a click, as fractions of the frame."""
        rows, cols = self.obs.rgb.shape[:2]
        return self.h.pixel_to_world(self.obs, v * (rows - 1), u * (cols - 1))

    def state(self) -> dict:
        lms = sorted(self.h.landmarks(), key=lambda lm: np.hypot(lm.centroid[0] - self.xy[0], lm.centroid[1] - self.xy[1]))
        return {
            "instruction": self.h.episode.description,
            "xy": self.xy, "alt_m": self.alt, "agl_m": self.alt - float(self.h.ground_z_m),
            "alt_lo": self.alt_lo, "alt_hi": self.alt_hi,
            "remaining": self.h.budget.remaining, "max_steps": self.h.budget.max_steps,
            "stops": len(self.stops), "view_m": 2.0 * self.h.view_half_m(self.alt),
            "landmarks": [{"name": lm.name, "kind": str(lm.tags.get("category") or ""),
                           "x": lm.centroid[0], "y": lm.centroid[1],
                           "dx": round(lm.centroid[0] - self.xy[0]), "dy": round(lm.centroid[1] - self.xy[1])}
                          for lm in lms],
        }

    def act(self, req: dict) -> dict:
        """One operator action: ``move`` to a clicked point, a searched
        landmark's centroid (``x``, ``y``) or the current point at ``alt_m``,
        or ``commit`` one of those."""
        kind = req.get("action")
        if "u" in req:
            xy = self._point(float(req["u"]), float(req["v"]))
        elif "x" in req:
            xy = (float(req["x"]), float(req["y"]))
        else:
            xy = self.xy
        entry = {"t_s": round(time.monotonic() - self.t0, 2), "action": kind, "xy": xy}
        if kind == "move":
            alt = min(max(float(req.get("alt_m", self.alt)), self.alt_lo), self.alt_hi)
            entry["alt_m"] = alt
            if not self.h.budget.affords(xy, alt):
                self.refused += 1
                self.log.append({**entry, "refused": True})
                return {"ok": False, "error": "that move exceeds the action budget left"}
            self.h.budget.move(xy, alt)
            self.xy, self.alt = xy, alt
            self.stops.append((xy, alt))
            self._look()
            self.h.save_image(f"stop_{len(self.stops) - 1}", self.frame)
        elif kind == "commit":
            self.commit_xy, self.done = xy, True
        else:
            return {"ok": False, "error": f"unknown action {kind!r}"}
        self.log.append(entry)
        return {"ok": True}

    def result(self) -> BaselineResult:
        self.h.record("session", {"instruction": self.h.episode.description, "actions": self.log})
        return BaselineResult(
            stops=self.stops, commit_xy=self.commit_xy, commit_label="human_commit",
            commit_tier="human:operator", lock_step=len(self.stops),
            extra_metrics={"moves_refused": self.refused},
        )


def _handler(console: Console):
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
            elif path == "/frame.jpg":
                selected = parse_qs(urlsplit(self.path).query).get("selected", [None])[0]
                frame = prior_frame(console.h, console.obs, selected) if selected else console.frame
                self._send(frame, "image/jpeg")
            elif path == "/state":
                self._send(json.dumps(console.state()).encode(), "application/json")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                out = console.act(req)
            except (ValueError, KeyError, TypeError) as exc:
                out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self._send(json.dumps(out).encode(), "application/json", 200 if out["ok"] else 400)

        def log_message(self, *_args) -> None:
            pass

    return Handler


def run_episode(h: BaselineHarness) -> BaselineResult:
    console = Console(h)
    host = os.environ.get("HUMAN_CONSOLE_HOST", "127.0.0.1")
    port = int(os.environ.get("HUMAN_CONSOLE_PORT", DEFAULT_PORT))
    HTTPServer.allow_reuse_address = True
    with HTTPServer((host, port), _handler(console)) as server:
        print(f"human console: http://{host}:{server.server_port}/  {h.episode.episode_id}", flush=True)
        while not console.done:
            server.handle_request()
    return console.result()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Operator Console</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f6f4;--fg:#1b1b1b;--muted:#666;--line:#ccc;--accent:#0a5cc2}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--muted:#999;--line:#333;--accent:#6aa8ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
main{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px;padding:16px;max-width:1400px;margin:auto}
@media (max-width:800px){main{grid-template-columns:1fr}}
#instr{font-size:18px;font-weight:600;margin:0 0 8px}
#wrap{position:relative;line-height:0}
#frame{width:100%;cursor:crosshair;border:1px solid var(--line)}
#pick{position:absolute;width:14px;height:14px;margin:-8px 0 0 -8px;border:2px solid #f0f;border-radius:50%;display:none;pointer-events:none}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}
button{font:inherit;padding:6px 12px;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:4px;cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
input{font:inherit;width:80px}
#msg{color:#c33;min-height:1.4em}
#lms{max-height:60vh;overflow:auto;font-size:13px;border-top:1px solid var(--line)}
#lms div{padding:3px 2px;cursor:pointer}
#lms div:hover,#lms div.on{background:var(--line)}
#lms span{color:var(--muted)}
.k{color:var(--muted)}
</style></head><body><main>
<section>
<p id="instr"></p>
<div class="row k" id="stat"></div>
<div id="wrap"><img id="frame" alt="current view"><div id="pick"></div></div>
<div class="row">
<label>Altitude AGL (m) <input id="alt" type="number" step="5"></label>
<button id="fly" class="primary">Fly to point</button>
<button id="commit">Commit point</button>
<button id="clear">Clear point</button>
</div>
<div id="msg"></div>
<p class="k">Click the view to pick a point. Fly to point moves there at the altitude set; with no point picked it changes altitude only. Commit point ends the episode at the picked point, or at the current position with no point picked. North is up.</p>
</section>
<aside><b>Search the prior</b>
<input id="q" type="search" placeholder="a named place" style="width:100%;margin:6px 0">
<div class="k">metres east, north of you; select one to outline it</div>
<div id="sel" class="row" style="display:none"><button id="over" class="primary"></button><button id="unsel">Clear</button></div>
<div id="lms"></div></aside>
</main><script>
let pick=null,st=null,sel=null;
const $=id=>document.getElementById(id);
async function load(){
  st=await (await fetch('/state')).json();
  $('instr').textContent=st.instruction;
  $('stat').textContent=`budget ${st.remaining}/${st.max_steps} actions · stops ${st.stops} · AGL ${st.agl_m.toFixed(0)} m · view ${st.view_m.toFixed(0)} m across`;
  $('alt').value=st.agl_m.toFixed(0);
  clearPick();select(null);
}
const esc=t=>t.replace(/&/g,'&amp;').replace(/</g,'&lt;');
function list(){
  const words=$('q').value.toLowerCase().split(/\s+/).filter(Boolean);
  const hits=st.landmarks.filter(l=>words.every(w=>l.name.toLowerCase().includes(w))).slice(0,200);
  $('lms').innerHTML=hits.map(l=>`<div data-n="${esc(l.name)}" class="${sel&&sel.name===l.name?'on':''}">${esc(l.name)} <span>${esc(l.kind)} (${l.dx}, ${l.dy})</span></div>`).join('');
}
function select(l){
  sel=l;
  $('sel').style.display=l?'flex':'none';
  if(l)$('over').textContent='Fly over '+l.name;
  $('frame').src='/frame.jpg?'+(l?'selected='+encodeURIComponent(l.name)+'&':'')+Date.now();
  list();
}
$('q').oninput=list;
$('lms').onclick=e=>{const d=e.target.closest('[data-n]');if(d)select(st.landmarks.find(l=>l.name===d.dataset.n))};
$('unsel').onclick=()=>select(null);
$('over').onclick=async()=>{
  const agl=parseFloat($('alt').value);
  if(await post({action:'move',x:sel.x,y:sel.y,alt_m:st.alt_m-st.agl_m+agl}))load();
};
function clearPick(){pick=null;$('pick').style.display='none'}
$('frame').onclick=e=>{
  const r=e.target.getBoundingClientRect();
  pick={u:(e.clientX-r.left)/r.width,v:(e.clientY-r.top)/r.height};
  Object.assign($('pick').style,{left:(pick.u*100)+'%',top:(pick.v*100)+'%',display:'block'});
};
async function post(body){
  $('msg').textContent='';
  const r=await (await fetch('/',{method:'POST',body:JSON.stringify(body)})).json();
  if(!r.ok){$('msg').textContent=r.error;return false}
  return true;
}
$('fly').onclick=async()=>{
  const agl=parseFloat($('alt').value);
  if(await post({action:'move',...(pick||{}),alt_m:st.alt_m-st.agl_m+agl}))load();
};
$('commit').onclick=async()=>{
  if(await post({action:'commit',...(pick||{})})){document.body.innerHTML='<main><p>Committed. The next episode opens here once it loads.</p></main>';setTimeout(next,3000)}
};
$('clear').onclick=clearPick;
async function next(){try{await fetch('/state');location.reload()}catch(e){setTimeout(next,3000)}}
load();
</script></body></html>
"""
