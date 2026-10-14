"""The seam a baseline fills, and everything both baselines share.

:func:`fly` is what ``benchmarks.citynav.run_benchmark.episode_runner`` hands
the built episode to, in place of the mission conversation, when the config
names a ``policy``. It builds the harness a policy computes against, calls
the policy and returns the stops it flew, the commit last, in the shape of
the mission's flights. The runner replays them through the same six-action
synthesis and scores them through the same row writer, so a baseline row and
a MAPS row differ only in what chose the stops.

A policy is a module ``baselines.<name>.policy`` exposing
``run_episode(harness) -> BaselineResult``. It teleports freely while it
computes (``harness.observe`` at any pose is one ortho crop) and prices each
teleport against the action budget with ``harness.budget``; the replay is
the scored truth, so a policy that stops issuing moves at the budget is
never scored on a pose the replay could not reach.
"""

from __future__ import annotations

import difflib
import importlib
import io
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from benchmarks.citynav import paths as bench_paths
from benchmarks.citynav.action_snap import Pose, snap_to_actions
from maps.llm.vision_content import build_vision_user_content
from maps.perception.geometry.geolocate import TileFrame
from maps.perception.observation import Observation, crop_observation

#: The request-side thinking dial each policy sends on every call. ``None``
#: sends nothing, so the call rides the server default the headline arm flew:
#: the controlled comparison holds the model AND its reasoning rung fixed.
#: ``geonav_released`` is GeoNav as shipped, whose DashScope client disabled
#: thinking; it is the arm that says how much of the gap a reasoning backbone
#: closes for that method.
REASONING_EFFORT: dict[str, str | None] = {"geonav": None, "geonav_released": "none", "visa": None}

#: Each paper's sampling temperature. GeoNav's code sends none, so it rode its
#: server's default, 1.0 here; ViSA states 0.6.
TEMPERATURE: dict[str, float] = {"geonav": 1.0, "geonav_released": 1.0, "visa": 0.6}



@dataclass
class BaselineResult:
    """What a policy hands back: the stops it flew, and the commit."""

    # Visited poses in order, start excluded: ``((x, y), altitude_m)``.
    stops: list[tuple[tuple[float, float], float]]
    # The committed target's world xy; ``None`` commits the last stop.
    commit_xy: tuple[float, float] | None
    # A label for the commit, carried as ``commit_fid`` (a mark id, a graph
    # node id, ...). Never a CityRefer id.
    commit_label: str
    # The ``commit_path`` tier: ``"<policy>:<mechanism>"``, one string per
    # mechanism, so a row names how its commit was made.
    commit_tier: str
    # The policy's own step index at which it committed, if it did.
    lock_step: int | None = None
    # Extra per-episode numbers, landed under ``metrics.<policy>_<key>``.
    extra_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Landmark:
    """A named OSM prior row: the landmark prior both papers read."""

    name: str
    contour: list[tuple[float, float]]
    centroid: tuple[float, float]
    tags: dict


class Budget:
    """Prices teleports in the six-action space.

    ``move`` commits a leg from the current pose: the actions
    :func:`snap_to_actions` needs to reach the target, the pose it actually
    reaches. ``price`` asks without committing. The replay flies the same legs
    from the same start with the same snap, so the two agree up to the
    replay's commit-leg reserve.
    """

    def __init__(self, max_steps: int, start: Pose) -> None:
        self.max_steps = int(max_steps)
        self.pose = start
        self.used = 0

    @property
    def remaining(self) -> int:
        return self.max_steps - self.used

    def _end(self, xy: tuple[float, float], altitude_m: float | None) -> Pose:
        return Pose(
            float(xy[0]),
            float(xy[1]),
            float(altitude_m) if altitude_m is not None else self.pose.altitude_m,
            self.pose.heading_deg,
        )

    def price(self, xy: tuple[float, float], altitude_m: float | None = None) -> int:
        actions, _ = snap_to_actions(self.pose, self._end(xy, altitude_m))
        return len(actions)

    def affords(self, xy: tuple[float, float], altitude_m: float | None = None) -> bool:
        return self.price(xy, altitude_m) <= self.remaining

    def move(self, xy: tuple[float, float], altitude_m: float | None = None) -> int:
        actions, reached = snap_to_actions(self.pose, self._end(xy, altitude_m))
        self.used += len(actions)
        self.pose = reached
        return len(actions)


class BaselineHarness:
    """The substrate, held fixed: one object a policy computes against."""

    def __init__(
        self,
        *,
        cfg,
        episode,
        assets,
        start_alt_m: float,
        ground_z_m: float | None,
        workdir: Path,
        policy: str,
        sensor=None,
        perception=None,
    ) -> None:
        self.cfg = cfg
        # The run's own sensor, block geometry and SAM client: the frames a
        # policy reads and the detections it gets are the mission's.
        self.sensor = sensor
        self.episode = episode
        self.assets = assets
        self.start_alt_m = float(start_alt_m)
        self.ground_z_m = float(ground_z_m if ground_z_m is not None else 0.0)
        self.workdir = Path(workdir)
        self.policy = policy
        self.log_dir = self.workdir / policy
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tile_frame = TileFrame(
            x_min=float(assets.bounds.x_min),
            y_min=float(assets.bounds.y_min),
            cell_size_m=float(assets.cell_size_m),
            grid_shape=tuple(assets.grid_shape),
        )
        self.budget = Budget(
            cfg.max_steps,
            Pose(float(episode.start_xy[0]), float(episode.start_xy[1]), self.start_alt_m, 0.0),
        )
        # Wall time per phase, landed on the row as ``<policy>_phase_<name>_s``.
        self.phase_s: dict[str, float] = {"render": 0.0, "segment": 0.0, "examine": 0.0}
        self.llm_calls = 0
        self.reasoning_tokens = 0
        self.extraction_calls = 0
        self.reasoning_truncated = 0
        self.calls: list[dict] = []
        self._sam = perception
        self._landmarks: list[Landmark] | None = None

    # -- the sensor ---------------------------------------------------------

    def observe(self, pose_xy: tuple[float, float], altitude_m: float | None = None) -> Observation:
        """The mission's frame at this pose: the run's sensor, north-up,
        rendered by the same call the mission makes."""
        t0 = time.perf_counter()
        obs = crop_observation(
            (float(pose_xy[0]), float(pose_xy[1])),
            altitude_m=float(altitude_m if altitude_m is not None else self.start_alt_m),
            yaw_deg=0.0,
            sensor=self.sensor,
            # Read at call time: benchmarks.calcity.use() selects the Cal City rasters.
            rgbd_dir=bench_paths.CITYNAV_NATIVE_RGBD_DIR,
            map_name=self.episode.map_name,
        )
        obs.tile = self.tile_frame
        self.phase_s["render"] += time.perf_counter() - t0
        return obs

    def view_half_m(self, altitude_m: float | None = None) -> float:
        """Half the ground footprint across the frame's columns: height
        above ground times ``tan(hfov / 2)``. Under CityNav's 90-degree
        sensor this is ``altitude - ground``, upstream's view square."""
        return self._half(altitude_m, self._fov_deg()[0])

    def view_half_short_m(self, altitude_m: float | None = None) -> float:
        """Half the footprint's SHORT side, the square a frame surely covers."""
        return self._half(altitude_m, min(self._fov_deg()))

    def _fov_deg(self) -> tuple[float, float]:
        intr = getattr(self.sensor, "intrinsics", None)
        if intr is None:
            return (90.0, 90.0)
        return (float(intr.hfov_deg), float(intr.vfov_deg))

    def _half(self, altitude_m: float | None, fov_deg: float) -> float:
        alt = float(altitude_m if altitude_m is not None else self.start_alt_m)
        # Exactly 1 at 90 degrees, where math.tan(pi / 4) is one ulp short.
        t = 1.0 if fov_deg == 90.0 else math.tan(math.radians(fov_deg) / 2.0)
        return max(1.0, (alt - self.ground_z_m) * t)

    def search_window(self, landmarks: list["Landmark"], *, clip_to_block: bool = True) -> SimpleNamespace:
        """The square a policy plans and maps over: centred on the start, a
        half-side reaching the farthest matched landmark plus a margin, held
        to ``[WINDOW_MIN_HALF_M, WINDOW_MAX_HALF_M]``. Every CityNav block is
        narrower than the minimum, so there the window covers the block; the
        13 km Cal City block would otherwise size a policy's arrays past
        16 GB. ``clip_to_block`` intersects it with the block's bounds."""
        sx, sy = (float(v) for v in self.episode.start_xy)
        reach = 0.0
        for lm in landmarks:
            for x, y in lm.contour:
                reach = max(reach, abs(float(x) - sx), abs(float(y) - sy))
        half = min(max(reach + WINDOW_MARGIN_M, WINDOW_MIN_HALF_M), WINDOW_MAX_HALF_M)
        x0, y0, x1, y1 = sx - half, sy - half, sx + half, sy + half
        if clip_to_block:
            b = self.assets.bounds
            x0, y0 = max(x0, float(b.x_min)), max(y0, float(b.y_min))
            x1, y1 = min(x1, float(b.x_max)), min(y1, float(b.y_max))
        return SimpleNamespace(x_min=x0, y_min=y0, x_max=x1, y_max=y1)

    def pixel_to_world(self, obs: Observation, row: float, col: float) -> tuple[float, float]:
        """A frame pixel's ground point, through the crop's own corners."""
        h, w = obs.rgb.shape[:2]
        u = float(col) / max(w - 1, 1)
        v = float(row) / max(h - 1, 1)
        (flx, fly), (frx, fry), _br, (blx, bly) = obs.ortho_view_corners_xy
        return (flx + u * (frx - flx) + v * (blx - flx), fly + u * (fry - fly) + v * (bly - fly))

    # -- the detector -------------------------------------------------------

    def detect(self, obs: Observation, concepts: list[str]) -> list:
        """SAM 3.1 over the frame for these open-vocabulary concepts."""
        if self._sam is None or not concepts:
            return []
        t0 = time.perf_counter()
        self._sam.concept_provider.set_concepts(concepts)
        out = self._sam.detect(obs)
        self.phase_s["segment"] += time.perf_counter() - t0
        return out

    # -- the prior ----------------------------------------------------------

    def prior_rows(self) -> list[dict]:
        """Every OSM row of the block, flat, each carrying its bake category
        (``roads``, ``buildings``, ``pois``, ...) under ``category``."""
        out: list[dict] = []
        for category, rows in (self.assets.base_features or {}).items():
            if not isinstance(rows, list):
                continue
            for r in rows:
                if isinstance(r, dict):
                    out.append({**r, "category": category})
        return out

    def landmarks(self) -> list[Landmark]:
        """The landmark prior: one landmark per OSM ``name``.

        A street is a polyline and a bus stop is a point, so each row's
        geometry is widened to an area first (a road to its carriageway from
        its OSM width, lanes or class; a point by ten metres); rows sharing a name (the segments of one
        street) are unioned and the largest piece stands for the name, the
        way GeoNav's own loader keeps the largest object per name.
        """
        if self._landmarks is None:
            self._landmarks = _landmarks_from_rows(self.prior_rows())
        return list(self._landmarks)

    # -- the model ----------------------------------------------------------

    def jpeg(self, rgb: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(
            buf, format="JPEG", quality=JPEG_QUALITY
        )
        return buf.getvalue()

    def user_message(self, text: str, images: list[np.ndarray | bytes] | None = None) -> dict:
        """One user turn: the text, then each image as a JPEG data URI."""
        jpegs = [im if isinstance(im, bytes) else self.jpeg(im) for im in (images or [])]
        return {"role": "user", "content": build_vision_user_content(text, jpegs)}

    def llm_json(self, messages: list[dict], schema: dict, schema_name: str, *, max_tokens: int | None = None) -> dict:
        """One JSON answer at the policy's temperature and thinking dial.

        The first call carries no output grammar: on this serve a grammar
        applies from the first token and the reasoning span never opens, so a
        constrained call would run every baseline without thinking whatever
        the dial says. The model reasons, then answers, and the answer is
        parsed as JSON and checked against the schema (``_fits``). An answer
        that is not that JSON goes through a second, grammar-constrained call
        with the dial off that only reshapes it. A first call whose cap the
        reasoning exhausted (no answer) falls back to the constrained call at
        the policy's dial and is counted under ``reasoning_truncated``.
        ``LLM_ATTEMPTS`` tries, then ``RuntimeError``.
        """
        from maps.llm import endpoint_spec
        from maps.llm.flight_summary import reasoning_tokens_of
        from maps.llm.http import _chat_completions_raw
        from maps.llm.model_cards import sampling_kwargs, speaks_openai
        from maps.llm.openai_wire import OUTPUT_CAP_KEY

        effort = REASONING_EFFORT.get(self.policy)
        spec = endpoint_spec.spec()
        model = spec.model_id
        cap_key = OUTPUT_CAP_KEY if speaks_openai(model) else "max_tokens"
        api_key = os.environ.get("SEARCH_REASONER_API_KEY", "EMPTY")
        site = f"{self.policy}:{schema_name}"

        def call(msgs, *, dial, cap, response_format=None):
            kw = sampling_kwargs(model, dial)
            # The card's envelope at the paper's temperature; an OpenAI-wire
            # card sends none (its reasoning models take only 1.0).
            if "temperature" in kw:
                kw["temperature"] = TEMPERATURE.get(self.policy, 1.0)
            if response_format is not None:
                kw["response_format"] = response_format
            resp = _chat_completions_raw(
                spec.base_url, api_key, timeout=LLM_TIMEOUT_S, model=model, messages=msgs, **{cap_key: cap}, **kw,
            )
            choice = resp["choices"][0]
            self.llm_calls += 1
            usage = _Usage(
                completion_tokens=int((resp.get("usage") or {}).get("completion_tokens") or 0),
                reasoning_tokens=reasoning_tokens_of(resp, model=model),
            )
            self.reasoning_tokens += usage.reasoning_tokens
            return ((choice.get("message") or {}).get("content") or ""), choice.get("finish_reason"), usage

        def shaped(text):
            parsed = _extract_json(text)
            return parsed if _fits(parsed, schema) else None

        strict = {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema, "strict": True}}
        t0 = time.perf_counter()
        last: Exception | None = None
        try:
            for _ in range(LLM_ATTEMPTS):
                try:
                    text, finish, usage = call(
                        _with_answer_shape(messages, schema), dial=effort, cap=max_tokens or spec.max_output_tokens,
                    )
                    parsed, mode, more_tokens = shaped(text), "free", 0
                    if parsed is None and text.strip():
                        # The answer exists but is not the schema's JSON: reshape it.
                        again, _, usage2 = call(
                            [{"role": "user", "content": EXTRACTION_PROMPT.format(answer=text[:12000])}],
                            dial="none", cap=EXTRACTION_MAX_TOKENS, response_format=strict,
                        )
                        parsed, mode, more_tokens = shaped(again), "free+extract", usage2.completion_tokens
                    elif parsed is None:
                        # No answer at all: the reasoning span ate the cap.
                        self.reasoning_truncated += 1
                        again, _, usage2 = call(messages, dial=effort, cap=EXTRACTION_MAX_TOKENS, response_format=strict)
                        parsed, mode, more_tokens = shaped(again), "truncated+constrained", usage2.completion_tokens
                    if parsed is None:
                        raise ValueError(f"{site}: no schema JSON after the {mode} call")
                    if mode == "free+extract":
                        self.extraction_calls += 1
                    self.calls.append({"site": site, "mode": mode, "finish": finish,
                                       "reasoning_tokens": usage.reasoning_tokens,
                                       "completion_tokens": usage.completion_tokens + more_tokens})
                    return parsed
                except ValueError as exc:
                    last = exc
            raise RuntimeError(f"{site}: no JSON after {LLM_ATTEMPTS} attempts") from last
        finally:
            self.phase_s["examine"] += time.perf_counter() - t0

    # -- the record ---------------------------------------------------------

    def record(self, name: str, payload: Any) -> Path:
        """Land one JSON artefact under ``<workdir>/<policy>/``."""
        p = self.log_dir / f"{name}.json"
        p.write_text(json.dumps(payload, indent=1, default=_jsonable))
        return p

    def save_image(self, name: str, rgb: np.ndarray | bytes) -> Path:
        p = self.log_dir / (f"{name}.jpg" if isinstance(rgb, bytes) else f"{name}.png")
        if isinstance(rgb, bytes):
            p.write_bytes(rgb)
        else:
            Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(p)
        return p


@dataclass(frozen=True)
class _Usage:
    """One call's completion tokens and the reasoning slice of them, read by
    ``maps.llm.flight_summary.reasoning_tokens_of``."""

    completion_tokens: int
    reasoning_tokens: int


# A point feature (a bus stop, a shop node) is given a 10 m disc so "beside
# the Halifax" has a boundary.
_POINT_RADIUS_M = 10.0

# A street's half-width when its OSM row states none: the paved surface by
# highway class, kerb to kerb with parking, since CityRefer's road landmarks
# (the polygons both papers read) are the paved surface from the point cloud
# and a car "on Willmore Road" lies inside them. A ``width`` tag wins, then
# ``lanes`` at 3.25 m each.
_ROAD_HALF_WIDTH_BY_CLASS_M = {
    "motorway": 8.0, "trunk": 7.0, "primary": 6.5, "secondary": 6.0, "tertiary": 5.5,
    "residential": 5.0, "unclassified": 5.0, "living_street": 5.0, "service": 3.5,
    "footway": 2.0, "path": 2.0, "cycleway": 2.0, "pedestrian": 3.0, "steps": 1.5, "track": 2.5,
}
_ROAD_HALF_WIDTH_DEFAULT_M = 4.5


def _road_half_width_m(row: dict) -> float:
    w = str(row.get("width") or "").strip().lower().replace("m", "").strip()
    try:
        if w:
            return max(1.0, float(w) / 2.0)
    except ValueError:
        pass
    by_class = _ROAD_HALF_WIDTH_BY_CLASS_M.get(str(row.get("highway") or "").split("_link")[0], _ROAD_HALF_WIDTH_DEFAULT_M)
    try:
        lanes = int(str(row.get("lanes") or "").strip())
        return max(by_class, lanes * 3.25 / 2.0)
    except ValueError:
        return by_class


def _row_area(row: dict):
    """The row's footprint as a shapely polygon, or ``None``."""
    from shapely.geometry import LineString, Point, Polygon

    pts = [(float(p[0]), float(p[1])) for p in (row.get("world_contour") or []) if len(p) >= 2]
    if not pts:
        return None
    is_line = bool(row.get("highway")) or row.get("category") == "roads"
    try:
        if len(pts) == 1:
            geom = Point(pts[0]).buffer(_POINT_RADIUS_M)
        elif len(pts) == 2 or is_line:
            geom = LineString(pts).buffer(_road_half_width_m(row) if is_line else _POINT_RADIUS_M)
        else:
            geom = Polygon(pts).buffer(0)
            if geom.is_empty:
                geom = LineString(pts).buffer(_road_half_width_m(row))
    except (ValueError, TypeError):
        return None
    return None if geom.is_empty else geom


def _landmarks_from_rows(rows: list[dict]) -> list[Landmark]:
    from shapely.geometry import MultiPolygon
    from shapely.ops import unary_union

    by_name: dict[str, list[tuple[dict, object]]] = {}
    for r in rows:
        name = str(r.get("name") or "").strip()
        if not name:
            continue
        geom = _row_area(r)
        if geom is None:
            continue
        by_name.setdefault(name, []).append((r, geom))
    out: list[Landmark] = []
    for name, parts in by_name.items():
        union = unary_union([g for _r, g in parts])
        if isinstance(union, MultiPolygon):
            union = max(union.geoms, key=lambda g: g.area)
        if union.is_empty or not hasattr(union, "exterior"):
            continue
        ring = [(float(x), float(y)) for x, y in union.exterior.coords]
        # The tags of the largest contributing row: category, highway, ...
        row = max(parts, key=lambda rg: rg[1].area)[0]
        tags = {k: v for k, v in row.items() if k not in ("cells", "world_contour")}
        out.append(Landmark(name=name, contour=ring, centroid=(float(union.centroid.x), float(union.centroid.y)), tags=tags))
    out.sort(key=lambda lm: lm.name)
    return out


#: The search window's half-side bounds and its margin past the farthest
#: matched landmark (``BaselineHarness.search_window``). The minimum exceeds
#: every CityNav block's extent (at most ~414 m), so a CityNav window covers
#: its block from any start. At the maximum, GeoNav's map render peaks at
#: ~1.6 GB per episode and ViSA's coverage plan at ~0.7 GB; twice the half-side
#: put GeoNav at ~4.2 GB.
#: ponytail: a start-centred square, so a Cal City dashboard row that starts
#: more than ~1.3 km from its target plans over ground that may not hold it;
#: the upgrade is a window spanning the start and the matched landmarks, with
#: GeoNav's map resolution scaled to hold its pixel count.
WINDOW_MIN_HALF_M = 512.0
WINDOW_MAX_HALF_M = 1280.0
WINDOW_MARGIN_M = 200.0

#: The JPEG quality every image a policy sends is encoded at: the quality the
#: mission's crops are encoded at (``maps.perception.viz.mllm_overlay``).
JPEG_QUALITY = 85

#: The reshaping call only restates an answer; it never reasons.
EXTRACTION_MAX_TOKENS = 4096

#: Each model call's HTTP timeout, and the tries ``llm_json`` makes at an answer.
LLM_TIMEOUT_S = 1800.0
LLM_ATTEMPTS = 2

EXTRACTION_PROMPT = (
    "Restate the answer below as one JSON object matching the schema you are "
    "given. Keep every value the answer states; invent nothing; where the "
    "answer names no value for a key, use an empty string, an empty list, 0 "
    "or false.\n\nAnswer:\n{answer}"
)


def _with_answer_shape(messages: list[dict], schema: dict) -> list[dict]:
    """The messages plus, on the last user turn, the JSON shape to answer in."""
    keys = ", ".join(schema.get("required") or [])
    shape = (
        f"\n\nAfter your reasoning, answer with ONE JSON object and nothing after it, "
        f"with exactly these keys: {keys}. Schema: {json.dumps(schema, separators=(',', ':'))}"
    )
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            m["content"] = c + shape
        elif isinstance(c, list):
            m["content"] = list(c) + [{"type": "text", "text": shape}]
        break
    return out


def _fits(value: Any, schema: dict) -> bool:
    """``value`` has the schema's shape: an enum holds it, an object carries
    every required key and each property fits, an array's items each fit."""
    if "enum" in schema:
        return value in schema["enum"]
    kind = schema.get("type")
    if kind == "object":
        return (isinstance(value, dict) and all(k in value for k in schema.get("required") or ())
                and all(_fits(value[k], sub) for k, sub in (schema.get("properties") or {}).items() if k in value))
    if kind == "array":
        return isinstance(value, list) and all(_fits(v, schema.get("items") or {}) for v in value)
    return True


def _extract_json(text: str) -> dict | None:
    """The first JSON object in ``text``: a fenced block, else the first
    ``{`` that decodes. Prose before or after it is ignored."""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [m.group(1)] if m else []
    candidates.append(text)
    dec = json.JSONDecoder()
    for cand in candidates:
        start = 0
        for _ in range(64):
            k = cand.find("{", start)
            if k < 0:
                break
            try:
                obj, _end = dec.raw_decode(cand[k:])
            except json.JSONDecodeError:
                start = k + 1
                continue
            if isinstance(obj, dict):
                return obj
            start = k + 1
    return None


# Words a landmark name shares with every other landmark of its kind. A match
# is made on what is left, so "Helmore building" reaches Helmore Building and
# never Mumford Building.
_GENERIC_NAME_TOKENS = frozenset(
    "the a an of and at on in building buildings centre center centres street streets st "
    "road roads rd lane lanes ln avenue ave drive dr way park parks house houses hall halls "
    "court courts close place places square squares hotel school college church station "
    "depot car lot parking garden gardens grove terrace small large big new old".split()
)


def _distinctive(name: str) -> tuple[list[str], bool]:
    """The name's non-generic tokens, and whether any were left. A name that
    is all generic ("Hall") falls back to its tokens and is flagged, so it is
    matched only on the whole name."""
    tokens = re.findall(r"[a-z0-9']+", str(name).lower())
    kept = [t for t in tokens if t not in _GENERIC_NAME_TOKENS]
    return (kept, True) if kept else (tokens, False)


def match_landmarks(names: list[str], landmarks: list[Landmark], *, floor: float = 0.6) -> list[Landmark]:
    """The instruction's named places, each to the OSM landmark it names.

    Both papers took their landmark strings from a curated file and matched
    on the full name; here the strings come from an extraction call, so the
    match is on the distinctive tokens (``difflib`` ratio, a shared
    distinctive token counting as 0.8) with a floor below which a place is
    honestly unmatched.
    """
    out: list[Landmark] = []
    seen: set[str] = set()
    for q in names:
        qd, _ = _distinctive(q)
        if not qd:
            continue
        # Ranked by the distinctive-token ratio, ties broken on the whole
        # name: "Grafton Street" scores 1.0 against both Grafton Centre and
        # Grafton Street on the distinctive token, and only the whole name
        # says which one was named.
        best, best_key = None, (0.0, 0.0)
        for lm in landmarks:
            ld, ld_distinct = _distinctive(lm.name)
            full = difflib.SequenceMatcher(None, str(q).lower(), lm.name.lower()).ratio()
            if not ld_distinct:
                # An all-generic landmark name is matched on the whole name only.
                r = full if full >= 0.9 else 0.0
            else:
                r = difflib.SequenceMatcher(None, " ".join(qd), " ".join(ld)).ratio()
                if {t for t in qd if len(t) >= 3} & {t for t in ld if len(t) >= 3}:
                    r = max(r, 0.8)
            if (r, full) > best_key:
                best, best_key = lm, (r, full)
        if best is not None and best_key[0] >= floor and best.name not in seen:
            out.append(best)
            seen.add(best.name)
    return out


def landmark_distance(lm: Landmark, xy: tuple[float, float]) -> float:
    """Metres from ``xy`` to the landmark's area, 0 inside it. A street's
    centroid can sit hundreds of metres from every point of the street, so
    distance to the area is the only reading that means "at the landmark"."""
    from shapely.geometry import Point, Polygon

    try:
        poly = Polygon(lm.contour).buffer(0)
        if not poly.is_empty:
            return float(poly.distance(Point(float(xy[0]), float(xy[1]))))
    except (ValueError, TypeError):
        pass
    return math.hypot(lm.centroid[0] - xy[0], lm.centroid[1] - xy[1])


def pixel_bbox(det) -> tuple[int, int, int, int] | None:
    """A detection's inclusive ``(row_min, col_min, row_max, col_max)`` in
    frame pixels: its mask's box, else the box the SAM client stamped."""
    mask = getattr(det, "mask", None)
    if mask is not None:
        return tuple(int(v) for v in mask.bbox)
    bb = (getattr(det, "metadata", None) or {}).get("pixel_bbox")
    return tuple(int(v) for v in bb) if bb else None


def _centroid(pts: list[tuple[float, float]]) -> tuple[float, float]:
    try:
        from shapely.geometry import Polygon

        c = Polygon(pts).centroid
        if not c.is_empty:
            return (float(c.x), float(c.y))
    except Exception:  # noqa: BLE001 -- a degenerate ring falls back to the mean
        pass
    return (float(sum(p[0] for p in pts) / len(pts)), float(sum(p[1] for p in pts) / len(pts)))


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)


@dataclass
class BaselineFlight:
    """What the runner reads back from one baseline episode."""

    # The stops in the shape of ``MissionSession.flights``, the commit last:
    # the replay's input.
    flights: list[dict]
    # Row fields: ``policy``, ``commit_path``, ``committed_fid`` and the
    # policy's own numbers as ``<policy>_<key>``.
    metrics: dict[str, Any]


def fly(h, policy: str) -> BaselineFlight:
    """Fly one built episode (``episode_runner.Hydrated``) with ``policy``.

    The policy starts at the runner's start pose and altitude, sees through
    the run's sensor and segments with the run's SAM client. Its stops and
    commit come back as flights for the runner's replay, and its record lands
    under ``<workdir>/<policy>/``.
    """
    workdir = Path(os.environ.get("SEARCH_COMPILE_WORKDIR") or ".")
    start_xy = (float(h.episode.start_xy[0]), float(h.episode.start_xy[1]))
    harness = BaselineHarness(
        cfg=h.cfg,
        episode=h.episode,
        assets=h.assets,
        start_alt_m=h.state.altitude_m,
        ground_z_m=h.search_loop._ortho_ground_z_m,
        workdir=workdir,
        policy=policy,
        sensor=h.sensor,
        perception=h.perception,
    )
    module = importlib.import_module(f"baselines.{policy}.policy")
    t0 = time.monotonic()
    result: BaselineResult = module.run_episode(harness)
    policy_s = time.monotonic() - t0

    stops = [((float(xy[0]), float(xy[1])), float(alt)) for xy, alt in result.stops]
    commit_xy = result.commit_xy
    if commit_xy is None:
        commit_xy = stops[-1][0] if stops else start_xy
    commit_xy = (float(commit_xy[0]), float(commit_xy[1]))
    # The commit leg flies at the altitude the last stop left the aircraft at.
    commit_alt = stops[-1][1] if stops else harness.start_alt_m
    flights = [
        {"target": f"{policy}_stop_{i}", "resolved_xy": [xy[0], xy[1]], "provenance": "point", "altitude_m": alt}
        for i, (xy, alt) in enumerate(stops)
    ] + [{"target": result.commit_label, "resolved_xy": [commit_xy[0], commit_xy[1]],
          "provenance": "point", "altitude_m": commit_alt}]

    harness.record("decision", {
        "commit_path": result.commit_tier,
        "target_fid": result.commit_label,
        "world_xy_resolved": [commit_xy[0], commit_xy[1]],
        "lock_step": result.lock_step,
        "stops": [[xy[0], xy[1], alt] for xy, alt in stops],
        "policy_actions_priced": harness.budget.used,
        "llm_calls": harness.llm_calls,
    })
    harness.record("calls", harness.calls)

    own: dict[str, Any] = {
        "actions_priced": harness.budget.used,
        "llm_calls": harness.llm_calls,
        "reasoning_tokens": harness.reasoning_tokens,
        "extraction_calls": harness.extraction_calls,
        "reasoning_truncated": harness.reasoning_truncated,
        "stops": len(stops),
        "lock_step": result.lock_step,
        "policy_s": round(policy_s, 2),
        **{f"phase_{k}_s": round(v, 2) for k, v in harness.phase_s.items()},
        **(result.extra_metrics or {}),
    }
    return BaselineFlight(
        flights=flights,
        metrics={
            "policy": policy,
            "commit_path": result.commit_tier,
            "committed_fid": result.commit_label,
            **{f"{policy}_{k}": v for k, v in own.items()},
        },
    )


__all__ = [
    "landmark_distance",
    "match_landmarks",
    "pixel_bbox",
    "REASONING_EFFORT",
    "TEMPERATURE",
    "BaselineHarness",
    "BaselineFlight",
    "BaselineResult",
    "Budget",
    "Landmark",
    "fly",
]
