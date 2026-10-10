"""``RunConfig`` -- the knob object for one bench run -- and its YAML loader.

A config file is either the control (``configs/citynav.yaml``, which declares
every key) or an override: ``base: <file>`` names the file it changes,
relative to itself, and the file holds only the keys it changes. Chains
resolve and deep-merge; ``_SCHEMA`` is the one statement of every key, its
type and its range, and the loader refuses an unknown key, a missing key or
a bad value at load, naming its dotted path and the file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from maps.perception.geometry.sensor_spec import resolve_sensor
from maps.perception.sam.detect_client import SamDetectClient

from benchmarks.citynav.flight_model import FlightModelConfig, known_flight_kinds

log = logging.getLogger("run_benchmark")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NUM = (int, float)
_NONE = type(None)


def _one_of(*allowed: object) -> tuple[Callable[[Any], bool], str]:
    return (lambda v: v in allowed), f"one of {list(allowed)}"


def _sensor_ok(v: Any) -> bool:
    resolve_sensor(v)  # raises, naming the problem
    return True


_POS = (lambda v: v > 0), "> 0"
_NONNEG = (lambda v: v >= 0), ">= 0"
_ANY = (lambda v: True), ""

# Every YAML key: (accepted types, (test, what the test asks)). A dict value
# is a nested block with its own keys. A float key reads an int too.
_SCHEMA: dict[str, Any] = {
    "perception": (str, _one_of("sam", "gt_perfect")),
    "max_steps": (int, _POS),
    "fov_m": (_NUM, _NONNEG),
    "altitude_m": (_NUM, _POS),
    "cell_size_m": (_NUM, _POS),
    "success_radius_m": (_NUM, _POS),
    "precise_tolerance_m": (_NUM, _NONNEG),
    "precise_floor_radius_m": (_NUM, _NONNEG),
    "spawn": (str, _one_of("episode", "uniform")),
    "spawn_seed": (int, _NONNEG),
    "start_altitude_band_m": (list, (
        lambda v: len(v) == 2 and all(isinstance(x, _NUM) and not isinstance(x, bool) for x in v)
        and 0 < v[0] <= v[1], "[lo, hi] with 0 < lo <= hi")),
    "start_altitude_agl_m": ((*_NUM, _NONE), ((lambda v: v is None or v > 0), "null or > 0")),
    "altitude_cap_m": (_NUM, _POS),
    "rgb_upscale_px": ((int, _NONE), ((lambda v: v is None or v > 0), "null or > 0")),
    "landing_altitude_m": (_NUM, _NONNEG),
    "descend_to_land": (bool, _ANY),
    "sensor": ((str, dict), (_sensor_ok, "a registered sensor or an inline override")),
    "flight_model": {
        "kind": (str, _one_of(*known_flight_kinds())),
        "camera_hz": (_NUM, _POS),
        "speed_mps": ((*_NUM, _NONE), ((lambda v: v is None or v > 0), "null or > 0")),
    },
    "sam": {
        "model_id": (str, _ANY),
        "color_palette": (str, _one_of("citynav", "default")),
        "score_threshold": (_NUM, ((lambda v: 0 <= v <= 1), "in [0, 1]")),
        "min_area_px": (int, _NONNEG),
        "max_per_step": (int, _POS),
        "sahi": (bool, _ANY),
        "sahi_overlap": (_NUM, ((lambda v: 0 <= v < 1), "in [0, 1)")),
        "sahi_nms_iou": (_NUM, ((lambda v: 0 < v <= 1), "in (0, 1]")),
    },
    "gt_perfect": {
        "color_table_path": ((str, _NONE), _ANY),
        "mask_px": (int, _POS),
        "min_blob_px": (int, _NONNEG),
        "max_per_step": (int, _NONNEG),
    },
    "compact": {
        "enable": (bool, _ANY),
    },
}


@dataclass
class RunConfig:
    """One resolved config: every ``_SCHEMA`` key, plus the run facts the
    CLI sets (``resume_*``) and the path the config resolved from."""

    perception: str
    max_steps: int
    fov_m: float
    altitude_m: float
    cell_size_m: float
    success_radius_m: float
    # How close to an annotated target SHAPE counts as arriving, for a
    # shape-scored SR (eval.score_precise); 0.0 is strict containment.
    precise_tolerance_m: float
    # A shape narrower than this is widened until a circle of this radius
    # fits inside it, so no episode is scored finer than the agent resolves.
    precise_floor_radius_m: float
    # "episode" flies the dataset's start pose; "uniform" a seeded uniform
    # point over the tile bounds, deterministic per (spawn_seed, episode_id).
    spawn: str
    spawn_seed: int
    # Upstream start-altitude band [lo, hi], sampled per episode (seeded by
    # episode_id); start_altitude_agl_m, when set, replaces it with
    # GROUND_LEVEL[map] + that value.
    start_altitude_band_m: list
    start_altitude_agl_m: float | None
    altitude_cap_m: float
    # The model-input frame size the native crop is resampled to; None
    # keeps the native span.
    rgb_upscale_px: int | None
    landing_altitude_m: float
    # CityNav SR is 2D, so the default traverses to the committed centroid
    # at cruise altitude; True descends to landing_altitude_m.
    descend_to_land: bool
    # A registered sensor name or an inline override (sensor_spec).
    sensor: str | dict
    flight_model: FlightModelConfig
    sam: dict
    gt_perfect: dict
    compact: dict
    # Universal resume from a prior run's workdir (CLI --resume-from-workdir)
    # and its turn (None picks the latest reconstructable one).
    resume_from_workdir: str | None = None
    resume_turn: int | None = None
    # The resolved top-level YAML path, written by ``load_config``.
    source_path: str | None = None


def _export_compact_env(compact: dict) -> None:
    """``compact.enable`` -> ``SEARCH_COMPACT_ENABLE``; a value already in the
    environment wins."""
    if os.environ.get("SEARCH_COMPACT_ENABLE") is None:
        os.environ["SEARCH_COMPACT_ENABLE"] = "1" if compact["enable"] else "0"


def _locate(path: str | Path | None) -> Path:
    if not path:
        raise RuntimeError("a --config YAML is required: pass --config configs/citynav.yaml.")
    candidates = [Path(path)]
    if not candidates[0].is_absolute():
        candidates.append(_REPO_ROOT / path)
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Config not found: tried {[str(p) for p in candidates]}")


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def _read_chain(p: Path, seen: tuple[Path, ...] = ()) -> dict:
    import yaml  # lazy -- only required for config runs

    p = p.resolve()
    if p in seen:
        raise ValueError(f"{p}: base chain loops ({' -> '.join(map(str, (*seen, p)))})")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: a config is a mapping of keys")
    base = data.pop("base", None)
    if base is None:
        return data
    if not isinstance(base, str):
        raise ValueError(f"{p}: base must name a config file")
    return _merge(_read_chain(p.parent / base, (*seen, p)), data)


def _check(data: dict, schema: dict, src: Path, prefix: str = "") -> None:
    unknown = sorted(set(data) - set(schema))
    missing = sorted(set(schema) - set(data))
    if unknown:
        raise ValueError(f"{src}: unknown config key(s) {[prefix + k for k in unknown]}")
    if missing:
        raise ValueError(f"{src}: missing config key(s) {[prefix + k for k in missing]}")
    for key, rule in schema.items():
        path, v = prefix + key, data[key]
        if isinstance(rule, dict):
            if not isinstance(v, dict):
                raise ValueError(f"{src}: {path} is a block of keys, got {v!r}")
            _check(v, rule, src, path + ".")
            continue
        types, (ok, says) = rule
        types = types if isinstance(types, tuple) else (types,)
        if not isinstance(v, types) or (isinstance(v, bool) and bool not in types):
            names = "/".join("null" if t is _NONE else t.__name__ for t in types)
            raise ValueError(f"{src}: {path} must be {names}, got {v!r}")
        if not ok(v):
            raise ValueError(f"{src}: {path} must be {says}, got {v!r}")


def resolve_config(path: str | Path | None) -> tuple[dict, Path]:
    """The checked, fully merged key dict of a config file, and its path."""
    p = _locate(path)
    data = _read_chain(p)
    _check(data, _SCHEMA, p)
    return data, p


def config_sha256(path: str | Path) -> str:
    """sha256 of the resolved config's canonical JSON: the identity of what a
    run flew, whichever base chain spelled it."""
    data, _ = resolve_config(path)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_config(path: str | Path | None) -> RunConfig:
    data, p = resolve_config(path)
    cfg = RunConfig(**{**data, "flight_model": FlightModelConfig(**data["flight_model"])})
    cfg.source_path = str(p)
    _assert_realstack_gt_isolation(cfg, p)
    return cfg


# The real-stack pivot GT-isolation runtime gate. TRIGGER CHOICE (revised
# 2026-07-04, blocker V1-R3/V4-G2/V5-F1): the check fires on the resolved
# config *CONTENT* — the perception tier — NOT on the config FILENAME. The
# earlier filename trigger (``citynav.yaml``) was scoped too
# narrowly: any DERIVED config (an A/B arm, a resume clone, a ``_lookoff``
# variant) inherits the real-SAM ``sam`` substrate but a filename
# gate skips it, so a carried-over ``color_table_path`` or a stray
# ``SEARCH_SKIP_VLM_PERCEPTION=1`` would fly GT-leaky with zero signal. Any
# config that resolves to the real-SAM tier now carries the guard. The
# gt_perfect DIAGNOSTIC tier (``perception: gt_perfect``) never trips it (it
# is not the real-SAM tier). An explicit, documented diagnostic opt-out
# (``SEARCH_ALLOW_GT_DIAGNOSTIC=1``) is the ONLY escape — for a deliberate
# GT-leaky probe under a sam config, a reviewable act.
_REALSTACK_PERCEPTION_TIER = "sam"
# Documented diagnostic opt-out for the C1 gate. Set to "1" ONLY for a
# deliberate GT-diagnostic probe under the real-SAM tier; NEVER on a fleet
# run. Off by default; its presence is logged loudly when it suppresses the
# guard.
_GT_DIAGNOSTIC_OPT_OUT_ENV = "SEARCH_ALLOW_GT_DIAGNOSTIC"


def _assert_realstack_gt_isolation(
    cfg: "RunConfig",
    config_path: Path | None = None,
    *,
    perception: object | None = None,
) -> None:
    """C1 GT-isolation runtime gate (SPEC_real_stack_pivot_2026-07-04.md §1 C1).

    Fires by CONTENT — whenever the resolved perception tier is the real-SAM
    ``sam`` tier — so every DERIVED config inherits the guard, not
    just the canonically-named one. Under that tier it asserts ALL of:

      (a) no GT color table is wired (``gt_perfect.color_table_path``
          unset) — color comes from pixels only;
      (b) the perception tier IS ``sam`` (defense-in-depth);
      (c) ``SEARCH_SKIP_VLM_PERCEPTION`` is NOT set — the GT-era toggle that
          once swapped in synthetic GT perception; it is refused outright so
          a stale environment can never masquerade as a real-SAM run;
      (d) if a resolved ``perception`` OBJECT is passed (post-construction),
          it IS the real SAM adapter, never a synthetic/GT-perfect adapter.

    Fails LOUD (``RuntimeError`` citing C1) at config load — before any
    compile/flight spend — and again post-construction with the object.
    ``SEARCH_ALLOW_GT_DIAGNOSTIC=1`` is the ONLY (documented, logged) escape.
    The gt_perfect diagnostic tier never trips it. The static half of the
    gate is ``distill/tests/test_c1_gt_isolation.py``.
    """
    # Tier-(c) oracle GT-injection derives a
    # viewpoint from the withheld target_xy — deliberately GT-leaky, so it
    # demands the same explicit ack as every other C1 diagnostic. Checked
    # before the tier early-return: tier-independent by design.
    if (
        os.environ.get("SEARCH_ORACLE_INJECT_GT", "").strip() == "1"
        and os.environ.get(_GT_DIAGNOSTIC_OPT_OUT_ENV) != "1"
    ):
        raise RuntimeError(
            "SEARCH_ORACLE_INJECT_GT=1 without SEARCH_ALLOW_GT_DIAGNOSTIC=1 "
            "— acknowledge the GT-injection diagnostic explicitly "
            "(tier (c))."
        )
    if cfg.perception != _REALSTACK_PERCEPTION_TIER:
        return  # not the real-SAM tier → guard does not apply (e.g. gt_perfect)
    if os.environ.get(_GT_DIAGNOSTIC_OPT_OUT_ENV) == "1":
        log.warning(
            "C1 GT-isolation gate SUPPRESSED by %s=1 — GT-diagnostic probe "
            "under the sam tier. This MUST NOT be set on a fleet run.",
            _GT_DIAGNOSTIC_OPT_OUT_ENV,
        )
        return
    where = f"{config_path}: " if config_path is not None else ""
    problems: list[str] = []
    # (b) perception tier (defense-in-depth; trivially true at the trigger, but
    # asserted so a future refactor of the trigger can't silently drop it).
    if cfg.perception != _REALSTACK_PERCEPTION_TIER:
        problems.append(
            f"perception tier is {cfg.perception!r}, must be "
            f"{_REALSTACK_PERCEPTION_TIER!r}"
        )
    # (a) no GT color table wired.
    if (cfg.gt_perfect or {}).get("color_table_path"):
        problems.append(
            f"gt_perfect.color_table_path is set "
            f"({cfg.gt_perfect['color_table_path']!r}), must be unset"
        )
    # (c) SEARCH_SKIP_VLM_PERCEPTION is the deleted GT-era toggle; refuse a
    # stale environment outright rather than let it imply a GT run.
    if os.environ.get("SEARCH_SKIP_VLM_PERCEPTION") == "1":
        problems.append(
            "SEARCH_SKIP_VLM_PERCEPTION=1 is set — this GT-era toggle no "
            "longer exists; unset the env (use the gt_perfect DIAGNOSTIC "
            "config for a GT run)"
        )
    # (d) if the resolved perception object is reachable, it must be the real
    # SAM adapter — not a synthetic / GT-perfect substitute.
    if perception is not None and not isinstance(
        perception, SamDetectClient
    ):
        problems.append(
            "resolved perception object is "
            f"{type(perception).__name__}, must be SamDetectClient "
            "(real SAM) under the sam tier — a synthetic/GT-perfect "
            "object here is a GT leak"
        )
    if problems:
        raise RuntimeError(
            f"{where}C1 GT-ISOLATION VIOLATION under the real-stack "
            f"sam tier — {'; '.join(problems)}. The pivot contract "
            "(distill/SPEC_real_stack_pivot_2026-07-04.md, contract C1) "
            "requires real SAM perception with pixels-only color and zero "
            "CityRefer-derived attributes on the agent path. Set "
            f"{_GT_DIAGNOSTIC_OPT_OUT_ENV}=1 ONLY for a deliberate diagnostic "
            "probe (never on a fleet run)."
        )
    # GT isolation is confirmed IN FORCE for this run — publish it so the
    # downstream bake-integrity guards (e.g. _load_block_assets' missing-
    # _features.json fail-loud, LEAK L1 re-verifier caveat) activate without
    # each re-deriving the tier. Idempotent; cleared by nothing (process-lived).
    os.environ["SEARCH_REALSTACK_GT_ISOLATION"] = "1"
