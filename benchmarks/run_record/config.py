"""``RunConfig`` -- the knob object for one bench run -- and its YAML loader.

A config file is either the control (``configs/citynav.yaml``, which declares
every key) or an override: ``base: <file>`` names the file it changes,
relative to itself, and the file holds only the keys it changes. Chains
resolve and deep-merge; ``_SCHEMA`` is the one statement of every key, its
type and its range, and the loader refuses an unknown key, a missing key or
a bad value at load, naming the key and the file. ``_OPTIONAL`` keys may be
left out; an absent one takes its ``RunConfig`` default.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from benchmarks import REPO_ROOT
from maps.perception.geometry.sensor_spec import resolve_sensor


_NUM = (int, float)
_NONE = type(None)


def _one_of(*allowed: object) -> tuple[Callable[[Any], bool], str]:
    return (lambda v: v in allowed), f"one of {list(allowed)}"


def _sensor_ok(v: Any) -> bool:
    return resolve_sensor(v) is not None  # a bad name or override raises, naming the problem


def _policy_ok(v: Any) -> bool:
    # By name: benchmarks carries no static import of baselines.
    try:
        return importlib.util.find_spec(f"baselines.{v}.policy") is not None
    except ModuleNotFoundError:
        return False


_POS = (lambda v: v > 0), "> 0"
_NONNEG = (lambda v: v >= 0), ">= 0"

# Every YAML key: (accepted types, (test, what the test asks)). A float key
# reads an int too.
_SCHEMA: dict[str, Any] = {
    "max_steps": (int, _POS),
    "success_radius_m": (_NUM, _POS),
    "spawn": (str, _one_of("episode", "uniform")),
    "spawn_seed": (int, _NONNEG),
    "start_altitude_agl_m": ((*_NUM, _NONE), ((lambda v: v is None or v > 0), "null or > 0")),
    "altitude_cap_m": (_NUM, _POS),
    "sensor": ((str, dict), (_sensor_ok, "a registered sensor or an inline override")),
    "policy": (str, (_policy_ok, "the name of a baselines/<policy>/policy.py")),
}
# Keys a config may leave out.
_OPTIONAL = {"policy"}


@dataclass
class RunConfig:
    """One resolved config: every ``_SCHEMA`` key, plus the path the config
    resolved from."""

    max_steps: int
    success_radius_m: float
    # "episode" flies the dataset's start pose; "uniform" a seeded uniform
    # point over the tile bounds, deterministic per (spawn_seed, episode_id).
    spawn: str
    spawn_seed: int
    # The start altitude is upstream's start z; start_altitude_agl_m, when
    # set, replaces it with GROUND_LEVEL[map] + that value.
    start_altitude_agl_m: float | None
    altitude_cap_m: float
    # A registered sensor name or an inline override (sensor_spec).
    sensor: str | dict
    # A published method that chooses the stops in place of the mission
    # conversation (baselines/<policy>/policy.py); None flies the mission.
    policy: str | None = None
    # The resolved top-level YAML path, written by ``load_config``.
    source_path: str | None = None


def assert_flight_config(cfg: RunConfig, environ: Mapping[str, str]) -> None:
    """The flight mission flies the flown engine: refuse a config that
    names a baseline policy."""
    if cfg.policy is not None:
        raise ValueError(f"{cfg.source_path}: the flight mission flies MAPS; policy={cfg.policy!r} is a bench arm")


def _locate(path: str | Path | None) -> Path:
    if not path:
        raise RuntimeError("a --config YAML is required: pass --config configs/citynav.yaml.")
    candidates = [Path(path)]
    if not candidates[0].is_absolute():
        candidates.append(REPO_ROOT / path)
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
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{p}: a config is a mapping of keys")
    base = data.pop("base", None)
    if base is None:
        return data
    if not isinstance(base, str) or not (p.parent / base).is_file():
        raise ValueError(f"{p}: base {base!r} names no config file")
    return _merge(_read_chain(p.parent / base, (*seen, p)), data)


def _check(data: dict, src: Path) -> None:
    unknown = sorted(set(data) - set(_SCHEMA))
    missing = sorted(set(_SCHEMA) - set(data) - _OPTIONAL)
    if unknown:
        raise ValueError(f"{src}: unknown config key(s) {unknown}")
    if missing:
        raise ValueError(f"{src}: missing config key(s) {missing}")
    for key, rule in _SCHEMA.items():
        if key not in data:
            continue
        v = data[key]
        types, (ok, says) = rule
        types = types if isinstance(types, tuple) else (types,)
        if not isinstance(v, types) or (isinstance(v, bool) and bool not in types):
            names = "/".join("null" if t is _NONE else t.__name__ for t in types)
            raise ValueError(f"{src}: {key} must be {names}, got {v!r}")
        try:
            passed = ok(v)
        except ValueError as e:
            raise ValueError(f"{src}: {key}: {e}") from e
        if not passed:
            raise ValueError(f"{src}: {key} must be {says}, got {v!r}")


def resolve_config(path: str | Path | None) -> tuple[dict, Path]:
    """The checked, fully merged key dict of a config file, and its path."""
    p = _locate(path)
    data = _read_chain(p)
    _check(data, p)
    return data, p


def config_sha256(path: str | Path) -> str:
    """sha256 of the resolved config's canonical JSON: the identity of what a
    run flew, whichever base chain spelled it."""
    data, _ = resolve_config(path)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_config(path: str | Path | None) -> RunConfig:
    data, p = resolve_config(path)
    cfg = RunConfig(**data)
    cfg.source_path = str(p)
    return cfg
