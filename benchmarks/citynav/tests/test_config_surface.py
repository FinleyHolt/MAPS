"""The config loader: base chains, deep merge, and the schema's refusals."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.citynav.run_benchmark.config import load_config, resolve_config

REPO_ROOT = Path(__file__).resolve().parents[3]

CITYNAV = {
    "altitude_cap_m": 200.0, "altitude_m": 125.0, "cell_size_m": 2.0, "max_steps": 200,
    "precise_floor_radius_m": 5.0, "precise_tolerance_m": 0.0,
    "sam": {"sahi": True},
    "sensor": "nadir_1008_90", "spawn": "episode", "spawn_seed": 0,
    "start_altitude_agl_m": None, "success_radius_m": 20.0, "withhold": [],
}
CALCITY = {
    **CITYNAV, "altitude_cap_m": 1000.0, "max_steps": 100000,
    "sam": {"sahi": False},
    "sensor": "oryx_orx32_25mm_nadir", "spawn": "uniform", "spawn_seed": 1234,
    "start_altitude_agl_m": 914.4,
}


@pytest.mark.parametrize("name,want", [("citynav", CITYNAV), ("calcity", CALCITY)])
def test_shipped_configs_resolve_to_what_the_bench_flies(name, want):
    resolved, _ = resolve_config(REPO_ROOT / "configs" / f"{name}.yaml")
    assert resolved == want
    load_config(REPO_ROOT / "configs" / f"{name}.yaml")


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _over(tmp_path: Path, body: str) -> Path:
    control = (REPO_ROOT / "configs" / "citynav.yaml").resolve()
    return _write(tmp_path, "arm.yaml", f"base: {control}\n{body}")


def test_base_resolves_relative_to_the_file_and_deep_merges(tmp_path):
    _write(tmp_path, "a/base.yaml", f"base: {(REPO_ROOT / 'configs/citynav.yaml').resolve()}\n"
           "sam:\n  sahi: false\n")
    p = _write(tmp_path, "a/b/arm.yaml", "base: ../base.yaml\nmax_steps: 7\n")
    data, _ = resolve_config(p)
    assert data["sam"]["sahi"] is False and data["max_steps"] == 7
    assert data["sensor"] == "nadir_1008_90"


@pytest.mark.parametrize("body,needle", [
    ("bogus: 1\n", "unknown config key(s) ['bogus']"),
    ("sam:\n  device: cuda\n", "unknown config key(s) ['sam.device']"),
    ("max_steps: 2.5\n", "max_steps must be int"),
    ("max_steps: 0\n", "max_steps must be > 0"),
    ("max_steps: true\n", "max_steps must be int"),
    ("sam:\n  sahi: 1\n", "sam.sahi must be bool"),
    ("spawn: random\n", "spawn must be one of"),
    ("sam: true\n", "sam is a block of keys"),
    ("withhold: [pictures]\n", "withhold must be"),
    ("withhold: [look, look]\n", "withhold must be"),
])
def test_a_bad_key_or_value_is_refused_with_its_path(tmp_path, body, needle):
    with pytest.raises(ValueError) as e:
        resolve_config(_over(tmp_path, body))
    assert needle in str(e.value), str(e.value)


def test_a_missing_key_is_refused(tmp_path):
    with pytest.raises(ValueError) as e:
        resolve_config(_write(tmp_path, "x.yaml", "sensor: nadir_1008_90\n"))
    assert "missing config key(s)" in str(e.value) and "'max_steps'" in str(e.value)


def test_a_base_loop_is_refused(tmp_path):
    _write(tmp_path, "a.yaml", "base: b.yaml\n")
    _write(tmp_path, "b.yaml", "base: a.yaml\n")
    with pytest.raises(ValueError, match="base chain loops"):
        resolve_config(tmp_path / "a.yaml")


def test_each_withhold_config_is_the_control_less_its_own_part():
    control = resolve_config(REPO_ROOT / "configs/citynav.yaml")[0]
    shipped = sorted((REPO_ROOT / "configs/withhold").glob("*.yaml"))
    assert len(shipped) == 6
    for path in shipped:
        resolved = resolve_config(path)[0]
        assert resolved == {**control, "withhold": [path.stem]}, path.name
