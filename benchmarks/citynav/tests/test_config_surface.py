"""The config loader: base chains, merge, and the schema's refusals."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.citynav.run_benchmark.config import load_config, resolve_config
from maps.search.withhold import WITHHOLDS

REPO_ROOT = Path(__file__).resolve().parents[3]

CITYNAV = {
    "altitude_cap_m": 200.0, "max_steps": 200,
    "sensor": "nadir_1008_90", "spawn": "episode", "spawn_seed": 0,
    "start_altitude_agl_m": None, "success_radius_m": 20.0, "withhold": [],
}
CALCITY = {
    **CITYNAV, "altitude_cap_m": 1000.0, "max_steps": 100000,
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


def test_base_resolves_relative_to_the_file_and_merges(tmp_path):
    _write(tmp_path, "a/base.yaml", f"base: {(REPO_ROOT / 'configs/citynav.yaml').resolve()}\n"
           "spawn_seed: 5\n")
    p = _write(tmp_path, "a/b/arm.yaml", "base: ../base.yaml\nmax_steps: 7\n")
    data, _ = resolve_config(p)
    assert data["spawn_seed"] == 5 and data["max_steps"] == 7
    assert data["sensor"] == "nadir_1008_90"


@pytest.mark.parametrize("body,needle", [
    ("bogus: 1\n", "unknown config key(s) ['bogus']"),
    ("max_steps: 2.5\n", "max_steps must be int"),
    ("max_steps: 0\n", "max_steps must be > 0"),
    ("max_steps: true\n", "max_steps must be int"),
    ("spawn: random\n", "spawn must be one of"),
    ("withhold: [pictures]\n", "withhold must be"),
    ("withhold: [look, look]\n", "withhold must be"),
    ("withhold: [{a: 1}]\n", "withhold must be"),
    ("withhold: [[look]]\n", "withhold must be"),
    ("sensor: no_such_sensor\n", ".yaml: sensor: "),
    ('sensor: ""\n', "sensor must be a registered sensor"),
    ("policy: no_such_method\n", "policy must be the name of a baselines/<policy>/policy.py"),
    ("policy: 3\n", "policy must be str"),
])
def test_a_bad_key_or_value_is_refused_with_its_name(tmp_path, body, needle):
    with pytest.raises(ValueError) as e:
        resolve_config(_over(tmp_path, body))
    assert needle in str(e.value), str(e.value)


def test_a_missing_key_is_refused(tmp_path):
    with pytest.raises(ValueError) as e:
        resolve_config(_write(tmp_path, "x.yaml", "sensor: nadir_1008_90\n"))
    assert "missing config key(s)" in str(e.value) and "'max_steps'" in str(e.value)


def test_policy_is_the_one_key_a_config_may_leave_out():
    """Absent, the bench flies the mission, and the resolved config carries no
    ``policy`` key, so its ``config_sha256`` is the one a MAPS root records."""
    assert load_config(REPO_ROOT / "configs" / "citynav.yaml").policy is None


def test_the_flight_refuses_a_baseline_policy():
    from benchmarks.citynav.run_benchmark.config import assert_flight_config

    cfg = load_config(REPO_ROOT / "configs" / "calcity.yaml")
    cfg.policy = "visa"
    with pytest.raises(ValueError, match="policy='visa' is a bench arm"):
        assert_flight_config(cfg)


def test_a_base_loop_is_refused(tmp_path):
    _write(tmp_path, "a.yaml", "base: b.yaml\n")
    _write(tmp_path, "b.yaml", "base: a.yaml\n")
    with pytest.raises(ValueError, match="base chain loops"):
        resolve_config(tmp_path / "a.yaml")


@pytest.mark.parametrize("base", ["nowhere.yaml", "3"])
def test_a_base_naming_no_file_is_refused_with_the_config(tmp_path, base):
    with pytest.raises(ValueError, match=r"a\.yaml: base .* names no config file"):
        resolve_config(_write(tmp_path, "a.yaml", f"base: {base}\n"))


# The research families flown as one arm each: family -> its withholds.
FAMILIES = {"seeing": ["look_for", "look", "images", "map"], "text": ["doc", "system_turn"]}


def test_each_withhold_config_is_the_control_less_its_own_part():
    control = resolve_config(REPO_ROOT / "configs/citynav.yaml")[0]
    shipped = sorted((REPO_ROOT / "configs/withhold").glob("*.yaml"))
    assert sorted(p.stem for p in shipped) == sorted([*WITHHOLDS, *FAMILIES])
    for path in shipped:
        resolved = resolve_config(path)[0]
        want = FAMILIES.get(path.stem, [path.stem])
        assert resolved == {**control, "withhold": want}, path.name


def test_a_blocks_cell_size_is_its_bakes(tmp_path, monkeypatch):
    """The config carries no cell size: bounds and grid_shape in the meta are
    computed against the bake's own."""
    import json

    from benchmarks.citynav.run_benchmark import assets

    (tmp_path / "blk_meta.json").write_text(json.dumps({
        "bounds": {"x_min": 0.0, "y_min": 0.0, "x_max": 30.0, "y_max": 30.0},
        "grid_shape": [10, 10], "cell_size_m": 3.0}))
    (tmp_path / "blk_features.json").write_text("{}")
    monkeypatch.setattr(assets, "tile_stem", lambda name: str(tmp_path / name))
    assert assets.load_episode_assets("blk").cell_size_m == 3.0
