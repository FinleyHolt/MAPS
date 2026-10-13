"""A baseline config is its parent arm plus ``policy`` and nothing else.

The comparison stands on the substrate being identical across methods: the
sensor, the prior, the budget, the replay and the scorer all come from the
parent config (``citynav.yaml``, or ``calcity.yaml`` for ``calcity_*``), and
a baseline config that changed a second key would be measuring that key
under the baseline's name. ``released_protocol.yaml`` is ``citynav.yaml``
with exactly the two protocol keys the released baselines evaluate under.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from benchmarks.citynav.run_benchmark.config import load_config, resolve_config

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
_ARMS = sorted(p for p in (_CONFIGS / "baselines").glob("*.yaml") if p.stem != "released_protocol")


def _parent(path: Path) -> Path:
    return _CONFIGS / ("calcity.yaml" if path.stem.startswith("calcity_") else "citynav.yaml")


def test_there_are_baseline_configs_to_pin() -> None:
    # Vacuity floor: three policies on each of two benches.
    assert len(_ARMS) == 6, [p.name for p in _ARMS]


@pytest.mark.parametrize("path", _ARMS, ids=[p.stem for p in _ARMS])
def test_a_baseline_config_differs_from_its_parent_by_policy_only(path: Path) -> None:
    policy = path.stem.removeprefix("calcity_")
    raw = yaml.safe_load(path.read_text())
    assert raw == {"base": f"../{_parent(path).name}", "policy": policy}
    arm, _ = resolve_config(path)
    parent, _ = resolve_config(_parent(path))
    assert arm.pop("policy") == policy
    assert arm == parent
    assert load_config(path).policy == policy


def test_the_released_protocol_arm_changes_exactly_the_released_protocol_keys() -> None:
    arm, _ = resolve_config(_CONFIGS / "baselines" / "released_protocol.yaml")
    parent, _ = resolve_config(_CONFIGS / "citynav.yaml")
    assert {k for k in parent if parent[k] != arm[k]} == {"max_steps", "start_altitude_agl_m"}
    assert (arm["max_steps"], arm["start_altitude_agl_m"]) == (100, 50.0)
    assert "policy" not in arm


def test_a_policy_name_without_a_policy_module_is_refused(tmp_path: Path) -> None:
    """``released`` and ``tests`` are packages under baselines/ but not
    policies; the loader refuses them instead of failing per episode."""
    for name in ("released", "tests", "no_such_method"):
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump({"base": str(_CONFIGS / "citynav.yaml"), "policy": name}))
        with pytest.raises(ValueError, match="policy"):
            load_config(p)


def test_a_config_without_a_policy_flies_the_mission() -> None:
    assert load_config(_CONFIGS / "citynav.yaml").policy is None
