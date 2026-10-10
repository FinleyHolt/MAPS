"""The config loader: base chains, deep merge, and the schema's refusals.

The resolved shipped configs equal the flat files they replaced. Those files
are pinned below as parsed data, each checked against the sha256 of its
canonical JSON taken from the file itself; the expected resolution is that
data less the keys nothing read, plus the value the code used where a flat
file left a key out.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.citynav.run_benchmark.config import (
    assert_flight_config,
    load_config,
    resolve_config,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

FLAT_CITYNAV = {
    "altitude_m": 125.0, "altitude_tolerance_m": 1.0, "cell_size_m": 2.0,
    "compact": {"enable": True}, "descent_step_m": 5.0,
    "flight_model": {"camera_hz": 5.0, "kind": "discrete6"}, "fov_m": 0.0,
    "landing_altitude_m": 5.0, "landing_xy_tolerance_m": 5.0, "max_steps": 200,
    "perception": "sam", "rgb_upscale_px": 2016,
    "sam": {"color_palette": "citynav", "device": "cuda", "dtype": "bfloat16",
            "max_per_step": 1024, "min_area_px": 32, "model_id": "facebook/sam3.1",
            "sahi": True, "sahi_nms_iou": 0.5, "sahi_overlap": 0.2,
            "score_threshold": 0.3},
    "sensor": "nadir_1008_90", "success_radius_m": 20.0,
}
FLAT_CALCITY = {
    "altitude_cap_m": 1000.0, "altitude_m": 125.0, "altitude_tolerance_m": 1.0,
    "cell_size_m": 2.0,
    "compact": {"concept_union_budget": 20000, "enable": True,
                "examiner_budget": 8000, "grader_budget": 38000,
                "router_budget": 8000, "stage1_budget": 8000,
                "stage2_3_budget": 38000, "subagent_budget": 20000,
                "teacher_budget": 38000},
    "descent_step_m": 5.0, "flight_model": {"camera_hz": 5.0}, "fov_m": 0.0,
    "landing_altitude_m": 5.0, "landing_xy_tolerance_m": 5.0, "max_steps": 100000,
    "perception": "sam", "rgb_upscale_px": 1008,
    "sam": {"device": "cuda", "dtype": "bfloat16", "max_per_step": 1024,
            "min_area_px": 32, "model_id": "facebook/sam3.1",
            "score_threshold": 0.3},
    "sensor": "oryx_orx32_25mm_nadir", "spawn": "uniform", "spawn_seed": 1234,
    "start_altitude_agl_m": 914.4, "success_radius_m": 20.0,
}
FLAT_SHA256 = {
    "citynav": "c38e9c91e487b09b8e149baeffb28f9ac6e741ed027fbec348b2cb656b907de5",
    "calcity": "366ceb132d850769dc3875e6615b2faaabf525576cf2fcb19cf0a446e2221e16",
}
# Keys nothing read.
DEAD_TOP = ("descent_step_m", "altitude_tolerance_m", "landing_xy_tolerance_m", "perception")
DEAD_SAM = ("device", "dtype")
DEAD_COMPACT = ("stage1_budget", "stage2_3_budget", "teacher_budget", "grader_budget",
                "concept_union_budget", "subagent_budget", "router_budget",
                "examiner_budget")
# Where a flat file left a key out, the value the code used (the RunConfig,
# SamDetectClient and FlightModelConfig defaults; an
# absent color_palette ran "citynav" in run_benchmark/perception.py).
ABSENT_MEANT = {
    "precise_tolerance_m": 0.0, "precise_floor_radius_m": 5.0, "spawn": "episode",
    "spawn_seed": 0, "start_altitude_band_m": [100.0, 150.0],
    "start_altitude_agl_m": None, "altitude_cap_m": 200.0, "descend_to_land": False,
    "withhold": [],
}
ABSENT_MEANT_FLIGHT = {"kind": "discrete6", "speed_mps": None}
ABSENT_MEANT_SAM = {"color_palette": "citynav", "sahi": False, "sahi_overlap": 0.2,
                    "sahi_nms_iou": 0.5}


def _sha(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _expected(flat: dict) -> dict:
    out = {**ABSENT_MEANT, **json.loads(json.dumps(flat))}
    for k in DEAD_TOP:
        del out[k]
    out["flight_model"] = {**ABSENT_MEANT_FLIGHT, **out["flight_model"]}
    out["sam"] = {k: v for k, v in {**ABSENT_MEANT_SAM, **out["sam"]}.items()
                  if k not in DEAD_SAM}
    out["compact"] = {k: v for k, v in out["compact"].items() if k not in DEAD_COMPACT}
    return out


@pytest.mark.parametrize("name,flat", [("citynav", FLAT_CITYNAV), ("calcity", FLAT_CALCITY)])
def test_shipped_configs_resolve_to_the_flat_files(name, flat):
    assert _sha(flat) == FLAT_SHA256[name]
    resolved, _ = resolve_config(REPO_ROOT / "configs" / f"{name}.yaml")
    assert resolved == _expected(flat)
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
    p = _write(tmp_path, "a/b/arm.yaml", "base: ../base.yaml\nsam:\n  min_area_px: 7\n")
    data, _ = resolve_config(p)
    assert data["sam"]["sahi"] is False and data["sam"]["min_area_px"] == 7
    assert data["sam"]["model_id"] == "facebook/sam3.1"


@pytest.mark.parametrize("body,needle", [
    ("bogus: 1\n", "unknown config key(s) ['bogus']"),
    ("sam:\n  device: cuda\n", "unknown config key(s) ['sam.device']"),
    ("flight_model:\n  speed: 3\n", "unknown config key(s) ['flight_model.speed']"),
    ("max_steps: 2.5\n", "max_steps must be int"),
    ("max_steps: 0\n", "max_steps must be > 0"),
    ("descend_to_land: 1\n", "descend_to_land must be bool"),
    ("sam:\n  score_threshold: 1.5\n", "sam.score_threshold must be in [0, 1]"),
    ("spawn: random\n", "spawn must be one of"),
    ("start_altitude_band_m: [150, 100]\n", "start_altitude_band_m must be [lo, hi]"),
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


def test_the_flight_mission_refuses_a_withholding_config(tmp_path):
    assert_flight_config(load_config(REPO_ROOT / "configs" / "calcity.yaml"))
    with pytest.raises(ValueError, match="withhold"):
        assert_flight_config(load_config(_over(tmp_path, "withhold: [images]\n")))



def test_every_cfg_attribute_the_bench_reads_is_a_field():
    import dataclasses
    import re

    from benchmarks.citynav.run_benchmark.config import RunConfig

    fields = {f.name for f in dataclasses.fields(RunConfig)}
    pattern = re.compile(r"\bcfg\.([a-z_]+)\b|getattr\(cfg, \"([a-z_]+)\"")
    read = {a or b for p in (REPO_ROOT / "benchmarks" / "citynav" / "run_benchmark").glob("*.py")
            for a, b in pattern.findall(p.read_text())}
    assert len(read) > 15, read
    assert read <= fields, sorted(read - fields)


def test_each_withhold_config_is_the_control_less_its_own_part():
    control = resolve_config(REPO_ROOT / "configs/citynav.yaml")[0]
    shipped = sorted((REPO_ROOT / "configs/withhold").glob("*.yaml"))
    assert len(shipped) == 6
    for path in shipped:
        resolved = resolve_config(path)[0]
        assert resolved == {**control, "withhold": [path.stem]}, path.name
