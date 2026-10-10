"""The two shipped configs load through the strict ``RunConfig(**yaml)``
constructor, which is the wall an unknown YAML key hits.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.citynav.run_benchmark import load_config

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_both_shipped_configs_load():
    for name in ("citynav.yaml", "calcity.yaml"):
        cfg = load_config(str(REPO_ROOT / "configs" / name))
        assert cfg.perception == "sam", name
