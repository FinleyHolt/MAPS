"""The arm stamp proves the bake: SEARCH_CITYNAV_TILES is on every row.

The tile bake is the ledger's fatal comparability class. A run flying the
DEFAULT bake used to stamp nothing (only set env vars were recorded), so
its column read "tile bake unrecorded" and could pair against nothing —
the stamp now records the resolution either way.
"""
from __future__ import annotations

from benchmarks.citynav import paths
from benchmarks.citynav.run_manifest import arm_env_stamp


def test_default_bake_stamps_the_resolved_tiles_dir(monkeypatch):
    monkeypatch.delenv("SEARCH_CITYNAV_TILES", raising=False)
    stamp = arm_env_stamp()
    assert stamp["SEARCH_CITYNAV_TILES"] == str(paths.TILES_DIR)


def test_explicit_bake_stamps_verbatim(monkeypatch):
    monkeypatch.setenv("SEARCH_CITYNAV_TILES", "/data/tiles/citynav_2m_osmv3")
    stamp = arm_env_stamp()
    assert stamp["SEARCH_CITYNAV_TILES"] == "/data/tiles/citynav_2m_osmv3"


def test_reasoning_effort_reaches_the_row(monkeypatch):
    # How hard the model was asked to think moves decode tokens per
    # episode and the score with them, so a run's records say what effort
    # it flew — the same row the bake rides, read by the bench predictions
    # stamp and the station's sortie record alike.
    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", "0.99")
    assert arm_env_stamp()["SEARCH_SAMPLING_REASONING_EFFORT"] == "0.99"


def test_unset_reasoning_effort_leaves_the_row_alone(monkeypatch):
    # Set-only, like every arm flag but the bake: a run that named no
    # effort flew the SERVE's default, and a value stamped here would
    # claim the arm asked for one.
    monkeypatch.delenv("SEARCH_SAMPLING_REASONING_EFFORT", raising=False)
    assert "SEARCH_SAMPLING_REASONING_EFFORT" not in arm_env_stamp()
