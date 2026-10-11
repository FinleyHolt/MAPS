"""The run's provenance is read off its ledger rows.

``_run_provenance`` is what the episode ledger and the A/B comparability
guard key on (git commit, split, bake). It comes off the first row that
carries a ``provenance`` block; a key the row does not record stays absent
rather than being guessed from anything else.
"""

from __future__ import annotations


from benchmarks.citynav.analysis.ab_report import _run_provenance
from benchmarks.citynav.analysis.tests.ledger import write_ledger

_PROV = {
    "git_commit": "abc1234",
    "split": "val_seen",
    "search_env_flags": {"SEARCH_CITYNAV_TILES": "tiles/citynav_2m_osmv2"},
}


def test_provenance_comes_off_the_first_row_that_carries_it(tmp_path):
    write_ledger(
        tmp_path,
        {"episode_id": "e0", "status": "infra", "error": "spawn"},
        {"episode_id": "e1", "status": "ok", "provenance": _PROV},
    )
    prov = _run_provenance(tmp_path)
    assert prov["git_commit"] == "abc1234"
    assert prov["split"] == "val_seen"


def test_an_unrecorded_split_stays_absent(tmp_path):
    prov = {k: v for k, v in _PROV.items() if k != "split"}
    write_ledger(tmp_path, {"episode_id": "e1", "status": "ok", "provenance": prov})
    assert not _run_provenance(tmp_path).get("split")


def test_a_run_without_a_ledger_has_no_provenance(tmp_path):
    """An episode.json alone is not a run."""
    write_ledger(tmp_path / "w0/maps_20260101_000000_citynav_x",
                 {"episode_id": "e1", "status": "ok", "provenance": _PROV}, name="episode.json")
    assert _run_provenance(tmp_path) == {}
