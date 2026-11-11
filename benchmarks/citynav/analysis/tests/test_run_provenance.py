"""The run's provenance is read off its ledger rows.

``_run_provenance`` is what the episode ledger and the A/B comparability
guard key on (git commit, split, bake). It comes off the first row that
carries a ``provenance`` block; a key the row does not record stays absent
rather than being guessed from anything else.
"""

from __future__ import annotations


from benchmarks.citynav.analysis.ab_report import _run_provenance
from benchmarks.run_record.tests.ledger import write_ledger

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


def test_a_run_resumed_under_other_code_is_an_arm_mismatch(tmp_path):
    """The first row names one commit; a resume the manifest records under another means the run's
    rows came from two codes, which an A/B against a one-code run must not wave through."""
    import json

    from benchmarks.citynav.analysis.ab_report import comparability_issues

    a, b = tmp_path / "a", tmp_path / "b"
    for root, resumes in ((a, [{"git_commit": "def5678"}]), (b, [])):
        write_ledger(root, {"episode_id": "e1", "status": "ok", "provenance": _PROV})
        (root / "run_manifest.json").write_text(json.dumps({"git_commit": "abc1234", "resumes": resumes}))
    _, arm = comparability_issues(_run_provenance(a), _run_provenance(b))
    assert any("code (git_commit) differs across resumes" in m for m in arm), arm
    assert comparability_issues(_run_provenance(b), _run_provenance(b)) == ([], [])
