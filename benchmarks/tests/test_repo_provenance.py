"""A tree deployed without git still names the commit it runs."""

from __future__ import annotations

import benchmarks


def test_a_tree_without_git_reads_its_commit_stamp(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(benchmarks, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(benchmarks, "COMMIT_STAMP", tmp_path / ".commit")
    assert benchmarks.git_provenance()["git_commit"] is None
    (tmp_path / ".commit").write_text("abc123\n")
    prov = benchmarks.git_provenance()
    assert prov["git_commit"] == "abc123" and prov["git_dirty"] is None
