"""The bake stamp is the ONE content check on a run's tile bake.

Every other provenance field compares names; two bakes of the same blocks
share all of them and score the same episodes differently.

These pin the three states a reader must distinguish -- verified,
mismatched, and never-stamped -- and that hashing sees content, not just
the file list.
"""

from __future__ import annotations

import json

from benchmarks.citynav import paths
from benchmarks.citynav.analysis import run_report


def _bake(root, features="{}"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "block_0_features.json").write_text(features)
    (root / "block_0_meta.json").write_text('{"crs": "utm"}')
    return root


def test_hash_is_stable_and_content_sensitive(tmp_path):
    a = _bake(tmp_path / "a")
    first = paths.bake_sha256(a)
    assert first and paths.bake_sha256(a) == first, "must be deterministic"

    # Same file list, different bytes -> different bake.
    (a / "block_0_features.json").write_text('{"prior": []}')
    assert paths.bake_sha256(a) != first, "content change must move the hash"


def test_hash_sees_renames_and_added_blocks(tmp_path):
    a = _bake(tmp_path / "a")
    base = paths.bake_sha256(a)
    (a / "block_0_meta.json").rename(a / "block_1_meta.json")
    assert paths.bake_sha256(a) != base, "a rename must move the hash"

    b = _bake(tmp_path / "b")
    added = paths.bake_sha256(b)
    (b / "block_2_nodes.json").write_text("{}")
    assert paths.bake_sha256(b) != added, "an added block file must move the hash"


def test_two_bakes_of_one_prior_hash_equal(tmp_path):
    """A meta's key order and layout are not its content, and a file the bake
    did not write is no part of it."""
    a, b = _bake(tmp_path / "a"), _bake(tmp_path / "b")
    (a / "block_0_meta.json").write_text(json.dumps({"crs": "utm", "provenance": {"prior": "osm"}}, indent=2))
    (b / "block_0_meta.json").write_text(json.dumps({"provenance": {"prior": "osm"}, "crs": "utm"}))
    (b / "notes.txt").write_text("scratch")
    assert paths.bake_sha256(a) == paths.bake_sha256(b)

    (b / "block_0_meta.json").write_text(json.dumps({
        "crs": "utm", "provenance": {"prior": "osm-other"}}))
    assert paths.bake_sha256(a) != paths.bake_sha256(b), "the prior is content"


def test_missing_bake_is_none_not_a_pass(tmp_path):
    assert paths.bake_sha256(tmp_path / "nope") is None


def _run(root, tiles, *, stamped):
    root.mkdir(parents=True, exist_ok=True)
    # Absolute on the machine that flew it, not this one.
    perception = {"tiles_dir_resolved": f"/scratch/someone/MAPS/{tiles}"}
    if stamped is not None:
        perception["bake_sha256"] = stamped
    (root / "run_manifest.json").write_text(json.dumps({"perception": perception}))
    return root


def test_reader_reports_all_three_states(tmp_path, monkeypatch):
    tiles = _bake(tmp_path / "tiles" / "citynav_2m_osmv3")
    monkeypatch.setattr(run_report, "_REPO_ROOT", tmp_path)
    rel = tiles.relative_to(tmp_path)
    real = paths.bake_sha256(tiles)

    verified = run_report.bake_line(_run(tmp_path / "r1", rel, stamped=real))
    assert "verified" in verified and "MISMATCH" not in verified

    wrong = run_report.bake_line(_run(tmp_path / "r2", rel, stamped="0" * 64))
    assert "MISMATCH" in wrong, "a different bake must be called out loudly"
    assert "not comparable" in wrong, "must say what the mismatch costs"

    # A run flown before the stamp existed: unverified, never a pass.
    absent = run_report.bake_line(_run(tmp_path / "r3", rel, stamped=None))
    assert "UNVERIFIED" in absent and "verified " not in absent

    # Nothing here refuses -- an audit of an old run must still read.
    for line in (verified, wrong, absent):
        assert isinstance(line, str) and line


def test_reader_does_not_confuse_benchmarks(tmp_path, monkeypatch):
    """A Cal City run must be checked against the Cal City bake.

    ``paths.TILES_DIR`` is the CityNav bake; resolving the manifest's own
    ``tiles_dir_resolved`` is what keeps a calcity run from reporting a
    mismatch against citynav tiles on every read.
    """
    monkeypatch.setattr(run_report, "_REPO_ROOT", tmp_path)
    citynav = _bake(tmp_path / "tiles" / "citynav_2m_osmv3", features='{"a": 1}')
    calcity = _bake(tmp_path / "benchmarks" / "calcity" / "tiles", features='{"b": 2}')
    assert paths.bake_sha256(citynav) != paths.bake_sha256(calcity)

    run = _run(
        tmp_path / "cal",
        calcity.relative_to(tmp_path),
        stamped=paths.bake_sha256(calcity),
    )
    assert "verified" in run_report.bake_line(run)


def test_the_resolved_tiles_dir_names_the_bake_to_check(tmp_path, monkeypatch):
    """``tiles_dir_resolved`` records the bake a run actually flew. Falling
    back to this checkout's default bake instead reports a mismatch on every
    Cal City read -- a correct number, disowned by its own reader.
    """
    monkeypatch.setattr(run_report, "_REPO_ROOT", tmp_path)
    _bake(tmp_path / "tiles" / "citynav_2m_osmv3", features='{"a": 1}')
    calcity = _bake(tmp_path / "benchmarks" / "calcity" / "tiles", features='{"b": 2}')

    root = tmp_path / "cal_nosuffix"
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "perception": {
                    # Absolute, on the machine that flew it -- not this one.
                    "tiles_dir_resolved": "/scratch/someone/MAPS"
                    "/benchmarks/calcity/tiles",
                    "bake_sha256": paths.bake_sha256(calcity),
                }
            }
        )
    )
    line = run_report.bake_line(root)
    assert "verified" in line and "MISMATCH" not in line, line

    # A path this checkout has no copy of stays honest rather than guessing.
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "perception": {
                    "tiles_dir_resolved": "/scratch/someone/elsewhere/no_such_bake",
                    "bake_sha256": paths.bake_sha256(calcity),
                }
            }
        )
    )
    assert "verified" not in run_report.bake_line(root)


def test_a_bake_outside_the_repo_is_checked_where_the_run_recorded_it(tmp_path, monkeypatch):
    """Tiles under an absolute path outside the checkout are read there; a
    recorded bake that is nowhere here is UNKNOWN, never compared against
    the CityNav default."""
    monkeypatch.setattr(run_report, "_REPO_ROOT", tmp_path / "repo")
    _bake(tmp_path / "repo" / "tiles" / "citynav_2m_osmv3", features='{"a": 1}')
    monkeypatch.setattr(paths, "TILES_DIR", tmp_path / "repo" / "tiles" / "citynav_2m_osmv3")
    outside = _bake(tmp_path / "elsewhere" / "calcity_tiles", features='{"b": 2}')

    def run(name, tiles_dir):
        root = tmp_path / name
        root.mkdir()
        (root / "run_manifest.json").write_text(json.dumps({"perception": {
            "tiles_dir_resolved": str(tiles_dir), "bake_sha256": paths.bake_sha256(outside)}}))
        return root

    line = run_report.bake_line(run("abs", outside))
    assert "verified" in line, line
    line = run_report.bake_line(run("gone", tmp_path / "nowhere" / "tiles"))
    assert "UNKNOWN" in line and "MISMATCH" not in line, line
