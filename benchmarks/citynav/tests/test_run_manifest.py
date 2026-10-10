"""Vacuity floor for the run manifest (derived tests fail open, so this one
pins by name): the writer emits every required field NON-EMPTY on a stub run,
the measured-identity fields exist under their exact names, secrets never
appear, and re-entry semantics hold — an identical re-entry appends a resume,
an identity drift refuses.
"""
from __future__ import annotations

import json
import pytest

from benchmarks.citynav import paths, run_manifest
from maps.llm import endpoint_spec as endpoint_spec_module
from maps.llm.endpoint_spec import EndpointSpec

STUB_SPEC = EndpointSpec(
    base_url="http://127.0.0.1:9/v1",
    model_id="stub-model",
    max_model_len=32768,
    tokenizer_name="stub-tokenizer",
    output_reserve=4096,
)


@pytest.fixture()
def bake(monkeypatch, tmp_path):
    """A tile bake on disk for the manifest to hash."""
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    (tiles / "block_0_meta.json").write_text("{}")
    monkeypatch.setattr(paths, "TILES_DIR", tiles)
    return tiles


@pytest.fixture()
def stub_run(monkeypatch, tmp_path, bake):
    monkeypatch.setattr(endpoint_spec_module, "_SPEC", STUB_SPEC)
    monkeypatch.setenv("SEARCH_REASONER_API_KEY", "sk-SECRET-NEVER-RECORDED")
    return run_manifest.build_manifest(
        out_root=tmp_path,
        config_path=paths.REPO_ROOT / "configs/citynav.yaml",
        split="train_seen",
        sample_seed=1234,
        sample_offset=0,
        limit=3,
        workers=2,
        allowlist_path=None,
        episodes=["ep_a", "ep_b", "ep_c"],
    )


def test_required_fields_nonempty(stub_run):
    m = stub_run
    for keypath in (
        ("schema",),
        ("created_ts",),
        ("hostname",),
        ("out_root",),
        ("git_commit",),
        ("config",),
        ("config_sha256",),
        ("config_yaml",),
        ("split",),
        ("search_env_flags", "SEARCH_CITYNAV_TILES"),
        ("selection", "selection_sha"),
        ("endpoints", "model_id"),
        ("endpoints", "max_model_len"),
        ("perception", "tiles_dir_resolved"),
    ):
        assert run_manifest._get(m, keypath), f"empty manifest field: {keypath}"
    # By-name pins on the MEASURED identity (the fields whose disappearance
    # a derived scan could not notice).
    for keypath in (("git_commit",), ("config_sha256",), ("endpoints", "model_id"),
                    ("perception", "bake_sha256")):
        assert run_manifest._get(m, keypath), f"measured field vanished: {keypath}"
    assert m["selection"]["episodes_n"] == 3
    assert m["withhold"] == []


def test_secret_never_appears(stub_run):
    dumped = json.dumps(stub_run)
    assert "sk-SECRET-NEVER-RECORDED" not in dumped
    assert "SEARCH_REASONER_API_KEY" not in stub_run["env"]


def test_replay_round_trip_and_refusals(stub_run, tmp_path):
    src_path = tmp_path / "source_manifest.json"
    src_path.write_text(json.dumps(stub_run))
    source = run_manifest.load_manifest(src_path)
    # Same checkout, same resolution: both gates pass.
    run_manifest.assert_replay_git(source)
    run_manifest.verify_replay(source, stub_run)
    # Tampered git sha: refused, no auto-checkout.
    tampered = json.loads(json.dumps(stub_run))
    tampered["git_commit"] = "deadbeef" * 5
    with pytest.raises(run_manifest.ManifestIdentityError, match="git mismatch"):
        run_manifest.assert_replay_git(tampered)
    # A re-derivation that selects different episodes: refused.
    drifted = json.loads(json.dumps(stub_run))
    drifted["selection"]["selection_sha"] = "000000000000"
    with pytest.raises(run_manifest.ManifestIdentityError, match="replay drift"):
        run_manifest.verify_replay(source, drifted)
    # A random JSON is not a manifest.
    (tmp_path / "not_manifest.json").write_text("{}")
    with pytest.raises(run_manifest.ManifestIdentityError, match="schema"):
        run_manifest.load_manifest(tmp_path / "not_manifest.json")


def test_apply_replay_args_forces_recorded_selection(stub_run, tmp_path):
    import argparse

    src_path = tmp_path / "source_manifest.json"
    src_path.write_text(json.dumps(stub_run))
    args = argparse.Namespace(
        from_manifest=str(src_path),
        split="val_seen",
        sample_seed=999,
        sample_offset=7,
        limit=0,
        episode_allowlist=None,
    )
    source = run_manifest.apply_replay_args(args)
    assert source["split"] == "train_seen"
    assert args.split == "train_seen"
    assert args.sample_seed == 1234
    assert args.sample_offset == 0
    assert args.limit == 3


def test_write_once_resume_and_refusal(stub_run, tmp_path):
    path = run_manifest.emit_or_verify(tmp_path, stub_run, log=lambda *_: None)
    assert path.is_file()
    # Identical re-entry: verified, resume appended, identity untouched.
    run_manifest.emit_or_verify(tmp_path, stub_run, log=lambda *_: None)
    stored = json.loads(path.read_text())
    assert len(stored["resumes"]) == 1
    assert stored["endpoints"]["model_id"] == "stub-model"
    # Identity drift: refused loud.
    drifted = json.loads(json.dumps(stub_run))
    drifted["endpoints"]["model_id"] = "other-model"
    with pytest.raises(run_manifest.ManifestIdentityError):
        run_manifest.emit_or_verify(tmp_path, drifted, log=lambda *_: None)


def test_a_direct_run_emits_a_manifest_the_reader_finds(
    stub_run, tmp_path
):
    """The bake stamp reaches the reader from a DIRECT run's own run dir.

    Every quotable SR needs a bake identity: two bakes of the same blocks
    share a path and a name, and the SR moves between them with nothing
    failing. ``bench_fullset`` stamped; ``run_benchmark`` invoked straight
    -- the smoke gate, ``launch``, a hand-run arm -- did not, so
    ``run_report`` printed ``bake: UNVERIFIED`` for all of them.

    Pins BOTH halves, because either alone passes while the pair is
    broken: the manifest carries a real content hash, and the reader
    finds it at the depth a direct run actually writes it
    (``<out>/<split>/<stamp>/``, two below the root a reader is handed).
    """
    from benchmarks.citynav.analysis import run_report

    stamped = (stub_run.get("perception") or {}).get("bake_sha256")
    assert stamped and len(stamped) == 64, stub_run.get("perception")
    assert stamped == paths.bake_sha256(), "not the bake on disk"

    run_dir = tmp_path / "out" / "train_seen" / "20260830_120000"
    run_dir.mkdir(parents=True)
    run_manifest.emit_or_verify(run_dir, stub_run, log=lambda *a: None)

    # the root a reader is handed is the out root, not the run dir
    assert run_report.read_manifest(tmp_path / "out") == stub_run
    line = run_report.bake_line(tmp_path / "out")
    assert line.startswith("bake: verified"), line
    assert "UNVERIFIED" not in line

    # VACUITY FLOOR: the reader really needs the extra depth -- a search
    # that stopped one level up would return {} here and this test would
    # be asserting nothing.
    assert run_report.read_manifest(tmp_path / "out" / "train_seen") == stub_run
    assert not (tmp_path / "out" / "run_manifest.json").exists()


def test_the_fleet_root_still_wins_over_a_nested_run_dir(
    stub_run, tmp_path
):
    """Shallowest-first: a fleet root answers with its own manifest.

    The reader now searches two levels deeper, and a fleet root has both
    -- its own at the top and one per shard below. Ordering is the whole
    contract: the fleet's manifest describes the fleet, a shard's
    describes one shard, and a report on the fleet must not quote a
    shard's selection as the run's.
    """
    from benchmarks.citynav.analysis import run_report

    fleet = dict(stub_run, run_id="the-fleet")
    shard = dict(stub_run, run_id="one-shard")
    root = tmp_path / "fleet"
    (root / "train_seen" / "20260830_120000").mkdir(parents=True)
    run_manifest.emit_or_verify(root, fleet, log=lambda *a: None)
    run_manifest.emit_or_verify(
        root / "train_seen" / "20260830_120000", shard, log=lambda *a: None
    )
    assert run_report.read_manifest(root)["run_id"] == "the-fleet"


def test_a_missions_own_effort_replaces_the_arms_in_its_manifest(monkeypatch, tmp_path, bake):
    """The field station's per-mission grain: the mission's stamp rides
    whole under ``mission``, and the effort it names is what
    ``search_env_flags`` records, since that is what its requests carried;
    a mission naming none leaves the arm's stamp, and no mission at all
    leaves no ``mission`` key."""
    monkeypatch.setattr(endpoint_spec_module, "_SPEC", STUB_SPEC)
    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", "0.99")
    common = dict(out_root=tmp_path, config_path=paths.REPO_ROOT / "configs/citynav.yaml", split="calcity_tmp",
                  sample_seed=None, sample_offset=None, limit=None, workers=None, allowlist_path=None,
                  episodes=["calcity::1#1"])
    stamp = {"episode_id": "calcity::1#1", "run_id": "20260917T160207Z", "run_no": 3,
             "reasoning_effort": "0.3", "reasoning_effort_source": "console"}
    m = run_manifest.build_manifest(**common, mission=stamp)
    assert m["mission"] == stamp and m["search_env_flags"]["SEARCH_SAMPLING_REASONING_EFFORT"] == "0.3"
    m = run_manifest.build_manifest(**common, mission={**stamp, "reasoning_effort": None, "reasoning_effort_source": "arm"})
    assert m["search_env_flags"]["SEARCH_SAMPLING_REASONING_EFFORT"] == "0.99"
    assert "mission" not in run_manifest.build_manifest(**common)
