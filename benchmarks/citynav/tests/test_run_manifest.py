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
    max_output_tokens=4096,
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
        limit=3,
        workers=2,
        allowlist_path=None,
        episodes=["ep_a", "ep_b", "ep_c"],
    )


def test_the_selection_records_what_the_driver_applies(stub_run):
    assert set(stub_run["selection"]) == {
        "sample_seed", "limit", "allowlist", "episodes_n", "selection_sha"}


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
    # A resume under another arm flag (here the reasoning effort) is another arm.
    other_arm = json.loads(json.dumps(stub_run))
    other_arm["search_env_flags"]["SEARCH_SAMPLING_REASONING_EFFORT"] = "0.1"
    with pytest.raises(run_manifest.ManifestIdentityError, match="search_env_flags"):
        run_manifest.emit_or_verify(tmp_path, other_arm, log=lambda *_: None)


def test_a_direct_run_emits_a_manifest_the_reader_finds(
    stub_run, tmp_path
):
    """The bake stamp reaches the reader from a DIRECT run's root.

    Every quotable SR needs a bake identity: two bakes of the same blocks
    share a path and a name, and the SR moves between them with nothing
    failing. Pins both halves: the manifest carries a real content hash,
    and the reader finds it at the run root a direct run writes it to.
    """
    from benchmarks.citynav.analysis import run_report

    stamped = (stub_run.get("perception") or {}).get("bake_sha256")
    assert stamped and len(stamped) == 64, stub_run.get("perception")
    assert stamped == paths.bake_sha256(), "not the bake on disk"

    run_dir = tmp_path / "out"
    run_dir.mkdir()
    run_manifest.emit_or_verify(run_dir, stub_run, log=lambda *a: None)

    assert run_report.read_manifest(tmp_path / "out") == stub_run
    line = run_report.bake_line(tmp_path / "out")
    assert line.startswith("bake: verified"), line
    assert "UNVERIFIED" not in line


def test_a_missions_own_effort_replaces_the_arms_in_its_manifest(monkeypatch, tmp_path, bake):
    """The field station's per-mission grain: the mission's stamp rides
    whole under ``mission``, and the effort it names is what
    ``search_env_flags`` records, since that is what its requests carried;
    a mission naming none leaves the arm's stamp, and no mission at all
    leaves no ``mission`` key."""
    monkeypatch.setattr(endpoint_spec_module, "_SPEC", STUB_SPEC)
    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", "0.99")
    common = dict(out_root=tmp_path, config_path=paths.REPO_ROOT / "configs/citynav.yaml", split="calcity_tmp",
                  sample_seed=None, limit=None, workers=None, allowlist_path=None,
                  episodes=["calcity::1#1"])
    stamp = {"episode_id": "calcity::1#1", "run_id": "20260917T160207Z", "run_no": 3,
             "reasoning_effort": "0.3"}
    m = run_manifest.build_manifest(**common, mission=stamp)
    assert m["mission"] == stamp and m["search_env_flags"]["SEARCH_SAMPLING_REASONING_EFFORT"] == "0.3"
    m = run_manifest.build_manifest(**common, mission={**stamp, "reasoning_effort": None})
    assert m["search_env_flags"]["SEARCH_SAMPLING_REASONING_EFFORT"] == "0.99"
    assert "mission" not in run_manifest.build_manifest(**common)


def test_a_resume_is_refused_under_another_window_or_bake_but_not_another_path(stub_run):
    """The served window and the bake's content are identity; the path the
    bake sits at is the host's (``perception.bake_sha256`` names the bake)."""
    def drift(edit):
        m = json.loads(json.dumps(stub_run))
        edit(m)
        return run_manifest.identity_drift(stub_run, m)

    assert drift(lambda m: m["endpoints"].update(max_model_len=8192))
    assert drift(lambda m: m["perception"].update(bake_sha256="0" * 64))
    assert not drift(lambda m: m["search_env_flags"].update(
        SEARCH_CITYNAV_TILES="/elsewhere/tiles"))


@pytest.mark.parametrize("raw, flown", [(" 0.9 ", "0.9"), ("high\n", "high"), ("  ", None)])
def test_the_effort_is_stamped_as_it_flies(monkeypatch, raw, flown):
    """The request carries the stripped effort, so the stamp, a resume's
    comparison and every reader carry the same one spelling."""
    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", raw)
    run_manifest.settle_effort("inkling-small")
    assert run_manifest.arm_env_stamp().get("SEARCH_SAMPLING_REASONING_EFFORT") == flown


def test_an_effort_the_card_does_not_take_is_refused(monkeypatch):
    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", "0.90")
    with pytest.raises(ValueError, match="'0.9'"):
        run_manifest.settle_effort("inkling-small")


def test_a_root_with_episodes_and_no_manifest_is_refused(stub_run, tmp_path):
    """A ledger with no record of what flew it cannot be resumed into: the
    resume would write a manifest describing only itself."""
    (tmp_path / "episodes.jsonl").write_text('{"episode_id": "e", "status": "ok"}\n')
    with pytest.raises(run_manifest.ManifestIdentityError, match="no run_manifest.json"):
        run_manifest.emit_or_verify(tmp_path, stub_run, log=lambda *_: None)
    assert not (tmp_path / run_manifest.MANIFEST_NAME).exists()


def test_an_unreadable_manifest_is_refused_in_one_line(stub_run, tmp_path):
    path = tmp_path / run_manifest.MANIFEST_NAME
    path.write_text('{"schema": "maps.run_manifest/v1", ')
    with pytest.raises(run_manifest.ManifestIdentityError, match="unreadable") as e:
        run_manifest.emit_or_verify(tmp_path, stub_run, log=lambda *_: None)
    assert str(path) in str(e.value) and "\n" not in str(e.value)


def test_a_failed_manifest_write_leaves_the_old_one_whole(stub_run, tmp_path, monkeypatch):
    """A reader of a running root must never catch a half-written manifest,
    and a resume that dies mid-write must not cut the record it verified."""
    import pathlib

    path = run_manifest.emit_or_verify(tmp_path, stub_run, log=lambda *_: None)
    before = path.read_text()

    def cut(self, text, *a, **k):
        with open(self, "w") as fh:
            fh.write(text[: len(text) // 2])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(pathlib.Path, "write_text", cut)
    monkeypatch.setattr("os.fsync", lambda fd: (_ for _ in ()).throw(OSError(28, "full")))
    with pytest.raises(OSError):
        run_manifest.emit_or_verify(tmp_path, stub_run, log=lambda *_: None)
    assert path.read_text() == before


def test_the_arm_stamp_records_the_resolved_bake(monkeypatch):
    """An unset SEARCH_CITYNAV_TILES stamps the directory it resolves to;
    a set one stamps verbatim."""
    monkeypatch.delenv("SEARCH_CITYNAV_TILES", raising=False)
    assert run_manifest.arm_env_stamp()["SEARCH_CITYNAV_TILES"] == str(paths.TILES_DIR)
    monkeypatch.setenv("SEARCH_CITYNAV_TILES", "/data/tiles/citynav_2m_osmv3")
    assert run_manifest.arm_env_stamp()["SEARCH_CITYNAV_TILES"] == "/data/tiles/citynav_2m_osmv3"


def test_an_unset_effort_is_not_stamped(monkeypatch):
    """A run that named no effort flew the serve's default."""
    monkeypatch.delenv("SEARCH_SAMPLING_REASONING_EFFORT", raising=False)
    assert "SEARCH_SAMPLING_REASONING_EFFORT" not in run_manifest.arm_env_stamp()
