"""The participant's zip: the launchers keep their execute bit, Windows files carry CRLF, the
code rides without the bench's corpus, a kit's results never ship, and the lead's contact
address reaches the READ ME and the kit. The Mac launcher works from its own folder, in UTF-8,
offline once set up, and says so plainly when setup did not finish."""

import hashlib
import json
import os
import shutil
import subprocess
import zipfile

import pytest

from baselines.human.study_package import (BUNDLE, REQUIRED, STUDY_REPORTS, build_bundle, main, off_manifest, unlocked,
                                           write_lock)


def _kit(root):
    """The four files a kit cannot do without, and a pilot's results."""
    for rel in REQUIRED:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("{}")
    return root


def _bundle(kit, out, **kw):
    """A bundle off a toy kit, with a study folder frozen from it beside the kit."""
    study = kit.parent / "study"
    shutil.rmtree(study, ignore_errors=True)
    study.mkdir()
    shutil.copy2(STUDY_REPORTS, study / "reports.json")
    (study / "targets.json").write_text('{"targets": {}}')
    shutil.copytree(kit / "plan", study / "plan")
    write_lock(kit, study)
    return build_bundle(kit, out, study=study, **kw)


def test_bundle(tmp_path) -> None:
    kit = _kit(tmp_path / "kit")
    (kit / "results" / "p01").mkdir(parents=True)
    (kit / "results" / "p01" / "answers.jsonl").write_text("{}\n")
    (kit / "roster.json").write_text('{"p01": "Tyler"}')
    # The output's folder need not exist yet.
    _bundle(kit, tmp_path / "new" / "b.zip", contact="lead@example.org")
    z = zipfile.ZipFile(tmp_path / "new" / "b.zip")
    names = set(z.namelist())
    mode = lambda n: z.getinfo(n).external_attr >> 16
    assert mode("MAPS-study/Start study (Mac).command") & 0o111 and mode("MAPS-study/app/run.sh") & 0o111
    assert mode("MAPS-study/Start study (Linux).sh") & 0o111
    assert b"\r\n" in z.read("MAPS-study/Start study (Windows).bat")
    assert "MAPS-study/app/baselines/human/study_server.py" in names and "MAPS-study/kit/plan/plan.json" in names
    assert not any("/tests/" in n or "/calcity/data/" in n or "/results/" in n for n in names)
    # First names stay with the lead; the download never carries them.
    assert "MAPS-study/kit/roster.json" not in names and "MAPS-study/app/baselines/human/information.md" in names
    readme = z.read("MAPS-study/READ ME FIRST.txt").decode()
    assert "lead@example.org" in readme and "{CONTACT}" not in readme and "\r\n" in readme
    assert z.read("MAPS-study/kit/contact.txt").decode().strip() == "lead@example.org"
    assert sorted(n for n in names if n.count("/") == 1) == [
        "MAPS-study/READ ME FIRST.txt", "MAPS-study/Start study (Linux).sh", "MAPS-study/Start study (Mac).command",
        "MAPS-study/Start study (Windows).bat"]
    video = tmp_path / "t.mp4"
    video.write_bytes(b"\0" * 64)
    _bundle(kit, tmp_path / "v.zip", contact="lead@example.org", video=video)
    info = zipfile.ZipFile(tmp_path / "v.zip").getinfo("MAPS-study/Watch first - tutorial.mp4")
    assert info.file_size == 64 and info.compress_type == zipfile.ZIP_STORED
    with pytest.raises(SystemExit) as e:
        main(["bundle", "--kit", str(kit), "--out", str(tmp_path / "c.zip")])
    assert e.value.code == 2


def test_bundle_refuses_an_incomplete_kit(tmp_path) -> None:
    kit = _kit(tmp_path / "kit")
    (kit / "calcity_ortho.mbtiles").unlink()
    with pytest.raises(SystemExit, match="calcity_ortho.mbtiles"):
        _bundle(kit, tmp_path / "b.zip", contact="lead@example.org")
    assert not (tmp_path / "b.zip").exists()


def test_bundle_follows_a_kit_of_symlinked_folders(tmp_path) -> None:
    real = _kit(tmp_path / "real")
    kit = tmp_path / "kit"
    kit.mkdir()
    for p in real.iterdir():
        (kit / p.name).symlink_to(p)
    _bundle(kit, tmp_path / "b.zip", contact="lead@example.org")
    names = set(zipfile.ZipFile(tmp_path / "b.zip").namelist())
    assert {f"MAPS-study/kit/{rel}" for rel in REQUIRED} <= names


def _unzipped(tmp_path):
    """The launchers as the zip lays them out, a stand-in curl whose installer drops a stand-in uv
    that logs how it was run (and fails offline when ``uv/fail`` exists), and a PATH holding it."""
    root = tmp_path / "MAPS study's (1)"
    (root / "app").mkdir(parents=True)
    shutil.copy(BUNDLE / "mac.command", root / "Start study (Mac).command")
    shutil.copy(BUNDLE / "run.sh", root / "app" / "run.sh")
    uv = ('#!/bin/sh\necho "$PWD|$PYTHONUTF8|$UV_CACHE_DIR|${PYTHONPATH-unset}|$*" >> "$(dirname "$0")/log"\n'
          'case "$*" in *--offline*) [ ! -f "$(dirname "$0")/fail" ];; esac\n')
    installer = f"mkdir -p \"$UV_INSTALL_DIR\"\ncat > \"$UV_INSTALL_DIR/uv\" <<'UV'\n{uv}UV\nchmod +x \"$UV_INSTALL_DIR/uv\"\n"
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "curl").write_text(f"#!/bin/sh\ncat <<'SH'\n{installer}SH\n")
    (tmp_path / "bin" / "curl").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin"}
    start = lambda: subprocess.run(["sh", str(root / "Start study (Mac).command")], cwd=tmp_path,
                                   capture_output=True, text=True, env=env)
    return root, start


def test_mac_launcher_sets_up_once_then_starts_offline(tmp_path) -> None:
    root, start = _unzipped(tmp_path)
    assert start().returncode == 0 and (root / "app" / "uv" / "ready").is_file()
    (root / "app" / "uv" / "log").unlink()
    (tmp_path / "bin" / "curl").write_text("#!/bin/sh\nexit 6\n")
    r = start()
    assert r.returncode == 0 and "First start" not in r.stdout
    runs = [line.split("|") for line in (root / "app" / "uv" / "log").read_text().splitlines()]
    assert all(a.startswith("run --offline --managed-python ") for *_, a in runs)
    cwd, utf8, cache, pythonpath, args = runs[-1]
    assert cwd == str(root / "app") and utf8 == "1" and cache == str(root / "app" / "uv" / "cache")
    assert pythonpath == "unset" and args.endswith(f"python -m baselines.human.study_server --kit {root}/kit")


def test_mac_launcher_offline_failure_asks_for_an_online_start(tmp_path) -> None:
    root, start = _unzipped(tmp_path)
    start()
    (root / "app" / "uv" / "fail").touch()
    r = start()
    assert r.returncode == 1 and not (root / "app" / "uv" / "ready").exists()
    assert ("The study stopped. Close this window and start the study again; that start needs the internet."
            in r.stdout)


def test_mac_launcher_says_when_setup_did_not_finish(tmp_path) -> None:
    root, start = _unzipped(tmp_path)
    (tmp_path / "bin" / "curl").write_text("#!/bin/sh\nexit 6\n")
    r = start()
    assert r.returncode == 1 and "Setup did not finish" in r.stdout
    assert not (root / "app" / "uv" / "ready").exists()


def test_windows_launcher_lines() -> None:
    bat = (BUNDLE / "windows.bat").read_text().splitlines()
    for line in ("setlocal EnableExtensions DisableDelayedExpansion", 'set "PYTHONUTF8=1"',
                 'set "UV_PYTHON_ARCH=x86_64"', 'set "UV_CACHE_DIR=%~dp0app\\uv\\cache"',
                 'if not exist "kit\\console\\index.html" goto inzip'):
        assert line in bat
    runs = [line for line in bat if line.startswith('"%UV%" run')]
    assert runs and all("--managed-python" in line for line in runs)
    assert sum("--offline" in line for line in runs) == 2 and not any("PYTHONPATH" in line for line in bat)


def _transcript(path, *contents) -> None:
    """An attempt's LLM wire log: one request and one response per reply, the last MAPS's answer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"kind": "params"}]
    for c in contents:
        rows += [{"kind": "request"}, {"kind": "response", "response": {"choices": [
            {"finish_reason": "stop", "message": {"role": "assistant", "content": c}}]}}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_maps_reply_is_its_last_message_and_lands_beside_the_attempts_records(tmp_path) -> None:
    from baselines.human.study_package import copy_records, reply_text

    root = tmp_path / "root"
    wd = "a/w0/maps_1_calcity_x"
    (root / wd).mkdir(parents=True)
    (root / wd / "conversation.jsonl").write_text("")
    (root / "episodes.jsonl").write_text(json.dumps({"episode_id": "calcity::1#0", "status": "ok", "workdir": wd}) + "\n"
                                         + json.dumps({"episode_id": "calcity::2#0", "status": "ok", "workdir": "a/w0/none"}) + "\n")
    _transcript(root / wd / "flight/llm/transcript.jsonl", "\n\n", "\n\n**Answer: `prior_7`**, the house.\n")
    assert reply_text(root / wd / "flight/llm/transcript.jsonl") == "**Answer: `prior_7`**, the house."
    assert reply_text(tmp_path / "missing.jsonl") == ""
    out = tmp_path / "kit" / "maps_root"
    assert copy_records(root, out) == 2
    assert (out / wd / "reply.md").read_text() == "**Answer: `prior_7`**, the house."
    assert not (out / "a/w0/none/reply.md").exists()
    # A kit built before replies were carried takes them from a mirror of the transcripts alone.
    (out / wd / "reply.md").unlink()
    mirror = tmp_path / "transcripts"
    _transcript(mirror / wd / "flight/llm/transcript.jsonl", "the house at the corner")
    assert main(["replies", "--transcripts", str(mirror), "--kit", str(tmp_path / "kit")]) == 0
    assert (out / wd / "reply.md").read_text() == "the house at the corner"


def test_the_study_manifest_names_real_reports_and_gates_the_bundle(tmp_path) -> None:
    doc = json.loads(STUDY_REPORTS.read_text())
    corpus = json.loads((STUDY_REPORTS.parents[3] / "benchmarks/calcity/data/calcity.json").read_text())
    for eid in doc["session"] + doc["practice"] + doc["omitted"]:
        ann, row = eid.removeprefix("calcity::").split("#")
        assert corpus[int(row)]["ann_ids"][0] == int(ann), eid
    assert len(set(doc["session"])) == len(doc["session"]) and not set(doc["session"]) & set(doc["omitted"])
    # The study's own record: a target for every session report, and the lock matching the files in git.
    study = STUDY_REPORTS.parent
    assert set(json.loads((study / "targets.json").read_text())["targets"]) == set(doc["session"])
    lock = json.loads((study / "lock.json").read_text())["files"]
    for rel, sha in lock.items():
        if not rel.startswith("kit/"):
            assert hashlib.sha256((study / rel).read_bytes()).hexdigest() == sha, rel
    assert {ln for f in (study / "plan").glob("*/*.txt") for ln in f.read_text().split()} <= set(doc["session"]) | set(doc["practice"])
    kit = _kit(tmp_path / "kit")
    (kit / "plan" / "p01").mkdir()
    (kit / "plan" / "p01" / "team.txt").write_text(doc["session"][0] + "\n")
    assert off_manifest(kit) == []
    (kit / "plan" / "p01" / "team.txt").write_text(doc["omitted"][0] + "\n")
    assert off_manifest(kit) == [doc["omitted"][0]]
    with pytest.raises(SystemExit, match="outside reports.json"):
        _bundle(kit, tmp_path / "b.zip", contact="lead@example.org")


def test_the_download_carries_no_target_and_the_lock_catches_any_drift(tmp_path) -> None:
    kit = _kit(tmp_path / "kit")
    row = {"episode_id": "calcity::1#0", "description": "the pool", "target_xy": [1.0, 2.0], "status": "ok",
           "workdir": "w0/x", "metrics": {"SR": 1, "NE": 0.5, "steps_taken": 12, "precise_target": "shape"}}
    (kit / "maps_root" / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    _bundle(kit, tmp_path / "b.zip", contact="lead@example.org")
    shipped = json.loads(zipfile.ZipFile(tmp_path / "b.zip").read("MAPS-study/kit/maps_root/episodes.jsonl"))
    assert "target_xy" not in shipped and shipped["metrics"] == {"steps_taken": 12} and shipped["description"] == "the pool"
    names = zipfile.ZipFile(tmp_path / "b.zip").namelist()
    assert not any(n.endswith(("targets.json", "lock.json")) for n in names)
    study = tmp_path / "study"
    assert unlocked(kit, study, tmp_path / "b.zip") == []
    (kit / "plan" / "plan.json").write_text('{"participants": {}}')
    assert unlocked(kit, study) == ["kit/plan/plan.json: not the study's plan"]
    with pytest.raises(SystemExit, match="lock.json"):
        build_bundle(kit, tmp_path / "c.zip", contact="lead@example.org", study=study)
    (study / "targets.json").write_text('{"targets": {"x": 1}}')
    assert "targets.json: changed since the freeze" in unlocked(kit, study)
