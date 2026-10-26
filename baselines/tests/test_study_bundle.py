"""The participant's zip: the launchers keep their execute bit, Windows files carry CRLF, the
code rides without the bench's corpus, a kit's results never ship, and the lead's contact
address reaches the READ ME and the kit. The Mac launcher works from its own folder, in UTF-8,
offline once set up, and says so plainly when setup did not finish."""

import os
import shutil
import subprocess
import zipfile

import pytest

from baselines.human.study_package import BUNDLE, build_bundle, main


def test_bundle(tmp_path) -> None:
    kit = tmp_path / "kit"
    (kit / "plan").mkdir(parents=True)
    (kit / "plan" / "plan.json").write_text("{}")
    (kit / "results" / "p01").mkdir(parents=True)
    (kit / "results" / "p01" / "answers.jsonl").write_text("{}\n")
    build_bundle(kit, tmp_path / "b.zip", contact="lead@example.org")
    z = zipfile.ZipFile(tmp_path / "b.zip")
    names = set(z.namelist())
    mode = lambda n: z.getinfo(n).external_attr >> 16
    assert mode("MAPS-study/Start study (Mac).command") & 0o111 and mode("MAPS-study/app/run.sh") & 0o111
    assert b"\r\n" in z.read("MAPS-study/Start study (Windows).bat")
    assert "MAPS-study/app/baselines/human/study_server.py" in names and "MAPS-study/kit/plan/plan.json" in names
    assert not any("/tests/" in n or "/calcity/data/" in n or "/results/" in n for n in names)
    readme = z.read("MAPS-study/READ ME FIRST.txt").decode()
    assert "lead@example.org" in readme and "{CONTACT}" not in readme and "\r\n" in readme
    assert z.read("MAPS-study/kit/contact.txt").decode().strip() == "lead@example.org"
    assert sorted(n for n in names if n.count("/") == 1) == [
        "MAPS-study/READ ME FIRST.txt", "MAPS-study/Start study (Mac).command", "MAPS-study/Start study (Windows).bat"]
    with pytest.raises(SystemExit) as e:
        main(["bundle", "--kit", str(kit), "--out", str(tmp_path / "c.zip")])
    assert e.value.code == 2


def _unzipped(tmp_path, uv: str | None):
    """The launchers as the zip lays them out, with a stand-in uv that echoes how it was run."""
    root = tmp_path / "MAPS study's (1)"
    (root / "app").mkdir(parents=True)
    shutil.copy(BUNDLE / "mac.command", root / "Start study (Mac).command")
    shutil.copy(BUNDLE / "run.sh", root / "app" / "run.sh")
    if uv is not None:
        (root / "app" / "uv").mkdir()
        (root / "app" / "uv" / "uv").write_text(uv)
        (root / "app" / "uv" / "uv").chmod(0o755)
    return root


def test_mac_launcher_once_set_up(tmp_path) -> None:
    root = _unzipped(tmp_path, '#!/bin/sh\necho "$PWD|$PYTHONUTF8|$*"\n')
    (root / "app" / "uv" / "ready").touch()
    out = subprocess.run(["sh", str(root / "Start study (Mac).command")], cwd=tmp_path, capture_output=True,
                         text=True, check=True).stdout.splitlines()[-1]
    cwd, utf8, args = out.split("|")
    assert cwd == str(root) and utf8 == "1" and args.startswith("run --offline ")
    assert args.endswith(f"--kit {root}/kit")


def test_mac_launcher_says_when_setup_did_not_finish(tmp_path) -> None:
    root = _unzipped(tmp_path, None)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "curl").write_text("#!/bin/sh\nexit 6\n")
    (tmp_path / "bin" / "curl").chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin"}
    r = subprocess.run(["sh", str(root / "Start study (Mac).command")], cwd=tmp_path, capture_output=True,
                       text=True, env=env)
    assert r.returncode == 1 and "Setup did not finish" in r.stdout
    assert not (root / "app" / "uv" / "ready").exists()
