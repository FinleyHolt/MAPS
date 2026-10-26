"""The participant's zip: the launchers keep their execute bit, Windows files carry CRLF, the
code rides without the bench's corpus, a kit's results never ship, and the lead's contact
address reaches the READ ME and the kit. The Mac launcher works from its own folder, in UTF-8,
offline once set up, and says so plainly when setup did not finish."""

import os
import shutil
import subprocess
import zipfile

import pytest

from baselines.human.study_package import BUNDLE, REQUIRED, build_bundle, main


def _kit(root):
    """The four files a kit cannot do without, and a pilot's results."""
    for rel in REQUIRED:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("{}")
    return root


def test_bundle(tmp_path) -> None:
    kit = _kit(tmp_path / "kit")
    (kit / "results" / "p01").mkdir(parents=True)
    (kit / "results" / "p01" / "answers.jsonl").write_text("{}\n")
    # The output's folder need not exist yet.
    build_bundle(kit, tmp_path / "new" / "b.zip", contact="lead@example.org")
    z = zipfile.ZipFile(tmp_path / "new" / "b.zip")
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


def test_bundle_refuses_an_incomplete_kit(tmp_path) -> None:
    kit = _kit(tmp_path / "kit")
    (kit / "calcity_ortho.mbtiles").unlink()
    with pytest.raises(SystemExit, match="calcity_ortho.mbtiles"):
        build_bundle(kit, tmp_path / "b.zip", contact="lead@example.org")
    assert not (tmp_path / "b.zip").exists()


def test_bundle_follows_a_kit_of_symlinked_folders(tmp_path) -> None:
    real = _kit(tmp_path / "real")
    kit = tmp_path / "kit"
    kit.mkdir()
    for p in real.iterdir():
        (kit / p.name).symlink_to(p)
    build_bundle(kit, tmp_path / "b.zip", contact="lead@example.org")
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
