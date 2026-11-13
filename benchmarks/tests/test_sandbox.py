"""The OS isolation model code runs under: bubblewrap where it runs, and its probe."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from benchmarks import sandbox


def test_the_probe_builds_the_sandbox_the_attempts_run_under(monkeypatch):
    """A host that refuses a part of the real sandbox (a ``/proc`` mount) falls back to
    running without bubblewrap instead of booking every attempt infra."""
    asked: list = []
    monkeypatch.setattr(sandbox.shutil, "which", lambda exe: f"/usr/bin/{exe}")
    monkeypatch.setattr(
        sandbox.subprocess,
        "run",
        lambda argv, **kw: asked.append(argv) or subprocess.CompletedProcess(argv, 0),
    )
    assert sandbox.bwrap_refusal() is None
    assert asked == [sandbox.bwrap_argv([]) + ["true"]]


def test_an_inherited_marker_outside_the_sandbox_is_not_trusted(monkeypatch):
    """A process started with ``MAPS_SANDBOXED=1`` in its environment, but no sandbox
    around it, logs the unconfined line."""
    lines: list[str] = []
    monkeypatch.setenv(sandbox.SANDBOXED_ENV, "1")
    monkeypatch.setattr(sandbox, "cap_memory", lambda gib: None)
    monkeypatch.setattr(sandbox, "bwrap_refusal", lambda: "bwrap is not installed")
    # An environment already the allowlist: the process is not restarted.
    monkeypatch.setattr(sandbox, "child_env", lambda environ: dict(environ))
    _catch_exec(monkeypatch)
    sandbox.confine([], 1.0, lines.append)
    assert lines == [sandbox.mode_line("bwrap is not installed", 1.0)]
    assert sandbox.SANDBOXED_ENV not in os.environ


# What a block reaches inside: each probe prints one word. The test never reads a secret;
# it asks only whether a path exists, is empty, or accepts a write.
_CONFINED = r"""
import os, socket, sys, urllib.request
rw, stray, url = sys.argv[1:4]
home = os.path.expanduser("~")
def wrote(p):
    try:
        open(p, "w").write("x")
        return "wrote"
    except OSError:
        return "refused"
def bus():
    s = socket.socket(socket.AF_UNIX)
    try:
        s.connect(f"/run/user/{os.getuid()}/bus")
        return "bus"
    except OSError:
        return "no-bus"
print(wrote(os.path.join(rw, "landed")))
print(wrote(os.path.join(stray, "stray")))
print(wrote("/tmp/maps_sandbox_probe"), wrote("/dev/shm/maps_sandbox_probe"))
print(wrote(os.path.join(home, "maps_sandbox_probe")))
print("ssh" if os.path.exists(os.path.join(home, ".ssh")) else "no-ssh")
env = os.path.join(sys.argv[4], ".env")
print("env-masked" if not os.path.exists(env) or os.path.getsize(env) == 0 else "env-open")
print(bus())
print(urllib.request.urlopen(url, timeout=10).status)
"""


@pytest.mark.skipif(
    sandbox.bwrap_refusal() is not None,
    reason="bwrap is absent or this host blocks user namespaces",
)
def test_under_bubblewrap_writes_land_only_in_the_binds_and_localhost_answers(tmp_path):
    """The explicit bind is the only writable route to the host: ``/tmp``, ``/dev/shm`` and
    home are empty scratch inside, secrets and the user's D-Bus are gone, localhost
    answers."""

    class Ok(BaseHTTPRequestHandler):
        def do_GET(self):  # http.server's name
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_a):
            pass

    rw, stray = tmp_path / "rw", tmp_path / "stray"
    stray.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Ok)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"
    try:
        out = subprocess.run(
            sandbox.bwrap_argv([rw], [stray])
            + [sys.executable, "-c", _CONFINED, str(rw), str(stray), url, str(sandbox.REPO)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    finally:
        server.shutdown()
    assert out == [
        "wrote",
        "refused",
        "wrote wrote",
        "wrote",
        "no-ssh",
        "env-masked",
        "no-bus",
        "200",
    ], out
    assert (rw / "landed").read_text() == "x"
    home = os.path.expanduser("~")
    for p in (
        "/tmp/maps_sandbox_probe",
        "/dev/shm/maps_sandbox_probe",
        f"{home}/maps_sandbox_probe",
    ):
        assert not os.path.exists(p), p


@pytest.mark.parametrize("raw", ["-1", "nan", "inf", "0", "0.5"])
def test_a_memory_cap_that_is_no_cap_or_starves_the_process_is_refused(raw, capsys):
    """-1 lifts the cap, nan books every attempt infra, a tiny cap crashes every child."""
    import argparse

    ap = argparse.ArgumentParser()
    sandbox.add_memory_flag(ap)
    with pytest.raises(SystemExit):
        ap.parse_args(["--memory-gib", raw])
    assert "at least 1 GiB" in capsys.readouterr().err
    assert ap.parse_args(["--memory-gib", "2.5"]).memory_gib == 2.5


def test_the_unconfined_line_names_why_bubblewrap_does_not_run(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda exe: None)
    assert sandbox.bwrap_refusal() == "bwrap is not installed"
    monkeypatch.setattr(sandbox.shutil, "which", lambda exe: f"/usr/bin/{exe}")
    monkeypatch.setattr(
        sandbox.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, "", "bwrap: setting up uid map\n"),
    )
    why = sandbox.bwrap_refusal()
    assert why == "bwrap cannot build the sandbox here (bwrap: setting up uid map)", why
    assert sandbox.mode_line(why, 4.0).startswith(f"sandbox: {why}; model code runs without")


def test_a_bound_path_reached_through_a_symlink_in_home_resolves_inside(tmp_path, monkeypatch):
    """Home is empty inside, so a symlink there on the way to a bound path (a venv's
    interpreter into uv's pythons, a data directory linked from home) is made again."""
    home, data = tmp_path / "home", tmp_path / "data"
    (home / "d").mkdir(parents=True)
    data.mkdir()
    (home / "d" / "link").symlink_to(data)
    monkeypatch.setenv("HOME", str(home))
    argv = sandbox.bwrap_argv([], [home / "d" / "link"])
    i = argv.index("--symlink")
    assert argv[i - 2 : i + 3] == [
        "--dir",
        str(home / "d"),
        "--symlink",
        str(data),
        str(home / "d" / "link"),
    ]
    binds = [k for k, a in enumerate(argv) if a == "--ro-bind" and argv[k + 1] == str(data)]
    assert argv.index(str(home)) < i < binds[0]


@pytest.mark.skipif(not sandbox.scope_works(), reason="no systemd user manager here")
def test_a_failed_attempt_scope_leaves_no_unit_behind():
    """An attempt that fails (an OOM kill, an error exit) unloads its scope with it, so
    the user manager does not collect one failed unit per crashed attempt."""
    unit = f"maps-attempt-test-{os.getpid()}"
    subprocess.run(
        sandbox.scope_argv(unit, 1.0) + [sys.executable, "-c", "b = b'x' * 2**31"],
        capture_output=True,
    )
    left = subprocess.run(
        ["systemctl", "--user", "list-units", f"{unit}.scope", "--all", "--no-legend"],
        capture_output=True,
        text=True,
    ).stdout
    subprocess.run(["systemctl", "--user", "reset-failed", f"{unit}.scope"], capture_output=True)
    assert left == "", left


_POOL_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
    "MALLOC_ARENA_MAX",
)


def test_a_capped_process_imports_its_native_libraries_on_any_core_count(monkeypatch):
    """``RLIMIT_DATA`` counts each thread's reserved stack and OpenBLAS's per-thread
    buffers: on a 32-core node numpy, OpenCV and SciPy reserve 6.7 GiB at import with
    0.1 GiB resident, so the environment model code runs under starts one thread per
    pool, and the shell's own setting wins."""
    for name in _POOL_VARS:
        monkeypatch.delenv(name, raising=False)
    env = sandbox.child_env(os.environ)
    assert {k: env[k] for k in _POOL_VARS} == dict.fromkeys(_POOL_VARS[:3], "1") | {
        "MALLOC_ARENA_MAX": "2"
    }
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    assert sandbox.child_env(os.environ)["OMP_NUM_THREADS"] == "8"
    probe = (
        "from benchmarks import sandbox; sandbox.cap_memory(1.0); import numpy, cv2, scipy.ndimage"
    )
    subprocess.run([sys.executable, "-c", probe], env=env, cwd=sandbox.REPO, check=True)


class _Execed(Exception):
    pass


def _catch_exec(monkeypatch) -> list:
    """Every exec this process asks for, as ``(argv, env)``, without exec'ing."""
    seen: list = []

    def execed(file, argv, env=None):
        seen.append((list(argv), dict(os.environ if env is None else env)))
        raise _Execed

    monkeypatch.setattr(os, "execvp", lambda f, a: execed(f, a))
    monkeypatch.setattr(os, "execvpe", execed)
    monkeypatch.setattr(os, "execve", execed)
    return seen


@pytest.mark.parametrize("bwrap", [True, False])
def test_a_confined_process_restarts_under_the_allowlist_and_a_read_only_hub(
    bwrap, tmp_path, monkeypatch
):
    """The station, the twin CLI and the runner re-execute themselves with the
    allowlisted environment (a launcher's FAKE_SECRET gone, the reasoner key kept, one
    thread per native pool); under bubblewrap the hub cache is bound read-only and
    the hub is offline, after the launcher cached the reasoner's tokenizer."""
    monkeypatch.setenv("FAKE_SECRET", "s")
    monkeypatch.setenv("SEARCH_REASONER_API_KEY", "k")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.setattr(sandbox, "cap_memory", lambda gib: None)
    monkeypatch.setattr(sandbox, "bwrap_refusal", lambda: None if bwrap else "bwrap is absent")
    cached: list = []
    monkeypatch.setattr(sandbox, "_cache_tokenizer", lambda log: cached.append(1), raising=False)
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setattr(sandbox, "hf_hub_cache", lambda: str(hub))
    seen = _catch_exec(monkeypatch)
    with pytest.raises(_Execed):
        sandbox.confine([tmp_path / "out"], 1.0, lambda line: None, flies=True)
    ((argv, env),) = seen
    assert "FAKE_SECRET" not in env and env["SEARCH_REASONER_API_KEY"] == "k"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    if bwrap:
        assert env["HF_HUB_OFFLINE"] == "1" and cached == [1]
        binds = [argv[i - 1] for i, a in enumerate(argv) if a == str(hub) and argv[i - 1] != a]
        assert binds == ["--ro-bind"], binds
    else:
        assert argv[0] == sys.executable
