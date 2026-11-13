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
    assert sandbox.bwrap_works()
    assert asked == [sandbox.bwrap_argv([]) + ["true"]]


def test_an_inherited_marker_outside_the_sandbox_is_not_trusted(monkeypatch):
    """A process started with ``MAPS_SANDBOXED=1`` in its environment, but no sandbox
    around it, logs the unconfined line."""
    lines: list[str] = []
    monkeypatch.setenv(sandbox.SANDBOXED_ENV, "1")
    monkeypatch.setattr(sandbox, "cap_memory", lambda gib: None)
    monkeypatch.setattr(sandbox, "bwrap_works", lambda: False)
    sandbox.confine([], 1.0, lines.append)
    assert lines == [sandbox.mode_line(False, 1.0)]
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
    not sandbox.bwrap_works(), reason="bwrap is absent or this host blocks user namespaces"
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
