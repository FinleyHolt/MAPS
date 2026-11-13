"""The OS isolation model code runs under: the memory cap, and bubblewrap where it runs."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from benchmarks import sandbox

REPO = Path(sandbox.__file__).resolve().parents[1]

# A mission's interpreter on a fake episode: one block past the cap, then a normal one.
_BLOCKS = (
    "from maps.library.interpreter import MissionPython\n"
    "from maps.mission.tests.fakes import quiet_env\n"
    "mp = MissionPython(quiet_env())\n"
    "print(mp.run_block('x = [0] * (6 * 10**8)\\nprint(len(x))').get('error'))\n"
    "print(mp.run_block('print(sum(range(10)))').get('printed'))\n"
)


def test_a_block_past_the_cap_is_a_served_memory_error_and_the_next_block_runs():
    out = subprocess.run(
        sandbox.cap_argv(sandbox.ATTEMPT_MEMORY_GIB) + [sys.executable, "-c", _BLOCKS],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert out[0].startswith("line 1: MemoryError"), out
    assert out[1] == "45", out


_CONFINED = (
    "import os, sys, urllib.request\n"
    "try:\n"
    "    open(os.path.expanduser('~/maps_sandbox_probe'), 'w')\n"
    "    print('home writable')\n"
    "except OSError:\n"
    "    print('home read-only')\n"
    "open(os.path.join(sys.argv[1], 'landed'), 'w').write('x')\n"
    "print(urllib.request.urlopen(sys.argv[2], timeout=10).status)\n"
)


@pytest.mark.skipif(
    not sandbox.bwrap_works(), reason="bwrap is absent or this host blocks user namespaces"
)
def test_under_bubblewrap_writes_land_only_in_the_binds_and_localhost_answers(tmp_path):
    class Ok(BaseHTTPRequestHandler):
        def do_GET(self):  # http.server's name
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Ok)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = subprocess.run(
            sandbox.bwrap_argv([tmp_path])
            + [
                sys.executable,
                "-c",
                _CONFINED,
                str(tmp_path),
                f"http://127.0.0.1:{server.server_port}/",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    finally:
        server.shutdown()
    assert out == ["home read-only", "200"], out
    assert (tmp_path / "landed").read_text() == "x"
    assert not os.path.exists(os.path.expanduser("~/maps_sandbox_probe"))
