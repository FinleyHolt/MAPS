"""A SIGKILLed bench driver leaves no warm runner behind.

The runner sits in its own session (the driver kills it by process group),
so nothing reaps it when the driver dies mid-episode: it would fly on,
holding its serve slots, and land a row nobody reads. The runner asks the
kernel to kill it with its parent.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

_CHILD = (
    "import time\n"
    "from benchmarks.citynav.run_benchmark.episode_server import _die_with_parent\n"
    "_die_with_parent()\n"
    "print('armed', flush=True)\n"
    "time.sleep(60)\n"
)
_PARENT = (
    "import os, subprocess, sys, time\n"
    "env = {**os.environ, 'CITYNAV_BENCH_PARENT_PID': str(os.getpid())}\n"
    f"c = subprocess.Popen([sys.executable, '-c', {_CHILD!r}], env=env,\n"
    "                     start_new_session=True, stdout=subprocess.PIPE, text=True)\n"
    "c.stdout.readline()\n"
    "print(c.pid, flush=True)\n"
    "time.sleep(60)\n"
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as f:
        return f.read().split()[2] != "Z"


def test_a_sigkilled_driver_takes_its_runner_with_it():
    parent = subprocess.Popen([sys.executable, "-c", _PARENT], cwd=REPO,
                              stdout=subprocess.PIPE, text=True)
    child = int(parent.stdout.readline())
    try:
        assert _alive(child)
        parent.send_signal(signal.SIGKILL)
        parent.wait()
        deadline = time.monotonic() + 5.0
        while _alive(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(child), "the warm runner outlived its driver"
    finally:
        parent.kill()
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
