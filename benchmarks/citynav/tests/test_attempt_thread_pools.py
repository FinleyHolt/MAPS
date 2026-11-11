"""An attempt child starts its native thread pools at one thread each.

Uncapped, every child starts BLAS, OpenCV and tokenizer pools of one
thread per core (about 1,000 per child on a 256-thread host), and a thread
that fails to start crashes its episode. A value the shell set wins.
"""

from __future__ import annotations

import os
import subprocess
import sys

from benchmarks.citynav import bench_fullset as bf
from benchmarks.citynav.paths import citynav_bench

_COUNT = (
    "import os, cv2, numpy as np\n"
    "a = np.random.default_rng(0).random((400, 400)); a @ a\n"
    "cv2.resize(np.zeros((2000, 2000, 3), np.uint8), (1000, 1000))\n"
    "print(len(os.listdir('/proc/self/task')))\n"
)
_POOLS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "OPENCV_FOR_THREADS_NUM", "RAYON_NUM_THREADS", "MALLOC_ARENA_MAX")


def test_a_runner_starts_with_its_native_thread_pools_capped(monkeypatch):
    for name in _POOLS:
        monkeypatch.delenv(name, raising=False)
    out = subprocess.run([sys.executable, "-c", _COUNT], env=bf._runner_env(citynav_bench(os.environ)),
                         capture_output=True, text=True, check=True).stdout
    assert int(out) <= 4, out


def test_a_pool_size_the_shell_set_wins(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    assert bf._runner_env(citynav_bench(os.environ))["OMP_NUM_THREADS"] == "8"
