"""``python -m benchmarks.calcity.bench_fullset`` — Cal City full-split driver.

The same wrapper seam as ``benchmarks.calcity.run_benchmark``: name the Cal
City contract, then hand
off to the shared CityNav harness. Every knob is the shared one
(``--workers``, ``--limit``, ``--sample-seed``, ``--episode-allowlist``,
``--out-root``); see
``benchmarks/citynav/bench_fullset.py`` for what they mean.

The env has to be set BEFORE the import: the shared module activates its
bridge and freezes its split constant at import time.
"""

from __future__ import annotations

import os
import sys

SPLIT = "calcity"


def main() -> int:
    # The parent process reads the split and preflights against Cal City's
    # paths contract; the per-episode child re-derives it from the runner
    # module, which sets the rest of the SEARCH_BENCHMARK_* family itself.
    os.environ.setdefault(
        "SEARCH_BENCHMARK_BRIDGE", "benchmarks.calcity._citynav_bridge"
    )
    os.environ.setdefault(
        "SEARCH_BENCHMARK_RUNNER_MODULE", "benchmarks.calcity.run_benchmark"
    )
    os.environ.setdefault("CITYNAV_BENCH_CONFIG", "configs/calcity.yaml")

    # The shared harness bakes DEFAULT_SPLIT to CityNav's train_seen at
    # import. Cal City has exactly one split, so supply it rather than make
    # every caller remember to.
    argv = sys.argv[1:]
    if "--split" not in argv:
        sys.argv[1:] = ["--split", SPLIT, *argv]

    from benchmarks.citynav.bench_fullset import main as shared_main

    return shared_main()


if __name__ == "__main__":
    raise SystemExit(main())
