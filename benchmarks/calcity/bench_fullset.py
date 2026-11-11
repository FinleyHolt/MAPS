"""``python -m benchmarks.calcity.bench_fullset``: the CityNav full-split driver
on the Cal City bench, ``configs/calcity.yaml`` and ``--split calcity`` by
default. Every flag is the shared one (``benchmarks/citynav/bench_fullset.py``);
each attempt's runner learns the bench through the environment the bench
exports.
"""

from __future__ import annotations

import os
import sys

from benchmarks.calcity.paths import calcity_bench
from benchmarks.citynav.bench_fullset import main as shared_main


def main() -> int:
    environ = {"CITYNAV_BENCH_CONFIG": "configs/calcity.yaml", **os.environ}
    if "--split" not in sys.argv[1:]:
        sys.argv[1:] = ["--split", "calcity", *sys.argv[1:]]
    return shared_main(bench=calcity_bench(environ))


if __name__ == "__main__":
    raise SystemExit(main())
