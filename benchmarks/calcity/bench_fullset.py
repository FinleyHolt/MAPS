"""``python -m benchmarks.calcity.bench_fullset``: the CityNav full-split driver
with Cal City selected, ``configs/calcity.yaml`` and ``--split calcity`` by
default. Every flag is the shared one (``benchmarks/citynav/bench_fullset.py``);
each attempt's runner inherits the selection through the environment.
"""

from __future__ import annotations

import os
import sys

from benchmarks.calcity import use


def main() -> int:
    use()
    os.environ.setdefault("CITYNAV_BENCH_CONFIG", "configs/calcity.yaml")
    if "--split" not in sys.argv[1:]:
        sys.argv[1:] = ["--split", "calcity", *sys.argv[1:]]

    from benchmarks.citynav.bench_fullset import main as shared_main

    return shared_main()


if __name__ == "__main__":
    raise SystemExit(main())
