"""``python -m benchmarks.calcity.run_benchmark``: the CityNav runner on the Cal City bench."""

from __future__ import annotations

import os

from benchmarks.calcity.paths import calcity_bench
from benchmarks.citynav.run_benchmark.cli import main as shared_main


def main() -> int:
    return shared_main(calcity_bench(os.environ))


if __name__ == "__main__":
    raise SystemExit(main())
