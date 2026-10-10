"""Cal City CLI backed by the shared navigation execution engine."""

from __future__ import annotations

import os

from benchmarks.calcity._citynav_bridge import activate


def main() -> int:
    os.environ["SEARCH_BENCHMARK_LABEL"] = "Cal City"
    os.environ["SEARCH_BENCHMARK_DEFAULT_CONFIG"] = "configs/calcity.yaml"
    os.environ["SEARCH_BENCHMARK_DEFAULT_SPLIT"] = "calcity"
    os.environ["SEARCH_BENCHMARK_OUTPUT_ROOT"] = "output/calcity"
    os.environ["SEARCH_BENCHMARK_NAME"] = "calcity"
    activate()
    from benchmarks.citynav.run_benchmark.cli import main as shared_main

    return shared_main()


__all__ = ["main"]
