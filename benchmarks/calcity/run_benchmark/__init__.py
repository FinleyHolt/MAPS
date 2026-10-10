"""Cal City CLI backed by the shared navigation execution engine."""

from __future__ import annotations

import os

from benchmarks.calcity._citynav_bridge import activate


def main() -> int:
    os.environ["SEARCH_BENCHMARK_NAME"] = "calcity"
    activate()
    from benchmarks.citynav.run_benchmark.cli import main as shared_main

    return shared_main(label="Cal City", default_split="calcity", output_root="output/calcity")


__all__ = ["main"]
