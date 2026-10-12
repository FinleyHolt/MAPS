"""Cal City CLI: the CityNav runner with Cal City selected."""

from __future__ import annotations

from benchmarks.calcity import use


def main() -> int:
    use()
    from benchmarks.citynav.run_benchmark.cli import main as shared_main

    return shared_main()


__all__ = ["main"]
