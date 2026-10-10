"""CityNav benchmark runner (the ``citynav-bench`` entry point).

The episode engine is a package of purpose-named modules: ``cli`` (main),
``dispatch`` -> ``episode_runner`` (per-episode flow) -> ``mission_engine``
(the one-conversation fly-on-request executor) -> ``trajectory`` (discrete
replay), with ``assets`` / ``perception`` / ``osm_prior`` / ``tour_cursor``
/ ``spawn`` / ``reasoner_warmup`` as leaves and ``config`` /
``error_taxonomy`` / ``episode_server`` alongside. This
surface re-exports the names external callers (serving, the citynav tests,
the console entry point) depend on.
"""

from benchmarks.citynav.run_benchmark.assets import (
    BlockAssets,
    load_episode_assets,
)
from benchmarks.citynav.run_benchmark.cli import main
from benchmarks.citynav.run_benchmark.config import RunConfig, load_config
from benchmarks.citynav.run_benchmark.trajectory import (
    synthesize_compliant_trajectory,
)

__all__ = [
    "BlockAssets",
    "RunConfig",
    "main",
    "load_config",
    "load_episode_assets",
    "synthesize_compliant_trajectory",
]
