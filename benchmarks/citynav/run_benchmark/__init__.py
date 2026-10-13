"""CityNav benchmark runner.

The episode engine is a package of purpose-named modules: ``cli`` (main),
``dispatch`` -> ``episode_runner`` (per-episode flow) -> ``mission_engine``
(the one-conversation fly-on-request executor) -> ``trajectory`` (discrete
replay), with ``assets`` / ``osm_prior`` as leaves and ``config``
alongside. This surface re-exports the names the flight harness and the
tests import from the package. Importing it loads ``cli``, which configures
INFO logging at import unless the process configured its own.
"""

from benchmarks.citynav.run_benchmark.assets import load_episode_assets
from benchmarks.citynav.run_benchmark.cli import main
from benchmarks.citynav.run_benchmark.config import load_config
from benchmarks.citynav.run_benchmark.trajectory import (
    synthesize_compliant_trajectory,
)

__all__ = [
    "main",
    "load_config",
    "load_episode_assets",
    "synthesize_compliant_trajectory",
]
