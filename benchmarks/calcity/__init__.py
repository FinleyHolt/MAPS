"""Cal City benchmark package: its data contract, launchers, preparation tools and tests.

It runs the CityNav engine: :func:`use` selects Cal City for this process and
the children it spawns.
"""

from __future__ import annotations

import os
import sys

from benchmarks.calcity import paths


def use() -> None:
    """Select Cal City for the shared engine: ``SEARCH_BENCHMARK_NAME`` and the
    four ``SEARCH_CITYNAV_*`` roots, which ``benchmarks.citynav.paths`` reads at
    import. Call it before that import; a second call is a no-op."""
    shared = sys.modules.get("benchmarks.citynav.paths")
    if shared is not None and shared.BENCHMARK != "calcity":
        raise RuntimeError(
            "benchmarks.citynav.paths was imported for CityNav before Cal City "
            "was selected: call benchmarks.calcity.use() first."
        )
    os.environ.update({
        "SEARCH_BENCHMARK_NAME": "calcity",
        "SEARCH_CITYNAV_DATA": str(paths.CALCITY_ROOT),
        "SEARCH_CITYNAV_SPLITS": str(paths.SPLITS_DIR),
        "SEARCH_CITYNAV_TILES": str(paths.TILES_DIR),
        "SEARCH_CITYNAV_RGBD_DIR": str(paths.CALCITY_NATIVE_RGBD_DIR),
    })
