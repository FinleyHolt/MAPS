"""Which mask ``SR`` is a verdict under, per benchmark.

``SUCCESS_CRITERION`` is the one fact that decides it. CityNav scores a
20 m ring around the target point; Cal City scores the target's own
annotated footprint. Both live on their package's ``paths`` and the
bridge rebinds the CityNav one, exactly like ``TILES_DIR`` — so the
engine reads the running benchmark's criterion without the CityNav
package holding a Cal City fact.
"""

from __future__ import annotations

from benchmarks.calcity import paths as calcity_paths

#: The complete set. A third value is a new scoring mode, not a typo to
#: absorb — every reader switches on exactly these two.
LEGAL = {"ring", "shape"}


def test_the_runner_reads_the_criterion_at_call_time():
    """``_success_mask`` must read ``paths.SUCCESS_CRITERION`` off the module
    at CALL time, not capture it at import.

    A ``from benchmarks.citynav.paths import SUCCESS_CRITERION`` at the top of
    ``episode_runner`` would keep the same signature, keep passing the bridge
    coverage test, and score every Cal City run on the 20 m ring — silently,
    because the ring is a real number and the row would still carry an ``SR``.
    This is the twin of ``test_bake_sha256_follows_the_rebound_tiles_dir``,
    which exists for the same failure one file over.
    """
    from types import SimpleNamespace

    from benchmarks.citynav import paths as shared_paths
    from benchmarks.citynav.run_benchmark import episode_runner

    episode = SimpleNamespace(
        target_object_ids=[], target_mask=None, map_name="calcity",
        success_radius_m=None,
    )
    cfg = SimpleNamespace(
        success_radius_m=20.0, precise_tolerance_m=0.0, precise_floor_radius_m=5.0
    )
    original = shared_paths.SUCCESS_CRITERION
    try:
        seen = []
        for value in ("ring", "shape"):
            shared_paths.SUCCESS_CRITERION = value
            seen.append(episode_runner._success_mask(cfg, episode, {})[0])
        assert seen == ["ring", "shape"], (
            "the runner captured the criterion at import -- a Cal City run "
            "would score on CityNav's mask"
        )
    finally:
        shared_paths.SUCCESS_CRITERION = original


def test_the_two_benchmarks_declare_different_criteria():
    """Vacuity floor for every criterion test below: if these ever agree,
    a test that "the criterion is honoured" proves nothing, because both
    branches would produce the same verdict."""
    from benchmarks.citynav import paths as citynav_paths

    assert citynav_paths.SUCCESS_CRITERION in LEGAL
    assert calcity_paths.SUCCESS_CRITERION in LEGAL
    assert citynav_paths.SUCCESS_CRITERION != calcity_paths.SUCCESS_CRITERION
    assert citynav_paths.SUCCESS_CRITERION == "ring"
    assert calcity_paths.SUCCESS_CRITERION == "shape"
