"""The same-class detection diagnostic has to arm for EVERY referent bucket.

``episode_runner`` hardcoded ``n_target_class_dets = 0`` /
``nearest_target_class_det_m = None`` and then computed them for car, bike
and building only. For a Ground or Parking target the hardcoded pair
survived, so all 25 such episodes in a 300-run (8.3%) unconditionally failed
any ``n_target_class_dets == 0`` test whatever the scene held. Round 8's own
first-draft failure attribution scored 8 of 24 "coverage" misses off that
artifact.

This is the repo tenet "a key that is present but always empty reads exactly
like a live one", live: the key was present on every row, schema-valid, and
dead for two buckets.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.citynav.adapter import OBJTYPE_TO_LABEL
from benchmarks.citynav.run_benchmark import episode_runner

#: The buckets the hardcoded subset stranded. Named, because a derived scan
#: over the map alone could not notice them going missing from it.
_STRANDED = ("Ground", "Parking")


def test_every_referent_bucket_can_arm_the_class_diagnostic():
    # The runner's arming rule IS this map, imported rather than re-spelled
    # — a second spelling is how the three-class subset survived.
    assert episode_runner.OBJTYPE_TO_LABEL is OBJTYPE_TO_LABEL
    # vacuity floor: CityRefer ships 13 object types, and a gutted map would
    # pass the "every bucket arms" scan below by having no buckets.
    assert len(OBJTYPE_TO_LABEL) == 13, sorted(OBJTYPE_TO_LABEL)

    unarmable = [t for t in OBJTYPE_TO_LABEL
                 if not OBJTYPE_TO_LABEL.get(str(t).strip())]
    assert not unarmable, unarmable
    for t in _STRANDED:
        assert OBJTYPE_TO_LABEL.get(t), t


def test_the_counters_are_never_hardcoded_before_the_diagnostic_runs():
    """A count the diagnostic never computed must be ABSENT, not zero.

    Absent reads as ``unrecorded`` in ``discernment.band`` and keeps the
    ``class`` family out of ``episode_signal``'s ``recorded`` set; a
    hardcoded 0 reads as a measured empty scene in both.
    """
    src = Path(episode_runner.__file__).read_text()
    for dead in ('scores["n_target_class_dets"] = 0',
                 'scores["nearest_target_class_det_m"] = None'):
        assert dead not in src, dead
    # and the writes that remain sit inside the armed branch, which states
    # the label it counted
    assert 'scores["target_class_for_diag"] = _target_class_for_diag' in src
    assert src.count('scores["n_target_class_dets"]') == 1
