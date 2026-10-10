"""A cross-bake or cross-split delta is refused, and no flag can waive it.

An A/B's SR is a score AGAINST a ground truth: a split and a tile bake. Vary
either and the two numbers answer different questions, so the delta is not
small or noisy, it is meaningless: two bakes of one name scored the same
episodes 0.351 and 0.169.

The waiver is the subtle half. ``--allow-provenance-mismatch`` exists because a
refactor or model A/B is SUPPOSED to differ on git_commit, and taking that
override must leave the tile check on. These tests pin the split: arm keys
waivable, ground truth never --
and UNRECORDED ground truth refuses like mismatched ground truth, because a
null that skips the comparison is the guard silently off.
"""

from __future__ import annotations

from benchmarks.citynav.analysis.ab_report import comparability_issues


def _prov(*, sha="abc123", tiles="tiles/citynav_2m_osmv2", split="val_seen",
          model="qwen36-35b-a3b", look=True) -> dict:
    return {
        "git_commit": sha,
        "config_sha256": "cfg0",
        "served_model": model,
        "look_tool": look,
        "split": split,
        "search_env_flags": {"SEARCH_CITYNAV_TILES": tiles},
    }


def test_identical_runs_raise_nothing():
    fatal, arm = comparability_issues(_prov(), _prov())
    assert fatal == [] and arm == []


def test_tile_bake_mismatch_is_fatal():
    fatal, arm = comparability_issues(
        _prov(tiles="tiles/citynav_2m"), _prov(tiles="tiles/citynav_2m_osmv2")
    )
    assert fatal, "a retired-bake arm must be refused"
    assert "tile bake" in fatal[0]
    # It is fatal, NOT an arm difference -- otherwise the waiver would clear it.
    assert arm == []


def test_split_mismatch_is_fatal():
    fatal, _ = comparability_issues(_prov(split="val_seen"), _prov(split="test_unseen"))
    assert fatal and "split" in fatal[0]


def test_same_bake_under_different_site_roots_is_comparable():
    """The bake's identity is its NAME: a cross-site pair (or one pool run's
    own lanes) carries the same bake under each site's mount prefix, and the
    retired-bake refusal must not fire on the prefix."""
    fatal, arm = comparability_issues(
        _prov(tiles="/local/a/MAPS/tiles/citynav_2m_osmv2"),
        _prov(tiles="/scratch/b/MAPS/tiles/citynav_2m_osmv2"),
    )
    assert fatal == [] and arm == []


def test_code_difference_is_an_arm_key_not_fatal():
    """The refactor A/B this repo actually runs: same tiles+split, different sha."""
    fatal, arm = comparability_issues(_prov(sha="8fc0fd8"), _prov(sha="2822471"))
    assert fatal == [], "a code A/B on one bake is legitimate"
    assert arm and "git_commit" in arm[0]


def test_waiving_the_arm_keys_cannot_waive_the_tile_bake():
    """The incident shape: a real code A/B that ALSO crossed bakes.

    The caller waives `arm` (correctly, the shas differ on purpose). `fatal`
    must still be non-empty so the CLI refuses regardless.
    """
    fatal, arm = comparability_issues(
        _prov(sha="8fc0fd8", tiles="tiles/citynav_2m"),
        _prov(sha="2822471", tiles="tiles/citynav_2m_osmv2"),
    )
    assert arm, "shas differ, so the waivable class is populated"
    assert fatal, "and the bake mismatch survives the waiver"


def test_look_tool_difference_is_reported():
    """Look off is a text-only reasoner."""
    _, arm = comparability_issues(_prov(look=True), _prov(look=False))
    assert arm and "look tool" in arm[0]


def test_absent_arm_provenance_does_not_invent_a_mismatch():
    """An old run without a recorded model or sha still baselines a code A/B."""
    a = _prov()
    del a["git_commit"], a["served_model"], a["config_sha256"], a["look_tool"]
    fatal, arm = comparability_issues(a, _prov())
    assert arm == [] and fatal == []


def test_unrecorded_ground_truth_is_fatal():
    """A guard that reads None on both sides and passes is not a guard.

    Every pre-stamping run stored null, so the split check compared nothing
    on every pairing ever made. Comparing nothing must refuse, not pass.
    """
    fatal, _ = comparability_issues({}, _prov())
    assert len(fatal) == 2 and all("unrecorded" in f for f in fatal)


def test_split_unrecorded_on_one_side_is_fatal():
    fatal, _ = comparability_issues(_prov(split=None), _prov())
    assert fatal and "split unrecorded" in fatal[0]


def test_split_unrecorded_on_both_sides_is_fatal():
    fatal, _ = comparability_issues(_prov(split=None), _prov(split=None))
    assert fatal and "split unrecorded" in fatal[0]


def test_the_same_bake_name_with_different_content_is_fatal():
    """Every Cal City bake is called ``tiles``, so the NAME check passes two
    genuinely different bakes. Two are live across the cluster today; the
    content hash is what separates them."""
    from benchmarks.citynav.analysis.ab_report import comparability_issues

    a = {
        "split": "calcity",
        "search_env_flags": {"SEARCH_CITYNAV_TILES": "/a/benchmarks/calcity/tiles"},
        "bake_sha256": "5fee5f556bc1b4" + "0" * 50,
    }
    b = {**a, "bake_sha256": "a8389803eb2322" + "0" * 50}
    fatal, _arm = comparability_issues(a, b)
    assert any("CONTENT differs" in f for f in fatal)
    # and the name check alone would have waved it through
    assert not any("tile bake (SEARCH_CITYNAV_TILES) differs" in f for f in fatal)


def test_the_same_bake_content_pairs():
    from benchmarks.citynav.analysis.ab_report import comparability_issues

    a = {
        "split": "calcity",
        "search_env_flags": {"SEARCH_CITYNAV_TILES": "/a/benchmarks/calcity/tiles"},
        "bake_sha256": "5fee5f556bc1b4" + "0" * 50,
    }
    b = {**a, "search_env_flags": {"SEARCH_CITYNAV_TILES": "/b/benchmarks/calcity/tiles"}}
    fatal, _arm = comparability_issues(a, b)
    assert fatal == [], fatal


def test_an_unstamped_bake_does_not_become_fatal():
    """Most stored columns predate the stamp. Absence must not invalidate
    them -- and it is not a loophole: the tiles NAME check above still
    refuses an unrecorded bake, so ground truth is never compared against
    nothing. The content key only ever tightens."""
    from benchmarks.citynav.analysis.ab_report import comparability_issues

    a = {
        "split": "calcity",
        "search_env_flags": {"SEARCH_CITYNAV_TILES": "/a/benchmarks/calcity/tiles"},
    }
    b = {**a, "bake_sha256": "a8389803eb2322" + "0" * 50}
    assert comparability_issues(a, b)[0] == []
    # but an unrecorded tiles PATH is still fatal, on either side
    c = {"split": "calcity", "search_env_flags": {}}
    assert comparability_issues(a, c)[0]
