"""A cross-bake or cross-split delta is refused, and no flag can waive it.

An A/B's SR is a score against a split and a tile bake; vary either and the
two numbers answer different questions. ``--allow-provenance-mismatch``
waives the arm keys (git_commit, model, ...) and never the ground truth, and
UNRECORDED ground truth refuses like mismatched ground truth.
"""

from __future__ import annotations

from benchmarks.citynav.analysis.ab_report import comparability_issues


def _prov(*, sha="abc123", tiles="tiles/citynav_2m_osmv2", split="val_seen",
          model="qwen36-35b-a3b") -> dict:
    return {
        "git_commit": sha,
        "config_sha256": "cfg0",
        "served_model": model,
        "split": split,
        "search_env_flags": {"SEARCH_CITYNAV_TILES": tiles},
    }


_CALCITY = {"split": "calcity",
            "search_env_flags": {"SEARCH_CITYNAV_TILES": "/a/benchmarks/calcity/tiles"}}
_SHA_A, _SHA_B = "5fee5f556bc1b4" + "0" * 50, "a8389803eb2322" + "0" * 50


def test_identical_runs_raise_nothing():
    fatal, arm = comparability_issues(_prov(), _prov())
    assert fatal == [] and arm == []


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
    """A code A/B that also crossed bakes: the caller waives ``arm``, and
    ``fatal`` still refuses."""
    fatal, arm = comparability_issues(
        _prov(sha="8fc0fd8", tiles="tiles/citynav_2m"),
        _prov(sha="2822471", tiles="tiles/citynav_2m_osmv2"),
    )
    assert arm, "shas differ, so the waivable class is populated"
    assert fatal and "tile bake" in fatal[0], "and the bake mismatch survives the waiver"
    assert not any("tile" in m for m in arm), "the bake is never a waivable arm key"


def test_provenance_records_no_constant():
    """Every run has the look tool, so provenance does not record it."""
    from benchmarks import repo_provenance

    assert "look_tool" not in repo_provenance()


def test_absent_arm_provenance_does_not_invent_a_mismatch():
    """An old run without a recorded model or sha still baselines a code A/B."""
    a = _prov()
    del a["git_commit"], a["served_model"], a["config_sha256"]
    fatal, arm = comparability_issues(a, _prov())
    assert arm == [] and fatal == []


def test_unrecorded_ground_truth_is_fatal():
    """Comparing nothing refuses rather than passes."""
    fatal, _ = comparability_issues({}, _prov())
    assert len(fatal) == 2 and all("unrecorded" in f for f in fatal)


def test_split_unrecorded_on_both_sides_is_fatal():
    fatal, _ = comparability_issues(_prov(split=None), _prov(split=None))
    assert fatal and "split unrecorded" in fatal[0]


def test_the_same_bake_name_with_different_content_is_fatal():
    """Every Cal City bake is called ``tiles``, so the NAME check passes two
    genuinely different bakes; the content hash separates them."""
    a = {**_CALCITY, "bake_sha256": _SHA_A}
    b = {**a, "bake_sha256": _SHA_B}
    fatal, _arm = comparability_issues(a, b)
    assert any("CONTENT differs" in f for f in fatal)
    # and the name check alone would have waved it through
    assert not any("tile bake (SEARCH_CITYNAV_TILES) differs" in f for f in fatal)


def test_the_same_bake_content_pairs():
    a = {**_CALCITY, "bake_sha256": _SHA_A}
    b = {**a, "search_env_flags": {"SEARCH_CITYNAV_TILES": "/b/benchmarks/calcity/tiles"}}
    fatal, _arm = comparability_issues(a, b)
    assert fatal == [], fatal


def test_an_unstamped_bake_does_not_become_fatal():
    """Most stored columns predate the stamp. Absence must not invalidate
    them -- and it is not a loophole: the tiles NAME check above still
    refuses an unrecorded bake, so ground truth is never compared against
    nothing. The content key only ever tightens."""
    a = _CALCITY
    b = {**a, "bake_sha256": _SHA_B}
    assert comparability_issues(a, b)[0] == []
    # but an unrecorded tiles PATH is still fatal, on either side
    c = {"split": "calcity", "search_env_flags": {}}
    assert comparability_issues(a, c)[0]


def test_the_served_alias_is_not_the_served_build():
    """One served alias can name two quantizations; the build is an arm."""
    gt = {"split": "train_seen",
          "search_env_flags": {"SEARCH_CITYNAV_TILES": "/x/tiles/citynav_2m_osmv3"}}
    a = {**gt, "served_model": "inkling-small",
         "served_build": "thinkingmachines/Inkling-Small-NVFP4"}
    b = {**gt, "served_model": "inkling-small",
         "served_build": "RedHatAI/Inkling-Small-FP8-dynamic"}
    fatal, arm = comparability_issues(a, b)
    assert not fatal
    assert any("BUILD" in m for m in arm), arm
    # Absence-tolerant, or every column stored before the key existed would
    # stop pairing against every column stored after it.
    fatal2, arm2 = comparability_issues(a, {**gt, "served_model": "inkling-small"})
    assert not fatal2 and not arm2
