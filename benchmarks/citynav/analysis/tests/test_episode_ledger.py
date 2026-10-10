"""The ledger stores per-episode facts and refuses incomparable pairings.

The point of storing columns is to pair a new arm against a baseline WITHOUT
re-running it. That saves a fleet, and it is also exactly how a stale baseline
gets trusted, so the refusal path is the load-bearing half of the tests here.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from benchmarks.citynav.analysis.episode_ledger import (
    DEFAULT_LEDGER,
    group,
    load,
    pair,
    render_group,
    render_table,
    verdict_cells,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "episode_ledger"

OSMV2 = "tiles/citynav_2m_osmv2"
RETIRED = "tiles/citynav_2m"


def _prov(sha: str, tiles: str = OSMV2, split: str = "val_seen") -> dict:
    return {
        "git_commit": sha,
        "config_sha256": "cfg0",
        "served_model": "qwen36-35b-a3b",
        "look_tool": True,
        "split": split,
        "seed": 1234,
        "search_env_flags": {"SEARCH_CITYNAV_TILES": tiles},
    }


def _write(path, column, prov, outcomes, precise=None, criterion=None):
    """Same shape ``append`` writes: one column header, then thin cells.

    ``criterion=None`` omits the header key, which is how a column stored
    before the stamp looks -- the reader derives it from the split.
    """
    with path.open("a") as fh:
        header = {
            "kind": "column", "column": column, "run_root": "/x", "provenance": prov,
        }
        if criterion is not None:
            header["criterion"] = criterion
        fh.write(json.dumps(header) + "\n")
        for eid, sr in outcomes.items():
            cell = {
                "kind": "cell", "column": column, "episode_id": eid, "sr": sr,
            }
            if precise is not None:
                cell["sr_precise"] = precise.get(eid)
            fh.write(json.dumps(cell) + "\n")


def test_round_trip_and_last_row_wins(tmp_path):
    led = tmp_path / "l.jsonl"
    _write(led, "aaa:runA", _prov("aaa"), {"e1": 1, "e2": 0})
    # A re-score of the same column corrects the cell without losing the file.
    _write(led, "aaa:runA", _prov("aaa"), {"e2": 1})
    cols, provs, precise, _ = load(led)
    assert cols["aaa:runA"] == {"e1": 1, "e2": 1}
    assert provs["aaa:runA"]["git_commit"] == "aaa"


def test_pair_across_commits_is_allowed_and_notes_the_arm_difference(tmp_path):
    """The refactor gate's own shape: same tiles and split, different sha."""
    led = tmp_path / "l.jsonl"
    _write(led, "old:runA", _prov("8fc0fd8"), {"e1": 1, "e2": 1, "e3": 0})
    _write(led, "new:runB", _prov("2822471"), {"e1": 1, "e2": 0, "e3": 1})
    st = pair(led, "old:runA", "new:runB")
    assert st["n"] == 3
    # e2 went 1 -> 0 (a regression) and e3 went 0 -> 1 (a gain). The
    # success->fail count is the load-bearing safety number, not net delta:
    # both here are 1, so net SR is flat while one episode actually broke.
    assert st["losses_success_to_fail"] == 1
    assert st["wins_fail_to_success"] == 1
    assert st["delta"] == 0.0
    assert any("git_commit" in d for d in st["arm_differences"])


def test_pair_refuses_a_cross_bake_comparison(tmp_path):
    """The near-miss: a baseline measured on the retired bake.

    0.351 vs 0.169 on the same episodes came from the bake, not the code.
    """
    led = tmp_path / "l.jsonl"
    _write(led, "old:runA", _prov("aaa", tiles=RETIRED), {"e1": 1})
    _write(led, "new:runB", _prov("bbb", tiles=OSMV2), {"e1": 0})
    with pytest.raises(SystemExit) as exc:
        pair(led, "old:runA", "new:runB")
    assert "tile bake" in str(exc.value)
    assert "not overridable" in str(exc.value)


def test_pair_refuses_a_cross_split_comparison(tmp_path):
    led = tmp_path / "l.jsonl"
    _write(led, "a:r", _prov("aaa", split="val_seen"), {"e1": 1})
    _write(led, "b:r", _prov("bbb", split="test_unseen"), {"e1": 1})
    with pytest.raises(SystemExit) as exc:
        pair(led, "a:r", "b:r")
    assert "split" in str(exc.value)


def test_pair_refuses_when_no_episodes_overlap(tmp_path):
    led = tmp_path / "l.jsonl"
    _write(led, "a:r", _prov("aaa"), {"e1": 1})
    _write(led, "b:r", _prov("bbb"), {"e9": 1})
    with pytest.raises(SystemExit, match="no shared episodes"):
        pair(led, "a:r", "b:r")


def test_pair_refuses_a_thin_accidental_overlap(tmp_path):
    """Two full-split columns from either side of a corpus edit intersect on a
    handful of accidental episode_id survivors (row_index shifts move the
    rest). Same split, same bake — nothing fatal upstream — so without its own
    floor, pair() would McNemar the remnant and read as a verdict on the split.
    The real case: 1029-row calcity vs its 1023-row successor shared 29 ids."""
    led = tmp_path / "l.jsonl"
    a = {f"cal::{i}#{i}": 1 for i in range(100)}
    b = {f"cal::{i}#{i + 3}": 1 for i in range(100)}  # renumbered tail...
    b.update({f"cal::{i}#{i}": 1 for i in range(20)})  # ...20 accidental survivors
    _write(led, "a:r", _prov("aaa"), a)
    _write(led, "b:r", _prov("bbb"), b)
    with pytest.raises(SystemExit, match="INVALID pairing -- only 20 shared"):
        pair(led, "a:r", "b:r")


def test_pair_allows_a_subset_run_against_a_full_column(tmp_path):
    """A deliberate subset arm (limit=N) shares ALL of its own episodes with
    the full baseline — the overlap floor is a fraction of the SMALLER column,
    so this stays a legal pairing."""
    led = tmp_path / "l.jsonl"
    _write(led, "full:r", _prov("aaa"), {f"e{i}": i % 2 for i in range(100)})
    _write(led, "sub:r", _prov("bbb"), {f"e{i}": (i + 1) % 2 for i in range(30)})
    assert pair(led, "full:r", "sub:r")["n"] == 30


def test_unknown_column_names_what_is_available(tmp_path):
    led = tmp_path / "l.jsonl"
    _write(led, "a:r", _prov("aaa"), {"e1": 1})
    with pytest.raises(SystemExit, match="a:r"):
        pair(led, "a:r", "nope")


def test_table_marks_not_run_apart_from_failed(tmp_path):
    """A blank cell is 'never ran here'; '.' is 'ran and failed'. Conflating
    them would let a partial column read as a regression."""
    led = tmp_path / "l.jsonl"
    _write(led, "a:r", _prov("aaa"), {"e1": 1, "e2": 0})
    _write(led, "b:r", _prov("bbb"), {"e1": 0})
    cols, _, _, _ = load(led)
    out = render_table(cols)
    e2 = [ln for ln in out.splitlines() if ln.startswith("e2")][0]
    assert e2.endswith(" ") or "  " in e2.split("e2")[1]
    assert "X" in out and "." in out


def test_missing_ledger_is_empty_not_an_error(tmp_path):
    cols, provs, _, _ = load(tmp_path / "absent.jsonl")
    assert cols == {} and provs == {}


def test_the_fixture_ledger_reads_to_its_expected_verdicts():
    """This fixture and its expected.json pin what the reader returns: the
    verdict a human sees for each stored column."""
    assert FIXTURES.is_dir(), f"the shared ledger pin is missing at {FIXTURES}"
    want = json.loads((FIXTURES / "expected.json").read_text())
    raw, provs, precise, crits = load(FIXTURES / "ledger.jsonl")

    assert set(raw) == set(want["column_order"])
    # Floor: an emptied fixture would satisfy a set comparison on both
    # sides of the pin without reading anything.
    assert len(raw) >= 2
    # ...and it must hold BOTH criteria, or the cell-selection half of the
    # pin is vacuous: every column reading the same cell would pass.
    assert set(crits.values()) == {"ring", "shape"}, crits
    for col_id, w in want["columns"].items():
        # VERDICTS, which is what the dashboard's reader returns: the pin is
        # that the two readers agree on the number a human sees, not on the
        # raw bytes they both parse.
        cells = {e: v for e, v in
                 verdict_cells(col_id, raw, precise, crits).items() if v is not None}
        assert cells == w["cells"], col_id
        assert sum(cells.values()) == w["successes"], col_id
        assert crits[col_id] == w["criterion"], col_id
        assert provs[col_id]["split"] == w["split"], col_id
        assert provs[col_id]["served_model"] == w["served_model"], col_id
        tiles = provs[col_id]["search_env_flags"]["SEARCH_CITYNAV_TILES"]
        assert tiles.rsplit("/", 1)[-1] == w["tile"], col_id


def test_every_verb_the_docstring_names_actually_exists():
    """A doc naming a command that was never written is how a reader learns to
    distrust the docs. The first draft of this module promised ``revalidate``.

    The CLI side is READ from the parser's own choices. Hardcoding it here
    inverted the test: adding the ``null`` verb legitimately, and documenting
    it, failed as "the docstring names verbs the CLI lacks" -- the check
    accusing the doc of the one thing that was actually right.
    """
    import re

    from benchmarks.citynav.analysis import episode_ledger as mod

    usage = mod.__doc__.split("Usage:")[1]
    named = set(re.findall(r"episode_ledger (\w+)", usage))
    # From the CLI itself, never a copy: the hardcoded list this replaces went
    # stale when ``null`` was added, and a stale expectation fails a test that
    # is documenting the truth. The floor is a COUNT rather than non-empty —
    # one surviving verb would satisfy `assert real` while the equality below
    # passed against a one-line docstring.
    real = set(mod.VERBS)
    assert len(real) >= 4, f"VERBS collapsed to {sorted(real)} -- vacuous"
    assert named <= real, f"docstring names verbs the CLI lacks: {sorted(named - real)}"
    assert named == real, f"CLI has verbs the docstring omits: {sorted(real - named)}"


# --- the precise column -----------------------------------------------------


def test_precise_column_keeps_unmeasured_apart_from_failed(tmp_path):
    """``None`` is "no shape yet", ``0`` is "measured and missed". Collapsing
    them would count an unannotated episode as a failure — and coverage is
    precisely what the shape cell is supposed to move with, so that would hide the
    only signal worth pairing on."""
    led = tmp_path / "l.jsonl"
    _write(
        led, "a:r", _prov("aaa"),
        {"e1": 1, "e2": 0, "e3": 0},
        precise={"e1": 1, "e2": 0, "e3": None},
    )
    cols, _, precise, _ = load(led)
    assert cols["a:r"] == {"e1": 1, "e2": 0, "e3": 0}
    assert precise["a:r"] == {"e1": 1, "e2": 0, "e3": None}
    scored = [v for v in precise["a:r"].values() if v is not None]
    assert len(scored) == 2, "the unmeasured episode must leave the denominator"


def test_sr_column_is_unchanged_by_a_column_with_no_precise_cells(tmp_path):
    """A column written before the precise cell existed still loads: its SR is
    intact and its precise cells read None, not 0."""
    led = tmp_path / "l.jsonl"
    _write(led, "old:r", _prov("aaa"), {"e1": 1, "e2": 0})
    cols, _, precise, _ = load(led)
    assert cols["old:r"] == {"e1": 1, "e2": 0}
    assert precise["old:r"] == {"e1": None, "e2": None}


def test_precise_cells_survive_the_last_row_wins_rescore(tmp_path):
    """Re-adding a column after annotation lands is the whole workflow: the
    corrected cell must win in the precise map too."""
    led = tmp_path / "l.jsonl"
    _write(led, "a:r", _prov("aaa"), {"e1": 0}, precise={"e1": None})
    _write(led, "a:r", _prov("aaa"), {"e1": 0}, precise={"e1": 1})
    _, _, precise, _ = load(led)
    assert precise["a:r"]["e1"] == 1
# --- group: pooling replicates, and the permutation that decides -----------


def test_group_pools_per_episode_means_not_run_means(tmp_path):
    """Each episode contributes its MEAN in each group, so replicates average.

    Three repeating patterns, ten episodes each. Pattern 0 succeeds in both A
    columns and neither B column (per-episode diff +1); pattern 1 does the
    reverse (-1); pattern 2 splits A (0.5) against a clean B (1.0), so -0.5.
    The gap is the mean of the per-episode differences, not a difference of
    run means -- which is the whole point of pooling replicates.
    """
    led = tmp_path / "l.jsonl"
    pat = {"a:r1": (1, 0, 1), "a:r2": (1, 0, 0),
           "b:r1": (0, 1, 1), "b:r2": (0, 1, 1)}
    for col, bits in pat.items():
        _write(led, col, _prov("aaa" if col.startswith("a") else "bbb"),
               {f"e{k}": bits[k % 3] for k in range(30)})
    st = group(led, ["a:r1", "a:r2"], ["b:r1", "b:r2"])
    assert st["n_episodes"] == 30
    assert st["sr_a"] == pytest.approx((1 + 0 + 0.5) / 3)
    assert st["sr_b"] == pytest.approx((0 + 1 + 1) / 3)
    assert st["gap"] == pytest.approx((1.0 - 1.0 - 0.5) / 3)


def test_group_permutation_counts_every_same_sized_split(tmp_path):
    """The number that decides: is THIS split special among its own columns?

    Four columns split 2-vs-2 is C(4,2) = 6 orderings, and a split whose gap
    is not extreme among them is nothing -- which is the whole reason the
    verb exists rather than four separate ``pair`` calls.
    """
    led = tmp_path / "l.jsonl"
    for i, col in enumerate(("a:r1", "a:r2", "b:r1", "b:r2")):
        # A pair that separates cleanly: the a-columns win every episode.
        out = {f"e{k}": int(i < 2) for k in range(30)}
        _write(led, col, _prov(f"sha{i}"), out)
    st = group(led, ["a:r1", "a:r2"], ["b:r1", "b:r2"])
    assert st["n_splits"] == 6
    assert st["gap"] == pytest.approx(1.0)
    # Only the split itself and its mirror reach |gap| = 1.
    assert st["splits_at_least_as_large"] == 2
    assert st["permutation_p"] == pytest.approx(2 / 6)
    # A perfect separation survives BOTH label nulls, selection included.
    assert st["label_p_fixed_split"] < 0.01
    assert st["label_p_split_selected"] < 0.01


def test_group_on_noise_is_not_special(tmp_path):
    """Vacuity floor for the permutation: identical columns must NOT read
    as a separation. A gap of zero is reached by every split, p = 1."""
    led = tmp_path / "l.jsonl"
    out = {f"e{k}": k % 2 for k in range(40)}
    for i, col in enumerate(("a:r1", "a:r2", "b:r1", "b:r2")):
        _write(led, col, _prov(f"sha{i}"), dict(out))
    st = group(led, ["a:r1", "a:r2"], ["b:r1", "b:r2"])
    assert st["gap"] == pytest.approx(0.0)
    assert st["permutation_p"] == pytest.approx(1.0)
    assert st["label_p_fixed_split"] == pytest.approx(1.0)
    assert st["label_p_split_selected"] == pytest.approx(1.0)


def test_group_refuses_a_cross_bake_column_anywhere_in_the_union(tmp_path):
    """The refusal must scan every PAIR in the union, not just A[0] vs B[0]:
    a retired-bake column hidden in position three is the same landmine."""
    led = tmp_path / "l.jsonl"
    _write(led, "a:r1", _prov("aaa"), {"e1": 1})
    _write(led, "a:r2", _prov("bbb"), {"e1": 1})
    _write(led, "b:r1", _prov("ccc", tiles=RETIRED), {"e1": 0})
    with pytest.raises(SystemExit, match="tile bake"):
        group(led, ["a:r1", "a:r2"], ["b:r1"])


def test_group_refuses_a_thin_overlap(tmp_path):
    led = tmp_path / "l.jsonl"
    _write(led, "a:r", _prov("aaa"), {f"e{k}": 1 for k in range(30)})
    _write(led, "b:r", _prov("bbb"), {"e1": 0, "e2": 0})
    with pytest.raises(SystemExit, match="shared"):
        group(led, ["a:r"], ["b:r"])


def test_group_names_a_missing_column(tmp_path):
    led = tmp_path / "l.jsonl"
    _write(led, "a:r", _prov("aaa"), {"e1": 1})
    with pytest.raises(SystemExit, match="nope"):
        group(led, ["a:r"], ["nope:r"])


def test_group_selection_null_is_far_weaker_than_the_fixed_one(tmp_path):
    """The correction this verb needed: a split CHOSEN by ranking the columns
    on SR tops the same-sized-split list by construction, so the descriptive
    2/56 count carries no evidence. The max-gap-over-all-splits label null is
    the one that prices that search, and it must be materially weaker.
    """
    led = tmp_path / "l.jsonl"
    rng = random.Random(4)
    # Eight exchangeable columns -- no real grouping exists.
    outs = [{f"e{k}": int(rng.random() < 0.4) for k in range(120)}
            for _ in range(8)]
    names = [f"s{i}:r{i}" for i in range(8)]
    for n, o in zip(names, outs):
        _write(led, n, _prov(n.split(":")[0]), o)
    # Pick the split the way a human would: the three highest-SR columns.
    top3 = sorted(names, key=lambda n: -sum(outs[names.index(n)].values()))[:3]
    st = group(led, top3, [n for n in names if n not in top3])
    assert st["permutation_p"] <= 2 / 56          # vacuous by construction
    assert st["label_p_split_selected"] > st["label_p_fixed_split"]
    assert st["label_p_split_selected"] > 0.2     # correctly reads as nothing


def test_group_flags_the_bootstrap_as_not_a_test_of_the_grouping(tmp_path):
    """The interval resamples EPISODES with the columns fixed, so it can
    exclude zero while the grouping is worthless. The rendering has to say
    so or the number gets quoted as significance."""
    led = tmp_path / "l.jsonl"
    rng = random.Random(9)
    outs = [{f"e{k}": int(rng.random() < 0.4) for k in range(120)}
            for _ in range(4)]
    names = [f"s{i}:r{i}" for i in range(4)]
    for n, o in zip(names, outs):
        _write(led, n, _prov(n.split(":")[0]), o)
    txt = render_group(group(led, names[:2], names[2:]))
    assert "not a test of the grouping" in txt.lower() or "NOT a test" in txt
    assert "fixed BEFORE" in txt and "chosen after" in txt


def test_pair_reads_each_column_under_its_own_criterion(tmp_path):
    """Cal City scores the TARGET'S OWN SHAPE, so a paired read on the 20 m
    disc answers a different question from the report above it. The column
    HEADER says which mask its cells are verdicts under; ``pair`` used to
    infer it from how many cells carried a shape verdict.

    The two arms below are built to DISAGREE across criteria: on the disc B
    wins 3-2, on the shape A wins 3-1. A pairing that silently used the disc
    would report the opposite sign."""
    led = tmp_path / "l.jsonl"
    _write(led, "a:A", _prov("aaa"), {"e1": 0, "e2": 0, "e3": 1, "e4": 1},
           precise={"e1": 1, "e2": 1, "e3": 1, "e4": 0}, criterion="shape")
    _write(led, "b:B", _prov("bbb"), {"e1": 1, "e2": 1, "e3": 1, "e4": 0},
           precise={"e1": 0, "e2": 0, "e3": 1, "e4": 0}, criterion="shape")
    st = pair(led, "a:A", "b:B")
    assert st["metric"] == "SR"
    assert st["criterion"] == "the target's own shape"
    assert (st["SR_a"], st["SR_b"]) == (0.75, 0.25), "the SHAPE, not the disc"


def test_pair_keeps_the_disc_on_a_ring_column(tmp_path):
    """THE VACUITY FLOOR FOR THE SWITCH. CityNav resolves no shape at all and
    the disc IS its criterion, so a change that read the shape cell
    everywhere would satisfy the test above and be wrong."""
    led = tmp_path / "l.jsonl"
    _write(led, "a:A", _prov("aaa"), {"e1": 1, "e2": 0}, criterion="ring")
    _write(led, "b:B", _prov("bbb"), {"e1": 0, "e2": 0}, criterion="ring")
    st = pair(led, "a:A", "b:B")
    assert st["metric"] == "SR"
    assert st["criterion"] == "the 20 m disc"
    assert (st["SR_a"], st["SR_b"]) == (0.5, 0.0)


def test_an_unstamped_column_takes_its_criterion_from_its_split(tmp_path):
    """Every column stored before the header carried the key. The split is
    already a FATAL comparability key, so it is recorded ground truth rather
    than the coverage guess this replaced."""
    led = tmp_path / "l.jsonl"
    _write(led, "a:A", _prov("aaa", split="calcity"),
           {"e1": 0, "e2": 0}, precise={"e1": 1, "e2": 1})
    _write(led, "b:B", _prov("bbb", split="calcity"),
           {"e1": 1, "e2": 1}, precise={"e1": 0, "e2": 0})
    st = pair(led, "a:A", "b:B")
    assert st["criterion"] == "the target's own shape"
    # The shape cells, which are the OPPOSITE of the ring cells here.
    assert (st["SR_a"], st["SR_b"]) == (1.0, 0.0)


def test_pair_refuses_a_shape_column_that_stores_no_shape_verdict(tmp_path):
    """Three stored Cal City columns predate shape scoring. Reading them
    anyway is the trap this closes: the number that came out was the 20 m
    disc while every report beside it said the shape."""
    led = tmp_path / "l.jsonl"
    _write(led, "old:O", _prov("aaa", split="calcity"), {"e1": 1, "e2": 0})
    _write(led, "new:N", _prov("bbb", split="calcity"),
           {"e1": 1, "e2": 0}, precise={"e1": 1, "e2": 0})
    with pytest.raises(SystemExit) as exc:
        pair(led, "old:O", "new:N")
    assert "no verdict under its own criterion" in str(exc.value)
    assert "old:O" in str(exc.value)


def test_pair_refuses_across_criteria(tmp_path):
    """Two columns whose cells are verdicts under different masks have no
    delta between them."""
    led = tmp_path / "l.jsonl"
    _write(led, "r:R", _prov("aaa"), {"e1": 1, "e2": 0}, criterion="ring")
    _write(led, "s:S", _prov("bbb"), {"e1": 1, "e2": 0},
           precise={"e1": 1, "e2": 0}, criterion="shape")
    with pytest.raises(SystemExit) as exc:
        pair(led, "r:R", "s:S")
    assert "INVALID pairing" in str(exc.value)


def test_pair_drops_only_the_unshaped_rows_and_says_how_many(tmp_path):
    """One unannotated episode leaves the read rather than counting as a
    miss, and the count is stated so a shrinking denominator cannot pass
    unseen."""
    led = tmp_path / "l.jsonl"
    outcomes = {f"e{i}": i % 2 for i in range(20)}
    precise_a = {f"e{i}": i % 2 for i in range(20)}
    precise_b = dict(precise_a)
    precise_b["e19"] = None
    _write(led, "a:A", _prov("aaa"), outcomes, precise=precise_a, criterion="shape")
    _write(led, "b:B", _prov("bbb"), outcomes, precise=precise_b, criterion="shape")
    st = pair(led, "a:A", "b:B")
    assert st["metric"] == "SR"
    assert (st["shared"], st["n"], st["unscored"]) == (20, 19, 1)
