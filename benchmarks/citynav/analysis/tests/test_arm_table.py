"""The N-arm table reports every arm, and refuses to pool two questions.

Two failures this pins. A ladder arm whose serve died reads as zero scored
episodes; skipping its row leaves a table that looks complete and is not.
And a ring root beside a shape root shares no success key, so an SR column
spanning both is two questions under one heading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.citynav.analysis import arm_table
from benchmarks.run_record.tests.ledger import write_ledger


def _ledger(eid: str, stamp: str, sr: int, criterion: str = "shape",
            ne: float = 10.0) -> dict:
    slug = eid.replace("::", "__").replace("#", "_")
    return {
        "episode_id": eid, "worker_id": "w0", "status": "ok", "elapsed_s": 1.0,
        "workdir": f"w0/calcity_{stamp}_{slug}",
        "metrics": {
            "episode_id": eid, "SR": sr, "NE": ne,
            "success_criterion": criterion,
            "precise_target": "shape",
        },
    }


def _arm(tmp_path: Path, name: str, *ledger_rows: dict,
         effort: float | None = None) -> Path:
    root = write_ledger(tmp_path / name, *ledger_rows)
    if effort is not None:
        (root / "run_manifest.json").write_text(json.dumps(
            {"search_env_flags": {"SEARCH_SAMPLING_REASONING_EFFORT": str(effort)}}))
    return root


def test_one_row_per_arm_with_the_arm_s_own_sr_and_ne(tmp_path):
    low = _arm(
        tmp_path, "low",
        _ledger("c::1#0", "20260827_100000", 1),
        _ledger("c::2#0", "20260827_100100", 0, ne=30.0),
        effort=0.0,
    )
    high = _arm(
        tmp_path, "high",
        _ledger("c::1#0", "20260827_100000", 1),
        _ledger("c::2#0", "20260827_100100", 1, ne=2.0),
        effort=0.9,
    )
    rows = arm_table.table([high, low])

    # Vacuity floor: an empty read would satisfy every assertion below if
    # they only checked shape, so pin the counts first.
    assert len(rows) == 2
    assert all(r["n"] == 2 for r in rows)
    assert all("tier" not in r for r in rows)

    assert [r["label"] for r in rows] == ["low", "high"]    # sorted by effort
    assert [r["effort"] for r in rows] == ["0.0", "0.9"]
    assert [r["SR_n"] for r in rows] == [1, 2]
    assert rows[0]["SR_pct"] == 50.0
    assert rows[1]["SR_pct"] == 100.0
    assert rows[0]["NE_mean"] == 20.0
    assert rows[0]["NE_med"] == 20.0
    assert rows[1]["NE_mean"] == 6.0


def test_a_dead_arm_is_reported_not_dropped(tmp_path):
    """Zero scored episodes is the finding, so the row still prints."""
    live = _arm(tmp_path, "live", _ledger("c::1#0", "20260827_100000", 1),
                effort=0.5)
    dead = _arm(tmp_path, "dead", effort=0.6)

    rows = arm_table.table([live, dead])
    assert len(rows) == 2
    assert rows[1]["label"] == "dead"
    assert rows[1]["n"] == 0
    assert rows[1]["SR_pct"] is None      # never a zero standing in for absent
    arm_table.render(rows)                 # must not raise on the empty row


def test_mixing_criteria_refuses_rather_than_pooling(tmp_path):
    shape = _arm(tmp_path, "shape",
                 _ledger("c::1#0", "20260827_100000", 1, criterion="shape"))
    ring = _arm(tmp_path, "ring",
                _ledger("n::1#0", "20260827_100000", 1, criterion="ring"))
    with pytest.raises(SystemExit, match="mix success criteria"):
        arm_table.table([shape, ring])


def test_an_unstamped_root_beside_a_stamped_one_refuses(tmp_path):
    """Scored rows without a criterion stamp are UNRECORDED, a criterion of their own."""
    shape = _arm(tmp_path, "shape", _ledger("c::1#0", "20260827_100000", 1))
    bare = _ledger("c::1#0", "20260827_100000", 1)
    del bare["metrics"]["success_criterion"]
    with pytest.raises(SystemExit, match="mix success criteria"):
        arm_table.table([shape, _arm(tmp_path, "bare", bare)])


def test_a_crash_only_root_beside_a_stamped_one_tables(tmp_path):
    """A root whose every pick is a crash miss records no criterion, so it mixes with none."""
    shape = _arm(tmp_path, "shape", _ledger("c::1#0", "20260827_100000", 1))
    crash = _ledger("c::1#0", "20260827_100000", 1)
    del crash["metrics"]
    crash["error"] = "Boom"
    rows = arm_table.table([shape, _arm(tmp_path, "crash", crash)])
    assert [(r["label"], r["n"], r["criterion"], r["SR_pct"]) for r in rows] == [
        ("crash", 1, None, 0.0), ("shape", 1, "the target's own shape", 100.0)]


def test_ring_arms_of_two_radii_refuse_rather_than_pooling(tmp_path):
    """Two ring arms scored on a 20 m and a 40 m disc answer different questions."""
    roots = []
    for radius in (20, 40):
        root = _arm(tmp_path, f"r{radius}", _ledger("n::1#0", "20260827_100000", 1, criterion="ring"))
        (root / "run_manifest.json").write_text(json.dumps({"config_yaml": f"success_radius_m: {radius}\n"}))
        roots.append(root)
    with pytest.raises(SystemExit, match=r"mix success criteria \['the 20 m disc', 'the 40 m disc'\]"):
        arm_table.table(roots)
    (row,) = arm_table.table(roots[:1])
    assert row["criterion"] == "the 20 m disc"


def test_osr_rides_a_ring_row_and_is_absent_from_a_shape_row(tmp_path):
    ring_row = _ledger("n::1#0", "20260827_100000", 1, criterion="ring")
    ring_row["metrics"]["OSR"] = 1
    (ring,) = arm_table.table([_arm(tmp_path, "ring", ring_row)])
    assert ring["OSR_pct"] == 100.0
    (shape,) = arm_table.table([_arm(tmp_path, "shape", _ledger("c::1#0", "20260827_100000", 1))])
    assert shape["OSR_pct"] is None


def test_label_falls_back_to_the_root_name(tmp_path):
    root = _arm(tmp_path, "no_receipt", _ledger("c::1#0", "20260827_100000", 1))
    (row,) = arm_table.table([root])
    assert row["label"] == "no_receipt"
    assert row["effort"] is None
    assert row["n"] == 1


def test_a_long_label_keeps_the_columns_aligned(capsys):
    label = "calcity_full_method_effort_09"
    arm_table.render([{"label": label, "criterion": "the target's own shape", "n": 1},
                      {"label": "e1", "criterion": "the target's own shape", "n": 1}])
    lines = capsys.readouterr().out.splitlines()[-3:]
    assert all(ln[len(label):len(label) + 2] == "  " for ln in lines), lines
    assert lines[1].startswith(label + "  ")
