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


def _ledger(eid: str, stamp: str, sr: int, criterion: str = "shape",
            ne: float = 10.0) -> dict:
    slug = eid.replace("::", "__").replace("#", "_")
    return {
        "episode_id": eid, "worker_id": "w0", "status": "ok", "elapsed_s": 1.0,
        "workdir": f"w0/calcity_{stamp}_{slug}", "oom_retries": 0,
        "metrics": {
            "episode_id": eid, "SR": sr, "NE": ne,
            "success_criterion": criterion,
            "precise_target": "shape",
            "precise_floor_applied": False,
        },
    }


def _arm(tmp_path: Path, name: str, *ledger_rows: dict,
         effort: float | None = None) -> Path:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "episodes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ledger_rows)
    )
    if effort is not None:
        (root / "arm.json").write_text(
            json.dumps({"label": f"e{effort}", "effort": effort})
        )
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

    assert [r["label"] for r in rows] == ["e0.0", "e0.9"]   # sorted by effort
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
    assert rows[1]["label"] == "e0.6"
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


def test_label_falls_back_to_the_root_name(tmp_path):
    root = _arm(tmp_path, "no_receipt", _ledger("c::1#0", "20260827_100000", 1))
    (row,) = arm_table.table([root])
    assert row["label"] == "no_receipt"
    assert row["effort"] is None
    assert row["n"] == 1


def test_an_ablation_arm_is_labelled_by_its_withholds(tmp_path):
    root = _arm(tmp_path, "ablate", _ledger("c::1#0", "20260827_100000", 1), effort=1.0)
    (root / "run_manifest.json").write_text(json.dumps({"withhold": ["look", "images"]}))
    (row,) = arm_table.table([root])
    assert row["label"] == "withhold=look,images"
    (root / "run_manifest.json").write_text(json.dumps({"withhold": []}))
    (row,) = arm_table.table([root])
    assert row["label"] == "e1.0"
