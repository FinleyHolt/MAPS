"""A serve that dies under an episode does not cost SR.

That is not the reasoner getting it wrong; it is the cluster's weather, and
charging it to the model prices one as the other. The PRODUCER stamps such
an attempt ``status: infra`` (``dispatch`` on ``ended_by ==
transient_failures``, the driver on a serve death or spawn failure), and
the reader reads ONLY that stamp -- two definitions of one question had
already diverged once, with the driver re-queueing an episode the reader
went on scoring.

Measured twice. ``f40088a:calcity_full_replyrepair``: a serve OOM storm
forced 50 of 1022 episodes to SR 0.000. And the reasoning-effort ladder,
where the max-effort arm's failure was DETERMINISTIC, so the re-queue fired
on every redo and the arm never terminated.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.citynav.analysis import rows


def _row(eid: str, stamp: str, sr: int, *, ended_by: str = "reply",
         status: str = "ok") -> dict:
    slug = eid.replace("::", "__").replace("#", "_")
    return {
        "episode_id": eid, "worker_id": "w0", "status": status, "error": None,
        "elapsed_s": 1.0, "workdir": f"w0/calcity_{stamp}_{slug}", "oom_retries": 0,
        "metrics": {
            "episode_id": eid, "SR": sr, "NE": 10.0,
            "success_criterion": "shape",
            "mission": {"ended_by": ended_by, "turns": 3},
        },
    }


def _run(tmp_path: Path, *ledger_rows: dict) -> Path:
    root = tmp_path / "run"
    root.mkdir(parents=True, exist_ok=True)
    (root / rows.EPISODES_JSONL).write_text(
        "".join(json.dumps(r) + "\n" for r in ledger_rows)
    )
    return root


def test_the_reader_reads_only_the_status_the_producer_stamped() -> None:
    """The verdict is one key. A row whose mission ended in
    transient_failures but whose producer stamped ``ok`` is NOT infra
    here -- the producer-side pin (that it never does that) lives beside
    the producer."""
    assert rows.is_infra_abort(
        _row("c::1#0", "20260827_100000", 0, ended_by="transient_failures",
             status="infra"))
    assert not rows.is_infra_abort(
        _row("c::2#0", "20260827_100000", 0, ended_by="transient_failures"))
    assert not rows.is_infra_abort(_row("c::3#0", "20260827_100000", 1))
    assert rows.is_infra_abort({"episode_id": "c::4#0", "status": "infra"})


def test_a_serve_failure_leaves_the_sr_denominator(tmp_path) -> None:
    root = _run(
        tmp_path,
        _row("c::1#0", "20260827_100000", 1),
        _row("c::2#0", "20260827_100100", 0, ended_by="transient_failures",
             status="infra"),
    )
    picked, stats = rows.load_attempts(root)

    # Vacuity floor: an empty read would satisfy the exclusion trivially.
    assert len(picked) == 1, "the healthy episode must still be there"
    assert "c::1#0" in picked
    assert "c::2#0" not in picked, "a serve casualty was scored"
    assert stats.infra == 1, "it left the denominator without being counted"


def test_the_rerun_represents_the_episode_not_the_casualty(tmp_path) -> None:
    """The exact ladder case: the driver re-queued, the re-run succeeded, and
    first-complete-by-start-time was handing the episode to the failure."""
    root = _run(
        tmp_path,
        _row("c::7#0", "20260827_100000", 0, ended_by="transient_failures",
             status="infra"),
        _row("c::7#0", "20260827_101500", 1),
    )
    picked, stats = rows.load_attempts(root)
    assert len(picked) == 1
    assert picked["c::7#0"].metrics["SR"] == 1, "the casualty outranked the redo"
    assert stats.infra == 1


def test_the_algorithm_s_own_budget_still_scores(tmp_path) -> None:
    """turn_cap, compaction_infeasible and oversize_rejection are the agent
    spending what it was given. Excluding those would launder real misses."""
    root = _run(
        tmp_path,
        _row("c::1#0", "20260827_100000", 0, ended_by="turn_cap"),
        _row("c::2#0", "20260827_100100", 0, ended_by="compaction_infeasible"),
        _row("c::3#0", "20260827_100200", 0, ended_by="oversize_rejection"),
    )
    picked, stats = rows.load_attempts(root)
    assert len(picked) == 3, "an algorithm miss was excluded from SR"
    assert stats.infra == 0
