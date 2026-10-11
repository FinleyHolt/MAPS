"""summary.txt is run_report's view of the root under the driver's infra lines."""

from __future__ import annotations

import json

from benchmarks.citynav.analysis.tests.ledger import ledger_row


def _summary(driver, *rows: dict) -> str:
    d = driver()
    if rows:
        d.episodes_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    d._write_summary()
    return d.summary_file.read_text()


def test_the_summary_scores_by_the_readers_rule_and_audits_infra(driver):
    out = _summary(driver,
        ledger_row("c::3#6534", "20260725_111500", 0),
        ledger_row("c::3#6534", "20260725_110958", 1),
        ledger_row("e::1#1", "20260725_110000", 1),
        ledger_row("g::2#2", "20260725_110100", status="infra", error="vllm_down_midrun"),
    )
    assert "infra (re-queued, not scored): 1  by cause: vllm_down_midrun:1" in out
    # The first-STARTED attempt of c scores, not the one that finished last.
    assert "SR = 2/2 (100.0%)" in out, out


def test_a_missing_ledger_still_writes_a_summary(driver):
    assert "no scored rows" in _summary(driver)
