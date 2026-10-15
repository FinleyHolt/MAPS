"""A run root's hosted-endpoint spend, priced off every attempt's transcript."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.citynav.analysis import spend


def _attempt(root: Path, rel: str, *, prompt: int, cached: int | None, write: int, out: int,
             model: str = "gpt-5.6-sol-2026-07-09") -> None:
    llm = root / rel / "flight" / "llm"
    llm.mkdir(parents=True)
    usage = {"prompt_tokens": prompt, "completion_tokens": out, "total_tokens": prompt + out}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached, "cache_write_tokens": write}
    recs = [{"kind": "request", "request_id": "r", "created_at": 1.0},
            {"kind": "response", "request_id": "r", "ts": 2.0,
             "response": {"model": model, "usage": usage, "choices": [{"message": {}}]}}]
    (llm / "transcript.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_every_attempt_is_priced_and_the_rest_projected(tmp_path: Path) -> None:
    _attempt(tmp_path, "w0/maps_1_a", prompt=1_000_000, cached=800_000, write=100_000, out=10_000)
    _attempt(tmp_path, "w0/maps_2_b", prompt=1_000_000, cached=None, write=0, out=0)  # re-queued
    _attempt(tmp_path, "w1/maps_3_c", prompt=2_000_000, cached=0, write=0, out=0,
             model="gpt-5.6-sol-2026-09-01")  # still flying
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps({"episode_id": "e::1", "workdir": "w0/maps_1_a", "status": "ok", "metrics": {"SR": 1}}) + "\n"
        + json.dumps({"episode_id": "e::2", "workdir": "w0/maps_2_b", "status": "infra"}) + "\n")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"selection": {"episodes_n": 11}}))
    out = spend.spend(tmp_path)
    scored = (100_000 * 5 + 800_000 * 0.5 + 100_000 * 6.25 + 10_000 * 30) / 1e6
    assert out["usd"] == {"scored": round(scored, 2), "infra": 5.0, "flying": 10.0}
    assert out["usd_per_scored_episode"] == round(scored, 3)
    assert out["usd_projected_remaining"] == round(scored * 10, 2)
    assert out["cache_hit"] == pytest.approx(800_000 / 4_000_000)
    assert out["served_models"] == {"gpt-5.6-sol-2026-07-09": 2, "gpt-5.6-sol-2026-09-01": 1}


def test_a_root_with_nothing_flown_prices_nothing(tmp_path: Path) -> None:
    out = spend.spend(tmp_path)
    assert out["usd_total"] == 0 and out["usd_per_scored_episode"] is None
    assert out["cache_hit"] is None and out["usd_projected_remaining"] is None
