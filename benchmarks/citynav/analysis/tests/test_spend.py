"""A run root's hosted-endpoint spend, priced off every attempt's transcript."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.citynav.analysis import spend


def _attempt(
    root: Path,
    rel: str,
    *,
    prompt: int,
    cached: int | None,
    write: int,
    out: int,
    model: str = "gpt-5.6-sol-2026-07-09",
) -> None:
    llm = root / rel / "flight" / "llm"
    llm.mkdir(parents=True)
    usage = {"prompt_tokens": prompt, "completion_tokens": out, "total_tokens": prompt + out}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached, "cache_write_tokens": write}
    recs = [
        {"kind": "request", "request_id": "r", "created_at": 1.0},
        {
            "kind": "response",
            "request_id": "r",
            "ts": 2.0,
            "response": {"model": model, "usage": usage, "choices": [{"message": {}}]},
        },
    ]
    (llm / "transcript.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_every_attempt_is_priced_and_the_rest_projected(tmp_path: Path) -> None:
    _attempt(tmp_path, "w0/maps_1_a", prompt=1_000_000, cached=800_000, write=100_000, out=10_000)
    _attempt(tmp_path, "w0/maps_2_b", prompt=1_000_000, cached=None, write=0, out=0)  # re-queued
    _attempt(
        tmp_path,
        "w1/maps_3_c",
        prompt=2_000_000,
        cached=0,
        write=0,
        out=0,
        model="gpt-5.6-sol-2026-09-01",
    )  # still flying
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps(
            {"episode_id": "e::1", "workdir": "w0/maps_1_a", "status": "ok", "metrics": {"SR": 1}}
        )
        + "\n"
        + json.dumps({"episode_id": "e::2", "workdir": "w0/maps_2_b", "status": "infra"})
        + "\n"
    )
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


def test_a_scored_attempt_that_sent_no_request_is_a_scored_episode(tmp_path: Path) -> None:
    """An episode scored without a request cost nothing, and still counts in the mean and the projection."""
    _attempt(tmp_path, "w0/maps_1_a", prompt=1_000_000, cached=0, write=0, out=0)
    (tmp_path / "w0/maps_2_b/flight/llm").mkdir(parents=True)
    (tmp_path / "w0/maps_2_b/flight/llm/transcript.jsonl").write_text("")
    (tmp_path / "episodes.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "episode_id": f"e::{i}",
                    "workdir": f"w0/maps_{i}_{c}",
                    "status": "ok",
                    "metrics": {"SR": 0},
                }
            )
            + "\n"
            for i, c in ((1, "a"), (2, "b"))
        )
    )
    (tmp_path / "run_manifest.json").write_text(json.dumps({"selection": {"episodes_n": 2}}))
    out = spend.spend(tmp_path)
    assert (
        out["scored_episodes"],
        out["usd_per_scored_episode"],
        out["usd_projected_remaining"],
    ) == (2, 2.5, 0)


def test_cache_meters_no_attempt_reported_read_none(tmp_path: Path) -> None:
    """A serve that reports no cache meter has an unknown hit rate, not a zero one."""
    _attempt(tmp_path, "w0/maps_1_a", prompt=1_000_000, cached=None, write=0, out=0)
    out = spend.spend(tmp_path)
    assert (
        out["tokens"]["cached_tokens"],
        out["tokens"]["cache_write_tokens"],
        out["cache_hit"],
    ) == (None, None, None)
    assert out["usd"]["flying"] == 5.0


def test_a_ledger_whose_workdirs_are_absent_is_refused(tmp_path: Path) -> None:
    # A copied ledger without its attempts would read as nothing spent.
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps(
            {"episode_id": "e::1", "workdir": "w0/maps_1_a", "status": "ok", "metrics": {"SR": 1}}
        )
        + "\n"
    )
    with pytest.raises(SystemExit, match="none of their workdirs is here"):
        spend.spend(tmp_path)


def test_attempts_that_made_no_workdir_cost_nothing(tmp_path: Path) -> None:
    # An infra row for an attempt that never started names no workdir.
    (tmp_path / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_id": f"e::{i}", "workdir": None, "status": "infra"}) + "\n"
            for i in range(3)
        )
    )
    assert spend.spend(tmp_path)["usd_total"] == 0


def test_attempts_are_read_where_the_ledger_places_them(tmp_path: Path) -> None:
    """A root gathered from several runs nests its workdirs deeper than ``w*/``;
    a single runner's root holds its attempt at the top."""
    _attempt(tmp_path, "a/w24/maps_1_a", prompt=1_000_000, cached=0, write=0, out=0)
    _attempt(tmp_path, "maps_2_b", prompt=1_000_000, cached=0, write=0, out=0)
    _attempt(tmp_path, "a/w3/maps_3_c", prompt=1_000_000, cached=0, write=0, out=0)  # flying
    (tmp_path / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_id": eid, "workdir": wd, "status": "ok", "metrics": {"SR": 1}})
            + "\n"
            for eid, wd in (("e::1", "a/w24/maps_1_a"), ("e::2", "maps_2_b"))
        )
    )
    out = spend.spend(tmp_path)
    assert out["usd"] == {"scored": 10.0, "infra": 0.0, "flying": 5.0}
    assert out["scored_episodes"] == 2
