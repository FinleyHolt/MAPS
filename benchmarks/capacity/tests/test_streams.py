from __future__ import annotations

import statistics

import pytest

from benchmarks.capacity.streams import stream

IDS = [f"calcity::{i}#{i}" for i in range(5000)]


def test_a_seed_reproduces_its_stream_and_another_does_not() -> None:
    assert stream(30, 1, 7, IDS) == stream(30, 1, 7, IDS)
    assert stream(30, 1, 7, IDS)["reports"] != stream(30, 1, 8, IDS)["reports"]


def test_gaps_average_the_offered_load_and_no_report_repeats() -> None:
    reports = stream(120, 30, 1, IDS)["reports"]
    times = [r["arrival_s"] for r in reports]
    assert times == sorted(times) and times[-1] < 30 * 3600
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert statistics.fmean(gaps) == pytest.approx(30.0, rel=0.05)
    assert len({r["episode_id"] for r in reports}) == len(reports)


def test_a_stream_longer_than_the_set_is_refused() -> None:
    with pytest.raises(ValueError, match="outrun"):
        stream(600, 1, 1, IDS[:100])


def test_the_reviewed_set_feeds_the_default_stream() -> None:
    from benchmarks.calcity.data_prep.reviewed import reviewed_episode_ids

    reviewed = set(reviewed_episode_ids())
    assert {r["episode_id"] for r in stream(60, 1, 1)["reports"]} <= reviewed
