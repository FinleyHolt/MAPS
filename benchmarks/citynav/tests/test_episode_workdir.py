"""``open_episode_dir`` is the one place an episode's directory is set up.

The pins here are the things a drift breaks silently: a stale env value
from the previous episode winning over the new one, and the log handler
outliving its episode.
"""

from __future__ import annotations

import logging
import os
import re

import pytest

from benchmarks.citynav import episode_workdir as ew


@pytest.fixture(autouse=True)
def declared_benchmark(monkeypatch):
    """A process that opens an episode directory declares what it is flying.
    ``benchmark()`` refuses an undeclared one, which has its own test."""
    monkeypatch.setenv("SEARCH_BENCHMARK_NAME", "citynav")


def test_each_benchmark_mints_its_own_name_and_the_stamp_reader_reads_it(tmp_path, monkeypatch):
    """``maps_<stamp>_<benchmark>_<episode>``, for every declared benchmark.

    Vacuity floor: the names are pinned here by hand, every one is
    exercised, and the minted names must DIFFER -- a ``benchmark()``
    that ignored its input and answered one word would collapse them. The
    canonical reader must recover each name's stamp, because that stamp is
    what orders an episode's attempts.
    """
    from benchmarks.citynav.analysis import rows

    assert ew.BENCHMARKS == ("citynav", "calcity")

    minted: dict[str, str] = {}
    for name in ew.BENCHMARKS:
        monkeypatch.setenv("SEARCH_BENCHMARK_NAME", name)
        d = ew.mint_workdir(tmp_path, "calcity::8#0")
        assert re.fullmatch(rf"maps_\d{{8}}_\d{{6}}_{name}_calcity__8_0", d.name), d.name
        stamp = rows.attempt_stamp({"workdir": f"w0/{d.name}"})
        assert re.fullmatch(r"\d{8}_\d{6}", stamp), (d.name, stamp)
        assert d.name.startswith(f"maps_{stamp}_{name}_")
        minted[name] = d.name

    assert len(set(minted.values())) == len(ew.BENCHMARKS), minted


@pytest.mark.parametrize("declared", [None, "", "CityNav"])
def test_an_undeclared_benchmark_refuses_instead_of_naming_the_wrong_one(
    tmp_path, monkeypatch, declared
):
    if declared is None:
        monkeypatch.delenv("SEARCH_BENCHMARK_NAME")
    else:
        monkeypatch.setenv("SEARCH_BENCHMARK_NAME", declared)
    with pytest.raises(RuntimeError, match="SEARCH_BENCHMARK_NAME"):
        ew.mint_workdir(tmp_path, "block_7::0#3")


def test_the_env_is_assigned_over_a_stale_value(tmp_path, monkeypatch):
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", "/stale/previous/episode")
    monkeypatch.setenv("SEARCH_LLM_ROUTING_KEY", "previous::9#9")
    ep = ew.open_episode_dir(tmp_path, episode_id="block_7::0#3")
    try:
        assert os.environ["SEARCH_COMPILE_WORKDIR"] == str(ep.dir)
        assert os.environ["SEARCH_LLM_ROUTING_KEY"] == "block_7::0#3"
    finally:
        ep.close()


def test_run_log_receives_the_loggers_and_close_detaches_it(tmp_path, caplog):
    caplog.set_level(logging.INFO)  # the runner's basicConfig level; pytest's root sits at WARNING
    root = logging.getLogger()
    before = list(root.handlers)
    ep = ew.open_episode_dir(tmp_path, episode_id="block_7::0#3")
    logging.getLogger("belief.some.tree").info("hello from the tree")
    ep.close()
    assert root.handlers == before
    text = (ep.dir / "run.log").read_text()
    assert "belief.some.tree" in text and "hello from the tree" in text
    logging.getLogger("belief.some.tree").info("after close")
    assert "after close" not in (ep.dir / "run.log").read_text()


def test_a_station_attempt_names_its_run_number_and_a_bench_attempt_names_none(
    tmp_path, monkeypatch
):
    """One identifier walks from a run row to the folder on disk.

    A sortie numbers its runs from 1, both cabin screens say that number,
    and the ledger row carries it -- so it is the number, not a timestamp,
    that a person browsing the tree greps for. A BENCH attempt carries none:
    there the run is the root the attempt already sits in.

    The stamp stays FIRST whichever it is, because ``analysis.rows`` orders
    an episode's attempts by it and so does ``ls``.
    """
    from benchmarks.citynav.analysis import rows

    monkeypatch.setenv("SEARCH_BENCHMARK_NAME", "calcity")
    flight = ew.mint_workdir(tmp_path, "calcity::8#0", 3)
    bench = ew.mint_workdir(tmp_path, "calcity::8#0")

    assert re.fullmatch(r"maps_\d{8}_\d{6}_run03_calcity_calcity__8_0", flight.name), flight.name
    assert re.fullmatch(r"maps_\d{8}_\d{6}_calcity_calcity__8_0", bench.name), bench.name
    assert "run03" in flight.name and "run" not in bench.name.replace("maps_", "")
    # A two-digit floor keeps `ls` in flying order through run 99; past it
    # the number simply grows.
    assert "_run12_" in ew.mint_workdir(tmp_path, "calcity::8#0", 12).name
    assert "_run104_" in ew.mint_workdir(tmp_path, "calcity::8#0", 104).name

    for d in (flight, bench):
        stamp = rows.attempt_stamp({"workdir": f"w0/{d.name}"})
        assert re.fullmatch(r"\d{8}_\d{6}", stamp), (d.name, stamp)
        assert d.name.startswith(f"maps_{stamp}_")


def test_the_run_number_reaches_the_directory_through_open_episode_dir(tmp_path):
    """``open_episode_dir`` is the one place a directory is set up, so the
    number has to survive the call rather than only the mint."""
    ep = ew.open_episode_dir(tmp_path, episode_id="block_7::0#3", run_no=7)
    try:
        assert "_run07_" in ep.dir.name, ep.dir.name
        assert ep.dir.parent == tmp_path
        assert (ep.dir / "flight").is_dir()
    finally:
        ep.close()
