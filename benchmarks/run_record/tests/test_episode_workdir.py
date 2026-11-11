"""``open_episode_dir`` is the one place an episode's directory is set up.

The pins here are the things a drift breaks silently: one episode's
recording landing in another's directory (or wherever a stale environment
value points), and the log handler outliving its episode.
"""

from __future__ import annotations

import logging
import os
import re

import pytest

from benchmarks.run_record import episode_workdir as ew


def test_each_benchmark_mints_its_own_name_and_the_stamp_reader_reads_it(tmp_path, monkeypatch):
    """``maps_<stamp>_<benchmark>_<episode>``, for every declared benchmark.

    Vacuity floor: the names are pinned here by hand, every one is
    exercised, and the minted names must DIFFER -- a ``mint_workdir``
    that ignored its ``benchmark`` and answered one word would collapse them. The
    canonical reader must recover each name's stamp, because that stamp is
    what orders an episode's attempts.
    """
    from benchmarks.run_record import rows

    assert ew.BENCHMARKS == ("citynav", "calcity")

    minted: dict[str, str] = {}
    for name in ew.BENCHMARKS:
        d = ew.mint_workdir(tmp_path, "calcity::8#0", benchmark=name)
        assert re.fullmatch(rf"maps_\d{{8}}_\d{{6}}_{name}_calcity__8_0", d.name), d.name
        stamp = rows.attempt_stamp({"workdir": f"w0/{d.name}"})
        assert re.fullmatch(r"\d{8}_\d{6}", stamp), (d.name, stamp)
        assert d.name.startswith(f"maps_{stamp}_{name}_")
        minted[name] = d.name

    assert len(set(minted.values())) == len(ew.BENCHMARKS), minted


@pytest.mark.parametrize("declared", [None, "", "CityNav"])
def test_an_undeclared_benchmark_refuses_instead_of_naming_the_wrong_one(tmp_path, declared):
    """The name is the caller's bench's: there is no default, and a name that
    is no benchmark's is refused."""
    if declared is None:
        with pytest.raises(TypeError, match="benchmark"):
            ew.mint_workdir(tmp_path, "block_7::0#3")
    else:
        with pytest.raises(RuntimeError, match="names no benchmark"):
            ew.mint_workdir(tmp_path, "block_7::0#3", benchmark=declared)


def test_each_episode_records_under_its_own_directory_whatever_a_stale_value_says(tmp_path, monkeypatch):
    """The directory is handed to the engine as a value. Two episodes open in
    one process each keep their own wire log, and a value a previous episode
    (or the shell) left in the environment is neither read nor written."""
    from maps.llm import http
    from tests.serves import ChatServe

    stale = tmp_path / "stale"
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(stale))
    monkeypatch.setenv("SEARCH_LLM_ROUTING_KEY", "previous::9#9")
    sent: list[dict] = []
    monkeypatch.setattr(http, "_request_headers", lambda api_key, routing_key=None: sent.append(
        {"routing_key": routing_key}) or {"Content-Type": "application/json"})
    first = ew.open_episode_dir(tmp_path / "run", episode_id="block_7::0#3", benchmark="citynav")
    second = ew.open_episode_dir(tmp_path / "run", episode_id="block_8::1#4", benchmark="citynav")
    try:
        assert first.dir != second.dir
        with ChatServe() as serve:
            for ep, episode_id in ((first, "block_7::0#3"), (second, "block_8::1#4"), (first, "block_7::0#3")):
                http.chat_completions_raw(f"{serve.url}/v1", None, timeout=30.0, workdir=ep.dir,
                                          routing_key=episode_id, model="m",
                                          messages=[{"role": "user", "content": episode_id}])
    finally:
        first.close()
        second.close()

    def requests(ep) -> int:
        lines = (ep.dir / "flight" / "llm" / "transcript.jsonl").read_text().splitlines()
        return sum('"kind": "request"' in line for line in lines)

    assert (requests(first), requests(second)) == (2, 1)
    assert [s["routing_key"] for s in sent] == ["block_7::0#3", "block_8::1#4", "block_7::0#3"]
    assert not stale.exists()
    assert os.environ["SEARCH_COMPILE_WORKDIR"] == str(stale) and os.environ["SEARCH_LLM_ROUTING_KEY"] == "previous::9#9"


def test_run_log_receives_the_loggers_and_close_detaches_it(tmp_path, caplog):
    caplog.set_level(logging.INFO)  # the runner's basicConfig level; pytest's root sits at WARNING
    root = logging.getLogger()
    before = list(root.handlers)
    ep = ew.open_episode_dir(tmp_path, episode_id="block_7::0#3", benchmark="citynav")
    logging.getLogger("belief.some.tree").info("hello from the tree")
    ep.close()
    assert root.handlers == before
    text = (ep.dir / "run.log").read_text()
    assert "belief.some.tree" in text and "hello from the tree" in text
    logging.getLogger("belief.some.tree").info("after close")
    assert "after close" not in (ep.dir / "run.log").read_text()


def test_a_flight_attempt_names_its_run_number_and_a_bench_attempt_names_none(
    tmp_path, monkeypatch
):
    """A flight mission's directory carries its sortie run number (two digits
    at least) through ``open_episode_dir``; a bench attempt's carries none.
    The stamp stays first, because ``run_record.rows`` orders attempts by it."""
    from benchmarks.run_record import rows

    ep = ew.open_episode_dir(tmp_path, episode_id="calcity::8#0", benchmark="calcity", run_no=3)
    ep.close()
    flight = ep.dir
    bench = ew.mint_workdir(tmp_path, "calcity::8#0", benchmark="calcity")

    assert re.fullmatch(r"maps_\d{8}_\d{6}_run03_calcity_calcity__8_0", flight.name), flight.name
    assert (flight / "flight").is_dir()
    assert re.fullmatch(r"maps_\d{8}_\d{6}_calcity_calcity__8_0", bench.name), bench.name
    assert "_run104_" in ew.mint_workdir(tmp_path, "calcity::8#0", 104, benchmark="calcity").name

    for d in (flight, bench):
        stamp = rows.attempt_stamp({"workdir": f"w0/{d.name}"})
        assert re.fullmatch(r"\d{8}_\d{6}", stamp), (d.name, stamp)
        assert d.name.startswith(f"maps_{stamp}_")
