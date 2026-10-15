"""The row-sized LLM activity block, against a hand-built flight dir.

``summarize`` is the ONLY producer of ``metrics.llm``, which readers use
instead of walking transcripts. These pin turn and token counts, the
declared and called tools, and the thing a naive scan gets wrong — a
request that never got a response.
"""

from __future__ import annotations

import json
from pathlib import Path

from maps.llm import flight_summary

# The tool schema a run sends: two tools, and ``analyze`` discriminating its
# program ops on a const ``op``. ``find`` declares a ``color`` arg the model
# may or may not ever supply — the question the arg table exists to answer.
PARAMS = {
    "model": "qwen",
    "tools": [
        {"type": "function", "function": {"name": "act"}},
        {
            "type": "function",
            "function": {
                "name": "analyze",
                "parameters": {
                    "properties": {
                        "program": {
                            "properties": {
                                "ops": {
                                    "items": {
                                        "anyOf": [
                                            {
                                                "properties": {
                                                    "op": {"const": "find"},
                                                    "args": {
                                                        "properties": {
                                                            "color": {},
                                                            "query": {},
                                                        }
                                                    },
                                                }
                                            },
                                            {
                                                "properties": {
                                                    "op": {"const": "count"},
                                                    "args": {
                                                        "properties": {"of": {}}
                                                    },
                                                }
                                            },
                                        ]
                                    }
                                }
                            }
                        }
                    }
                },
            },
        },
    ],
}


def _response(rid: str, ts: float, *, usage: dict, tool_calls: list) -> dict:
    return {
        "kind": "response",
        "request_id": rid,
        "ts": ts,
        "response": {
            "usage": usage,
            "choices": [{"message": {"tool_calls": tool_calls}}],
        },
    }


def _analyze_call(ops: list) -> dict:
    return {
        "function": {
            "name": "analyze",
            "arguments": json.dumps({"program": {"ops": ops}}),
        }
    }


def _write(flight: Path, records: list[dict]) -> None:
    llm = flight / "llm"
    llm.mkdir(parents=True, exist_ok=True)
    (llm / "transcript.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )


def _flight(tmp_path: Path) -> Path:
    """Two answered turns and one that never came back."""
    flight = tmp_path / "flight"
    _write(
        flight,
        [
            {"kind": "params", "sha": "aa", "params": PARAMS},
            {"kind": "request", "request_id": "r1", "created_at": 100.0},
            _response(
                "r1",
                102.5,
                usage={
                    "prompt_tokens": 900,
                    "completion_tokens": 100,
                    "total_tokens": 1000,
                    "prompt_tokens_details": {"cached_tokens": 640},
                },
                tool_calls=[
                    _analyze_call(
                        [
                            {"op": "find", "args": {"query": "car", "color": None}},
                            {"op": "count", "args": {"of": "car"}},
                        ]
                    )
                ],
            ),
            {"kind": "request", "request_id": "r2", "created_at": 200.0},
            _response(
                "r2",
                210.0,
                usage={
                    "prompt_tokens": 1100,
                    "completion_tokens": 50,
                    "total_tokens": 1150,
                    "prompt_tokens_details": {"cached_tokens": 800},
                },
                tool_calls=[{"function": {"name": "act"}}],
            ),
            # Fired, never answered: the episode died mid-call.
            {"kind": "request", "request_id": "r3", "created_at": 300.0},
        ],
    )
    return flight


def test_turns_tokens_and_latency(tmp_path: Path) -> None:
    s = flight_summary.summarize(_flight(tmp_path))

    # Vacuity floor: a scan that found nothing must not pass this file.
    assert s["turns"] > 0

    assert s["turns"] == 2
    assert s["requests"] == 3
    assert s["unanswered"] == 1

    assert s["prompt_tokens"] == 2000
    assert s["completion_tokens"] == 150
    assert s["total_tokens"] == 2150

    assert s["latency_s_sum"] == 12.5
    # Every key has a reader (measures, the figures, the endpoint check).
    assert set(s) == set(flight_summary.blank()) == {
        "turns", "requests", "unanswered", "prompt_tokens", "completion_tokens",
        "total_tokens", "reasoning_tokens", "latency_s_sum",
        "cached_tokens", "cache_write_tokens", "served_model"}


def test_the_cache_meters_sum_what_the_responses_report(tmp_path: Path) -> None:
    """A hosted provider's cache reads and writes ride the row summed; the
    model it names on its responses rides as it last named it."""
    flight = _flight(tmp_path)
    with (flight / "llm" / "transcript.jsonl").open("a") as fh:
        fh.write(json.dumps({"kind": "response", "response": {
            "model": "gpt-5.6-sol-2026-07-09", "choices": [{"message": {}}],
            "usage": {"prompt_tokens": 2000, "prompt_tokens_details": {
                "cached_tokens": 1408, "cache_write_tokens": 512}}}}) + "\n")
    s = flight_summary.summarize(flight)
    assert s["cached_tokens"] == 640 + 800 + 1408
    assert s["cache_write_tokens"] == 512
    assert s["served_model"] == "gpt-5.6-sol-2026-07-09"


def test_a_serve_that_reports_no_cache_reads_none(tmp_path: Path) -> None:
    """Cache hits are a serve fact only some serves report: none reported is
    None, never a zero hit rate."""
    flight = tmp_path / "flight"
    _write(flight, [_response("r1", 1.0, usage={"prompt_tokens": 5}, tool_calls=[])])
    s = flight_summary.summarize(flight)
    assert s["cached_tokens"] is None and s["cache_write_tokens"] is None
    assert s["served_model"] is None
    assert flight_summary.blank()["cached_tokens"] is None


def test_reasoning_tokens_are_summed_over_the_responses_that_carry_a_span(tmp_path: Path) -> None:
    """Each response's count is ``reasoning_tokens_of``: the serve's own where a
    thinking span exists, zero where none does."""
    flight = _flight(tmp_path)
    assert flight_summary.summarize(flight)["reasoning_tokens"] == 0

    spans = [({"reasoning_content": "thought"}, 207), ({"reasoning": "more"}, 93), ({}, 50)]
    with (flight / "llm" / "transcript.jsonl").open("a") as fh:
        for message, n in spans:
            fh.write(json.dumps({"kind": "response", "response": {
                "usage": {"reasoning_tokens": n}, "choices": [{"message": message}]}}) + "\n")
    assert flight_summary.summarize(flight)["reasoning_tokens"] == 300


def test_wrong_typed_fields_are_skipped_not_fatal(tmp_path: Path) -> None:
    """Every ``or {}`` site in the scan, handed a string instead of an object.

    ``x.get(k) or {}`` guards absence, never type. Each of these would have
    raised on the following ``.get``."""
    flight = tmp_path / "flight"
    _write(
        flight,
        [
            # usage, message, and the tool-call function each arrive as strings.
            {"kind": "response", "response": {"usage": "n/a",
                                              "choices": [{"message": "?"}]}},
            {"kind": "response", "response": {
                "choices": [{"message": {"tool_calls": ["?"]}}]}},
            # a params record whose tool entries and analyze schema are strings
            {"kind": "params", "params": {"tools": ["find", {"function": "analyze"}]}},
            {"kind": "params", "params": {
                "tools": [{"function": {"name": "analyze", "parameters": "?"}}]}},
            # and one good round, so the scan is proven still to be counting
            _response("r9", 1.0, usage={"total_tokens": 5}, tool_calls=[
                _analyze_call([{"op": "find", "args": {"color": "red"}}])
            ]),
        ],
    )
    s = flight_summary.summarize(flight)
    assert s["turns"] == 3
    assert s["total_tokens"] == 5


def test_non_dict_record_line_is_skipped(tmp_path: Path) -> None:
    """A line that is valid JSON but not an object reaches ``rec.get``.

    A JSON *array* clears both gates that would otherwise have caught it: the
    substring sniff sees the kind key inside it, and ``json.loads`` succeeds.
    Only the type check stops it."""
    flight = tmp_path / "flight"
    llm = flight / "llm"
    llm.mkdir(parents=True)
    (llm / "transcript.jsonl").write_text(
        '[{"kind": "response"}]\n'
        + json.dumps(_response("r1", 1.0, usage={"total_tokens": 5}, tool_calls=[]))
        + "\n"
    )
    # The array contributed nothing; the object did.
    assert flight_summary.summarize(flight)["turns"] == 1


def test_no_transcript_is_the_zero_shape(tmp_path: Path) -> None:
    """An episode that kept no transcript writes the same keys at zero, so a
    reader never has to tell "no block" from "empty block"."""
    assert flight_summary.summarize(tmp_path / "nothing") == flight_summary.blank()


def test_truncated_tail_is_not_an_error(tmp_path: Path) -> None:
    """A live writer mid-append leaves half a line; the turns before it still
    count."""
    flight = tmp_path / "flight"
    _write(
        flight,
        [
            {"kind": "request", "request_id": "r1", "created_at": 1.0},
            _response("r1", 2.0, usage={"total_tokens": 7}, tool_calls=[]),
        ],
    )
    with (flight / "llm" / "transcript.jsonl").open("a") as fh:
        fh.write('{"kind": "response", "request_id": "r2"')
    s = flight_summary.summarize(flight)
    assert s["turns"] == 1
    assert s["total_tokens"] == 7
