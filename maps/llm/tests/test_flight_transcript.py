"""The flight transcript stores the conversation once — deltas, not snapshots.

``maps.llm.http`` appends three record kinds to
``flight/llm/transcript.jsonl``: ``params`` (one per distinct
tools/schema/sampling body), ``request`` (message delta with
``n_prefix``), ``response``. These tests pin the write-time contract the
replay reader reconstructs from: exact prefix+delta round-trip, image
parts scrubbed to sha refs, params dedup, pending-then-closed ordering, and
compaction (a shrunken history yields a shorter prefix, never a mis-slice).
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from maps.llm import http as llm_http

def _records(directory: Path) -> list[dict]:
    text = (directory / "transcript.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _reconstruct(records: list[dict]) -> dict[str, list]:
    """Reader-side reconstruction: request_id -> full message list."""
    prev: list = []
    out: dict[str, list] = {}
    for rec in records:
        if rec.get("kind") != "request":
            continue
        prev = prev[: rec["n_prefix"]] + list(rec["new_messages"])
        out[rec["request_id"]] = prev
    return out


def _body(messages: list, **params) -> dict:
    return {
        "model": "qwen",
        "temperature": 0.6,
        "tools": [{"type": "function", "function": {"name": "act"}}],
        **params,
        "messages": messages,
    }


def test_sequential_calls_store_only_the_delta_and_round_trip(tmp_path):
    msgs1 = [
        {"role": "system", "content": "You fly."},
        {"role": "user", "content": "stop 0"},
    ]
    msgs2 = [*msgs1,
             {"role": "assistant", "content": "looking"},
             {"role": "user", "content": "stop 1"}]
    llm_http._flight_request_record(tmp_path, "rid1", _body(msgs1))
    llm_http._flight_request_record(tmp_path, "rid2", _body(msgs2))

    recs = _records(tmp_path)
    reqs = [r for r in recs if r["kind"] == "request"]
    assert [r["request_id"] for r in reqs] == ["rid1", "rid2"]
    assert reqs[0]["n_prefix"] == 0 and reqs[0]["n_messages"] == 2
    # The second call stores ONLY the two new messages.
    assert reqs[1]["n_prefix"] == 2 and len(reqs[1]["new_messages"]) == 2

    rebuilt = _reconstruct(recs)
    assert rebuilt["rid1"] == msgs1
    assert rebuilt["rid2"] == msgs2


def test_image_parts_become_sha_refs_and_no_data_urls_survive(tmp_path):
    raw = b"\xff\xd8\xff\xe0 jpeg bytes for the scrub test"
    data_url = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "Overhead crops"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]
    body = _body(msgs)
    # Vacuity floor: the input really carries an image_url data part —
    # without this, a scrub that dropped the message entirely would pass.
    assert "data:image" in json.dumps(body)

    llm_http._flight_request_record(tmp_path, "rid1", body)
    text = (tmp_path / "transcript.jsonl").read_text()
    assert "data:image" not in text

    (req,) = [r for r in _records(tmp_path) if r["kind"] == "request"]
    part = req["new_messages"][0]["content"][1]
    assert part["type"] == "image_ref"
    assert part["sha256"] == hashlib.sha256(raw).hexdigest()
    assert part["bytes"] == len(raw)
    # The caller's body is untouched — the scrub never mutates the wire dict.
    assert body["messages"][0]["content"][1]["type"] == "image_url"


def test_params_are_written_once_per_distinct_body(tmp_path):
    llm_http._flight_request_record(
        tmp_path, "rid1", _body([{"role": "user", "content": "a"}]))
    llm_http._flight_request_record(
        tmp_path, "rid2", _body([{"role": "user", "content": "b"}]))
    llm_http._flight_request_record(
        tmp_path, "rid3",
        _body([{"role": "user", "content": "c"}], response_format={"x": 1}))

    recs = _records(tmp_path)
    params = [r for r in recs if r["kind"] == "params"]
    assert len(params) == 2  # one shared, one for the differing body
    reqs = [r for r in recs if r["kind"] == "request"]
    assert reqs[0]["params_sha"] == reqs[1]["params_sha"]
    assert reqs[2]["params_sha"] != reqs[0]["params_sha"]
    # messages / stream flags never leak into params.
    for p in params:
        assert "messages" not in p["params"]


def test_a_request_precedes_its_response_and_pending_is_readable(tmp_path):
    llm_http._flight_request_record(
        tmp_path, "rid1", _body([{"role": "user", "content": "a"}]))
    recs = _records(tmp_path)
    assert [r["kind"] for r in recs] == ["params", "request"]  # pending round

    llm_http._flight_response_record(
        tmp_path, "rid1", {"choices": [], "usage": {"total_tokens": 7}})
    recs = _records(tmp_path)
    assert recs[-1]["kind"] == "response"
    assert recs[-1]["request_id"] == "rid1"
    assert recs[-1]["response"]["usage"]["total_tokens"] == 7


def test_compaction_yields_a_shorter_prefix_never_a_mis_slice(tmp_path):
    msgs1 = [
        {"role": "system", "content": "You fly."},
        {"role": "user", "content": "stop 0"},
        {"role": "assistant", "content": "long history"},
        {"role": "user", "content": "stop 1"},
    ]
    # Compaction: the history SHRANK and diverges after the system message.
    msgs2 = [
        {"role": "system", "content": "You fly."},
        {"role": "user", "content": "compacted summary"},
    ]
    llm_http._flight_request_record(tmp_path, "rid1", _body(msgs1))
    llm_http._flight_request_record(tmp_path, "rid2", _body(msgs2))

    recs = _records(tmp_path)
    reqs = [r for r in recs if r["kind"] == "request"]
    assert reqs[1]["n_prefix"] == 1  # only the system message survived
    assert reqs[1]["n_prefix"] < reqs[0]["n_messages"]
    assert _reconstruct(recs)["rid2"] == msgs2
