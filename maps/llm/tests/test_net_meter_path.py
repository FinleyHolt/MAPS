"""The meter follows NET_METER_LOG when it moves.

The episode server rewrites ``NET_METER_LOG`` per episode inside one
process; a singleton that latched the first path wrote two episodes'
calls into one file (652 meter files for 1007 workdirs on the pulled
fleet), which is the one number ``flight_summary`` sources from the
meter — ``reasoning_tokens`` — reading 0 or over-counting.
"""

from __future__ import annotations

import json

from maps.llm.net_meter import NetMeter


def _record(meter: NetMeter) -> None:
    meter.record("vllm", up_bytes=1, down_bytes=1, ts_start=0.0, ts_end=1.0)


def test_meter_follows_env_path_change(tmp_path, monkeypatch):
    a = tmp_path / "ep_a" / "net_meter.jsonl"
    b = tmp_path / "ep_b" / "net_meter.jsonl"

    monkeypatch.setenv("NET_METER_LOG", str(a))
    _record(NetMeter.instance())
    monkeypatch.setenv("NET_METER_LOG", str(b))
    _record(NetMeter.instance())

    for path in (a, b):
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        assert len(lines) == 1, f"{path} holds {len(lines)} records, want 1"

    monkeypatch.delenv("NET_METER_LOG")
    assert not NetMeter.instance().enabled
