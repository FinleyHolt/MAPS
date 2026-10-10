"""A serve that reports no ``prompt_tokens_details`` has no cached count: the
meter records none and the rollup reads n/a, never 0%."""

from __future__ import annotations

from maps.llm.net_meter import NetMeter, cached_tokens_of, summarize_window


def test_a_serve_without_cache_reporting_carries_no_cached_count(tmp_path, monkeypatch):
    assert cached_tokens_of({"usage": {"prompt_tokens": 10}}) is None
    assert cached_tokens_of({"usage": {"prompt_tokens": 10, "prompt_tokens_details": {}}}) == 0
    path = tmp_path / "net_meter.jsonl"
    monkeypatch.setenv("NET_METER_LOG", str(path))
    NetMeter.instance().record("reasoner_chat", 1, 1, 0.0, 1.0, prompt_tokens=10, cached_tokens=None)
    assert "total_cached_tokens" not in summarize_window(path)
    NetMeter.instance().record("reasoner_chat", 1, 1, 0.0, 1.0, prompt_tokens=10, cached_tokens=4)
    assert summarize_window(path)["total_cached_tokens"] == 4
