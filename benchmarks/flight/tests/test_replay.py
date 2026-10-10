"""The replay over a recording: the observation bank's coverage rule, the forced diff, the camera ring's read."""

from __future__ import annotations

import json
import struct
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from benchmarks.flight import mission as M
from benchmarks.flight import replay as R
from benchmarks.flight.tests.test_mission import _hydrated


def _held(tmp_path, name, extent, black_east=False):
    rgb = np.full((200, 200, 3), 150, np.uint8)
    if black_east:
        rgb[:, 100:] = 0
    path = tmp_path / f"{name}.jpg"
    Image.fromarray(rgb).save(path, quality=90)
    return R.Held(path=path, extent=extent, camera_z=914.4,
                  frame={"seq": 7, "log_time": 7, "boresight": "/some/where/boresight.json", "corners_xy": None})


@pytest.fixture
def bank_mission(built, floor, tmp_path):
    def make(bank):
        live = R.BankSource(bank, tiles_dir=built[0], map_name="calcity", tile_ground_z=0.0)
        workdir = tmp_path / "sortie" / "run"
        workdir.mkdir(parents=True)
        m = M.Mission(episode_id="calcity::t", split="calcity", config="configs/calcity.yaml", workdir=workdir,
                      live=live, run_no=1)
        m.map_name, m._tiles_dir = "calcity", built[0]
        return m, live
    return make


def test_the_bank_serves_the_observation_holding_most_of_the_area_and_black_ground_is_unheld(built, bank_mission,
                                                                                            tmp_path):
    half = _held(tmp_path, "half", (300.0, 300.0, 900.0, 900.0), black_east=True)
    whole = _held(tmp_path, "whole", (300.0, 300.0, 900.0, 900.0))
    m, live = bank_mission([half, whole])
    area = m._area_lonlat(M.leg_area(_hydrated(built[3]), "", (600.0, 600.0)))
    assert live.coverage(area)["best"][0] == (0.0, 1)
    alone = R.BankSource([half], tiles_dir=built[0], map_name="calcity", tile_ground_z=0.0).coverage(area)
    assert 0.0 < alone["fraction"] < M.COVERED_FRACTION     # the black east half holds nothing
    served = []
    m._fly_live(lambda item: served.append(m._held) or {}, _hydrated(built[3]), {"target_fid": "[600, 600]"})
    assert "no leg is flown" in [r["said"] for r in m.transcript if r.get("kind") == "leg"][-1]
    obs, last = served[0]
    assert obs.world_window == (300.0, 300.0, 900.0, 900.0) and last["boresight"] == "boresight.json"


def test_ground_no_recorded_observation_holds_leaves_the_flight_without_one(built, bank_mission, tmp_path):
    m, live = bank_mission([_held(tmp_path, "far", (300.0, 300.0, 500.0, 500.0))])
    served = []
    m._fly_live(lambda item: served.append(m._held) or {}, _hydrated(built[3]), {"target_fid": "[1000, 1000]"})
    assert served == [None] and len(live.vehicle.legs) == 1
    assert "no frame held any of it" in [r["said"] for r in m.transcript if r.get("kind") == "leg"][-1]


def test_a_replay_names_the_frame_tier_only_when_the_camera_recording_answers(built, floor, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from benchmarks.calcity import _citynav_bridge
    from benchmarks.citynav import episode_workdir

    monkeypatch.setattr(_citynav_bridge, "activate", lambda: None)
    monkeypatch.setattr(episode_workdir, "open_episode_dir",
                        lambda *a, **k: SimpleNamespace(dir=tmp_path, recorder=None, close=lambda: None))
    modes = []

    def named(**kw):
        modes.append(kw["replay_of"]["mode"])
        raise StopIteration

    monkeypatch.setattr(R, "ReplayMission", named)
    rec = R.Recording(sortie=tmp_path / "s", run_dir=tmp_path / "s" / "run", row={"episode_id": "calcity::t", "start_xy": [0, 0]},
                      params={}, requests=[], responses=[], conversation=[], frames=[])
    for live in (None, object()):
        with pytest.raises(StopIteration):
            R.fly(rec, tmp_path, bank=[], live=live)
    assert modes == ["bank", "frames"]


# -- the forced diff -------------------------------------------------------------


def _transcript(path, exchanges, *, unanswered=False, retried=False):
    """A transcript of ``exchanges`` (``(new_messages, response)``) as the harness writes it;
    ``retried``, the first request sent twice, the first time unanswered."""
    lines = [{"kind": "params", "params": {"model": "m", "seed": 1}}]
    history: list[dict] = []
    for i, (new, response) in enumerate(exchanges):
        if retried and i == 0:
            lines.append({"kind": "request", "request_id": "lost", "n_prefix": 0, "new_messages": new})
        lines.append({"kind": "request", "request_id": f"r{i}", "n_prefix": len(history), "new_messages": new})
        history = history + new
        if response is not None:
            lines.append({"kind": "response", "request_id": f"r{i}", "response": response})
    if unanswered:
        lines.append({"kind": "request", "request_id": "rx", "n_prefix": len(history), "new_messages": []})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))


def _run(root, name, receipt, *, unanswered=False, retried=False):
    run = root / name
    call = {"id": "c", "type": "function", "function": {"name": "run", "arguments": "{\"code\": \"go_to(prior_1)\"}"}}
    answer = {"role": "assistant", "content": "", "tool_calls": [call]}
    _transcript(run / "flight" / "llm" / "transcript.jsonl", [
        ([{"role": "system", "content": "s"}, {"role": "user", "content": "find it"}],
         {"choices": [{"message": answer}]}),
        ([answer, {"role": "tool", "tool_call_id": "c", "content": [
            {"type": "text", "text": receipt}, {"type": "image_ref", "sha256": "ab"}]}],
         {"choices": [{"message": {"role": "assistant", "content": "there"}}]})], unanswered=unanswered,
                retried=retried)
    (run / "flight" / "tour_cursor.json").write_text(json.dumps({"stops_visited": [{"stop_id": "prior_1",
                                                                                     "xy": [10.0, 20.0]}]}))
    return run


def test_the_forced_diff_passes_a_replay_that_sent_what_flew_and_names_the_first_receipt_that_did_not(tmp_path):
    sortie = tmp_path / "20260101T000000Z"
    flown = _run(sortie, "maps_run01", "flew to prior_1", retried=True)
    (sortie / "episodes.jsonl").write_text(json.dumps({"run_no": 1, "workdir": "maps_run01"}) + "\n")
    rec = R.open_recording(flown)
    assert len(rec.requests) == 2 and rec.stops == [("prior_1", (10.0, 20.0))]
    same = R.diff(rec, _run(tmp_path / "replay", "same", "flew to prior_1", unanswered=True))
    assert same["reproduced"] and same["first"] is None
    other = R.diff(rec, _run(tmp_path / "replay", "other", "flew to prior_2"))
    assert not other["reproduced"]
    assert (other["first"]["request"], other["first"]["role"]) == (1, "tool")
    assert "+[tool] flew to prior_2" in other["first"]["diff"]


def test_the_recorded_reasoner_speaks_an_operator_turn_where_the_next_request_carried_it():
    picture = {"role": "user", "content": [{"type": "text", "text": "a map"}, {"type": "image_ref", "sha256": "x"}]}
    asked = [{"role": "system", "content": "s"}, {"role": "user", "content": "find it"}]
    answer = {"role": "assistant", "content": "", "tool_calls": []}
    rec = R.Recording(sortie=None, run_dir=None, row={}, params={}, conversation=[], frames=[],
                      requests=[asked, asked + [answer, {"role": "tool", "content": "r"}, picture],
                                asked + [answer, {"role": "tool", "content": "r"}, picture, answer,
                                         {"role": "tool", "content": "r"}, picture,
                                         {"role": "user", "content": "the red one"}]],
                      responses=[{}, {}, {}])
    script = R.Script(rec)
    assert [script._user_turns(k) for k in range(3)] == [0, 0, 1]


# -- the camera ring -------------------------------------------------------------


def _cdr_image(jpeg: bytes) -> bytes:
    def string(s: bytes, at: int) -> bytes:
        pad = b"\0" * (-at % 4)
        return pad + struct.pack("<I", len(s) + 1) + s + b"\0"
    out = b"\0\1\0\0" + struct.pack("<iI", 1, 2)
    out += string(b"oryx", len(out))
    out += string(b"jpeg", len(out))
    out += b"\0" * (-len(out) % 4) + struct.pack("<I", len(jpeg)) + jpeg
    return out


def _recording(path, t0, chunk_size=1 << 20):
    """Five seconds at 10 Hz of fixes and camera frames, flying north at 80 m/s; the frames' JPEGs by sequence."""
    from mcap.writer import Writer

    from benchmarks.flight import recorded_ring as RR

    jpegs = {}
    with path.open("wb") as fh:
        w = Writer(fh, chunk_size=chunk_size)
        w.start()
        js = w.register_schema(name="doc", encoding="jsonschema", data=b"{}")
        img = w.register_schema(name="sensor_msgs/msg/CompressedImage", encoding="ros2msg", data=b"")
        pvt = w.register_channel(topic=RR.TOPIC_PVT, message_encoding="json", schema_id=js)
        cam = w.register_channel(topic=RR.TOPIC_CAMERA, message_encoding="cdr", schema_id=img)
        for k in range(50):
            t = t0 + k * 10**8
            fix = {"lat": 35.0 + k * 8.0 / 111_320.0, "lon": -118.0, "altitudeMslMeters": 1500.0,
                   "altitudeHaeMeters": 1470.0, "speedMps": 80.0, "trackDeg": 0.0, "accuracyMeters": 0.5,
                   "climbMps": 0.0, "timestamp": f"t{k}"}
            w.add_message(pvt, log_time=t, publish_time=t, data=json.dumps(fix).encode(), sequence=k)
            buf = BytesIO()
            Image.fromarray(np.full((4, 4, 3), k, np.uint8)).save(buf, format="JPEG")
            jpegs[k] = buf.getvalue()
            w.add_message(cam, log_time=t + 1, publish_time=t + 1, data=_cdr_image(jpegs[k]), sequence=k)
        w.finish()
    return jpegs


def test_the_ring_reads_a_recording_at_the_spill_rate_posed_by_its_own_fixes_and_up_to_its_clock(tmp_path):
    from benchmarks.flight import recorded_ring as RR

    t0 = 1_800_000_000 * 10**9
    path = tmp_path / "sensors.mcap"
    jpegs = _recording(path, t0)
    ring = RR.RecordedRing(path, start_ns=t0, end_ns=t0 + 5 * 10**9)
    assert [r["seq"] for r in ring.rows] == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    assert ring.index() == []
    ring.sleep(2.0)
    assert [r["seq"] for r in ring.index()] == [0, 5, 10, 15]
    assert [r["seq"] for r in ring.index(since=t0 + 10**9)] == [10, 15]
    jpeg, t_host, log_time, seq, tag = ring.frame(10)
    assert jpeg == jpegs[10] and seq == 10 and log_time == t0 + 10**9 + 1
    assert tag["lat"] == pytest.approx(35.0 + 80.0 / 111_320.0, abs=1e-9) and tag["heading_deg"] == 0.0
    assert ring.frame(11) is None and ring.last_error


def test_the_recorded_reasoner_settings_stand_only_in_a_forced_replay(monkeypatch):
    from types import SimpleNamespace

    flags = {"SEARCH_COMPACT_TOKENIZER_NAME": "recorded/tok", "SEARCH_SAMPLING_REASONING_EFFORT": "0.9",
             "SEARCH_MAP_MAX_PX": "1024", "SEARCH_CALCITY_TILES": "/aircraft/tiles"}
    rec = SimpleNamespace(row={"search_env_flags": flags})
    for k in flags:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SEARCH_COMPACT_TOKENIZER_NAME", "mine/tok")
    R._arm_env(rec, forced=False)
    import os
    assert os.environ["SEARCH_COMPACT_TOKENIZER_NAME"] == "mine/tok"
    assert "SEARCH_SAMPLING_REASONING_EFFORT" not in os.environ
    assert os.environ["SEARCH_MAP_MAX_PX"] == "1024" and "SEARCH_CALCITY_TILES" not in os.environ
    R._arm_env(rec, forced=True)
    assert os.environ["SEARCH_COMPACT_TOKENIZER_NAME"] == "recorded/tok"
    assert os.environ["SEARCH_SAMPLING_REASONING_EFFORT"] == "0.9"


def test_a_torn_recording_serves_a_frame_from_its_own_chunk_without_reading_from_the_top(tmp_path, monkeypatch):
    from benchmarks.flight import recorded_ring as RR

    t0 = 1_800_000_000 * 10**9
    path = tmp_path / "sensors.mcap"
    jpegs = _recording(path, t0, chunk_size=2048)
    data = path.read_bytes()
    path.write_bytes(data[: int(len(data) * 0.7)])          # the summary and footer torn off
    ring = RR.RecordedRing(path, start_ns=t0, end_ns=t0 + 5 * 10**9)
    assert [r["seq"] for r in ring.rows][:3] == [0, 5, 10]

    def streamed(*a, **k):
        raise AssertionError("frame() streamed the recording from the top")

    monkeypatch.setattr(RR, "messages", streamed)
    assert ring.frame(10)[0] == jpegs[10]
