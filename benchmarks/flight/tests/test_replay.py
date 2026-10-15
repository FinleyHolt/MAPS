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
from benchmarks.flight.tests import world


def _held(tmp_path, name, extent, black_east=False):
    rgb = np.full((200, 200, 3), 150, np.uint8)
    if black_east:
        rgb[:, 100:] = 0
    path = tmp_path / f"{name}.jpg"
    Image.fromarray(rgb).save(path, quality=90)
    return R.Held(path=path, extent=extent, camera_z=914.4,
                  frame={"seq": 7, "log_time": 7, "boresight": "/some/where/boresight.json", "corners_xy": None})


@pytest.fixture
def bank_mission(built, tmp_path):
    def make(bank):
        live = R.BankSource(bank, tiles_dir=built[0], map_name="calcity", tile_ground_z=0.0)
        world.recording_legs(live.vehicle)
        workdir = tmp_path / "sortie" / "run"
        workdir.mkdir(parents=True)
        m = M.Mission(episode_id="calcity::t", split="calcity", config="configs/calcity.yaml", workdir=workdir,
                      live=live, recorder=None, close_log=lambda: None, run_no=1,
                      agl_m=world.AGL_M)
        m.map_name, m._tiles_dir = "calcity", built[0]
        return m, live
    return make


def test_the_bank_serves_the_observation_holding_most_of_the_area_and_black_ground_is_unheld(built, bank_mission,
                                                                                            tmp_path):
    half = _held(tmp_path, "half", (300.0, 300.0, 900.0, 900.0), black_east=True)
    whole = _held(tmp_path, "whole", (300.0, 300.0, 900.0, 900.0))
    m, live = bank_mission([half, whole])
    area = m._area_lonlat(M.leg_area(world.hydrated(built[3]), "", (600.0, 600.0)))
    assert live.coverage(area)["best"][0] == (0.0, 1)
    alone = R.BankSource([half], tiles_dir=built[0], map_name="calcity", tile_ground_z=0.0).coverage(area)
    assert 0.0 < alone["fraction"] < M.COVERED_FRACTION     # the black east half holds nothing
    served = []
    m._fly_live(world.session(world.hydrated(built[3]), lambda item: served.append(m._held) or {}), {"target_fid": "[600, 600]"})
    assert "no leg is flown" in [r["said"] for r in m.transcript if r.get("kind") == "leg"][-1]
    obs, last = served[0]
    assert obs.world_window == (300.0, 300.0, 900.0, 900.0) and last["boresight"] == "boresight.json"


def test_ground_no_recorded_observation_holds_leaves_the_flight_without_one(built, bank_mission, tmp_path):
    m, live = bank_mission([_held(tmp_path, "far", (300.0, 300.0, 500.0, 500.0))])
    served = []
    m._fly_live(world.session(world.hydrated(built[3]), lambda item: served.append(m._held) or {}), {"target_fid": "[1000, 1000]"})
    assert served == [None] and len(live.vehicle.legs) == 1
    assert "no frame held any of it" in [r["said"] for r in m.transcript if r.get("kind") == "leg"][-1]


def test_a_replay_names_the_frame_tier_only_when_the_camera_recording_answers_and_flies_the_caller_s_effort(
        built, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import benchmarks.calcity
    from benchmarks.citynav import episode_workdir, paths

    monkeypatch.setattr(benchmarks.calcity, "use", lambda: None)
    monkeypatch.setattr(paths, "TILES_DIR", built[0])
    monkeypatch.setattr(episode_workdir, "open_episode_dir",
                        lambda *a, **k: SimpleNamespace(dir=tmp_path, recorder=None, close=lambda: None))
    modes = []

    def named(**kw):
        modes.append((kw["replay_of"]["mode"], kw["agl_m"], kw["reasoning_effort"]))
        raise StopIteration

    monkeypatch.setattr(R, "ReplayMission", named)
    rec = R.Recording(sortie=tmp_path / "s", run_dir=tmp_path / "s" / "run", row={"episode_id": "calcity::t", "start_xy": [0, 0]},
                      params={"reasoning_effort": "0.99"}, requests=[], responses=[], conversation=[],
                      frames=[{"altitude_m": 1980.0}, {"altitude_m": 2000.0}, {"altitude_m": 1990.0}])
    for live in (None, object()):
        with pytest.raises(StopIteration):
            R.fly(rec, tmp_path, bank=[], live=live)
    assert modes == [("bank", 1990.0, None), ("frames", 1990.0, None)]


def test_a_replay_over_a_bake_with_no_georef_is_refused_before_its_sortie_folder_is_opened(tmp_path, monkeypatch):
    import benchmarks.calcity
    from benchmarks.citynav import paths

    monkeypatch.setattr(benchmarks.calcity, "use", lambda: None)
    monkeypatch.setenv("SEARCH_BENCHMARK_NAME", "calcity")
    monkeypatch.setattr(paths, "TILES_DIR", tmp_path / "tiles")
    monkeypatch.setattr(R, "ReplayMission", lambda **kw: pytest.fail("the mission was built"))
    rec = R.Recording(sortie=tmp_path / "s", run_dir=tmp_path / "s" / "run",
                      row={"episode_id": "calcity::t", "start_xy": [0, 0]}, params={}, requests=[], responses=[],
                      conversation=[], frames=[{"altitude_m": 1990.0}])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="carries no georef"):
        R.fly(rec, out, bank=[])
    assert not (out / "s").exists()


def test_a_replay_flies_at_the_recorded_height_and_a_forced_one_at_the_config_s(tmp_path):
    """The recorded run's height above ground: its observations' camera
    heights, else its GNSS track over the run; forced, the config's start
    height, where the flown engine was hydrated."""
    sortie = tmp_path / "s"
    sortie.mkdir()
    floor = R.FLOOR_MSL_M["calcity"]
    (sortie / "flight_track.jsonl").write_text("".join(
        json.dumps({"t_unix": t, "alt_msl_m": floor + agl}) + "\n" for t, agl in ((5, 50.0), (10, 990.0), (20, 1010.0),
                                                                                   (30, 1000.0), (99, 50.0))))
    row = {"map_name": "calcity", "started_at_utc": 10.0, "elapsed_s": 20.0}

    def rec(frames):
        return R.Recording(sortie=sortie, run_dir=sortie / "run", row=row, params={}, requests=[], responses=[],
                           conversation=[], frames=frames)

    assert R.replay_agl_m(rec([{"altitude_m": 1980.0}]), forced=False, config="configs/calcity.yaml") == 1980.0
    assert R.replay_agl_m(rec([]), forced=False, config="configs/calcity.yaml") == pytest.approx(1000.0)
    assert R.replay_agl_m(rec([{"altitude_m": 1980.0}]), forced=True, config="configs/calcity.yaml") == 914.4
    row["started_at_utc"] = 500.0
    with pytest.raises(ValueError, match="no height above ground"):
        R.replay_agl_m(rec([]), forced=False, config="configs/calcity.yaml")


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
                                         {"role": "user", "content": "[Repair] call run with code"},
                                         {"role": "user", "content": "the red one"}]],
                      responses=[{}, {}, {}])
    script = R.Script(rec)
    # The harness's own [Repair] nudge is not an operator's turn.
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


def test_the_recorded_segmenter_keeps_a_recorded_object_id_of_zero():
    from types import SimpleNamespace

    from maps.util.mask_codec import encode_mask_record

    mask = np.zeros((8, 8), bool)
    mask[2:5, 2:5] = True
    record = encode_mask_record(mask, bbox_rcrc=(2, 2, 4, 4))
    rows = [{"label": "car", "confidence": 0.9, "metadata": {"sam3_1_object_id": oid}, **record} for oid in (3, 0)]
    rec = SimpleNamespace(detections={0: rows})
    seen = []
    client = SimpleNamespace(_detections=lambda obs, masks: seen.extend(masks) or [])
    R.recorded_segmenter(rec, lambda: SimpleNamespace(step_idx=0))(client, SimpleNamespace(rgb=np.zeros((8, 8, 3))), [])
    assert [m[2] for m in seen] == [3, 0]


def test_a_replay_records_no_start_time_and_no_track(built, make_twin, make_live, tmp_path):
    """A replay's clock is not the flight's: its row carries no UTC start and
    it writes no track, so the measures read it on SR and NE alone."""
    workdir = tmp_path / "sortie" / "run"
    workdir.mkdir(parents=True)
    m = R.ReplayMission(replay_of={"mode": "frames"}, episode_id="calcity::t", split="calcity",
                        config="configs/calcity.yaml", workdir=workdir, live=make_live(make_twin()), recorder=None,
                        close_log=lambda: None, agl_m=world.AGL_M)
    m.map_name, m._tiles_dir = "calcity", built[0]
    m.started_at, m._t0, m.started_at_utc = 0.0, 0.0, m._utc_now()
    m._live.vehicle.sleep(5.0)
    m.bank()
    row = json.loads((workdir.parent / "episodes.jsonl").read_text())
    assert row["started_at_utc"] is None and row["replay_of"] == {"mode": "frames"}
    assert not (workdir.parent / "flight_track.jsonl").exists()


def test_a_forced_replay_parked_with_no_recorded_tasking_ends_instead_of_waiting(built, make_twin, make_live,
                                                                                 tmp_path):
    """A forced replay that diverges into a report the recording holds no
    operator answer to has nothing to wait for: it ends at that report."""
    import threading

    workdir = tmp_path / "sortie" / "run"
    workdir.mkdir(parents=True)
    m = R.ReplayMission(replay_of={"mode": "forced"}, stops=[], episode_id="calcity::t", split="calcity",
                        config="configs/calcity.yaml", workdir=workdir, live=make_live(make_twin()), recorder=None,
                        close_log=lambda: None, agl_m=world.AGL_M)
    got = []
    t = threading.Thread(target=lambda: got.append(m._await_tasking()), daemon=True)
    t.start()
    t.join(5.0)
    assert not t.is_alive() and got == [None]
    m.tasking("go on")
    assert m._await_tasking() == "go on"


def test_the_bank_reads_an_area_s_pixels_once_however_often_it_is_asked(built, tmp_path, monkeypatch):
    """Arrival polls the same area every simulated second: a week's bank
    answers the poll from memory, not by decoding every observation again."""
    reads = []
    pixels = R._pixels
    monkeypatch.setattr(R, "_pixels", lambda path: reads.append(path) or pixels(path))
    live = R.BankSource([_held(tmp_path, "whole", (300.0, 300.0, 900.0, 900.0))], tiles_dir=built[0],
                        map_name="calcity", tile_ground_z=0.0)
    area = _disc_lonlat(built[0], 600.0, 600.0)
    first = live.coverage(area)
    n = len(reads)
    assert n and live.coverage(area) == first and len(reads) == n


def _disc_lonlat(tiles, x, y):
    from benchmarks.flight.tile_georef import local_to_lonlat

    return [local_to_lonlat(tiles, "calcity", x + dx, y + dy) for dx, dy in ((0, 0), (50, 0), (0, 50), (-50, 0))]


def test_a_torn_line_in_a_recorded_transcript_is_refused_by_file_and_line(tmp_path):
    rows = [{"kind": "params", "params": {"model": "m"}},
            {"kind": "request", "request_id": "a", "n_prefix": 0, "new_messages": [{"role": "user", "content": "go"}]},
            {"kind": "response", "request_id": "a", "response": {"x": 1}}]
    lines = [json.dumps(r) for r in rows]
    lines[1] = lines[1][:25]
    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=f"{path}:2"):
        R.read_transcript(path)


@pytest.mark.parametrize("flags", [["--forced", "--bank", "week"], ["--frames", "x.mcap", "--bank", "week"],
                                   ["--forced", "--reasoning-effort", "high"], ["--raster", "r"],
                                   ["--boresight", "b.json"], ["--nav", "n.mcap"]])
def test_a_replay_refuses_a_flag_its_tier_would_ignore(flags, capsys):
    """``--bank`` chooses the bank tier's observations and ``--reasoning-effort``
    the effort of the tiers your reasoner answers: a tier that reads neither refuses them."""
    with pytest.raises(SystemExit) as exc:
        R.main(["run", *flags])
    assert exc.value.code == 2 and flags[-2] in capsys.readouterr().err


def test_a_week_bank_serves_every_run_s_observations_past_a_torn_line_in_another_run(tmp_path, monkeypatch):
    """The week's other runs serve only their observations: a torn line in
    one's records skips that line, not the replay."""
    from types import SimpleNamespace

    frame = {"corners_xy": [[0, 0], [100, 0], [100, 100], [0, 100]], "altitude_m": 900.0}
    runs = []
    for sortie, sha in (("20260101T000000Z", "aa"), ("20260102T000000Z", "bb")):
        run = _run(tmp_path / sortie, "maps_run01", "flew to prior_1")
        (run / "observations").mkdir()
        (run / "observations" / f"{sha}.jpg").write_bytes(b"jpeg")
        (run / "frames.jsonl").write_text(json.dumps({**frame, "observation": sha}) + "\n")
        (run.parent / "episodes.jsonl").write_text(json.dumps({"run_no": 1, "workdir": "maps_run01", "start_xy": [0, 0]}) + "\n")
        runs.append(run)
    for name in ("frames.jsonl", "flight/llm/transcript.jsonl"):
        with (runs[1] / name).open("a") as f:
            f.write('{"torn\n')
    banks = []
    monkeypatch.setattr(R, "fly", lambda rec, out, *, bank, **kw: banks.append(bank) or (SimpleNamespace(error=None), None))
    monkeypatch.setattr(R, "summary_line", lambda mission: "")
    R.main([str(runs[0]), "--bank", "week", "--out", str(tmp_path / "out")])
    assert [h.path.parent.parent for h in banks[0]] == runs


def test_a_week_bank_reads_past_a_torn_ledger_line_in_another_sortie(tmp_path, monkeypatch):
    """A torn line in another sortie's ``episodes.jsonl`` skips that line, not the replay."""
    from types import SimpleNamespace

    frame = {"corners_xy": [[0, 0], [100, 0], [100, 100], [0, 100]], "altitude_m": 900.0}
    runs = []
    for sortie, sha in (("20260101T000000Z", "aa"), ("20260102T000000Z", "bb")):
        run = _run(tmp_path / sortie, "maps_run01", "flew to prior_1")
        (run / "observations").mkdir()
        (run / "observations" / f"{sha}.jpg").write_bytes(b"jpeg")
        (run / "frames.jsonl").write_text(json.dumps({**frame, "observation": sha}) + "\n")
        (run.parent / "episodes.jsonl").write_text(json.dumps({"run_no": 1, "workdir": "maps_run01", "start_xy": [0, 0]}) + "\n")
        runs.append(run)
    with (runs[1].parent / "episodes.jsonl").open("a") as f:
        f.write('{"torn\n')
    banks = []
    monkeypatch.setattr(R, "fly", lambda rec, out, *, bank, **kw: banks.append(bank) or (SimpleNamespace(error=None), None))
    monkeypatch.setattr(R, "summary_line", lambda mission: "")
    R.main([str(runs[0]), "--bank", "week", "--out", str(tmp_path / "out")])
    assert [h.path.parent.parent for h in banks[0]] == runs


def test_a_replay_of_a_run_that_never_reached_the_reasoner_is_refused_by_name_before_its_frames_are_read(
        tmp_path, monkeypatch, capsys):
    import benchmarks.flight.recorded_ring as ring

    rec = R.Recording(sortie=tmp_path / "s", run_dir=tmp_path / "s" / "run", row={"episode_id": "calcity::t"},
                      params={}, requests=[], responses=[], conversation=[], frames=[])
    monkeypatch.setattr(R, "open_recording", lambda run: rec)
    monkeypatch.setattr(ring, "RecordedRing", lambda *a, **kw: pytest.fail("the frames were read"))
    with pytest.raises(SystemExit) as refused:
        R.main(["run", "--frames", str(tmp_path / "cam.mcap"), "--out", str(tmp_path / "out")])
    assert refused.value.code == 2 and "records no start: the run never reached the reasoner" in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["frames", "boresight", "raster"])
def test_a_frame_tier_input_that_is_not_there_is_refused_before_the_ring_is_read(
        tmp_path, monkeypatch, capsys, missing):
    import benchmarks.flight.recorded_ring as ring

    rec = R.Recording(sortie=tmp_path / "s", run_dir=tmp_path / "s" / "run",
                      row={"episode_id": "calcity::t", "start_xy": [0.0, 0.0]},
                      params={}, requests=[], responses=[], conversation=[], frames=[])
    monkeypatch.setattr(R, "open_recording", lambda run: rec)
    monkeypatch.setattr(ring, "RecordedRing", lambda *a, **kw: pytest.fail("the frames were read"))
    given = {"frames": tmp_path / "cam.mcap", "boresight": tmp_path / "chain.json", "raster": tmp_path / "raster"}
    given["raster"].mkdir()
    for name in ("calcity.tif", "calcity_rgb.tif"):
        (given["raster"] / name).write_bytes(b"")
    given["frames"].write_bytes(b"")
    given["boresight"].write_text("{}")
    if missing == "raster":
        (given["raster"] / "calcity_rgb.tif").unlink()
    else:
        given[missing].unlink()
    with pytest.raises(SystemExit) as refused:
        R.main(["run", *(a for k, p in given.items() for a in (f"--{k}", str(p))), "--out", str(tmp_path / "out")])
    assert refused.value.code == 2 and str(given[missing]) in capsys.readouterr().err
