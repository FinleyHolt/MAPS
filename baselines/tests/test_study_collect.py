"""Collecting results files: a good results file (the visit's one JSON document, or an earlier
visit's zip) unpacks, scores, is recorded once and lands in the
figure's study.json beside its other entries; a second copy of it is skipped; a zip whose run id
is already there with other contents is refused; a zip with anything but one run's ledgers is
refused before a byte is written; a missing session fails the QC and stays out of study.json;
a second pass over the inbox changes nothing; actions.csv lists every collected answer."""

import csv
import json
import zipfile

from baselines.human import study_collect
from baselines.tests.test_study_qc import _kit

RUN = "p04-3f9a2c1e"


def _files(answers: str = "skip") -> dict[str, str]:
    session = {"agents": 1, "session_s": 3600.0, "started_epoch": 1000.0, "started_at": "1970-01-01T00:16:40Z",
               "episodes_n": 1, "order": ["block_0::1#0"]}
    answer = {"episode_id": "block_0::1#0", "source": answers, "arm": "solo", "answered_at": "1970-01-01T00:20:00Z"}
    return {f"results/{RUN}/solo/session.json": json.dumps({**session, "arm": "solo"}, indent=1),
            f"results/{RUN}/solo/answers.jsonl": json.dumps(answer) + "\n",
            f"results/{RUN}/practice/answers.jsonl": ""}


def _zip(inbox, name: str, files: dict[str, str]):
    inbox.mkdir(exist_ok=True)
    with zipfile.ZipFile(inbox / name, "w") as z:
        for arc, text in files.items():
            z.writestr(arc, text)
    return inbox / name


def _doc(answers: str = "skip", run: str = RUN) -> dict:
    """The JSON results file the visit saves for the same sessions as ``_files``."""
    sessions = {}
    for path, text in _files(answers).items():
        name, leaf = path.split("/")[2:]
        rows = [json.loads(ln) for ln in text.splitlines()] if leaf == "answers.jsonl" else None
        if leaf == "session.json":
            sessions.setdefault(name, {"answers": []})["session"] = json.loads(text)
        elif name in sessions or rows:
            sessions.setdefault(name, {})["answers"] = rows
    return {"format": "maps-capacity-results", "version": 1, "run_id": run, "participant": run.split("-")[0],
            "saved_at": "2026-10-26T17:00:00Z", "sessions": sessions}


def _json(inbox, name: str, doc) -> None:
    inbox.mkdir(exist_ok=True)
    (inbox / name).write_text(json.dumps(doc, indent=1) + "\n")


def _setup(tmp_path, monkeypatch):
    kit = _kit(tmp_path, monkeypatch)
    for f in sorted((kit / "results").rglob("*"), reverse=True):
        f.unlink() if f.is_file() else f.rmdir()
    (kit / "results").rmdir()
    (kit / "plan" / "plan.json").write_text(json.dumps({"participants": {"p04": {"arm": "solo"}}}))
    monkeypatch.setenv("HOME", str(tmp_path))
    study = tmp_path / "study.json"
    study.write_text(json.dumps({"study": "s", "participants": {"p01": {"solo": "~/old"}}, "plan": "~/plan"}, indent=1) + "\n")
    return kit, tmp_path / "inbox", study


def _collect(kit, inbox, study) -> int:
    return study_collect.main(["--kit", str(kit), "--inbox", str(inbox), "--study-json", str(study)])


def test_a_good_zip_is_collected_once_and_fills_study_json(tmp_path, monkeypatch, capsys) -> None:
    kit, inbox, study = _setup(tmp_path, monkeypatch)
    _zip(inbox, f"maps-study-{RUN}.zip", _files())
    assert _collect(kit, inbox, study) == 0
    out = capsys.readouterr().out
    assert (kit / "results" / RUN / "solo" / "answers.jsonl").is_file() and (kit / "scored" / f"{RUN}-solo").is_dir()
    (rec,) = [json.loads(ln) for ln in (kit / "collected.jsonl").read_text().splitlines()]
    assert (rec["file"], rec["run_id"], rec["label"], rec["qc"], rec["failed"]) == (f"maps-study-{RUN}.zip", RUN, "p04", "pass", [])
    assert rec["conditions"] == {"solo": [0, 1]} and len(rec["sha256"]) == 64
    doc = json.loads(study.read_text())
    assert doc["participants"] == {"p01": {"solo": "~/old"},
                                   RUN: {"solo": f"~/kit/scored/{RUN}-solo", "qc": "pass"}}
    assert doc["plan"] == "~/plan" and list(doc) == ["study", "participants", "plan"]
    assert RUN in out and "0/1" in out and "pass" in out
    before = {p: p.read_bytes() for p in (kit / "collected.jsonl", study)}
    (inbox / "maps-study-copy.zip").write_bytes((inbox / f"maps-study-{RUN}.zip").read_bytes())
    assert _collect(kit, inbox, study) == 0
    assert "already collected" in capsys.readouterr().out
    assert {p: p.read_bytes() for p in before} == before


def test_a_zip_conflicting_with_collected_results_is_refused(tmp_path, monkeypatch, capsys) -> None:
    kit, inbox, study = _setup(tmp_path, monkeypatch)
    _zip(inbox, f"maps-study-{RUN}.zip", _files())
    assert _collect(kit, inbox, study) == 0
    kept = (kit / "results" / RUN / "solo" / "answers.jsonl").read_bytes()
    _zip(inbox, "maps-study-again.zip", _files(answers="own_search"))
    assert _collect(kit, inbox, study) == 1
    assert "REFUSED maps-study-again.zip" in capsys.readouterr().out
    assert (kit / "results" / RUN / "solo" / "answers.jsonl").read_bytes() == kept
    assert len((kit / "collected.jsonl").read_text().splitlines()) == 1


def test_a_zip_holding_anything_else_is_refused_before_anything_is_written(tmp_path, monkeypatch, capsys) -> None:
    kit, inbox, study = _setup(tmp_path, monkeypatch)
    bad = {"traversal": {f"results/{RUN}/../../evil/session.json": "{}"}, "absolute": {"/tmp/session.json": "{}"},
           "extra": {**_files(), "notes.txt": "hi"}, "unknown": {f"results/{RUN}/solo/notes.txt": "hi"},
           "two-runs": {**_files(), "results/p04-0b1c2d3e/solo/session.json": "{}"},
           "off-plan": {"results/p99-3f9a2c1e/solo/session.json": "{}"}}
    for name, files in bad.items():
        _zip(inbox, f"maps-study-{name}.zip", files)
    with zipfile.ZipFile(inbox / "maps-study-link.zip", "w") as z:
        info = zipfile.ZipInfo(f"results/{RUN}/solo/session.json")
        info.external_attr = 0o120777 << 16
        z.writestr(info, "/etc/passwd")
    (inbox / "maps-study-torn.zip").write_bytes(b"not a zip")
    assert _collect(kit, inbox, study) == 1
    out = capsys.readouterr().out
    assert out.count("REFUSED") == len(bad) + 2
    assert not (kit / "results").exists() and not (tmp_path / "evil").exists() and not (kit / "collected.jsonl").exists()


def test_a_missing_session_fails_the_qc_and_is_recorded_flagged(tmp_path, monkeypatch, capsys) -> None:
    kit, inbox, study = _setup(tmp_path, monkeypatch)
    _zip(inbox, f"maps-study-{RUN}.zip", {k: v for k, v in _files().items() if "/solo/" not in k})
    assert _collect(kit, inbox, study) == 1
    (rec,) = [json.loads(ln) for ln in (kit / "collected.jsonl").read_text().splitlines()]
    assert rec["qc"] == "fail" and rec["failed"] == ["solo: no session yet"] and rec["conditions"] == {}
    # Every run is on record; the figure leaves out a flagged one.
    assert json.loads(study.read_text())["participants"][RUN] == {"qc": "fail", "failed": ["solo: no session yet"]}
    assert "QC FAILED" in capsys.readouterr().out
    assert _collect(kit, inbox, study) == 0


def test_a_results_file_a_mac_unpacked_and_compressed_again_is_collected(tmp_path, monkeypatch, capsys) -> None:
    kit, inbox, study = _setup(tmp_path, monkeypatch)
    wrap = f"maps-study-{RUN}"
    files = {f"{wrap}/{k}": v for k, v in _files().items()}
    files.update({f"{wrap}/": "", f"{wrap}/results/": "", f"{wrap}/.DS_Store": "x",
                  f"__MACOSX/{wrap}/._results": "x", f"__MACOSX/{wrap}/results/{RUN}/solo/._answers.jsonl": "x"})
    _zip(inbox, f"{wrap}.zip", files)
    assert _collect(kit, inbox, study) == 0
    assert (kit / "results" / RUN / "solo" / "answers.jsonl").is_file()
    assert not (kit / "results" / ".DS_Store").exists() and not list((kit / "results").rglob("._*"))


def test_a_good_json_is_collected_as_the_visit_wrote_it(tmp_path, monkeypatch, capsys) -> None:
    kit, inbox, study = _setup(tmp_path, monkeypatch)
    _json(inbox, f"maps-study-{RUN}.json", _doc())
    doc = json.loads((inbox / f"maps-study-{RUN}.json").read_text())
    assert _collect(kit, inbox, study) == 0
    solo = kit / "results" / RUN / "solo"
    assert (solo / "session.json").read_text() == json.dumps(doc["sessions"]["solo"]["session"], indent=1)
    assert (solo / "answers.jsonl").read_text() == "".join(json.dumps(a) + "\n" for a in doc["sessions"]["solo"]["answers"])
    assert sorted(p.name for p in (kit / "results" / RUN).iterdir()) == ["solo"]
    (rec,) = [json.loads(ln) for ln in (kit / "collected.jsonl").read_text().splitlines()]
    assert (rec["file"], rec["run_id"], rec["qc"], rec["conditions"]) == (
        f"maps-study-{RUN}.json", RUN, "pass", {"solo": [0, 1]})
    assert json.loads(study.read_text())["participants"][RUN]["qc"] == "pass"
    with (kit / "actions.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["run_id"], r["participant"], r["session"], r["episode_id"], r["source"], r["x"]) for r in rows] == [
        (RUN, "p04", "solo", "block_0::1#0", "skip", "")]
    assert list(rows[0]) == ["run_id", "participant", "session", "order_index", "episode_id", "source", "x", "y",
                             "opened_at", "answered_at", "answer_s", "maps_ready_at_open", "revises"]


def test_two_copies_of_one_results_file_collect_once(tmp_path, monkeypatch, capsys) -> None:
    kit, inbox, study = _setup(tmp_path, monkeypatch)
    _json(inbox, f"maps-study-{RUN}.json", _doc())
    _json(inbox, f"maps-study-{RUN} (1).json", _doc())
    # A second save of the same results differs only in when it was saved.
    _json(inbox, f"maps-study-{RUN}-later.json", _doc() | {"saved_at": "2026-10-26T18:00:00Z"})
    assert _collect(kit, inbox, study) == 0
    out = capsys.readouterr().out
    assert "already collected" in out and "already in the kit with the same files" in out
    assert len((kit / "collected.jsonl").read_text().splitlines()) == 1
    assert len((kit / "actions.csv").read_text().splitlines()) == 2


def test_a_json_conflicting_with_collected_results_is_refused(tmp_path, monkeypatch, capsys) -> None:
    kit, inbox, study = _setup(tmp_path, monkeypatch)
    _json(inbox, f"maps-study-{RUN}.json", _doc())
    assert _collect(kit, inbox, study) == 0
    kept = (kit / "results" / RUN / "solo" / "answers.jsonl").read_bytes()
    _json(inbox, "maps-study-again.json", _doc(answers="own_search"))
    assert _collect(kit, inbox, study) == 1
    assert "REFUSED maps-study-again.json: results/" in capsys.readouterr().out
    assert (kit / "results" / RUN / "solo" / "answers.jsonl").read_bytes() == kept
    assert len((kit / "collected.jsonl").read_text().splitlines()) == 1


def test_a_json_that_is_not_one_runs_results_is_refused_before_anything_is_written(tmp_path, monkeypatch, capsys) -> None:
    kit, inbox, study = _setup(tmp_path, monkeypatch)
    good = _doc()
    bad = {"list": [good], "foreign": good | {"format": "something-else"}, "version": good | {"version": 2},
           "extra-key": good | {"notes": "hi"}, "no-sessions": good | {"sessions": {}},
           "traversal": _doc(run="p04-3f9a2c1e/../../evil"), "off-plan": _doc(run="p99-3f9a2c1e"),
           "label": good | {"participant": "p01"}, "unknown-session": good | {"sessions": {"notes": good["sessions"]["solo"]}},
           "no-answers": good | {"sessions": {"solo": {"session": {}}}},
           "answers-not-rows": good | {"sessions": {"solo": {"session": {}, "answers": ["x"]}}},
           "session-extra": good | {"sessions": {"solo": {**good["sessions"]["solo"], "path": "/home/x"}}}}
    for name, doc in bad.items():
        _json(inbox, f"maps-study-{name}.json", doc)
    (inbox / "maps-study-torn.json").write_text('{"format": "maps-capacity-results", ')
    (inbox / "maps-study-binary.json").write_bytes(b"\xff\xfe\x00")
    assert _collect(kit, inbox, study) == 1
    out = capsys.readouterr().out
    assert out.count("REFUSED") == len(bad) + 2, out
    assert "REFUSED maps-study-off-plan.json: p99 is not a participant on this kit's plan" in out
    assert not (kit / "results").exists() and not (tmp_path / "evil").exists() and not (kit / "collected.jsonl").exists()


def test_unreadable_lines_the_visit_counted_fail_the_qc(tmp_path, monkeypatch, capsys) -> None:
    kit, inbox, study = _setup(tmp_path, monkeypatch)
    doc = _doc()
    doc["sessions"]["solo"]["unreadable_lines"] = 2
    _json(inbox, f"maps-study-{RUN}.json", doc)
    assert _collect(kit, inbox, study) == 1
    (rec,) = [json.loads(ln) for ln in (kit / "collected.jsonl").read_text().splitlines()]
    assert rec["qc"] == "fail" and rec["failed"] == ["solo: 2 unreadable lines in answers.jsonl"]


def test_the_visits_results_file_collects_back_byte_for_byte(tmp_path, monkeypatch, capsys) -> None:
    from baselines.human.study_server import Visit

    kit, inbox, study = _setup(tmp_path, monkeypatch)
    events = {f"results/{RUN}/solo/events.jsonl": json.dumps({"kind": "search", "t": 1.5, "query": "88th & Lime"}) + "\n"}
    for path, text in {**_files(), **events}.items():
        (kit / path).parent.mkdir(parents=True, exist_ok=True)
        (kit / path).write_text(text)
    written = {p.relative_to(kit).as_posix(): p.read_bytes() for p in (kit / "results").rglob("*") if p.is_file()}
    inbox.mkdir()
    (inbox / f"maps-study-{RUN}.json").write_bytes(Visit(kit).results_json())
    for f in sorted((kit / "results").rglob("*"), reverse=True):
        f.unlink() if f.is_file() else f.rmdir()
    assert _collect(kit, inbox, study) == 0
    back = {p.relative_to(kit).as_posix(): p.read_bytes() for p in (kit / "results").rglob("*") if p.is_file()}
    # Practice without a clock (a session that never opened) is not a session the file carries.
    assert back == {k: v for k, v in written.items() if "/practice/" not in k}
