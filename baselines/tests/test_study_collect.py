"""Collecting results files: a good zip unpacks, scores, is recorded once and lands in the
figure's study.json beside its other entries; a second copy of it is skipped; a zip whose run id
is already there with other contents is refused; a zip with anything but one run's ledgers is
refused before a byte is written; a missing condition fails the QC and stays out of study.json;
a second pass over the inbox changes nothing."""

import json
import zipfile

from baselines.human import study_collect
from baselines.tests.test_study_qc import _kit

RUN = "p04-3f9a2c1e"


def _files(answers: str = "skip") -> dict[str, str]:
    session = {"agents": 1, "session_s": 2400.0, "started_epoch": 1000.0, "started_at": "1970-01-01T00:16:40Z",
               "episodes_n": 1, "order": ["block_0::1#0"]}
    answer = {"episode_id": "block_0::1#0", "source": answers, "arm": "solo", "answered_at": "1970-01-01T00:20:00Z"}
    return {f"results/{RUN}/solo/session.json": json.dumps({**session, "arm": "solo"}),
            f"results/{RUN}/solo/answers.jsonl": json.dumps(answer) + "\n",
            f"results/{RUN}/team/session.json": json.dumps({**session, "arm": "team"}),
            f"results/{RUN}/practice/answers.jsonl": ""}


def _zip(inbox, name: str, files: dict[str, str]):
    inbox.mkdir(exist_ok=True)
    with zipfile.ZipFile(inbox / name, "w") as z:
        for arc, text in files.items():
            z.writestr(arc, text)
    return inbox / name


def _setup(tmp_path, monkeypatch):
    kit = _kit(tmp_path, monkeypatch)
    for f in sorted((kit / "results").rglob("*"), reverse=True):
        f.unlink() if f.is_file() else f.rmdir()
    (kit / "results").rmdir()
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
    assert (kit / "results" / RUN / "solo" / "answers.jsonl").is_file() and (kit / "scored" / f"{RUN}-team").is_dir()
    (rec,) = [json.loads(ln) for ln in (kit / "collected.jsonl").read_text().splitlines()]
    assert (rec["zip"], rec["run_id"], rec["label"], rec["qc"], rec["failed"]) == (f"maps-study-{RUN}.zip", RUN, "p04", "pass", [])
    assert rec["conditions"] == {"solo": [0, 1], "team": [1, 1]} and len(rec["sha256"]) == 64
    doc = json.loads(study.read_text())
    assert doc["participants"] == {"p01": {"solo": "~/old"},
                                   RUN: {"solo": f"~/kit/scored/{RUN}-solo", "team": f"~/kit/scored/{RUN}-team", "qc": "pass"}}
    assert doc["plan"] == "~/plan" and list(doc) == ["study", "participants", "plan"]
    assert RUN in out and "0/1" in out and "1/1" in out and "pass" in out
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


def test_a_missing_condition_fails_the_qc_and_is_recorded_flagged(tmp_path, monkeypatch, capsys) -> None:
    kit, inbox, study = _setup(tmp_path, monkeypatch)
    _zip(inbox, f"maps-study-{RUN}.zip", {k: v for k, v in _files().items() if "/team/" not in k})
    assert _collect(kit, inbox, study) == 1
    (rec,) = [json.loads(ln) for ln in (kit / "collected.jsonl").read_text().splitlines()]
    assert rec["qc"] == "fail" and rec["failed"] == ["team: no session yet"] and rec["conditions"] == {"solo": [0, 1]}
    # Every run is on record; the figure leaves out a flagged one.
    assert json.loads(study.read_text())["participants"][RUN] == {
        "solo": f"~/kit/scored/{RUN}-solo", "qc": "fail", "failed": ["team: no session yet"]}
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
