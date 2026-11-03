"""Collect the participants' results files into the kit, check and score them (``STUDY.md``).

    python -m baselines.human.study_collect --kit <kit> --inbox <dir> [--study-json <figures/capacity/study.json>]

Every ``maps-study-*.json`` in the inbox not yet collected must be the visit's results file
(``study_server.Visit.results_json``): one run of a label on the kit's plan, its sessions by
name, each one's clock and answers. An earlier visit's ``maps-study-*.zip`` must hold
``results/<run id>/`` and nothing else, each file a condition's ``answers.jsonl`` or
``session.json``. Either lands in ``<kit>/results/<run id>/<session>/`` as the visit wrote it,
without replacing a file; a run already there with the same files is noted, one with other files
is refused. ``study_qc`` then checks and scores the run into ``<kit>/scored/<run id>-<arm>``,
``<kit>/collected.jsonl`` records the file, and the run enters the figure's study.json under its
run id, flagged when a check failed. ``<kit>/actions.csv`` lists every collected answer, one row
each, written afresh from ``<kit>/results`` on every pass. A second pass over the same inbox
changes nothing. It exits non-zero when a file is refused or a newly collected run fails its checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import zipfile
from pathlib import Path

from baselines.human import study_qc
from baselines.human.study_server import RESULTS_FORMAT, RESULTS_VERSION, label_of

#: A run id: a label, then the visit's 8 hex.
RUN = r"[a-z0-9]+(?:-[0-9a-f]{8})?"
SESSIONS = ("practice", "solo", "team")
#: The one shape of a results zip's member: a run's condition's ledger or clock, under one
#: folder when a Mac unpacked the file and the participant compressed that folder again.
MEMBER = re.compile(rf"(?:[^/]+/)?(results/({RUN})/(?:practice|solo|team)/(?:answers\.jsonl|session\.json))")
#: What a Mac adds when it compresses a folder: folder entries, Finder's records, resource forks.
MAC_EXTRA = re.compile(r"(?:.*/)?(?:\.DS_Store)?|__MACOSX/.*")


def members(zf: zipfile.ZipFile, plan: dict) -> tuple[str, dict[str, bytes]]:
    """The zip's run id and its files by path under ``results/``; ValueError says what is wrong."""
    runs, files = set(), {}
    for info in zf.infolist():
        m = MEMBER.fullmatch(info.filename)
        if m is None and MAC_EXTRA.fullmatch(info.filename):
            continue
        if m is None:
            raise ValueError(f"{info.filename!r} is not a results file's entry")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f"{info.filename!r} is a link")
        runs.add(m[2])
        files[m[1].removeprefix("results/")] = zf.read(info)
    if len(runs) != 1:
        raise ValueError(f"holds {len(runs)} runs, not one")
    run = runs.pop()
    if label_of(run) not in plan:
        raise ValueError(f"{label_of(run)} is not a participant on this kit's plan")
    return run, files


def json_members(data: bytes, plan: dict) -> tuple[str, dict[str, bytes], list[str]]:
    """The results file's run id, its files by path under ``results/`` as the visit wrote them,
    and the lines the visit could not read, per scored session; ValueError says what is wrong."""
    doc = json.loads(data)
    if not isinstance(doc, dict) or doc.get("format") != RESULTS_FORMAT:
        raise ValueError("is not a MAPS study results file")
    if doc.get("version") != RESULTS_VERSION:
        raise ValueError(f"is results file version {doc.get('version')!r}, not {RESULTS_VERSION}")
    if set(doc) != {"format", "version", "run_id", "participant", "saved_at", "sessions"}:
        raise ValueError(f"holds {sorted(doc)}, not a results file's keys")
    run, sessions = doc["run_id"], doc["sessions"]
    if not isinstance(run, str) or not re.fullmatch(RUN, run) or doc["participant"] != label_of(run):
        raise ValueError(f"run id {run!r} with label {doc['participant']!r} is not a visit's")
    if label_of(run) not in plan:
        raise ValueError(f"{label_of(run)} is not a participant on this kit's plan")
    if not isinstance(sessions, dict) or not sessions or not set(sessions) <= set(SESSIONS):
        raise ValueError(f"holds sessions {sorted(sessions) if isinstance(sessions, dict) else sessions!r}, "
                         f"not some of {', '.join(SESSIONS)}")
    files, failed = {}, []
    for name, s in sessions.items():
        if (not isinstance(s, dict) or not {"session", "answers"} <= set(s) <= {"session", "answers", "events", "unreadable_lines"}
                or not isinstance(s.get("events", []), list)
                or not isinstance(s["session"], dict) or not isinstance(s["answers"], list)
                or not all(isinstance(a, dict) for a in s["answers"])
                or not isinstance(s.get("unreadable_lines", 0), int)):
            raise ValueError(f"session {name!r} is not a session's clock and answers")
        # Written as capacity.Capacity writes them.
        files[f"{run}/{name}/session.json"] = json.dumps(s["session"], indent=1).encode()
        if s["answers"]:
            files[f"{run}/{name}/answers.jsonl"] = "".join(json.dumps(a) + "\n" for a in s["answers"]).encode()
        if s.get("events"):
            files[f"{run}/{name}/events.jsonl"] = "".join(json.dumps(ev) + "\n" for ev in s["events"]).encode()
        if s.get("unreadable_lines") and name != "practice":
            failed.append(f"{name}: {s['unreadable_lines']} unreadable lines in answers.jsonl")
    return run, files, failed


def home_path(p: Path) -> str:
    """A path as study.json writes it: under ``~`` when it is in the home folder."""
    p = p.absolute()
    return f"~/{p.relative_to(Path.home()).as_posix()}" if p.is_relative_to(Path.home()) else str(p)


def collect(kit: Path, path: Path, sha: str, plan: dict) -> dict | None:
    """One new results file unpacked, checked and scored: its record, or None when it adds nothing."""
    if path.suffix == ".json":
        run, files, unreadable = json_members(path.read_bytes(), plan)
    else:
        with zipfile.ZipFile(path) as zf:
            (run, files), unreadable = members(zf, plan), []
    results = kit / "results"
    have = {f.relative_to(results).as_posix(): f.read_bytes()
            for f in (results / run).rglob("*") if f.is_file()} if (results / run).is_dir() else None
    if have == files:
        print(f"{path.name}: results/{run} is already in the kit with the same files")
        return None
    if have is not None:
        raise ValueError(f"results/{run} is already in the kit with other contents: compare the two by hand")
    for rel, data in files.items():
        (results / rel).parent.mkdir(parents=True, exist_ok=True)
        with (results / rel).open("xb") as fh:
            fh.write(data)
    res = study_qc.score(kit, run)
    failed = unreadable + [f"{arm}: {p}" for arm, r in res.items() for p in r["failed"]]
    return {"file": path.name, "sha256": sha, "run_id": run, "label": label_of(run),
            "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "qc": "fail" if failed else "pass",
            "failed": failed, "conditions": {arm: [r["summary"]["correct"], r["summary"]["n"]]
                                             for arm, r in res.items() if r["summary"]}}


#: actions.csv's columns: an answer's ids, then its fields as answers.jsonl holds them, ``xy`` split.
ACTIONS = ("run_id", "participant", "session", "order_index", "episode_id", "source", "x", "y",
           "opened_at", "answered_at", "answer_s", "maps_ready_at_open", "revises")


def write_actions(kit: Path, plan: dict) -> None:
    """``<kit>/actions.csv``: every answer of every collected run, a row each, in the ledgers' order."""
    from baselines.human.audit_console import _jsonl

    results = kit / "results"
    runs = sorted(p for p in results.iterdir() if p.is_dir() and label_of(p.name) in plan) if results.is_dir() else []
    with (kit / "actions.csv").open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(ACTIONS)
        for run in runs:
            for name in SESSIONS:
                for a in _jsonl(run / name / "answers.jsonl"):
                    x, y = a.get("xy") or ("", "")
                    row = {**a, "run_id": run.name, "participant": label_of(run.name), "session": name, "x": x, "y": y}
                    out.writerow([row.get(k, "") for k in ACTIONS])


def add_to_study(study_json: Path, kit: Path, rec: dict) -> None:
    """The run's scored roots and its qc in the figure's study.json, every other key as it was.
    A run that failed a check is recorded too, flagged with what failed; the figure leaves it out."""
    doc = json.loads(study_json.read_text())
    entry = {arm: home_path(kit / "scored" / f"{rec['run_id']}-{arm}") for arm in rec["conditions"]}
    entry["qc"] = rec["qc"]
    if rec["failed"]:
        entry["failed"] = rec["failed"]
    if doc.get("participants", {}).get(rec["run_id"]) != entry:
        doc.setdefault("participants", {})[rec["run_id"]] = entry
        study_json.write_text(json.dumps(doc, indent=1) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", type=Path, required=True)
    ap.add_argument("--inbox", type=Path, required=True, help="the folder the emailed results files are saved in")
    ap.add_argument("--study-json", type=Path, help="the capacity figure's study.json, to record each run in (a failed run flagged)")
    args = ap.parse_args(argv)
    kit, inbox = args.kit.expanduser(), args.inbox.expanduser()
    plan = json.loads((kit / "plan" / "plan.json").read_text())["participants"]
    ledger = kit / "collected.jsonl"
    records = [json.loads(ln) for ln in ledger.read_text().splitlines() if ln.strip()] if ledger.is_file() else []
    # A record from before the results file was JSON names it under "zip".
    seen = {r["sha256"]: r.get("file") or r["zip"] for r in records}
    bad = 0
    for zpath in sorted([*inbox.glob("maps-study-*.json"), *inbox.glob("maps-study-*.zip")]):
        sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
        if sha in seen:
            if seen[sha] != zpath.name:
                print(f"{zpath.name}: already collected as {seen[sha]}")
            continue
        try:
            rec = collect(kit, zpath, sha, plan)
        except (ValueError, zipfile.BadZipFile) as e:
            print(f"REFUSED {zpath.name}: {e}")
            bad += 1
            continue
        if rec is None:
            continue
        with ledger.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        records.append(rec)
        seen[sha] = zpath.name
        if rec["qc"] == "fail":
            bad += 1
            print(f"QC FAILED {rec['run_id']} ({zpath.name}), recorded flagged: " + "; ".join(rec["failed"]))
        if args.study_json:
            add_to_study(args.study_json.expanduser(), kit, rec)
    write_actions(kit, plan)
    last = {r["run_id"]: r for r in records}
    print(f"\n{'run id':<14} {'label':<6} {'conditions':<11} {'solo':>8} {'team':>8}  qc")
    for run, r in sorted(last.items()):
        c = r["conditions"]
        cell = {arm: "{}/{}".format(*c[arm]) if arm in c else "-" for arm in ("solo", "team")}
        print(f"{run:<14} {r['label']:<6} {','.join(c) or '-':<11} {cell['solo']:>8} {cell['team']:>8}  {r['qc']}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
