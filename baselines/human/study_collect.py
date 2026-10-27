"""Collect the participants' results files into the kit, check and score them (``STUDY.md``).

    python -m baselines.human.study_collect --kit <kit> --inbox <dir> [--study-json <figures/capacity/study.json>]

Every ``maps-study-*.zip`` in the inbox not yet collected must hold ``results/<run id>/`` and
nothing else: one run of a label on the kit's plan, each file a condition's ``answers.jsonl``
or ``session.json``. It unpacks into ``<kit>/results`` without replacing a file; a run already
there with the same files is noted, one with other files is refused. ``study_qc`` then checks
and scores the run into ``<kit>/scored/<run id>-<arm>``, ``<kit>/collected.jsonl`` records the
zip, and a run that passes its checks enters the figure's study.json under its run id. A second
pass over the same inbox changes nothing. It exits non-zero when a zip is refused or a newly
collected run fails its checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import zipfile
from pathlib import Path

from baselines.human import study_qc
from baselines.human.study_server import label_of

#: The one shape of a results file's member: a run's condition's ledger or clock.
MEMBER = re.compile(r"results/([a-z0-9]+(?:-[0-9a-f]{8})?)/(?:practice|solo|team)/(?:answers\.jsonl|session\.json)")


def members(zf: zipfile.ZipFile, plan: dict) -> tuple[str, dict[str, bytes]]:
    """The zip's run id and its files by path under ``results/``; ValueError says what is wrong."""
    runs, files = set(), {}
    for info in zf.infolist():
        m = MEMBER.fullmatch(info.filename)
        if m is None:
            raise ValueError(f"{info.filename!r} is not a results file's entry")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f"{info.filename!r} is a link")
        runs.add(m[1])
        files[info.filename.removeprefix("results/")] = zf.read(info)
    if len(runs) != 1:
        raise ValueError(f"holds {len(runs)} runs, not one")
    run = runs.pop()
    if label_of(run) not in plan:
        raise ValueError(f"{label_of(run)} is not a participant on this kit's plan")
    return run, files


def home_path(p: Path) -> str:
    """A path as study.json writes it: under ``~`` when it is in the home folder."""
    p = p.absolute()
    return f"~/{p.relative_to(Path.home()).as_posix()}" if p.is_relative_to(Path.home()) else str(p)


def collect(kit: Path, zpath: Path, sha: str, plan: dict) -> dict | None:
    """One new zip unpacked, checked and scored: its record, or None when it adds nothing."""
    with zipfile.ZipFile(zpath) as zf:
        run, files = members(zf, plan)
    results = kit / "results"
    have = {f.relative_to(results).as_posix(): f.read_bytes()
            for f in (results / run).rglob("*") if f.is_file()} if (results / run).is_dir() else None
    if have is not None and have != files:
        raise ValueError(f"results/{run} is already in the kit with other contents: compare the two by hand")
    if have is None:
        for rel, data in files.items():
            (results / rel).parent.mkdir(parents=True, exist_ok=True)
            with (results / rel).open("xb") as fh:
                fh.write(data)
    else:
        print(f"{zpath.name}: results/{run} is already in the kit with the same files")
    res = study_qc.score(kit, run)
    failed = [f"{arm}: {p}" for arm, r in res.items() for p in r["failed"]]
    return {"zip": zpath.name, "sha256": sha, "run_id": run, "label": label_of(run),
            "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "qc": "fail" if failed else "pass",
            "failed": failed, "conditions": {arm: [r["summary"]["correct"], r["summary"]["n"]]
                                             for arm, r in res.items() if r["summary"]}}


def add_to_study(study_json: Path, kit: Path, rec: dict) -> None:
    """The run's scored roots in the figure's study.json, every other key as it was."""
    doc = json.loads(study_json.read_text())
    entry = {arm: home_path(kit / "scored" / f"{rec['run_id']}-{arm}") for arm in rec["conditions"]}
    if doc.get("participants", {}).get(rec["run_id"]) != entry:
        doc.setdefault("participants", {})[rec["run_id"]] = entry
        study_json.write_text(json.dumps(doc, indent=1) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", type=Path, required=True)
    ap.add_argument("--inbox", type=Path, required=True, help="the folder the emailed results files are saved in")
    ap.add_argument("--study-json", type=Path, help="the capacity figure's study.json, to add each passing run to")
    args = ap.parse_args(argv)
    kit, inbox = args.kit.expanduser(), args.inbox.expanduser()
    plan = json.loads((kit / "plan" / "plan.json").read_text())["participants"]
    ledger = kit / "collected.jsonl"
    records = [json.loads(ln) for ln in ledger.read_text().splitlines() if ln.strip()] if ledger.is_file() else []
    seen = {r["sha256"]: r["zip"] for r in records}
    bad = 0
    for zpath in sorted(inbox.glob("maps-study-*.zip")):
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
        with ledger.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        records.append(rec)
        seen[sha] = zpath.name
        if rec["qc"] == "fail":
            bad += 1
            print(f"QC FAILED {rec['run_id']} ({zpath.name}), not added to study.json: " + "; ".join(rec["failed"]))
        elif args.study_json:
            add_to_study(args.study_json.expanduser(), kit, rec)
    last = {r["run_id"]: r for r in records}
    print(f"\n{'run id':<14} {'label':<6} {'conditions':<11} {'solo':>8} {'team':>8}  qc")
    for run, r in sorted(last.items()):
        c = r["conditions"]
        cell = {arm: "{}/{}".format(*c[arm]) if arm in c else "-" for arm in ("solo", "team")}
        print(f"{run:<14} {r['label']:<6} {','.join(c) or '-':<11} {cell['solo']:>8} {cell['team']:>8}  {r['qc']}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
