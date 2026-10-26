"""Score one participant's capacity sessions off the study download and check them (``STUDY.md``).

Each condition in ``<kit>/results/<p>/`` is scored by ``team_rows`` into ``<kit>/scored/<p>-<arm>``
(a bench root the paper's capacity figure reads), then summarised: answers and correct finds by
source, success, correct finds per session hour, how the person met MAPS's answers, and the
ledger's own checks (a duplicate answer, an answer after the session's end or before the one ahead
of it, an unreadable line, a missing clock).

    python -m baselines.human.study_qc p01 [--kit ~/maps-study] [--misses]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from baselines.human.study_server import KIT, use_kit

SOURCES = ("own_search", "confirm", "replace", "skip", "unchecked")


def checks(results: Path) -> list[str]:
    """What is wrong with one condition's ledger, in words; empty when nothing is."""
    from baselines.human.audit_console import _jsonl
    from baselines.human.capacity import _utc

    meta, ledger = results / "session.json", results / "answers.jsonl"
    if not meta.is_file():
        return ["no session.json: the session's clock is lost"]
    session = json.loads(meta.read_text())
    answers = _jsonl(ledger)
    out = []
    lines = sum(1 for ln in ledger.read_text().splitlines() if ln.strip()) if ledger.is_file() else 0
    if lines > len(answers):
        out.append(f"{lines - len(answers)} unreadable lines in answers.jsonl")
    dup = [e for e, n in Counter(a["episode_id"] for a in answers).items() if n > 1]
    if dup:
        out.append(f"{len(dup)} reports answered twice, e.g. {dup[0]}")
    # UTC stamps to the second compare as text.
    end = _utc(float(session["started_epoch"]) + float(session["session_s"]) + 1)
    late = [a for a in answers if a["answered_at"] > end]
    if late:
        out.append(f"{len(late)} answers after the session's end")
    stamps = [session["started_at"]] + [a["answered_at"] for a in answers]
    back = sum(b < a for a, b in zip(stamps, stamps[1:]))
    if back:
        out.append(f"{back} answers before the session's start or before the answer ahead of them: the clock moved back")
    off = [a for a in answers if a["episode_id"] not in session["order"]]
    if off:
        out.append(f"{len(off)} answers to reports not on the list")
    return out


def summary(scored: Path) -> dict:
    """One scored condition's numbers."""
    from baselines.human.audit_console import _jsonl

    rows = _jsonl(scored / "episodes.jsonl")
    hours = float(json.loads((scored / "session.json").read_text())["session_s"]) / 3600
    m = [r["metrics"] for r in rows]
    by = {s: [x for x in m if x.get("capacity_source") == s] for s in SOURCES}
    out = {"n": len(m), "correct": sum(int(x["SR"]) for x in m), "hours": hours,
           "sources": {s: (len(v), sum(int(x["SR"]) for x in v)) for s, v in by.items() if v},
           "median_answer_s": {s: statistics.median(x["capacity_answer_s"] for x in v)
                               for s, v in by.items() if v and v[0].get("capacity_answer_s") is not None}}
    verified = by["confirm"] + by["replace"]
    if verified:
        # MAPS right or wrong, against what the person did with it.
        out["maps_wrong_confirmed"] = sum(not x["maps_SR"] for x in by["confirm"])
        out["maps_right_replaced_wrong"] = sum(x["maps_SR"] and not x["SR"] for x in by["replace"])
        out["maps_wrong_fixed"] = sum(not x["maps_SR"] and x["SR"] for x in by["replace"])
    out["misses"] = [(x.get("capacity_source"), r["episode_id"]) for r, x in zip(rows, m) if not x["SR"]
                     and x.get("capacity_source") != "unchecked"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("participant")
    ap.add_argument("--kit", type=Path, default=KIT)
    ap.add_argument("--misses", action="store_true", help="list every report the person got wrong")
    args = ap.parse_args(argv)
    kit = args.kit.expanduser()
    label = args.participant.strip().lower()
    if label not in json.loads((kit / "plan" / "plan.json").read_text())["participants"]:
        raise SystemExit(f"{label} is not a participant on this kit's plan")
    use_kit(kit)
    from baselines.human import team_rows

    bad = 0
    for arm in ("solo", "team"):
        results = kit / "results" / label / arm
        if not results.is_dir():
            print(f"{arm}: no session yet")
            continue
        problems = checks(results)
        for p in problems:
            print(f"{arm}: CHECK FAILED: {p}")
        bad += len(problems)
        if not (results / "session.json").is_file():
            continue
        scored = kit / "scored" / f"{label}-{arm}"
        team_rows.main(["--maps-root", str(kit / "maps_root"), "--decisions", str(results / "answers.jsonl"),
                        "--out-root", str(scored)])
        s = summary(scored)
        print(f"\n{arm}: {s['correct']}/{s['n']} correct ({100 * s['correct'] / max(1, s['n']):.1f}%), "
              f"{s['correct'] / s['hours']:.1f} correct per hour over {60 * s['hours']:.0f} min")
        for src, (n, ok) in s["sources"].items():
            med = s["median_answer_s"].get(src)
            print(f"  {src:<11} {n:>4} answers, {ok:>4} correct" + (f", median {med:.0f} s" if med is not None else ""))
        if "maps_wrong_confirmed" in s:
            print(f"  MAPS wrong and confirmed: {s['maps_wrong_confirmed']}; MAPS wrong and fixed: {s['maps_wrong_fixed']}; "
                  f"MAPS right and replaced with a miss: {s['maps_right_replaced_wrong']}")
        if args.misses:
            for src, e in s["misses"]:
                print(f"  miss  {src:<11} {e}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
