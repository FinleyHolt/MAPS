"""The capacity reader: correct finds per hour against offered load, the
saturation load, and alone against assisted on the same streams.

A session is one run root holding ``arrivals.json`` (the stream it flew,
``benchmarks.capacity.streams``) and the bench ledger read through
``benchmarks.citynav.analysis.rows.load_attempts``. Each report's row carries
``arrival_s``, ``claimed_s`` and ``elapsed_s`` (session seconds) and the
bench's ``SR`` metric; the alone arm's front end writes the same shape. A
report is found at ``claimed_s + elapsed_s`` and counts toward the session's
finds per hour when its SR is 1 and it lands inside the session's ``hours``::

    python -m benchmarks.capacity.reader --assisted <root>... [--alone <root>...] [--json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from benchmarks.citynav.analysis.ab_report import ab_stats, bootstrap_ci, wilson_ci
from benchmarks.citynav.analysis.rows import load_attempts

#: A load saturates the arm when the share of its due reports (those that
#: arrived early enough to land inside the window at the lightest load's median
#: time to find) found correct inside the window falls below this share of the
#: arm's accuracy at its lightest load.
SATURATED_SHARE = 0.9


def session(root: Path) -> dict[str, Any]:
    """One session's capacity row, with its per-report ``reports``. A session
    cannot resume, so it reads as it stands: a report whose every attempt was
    an infrastructure casualty is offered and never found, counted under
    ``infra``."""
    stream = json.loads((root / "arrivals.json").read_text())
    picked, stats = load_attempts(root, allow_partial=True)
    window = float(stream["hours"]) * 3600.0
    reports = []
    for r in stream["reports"]:
        a = picked.get(r["episode_id"])
        row = a.row if a else {}
        found = (float(row["claimed_s"]) + float(row["elapsed_s"])
                 if row.get("claimed_s") is not None and row.get("elapsed_s") is not None else None)
        correct = bool(a and a.metrics.get("SR"))
        reports.append({"report": r["report"], "episode_id": r["episode_id"], "arrival_s": r["arrival_s"],
                        "found_s": found, "correct": correct,
                        "in_window": found is not None and found <= window})
    answered = [r for r in reports if r["found_s"] is not None]
    k = sum(r["correct"] for r in answered)
    lo, hi = wilson_ci(k, len(answered))
    finds = [r for r in reports if r["in_window"] and r["correct"]]
    right = [r for r in answered if r["correct"]]
    return {
        "root": str(root), "load_per_h": float(stream["load_per_h"]), "seed": int(stream["seed"]),
        "hours": float(stream["hours"]), "offered": len(reports),
        "offered_per_h": round(len(reports) / float(stream["hours"]), 2), "answered": len(answered),
        "finds_per_h": round(len(finds) / float(stream["hours"]), 2),
        "accuracy": round(k / len(answered), 3) if answered else None,
        "accuracy_ci95": [round(lo, 3), round(hi, 3)] if answered else None,
        "time_to_find_median_s": (round(statistics.median(r["found_s"] - r["arrival_s"] for r in right), 1)
                                  if right else None),
        "backlog": sum(1 for r in reports if not r["in_window"]),
        "infra": sum(1 for r in reports if r["episode_id"] in stats.infra_episodes - picked.keys()),
        "reports": reports,
    }


def curve(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """One arm's finds per hour by offered load (the mean over seeds), its
    capacity (the most finds per hour any load drew) and its saturation load
    (the lightest load whose due reports, the mean over seeds, are found
    correct inside the window short of ``SATURATED_SHARE`` of the arm's
    lightest-load accuracy); None when no load saturates it. A report too late
    to land inside the window at the lightest load's median time to find is
    not due, so the window's edge saturates no load.

    ponytail: a threshold knee, not a fitted ``min(load * p, mu)``; fit it if
    the paper needs an interval on the capacity."""
    loads = sorted({s["load_per_h"] for s in sessions})
    light = [s for s in sessions if s["load_per_h"] == loads[0]]
    base = [s["accuracy"] for s in light if s["accuracy"] is not None]
    p = statistics.fmean(base) if base else None
    lags = [r["found_s"] - r["arrival_s"] for s in light for r in s["reports"] if r["found_s"] is not None]
    ttf = statistics.median(lags) if lags else None
    points = []
    for load in loads:
        at = [s for s in sessions if s["load_per_h"] == load]
        yields = []
        for s in at:
            due = [r for r in s["reports"] if ttf is not None and r["arrival_s"] <= s["hours"] * 3600.0 - ttf]
            if due:
                yields.append(sum(r["in_window"] and r["correct"] for r in due) / len(due))
        points.append({"load_per_h": load, "sessions": len(at),
                       "offered_per_h": round(statistics.fmean(s["offered_per_h"] for s in at), 2),
                       "finds_per_h": round(statistics.fmean(s["finds_per_h"] for s in at), 2),
                       "due_yield": round(statistics.fmean(yields), 3) if yields else None})
    saturation = next((pt["load_per_h"] for pt in points if p is not None and pt["due_yield"] is not None
                       and pt["due_yield"] < SATURATED_SHARE * p), None)
    return {"points": points, "accuracy_light": None if p is None else round(p, 3),
            "capacity_finds_per_h": max(pt["finds_per_h"] for pt in points), "saturation_load_per_h": saturation}


def paired(alone: list[dict[str, Any]], assisted: list[dict[str, Any]]) -> dict[str, Any]:
    """Alone against assisted over the sessions that flew the same stream,
    paired by ``(load_per_h, seed, hours)``, one session per arm per stream:
    the per-session difference in finds per hour (assisted - alone) with its
    bootstrap interval, and the per-report in-window correct finds through
    ``ab_stats`` (a = alone)."""
    def key(s):
        return (s["load_per_h"], s["seed"], s["hours"])

    def by_key(arm):
        out: dict = {}
        for s in arm:
            if key(s) in out:
                raise SystemExit(f"{out[key(s)]['root']} and {s['root']} both fly stream {key(s)} in one arm")
            out[key(s)] = s
        return out

    alone_by = by_key(alone)
    by_key(assisted)
    pairs = []
    for b in assisted:
        a = alone_by.get(key(b))
        if a is None:
            continue
        if [r["episode_id"] for r in a["reports"]] != [r["episode_id"] for r in b["reports"]]:
            raise SystemExit(f"{a['root']} and {b['root']} claim stream {key(b)} but flew different reports")
        pairs.append((a, b))
    diffs = [b["finds_per_h"] - a["finds_per_h"] for a, b in pairs]
    lo, hi = bootstrap_ci(diffs)
    found = [(int(ra["in_window"] and ra["correct"]), int(rb["in_window"] and rb["correct"]))
             for a, b in pairs for ra, rb in zip(a["reports"], b["reports"])]
    return {"sessions": len(pairs),
            "finds_per_h_delta": round(statistics.fmean(diffs), 2) if diffs else None,
            "finds_per_h_delta_ci95": [round(lo, 2), round(hi, 2)] if diffs else None,
            "reports": ab_stats(found, label="alone -> assisted") if found else None}


def _table(arm: str, sessions: list[dict[str, Any]], c: dict[str, Any]) -> str:
    lines = [f"{arm}: capacity {c['capacity_finds_per_h']} finds/h, saturates at "
             f"{c['saturation_load_per_h'] or 'no load flown'}"
             f"{'/h' if c['saturation_load_per_h'] else ''} (light-load accuracy {c['accuracy_light']})",
             f"  {'load/h':>7} {'seed':>5} {'offered':>7} {'finds/h':>8} {'accuracy':>9} {'ttf_med_s':>10} "
             f"{'backlog':>7} {'infra':>5}"]
    for s in sorted(sessions, key=lambda s: (s["load_per_h"], s["seed"])):
        lines.append(f"  {s['load_per_h']:>7g} {s['seed']:>5} {s['offered']:>7} {s['finds_per_h']:>8} "
                     f"{s['accuracy'] if s['accuracy'] is not None else '-':>9} "
                     f"{s['time_to_find_median_s'] if s['time_to_find_median_s'] is not None else '-':>10} "
                     f"{s['backlog']:>7} {s['infra']:>5}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--assisted", type=Path, nargs="+", required=True, help="the assisted arm's session roots")
    ap.add_argument("--alone", type=Path, nargs="*", default=[], help="the alone arm's session roots")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)
    arms = {name: [session(r) for r in roots]
            for name, roots in (("assisted", args.assisted), ("alone", args.alone)) if roots}
    out: dict[str, Any] = {name: {"curve": curve(s), "sessions": [{k: v for k, v in x.items() if k != "reports"}
                                                                   for x in s]}
                           for name, s in arms.items()}
    if "alone" in arms:
        out["paired"] = paired(arms["alone"], arms["assisted"])
    if args.json:
        print(json.dumps(out, indent=1, allow_nan=False))
        return 0
    for name, s in arms.items():
        print(_table(name, s, out[name]["curve"]))
    if "paired" in out:
        pr = out["paired"]
        st = pr["reports"]
        print(f"paired over {pr['sessions']} sessions" + (
            f": assisted - alone = {pr['finds_per_h_delta']} finds/h 95%CI{pr['finds_per_h_delta_ci95']}; "
            f"per report {st['SR_a']:.3f} -> {st['SR_b']:.3f} (McNemar p={st['mcnemar']['p_value']}, n={st['n']})"
            if st else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
