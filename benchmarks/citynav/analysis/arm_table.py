"""One metrics row per run root — the N-arm view a ladder is read with.

``run_report`` speaks for one root or a pair; ``episode_ledger table``
renders per-episode X/. cells. Neither prints SR and NE side by side for a
dozen arms, which is what a sweep over one knob produces.

This defines NO metric. Every number comes from the canonical readers:
``rows.load_attempts`` decides which attempt IS an episode's result (the
first fully complete attempt by start time — the rule lives there and
nowhere else), and ``eval.aggregate`` macro-averages. Its own work is
labelling, sorting and the criterion guard.

The criterion is asserted IDENTICAL across every root and printed once. A
ring row and a shape row share no success key, so a table pooling them would
be differencing two questions under one column heading.

Usage:
    python -m benchmarks.citynav.analysis.arm_table <root> [<root> ...]
    python -m benchmarks.citynav.analysis.arm_table --json <root> [...]

An arm's label is its ``arm.json`` ``label`` (the ladder writes one), else
the root's basename.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from benchmarks.citynav import eval as _eval
from benchmarks.citynav.analysis import rows as _rows
from benchmarks.citynav.analysis.run_report import run_criterion

#: (json key, header, format). ``None`` values print as ``--`` rather than 0:
#: a metric this bench does not resolve is absent, not zero.
COLUMNS = (
    ("label", "arm", "{:<10}"),
    ("effort", "effort", "{:>6}"),
    ("n", "n", "{:>5}"),
    ("SR_pct", "SR%", "{:>6}"),
    ("SR_n", "SR", "{:>9}"),
    ("SR_by_shape_pct", "shape%", "{:>7}"),
    ("SR_by_radius_pct", "radius%", "{:>8}"),
    ("precise_shape_n", "shape_n", "{:>8}"),
    ("precise_floored_n", "floored", "{:>8}"),
    ("NE_mean", "NE_mean", "{:>8}"),
    ("NE_med", "NE_med", "{:>7}"),
    ("no_metrics", "nometr", "{:>7}"),
    ("timed_out", "t/out", "{:>6}"),
    ("infra", "infra", "{:>6}"),
)


#: Copied straight out of ``eval.aggregate``'s output, never recomputed.
_AGG_KEYS = (
    "SR",
    "SR_pct",
    "SR_by_shape",
    "SR_by_shape_pct",
    "SR_by_radius",
    "SR_by_radius_pct",
    "precise_shape_n",
    "precise_radius_n",
    "precise_floored_n",
)
_METRIC_KEYS = (*_AGG_KEYS, "SR_n", "NE_mean", "NE_med")


def arm_row(root: Path) -> dict:
    """One arm's summary, or a row saying plainly that it has no result.

    A root with zero scored episodes is REPORTED, not skipped and not
    raised on: in a ladder, an arm whose serve died is the finding, and
    dropping its row would leave a table that looks complete.
    """
    picked, stats = _rows.load_attempts(root)
    scored = [a.metrics for a in picked.values() if a.metrics]
    meta = _arm_meta(root)
    row: dict = {
        "root": str(root),
        "label": meta.get("label") or root.name,
        "effort": meta.get("effort"),
        "criterion": run_criterion(scored) if scored else None,
        "n": len(scored),
        "no_metrics": stats.no_metrics,
        "timed_out": stats.timed_out,
        "infra": stats.infra,
        "tier": stats.tier,
        # Every row carries every metric key, absent ones as null. A dead
        # arm's row and a live one's have the same shape, so a consumer
        # never has to tell a missing key from a real zero.
        **{k: None for k in _METRIC_KEYS},
    }
    if not scored:
        return row
    agg = _eval.aggregate(scored)
    row["SR_n"] = sum(1 for m in scored if m.get("SR"))
    ne = [m["NE"] for m in scored if m.get("NE") is not None]
    row["NE_med"] = round(statistics.median(ne), 2) if ne else None
    row["NE_mean"] = agg.get("NE")
    row.update({k: agg.get(k) for k in _AGG_KEYS})
    return row


def _arm_meta(root: Path) -> dict:
    """The ladder's own receipt for this arm, if it wrote one."""
    try:
        meta = json.loads((root / "arm.json").read_text())
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _sort_key(row: dict) -> tuple[int, float, str]:
    """Numeric label first (an effort ladder), then the rest by name."""
    effort = row.get("effort")
    try:
        return (0, float(effort), row["label"])
    except (TypeError, ValueError):
        return (1, 0.0, row["label"])


def table(roots: list[Path]) -> list[dict]:
    """Every arm's row, criterion-checked across the whole set."""
    rows = [arm_row(r) for r in roots]
    seen = {r["criterion"] for r in rows if r["criterion"] is not None}
    if len(seen) > 1:
        raise SystemExit(
            f"arm_table: these roots mix success criteria {sorted(seen)}. "
            "A ring arm and a shape arm answer different questions and share "
            "no success key -- table them separately."
        )
    return sorted(rows, key=_sort_key)


def _cell(row: dict, key: str) -> str:
    value = row.get(key)
    if value is None:
        return "--"
    if key == "SR_n":
        return f"{value}/{row['n']}"
    return str(value)


def render(rows: list[dict]) -> None:
    seen = {r["criterion"] for r in rows if r["criterion"] is not None}
    criterion = seen.pop() if seen else None
    print(f"criterion: {criterion or 'UNRECORDED (pre-cutover runs)'}   arms: {len(rows)}")
    if criterion == "shape":
        # Said once, here, so nobody reads the missing column as a zero.
        print("OSR is ring-basis and absent from every shape-scored row.")
    print("  ".join(fmt.format(head) for _, head, fmt in COLUMNS))
    for row in rows:
        print("  ".join(fmt.format(_cell(row, key)) for key, _, fmt in COLUMNS))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path, help="run roots, one per arm")
    ap.add_argument(
        "--json", action="store_true", help="emit the rows as JSON instead of a table"
    )
    args = ap.parse_args(argv)
    rows = table(args.roots)
    if args.json:
        json.dump(rows, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        render(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
