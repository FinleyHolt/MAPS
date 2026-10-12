"""One metrics row per run root — the N-arm view a ladder is read with.

It defines no metric: ``rows.load_attempts`` picks each episode's result
and ``eval.aggregate`` averages. The criterion must be identical across
every root and is printed once. An arm's label is the withholds its run
manifest records, else the root's basename; its effort is the manifest's
``SEARCH_SAMPLING_REASONING_EFFORT``, as written.

Usage:
    python -m benchmarks.citynav.analysis.arm_table <root> [<root> ...]
    python -m benchmarks.citynav.analysis.arm_table --json <root> [...]
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

#: (json key, header, format). An absent metric prints ``--``, never 0.
COLUMNS = (
    ("label", "arm", "{:<10}"),
    ("effort", "effort", "{:>6}"),
    ("n", "n", "{:>5}"),
    ("SR_pct", "SR%", "{:>6}"),
    ("SR_n", "SR", "{:>9}"),
    ("OSR_pct", "OSR%", "{:>6}"),
    ("SR_by_shape_pct", "shape%", "{:>7}"),
    ("SR_by_radius_pct", "radius%", "{:>8}"),
    ("precise_shape_n", "shape_n", "{:>8}"),
    ("NE_mean", "NE_mean", "{:>8}"),
    ("NE_med", "NE_med", "{:>7}"),
    ("no_metrics", "nometr", "{:>7}"),
    ("timed_out", "t/out", "{:>6}"),
    ("infra", "infra", "{:>6}"),
)


#: Copied straight out of ``eval.aggregate``'s output, never recomputed.
_AGG_KEYS = ("SR", "SR_pct", "OSR", "OSR_pct", "SR_by_shape", "SR_by_shape_pct", "SR_by_radius",
             "SR_by_radius_pct", "precise_shape_n", "precise_radius_n")
_METRIC_KEYS = (*_AGG_KEYS, "SR_n", "NE_mean", "NE_med")


def arm_row(root: Path, *, allow_partial: bool = False) -> dict:
    """One arm's summary; an arm with no scored episode still gets its row."""
    picked, stats = _rows.load_attempts(root, allow_partial=allow_partial)
    scored = list(_rows.scored_metrics(picked).values())
    manifest = _rows.read_manifest(root)
    withhold = list(manifest.get("withhold") or [])
    row: dict = {
        "root": str(root),
        "label": f"withhold={','.join(withhold)}" if withhold else root.name,
        "effort": (manifest.get("search_env_flags") or {}).get(
            "SEARCH_SAMPLING_REASONING_EFFORT"),
        "criterion": run_criterion(scored) if scored else None,
        "n": len(scored),
        "no_metrics": stats.no_metrics,
        "timed_out": stats.timed_out,
        "infra": stats.infra,
        # Every row carries every metric key, absent ones as null.
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


def _sort_key(row: dict) -> tuple[int, float, str]:
    """Numeric effort first (an effort ladder), then the rest by label."""
    effort = row.get("effort")
    try:
        return (0, float(effort), row["label"])
    except (TypeError, ValueError):
        return (1, 0.0, row["label"])


def table(roots: list[Path], *, allow_partial: bool = False) -> list[dict]:
    """Every arm's row, criterion-checked across the whole set."""
    rows = [arm_row(r, allow_partial=allow_partial) for r in roots]
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
    label_w = max([10] + [len(str(r["label"])) for r in rows])
    seen = {r["criterion"] for r in rows if r["criterion"] is not None}
    criterion = seen.pop() if seen else None
    print(f"criterion: {criterion or 'UNRECORDED'}   arms: {len(rows)}")
    if criterion == "shape":
        print("OSR is ring-basis and absent from every shape-scored row.")
    columns = [(k, h, f"{{:<{label_w}}}" if k == "label" else fmt) for k, h, fmt in COLUMNS]
    print("  ".join(fmt.format(head) for _, head, fmt in columns))
    for row in rows:
        print("  ".join(fmt.format(_cell(row, key)) for key, _, fmt in columns))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path, help="run roots, one per arm")
    ap.add_argument(
        "--json", action="store_true", help="emit the rows as JSON instead of a table"
    )
    _rows.add_allow_partial(ap)
    args = ap.parse_args(argv)
    rows = table(args.roots, allow_partial=args.allow_partial)
    if args.json:
        json.dump(rows, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        render(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
