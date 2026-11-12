"""One metrics row per run root: the table a set of runs (an effort ladder, several models) is read with.

It defines no metric: ``rows.load_attempts`` picks each episode's result
and :func:`run_row` averages it. The criterion must be identical across
every root that records one (a disc's radius included) and is printed once.
A run's label is the root's basename; its effort is the manifest's
``SEARCH_SAMPLING_REASONING_EFFORT``, as written.

Usage:
    python -m benchmarks.citynav.analysis.run_table <root> [<root> ...]
    python -m benchmarks.citynav.analysis.run_table --json <root> [...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from benchmarks.citynav.analysis.run_report import criterion_label, run_criterion
from benchmarks.run_record import rows as _rows

#: (json key, header, format). An absent metric prints ``--``, never 0.
COLUMNS = (
    ("label", "run", "{:<10}"),
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


_METRIC_KEYS = (
    "SR",
    "SR_pct",
    "OSR",
    "OSR_pct",
    "SR_by_shape",
    "SR_by_shape_pct",
    "SR_by_radius",
    "SR_by_radius_pct",
    "precise_shape_n",
    "precise_radius_n",
    "SR_n",
    "NE_mean",
    "NE_med",
)


def _mean(vals: list) -> float | None:
    return round(float(np.mean(vals)), 4) if vals else None


def run_row(
    root: Path, *, allow_partial: bool = False, attempts: tuple[dict, _rows.ReadStats] | None = None
) -> dict:
    """One run's summary; a run with no scored episode still gets its row.
    ``attempts`` is the root's ``rows.load_attempts`` result when the caller
    already read it.

    Each mean is a macro mean over the rows carrying the key (OSR is absent
    from a shape-scored row); a shape-scored SR, which pools footprint
    containment and the disc fallback, is also split by ``precise_target``;
    rates are also given as percentages, as CityNav reports them."""
    picked, stats = attempts or _rows.load_attempts(root, allow_partial=allow_partial)
    scored = list(_rows.scored_metrics(picked).values())
    recorded = [a.metrics for a in picked.values() if a.metrics]
    manifest = _rows.read_manifest(root)
    row: dict = {
        "root": str(root),
        "label": root.name,
        "effort": (manifest.get("search_env_flags") or {}).get("SEARCH_SAMPLING_REASONING_EFFORT"),
        "criterion": criterion_label(run_criterion(recorded), root) if recorded else None,
        "n": len(scored),
        "no_metrics": stats.no_metrics,
        "timed_out": stats.timed_out,
        "infra": stats.infra,
        # Every row carries every metric key, absent ones as null.
        **{k: None for k in _METRIC_KEYS},
    }
    if not scored:
        return row
    row["SR_n"] = sum(1 for m in scored if m.get("SR"))
    ne = [m["NE"] for m in scored if m.get("NE") is not None]
    row["NE_med"] = round(statistics.median(ne), 2) if ne else None
    row["NE_mean"] = _mean(ne)
    for k in ("SR", "OSR"):
        row[k] = _mean([m[k] for m in scored if m.get(k) is not None])
    for kind in ("shape", "radius"):
        vals = [
            m["SR"] for m in scored if m.get("precise_target") == kind and m.get("SR") is not None
        ]
        row[f"precise_{kind}_n"] = len(vals)
        row[f"SR_by_{kind}"] = _mean(vals)
    for k in ("SR", "OSR", "SR_by_shape", "SR_by_radius"):
        if row[k] is not None:
            row[k + "_pct"] = round(row[k] * 100.0, 2)
    return row


def _sort_key(row: dict) -> tuple[int, float, str]:
    """Numeric effort first (an effort ladder), then the rest by label."""
    effort = row.get("effort")
    try:
        return (0, float(effort), row["label"])
    except (TypeError, ValueError):
        return (1, 0.0, row["label"])


def table(roots: list[Path], *, allow_partial: bool = False) -> list[dict]:
    """Every run's row, criterion-checked across the whole set."""
    rows = [run_row(r, allow_partial=allow_partial) for r in roots]
    seen = {r["criterion"] for r in rows if r["criterion"] is not None}
    if len(seen) > 1:
        raise SystemExit(
            f"run_table: these roots mix success criteria {sorted(seen)}. "
            "Runs scored on different masks (a shape and a disc, or discs of "
            "two radii) answer different questions -- table them separately."
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
    print(f"criterion: {criterion or 'UNRECORDED'}   runs: {len(rows)}")
    if criterion == criterion_label("shape", Path()):
        print("OSR is ring-basis and absent from every shape-scored row.")
    columns = [(k, h, f"{{:<{label_w}}}" if k == "label" else fmt) for k, h, fmt in COLUMNS]
    print("  ".join(fmt.format(head) for _, head, fmt in columns))
    for row in rows:
        print("  ".join(fmt.format(_cell(row, key)) for key, _, fmt in columns))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path, help="the run roots to table")
    ap.add_argument("--json", action="store_true", help="emit the rows as JSON instead of a table")
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
