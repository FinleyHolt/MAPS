"""The capability x effort figure's data, re-derived from run roots every call.

One record per run root, from the readers: SR from ``arm_table.arm_row``,
the interval from ``ab_report.wilson_ci``, the bake line from
``run_report.bake_line``, and cost off the same picked attempts. An arm is
identified by its run manifest (``served_model``, ``split``, the effort its
requests carried); a root naming no ``served_model`` is listed as unmatched.
Every record carries its criterion, and a figure draws one split at a time.

Usage:
    python -m figures.capability_effort_arms <root> [<root> ...]
    python -m figures.capability_effort_arms --out arms.json <root>...
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from benchmarks.citynav.analysis import rows as _rows
from benchmarks.citynav.analysis.ab_report import wilson_ci
from benchmarks.citynav.analysis.arm_table import arm_row
from benchmarks.citynav.analysis.run_report import bake_line


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 1) if values else None


def _cost(root: Path, *, allow_partial: bool) -> dict:
    """Price the attempts arm_row scored that kept a transcript; ``priced_n`` counts them."""
    picked, _ = _rows.load_attempts(root, allow_partial=allow_partial)
    keys = ("completion_tokens", "total_tokens", "reasoning_tokens", "turns")
    seen: dict[str, list[float]] = {k: [] for k in keys}
    wall: list[float] = []
    for att in picked.values():
        if not att.metrics:
            continue
        if isinstance(att.row.get("elapsed_s"), (int, float)):
            wall.append(float(att.row["elapsed_s"]))
        llm = att.metrics.get("llm") or {}
        if not llm.get("requests"):
            continue  # no transcript: flight_summary.blank() zeros, not a price
        for key in keys:
            if isinstance(llm.get(key), (int, float)):
                seen[key].append(float(llm[key]))
    return {
        "priced_n": len(seen["turns"]),
        "wall_med_s": _median(wall),
        "completion_tok_mean": _mean(seen["completion_tokens"]),
        "total_tok_mean": _mean(seen["total_tokens"]),
        "completion_tok_med": _median(seen["completion_tokens"]),
        "reasoning_tok_med": _median(seen["reasoning_tokens"]),
        "turns_med": _median(seen["turns"]),
    }


def effort_value(raw: object) -> object:
    """A numeric effort as a float, a named rung as written, unset as None."""
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return str(raw)


def read_root(root: Path, *, allow_partial: bool = False) -> dict | None:
    """One arm's record, or None when its run manifest names no served_model."""
    man = _rows.read_manifest(root)
    model = man.get("served_model")
    if not model:
        return None
    row = arm_row(root, allow_partial=allow_partial)
    n, k = row["n"], row["SR_n"]
    rec = {
        "model": str(model).rstrip("/").rsplit("/", 1)[-1],
        "split": man.get("split"),
        "effort": effort_value(row["effort"]),
        "withhold": list(man.get("withhold") or []),
        "git_commit": man.get("git_commit"),
        "n": n, "sr": row["SR_pct"], "sr_n": k, "osr": row["OSR_pct"],
        "ne_med": row["NE_med"], "criterion": row["criterion"],
        "bake": bake_line(root),
        "timed_out": row["timed_out"], "infra": row["infra"],
        "root": str(root),
        "sr_lo": None, "sr_hi": None,
        **_cost(root, allow_partial=allow_partial),
    }
    if n and isinstance(k, int):
        lo, hi = wilson_ci(k, n)
        rec["sr_lo"], rec["sr_hi"] = round(100 * lo, 2), round(100 * hi, 2)
    return rec


def build(roots: list[Path], *, allow_partial: bool = False) -> dict:
    records, unmatched = [], []
    for root in roots:
        rec = read_root(root, allow_partial=allow_partial)
        if rec is None:
            unmatched.append(str(root))
        else:
            records.append(rec)
    records.sort(key=lambda r: (str(r["split"]), r["model"], str(r["effort"])))
    return {
        "schema": "maps.capability_effort/v1",
        "arms": len(records),
        "episodes": sum(r["n"] for r in records),
        "unmatched_roots": unmatched,
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path, help="run roots, one per arm")
    ap.add_argument("--out", type=Path, help="write JSON here instead of stdout")
    _rows.add_allow_partial(ap)
    args = ap.parse_args(argv)

    data = build([r for r in args.roots if r.is_dir()], allow_partial=args.allow_partial)
    text = json.dumps(data, indent=2)
    if args.out:
        args.out.write_text(text)
        print(f"{data['arms']} arms, {data['episodes']:,} episodes -> {args.out}")
    else:
        print(text)
    if data["unmatched_roots"]:
        print(f"note: {len(data['unmatched_roots'])} root(s) carry no served_model in their run manifest",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
