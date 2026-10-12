#!/usr/bin/env python3
"""Paired A/B SR report: bootstrap CI on the delta, McNemar, and the flip counts.

A change can trade successes for failures at a flat net SR, so the
success->fail count is reported beside the delta. ``ab_stats`` takes
(sr_a, sr_b) pairs over the shared episodes; the CLI pairs two run roots by
episode_id and refuses two roots scored under different success criteria.

Usage:
  python -m benchmarks.citynav.analysis.ab_report <run_a> <run_b> [--label "..."]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

from benchmarks.citynav.analysis import rows as _rows


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    """The 95% Wilson score interval for k/n."""
    z = 1.96
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _bootstrap_ci(diffs: list[float]) -> tuple[float, float]:
    """Seeded 95% percentile bootstrap CI (10,000 draws) for the mean of per-episode (b - a)."""
    import random

    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"))
    iters = 10000
    rng = random.Random(12345)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def _mcnemar(pairs: list[tuple[int, int]]) -> dict:
    """McNemar over discordant pairs: b losses, c wins; exact two-sided
    binomial p up to 25 discordant, else continuity-corrected chi2(1)."""
    b = sum(1 for a, bb in pairs if a == 1 and bb == 0)  # success->fail
    c = sum(1 for a, bb in pairs if a == 0 and bb == 1)  # fail->success
    n = b + c
    if n == 0:
        return {"b_loss": 0, "c_win": 0, "p_value": 1.0, "method": "none"}
    if n <= 25:  # exact two-sided binomial (p=0.5)
        k = min(b, c)
        # P(X<=k) + P(X>=n-k), X~Binom(n,0.5); symmetric -> 2*sum_{i<=k} C(n,i)/2^n
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
        return {"b_loss": b, "c_win": c, "p_value": round(p, 4), "method": "binomial_exact"}
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2))
    return {"b_loss": b, "c_win": c, "p_value": round(p, 4),
            "chi2": round(chi2, 3), "method": "chi2_1df"}


def ab_stats(pairs: list[tuple[int, int]], *, label: str = "") -> dict:
    """Paired SR statistics over (a, b) in {0,1}. The shared-set core."""
    n = len(pairs)
    sr_a = sum(a for a, _ in pairs) / n if n else float("nan")
    sr_b = sum(b for _, b in pairs) / n if n else float("nan")
    diffs = [float(b - a) for a, b in pairs]
    lo, hi = _bootstrap_ci(diffs)
    mc = _mcnemar(pairs)
    return {
        "label": label,
        "n": n,
        "SR_a": round(sr_a, 4),
        "SR_b": round(sr_b, 4),
        "delta": round(sr_b - sr_a, 4),
        "delta_ci95": [round(lo, 4), round(hi, 4)],
        "wins_fail_to_success": mc["c_win"],
        "losses_success_to_fail": mc["b_loss"],
        "mcnemar": mc,
        "significant_95": bool(mc["p_value"] < 0.05),
    }


def render(st: dict) -> str:
    ci = st["delta_ci95"]
    sig = "SIGNIFICANT" if st["significant_95"] else "not significant"
    return (
        f"A/B paired ({st['label']}): n={st['n']}  "
        f"SR_a={st['SR_a']:.3f}  SR_b={st['SR_b']:.3f}  "
        f"Δ={st['delta']:+.3f}  95%CI[{ci[0]:+.3f},{ci[1]:+.3f}]\n"
        f"  wins(fail->success)={st['wins_fail_to_success']}  "
        f"losses(success->fail)={st['losses_success_to_fail']}  "
        f"McNemar p={st['mcnemar']['p_value']} ({st['mcnemar']['method']}) -> {sig}"
    )


def _run_provenance(run_root: Path) -> dict:
    """The first row's provenance block, else {}, with what the manifest adds:
    the tile-bake content hash (every Cal City bake sits at
    ``benchmarks/calcity/tiles``, so the path the rows carry cannot tell two
    apart), the arm's withholds, and the sha256 of the resolved config
    without them (``config_body``: an E4 arm differs from E1 only there)."""
    prov = next((r["provenance"] for r in _rows.iter_json_rows(run_root / _rows.EPISODES_JSONL)
                 if r.get("provenance")), None)
    if not prov:
        return {}
    manifest = _rows.read_manifest(run_root)
    prov = dict(prov, withhold=manifest.get("withhold"))
    bake = (manifest.get("perception") or {}).get("bake_sha256")
    if bake:
        prov["bake_sha256"] = bake
    if manifest.get("config_yaml"):
        body = {k: v for k, v in yaml.safe_load(manifest["config_yaml"]).items() if k != "withhold"}
        prov["config_body"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return prov


def _ground_truth_keys(prov: dict) -> dict:
    """The facts that decide what was measured: split, bake directory name
    (one bake mounts under many roots) and bake content."""
    flags = prov.get("search_env_flags") or {}
    tiles = flags.get("SEARCH_CITYNAV_TILES")
    return {
        "split": prov.get("split"),
        "tiles": Path(tiles).name if tiles else None,
        "bake_sha256": prov.get("bake_sha256"),
    }


def comparability_issues(prov_a: dict, prov_b: dict) -> tuple[list[str], list[str]]:
    """``(fatal, arm)`` mismatches between two runs' provenance.

    ``fatal`` is the ground truth SR is scored against (split, tile bake):
    differing or unrecorded,
    no A/B exists, and nothing waives it. ``arm`` is code, config (apart from
    its withholds where both manifests record it), served model and reasoning
    effort (unset reads as blank), which an A/B is meant to vary; compared only
    where both sides recorded them, and waived by ``--allow-provenance-mismatch``."""
    fatal: list[str] = []
    ga, gb = _ground_truth_keys(prov_a), _ground_truth_keys(prov_b)
    for key, label in (
        ("split", "split"),
        ("tiles", "tile bake (SEARCH_CITYNAV_TILES)"),
    ):
        va, vb = ga.get(key), gb.get(key)
        if not va or not vb:
            fatal.append(f"{label} unrecorded:  A={va}  B={vb}")
        elif va != vb:
            fatal.append(f"{label} differs:  A={va}  B={vb}")

    # Bake content tightens the name check where both sides recorded it.
    ba, bb = ga.get("bake_sha256"), gb.get("bake_sha256")
    if ba and bb and ba != bb:
        fatal.append(
            f"tile bake CONTENT differs at the same name:  "
            f"A={ba[:12]}  B={bb[:12]}"
        )

    arm: list[str] = []
    config = (("config_body", "config apart from its withholds (sha256)")
              if prov_a.get("config_body") and prov_b.get("config_body") else ("config_sha256", "config (sha256)"))
    for key, label in (
        ("git_commit", "code (git_commit)"),
        config,
        ("served_model", "served model"),
        ("served_build", "served model BUILD (tokenizer/quant)"),
    ):
        va, vb = prov_a.get(key), prov_b.get(key)
        if va is not None and vb is not None and va != vb:
            arm.append(f"{label} differs:  A={va}  B={vb}")
    fa, fb = prov_a.get("search_env_flags"), prov_b.get("search_env_flags")
    if fa is not None and fb is not None:
        ea, eb = (f.get("SEARCH_SAMPLING_REASONING_EFFORT") or "unset" for f in (fa, fb))
        if ea != eb:
            arm.append(f"reasoning effort (SEARCH_SAMPLING_REASONING_EFFORT) differs:  A={ea}  B={eb}")
    return fatal, arm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_a", help="baseline run root")
    ap.add_argument("run_b", help="treatment run root")
    ap.add_argument("--label", default="A_vs_B")
    ap.add_argument("--min-overlap-frac", type=float, default=0.8,
                    help="refuse if shared/min(n_a,n_b) is below this (a seed or limit mismatch)")
    ap.add_argument("--allow-low-overlap", action="store_true")
    ap.add_argument("--allow-provenance-mismatch", action="store_true",
                    help="permit comparing runs with different code/config/model (NOT a valid A/B)")
    _rows.add_allow_partial(ap)
    args = ap.parse_args()

    from benchmarks.citynav.analysis.run_report import _CRITERION_LABEL, load_run, run_criterion

    a, _ = load_run(Path(args.run_a), allow_partial=args.allow_partial)
    b, _ = load_run(Path(args.run_b), allow_partial=args.allow_partial)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("no shared scored episodes between the two runs")
    crit_a, crit_b = (run_criterion([run[e] for e in shared]) for run in (a, b))
    if crit_a != crit_b:
        raise SystemExit(
            f"INVALID A/B -- A is scored on {_CRITERION_LABEL.get(crit_a, crit_a)} and B on "
            f"{_CRITERION_LABEL.get(crit_b, crit_b)}: the two numbers answer different questions."
        )

    prov_a, prov_b = _run_provenance(Path(args.run_a)), _run_provenance(Path(args.run_b))
    fatal, arm = comparability_issues(prov_a, prov_b)
    if fatal:
        raise SystemExit(
            "INVALID A/B -- the two runs score against different ground truth:\n  - "
            + "\n  - ".join(fatal)
            + "\nThis is not overridable: the numbers answer different questions."
        )
    if arm and not args.allow_provenance_mismatch:
        raise SystemExit(
            "INVALID A/B -- runs are not comparable:\n  - " + "\n  - ".join(arm)
            + "\n(pass --allow-provenance-mismatch to override)"
        )

    # A low overlap means a seed/limit mismatch, not noise.
    min_n = min(len(a), len(b)) or 1
    if len(shared) < args.min_overlap_frac * min_n and not args.allow_low_overlap:
        raise SystemExit(
            f"low paired overlap shared={len(shared)} / min(n)={min_n} "
            f"(< {args.min_overlap_frac:.0%}) -- a seed or limit mismatch "
            "suspected; pass --allow-low-overlap to override"
        )

    pairs = [(int(bool(a[e].get("SR"))), int(bool(b[e].get("SR")))) for e in shared]
    print(render(ab_stats(pairs, label=args.label)))
    print(f"shared={len(shared)}  only_a={len(set(a) - set(b))}  "
          f"only_b={len(set(b) - set(a))}")
    if prov_a.get("withhold") is not None or prov_b.get("withhold") is not None:
        print(f"treatment: withhold A={prov_a.get('withhold')} B={prov_b.get('withhold')}")
    if arm and args.allow_provenance_mismatch:
        print("WARNING comparing non-comparable runs (override active):\n  - "
              + "\n  - ".join(arm))


if __name__ == "__main__":
    main()
