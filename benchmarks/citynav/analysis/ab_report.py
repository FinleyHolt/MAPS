#!/usr/bin/env python3
"""Paired A/B SR report with a bootstrap CI on the delta, McNemar, and the
success->fail regression guard -- the gate every SR/DeltaSR claim passes through.

`run_report.paired` prints point SR for A and B; this adds the CI on the
*difference* and the flip counts. A change can trade successes for failures
at a flat net SR, so the success->fail count is reported beside the delta.

Two entry points:
  * `ab_stats(pairs)` -- library core. `pairs` = list of (sr_a, sr_b) in {0,1}
    over the SHARED episode set (already paired). Returns SR_a, SR_b, delta, a
    bootstrap CI on the delta (episode resampling), McNemar (b/c discordant +
    chi-square-1df p via math.erfc), and the win/loss (fail->success /
    success->fail) flip counts. No numpy/scipy.
  * CLI -- pair two run roots by episode_id on SR and print the verdict.

Usage:
  python -m benchmarks.citynav.analysis.ab_report <run_a> <run_b> [--label "..."]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from benchmarks.citynav.analysis import rows as _rows


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (95% at z=1.96):
    the capability figure's interval."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _bootstrap_ci(
    diffs: list[float], *, iters: int = 10000, seed: int = 12345, alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-episode (b - a).

    Deterministic (seeded) so a re-run reproduces the interval. Pure stdlib:
    a small LCG-free resampler via random.Random.
    """
    import random

    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        draw = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (lo, hi)


def _mcnemar(pairs: list[tuple[int, int]]) -> dict:
    """Exact-ish McNemar over discordant pairs.

    b = A-success & B-fail (losses), c = A-fail & B-success (wins). With
    continuity correction, chi2 = (|b-c|-1)^2 / (b+c) ~ chi2(1); the p-value of
    chi2(1) is erfc(sqrt(chi2/2)) (stdlib). For small b+c report the exact
    two-sided binomial p instead (more honest than the chi2 approx).
    """
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


# --- two-run CLI: pair by episode_id on SR --------------------------------------


def _iter_rows(run_root: Path):
    """Every parsed ledger row under the run root, as ``(file, row)``, in
    the canonical tier order (``rows.result_files``)."""
    for f in _rows.result_files(run_root)[0]:
        for r in _rows.iter_json_rows(f):
            yield f, r


def _sr_by_episode(run_root: Path, *, allow_partial: bool = False) -> dict[str, int]:
    """The served SR bit per episode via the CANONICAL pick
    (rows.load_attempts: first fully complete attempt by start time); it is
    never recomputed from pred_xy vs target_xy."""
    from benchmarks.citynav.analysis.run_report import load_run

    scored, _ = load_run(run_root, allow_partial=allow_partial)
    return {eid: int(bool(m.get("SR"))) for eid, m in scored.items()}


def _run_provenance(run_root: Path) -> dict:
    """The first row's provenance block (git/config/model identity) stamped
    by run_benchmark, with the tile bake's own stamp attached. Empty if the
    run predates provenance stamping."""
    for _f, r in _iter_rows(run_root):
        prov = r.get("provenance")
        if prov:
            bake = _bake_stamp(run_root)
            if bake:
                prov = {**prov, "bake_sha256": bake}
            return prov
    return {}


def _bake_stamp(run_root: Path) -> str | None:
    """The tile-bake CONTENT hash the run recorded, from its manifest.

    The rows carry the bake's PATH; the manifest carries what was in it. Two
    bakes of the same block set live at the same path under the same name
    (`paths.bake_sha256`), which is exactly the case the path cannot see:
    every Cal City bake sits at ``benchmarks/calcity/tiles``.
    """
    try:
        manifest = json.loads((run_root / "run_manifest.json").read_text())
    except (OSError, ValueError):
        return None
    return (manifest.get("perception") or {}).get("bake_sha256")


def _ground_truth_keys(prov: dict) -> dict:
    """The facts that decide WHAT was measured, not how well it scored.

    The tile bake's identity is its NAME, not the absolute path: every site
    mounts the same bake under its own root (a cross-site pool run mixes
    those prefixes inside ONE run), while two CityNav bakes differ in the
    name (citynav_2m_osmv2 vs citynav_2m_osmv3)."""
    flags = prov.get("search_env_flags") or {}
    tiles = flags.get("SEARCH_CITYNAV_TILES")
    return {
        "split": prov.get("split"),
        "tiles": Path(tiles).name if tiles else None,
        # Content, where the name cannot answer. Every CityNav bake encodes
        # itself in its directory (citynav_2m_osmv2 vs _osmv3) and the name
        # check catches those; every Cal City bake is called ``tiles``, so two
        # different ones share a name.
        "bake_sha256": prov.get("bake_sha256"),
    }


def comparability_issues(prov_a: dict, prov_b: dict) -> tuple[list[str], list[str]]:
    """``(fatal, arm)`` -- two classes of mismatch, because they are not the
    same kind of problem.

    ``fatal`` is the ground truth the SR is a score AGAINST: the split and the
    tile bake. Differ on those and the two numbers answer different questions,
    so there is no A/B to run and no intent that makes one valid. Measured: the
    retired ``citynav_2m`` bake scored the same episodes 0.351 where
    ``citynav_2m_osmv2`` scores 0.169, and a baseline recalled from the wrong
    bake was nearly used to validate a 5,311-episode run.

    ``arm`` is code / config / served model / look tool. These SHOULD differ in
    a refactor or model A/B, so ``--allow-provenance-mismatch`` waives them. It
    does NOT waive ``fatal``: an all-or-nothing override means the one
    legitimate reason to override (a code A/B) also silences the tile check,
    which is precisely how the incident above happens.

    UNRECORDED ground truth is fatal too. A null that skips the comparison
    turns the guard off exactly where it is blindest -- every pre-stamping
    run read None on both sides and passed, which is how a cross-split
    pairing would have sailed through. Comparing nothing must refuse, not
    pass; arm keys keep their absence-tolerant guard (an old run without a
    recorded model is still a legitimate baseline for a code A/B)."""
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

    # Bake CONTENT, checked only when both sides recorded it. Unrecorded is
    # not fatal here, and that is not the loophole it looks like: the tiles
    # check above already refuses an unrecorded bake, so ground truth is
    # never compared against nothing. This key only ever TIGHTENS -- it
    # catches the pair the name says is fine. Made fatal-on-absent instead,
    # it would invalidate every column stamped before the manifest carried
    # the hash, which is most of the ledger.
    ba, bb = ga.get("bake_sha256"), gb.get("bake_sha256")
    if ba and bb and ba != bb:
        fatal.append(
            f"tile bake CONTENT differs at the same name:  "
            f"A={ba[:12]}  B={bb[:12]}"
        )

    arm: list[str] = []
    for key, label in (
        ("git_commit", "code (git_commit)"),
        ("config_sha256", "config (sha256)"),
        ("served_model", "served model"),
        ("served_build", "served model BUILD (tokenizer/quant)"),
        ("look_tool", "look tool"),
    ):
        va, vb = prov_a.get(key), prov_b.get(key)
        if va is not None and vb is not None and va != vb:
            arm.append(f"{label} differs:  A={va}  B={vb}")
    return fatal, arm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_a", help="baseline run root")
    ap.add_argument("run_b", help="treatment run root")
    ap.add_argument("--label", default="A_vs_B")
    ap.add_argument("--min-overlap-frac", type=float, default=0.8,
                    help="refuse if shared/min(n_a,n_b) is below this (coord-root/seed bug)")
    ap.add_argument("--allow-low-overlap", action="store_true")
    ap.add_argument("--allow-provenance-mismatch", action="store_true",
                    help="permit comparing runs with different code/config/model (NOT a valid A/B)")
    _rows.add_allow_partial(ap)
    args = ap.parse_args()

    a = _sr_by_episode(Path(args.run_a), allow_partial=args.allow_partial)
    b = _sr_by_episode(Path(args.run_b), allow_partial=args.allow_partial)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("no shared scored episodes between the two runs")

    # Guard: the two runs must be COMPARABLE. `fatal` is the ground truth the
    # score is against (split, tile bake) and is NOT waivable -- no intent makes
    # a cross-bake delta mean anything. `arm` is code/config/model/look, which a
    # refactor or model A/B is expected to vary, so that one is waivable.
    fatal, arm = comparability_issues(
        _run_provenance(Path(args.run_a)), _run_provenance(Path(args.run_b))
    )
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

    # Guard: a low shared/total overlap means episodes were stolen (shared
    # coord-root) or the seed/limit differed, not noise.
    min_n = min(len(a), len(b)) or 1
    if len(shared) < args.min_overlap_frac * min_n and not args.allow_low_overlap:
        raise SystemExit(
            f"low paired overlap shared={len(shared)} / min(n)={min_n} "
            f"(< {args.min_overlap_frac:.0%}) -- coord-root theft or seed/limit mismatch "
            "suspected; pass --allow-low-overlap to override"
        )

    pairs = [(a[e], b[e]) for e in shared]
    print(render(ab_stats(pairs, label=args.label)))
    print(f"shared={len(shared)}  only_a={len(set(a) - set(b))}  "
          f"only_b={len(set(b) - set(a))}")
    if arm and args.allow_provenance_mismatch:
        print("WARNING comparing non-comparable runs (override active):\n  - "
              + "\n  - ".join(arm))


if __name__ == "__main__":
    main()
