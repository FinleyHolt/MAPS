#!/usr/bin/env python3
"""Paired A/B SR report with a bootstrap CI on the delta, McNemar, and the
success->fail regression guard -- the gate every SR/DeltaSR claim passes through.

WHY this exists: `run_report.paired` prints point SR for A and B but no CI on the
*difference* and no flip count. The pose-lever work nearly shipped a phantom "+5"
for exactly that reason (a stochastic verifier + thresholds overfit on n=298). A
commit-protected override can only help in the oracle limit, but a *realistic*
re-rank uses a fallible signal and can flip a SUCCESS to a FAIL -- so the
success->fail count is the load-bearing safety number, not net SR.

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
    stat: str = "mean",
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean (or median) of per-episode (b - a).

    Deterministic (seeded) so a re-run reproduces the interval. Pure stdlib:
    a small LCG-free resampler via random.Random. ``stat="median"`` is for
    the continuous metrics (NE), whose tail of 1 km misses drags a mean
    far off the typical episode.
    """
    import random
    import statistics

    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        draw = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.median(draw) if stat == "median" else sum(draw) / n)
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


def ab_stats(
    pairs: list[tuple[int, int]],
    *,
    label: str = "",
    metric: str = "SR",
) -> dict:
    """Paired statistics over (a, b) in {0,1}. The shared-set core.

    ``metric`` names WHICH bit was paired (SR, OSR, ...). It is a label
    only -- the arithmetic is identical -- but an OSR delta printed under
    an ``SR_a=`` heading is the kind of mislabel that outlives the session
    that made it. The dict keys stay ``SR_a``/``SR_b`` so every stored
    ledger column and its readers keep parsing."""
    n = len(pairs)
    sr_a = sum(a for a, _ in pairs) / n if n else float("nan")
    sr_b = sum(b for _, b in pairs) / n if n else float("nan")
    diffs = [float(b - a) for a, b in pairs]
    lo, hi = _bootstrap_ci(diffs)
    mc = _mcnemar(pairs)
    return {
        "label": label,
        "metric": metric,
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


def paired_continuous(
    pairs: list[tuple[float, float]],
    *,
    label: str = "",
    metric: str = "",
    boot_seed: int = 12345,
) -> dict:
    """Paired Wilcoxon signed-rank on a CONTINUOUS metric (e.g. ``NE``).

    WHY: a 20 m success cut throws away most of a distance's information,
    so a run can move the distribution while SR barely moves. The
    default-frame arm shifted mean NE 47.6 -> 44.4 m against +0.7 SR
    points; only the continuous read can say whether that is real.

    The MEDIAN of the per-episode difference is the reported point (the
    statistic Wilcoxon actually tests is the symmetry of the signed
    ranks, and the mean is dragged by a handful of 1 km misses). Its CI
    is the same seeded percentile bootstrap the SR delta uses, so the
    two intervals are constructed the same way. scipy supplies only the
    exact/normal signed-rank p."""
    import statistics

    from scipy.stats import wilcoxon

    n = len(pairs)
    diffs = [float(b - a) for a, b in pairs]
    nz = [d for d in diffs if d != 0.0]
    med_lo, med_hi = _bootstrap_ci(diffs, seed=boot_seed, stat="median")
    # The MEAN gets an interval too. A bare mean delta beside an
    # interval-bearing median reads as the solid number and is the
    # opposite: -3.21 m on the default-frame arm looked like a real gain
    # until its own CI came back [-8.61,+2.00]. Never print one naked.
    mean_lo, mean_hi = _bootstrap_ci(diffs, seed=boot_seed)
    p = float(wilcoxon(diffs).pvalue) if nz else float("nan")
    med = statistics.median(diffs) if n else float("nan")
    return {
        "label": label,
        "metric": metric,
        "n": n,
        "mean_a": round(sum(a for a, _ in pairs) / n, 4) if n else float("nan"),
        "mean_b": round(sum(b for _, b in pairs) / n, 4) if n else float("nan"),
        "median_delta": round(med, 4),
        "median_delta_ci95": [round(med_lo, 4), round(med_hi, 4)],
        "mean_delta": round(sum(diffs) / n, 4) if n else float("nan"),
        "mean_delta_ci95": [round(mean_lo, 4), round(mean_hi, 4)],
        "n_nonzero": len(nz),
        "n_improved": sum(1 for d in nz if d < 0),
        "n_worsened": sum(1 for d in nz if d > 0),
        "wilcoxon_p": p,
        "significant_95": bool(p < 0.05),
    }


def render_continuous(st: dict) -> str:
    ci = st["median_delta_ci95"]
    mci = st["mean_delta_ci95"]
    sig = "SIGNIFICANT" if st["significant_95"] else "not significant"
    return (
        f"paired {st['metric']} ({st['label']}): n={st['n']}  "
        f"mean_a={st['mean_a']:.2f}  mean_b={st['mean_b']:.2f}\n"
        f"  mean   Δ={st['mean_delta']:+.2f}  "
        f"95%CI[{mci[0]:+.2f},{mci[1]:+.2f}]\n"
        f"  median Δ={st['median_delta']:+.2f}  "
        f"95%CI[{ci[0]:+.2f},{ci[1]:+.2f}]  "
        f"improved={st['n_improved']}  worsened={st['n_worsened']}  "
        f"ties={st['n'] - st['n_nonzero']}\n"
        f"  Wilcoxon signed-rank p={st['wilcoxon_p']:.4g} -> {sig}"
    )


def render(st: dict) -> str:
    ci = st["delta_ci95"]
    sig = "SIGNIFICANT" if st["significant_95"] else "not significant"
    m = st.get("metric", "SR")
    return (
        f"A/B paired ({st['label']}): n={st['n']}  "
        f"{m}_a={st['SR_a']:.3f}  {m}_b={st['SR_b']:.3f}  "
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


def _row_metrics(r: dict) -> tuple[str | None, dict]:
    """``(episode_id, metrics)`` off a ledger row."""
    eid = r.get("episode_id")
    return (str(eid) if eid is not None else None), (r.get("metrics") or {})


def _sr_by_episode(run_root: Path, key: str = "SR") -> dict[str, int]:
    """Binary metric bit per episode via the CANONICAL pick
    (rows.load_attempts: first fully complete attempt by start time). The
    prior last-row-wins walk here disagreed with every other reader on
    duplicated episodes — on the oraclec matrix it flipped 2 episodes and
    moved a pair's McNemar across 0.05 (2026-08-02 adversarial
    verification). ``key`` selects which served bit (SR, OSR); it is
    never recomputed from pred_xy vs target_xy."""
    from benchmarks.citynav.analysis.run_report import load_run

    scored, _ = load_run(run_root)
    if not any(key in m for m in scored.values()):
        raise SystemExit(f"--metric {key}: no scored row of {run_root} carries {key}")
    return {eid: int(bool(m.get(key))) for eid, m in scored.items()}


def _continuous_by_episode(run_root: Path, key: str) -> dict[str, float]:
    """Served continuous metric per episode, same canonical pick.

    Episodes whose pick carries no value for ``key`` are ABSENT rather
    than defaulted -- a 0.0 fill on a distance metric reads as a perfect
    landing."""
    from benchmarks.citynav.analysis.run_report import load_run

    scored, _ = load_run(run_root)
    return {eid: float(m[key]) for eid, m in scored.items()
            if m.get(key) is not None}


def _cue_type_by_episode(run_root: Path) -> dict[str, str]:
    """episode_id -> ``target_object_type``, the AUTHORITATIVE CityRefer class
    of the GT target (stamped by episode_runner into metrics).
    ``referent_bucket`` is a text-regex proxy that mislabels ~23% of building
    targets -- never group on it."""
    out: dict[str, str] = {}
    for _f, r in _iter_rows(run_root):
        eid, m = _row_metrics(r)
        ct = m.get("target_object_type")
        if eid is not None and ct:
            out[eid] = str(ct)
    return out


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
    Cal City bakes all sit at ``benchmarks/calcity/tiles`` and two different
    ones are live across the cluster today.
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
    those prefixes inside ONE run), while the retired bake differs exactly
    in the name (citynav_2m vs citynav_2m_osmv2)."""
    flags = prov.get("search_env_flags") or {}
    tiles = flags.get("SEARCH_CITYNAV_TILES")
    return {
        "split": prov.get("split"),
        "tiles": Path(tiles).name if tiles else None,
        # Content, where the name cannot answer. Every CityNav bake encodes
        # itself in its directory (citynav_2m_osmv2 vs _osmv3) and the name
        # check catches those; every Cal City bake is called ``tiles``, and
        # two different ones are live across the cluster right now.
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
    # Presence-sensitive on purpose (unlike the arm keys above, whose
    # absence-tolerant guard would miss flag-on-A / absent-on-B): an
    # oracle-injected run vs a clean run is never a like-for-like A/B
    # without the deliberate --allow-provenance-mismatch waiver.
    oa = (prov_a.get("search_env_flags") or {}).get("SEARCH_ORACLE_INJECT_GT")
    ob = (prov_b.get("search_env_flags") or {}).get("SEARCH_ORACLE_INJECT_GT")
    if oa != ob:
        arm.append(
            f"oracle GT inject (SEARCH_ORACLE_INJECT_GT) differs:  A={oa}  B={ob}"
        )
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
    ap.add_argument("--dump-flips", default=None, metavar="FILE",
                    help="write the paired flip lists as JSON: "
                         "{wins: [episode_id...], losses: [...]} — wins are "
                         "fail(A)->success(B), losses success(A)->fail(B). "
                         "The forensics entry point: every trace-level "
                         "mechanism hunt starts from these ids.")
    ap.add_argument("--subset", default=None, metavar="FILE",
                    help="episode-id list file (one per line): pair and "
                         "test ONLY these episodes, reported alongside the "
                         "full-set headline")
    ap.add_argument("--metric", default="SR", metavar="KEY",
                    help="which SERVED binary metric to pair on (SR, OSR). "
                         "OSR answers a different question than SR -- "
                         "'did the flight path ever pass within R' vs 'did "
                         "the commit land there' -- and an arm can move "
                         "them in opposite directions.")
    ap.add_argument("--continuous", default=None, metavar="KEY",
                    help="ALSO run a paired Wilcoxon signed-rank on a "
                         "continuous served metric (ne_pos_to_goal). A 20 m "
                         "cut discards most of a distance's information, so "
                         "the continuous read can resolve what the binary "
                         "one cannot.")
    ap.add_argument("--by-cue-type", action="store_true",
                    help="also break the paired stats down by "
                         "target_object_type (the authoritative CityRefer "
                         "class stamped in metrics; referent_bucket is a "
                         "noisy text proxy and is never used)")
    args = ap.parse_args()

    a = _sr_by_episode(Path(args.run_a), args.metric)
    b = _sr_by_episode(Path(args.run_b), args.metric)
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
    # coord-root) or the seed/limit differed, not noise. Assessed on the FULL
    # pairing, before any --subset narrowing: the subset is deliberate, so its
    # size says nothing about coord-root or seed health.
    min_n = min(len(a), len(b)) or 1
    if len(shared) < args.min_overlap_frac * min_n and not args.allow_low_overlap:
        raise SystemExit(
            f"low paired overlap shared={len(shared)} / min(n)={min_n} "
            f"(< {args.min_overlap_frac:.0%}) -- coord-root theft or seed/limit mismatch "
            "suspected; pass --allow-low-overlap to override"
        )

    if args.subset:
        want = {ln.strip() for ln in Path(args.subset).read_text().splitlines()
                if ln.strip()}
        n_before = len(shared)
        shared = [e for e in shared if e in want]
        print(f"subset filter: {len(shared)}/{n_before} shared episodes "
              f"(subset file lists {len(want)})")
        if not shared:
            raise SystemExit("no shared scored episodes inside --subset")

    pairs = [(a[e], b[e]) for e in shared]
    st = ab_stats(pairs, label=args.label, metric=args.metric)
    if args.dump_flips:
        flips = {
            "wins": [e for e in shared if a[e] == 0 and b[e] == 1],
            "losses": [e for e in shared if a[e] == 1 and b[e] == 0],
        }
        Path(args.dump_flips).write_text(json.dumps(flips, indent=1))
        print(f"-> flips dumped to {args.dump_flips} "
              f"(wins={len(flips['wins'])} losses={len(flips['losses'])})")
    extra = (f"shared={len(shared)}  only_a={len(set(a) - set(b))}  "
             f"only_b={len(set(b) - set(a))}")
    print(render(st))
    print(extra)
    if args.continuous:
        ca = _continuous_by_episode(Path(args.run_a), args.continuous)
        cb = _continuous_by_episode(Path(args.run_b), args.continuous)
        ids = [e for e in shared if e in ca and e in cb]
        if len(ids) < 0.9 * len(shared):
            raise SystemExit(
                f"--continuous {args.continuous}: only {len(ids)}/{len(shared)} "
                "shared episodes carry the key on BOTH sides — a paired test "
                "over a silently shrunken set is not the test that was asked for"
            )
        print(render_continuous(paired_continuous(
            [(ca[e], cb[e]) for e in ids],
            label=args.label, metric=args.continuous)))
    if args.by_cue_type:
        # Treatment stamps the type; older baselines may predate it, so B
        # fills first and A backfills. Both arms score the same GT object, so
        # a disagreement is impossible and order is only about coverage.
        ct = {**_cue_type_by_episode(Path(args.run_a)),
              **_cue_type_by_episode(Path(args.run_b))}
        groups: dict[str, list[str]] = {}
        for e in shared:
            groups.setdefault(ct.get(e, "unstamped"), []).append(e)
        for cue in sorted(groups, key=lambda k: -len(groups[k])):
            eps = groups[cue]
            print(render(ab_stats([(a[e], b[e]) for e in eps],
                                  label=f"{args.label}/{cue}")))
    if arm and args.allow_provenance_mismatch:
        print("WARNING comparing non-comparable runs (override active):\n  - "
              + "\n  - ".join(arm))


if __name__ == "__main__":
    main()
