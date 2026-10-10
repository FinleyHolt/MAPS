"""Canonical single-run / paired SR report for a run root.

ONE trustworthy view so analysis stops hand-rolling reads against the
wrong nesting. The authoritative per-episode record is the ledger row's
``metrics`` dict (``episodes.jsonl``, read through ``analysis.rows``); the
commit-provenance fields live at ``metrics.committed_fid``. The success
bit is ``metrics.SR`` (the runner's authoritative score) -- do NOT
recompute SR from ``pred_xy`` vs ``target_xy``.

WHICH attempt represents an episode is decided in ``analysis.rows``: the
FIRST fully complete attempt, ordered by attempt START time. Every reader
in the tree shares that one rule. Read its module docstring before
touching selection here.

Rows with NO ``metrics`` are worker crashes and no-commit misses. They
carry an episode_id but no result, so they are COUNTED and reported
explicitly as ``no_metrics`` rather than silently dropped -- the
denominator is never quietly inflated, and never quietly deflated either.

Usage:
    python -m benchmarks.citynav.analysis.run_report output/run_a
    python -m benchmarks.citynav.analysis.run_report output/run_a output/run_b  # paired
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

from benchmarks.citynav.analysis import rows as _rows


def load_run(root: Path) -> tuple[dict[str, dict], int]:
    """Return (episode_id -> that episode's scored metrics, no_metrics rows).

    Selection is the shared rule (``rows.pick_attempts``): each episode's
    result is its FIRST FULLY COMPLETE attempt, ordered by attempt START
    time -- NOT the last row in the file. The ledger appends on
    COMPLETION, so last-row-wins scored whichever replicate happened to
    finish last, which under the pre-f22e8fd dispatch was a coin flip.

    ``no_metrics`` is a ROW count: non-infra ledger rows carrying no
    metrics dict at all. It is NOT a count of unscored episodes -- one episode can
    contribute both a crash row and a scored row.

    An episode whose pick carries no metrics (it never committed) scores
    as a miss (``rows.scored_metrics``).
    """
    picked, stats = _rows.load_attempts(root)
    return _rows.scored_metrics(picked), stats.no_metrics


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.1f}%)" if den else f"{num}/0 (--)"


#: How each criterion prints. ``None`` is a run flown before the rows carried
#: the stamp -- named as unrecorded rather than guessed at.
_CRITERION_LABEL = {
    "ring": "the 20 m disc",
    "shape": "the target's own shape",
    None: "UNRECORDED (pre-cutover run)",
}


def run_criterion(metrics: list[dict]) -> str | None:
    """The criterion these rows were scored under, or ``None`` if unstamped.

    Read off the rows, never inferred from coverage. A run whose rows
    disagree is a merged artifact and says so loudly rather than picking the
    majority: two criteria in one report is exactly the confusion this key
    exists to end.
    """
    seen = {m.get("success_criterion") for m in metrics}
    seen.discard(None)
    if len(seen) > 1:
        raise SystemExit(
            f"run mixes success criteria {sorted(seen)} -- these rows were "
            "not all scored on the same mask, so no single SR speaks for "
            "them. Split the run tree by criterion and report each."
        )
    return seen.pop() if seen else None


def print_precise(metrics: list[dict], n: int) -> bool:
    """How a SHAPE-scored SR breaks down by what the target resolved to.

    It pools two different tests — episodes with an annotated shape (strict
    containment) and episodes without one (a tighter disc) — and the mix moves
    as annotation lands. Both are printed, so a change in the headline is
    attributable to ground truth rather than to the radius standing in for it.
    See :func:`benchmarks.citynav.eval.score_precise`.

    Gated on ``precise_target``, NOT on the verdict. ``SR`` is now the verdict
    under whichever mask ran, so gating on it would make every ring row
    "precise-scored" and print the 20 m disc number under a shape's label —
    on a CityNav run, which resolves no shape at all.

    ``n`` is the full episode count, so ``coverage`` states how much of the run
    this number actually speaks for.

    Returns whether it printed anything, so a caller can say what is on
    screen rather than assume.
    """
    pr_scored = [
        m for m in metrics
        if m.get("precise_target") is not None and m.get("SR") is not None
    ]
    if not pr_scored:
        return False
    pr_sr = sum(1 for m in pr_scored if m.get("SR"))
    print(
        f"by target shape = {_pct(pr_sr, len(pr_scored))}  "
        f"(coverage {_pct(len(pr_scored), n)})"
    )
    parts = []
    for kind, label in (("shape", "by shape"), ("radius", "by radius")):
        sub = [m for m in pr_scored if m.get("precise_target") == kind]
        if sub:
            hit = sum(1 for m in sub if m.get("SR"))
            parts.append(f"{label} {_pct(hit, len(sub))}")
    # Shapes narrower than the precision floor were widened to it. Say how
    # many, so a "by shape" number that is mostly floor does not read as a
    # footprint result.
    floored = sum(1 for m in pr_scored if m.get("precise_floor_applied"))
    if floored:
        parts.append(f"floor-widened {floored}")
    if parts:
        print("  " + "   ".join(parts))
    return True


_REPO_ROOT = Path(__file__).resolve().parents[3]


def read_manifest(root: Path) -> dict:
    """The run's receipt, searched shallowest-first.

    Three depths, because three shapes emit one. A fleet root carries it
    at the top; a lane/coord root nests it one down; a DIRECT run
    (``run_benchmark`` invoked straight — the smoke gate, ``launch``, a
    hand-run arm) writes it into its own ``<out>/<split>/<stamp>/`` dir,
    two down. Shallowest wins, so a fleet root still answers with the
    manifest describing the whole fleet rather than one shard's.
    """
    for p in (
        root / "run_manifest.json",
        *sorted(root.glob("*/run_manifest.json")),
        *sorted(root.glob("*/*/run_manifest.json")),
    ):
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            continue
    return {}


def _local_bake_dir(resolved: str | None) -> Path | None:
    """This machine's copy of the tiles dir a run recorded flying.

    ``tiles_dir_resolved`` is absolute on the machine that flew the run: the
    path itself when it is a directory here, else the longest tail of it
    that is a directory under this repo --
    ``/scratch/<user>/MAPS/benchmarks/calcity/tiles`` resolves to
    ``benchmarks/calcity/tiles``. Longest tail first, so a run under
    ``.../tiles`` cannot match a shorter unrelated path.
    """
    if not resolved:
        return None
    if Path(resolved).is_dir():
        return Path(resolved)
    parts = PurePosixPath(resolved).parts
    for i in range(1, len(parts) - 1):          # a tail of two parts at least: "tiles" alone names nothing
        cand = _REPO_ROOT.joinpath(*parts[i:])
        if cand.is_dir():
            return cand
    return None


def bake_line(root: Path) -> str:
    """One line saying whether this run's tile bake is the one on disk now.

    Every other provenance check compares NAMES. Two bakes of the same
    blocks live at the same path under the same name and pass all of them,
    and the SR moves anyway (0.351 vs 0.169 on identical episodes). This
    compares content.

    WHICH bake to compare against is the run's own business: the tiles dir
    it recorded resolving. Only a run that recorded none falls back to this
    checkout's default bake -- reaching for that default first is how a Cal
    City run reported a mismatch against CityNav tiles on every read.

    Never raises and never refuses: a run flown before the stamp existed
    has no hash to check, and saying so is the honest state. Missing tiles
    are reported as unknown, not as a pass.
    """
    from benchmarks.citynav import paths

    perception = read_manifest(root).get("perception") or {}
    stamped = perception.get("bake_sha256")
    recorded = perception.get("tiles_dir_resolved")

    if not stamped:
        return "bake: UNVERIFIED (run stamps no bake_sha256)"
    local = _local_bake_dir(recorded) if recorded else paths.TILES_DIR
    if local is None:
        return f"bake: UNKNOWN (the run's bake {recorded} is not on this machine)  run={stamped[:12]}"
    here = paths.bake_sha256(local)
    if here is None:
        return f"bake: UNKNOWN (no bake at {local})  run={stamped[:12]}"
    if here == stamped:
        return f"bake: verified {stamped[:12]} ({local})"
    return (
        f"bake: *** MISMATCH *** run={stamped[:12]} local={here[:12]} at {local}. "
        f"The SR below was measured against a DIFFERENT bake of the same name; "
        f"it is not comparable to a number measured against this one."
    )


def report(root: Path) -> dict[str, dict]:
    picked, stats = _rows.load_attempts(root)
    scored = _rows.scored_metrics(picked)
    n = len(scored)
    if n == 0:
        raise SystemExit(
            f"run_report: no scored rows under {root} "
            f"(tier: {stats.tier}, files: {stats.files}, "
            f"attempt rows: {stats.rows}, no_metrics: {stats.no_metrics}, "
            f"infra: {stats.infra})"
        )
    sr = sum(1 for m in scored.values() if m.get("SR"))
    _criterion = run_criterion(list(scored.values()))
    print(f"\n=== {root} ===")
    print(f"source: {stats.tier} tier, {stats.files} file(s)")
    print(bake_line(root))
    # ``rows`` counts ATTEMPTS; ``scored`` counts EPISODES. Their sum was
    # never a total, which is what made the old rows_total misleading.
    print(
        f"scored={n}  attempts={stats.rows}  "
        f"replicates_discarded={stats.keyed - len(picked)}  "
        f"no_metrics(crash/no-commit)={stats.no_metrics}  "
        f"timeouts(wall-clock)={stats.timed_out}  "
        f"infra(re-queued)={stats.infra}"
    )
    # WHICH CRITERION THIS RUN IS SCORED UNDER, read off the rows' own stamp.
    # There is ONE number now -- ``SR`` is the verdict under the bench's mask,
    # and ``success_criterion`` says which mask that was. The >=90%-of-rows-
    # resolve-a-shape heuristic this replaces existed because nothing on a row
    # said; it had already been copied to two other readers and diverged once.
    print(f"criterion: {_CRITERION_LABEL.get(_criterion, _criterion)}")
    print(f"SR = {_pct(sr, n)}")
    # The one distance on every scored row, under either criterion: metres
    # from the commit to the target centroid, no radius in it.
    ne = [float(m["NE"]) for m in scored.values() if isinstance(m.get("NE"), (int, float))]
    print(f"NE median = {statistics.median(ne):.1f} m  mean = {sum(ne) / len(ne):.1f} m (n={len(ne)})"
          if ne else "NE = -- (no scored row carries NE)")
    # The one route term on a scored row. Absent on a row that never flew.
    spl = [float(m["spl_pos"]) for m in scored.values()
           if isinstance(m.get("spl_pos"), (int, float))]
    print(f"mean spl_pos = {sum(spl) / len(spl):.4f} (n={len(spl)})" if spl
          else "mean spl_pos = -- (no scored row carries spl_pos)")
    # SAHI tiles the segmenter could not answer: those frames were
    # partial-blind, so a run that lost any says so beside its SR.
    tiles = [int(m.get("sam_tiles_failed") or 0) for m in scored.values()]
    if any(tiles):
        print(f"SAHI tiles failed = {sum(tiles)} over {sum(1 for t in tiles if t)} "
              f"of {n} scored episode(s)")
    shaped = print_precise(list(scored.values()), n)
    if _criterion is None:
        # A run flown before the rows carried the stamp. Both readings are
        # printed and NEITHER is called the headline: the criterion is
        # unrecorded ground truth, and picking one here would be a guess
        # that misreports the headline whenever it is wrong.
        #
        # What it says depends on whether a shape reading actually printed:
        # pointing at "the shape line beside it" when the run carries no
        # shaped rows sends the reader looking for a line that is not there.
        print("  ^ this run predates the criterion stamp: the SR above is the "
              "20 m disc"
              + (", and the shape line beside it is the other reading. Which "
                 "one this bench is scored on is NOT recorded"
                 if shaped else
                 ", and the run carries no shape verdict at all. Whether this "
                 "bench is scored on one is NOT recorded")
              + " -- quote neither as the bench's SR.")
    # OSR and the (OSR, SR) joint. SR alone cannot say whether a run got
    # WORSE at reaching the target or better at committing once it did:
    # OSR fell 0.68 -> 0.64 on the default-frame arm while SR rose, and
    # neither marginal shows that. Conversion = SR/OSR is the cell that
    # moved. OSR=0,SR=1 is impossible by construction (SR is the final
    # pose inside R, OSR the trajectory minimum inside R), so a nonzero
    # count there means the two metrics came from different geometry.
    #
    # OSR is RING-basis and rides ring rows only. On a bench scored against
    # the target's own shape the rows carry none, and printing 0/n there
    # would put a total-failure-to-reach on the report of a run that never
    # asked the question. The block is skipped instead of zeroed.
    osr_rows = [m for m in scored.values() if "OSR" in m]
    if osr_rows:
        osr = sum(1 for m in osr_rows if m.get("OSR"))
        joint = Counter((bool(m.get("OSR")), bool(m.get("SR"))) for m in osr_rows)
        print(f"overall OSR = {_pct(osr, len(osr_rows))}")
        print(
            f"OSRxSR: reached+committed={joint[(True, True)]}  "
            f"reached+missed={joint[(True, False)]}  "
            f"never_reached={joint[(False, False)]}  "
            f"impossible(SR w/o OSR)={joint[(False, True)]}  "
            f"conversion(SR|OSR)={_pct(joint[(True, True)], osr)}"
        )

    # SR by referent bucket.
    bt: Counter = Counter()
    bs: Counter = Counter()
    for m in scored.values():
        b = m.get("referent_bucket") or "?"
        bt[b] += 1
        bs[b] += bool(m.get("SR"))
    print("by bucket: " + "   ".join(_label(k, bs[k], bt[k]) for k in _BUCKETS if bt[k]))

    # Commit provenance (prior landmark vs runtime detection vs none).
    prov = Counter(m.get("committed_provenance") or "none" for m in scored.values())
    print("provenance: " + "  ".join(f"{k}={v}" for k, v in prov.most_common()))

    # Among SCORED FAILS: was the committed xy itself within the success
    # radius? A stage fact, not a cause — why the commit landed where it
    # did takes reading the episode.
    fails = [m for m in scored.values() if not m.get("SR")]
    commit_in = sum(1 for m in fails if m.get("commit_would_score") is True)
    commit_out = sum(1 for m in fails if m.get("commit_would_score") is False)
    unk = len(fails) - commit_in - commit_out
    print(
        f"fails={len(fails)}: commit_within_R={commit_in}  "
        f"commit_beyond_R={commit_out}  (no commit_xy={unk})"
    )


    # Failure mechanism comes from reading each episode's own record. A
    # split keyed on OSR sorts misses by transit proximity while reading as
    # mechanism, so the report prints counts only.
    return scored


_BUCKETS = ("car", "building", "other", "?")


def _label(name: str, suc: int, tot: int) -> str:
    return f"{name}:{suc}/{tot}({100 * suc / tot:.0f}%)" if tot else f"{name}:0"


def paired(root_a: Path, root_b: Path) -> None:
    """Paired SR on the shared scored episode set (apples-to-apples).

    UNDER THE SAME CRITERION THE HEADLINE USES -- now by construction
    rather than by two readers deriving it the same way. ``SR`` is the
    verdict under whichever mask ran, so this pairs the same quantity
    ``report`` printed. The criterion is still NAMED in the header so no
    one has to infer which question moved, and a pairing ACROSS criteria
    refuses: it would difference two different questions.
    """
    a, _ = load_run(root_a)
    b, _ = load_run(root_b)
    shared = sorted(set(a) & set(b))
    crit_a = run_criterion([a[e] for e in shared])
    crit_b = run_criterion([b[e] for e in shared])
    if crit_a != crit_b:
        raise SystemExit(
            f"INVALID pairing -- {root_a.name} is scored on "
            f"{_CRITERION_LABEL.get(crit_a, crit_a)} and {root_b.name} on "
            f"{_CRITERION_LABEL.get(crit_b, crit_b)}. The two numbers answer "
            "different questions."
        )
    _key = "SR"
    print(
        f"\n=== PAIRED on {len(shared)} shared episodes, "
        f"{_CRITERION_LABEL.get(crit_a, crit_a)} "
        f"({root_a.name} vs {root_b.name}) ==="
    )
    for name, D in ((root_a.name, a), (root_b.name, b)):
        bt: Counter = Counter()
        bs: Counter = Counter()
        for eid in shared:
            m = D[eid]
            bk = m.get("referent_bucket") or "?"
            bt["all"] += 1
            bs["all"] += bool(m.get(_key))
            bt[bk] += 1
            bs[bk] += bool(m.get(_key))
        line = "   ".join(
            _label(k, bs[k], bt[k]) for k in ("all", *_BUCKETS) if bt[k]
        )
        print(f"{name:18s} {line}")


def main(argv: list[str]) -> int:
    args = [Path(a) for a in argv[1:]]
    if not args:
        print(__doc__)
        return 2
    for root in args:
        report(root)
    if len(args) == 2:
        paired(args[0], args[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
