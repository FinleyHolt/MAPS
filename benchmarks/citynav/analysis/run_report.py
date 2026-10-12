"""SR report for a run root, over the ``metrics`` of the attempts ``analysis.rows`` picks.

Usage:
    python -m benchmarks.citynav.analysis.run_report output/run_a [output/run_b ...]

A root that scores fewer episodes than its manifest selected is refused
unless ``--allow-partial`` (``rows.load_attempts``).
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

from benchmarks.citynav.analysis import rows as _rows
from benchmarks.citynav.analysis.rows import read_manifest


def load_run(root: Path, *, allow_partial: bool = False) -> tuple[dict[str, dict], int]:
    """(episode_id -> scored metrics, no_metrics) where ``no_metrics`` counts
    ROWS: one episode can have both a crash row and a scored row."""
    picked, stats = _rows.load_attempts(root, allow_partial=allow_partial)
    return _rows.scored_metrics(picked), stats.no_metrics


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.1f}%)" if den else f"{num}/0 (--)"


#: How each criterion prints; ``None`` is a row without the stamp.
_CRITERION_LABEL = {
    "ring": "the 20 m disc",
    "shape": "the target's own shape",
    None: "UNRECORDED",
}


def run_criterion(metrics: list[dict]) -> str | None:
    """The criterion stamped on these rows, or ``None``; rows that disagree raise."""
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
    """A shape-scored SR split into shape containment and disc fallback
    (``eval.score_precise``), with coverage over ``n`` episodes. Gated on
    ``precise_target``, which ring rows lack. Returns whether it printed."""
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
    floored = sum(1 for m in pr_scored if m.get("precise_floor_applied"))
    if floored:
        parts.append(f"floor-widened {floored}")
    if parts:
        print("  " + "   ".join(parts))
    return True


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _local_bake_dir(resolved: str | None) -> Path | None:
    """This machine's copy of the recorded ``tiles_dir_resolved``: the path
    itself, else its longest tail that is a directory under this repo
    (``/scratch/<user>/MAPS/benchmarks/calcity/tiles`` -> ``benchmarks/calcity/tiles``)."""
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
    """One line: does the run's ``bake_sha256`` match the content of the bake
    on disk (``paths.bake_sha256``)? Compared against the tiles dir the run
    recorded, else this checkout's default. Never raises; missing tiles or
    stamp read as unknown or unverified, never as a pass."""
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


def report(root: Path, *, allow_partial: bool = False) -> dict[str, dict]:
    picked, stats = _rows.load_attempts(root, allow_partial=allow_partial)
    scored = _rows.scored_metrics(picked)
    n = len(scored)
    if n == 0:
        raise SystemExit(
            f"run_report: no scored rows under {root} "
            f"(attempt rows: {stats.rows}, no_metrics: {stats.no_metrics}, "
            f"infra: {stats.infra})"
        )
    sr = sum(1 for m in scored.values() if m.get("SR"))
    _criterion = run_criterion(list(scored.values()))
    print(f"\n=== {root} ===")
    print(bake_line(root))
    # ``rows`` counts attempts, ``scored`` episodes.
    print(
        f"scored={n}  attempts={stats.rows}  "
        f"replicates_discarded={stats.keyed - len(picked)}  "
        f"no_metrics(crash/timeout-no-record)={stats.no_metrics}  "
        f"timeouts(wall-clock)={stats.timed_out}  "
        f"infra(re-queued)={stats.infra}"
    )
    print(f"criterion: {_CRITERION_LABEL.get(_criterion, _criterion)}")
    print(f"SR = {_pct(sr, n)}")
    # Metres from the commit to the target centroid, on every scored row.
    ne = [float(m["NE"]) for m in scored.values() if isinstance(m.get("NE"), (int, float))]
    print(f"NE median = {statistics.median(ne):.1f} m  mean = {sum(ne) / len(ne):.1f} m (n={len(ne)})"
          if ne else "NE = -- (no scored row carries NE)")
    # The route term; absent on a row that never flew.
    spl = [float(m["spl_pos"]) for m in scored.values()
           if isinstance(m.get("spl_pos"), (int, float))]
    print(f"mean spl_pos = {sum(spl) / len(spl):.4f} (n={len(spl)})" if spl
          else "mean spl_pos = -- (no scored row carries spl_pos)")
    # SAHI tiles the segmenter could not answer left those frames partial-blind.
    tiles = [int(m.get("sam_tiles_failed") or 0) for m in scored.values()]
    if any(tiles):
        print(f"SAHI tiles failed = {sum(tiles)} over {sum(1 for t in tiles if t)} "
              f"of {n} scored episode(s)")
    shaped = print_precise(list(scored.values()), n)
    if _criterion is None:
        # Unstamped rows: neither reading is the headline, and the note names
        # the shape line only when one printed.
        print("  ^ these rows carry no criterion stamp: the SR above is the "
              "20 m disc"
              + (", and the shape line beside it is the other reading. Which "
                 "one this bench is scored on is NOT recorded"
                 if shaped else
                 ", and the run carries no shape verdict at all. Whether this "
                 "bench is scored on one is NOT recorded")
              + " -- quote neither as the bench's SR.")
    # OSR and the (OSR, SR) joint separate reaching the target from committing
    # once there; SR without OSR means mismatched geometry. Ring rows only.
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

    # Among scored fails: was the committed xy itself within the success radius?
    fails = [m for m in scored.values() if not m.get("SR")]
    commit_in = sum(1 for m in fails if m.get("commit_would_score") is True)
    commit_out = sum(1 for m in fails if m.get("commit_would_score") is False)
    unk = len(fails) - commit_in - commit_out
    print(
        f"fails={len(fails)}: commit_within_R={commit_in}  "
        f"commit_beyond_R={commit_out}  (no commit_xy={unk})"
    )
    return scored


_BUCKETS = ("car", "building", "other", "?")


def _label(name: str, suc: int, tot: int) -> str:
    return f"{name}:{suc}/{tot}({100 * suc / tot:.0f}%)" if tot else f"{name}:0"


def main(argv: list[str]) -> int:
    allow_partial = "--allow-partial" in argv[1:]
    args = [Path(a) for a in argv[1:] if a != "--allow-partial"]
    if not args:
        print(__doc__)
        return 2
    for root in args:
        report(root, allow_partial=allow_partial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
