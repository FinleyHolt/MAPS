"""SR report for a run root, over the ``metrics`` of the attempts ``run_record.rows`` picks.

Usage:
    python -m benchmarks.citynav.analysis.run_report output/run_a [output/run_b ...]

A root that scores fewer episodes than its manifest selected is refused
unless ``--allow-partial`` (``rows.load_attempts``).
"""

from __future__ import annotations

import argparse
import functools
import os
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import yaml

from benchmarks import REPO_ROOT
from benchmarks.bench import Bench, bake_sha256
from benchmarks.calcity.paths import calcity_bench
from benchmarks.citynav.adapter import find_row, load_split, parse_episode
from benchmarks.citynav.paths import citynav_bench
from benchmarks.run_record import rows as _rows
from benchmarks.run_record.rows import read_manifest


def benches(environ: Mapping[str, str], *, first: Bench | None = None) -> tuple[Bench, ...]:
    """The benches a report reads runs of, as ``environ`` places them: the
    first is the one whose bake a run that records no tiles directory is
    checked against (``first``, else CityNav); each split is read from the
    bench that names it."""
    both = (citynav_bench(environ), calcity_bench(environ))
    return both if first is None else (first, *(b for b in both if b.name != first.name))


def load_run(root: Path, *, allow_partial: bool = False) -> dict[str, dict]:
    """episode_id -> scored metrics for the run at ``root``."""
    picked, _ = _rows.load_attempts(root, allow_partial=allow_partial)
    return _rows.scored_metrics(picked)


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.1f}%)" if den else f"{num}/0 (--)"


def configured_radius(root: Path) -> float | None:
    """The run's configured ``success_radius_m`` (manifest ``config_yaml``), or None."""
    return (yaml.safe_load(read_manifest(root).get("config_yaml") or "{}") or {}).get(
        "success_radius_m"
    )


def criterion_label(criterion: str | None, root: Path) -> str:
    """How a criterion prints; ``None`` is a row without the stamp. The disc's
    radius is the run's configured one."""
    if criterion == "ring":
        radius = configured_radius(root)
        return "the disc of unrecorded radius" if radius is None else f"the {radius:g} m disc"
    return {"shape": "the target's own shape", None: "UNRECORDED"}.get(criterion, criterion)


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
        m for m in metrics if m.get("precise_target") is not None and m.get("SR") is not None
    ]
    if not pr_scored:
        return False
    pr_sr = sum(1 for m in pr_scored if m.get("SR"))
    print(f"by target shape = {_pct(pr_sr, len(pr_scored))}  (coverage {_pct(len(pr_scored), n)})")
    parts = []
    for kind, label in (("shape", "by shape"), ("radius", "by radius")):
        sub = [m for m in pr_scored if m.get("precise_target") == kind]
        if sub:
            hit = sum(1 for m in sub if m.get("SR"))
            parts.append(f"{label} {_pct(hit, len(sub))}")
    if parts:
        print("  " + "   ".join(parts))
    return True


def _local_bake_dir(resolved: str | None) -> Path | None:
    """This machine's copy of the recorded ``tiles_dir_resolved``: the path
    itself, else its longest tail that is a directory under this repo
    (``/srv/<user>/MAPS/benchmarks/calcity/tiles`` -> ``benchmarks/calcity/tiles``)."""
    if not resolved:
        return None
    if Path(resolved).is_dir():
        return Path(resolved)
    parts = PurePosixPath(resolved).parts
    for i in range(1, len(parts) - 1):  # a tail of two parts at least: "tiles" alone names nothing
        cand = REPO_ROOT.joinpath(*parts[i:])
        if cand.is_dir():
            return cand
    return None


def bake_line(root: Path, default_tiles: Path) -> str:
    """One line: does the run's ``bake_sha256`` match the content of the bake
    on disk (``bench.bake_sha256``)? Compared against the tiles dir the run
    recorded, else ``default_tiles``. Never raises; missing tiles or
    stamp read as unknown or unverified, never as a pass."""
    perception = read_manifest(root).get("perception") or {}
    stamped = perception.get("bake_sha256")
    recorded = perception.get("tiles_dir_resolved")

    if not stamped:
        return "bake: UNVERIFIED (run stamps no bake_sha256)"
    local = _local_bake_dir(recorded) if recorded else Path(default_tiles)
    if local is None:
        return (
            f"bake: UNKNOWN (the run's bake {recorded} is not on this machine)  run={stamped[:12]}"
        )
    here = bake_sha256(local)
    if here is None:
        return f"bake: UNKNOWN (no bake at {local})  run={stamped[:12]}"
    if here == stamped:
        return f"bake: verified {stamped[:12]} ({local})"
    return (
        f"bake: *** MISMATCH *** run={stamped[:12]} local={here[:12]} at {local}. "
        f"The SR below was measured against a DIFFERENT bake of the same name; "
        f"it is not comparable to a number measured against this one."
    )


def report(root: Path, known: Sequence[Bench], *, allow_partial: bool = False) -> dict[str, dict]:
    """Print the report of the run at ``root``; ``known`` is :func:`benches`."""
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
    print(bake_line(root, known[0].tiles_dir))
    # ``rows`` counts attempts, ``scored`` episodes.
    print(
        f"scored={n}  attempts={stats.rows}  "
        f"replicates_discarded={stats.keyed - len(picked)}  "
        f"no_metrics(crash/timeout-no-record)={stats.no_metrics}  "
        f"timeouts(wall-clock)={stats.timed_out}  "
        f"infra(excluded)={stats.infra}"
    )
    print(f"criterion: {criterion_label(_criterion, root)}")
    print(f"SR = {_pct(sr, n)}")
    # Metres from the final pose to the target centroid, on every row with metrics.
    ne = [float(m["NE"]) for m in scored.values() if isinstance(m.get("NE"), (int, float))]
    print(
        f"NE median = {statistics.median(ne):.1f} m  mean = {sum(ne) / len(ne):.1f} m (n={len(ne)})"
        if ne
        else "NE = -- (no scored row carries NE)"
    )
    # The route term: 0 on a row that never flew, absent on a shape-scored row.
    spl = [
        float(m["spl_pos"]) for m in scored.values() if isinstance(m.get("spl_pos"), (int, float))
    ]
    if spl:
        print(f"mean spl_pos = {sum(spl) / len(spl):.4f} (n={len(spl)})")
    # SAHI tiles the segmenter could not answer left those frames partial-blind.
    tiles = [int(m.get("sam_tiles_failed") or 0) for m in scored.values()]
    if any(tiles):
        print(
            f"SAHI tiles failed = {sum(tiles)} over {sum(1 for t in tiles if t)} "
            f"of {n} scored episode(s)"
        )
    shaped = print_precise(list(scored.values()), n)
    if _criterion is None and any(a.metrics for a in picked.values()):
        # Unstamped recorded rows: neither reading is the headline, and the
        # note names the shape line only when one printed.
        print(
            "  ^ these rows carry no criterion stamp: the SR above is "
            + criterion_label("ring", root)
            + (
                ", and the shape line beside it is the other reading. Which "
                "one this bench is scored on is NOT recorded"
                if shaped
                else ", and the run carries no shape verdict at all. Whether this "
                "bench is scored on one is NOT recorded"
            )
            + " -- quote neither as the bench's SR."
        )
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

    print_classes(root, scored, picked, known)

    # Commit provenance (prior landmark, runtime detection, world point, or none).
    prov = Counter(m.get("committed_provenance") or "none" for m in scored.values())
    print(
        "provenance: "
        + "  ".join(f"{k}={v}" for k, v in sorted(prov.items(), key=lambda kv: (-kv[1], kv[0])))
    )
    return scored


_BUCKETS = ("car", "building", "other")


# ponytail: a split file is read once per process (the bench reports after every landed
# episode); one edited on disk is read again by a new process only.
_read_split = functools.lru_cache(maxsize=4)(load_split)


def _split_rows(split: str | None, known: Sequence[Bench]) -> list[dict] | None:
    """The rows of the split a run names, read from the bench of ``known`` that
    names it, else ``None`` when no bench does or its file is not on this machine."""
    bench = next((b for b in known if split in b.split_files), None)
    try:
        return None if bench is None else _read_split(bench.split_path(split))
    except (OSError, ValueError):
        return None


def print_classes(
    root: Path, scored: dict[str, dict], picked: dict, known: Sequence[Bench]
) -> None:
    """SR by referent bucket and, where the corpus grades it, by intel quality
    (Cal City, as the flight measures' by_intel_quality). A recorded row's
    class is its own ``referent_bucket`` and ``intel_quality``; a pick without
    them takes its corpus row's, read by episode id from the run's split (its
    manifest's, else the one its rows record), so it counts in its own class;
    a row that records its own description is bucketed by that wording.
    A split on this machine that lacks such a pick is not the run's corpus: no
    class line."""
    # Imported here: the episode runner brings the mission stack, which a report of recorded classes never needs.
    from benchmarks.citynav.run_benchmark.episode_runner import classify_referent

    classes = {
        eid: (m["referent_bucket"], m.get("intel_quality"))
        for eid, m in scored.items()
        if "referent_bucket" in m
    }
    unrecorded = [eid for eid in scored if eid not in classes]
    if unrecorded:
        split = read_manifest(root).get("split") or next(
            (a.row["split"] for a in picked.values() if a.row.get("split")), None
        )
        corpus = _split_rows(split, known) or []
        found = {eid: find_row(corpus, eid) for eid in unrecorded}
        missing = sum(f is None for f in found.values())
        if missing:
            where = (
                f"are not in its split {split} on this machine"
                if split
                else "carry no class and no split recorded"
            )
            print(f"by class: -- ({missing} of the run's {len(scored)} episodes {where})")
            return
        for eid, f in found.items():
            episode = parse_episode(f[1])
            said = picked[eid].row.get("description") or episode.description
            classes[eid] = (classify_referent(said), episode.intel_quality)
    for kind, index, order in (("bucket", 0, _BUCKETS), ("intel quality", 1, ("G", "M", "P", "?"))):
        tot: Counter = Counter()
        suc: Counter = Counter()
        for eid, m in scored.items():
            cls = classes[eid][index] or "?"
            tot[cls] += 1
            suc[cls] += bool(m.get("SR"))
        if kind == "bucket" or set(tot) != {"?"}:
            print(f"by {kind}: " + "   ".join(_label(k, suc[k], tot[k]) for k in order if tot[k]))


def _label(name: str, suc: int, tot: int) -> str:
    return f"{name}:{suc}/{tot}({100 * suc / tot:.0f}%)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", type=Path, nargs="+", help="run roots")
    _rows.add_allow_partial(ap)
    args = ap.parse_args(argv)
    known = benches(os.environ)
    rc = 0
    for root in args.roots:
        try:
            report(root, known, allow_partial=args.allow_partial)
        except SystemExit as refused:  # a root it cannot report; the next is still read
            print(refused, file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
