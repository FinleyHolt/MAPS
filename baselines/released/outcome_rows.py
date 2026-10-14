"""A released baseline's native outputs as ledger rows the analysis readers take.

Upstream keys an episode ``(map_name, object_id, desc_id)``; this harness keys
it ``<map>::<ann>#<row>`` against the split file (``adapter.parse_episode``).
The converter maps one to the other through the split file that defines the
harness ids, so a released checkpoint's outcome pairs against a MAPS run on
the same episode (``outcome_rows pair <released> <maps>``). Every row is
scored by ``benchmarks.citynav.eval.score_arrival``: the ring on CityNav, and
on Cal City the shape criterion ``benchmarks.calcity.use`` selects, so a Cal
City number is on the Cal City contract rather than upstream's 20 m disc.

Written per run directory:

  ``episodes.jsonl``     one ``status: ok`` row per episode upstream scored
  ``episode_sets.json``  the ids upstream evaluated, and every id it dropped
                         or the export excluded, with the reason, so a reader
                         can score on the unfiltered set (a drop is a miss)

Native inputs: MGP's ``<model>_<ckpt>_<split>_*.json`` (``trajectory_logs``),
or a capture written by ``upstream_eval/launch.py`` (HETT, FlightGPT).

Usage::

    python -m baselines.released.outcome_rows citynav --method mgp \\
        --native mgp_mturk_test_unseen_0.75_gdino.json --split test_unseen \\
        --split-file citynav_test_unseen.json --dataset-version original \\
        --checkpoint-id "$(cat checkpoint.id)" --out output/released/mgp_test_unseen
    python -m baselines.released.outcome_rows calcity --method hett \\
        --native captures/hett_test_unseen.json --export EXPORT_DIR \\
        --checkpoint-id "$(cat checkpoint.id)" --out output/released/hett_calcity
    python -m baselines.released.outcome_rows pair output/released/mgp_test_unseen MAPS_RUN
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

from baselines.released.upstream import PROTOCOL, RELEASES

Key = tuple[str, int, int]


def upstream_key(row: dict) -> Key:
    """The ``Episode.id`` upstream's loader gives this split row."""
    return (f"{row['area']}_block_{row['block']}", int(row["object_ids"][0]), int(row["ann_ids"][0]))


def read_native(path: Path) -> tuple[dict[Key, list[tuple[float, float]]], list[Key]]:
    """``({key: flown xy path}, [errored keys])`` from one native output."""
    data = json.loads(path.read_text())
    if "trajectory_logs" in data:  # MGP's own result file
        trajs = {
            tuple(ast.literal_eval(k)): [(float(p[0]), float(p[1])) for p in poses]
            for k, poses in data["trajectory_logs"].items()
        }
        return trajs, []
    trajs = {
        tuple(e["key"]): [(float(x), float(y)) for x, y in e["trajectory"]]
        for e in data["episodes"]
    }
    return trajs, [tuple(k) for k in data.get("errors", [])]


def read_natives(paths: list[Path]):
    trajs: dict[Key, list] = {}
    errors: list[Key] = []
    for path in paths:
        t, e = read_native(path)
        trajs.update(t)
        errors.extend(e)
    return trajs, errors


def stamp(method: str, checkpoint_id: str, dataset_version: str, split: str) -> dict:
    """The identity every row of one run carries, top level and in provenance."""
    from benchmarks import _git_provenance

    release = RELEASES[method]
    ident = {
        "method": method,
        "upstream_repo": release.repo,
        "upstream_sha": release.sha,
        "checkpoint": checkpoint_id,
        "protocol": PROTOCOL,
        "dataset_version": dataset_version,
    }
    provenance = {
        **ident,
        "git_commit": _git_provenance().get("git_commit"),
        "split": split,
        "served_model": f"{method}:{checkpoint_id}",
        # No tile bake: a released checkpoint reads its own rasters, so
        # ab_report's bake check refuses a stored-column pair with it by
        # design; pair through run_report on the shared episodes instead.
        "search_env_flags": {},
    }
    return {**ident, "split": split, "provenance": provenance}


def _row(eid: str, key: Key, path: list, description: str, scores: dict, ident: dict) -> dict:
    return {
        **ident,
        "episode_id": eid,
        "status": "ok",
        "upstream_key": list(key),
        "description": description,
        "visited_xy": [list(p) for p in path],
        "metrics": {"episode_id": eid, **scores},
    }


def citynav_rows(
    trajs: dict[Key, list],
    errors: list[Key],
    id_rows: list[dict],
    loaded_rows: list[dict],
    ident: dict,
) -> tuple[list[dict], dict]:
    """Ring-scored rows plus the evaluated/dropped sets.

    ``id_rows`` is the split file whose row order defines the harness ids;
    ``loaded_rows`` is everything upstream was handed (a difficulty tier, or
    the same file), the denominator for what it dropped.
    """
    from benchmarks.citynav.adapter import parse_episode
    from benchmarks.citynav.eval import score_arrival

    episodes: dict[Key, list] = {}
    for index, row in enumerate(id_rows):
        episodes.setdefault(upstream_key(row), []).append(parse_episode(row, index))
    unknown = set(trajs) - set(episodes)
    if unknown:
        raise ValueError(
            f"{len(unknown)} upstream keys are not in the id split file, e.g. "
            f"{sorted(unknown)[:3]} -- is --split-file the file upstream loaded?"
        )
    rows, evaluated = [], []
    for key, path in trajs.items():
        shared = len(episodes[key])
        for ep in episodes[key]:
            scores = score_arrival(
                criterion="ring", final_pos_xy=path[-1], target_xy=ep.target_xy,
                pos_trajectory=path,
            )
            row = _row(ep.episode_id, key, path, ep.description, scores, ident)
            if shared > 1:
                # Upstream logs one trajectory per key; every row sharing the
                # key inherits it, and says so.
                row["upstream_key_shared"] = shared
            rows.append(row)
            evaluated.append(ep.episode_id)
    errored = set(errors)
    dropped = {}
    for row in loaded_rows:
        key = upstream_key(row)
        if key in trajs or key not in episodes:
            continue
        reason = "upstream_exception" if key in errored else "upstream_loader_filter"
        for ep in episodes[key]:
            dropped[ep.episode_id] = reason
    return rows, {"evaluated": sorted(evaluated), "dropped": dropped}


def calcity_rows(
    trajs: dict[Key, list],
    errors: list[Key],
    manifest: dict,
    shapes: dict[str, dict],
    descriptions: dict[str, str],
    criterion: str,
    ident: dict,
) -> tuple[list[dict], dict]:
    """Rows scored on the Cal City contract, plus the evaluated/dropped sets.
    ``shapes`` is :func:`calcity_export.calcity_shapes`'s; the released
    methods record no commit, so the final pose is the scored point."""
    from benchmarks.citynav.eval import score_arrival

    blocks = manifest["blocks"]
    rows, evaluated = [], []
    for key, path in trajs.items():
        eid = blocks[key[0]]["episode_id"]
        shape = shapes[eid]
        scores = score_arrival(
            criterion=criterion, final_pos_xy=path[-1], target_xy=tuple(shape["target_xy"]),
            pos_trajectory=path, gt_polygon=shape.get("gt_polygon"),
        )
        rows.append(_row(eid, key, path, descriptions.get(eid, ""), scores, ident))
        evaluated.append(eid)
    errored = {k[0] for k in errors}
    scored = set(evaluated)
    dropped = {
        b["episode_id"]: ("upstream_exception" if name in errored else "upstream_loader_filter")
        for name, b in blocks.items()
        if b["episode_id"] not in scored
    }
    return rows, {
        "evaluated": sorted(evaluated),
        "dropped": dropped,
        "excluded_by_export": manifest["excluded"],
    }


def write_run(out: Path, rows: list[dict], sets: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "episodes.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    (out / "episode_sets.json").write_text(json.dumps(sets, indent=1, sort_keys=True))


def pair(released: Path, other: Path, *, allow_partial: bool = False) -> dict:
    """``ab_report.ab_stats`` over the episodes both runs scored, the released
    run as A, with ``only_a`` and ``only_b``, the episodes one run scored and
    the other did not. ``ab_report``'s CLI refuses the pair because these rows
    record no tile bake (a released checkpoint reads its own rasters), so the
    two runs are paired here on episode id and success criterion alone;
    ``allow_partial`` reads a run short of its manifest as it stands."""
    from benchmarks.citynav.analysis import ab_report, run_report

    a, b = (run_report.load_run(r, allow_partial=allow_partial) for r in (released, other))
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("no shared scored episodes between the two runs")
    crit = {run_report.run_criterion([run[e] for e in shared]) for run in (a, b)}
    if len(crit) != 1:
        raise SystemExit(f"the two runs are scored on different criteria {sorted(map(str, crit))}")
    pairs = [(int(bool(a[e].get("SR"))), int(bool(b[e].get("SR")))) for e in shared]
    return {**ab_report.ab_stats(pairs, label=f"{released.name} vs {other.name}"),
            "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a))}


def main(argv: list[str] | None = None) -> int:
    from benchmarks.citynav.analysis import rows as _rows

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="bench", required=True)
    pr = sub.add_parser("pair", help="paired SR of a released run (A) against another run (B)")
    pr.add_argument("released", type=Path)
    pr.add_argument("other", type=Path)
    _rows.add_allow_partial(pr)
    for name in ("citynav", "calcity"):
        p = sub.add_parser(name)
        p.add_argument("--method", choices=sorted(RELEASES), required=True)
        p.add_argument("--native", type=Path, nargs="+", required=True)
        p.add_argument("--checkpoint-id", required=True,
                       help="the identity setup measured, e.g. a sha256 of the weights")
        p.add_argument("--out", type=Path, required=True)
    cn = sub.choices["citynav"]
    cn.add_argument("--split", required=True)
    cn.add_argument("--split-file", type=Path, required=True,
                    help="the split JSON whose row order defines the harness ids")
    cn.add_argument("--loaded", type=Path, nargs="+",
                    help="the split JSONs upstream loaded (default: --split-file)")
    cn.add_argument("--dataset-version", choices=("original", "revised"), required=True)
    sub.choices["calcity"].add_argument("--export", type=Path, required=True,
                                        help="the calcity_export output directory")
    args = ap.parse_args(argv)
    if args.bench == "pair":
        from benchmarks.citynav.analysis.ab_report import render

        st = pair(args.released, args.other, allow_partial=args.allow_partial)
        print(render(st))
        print(f"shared={st['n']}  only_a={st['only_a']}  only_b={st['only_b']}")
        return 0

    trajs, errors = read_natives(args.native)
    if args.bench == "citynav":
        id_rows = json.loads(args.split_file.read_text())
        loaded = [r for f in (args.loaded or [args.split_file]) for r in json.loads(f.read_text())]
        ident = stamp(args.method, args.checkpoint_id, args.dataset_version, args.split)
        rows, sets = citynav_rows(trajs, errors, id_rows, loaded, ident)
    else:
        from baselines.released.calcity_export import calcity_shapes

        shapes = calcity_shapes()  # selects Cal City for the shared paths
        from benchmarks.calcity.paths import split_path
        from benchmarks.citynav import paths as shared_paths
        from benchmarks.citynav.adapter import parse_episode

        split_rows = json.loads(split_path("calcity").read_text())
        episodes = [parse_episode(r, i) for i, r in enumerate(split_rows)]
        descriptions = {ep.episode_id: ep.description for ep in episodes}
        ident = stamp(args.method, args.checkpoint_id, "calcity", "calcity")
        rows, sets = calcity_rows(
            trajs, errors, json.loads((args.export / "manifest.json").read_text()),
            shapes, descriptions, shared_paths.SUCCESS_CRITERION,
            ident,
        )
    write_run(args.out, rows, sets)
    print(f"{len(rows)} rows, {len(sets['dropped'])} dropped -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
