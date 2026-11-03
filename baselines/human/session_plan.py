"""Each participant's episode lists for the solo and team arms, dealt so nobody can carry an answer.

A participant who found a target alone must never audit it, or audit
anything near it: the second look would score memory, not judgement. So the
ground of the paired MAPS root is split into two pools, and a participant
flies one pool solo and audits the other, never the same ground in both arms.
A pool unit is a block (CityNav), or with ``--region-m`` a square region of
one block keyed by where the target lies (Cal City's single 13 km block).
Within an arm a participant sees each target object at most once (CityNav
writes several descriptions per object, ``<map>::<object>#<description>``).
Participants alternate which pool is solo and which arm comes first, so pool
difficulty and practice cancel across pairs of participants.

    python -m baselines.human.session_plan --maps-root output/e3/<root> --region-m 1000 \\
        --participants p01,p02,p03,p04 --per-arm 20 --out output/human_study/<study>

writes ``<out>/<participant>/{solo,team}.txt`` (the bench's and the audit
console's ``--episode-allowlist``) and ``<out>/plan.json``.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import random
from pathlib import Path

from benchmarks.citynav.analysis.rows import iter_json_rows


def target_key(episode_id: str) -> str:
    """The target object an episode describes: its id without the description index."""
    return episode_id.split("#", 1)[0]


def scored_rows(maps_root: Path) -> dict[str, dict]:
    """The last ``ok`` row per episode in a root's ledger: the attempts an auditor can review.
    Read once per version of the ledger and shared, so callers read the rows and never write them:
    every session the study server opens on a root would otherwise decode it again."""
    ledger = (maps_root / "episodes.jsonl").resolve()
    stat = ledger.stat() if ledger.is_file() else None
    return _scored_rows(ledger, stat and (stat.st_mtime_ns, stat.st_size))


@functools.lru_cache(maxsize=4)
def _scored_rows(ledger: Path, version: tuple[int, int] | None) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in iter_json_rows(ledger):
        if row.get("status") == "ok" and row.get("workdir"):
            rows[str(row["episode_id"])] = row
    return rows


def ground_unit(row: dict, region_m: float | None) -> str:
    """The pool unit an episode belongs to: its block, or its target's region of the block."""
    if not region_m:
        return str(row["map_name"])
    x, y = (float(v) for v in row["target_xy"])
    return f"{row['map_name']}@{int(x // region_m)},{int(y // region_m)}"


def split_ground(rows: dict[str, dict], seed: int, region_m: float | None = None) -> tuple[list[str], list[str]]:
    """The pool units in two pools of near-equal episode count, in a seeded order."""
    counts: dict[str, int] = {}
    for row in rows.values():
        u = ground_unit(row, region_m)
        counts[u] = counts.get(u, 0) + 1
    order = sorted(counts, key=lambda u: hashlib.sha256(f"{seed}:{u}".encode()).hexdigest())
    pools: tuple[list[str], list[str]] = ([], [])
    sizes = [0, 0]
    for unit in order:
        i = 0 if sizes[0] <= sizes[1] else 1
        pools[i].append(unit)
        sizes[i] += counts[unit]
    return pools


def deal(rows: dict[str, dict], units: list[str], n: int, rng: random.Random, region_m: float | None = None) -> list[str]:
    """``n`` episodes on ``units``, one per target object, in a shuffled order."""
    by_target: dict[str, list[str]] = {}
    for eid, row in sorted(rows.items()):
        if ground_unit(row, region_m) in units:
            by_target.setdefault(target_key(eid), []).append(eid)
    targets = sorted(by_target)
    if len(targets) < n:
        raise SystemExit(f"FATAL: the pool holds {len(targets)} target objects, fewer than --per-arm {n}")
    return [rng.choice(by_target[t]) for t in rng.sample(targets, n)]


def plan(rows: dict[str, dict], participants: list[str], per_arm: int, seed: int, region_m: float | None = None) -> dict:
    """Each participant's arms: which pool is solo, which arm comes first, and the episodes."""
    pools = split_ground(rows, seed, region_m)
    out: dict = {"seed": seed, "per_arm": per_arm, "region_m": region_m, "pools": [sorted(p) for p in pools],
                 "participants": {}}
    for i, p in enumerate(participants):
        rng = random.Random(f"{seed}:{p}")
        solo_pool = i % 2
        out["participants"][p] = {
            "first": "solo" if (i // 2) % 2 == 0 else "team",
            "solo_pool": solo_pool,
            "solo": deal(rows, pools[solo_pool], per_arm, rng, region_m),
            "team": deal(rows, pools[1 - solo_pool], per_arm, rng, region_m),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--maps-root", type=Path, required=True, help="the MAPS root the team arm audits")
    ap.add_argument("--participants", required=True, help="comma-separated labels, never names (p01,p02,...)")
    ap.add_argument("--per-arm", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--region-m", type=float, default=None,
                    help="pool square regions of this side instead of whole blocks (a one-block bench)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    labels = [p.strip() for p in args.participants.split(",") if p.strip()]
    result = plan(scored_rows(args.maps_root), labels, args.per_arm, args.seed, args.region_m)
    result["maps_root"] = str(args.maps_root.resolve())
    for p, arms in result["participants"].items():
        (args.out / p).mkdir(parents=True, exist_ok=True)
        for arm in ("solo", "team"):
            (args.out / p / f"{arm}.txt").write_text("\n".join(arms[arm]) + "\n")
    (args.out / "plan.json").write_text(json.dumps(result, indent=1))
    for p, arms in result["participants"].items():
        print(f"{p}: {arms['first']} first, solo on pool {arms['solo_pool']}, {args.per_arm} episodes per arm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#: Reports a reviewer flagged or answered wrong after the study's root was flown. The plan deals
#: none of them and a scored root carries none of them; the file says where each verdict came from.
OMITTED_REPORTS = Path(__file__).with_name("omitted_reports.json")


def omitted_reports(path: Path = OMITTED_REPORTS) -> dict[str, str]:
    """Episode id -> the reviewer's reason, for every report the study leaves out."""
    if not path.is_file():
        return {}
    return {e["episode_id"]: e["reason"] for e in json.loads(path.read_text())["episodes"]}

