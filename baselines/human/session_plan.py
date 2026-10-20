"""Each participant's episode lists for the solo and team arms, dealt so nobody can carry an answer.

A participant who found a target alone must never audit it, or audit
anything near it: the second look would score memory, not judgement. So the
blocks of the paired MAPS root are split into two pools, and a participant
flies one pool solo and audits the other, never the same block in both arms.
Within an arm a participant sees each target object at most once (CityNav
writes several descriptions per object, ``<map>::<object>#<description>``).
Participants alternate which pool is solo and which arm comes first, so pool
difficulty and practice cancel across pairs of participants.

    python -m baselines.human.session_plan --maps-root output/e2/<root> \\
        --participants p01,p02,p03,p04 --per-arm 20 --out output/human_study/<study>

writes ``<out>/<participant>/{solo,team}.txt`` (the bench's and the audit
console's ``--episode-allowlist``) and ``<out>/plan.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from benchmarks.citynav.analysis.rows import iter_json_rows


def target_key(episode_id: str) -> str:
    """The target object an episode describes: its id without the description index."""
    return episode_id.split("#", 1)[0]


def scored_rows(maps_root: Path) -> dict[str, dict]:
    """The last ``ok`` row per episode in a root's ledger: the attempts an auditor can review."""
    rows: dict[str, dict] = {}
    for row in iter_json_rows(maps_root / "episodes.jsonl"):
        if row.get("status") == "ok" and row.get("workdir"):
            rows[str(row["episode_id"])] = row
    return rows


def split_blocks(rows: dict[str, dict], seed: int) -> tuple[list[str], list[str]]:
    """The blocks in two pools of near-equal episode count, in a seeded order."""
    counts: dict[str, int] = {}
    for row in rows.values():
        counts[row["map_name"]] = counts.get(row["map_name"], 0) + 1
    order = sorted(counts, key=lambda b: hashlib.sha256(f"{seed}:{b}".encode()).hexdigest())
    pools: tuple[list[str], list[str]] = ([], [])
    sizes = [0, 0]
    for block in order:
        i = 0 if sizes[0] <= sizes[1] else 1
        pools[i].append(block)
        sizes[i] += counts[block]
    return pools


def deal(rows: dict[str, dict], blocks: list[str], n: int, rng: random.Random) -> list[str]:
    """``n`` episodes on ``blocks``, one per target object, in a shuffled order."""
    by_target: dict[str, list[str]] = {}
    for eid, row in sorted(rows.items()):
        if row["map_name"] in blocks:
            by_target.setdefault(target_key(eid), []).append(eid)
    targets = sorted(by_target)
    if len(targets) < n:
        raise SystemExit(f"FATAL: the pool holds {len(targets)} target objects, fewer than --per-arm {n}")
    return [rng.choice(by_target[t]) for t in rng.sample(targets, n)]


def plan(rows: dict[str, dict], participants: list[str], per_arm: int, seed: int) -> dict:
    """Each participant's arms: which pool is solo, which arm comes first, and the episodes."""
    pools = split_blocks(rows, seed)
    out: dict = {"seed": seed, "per_arm": per_arm, "pools": [sorted(p) for p in pools], "participants": {}}
    for i, p in enumerate(participants):
        rng = random.Random(f"{seed}:{p}")
        solo_pool = i % 2
        out["participants"][p] = {
            "first": "solo" if (i // 2) % 2 == 0 else "team",
            "solo_pool": solo_pool,
            "solo": deal(rows, pools[solo_pool], per_arm, rng),
            "team": deal(rows, pools[1 - solo_pool], per_arm, rng),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--maps-root", type=Path, required=True, help="the MAPS root the team arm audits")
    ap.add_argument("--participants", required=True, help="comma-separated labels, never names (p01,p02,...)")
    ap.add_argument("--per-arm", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    labels = [p.strip() for p in args.participants.split(",") if p.strip()]
    result = plan(scored_rows(args.maps_root), labels, args.per_arm, args.seed)
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
