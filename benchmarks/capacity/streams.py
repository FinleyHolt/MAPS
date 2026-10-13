"""Seeded Poisson report streams over the reviewed Cal City episodes.

One stream is one capacity session: reports arrive at exponential gaps of
mean ``3600 / load_per_h`` seconds for ``hours``, each a reviewed episode
drawn without replacement. One ``(load_per_h, hours, seed)`` always gives
one stream, so an alone session and an assisted one face the same reports::

    python -m benchmarks.capacity.streams --load 30 --hours 1 --seed 1 --out stream.json
    python -m benchmarks.calcity.bench_fullset --workers 4 --arrivals stream.json --out-root <root>
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def stream(load_per_h: float, hours: float, seed: int, episode_ids: list[str] | None = None) -> dict:
    """The session's stream: ``{"load_per_h", "hours", "seed", "reports"}``,
    each report ``{"report", "arrival_s", "episode_id"}`` in arrival order.
    ``episode_ids`` defaults to the reviewed set; more arrivals than it holds is refused."""
    if load_per_h <= 0 or hours <= 0:
        raise ValueError(f"a stream needs a positive load and length, got {load_per_h}/h for {hours} h")
    if episode_ids is None:
        from benchmarks.calcity.data_prep.reviewed import reviewed_episode_ids  # noqa: PLC0415

        episode_ids = reviewed_episode_ids()
    rng = random.Random(seed)
    pool = list(episode_ids)
    rng.shuffle(pool)
    arrivals: list[float] = []
    t = rng.expovariate(load_per_h / 3600.0)
    while t < hours * 3600.0:
        arrivals.append(t)
        t += rng.expovariate(load_per_h / 3600.0)
    if len(arrivals) > len(pool):
        raise ValueError(f"{len(arrivals)} arrivals at {load_per_h}/h for {hours} h outrun the "
                         f"{len(pool)} episodes; a report is never flown twice in a session")
    return {"load_per_h": float(load_per_h), "hours": float(hours), "seed": int(seed),
            "reports": [{"report": k, "arrival_s": round(a, 3), "episode_id": pool[k]}
                        for k, a in enumerate(arrivals)]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--load", type=float, required=True, help="offered load, reports per hour")
    ap.add_argument("--hours", type=float, required=True, help="session length")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    s = stream(args.load, args.hours, args.seed)
    args.out.write_text(json.dumps(s, indent=1) + "\n")
    print(f"{len(s['reports'])} reports at {args.load:g}/h over {args.hours:g} h (seed {args.seed}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
