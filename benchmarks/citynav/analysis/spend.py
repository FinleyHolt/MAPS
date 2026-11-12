"""What a run root has spent at a hosted endpoint, and what the rest of its
selection will cost at that rate, read live off every attempt's transcript.

    python -m benchmarks.citynav.analysis.spend <out-root> [--price-in 5 ...]

Every attempt directory (``w*/maps_*``) is metered with
``maps.llm.flight_summary.summarize``, the one transcript reader, so an
attempt still flying and one the run re-queued as infra are both counted:
the endpoint billed them. Prompt tokens are priced in three parts: the
cache reads (``cached_tokens``), the cache writes (``cache_write_tokens``)
and the rest; a serve that reports no cache meter prices every prompt token
as uncached, and its cache totals and hit rate read null. The projection is the mean cost of a scored episode times the
selected episodes not yet scored (``run_manifest.json`` ``selection``).
Every ``served_model`` the attempts were answered by is listed: more than
one is a deployment that changed model version mid-run.

The default prices are GPT-5.6 Sol's Global Standard short-context rates in
USD per million tokens as third parties report them (Azure's own page lists
none yet); pass the invoice's.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from benchmarks.run_record import rows as _rows
from benchmarks.run_record.episodes import EPISODES_JSONL, MANIFEST_NAME
from maps.llm.flight_summary import summarize
from maps.settings import from_environ

#: USD per million tokens: input, cache read, cache write, output.
PRICES = {"in": 5.0, "cached": 0.5, "write": 6.25, "out": 30.0}


def attempt_cost(llm: dict, prices: dict = PRICES) -> float:
    """One attempt's USD from its ``flight_summary`` block. Cache writes and
    reads are taken as slices of ``prompt_tokens``, disjoint."""
    cached = llm.get("cached_tokens") or 0
    write = llm.get("cache_write_tokens") or 0
    plain = max(0, llm["prompt_tokens"] - cached - write)
    return (plain * prices["in"] + cached * prices["cached"] + write * prices["write"]
            + llm["completion_tokens"] * prices["out"]) / 1e6


def spend(out_root: Path, prices: dict = PRICES, tokenizer_name: str = "") -> dict:
    """The root's spend: tokens and USD over every attempt, the scored
    episodes' mean, and the projection for the selection's remainder.
    ``tokenizer_name`` is the reader's tokenizer override
    (``SEARCH_COMPACT_TOKENIZER_NAME``), for a reasoning span no serve counted."""
    out_root = Path(out_root)
    ledger = {r.get("workdir"): r for r in _rows.iter_json_rows(out_root / EPISODES_JSONL)}
    # A cache meter stays None until an attempt reports it.
    totals: dict = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
                    "cached_tokens": None, "cache_write_tokens": None}
    usd = {"scored": 0.0, "infra": 0.0, "flying": 0.0}
    served: Counter = Counter()
    scored_ids: set[str] = set()
    for attempt in sorted(out_root.glob("w*/maps_*")):
        llm = summarize(attempt / "flight", tokenizer_name)
        for k in totals:
            if llm[k] is not None:
                totals[k] = (totals[k] or 0) + llm[k]
        if llm["served_model"]:
            served[llm["served_model"]] += 1
        row = ledger.get(str(attempt.relative_to(out_root)))
        state = "flying" if row is None else "infra" if _rows.is_infra_abort(row) else "scored"
        usd[state] += attempt_cost(llm, prices)
        if state == "scored" and isinstance(row.get("episode_id"), str):
            scored_ids.add(row["episode_id"])
    named = [w for w in ledger if w]
    if named and not any((out_root / str(w)).is_dir() for w in named):
        raise SystemExit(f"{out_root}: the ledger names {len(named)} attempt(s) but none of their "
                         f"workdirs is here; spend reads each attempt's transcript")
    try:
        selected = json.loads((out_root / MANIFEST_NAME).read_text())["selection"]["episodes_n"]
    except (OSError, ValueError, KeyError):
        selected = None
    per_episode = usd["scored"] / len(scored_ids) if scored_ids else None
    remaining = None if selected is None else max(0, selected - len(scored_ids))
    return {
        "tokens": totals,
        "cache_hit": (totals["cached_tokens"] / totals["prompt_tokens"]
                      if totals["cached_tokens"] is not None and totals["prompt_tokens"] else None),
        "usd": {k: round(v, 2) for k, v in usd.items()},
        "usd_total": round(sum(usd.values()), 2),
        "scored_episodes": len(scored_ids),
        "usd_per_scored_episode": None if per_episode is None else round(per_episode, 3),
        "selected": selected,
        "usd_projected_remaining": (None if per_episode is None or remaining is None
                                    else round(per_episode * remaining, 2)),
        "served_models": dict(served),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_root", type=Path)
    for key, value in PRICES.items():
        ap.add_argument(f"--price-{key}", type=float, default=value,
                        help=f"USD per million tokens (default {value})")
    args = ap.parse_args(argv)
    prices = {k: getattr(args, f"price_{k}") for k in PRICES}
    print(json.dumps(spend(args.out_root, prices, from_environ(os.environ).endpoint.tokenizer_name), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
