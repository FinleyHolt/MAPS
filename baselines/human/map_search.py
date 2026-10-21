"""The map-search arm: a person alone finds each episode's target on the whole map.

The participant gets every episode on their list at once and works them in
any order. Each opens on its instruction and the block's imagery, with the
bench's OSM prior (the one MAPS reads) to toggle on and to search by name.
They commit one point per episode, and may commit again until they leave it;
the last commit stands. Nothing they see carries the target or a score.
Unlike the operator arm (``baselines.human.policy``), nothing is flown and no
budget is spent: this is the person with the map, the ceiling a searcher
reading the same pixels reaches.

The session is the queue and the ledger; a front end draws it (the FTE
station's study mode, ``--study search``). ``team_rows`` scores the ledger:
each committed point by ``eval.score_arrival`` as the MAPS row's own episode.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from baselines.human.session_plan import scored_rows
from benchmarks.citynav.analysis.rows import iter_json_rows


class Search:
    """A participant's solo list over the whole map: every instruction up front, a point each."""

    def __init__(self, maps_root: Path, episode_ids: list[str], out: Path) -> None:
        rows = scored_rows(maps_root)
        missing = [e for e in episode_ids if e not in rows]
        if missing:
            raise SystemExit(f"FATAL: {len(missing)} allowlisted episodes have no scored MAPS row, e.g. {missing[:3]}")
        # The row's instruction and block, never its target or metrics.
        self.episodes = [{"episode_id": e, "instruction": str(rows[e]["description"]), "map_name": str(rows[e]["map_name"])}
                         for e in episode_ids]
        self.map_name = self.episodes[0]["map_name"] if self.episodes else None
        self.out, self.decisions = out, out / "decisions.jsonl"
        self.committed = {d["episode_id"]: d for d in iter_json_rows(self.decisions)} if self.decisions.is_file() else {}
        self._opened: dict[str, float] = {}
        self._opened_at: dict[str, str] = {}

    def state(self) -> dict:
        return {"total": len(self.episodes), "done_n": sum(e["episode_id"] in self.committed for e in self.episodes),
                "episodes": [{**e, "committed": self.committed[e["episode_id"]]["xy"] if e["episode_id"] in self.committed else None}
                             for e in self.episodes]}

    def open(self, episode_id: str) -> dict:
        """Start an episode's clock, once: its search time runs from when it was first opened."""
        if episode_id not in {e["episode_id"] for e in self.episodes}:
            return {"ok": False, "error": f"{episode_id} is not on this list"}
        self._opened.setdefault(episode_id, time.monotonic())
        self._opened_at.setdefault(episode_id, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        return {"ok": True}

    def commit(self, episode_id: str, xy: tuple[float, float]) -> dict:
        if episode_id not in {e["episode_id"] for e in self.episodes}:
            return {"ok": False, "error": f"{episode_id} is not on this list"}
        opened = self._opened.setdefault(episode_id, time.monotonic())
        opened_at = self._opened_at.setdefault(episode_id, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        decision = {"episode_id": episode_id, "decision": "commit", "xy": [float(xy[0]), float(xy[1])],
                    "search_s": round(time.monotonic() - opened, 1), "opened_at": opened_at, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.out.mkdir(parents=True, exist_ok=True)
        with self.decisions.open("a") as fh:
            fh.write(json.dumps(decision) + "\n")
        self.committed[episode_id] = decision
        return {"ok": True}
