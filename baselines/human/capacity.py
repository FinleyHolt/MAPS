"""The capacity study: how many correct finds a fixed session holds, a person alone and with MAPS.

The backlog is the MAPS root's episodes, split into two halves of equal MAPS SR
(``plan``), one half per condition, the halves and the order of the conditions
counterbalanced in groups of four participants. Each participant works their
half in a fixed shuffled order, always on the earliest open report, for one
fixed-length session per condition. A skip is allowed: alone it scores as a miss, and with
MAPS it leaves MAPS's answer standing (``team_rows``).

With MAPS (``arm = "team"``) its results are not flown live: each is released
when a replay of the root's own recorded timing says it lands, ``agents``
searches at a time working the list in the participant's order
(``release_times``). A report whose result is ready when the person opens it is
verified (confirm or pick another, through the audit session); one that is not
is searched by hand, and a result landing meanwhile does not interrupt it. Alone
(``arm = "solo"``) every report is searched by hand.

Each answer lands in ``<out>/answers.jsonl`` with its source (``confirm``,
``replace``, ``own_search``, ``skip``), when the report was opened and answered
and when MAPS's result was ready, in UTC, and the answer's own instant (``answered_epoch``),
at which the next report opens; ``<out>/session.json`` holds the
session's start, length and list, so a restart keeps the clock. ``team_rows`` scores the
ledger, and with MAPS also every result that landed within the session that the person
never reached: an ``unchecked`` answer, scored as MAPS scored it, since the team's output
is every MAPS result plus the person's corrections and searches.

    python -m baselines.human.capacity plan --maps-root <root> --participants p01,p02,p03,p04 --out <study>
    python -m baselines.human.capacity practice --maps-root <root> --n 4 --out <study>/practice

``practice`` writes a root of the bench's rows the MAPS root never flew, for a participant
to learn the console on, searching by hand; it carries no MAPS result and is never scored.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import random
import time
from datetime import datetime
from pathlib import Path

from baselines.human.audit_console import Session, _jsonl
from baselines.human.session_plan import scored_rows

ARMS = ("solo", "team")
SOURCES = ("confirm", "replace", "own_search", "skip")
#: One condition's session, seconds: the same for every participant. Forty minutes, so a
#: visit (setup, briefing, practice, both conditions and a break) fits in two hours.
SESSION_S = 2400.0
#: MAPS searches running at once in the replay, the concurrency the replayed batch flew at:
#: nearly all of a participant's half lands within the session. ``plan`` records each list's count.
AGENTS = 96


def _utc(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _answered_epoch(answer: dict) -> float:
    """When an answer landed; a ledger written before ``answered_epoch`` has its stamp to the second."""
    if "answered_epoch" in answer:
        return float(answer["answered_epoch"])
    return datetime.fromisoformat(answer["answered_at"]).timestamp()


def halves(rows: dict[str, dict], seed: int) -> tuple[list[str], list[str]]:
    """The episodes in two halves whose MAPS SR differs by at most one episode's worth:
    each SR stratum shuffled and dealt alternately."""
    a: list[str] = []
    b: list[str] = []
    for sr in (1, 0):
        stratum = sorted(e for e, r in rows.items() if int(r["metrics"]["SR"]) == sr)
        random.Random(f"{seed}:{sr}").shuffle(stratum)
        for e in stratum:
            # Alternate on the running count, so the second stratum carries on where the first left off.
            (a if (len(a) + len(b)) % 2 == 0 else b).append(e)
    return sorted(a), sorted(b)


def plan(rows: dict[str, dict], participants: list[str], seed: int, agents: int = AGENTS) -> dict:
    """Each participant's two lists, in their own fixed shuffled order. In every group of
    four, each half meets each condition twice and each condition goes first twice."""
    half = halves(rows, seed)
    out = {"seed": seed, "halves_n": [len(half[0]), len(half[1])],
           "halves_maps_sr": [round(sum(rows[e]["metrics"]["SR"] for e in h) / max(1, len(h)), 4) for h in half],
           "participants": {}}
    for i, p in enumerate(participants):
        solo_half = i % 2
        first = "solo" if (i // 2) % 2 == 0 else "team"
        lists = {}
        for arm, h in (("solo", half[solo_half]), ("team", half[1 - solo_half])):
            order = list(h)
            random.Random(int(hashlib.sha256(f"{seed}:{p}:{arm}".encode()).hexdigest()[:12], 16)).shuffle(order)
            lists[arm] = order
        _starts, ends = release_times([float(rows[e].get("elapsed_s") or 0) for e in lists["team"]], agents)
        out["participants"][p] = {"solo_half": "AB"[solo_half], "team_half": "AB"[1 - solo_half], "first": first,
                                  # How much of MAPS's output this participant's team session holds.
                                  "agents": agents, "maps_ready_in_session": sum(t <= SESSION_S for t in ends),
                                  "lists": lists}
    return out


def release_times(elapsed_s: list[float], agents: int = AGENTS) -> tuple[list[float], list[float]]:
    """When each search starts and lands, seconds from the session's start: ``agents``
    searches at a time, each taking the next report in order as soon as one is free."""
    free = [0.0] * max(1, agents)
    heapq.heapify(free)
    starts, ends = [], []
    for s in elapsed_s:
        t = heapq.heappop(free)
        starts.append(t)
        ends.append(t + float(s))
        heapq.heappush(free, t + float(s))
    return starts, ends


class Capacity:
    """One participant's session in one condition. ``ids`` (the visit's run id and label) ride
    on the session's clock and on every answer."""

    def __init__(self, maps_root: Path, order: list[str], out: Path, arm: str, *, agents: int = AGENTS,
                 session_s: float = SESSION_S, labels: dict | None = None, ids: dict | None = None,
                 clock=time.time) -> None:
        if arm not in ARMS:
            raise ValueError(f"arm {arm!r} is not one of {ARMS}")
        self.rows = scored_rows(maps_root)
        missing = [e for e in order if e not in self.rows]
        if missing:
            raise SystemExit(f"FATAL: {len(missing)} listed episodes have no scored MAPS row, e.g. {missing[:3]}")
        self.maps_root, self.order, self.out, self.arm = maps_root, order, out, arm
        self.labels, self.session_s, self.clock, self.ids = labels, float(session_s), clock, ids or {}
        self.starts, self.ends = (release_times([float(self.rows[e]["elapsed_s"]) for e in order], agents)
                                  if arm == "team" else ([], []))
        self.ledger, meta = out / "answers.jsonl", out / "session.json"
        out.mkdir(parents=True, exist_ok=True)
        if meta.is_file():
            # A resumed session keeps its recorded clock and length.
            session = json.loads(meta.read_text())
            self.started, self.session_s = float(session["started_epoch"]), float(session["session_s"])
        else:
            self.started = clock()
            meta.write_text(json.dumps({**self.ids, "arm": arm, "agents": agents if arm == "team" else 0, "session_s": self.session_s,
                                        "started_epoch": self.started, "started_at": _utc(self.started),
                                        "episodes_n": len(order), "order": order}, indent=1))
        if self.ledger.is_file() and self.ledger.read_bytes()[-1:] not in (b"", b"\n"):
            # A line torn by a power loss: the next answer starts on a line of its own.
            with self.ledger.open("a") as fh:
                fh.write("\n")
        self.answered = {a["episode_id"]: a for a in _jsonl(self.ledger)}
        # The open report opened at the last answer, or at the start: a relaunch keeps it so.
        self._open(max([self.started, *(_answered_epoch(a) for a in self.answered.values())]))

    # -- the clock and the list --------------------------------------------

    def elapsed(self) -> float:
        return self.clock() - self.started

    def over(self) -> bool:
        return self.elapsed() >= self.session_s or self.current is None

    def status(self, i: int) -> str:
        e = self.order[i]
        if e in self.answered:
            return "skipped" if self.answered[e]["source"] == "skip" else "answered"
        if self.arm != "team":
            return "open" if i == self.index else "waiting"
        t = self.elapsed()
        return "ready" if t >= self.ends[i] else "searching" if t >= self.starts[i] else "not_started"

    def _open(self, at: float) -> None:
        """The earliest open report, opened ``at`` (the start or the last answer): verified if
        MAPS's result was ready then, searched by hand if not, and kept so until it is answered."""
        self.index = next((i for i, e in enumerate(self.order) if e not in self.answered), None)
        self.current = None if self.index is None else self.order[self.index]
        self.review: Session | None = None
        self.point: tuple[float, float] | None = None
        if self.current is None:
            return
        self.opened = at
        self.opened_at = _utc(self.opened)
        self.verify = self.arm == "team" and self.opened - self.started >= self.ends[self.index]
        if self.verify:
            self.review = Session(self.maps_root, [self.current], self.out, labels=self.labels, rows=self.rows,
                                  record=self._reviewed)

    def ready_at(self, i: int) -> str | None:
        return _utc(self.started + self.ends[i]) if self.arm == "team" else None

    # -- the answers -------------------------------------------------------

    def _answer(self, source: str, xy, **more) -> dict:
        if self.over():
            return {"ok": False, "error": "the session is over"}
        i, e, now = self.index, self.current, self.clock()
        answer = {**self.ids, "episode_id": e, "order_index": i, "arm": self.arm, "source": source,
                  "xy": None if xy is None else [float(xy[0]), float(xy[1])],
                  "opened_at": self.opened_at, "answered_at": _utc(now), "answered_epoch": now,
                  "answer_s": round(now - self.opened, 1),
                  "maps_ready_at": self.ready_at(i), "maps_ready_at_open": self.verify, **more}
        with self.ledger.open("a") as fh:
            fh.write(json.dumps(answer) + "\n")
        self.answered[e] = answer
        self._open(now)
        return {"ok": True}

    def _reviewed(self, decision: dict) -> None:
        # The audit session decided (confirm or replace): it lands as this report's answer.
        self._answer(decision["decision"], decision["xy"], candidate=decision.get("candidate"),
                     maps_xy=decision.get("maps_xy"))

    def act(self, req: dict) -> dict:
        if self.over():
            return {"ok": False, "error": "the session is over"}
        if req.get("episode_id") != self.current:
            # A stale or second tab, or one that names no report: its action may be meant for one already answered.
            return {"ok": False, "error": "That report already changed. The page now shows the current one."}
        kind = req.get("action")
        if kind == "skip":
            return self._answer("skip", None)
        if self.review is not None:
            # Verifying MAPS's result: the audit session's own verbs; its decision is the answer.
            return self.review.act(req)
        if kind == "select":
            self.point = (float(req["x"]), float(req["y"]))
            return {"ok": True}
        if kind == "clear":
            self.point = None
            return {"ok": True}
        if kind == "commit":
            if self.point is None:
                return {"ok": False, "error": "click the map where the target is first"}
            return self._answer("own_search", self.point)
        return {"ok": False, "error": f"unknown action {kind!r}"}

    def state(self) -> dict:
        t = self.elapsed()
        # The list near the open report: five back, forty in all.
        lo = max(0, (self.index or 0) - 5)
        hi = min(len(self.order), lo + 40)
        return {
            "arm": self.arm, "elapsed_s": round(t), "remaining_s": max(0, round(self.session_s - t)),
            "over": self.over(), "total": len(self.order),
            "answered_n": sum(1 for a in self.answered.values() if a["source"] != "skip"),
            "skipped_n": sum(1 for a in self.answered.values() if a["source"] == "skip"),
            "list": [{"n": i + 1, "episode_id": self.order[i], "status": self.status(i), "current": i == self.index}
                     for i in range(lo, hi)],
            "current": None if self.current is None or self.over() else {
                "n": self.index + 1, "episode_id": self.current,
                "instruction": str(self.rows[self.current]["description"]),
                "mode": "verify" if self.verify else "search",
                "point": None if self.point is None else list(self.point),
                "review": self.review.state() if self.review is not None else None,
            },
        }


def practice_root(maps_root: Path, n: int, seed: int, out: Path) -> list[str]:
    """``n`` rows of the bench's own split that ``maps_root`` did not fly, as a root the capacity
    session opens alone: the instruction and the block, no MAPS result, marked practice."""
    from baselines.human.team_rows import select_bench
    # The bench first: its paths are read at import.
    select_bench(maps_root)
    from benchmarks.citynav.adapter import load_split, parse_episode
    from benchmarks.citynav.paths import split_path

    manifest = json.loads((maps_root / "run_manifest.json").read_text())
    flown = {e.split("#")[0] for e in scored_rows(maps_root)}
    episodes = [parse_episode(r, row_index=i) for i, r in enumerate(load_split(split_path(manifest["split"])))]
    # Not a target the root flew under any of its descriptions, so practice teaches nothing of a half.
    spare = [ep for ep in episodes if ep.episode_id.split("#")[0] not in flown]
    picked = random.Random(f"practice:{seed}").sample(spare, min(n, len(spare)))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ep in enumerate(picked):
        (out / "practice" / str(i)).mkdir(parents=True, exist_ok=True)
        rows.append({"episode_id": ep.episode_id, "map_name": ep.map_name, "description": ep.description, "status": "ok",
                     "workdir": f"practice/{i}", "elapsed_s": 0.0, "metrics": {"SR": 0, "practice": True}})
    (out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "run_manifest.json").write_text(json.dumps({**manifest, "practice": True, "selection": {"episodes_n": len(rows)}}, indent=1))
    (out / "practice.txt").write_text("".join(r["episode_id"] + "\n" for r in rows))
    return [r["episode_id"] for r in rows]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="deal the halves and each participant's ordered lists")
    p.add_argument("--maps-root", type=Path, required=True)
    p.add_argument("--participants", required=True, help="comma-separated labels, never names: p01,p02,...")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("practice", help="a practice root of rows the MAPS root never flew")
    q.add_argument("--maps-root", type=Path, required=True)
    q.add_argument("--n", type=int, default=4)
    q.add_argument("--seed", type=int, default=1234)
    q.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "practice":
        picked = practice_root(args.maps_root, args.n, args.seed, args.out)
        print(f"{len(picked)} practice reports -> {args.out}/practice.txt")
        return 0
    doc = plan(scored_rows(args.maps_root), [x for x in args.participants.split(",") if x], args.seed)
    for name, d in doc["participants"].items():
        (args.out / name).mkdir(parents=True, exist_ok=True)
        for arm, order in d["lists"].items():
            (args.out / name / f"{arm}.txt").write_text("".join(e + "\n" for e in order))
    (args.out / "plan.json").write_text(json.dumps(
        {**doc, "participants": {k: {kk: vv for kk, vv in v.items() if kk != "lists"} for k, v in doc["participants"].items()}},
        indent=1))
    print(f"halves {doc['halves_n']} at MAPS SR {doc['halves_maps_sr']}; "
          f"{len(doc['participants'])} participants -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
