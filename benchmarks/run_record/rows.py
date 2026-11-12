"""Which attempt is an episode's result — one definition, every reader.

A run's ledger ``<run>/episodes.jsonl`` holds one flat row per ATTEMPT,
appended when it lands; ``workdir`` is the attempt's directory relative to
the ledger's (the runner's copy of the row is ``<workdir>/episode.json``).
An infra casualty is re-queued, so an episode can have several attempts.

THE RULE: an episode's result is its FIRST attempt with metrics and status
``ok``, ordered by attempt START time (the ``maps_<YYYYmmdd>_<HHMMSS>...``
stamp on its workdir), not ledger order, which is finish order. With no
complete attempt: the LAST attempt carrying metrics, then the LAST attempt at
all (an ``ok`` miss with no metrics keeps its slot in the denominator).

``status`` is the producer's verdict: ``ok`` (scored; running out of turns
or window, or a refused request, stays ``ok``), ``timed_out`` (scored,
counted apart), ``infra`` (never a fair run; dropped at read time).

Stdlib only but for the names ``episodes`` owns, so the benchmark
driver and the report CLIs both depend on it without a cycle.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator

from benchmarks.run_record.episodes import (
    EPISODES_JSONL,
    MANIFEST_NAME,
    STATUS_INFRA,
    STATUS_OK,
    STATUS_TIMED_OUT,
)


def metrics(row: dict) -> dict:
    """This row's scored metrics dict, or ``{}``."""
    return row.get("metrics") or {}


def episode_id(row: dict, m: dict | None = None) -> str | None:
    """This row's episode id (metrics first, then the row's own key, which a
    crash row without metrics still carries), else None."""
    if m is None:
        m = metrics(row)
    eid = m.get("episode_id") or row.get("episode_id")
    return str(eid) if eid else None


#: A mission the serve failed out from under.
INFRA_ENDINGS = ("transient_failures",)


def ended_by(row: dict) -> str | None:
    """How the attempt's mission ended: the row's own key (a flight row), else
    the scored mission block's (a bench row)."""
    return row.get("ended_by") or (metrics(row).get("mission") or {}).get("ended_by")


def is_infra_abort(row: dict) -> bool:
    """True when the attempt never got a fair run (the driver's, the readers'
    and the closed-loop measures' one predicate): ``status: infra`` on a bench
    row, an ``ended_by`` in ``INFRA_ENDINGS`` on a flight row. Such a row is
    dropped at read time.
    """
    return row.get("status") == STATUS_INFRA or ended_by(row) in INFRA_ENDINGS


def is_complete(row: dict, m: dict | None = None) -> bool:
    """The attempt carries metrics and status ``ok``. A ``timed_out`` row
    that committed carries metrics but loses to any clean attempt."""
    if m is None:
        m = metrics(row)
    return bool(m) and row.get("status") == STATUS_OK


# The start stamp on an attempt's directory; a run root's name never matches.
_ATTEMPT_STAMP_RE = re.compile(r"(\d{8}_\d{6})")


def attempt_stamp(row: dict) -> str:
    """The attempt's start time off ``row["workdir"]`` as a sortable
    ``YYYYmmdd_HHMMSS``, or "" for a row without one (an ``infra`` row of
    an attempt that never made its workdir)."""
    recorded = row.get("workdir")
    if recorded:
        m = _ATTEMPT_STAMP_RE.search(PurePosixPath(str(recorded)).name)
        if m:
            return m.group(1)
    return ""


@dataclass(frozen=True)
class Attempt:
    """One ledger row, with everything the pick needs decided at read."""

    episode_id: str
    row: dict
    metrics: dict
    stamp: str
    seq: int
    complete: bool

    @property
    def key(self) -> tuple[str, int]:
        """Deterministic ordering: attempt START, then read order."""
        return (self.stamp, self.seq)


@dataclass(frozen=True)
class ReadStats:
    """What the read saw, so a caller can print an honest denominator."""

    rows: int
    keyed: int
    no_metrics: int
    # Attempts that hit the wall-clock cap, counted apart: one with a record
    # is scored as recorded, one without is a miss.
    timed_out: int
    # Attempts stamped ``infra``: out of SR, counted so none leaves silently.
    infra: int
    # The episodes an ``infra`` row names: what a resume re-queues.
    infra_episodes: frozenset[str]


def iter_json_rows(path: Path) -> Iterator[dict]:
    """Every well-formed JSON object in a .jsonl; a live run's torn last line
    and an unreadable file yield nothing."""
    try:
        text = path.read_text()
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            yield row


def read_attempts(ledger: Path) -> tuple[list[Attempt], ReadStats]:
    """A ledger's rows as Attempts, plus the audit counters. Infra rows are
    counted and dropped (not in ``no_metrics``); rows with no episode id
    are dropped but count in ``rows`` and ``no_metrics``."""
    attempts: list[Attempt] = []
    rows = no_metrics = timed_out = infra = 0
    infra_episodes: set[str] = set()
    for row in iter_json_rows(ledger):
        rows += 1
        if row.get("status") == STATUS_TIMED_OUT:
            timed_out += 1
        m = metrics(row)
        eid = episode_id(row, m)
        if is_infra_abort(row):
            infra += 1
            if eid:
                infra_episodes.add(eid)
            continue
        if not m:
            no_metrics += 1
        if not eid:
            continue
        attempts.append(
            Attempt(
                episode_id=eid,
                row=row,
                metrics=m,
                stamp=attempt_stamp(row),
                seq=len(attempts),
                complete=is_complete(row, m),
            )
        )
    return attempts, ReadStats(
        rows=rows,
        keyed=len(attempts),
        no_metrics=no_metrics,
        timed_out=timed_out,
        infra=infra,
        infra_episodes=frozenset(infra_episodes),
    )


def pick_attempts(attempts: Iterable[Attempt]) -> dict[str, Attempt]:
    """episode_id -> its result under THE RULE (module docstring); read
    order breaks start-time ties."""
    by_episode: dict[str, list[Attempt]] = {}
    for attempt in attempts:
        by_episode.setdefault(attempt.episode_id, []).append(attempt)
    picked: dict[str, Attempt] = {}
    for eid, group in by_episode.items():
        group.sort(key=lambda a: a.key)
        complete = [a for a in group if a.complete]
        if complete:
            picked[eid] = complete[0]
            continue
        scored = [a for a in group if a.metrics]
        picked[eid] = scored[-1] if scored else group[-1]
    return picked


def read_manifest(root: Path) -> dict:
    """The run's ``run_manifest.json`` at its root, else ``{}``."""
    try:
        return json.loads((root / MANIFEST_NAME).read_text())
    except (OSError, ValueError):
        return {}


def _shortfall(root: Path, picked: dict[str, Attempt], stats: ReadStats) -> str | None:
    """Why the picked episodes are fewer than the manifest selected, else
    ``None``."""
    selected = (read_manifest(root).get("selection") or {}).get("episodes_n")
    if not selected or len(picked) >= selected:
        return None
    infra_only = len(stats.infra_episodes - picked.keys())
    return (
        f"{root}: scores {len(picked)} of the {selected} episodes its manifest selects "
        f"({infra_only} only ever infrastructure casualties, "
        f"{max(selected - len(picked) - infra_only, 0)} never landed a row)"
    )


def load_attempts(
    root: Path, *, allow_partial: bool = False
) -> tuple[dict[str, Attempt], ReadStats]:
    """The one canonical read of a run root: episode -> its result. A root
    short of its manifest's selection raises unless ``allow_partial``,
    which prints the shortfall to stderr; a root with no ledger raises."""
    if not (root / EPISODES_JSONL).is_file():
        raise SystemExit(f"{root}: not a run root (no {EPISODES_JSONL})")
    attempts, stats = read_attempts(root / EPISODES_JSONL)
    picked = pick_attempts(attempts)
    shortfall = _shortfall(root, picked, stats)
    if shortfall and not allow_partial:
        raise SystemExit(
            f"{shortfall}. Finish it by a resume into the same root (an episode that "
            "already spent bench_fullset's --max-infra-attempts needs a higher one), "
            "or pass --allow-partial to read it as it stands."
        )
    if shortfall:
        print(f"*** PARTIAL (--allow-partial): {shortfall} ***", file=sys.stderr)
    return picked, stats


def add_allow_partial(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="read a run that scores fewer episodes than its manifest "
        "selected (the shortfall is printed to stderr)",
    )


def scored_metrics(picked: dict[str, Attempt]) -> dict[str, dict]:
    """episode_id -> the metrics every reader scores; a pick without metrics
    is a miss inside every rate's denominator (SR, and OSR and spl_pos on a
    ring run)."""
    ring = any("OSR" in a.metrics for a in picked.values())
    miss = {"SR": 0, "OSR": 0, "spl_pos": 0.0} if ring else {"SR": 0}
    return {eid: a.metrics or dict(miss) for eid, a in picked.items()}
