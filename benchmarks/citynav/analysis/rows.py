"""Which attempt is an episode's result — one definition, every reader.

A run's ledger is ``<run>/episodes.jsonl``: one line per ATTEMPT, appended
by the run-root owner when the attempt lands. Every row is flat -- the
scored ``metrics`` dict, ``pred_xy`` / ``final_pos_xy`` / ``visited_xy``,
``description``, ``provenance`` and ``search_env_flags`` sit at the top
level -- and ``workdir`` is the attempt's directory RELATIVE to the
directory holding the ledger. The runner's own copy of the row is
``<workdir>/episode.json``.

An episode can be attempted more than once: an infra casualty is
re-queued, and before ``f22e8fd`` the standalone dispatch took its claim
cursor modulo the selection length, so it wrapped and a worker that
finished early was handed a sibling's LIVE episode (one 4-worker /
4-episode run produced nine attempts).

THE RULE, defined here and nowhere else: an episode's result is its
FIRST FULLY COMPLETE attempt, ordered by attempt START time.

    fully complete  the row carries metrics and its ``status`` is ``ok``
    start time      the ``maps_<YYYYmmdd>_<HHMMSS>[_run<NN>]_<benchmark>_<slug>``
                    stamp on the workdir the attempt was launched into

START, not completion. The ledger appends when an attempt FINISHES, so
its order ranks a fast duplicate ahead of the slow attempt that was
claimed first -- measured on one train_seen run, one
episode's three attempts sit at file positions 4, 6, 7 with start stamps
132524, 133120, 133016. And because a duplicate is ALWAYS the
later-STARTED attempt, first-by-start recovers the attempt the scheduler
legitimately claimed, while completion order picks whichever replicate
happened to run fastest.

``status`` is the one verdict on a row, stamped by the producer:
``ok`` (scored, complete -- ``ended_by`` turn_cap / compaction_infeasible
/ oversize_rejection are the algorithm spending its own budget and stay
``ok``), ``timed_out`` (scored, in the denominator, counted apart),
``infra`` (never a fair run: serve death, OOM, ingest or SAM outage,
spawn failure, crash; dropped at read time), ``flying`` (the pre-flight
floor -- lives only in an ``episode.json``, and no owner appends one to
a ledger).

Fallbacks for an episode with no complete attempt, in order: the LAST
attempt carrying metrics (an all-failure episode still reports, and a
re-claimed retry still supersedes its earlier failure), then the LAST
attempt at all (an ``ok`` row with no metrics is a no-commit miss and
keeps its coverage slot instead of vanishing from the denominator).

Stdlib only, and imports nothing from this repo, so the benchmark driver
(which writes summary.txt live from worker threads) and the analysis CLIs
can both depend on it without a cycle. Deliberately NOT folded into
``run_report`` -- a report CLI the driver must not import.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Sequence

#: The run's one ledger. Byte-identical to ``benchmarks.citynav.episodes.
#: EPISODES_JSONL`` (the producer); tests/test_rows.py pins the two equal.
EPISODES_JSONL = "episodes.jsonl"

# ---------------------------------------------------------------------------
# One row
# ---------------------------------------------------------------------------


def metrics(row: dict) -> dict:
    """This row's scored metrics dict, or ``{}``."""
    return row.get("metrics") or {}


def episode_id(row: dict, m: dict | None = None) -> str | None:
    """This row's episode id, else None.

    Metrics first, then the row's own key. A crash row carries no metrics
    but DOES carry an episode_id: every one of the fourteen no-metrics
    rows in ``citynav_val_seen_2026-07-23_16-56-19`` did.
    """
    if m is None:
        m = metrics(row)
    eid = m.get("episode_id") or row.get("episode_id")
    return str(eid) if eid else None


def final_xy(row: dict) -> tuple[float, float] | None:
    """Where this episode's flight ENDED, else ``None``.

    The point ``SR`` was scored from. ``pred_xy`` before ``final_pos_xy``: they are the same point
    on a scored row, and ``pred_xy`` is the one the runner writes as the
    prediction.
    """
    for key in ("pred_xy", "final_pos_xy"):
        xy = row.get(key)
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            return (float(xy[0]), float(xy[1]))
    return None


def is_provisional(row: dict) -> bool:
    """True for the pre-flight floor the runner writes to ``episode.json``
    before takeoff -- a stand-in for an episode whose flight then dies, not
    a result. It never lands in a ledger; the driver's landed-row read
    asks this."""
    return row.get("status") == "flying"


def is_infra_abort(row: dict) -> bool:
    """True when the attempt never got a fair run.

    The PRODUCER stamps ``status: infra`` -- for a serve dying under the
    episode (``ended_by=transient_failures``), a CUDA/CuPy contention OOM,
    an ingest or SAM outage, a warm-runner spawn failure, a crash. All are
    infra casualties rather than algorithm misses, so they are excluded
    from coverage and SR -- and dropped at READ time, not merely marked
    incomplete, so one can never represent its episode through the
    last-resort fallback. A serve dying under an episode is not the
    reasoner getting it wrong, and charging SR for it prices the
    cluster's weather as the model's skill.

    THE DRIVER'S PREDICATE AND THIS ONE ARE THE SAME PREDICATE.
    ``bench_fullset._infra_casualty`` delegates here. They were two
    definitions of one question and they had already diverged: the driver
    excluded a ``transient_failures`` row from its done-set and re-queued
    the episode, while this reader scored the excluded attempt and
    discarded the re-run as a replicate. Measured on the reasoning-effort
    ladder, where 7 of one arm's 32 episodes went that way and the arm
    re-queued them forever; and before that on
    ``f40088a:calcity_full_replyrepair``, where a serve OOM storm forced
    50 of 1022 episodes to SR 0.000.

    What is NOT here, deliberately: ``turn_cap``, ``compaction_infeasible``
    and ``oversize_rejection`` stay ``ok``. Those are the algorithm
    spending its own budget -- running out of turns, or growing a
    conversation past the window -- not the serve failing underneath it.
    """
    return row.get("status") == "infra"


def is_complete(row: dict, m: dict | None = None) -> bool:
    """Did this attempt produce a real scored record?

    It carries metrics, it is not the pre-flight floor, and its ``status``
    is ``ok``. A ``timed_out`` row that did commit before the kill carries
    metrics but is NOT complete: it loses to any clean attempt of the same
    episode regardless of start order, and represents its episode only
    through the metrics-bearing fallback. Fully complete means the driver
    saw the runner exit cleanly.
    """
    if m is None:
        m = metrics(row)
    return bool(m) and not is_provisional(row) and row.get("status") == "ok"


# ---------------------------------------------------------------------------
# When the attempt started
# ---------------------------------------------------------------------------

# The workdir every attempt is launched into: ``maps_<YYYYmmdd>_
# <HHMMSS>[_run<NN>]_<benchmark>_<episode-slug>``, stamped at mkdir time by
# episode_workdir. The stamp is what this reads, so the pattern matches the
# stamp itself and not the words around it -- the benchmark word is
# episode_workdir's to choose and no ordering depends on it.
#
# Deliberately NARROW: it must not match a RUN-root name
# ``<benchmark>_<split>_<Y-m-d>_<H-M-S>``. A run root is not an attempt:
# matching it would hand every row in a run the same stamp and collapse the
# ordering entirely. The run root's dashes are what keep it out -- it
# carries no run of 8 digits.
_ATTEMPT_STAMP_RE = re.compile(r"(\d{8}_\d{6})")


def attempt_stamp(row: dict) -> str:
    """The attempt's START time as a sortable ``YYYYmmdd_HHMMSS``, or "".

    Read off ``row["workdir"]``, the ledger row's own record of where it
    ran. Returns "" when no stamp is derivable -- the warm-runner spawn
    row and the worker-loop crash row carry no ``workdir`` at all. ""
    sorts before every real stamp, but such a row is ``infra`` and never
    keyed.
    """
    recorded = row.get("workdir")
    if recorded:
        m = _ATTEMPT_STAMP_RE.search(PurePosixPath(str(recorded)).name)
        if m:
            return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# Where the attempts are recorded
# ---------------------------------------------------------------------------

# Read order; the FIRST non-empty tier answers and tiers are never mixed
# -- mixing would count one attempt twice, inflating both no_metrics and
# the denominator.
RESULT_TIERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # The run's own ledger: one row per attempt, carrying status / workdir
    # / oom_retries. Archived --loop passes are ``episodes.jsonl.pass<N>.
    # <ts>``, which the exact name excludes.
    ("ledger", (EPISODES_JSONL,)),
)


def result_files(root: Path) -> tuple[list[Path], str]:
    """(files, tier name) for a run root -- the canonical read order."""
    for tier, patterns in RESULT_TIERS:
        found: set[Path] = set()
        for pattern in patterns:
            try:
                found.update(root.glob(pattern))
            except OSError:
                continue
        if found:
            return sorted(found), tier
    return [], "none"


# ---------------------------------------------------------------------------
# The pick
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Attempt:
    """One ledger row, with everything the pick needs decided at read."""

    episode_id: str
    row: dict
    metrics: dict
    stamp: str
    seq: int
    source: Path | None
    complete: bool

    @property
    def key(self) -> tuple[str, int]:
        """Deterministic ordering: attempt START, then read order."""
        return (self.stamp, self.seq)


@dataclass(frozen=True)
class ReadStats:
    """What the read SAW, so a caller can print an honest denominator."""

    tier: str
    files: int
    rows: int
    keyed: int
    no_metrics: int
    # Attempts that hit the wall-clock cap (bench episode_timeout_s). An
    # algorithm failure inside the SR denominator, counted apart so the
    # binding bound is visible beside SR.
    timed_out: int
    # Attempts the producer stamped ``infra``. Dropped from SR, and
    # COUNTED here because an episode that silently leaves the
    # denominator is the same silent miss in the other direction.
    infra: int


def iter_json_rows(path: Path) -> Iterator[dict]:
    """Every well-formed JSON object in a .jsonl.

    Blanks, non-objects and undecodable lines are skipped: a LIVE run's
    last line is routinely a partial write, and a run root mid-rsync can
    raise OSError mid-read. The bare ``read_text()`` this replaces died on
    both.
    """
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


def read_attempts(
    files: Sequence[Path], tier: str = ""
) -> tuple[list[Attempt], ReadStats]:
    """Parse ledger files into Attempts, plus the audit counters.

    Infra rows are counted and DROPPED -- they never got a fair run, so
    they must not represent their episode even through the last-resort
    fallback. Rows with no derivable episode_id cannot be keyed and are
    dropped too, but still count in ``rows`` and ``no_metrics``.
    """
    attempts: list[Attempt] = []
    rows = no_metrics = timed_out = infra = 0
    seq = 0
    for source in files:
        for row in iter_json_rows(source):
            rows += 1
            m = metrics(row)
            if not m:
                no_metrics += 1
            if row.get("status") == "timed_out":
                timed_out += 1
            if is_infra_abort(row):
                infra += 1
                continue
            eid = episode_id(row, m)
            if not eid:
                continue
            attempts.append(
                Attempt(
                    episode_id=eid,
                    row=row,
                    metrics=m,
                    stamp=attempt_stamp(row),
                    seq=seq,
                    source=source,
                    complete=is_complete(row, m),
                )
            )
            seq += 1
    return attempts, ReadStats(
        tier=tier,
        files=len(files),
        rows=rows,
        keyed=len(attempts),
        no_metrics=no_metrics,
        timed_out=timed_out,
        infra=infra,
    )


def pick_attempts(attempts: Iterable[Attempt]) -> dict[str, Attempt]:
    """episode_id -> the ONE attempt that IS that episode's result.

    THE RULE (see the module docstring): the FIRST fully complete attempt,
    ordered by attempt START time. When no attempt is complete: the LAST
    attempt carrying metrics, then the LAST attempt at all -- "last" by
    that same start ordering, read order breaking ties.
    """
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


def load_attempts(root: Path) -> tuple[dict[str, Attempt], ReadStats]:
    """The one canonical read of a run root: episode -> its result."""
    files, tier = result_files(root)
    attempts, stats = read_attempts(files, tier)
    return pick_attempts(attempts), stats
