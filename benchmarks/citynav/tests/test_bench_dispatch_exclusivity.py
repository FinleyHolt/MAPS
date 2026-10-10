"""One worker per episode: the standalone dispatch never double-runs.

`_next_claimed_episode` walks the selection with a cursor taken modulo
its length, so it wraps. "Done" means a qualifying row reached
episodes.jsonl — and the claim path must see that row IMMEDIATELY: run
contract300 re-flew 10/300 episodes when the in-flight release outran a
stale done cache (same worker re-claimed its own episode in the same
second), and the replicate attempts disagreed on SR. Completion here goes
through the REAL writer (`_append_result`), and nothing re-reads the
ledger, so only the synchronous done-cache add keeps a finished episode
out of the next claim.

These drive the dispatch directly: constructing a real Driver needs a
split, a GPU and a vLLM, and none of that is what broke.
"""

from __future__ import annotations

import threading
from collections import deque

from benchmarks.citynav.bench_fullset import ABORT_WINDOW, Driver


class _Dispatch:
    """The dispatch + result-write surface the claim invariant spans."""

    def __init__(self, selection, out_root):
        self._selection = list(selection)
        self._cursor = 0
        self._dispatch_lock = threading.Lock()
        self._done_cache = set()
        self._in_flight = set()
        self._deadline = None
        self.dispatcher = None
        self.coord_root = None
        self.out_root = out_root
        self._session_done = 0
        # The mass-instant-failure gate rides the same two methods: the
        # claim path reads _abort, the writer feeds _recent. Present here
        # so this stub exercises the real code, healthy.
        self._recent = deque(maxlen=ABORT_WINDOW)
        self._abort = None
        self.max_infra_attempts = 3
        self._infra_errors = {}

    _next_claimed_episode = Driver._next_claimed_episode
    _infra_attempts = Driver._infra_attempts
    _append_result = Driver._append_result
    # The result-write surface now includes the mass-instant-failure gate:
    # every row the writer takes feeds it, and the claim path reads its
    # verdict. Nothing here fills a window, so it stays healthy throughout.
    _check_mass_infra = Driver._check_mass_infra
    _log_excerpt = Driver._log_excerpt

    def log(self, msg):
        print(msg)

    def complete(self, eid, **extra):
        # A worker finishing, in production order: run_one writes the
        # row, then worker_loop's finally releases the in-flight hold.
        self._append_result({"episode_id": eid, "status": "ok", **extra})
        with self._dispatch_lock:
            self._in_flight.discard(eid)


def _dispatch(tmp_path, selection):
    return _Dispatch(selection, tmp_path)


def test_each_episode_is_handed_out_once(tmp_path):
    d = _dispatch(tmp_path, ["a", "b", "c", "d"])
    assert [d._next_claimed_episode(i) for i in range(4)] == ["a", "b", "c", "d"]


def test_a_worker_that_finishes_early_never_takes_a_live_episode(tmp_path):
    # The exact shape that produced 9 attempts for 4 episodes: every
    # episode is out, one worker finishes, and the cursor has wrapped.
    d = _dispatch(tmp_path, ["a", "b", "c", "d"])
    for i in range(4):
        d._next_claimed_episode(i)
    # w0 finishes "a": its row lands, its hold is released. b, c and d
    # are still running on the other three workers, so the worker
    # retires rather than re-running a sibling's episode.
    d.complete("a")
    assert d._next_claimed_episode(0) is None


def test_a_finished_episode_is_never_reclaimed(tmp_path):
    # contract300's 10 doubles: END and the next claim in the same
    # second. Only the synchronous add at write time keeps "a" out of the
    # very next claim.
    d = _dispatch(tmp_path, ["a", "b"])
    assert d._next_claimed_episode(0) == "a"
    d.complete("a")
    # Vacuity floor: the pool still holds real work, and the claim
    # immediately after the completed write serves it — never "a".
    assert d._next_claimed_episode(0) == "b"
    d.complete("b")
    assert d._next_claimed_episode(0) is None


def test_an_infra_row_leaves_the_episode_claimable(tmp_path):
    # An infra row never got a fair run: it must NOT enter the done set,
    # so the cursor re-offers the episode for a full redo.
    d = _dispatch(tmp_path, ["a"])
    assert d._next_claimed_episode(0) == "a"
    d.complete("a", status="infra", error="killed_midrun")
    assert d._next_claimed_episode(0) == "a"


def test_a_released_episode_with_no_result_row_is_reoffered(tmp_path):
    # A worker that dies without writing a row must not strand its
    # episode: releasing the hold puts it back in the selection.
    d = _dispatch(tmp_path, ["a", "b"])
    assert d._next_claimed_episode(0) == "a"
    with d._dispatch_lock:
        d._in_flight.discard("a")  # crashed before writing a result row
    assert d._next_claimed_episode(1) == "b"
    assert d._next_claimed_episode(1) == "a"


def test_the_selection_is_exhausted_once_every_episode_is_done(tmp_path):
    d = _dispatch(tmp_path, ["a", "b"])
    for i in range(2):
        d._next_claimed_episode(i)
    d.complete("a")
    d.complete("b")
    assert d._next_claimed_episode(0) is None
