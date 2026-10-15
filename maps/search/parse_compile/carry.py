"""The op-program runner and the carry: a result handle is a query, so a
read after the surface moves re-runs the ops behind it (and the handles
they read); a replay that fails keeps the last rows."""

from __future__ import annotations

import json
import logging
import re

from maps.search.analyze_scene.geom_program import (
    ProgramError,
    compact_results,
    is_color_column_key,
    join_hints,
    run_program,
)
from maps.search.parse_compile.code_surface import HANDLE_RE
from maps.search.withhold import withheld

log = logging.getLogger(__name__)


def _preflight_color_note() -> str:
    """Served on a refused preflight program that names a color column:
    before landing no row carries perceived color."""
    return (("nothing is detected before landing" if withheld("prior")
             else "priors carry no perceived color before landing")
            + ", so no color column attaches yet; color enriches per-stop "
              "after arrival.")


def _requests_color(op: dict) -> bool:
    """Whether this op names a color column (a filter key, a score_by
    emphasis key)."""
    args = op["args"]
    return any(is_color_column_key(k) and k[len("color_"):].strip()
               for k in [args.get("key")]
               + [e["key"] for e in args.get("emphasis") or []])


def run_ops(ops: list, *, namespace: dict,
            raw_sink: dict | None = None,
            seed_results: dict | None = None,
            preflight: bool = False,
            answer_ref: str = "") -> dict:
    """Run one op list over the namespace's surface and compact the results
    for serving (``raw_sink["results"]`` gets them raw); a refused program
    returns ``{"error": ...}``."""
    ns = namespace
    cands = ns.get("cands") or []
    anchors = ns.get("anchors") or []
    try:
        out = run_program(cands, anchors, {"ops": ops},
                           seed_results=seed_results,
                           staged=ns.get("staged"),
                           standing=ns.get("standing"))
    except ProgramError as exc:
        _err = str(exc)
        if raw_sink is not None:
            raw_sink["error_op"] = getattr(exc, "op", "")
        if preflight and any(_requests_color(op) for op in ops):
            # The color request errored before any envelope existed (e.g.
            # the emphasis absent-color guard) -- the phase fact rides the
            # error so "no color" reads as WHEN, not broken.
            _err = join_hints(_err, _preflight_color_note())
        return {"error": _err}
    except Exception as exc:  # noqa: BLE001 -- served to the model; never crash the turn
        return {"error": f"program failed: {exc}"}
    _raw_results = out["results"]
    if raw_sink is not None:
        raw_sink["results"] = _raw_results
    # ``answer_ref`` names the one table a plan returns; the other ops' hints
    # ride it.
    out["results"] = compact_results(_raw_results, answer_ref=answer_ref)
    return out


def program_dedup_key(ops: list) -> str | None:
    """Canonical identity of a SELF-CONTAINED op list (each ref renamed to
    its emission index), or ``None`` when it reads a handle it does not
    define and is therefore never deduped."""
    if not ops:
        return None
    ren = {op["ref"]: f"r{i}" for i, op in enumerate(ops)}
    blob = json.dumps(ops, sort_keys=True, default=str)
    handle = rf"\b{HANDLE_RE.pattern}\b"
    if any(t not in ren for t in re.findall(handle, blob)):
        return None
    return re.sub(handle, lambda m: ren[m.group(0)], blob)


def _refs_read(op: dict) -> set:
    """The handles an op's args read (``rN`` or ``rN[a:b]``)."""
    return set(re.findall(rf'"({HANDLE_RE.pattern})(?:\[[^"\]]*\])?"',
                          json.dumps(op["args"], default=str))) - {op["ref"]}


def program_reads(ops: list) -> set:
    """The carry handles an op list reads and does not define."""
    return set().union(*(_refs_read(o) for o in ops)) - {o["ref"] for o in ops}


class Carry:
    """Every successful program's raw results by ref (``rows``), and the op
    history that re-derives them per surface ``version``."""

    def __init__(self, namespace: dict):
        self.ns = namespace
        # The surface version: bumped by whatever moves the surface (a
        # flight, a boost, a stance).
        self.version = 0
        self.rows: dict = {}
        self._history: list = []          # every op, in execution order
        self._reads: list = []            # per op: the handles its args name
        self._producers: dict = {}        # ref -> indices into _history
        self._armed = 0                   # the version _fresh is for
        self._fresh: set = set()          # refs re-derived, or written, under it
        # A flight-minted handle records an event (the rows one flight added)
        # and never re-derives.
        self._events: set = set()

    def ensure_fresh(self, refs) -> None:
        """Re-derive ``refs`` and every op they read against the current
        surface, once per surface version."""
        if self._armed != self.version:
            self._armed = self.version
            self._fresh.clear()
            self._fresh.update(self._events)
        need: set = set()
        todo = [r for r in refs if r not in self._fresh]
        while todo:
            r = todo.pop()
            if r in need or r in self._fresh:
                continue
            idxs = self._producers.get(r)
            if not idxs:
                continue    # an event handle, an alias, an unknown: nothing to re-derive
            need.add(r)
            for i in idxs:
                todo.extend(self._reads[i])
        if not need:
            return
        _sink: dict = {}
        # Seed everything else (a flight-minted handle has no op).
        _out = run_ops(
            [op for op in self._history if op["ref"] in need],
            namespace=self.ns, raw_sink=_sink,
            seed_results={h: v for h, v in self.rows.items()
                          if h not in need})
        if _out.get("error"):
            # A refused replay keeps the last rows, unclaimed.
            log.debug("carry refresh refused; handles keep their last "
                      "rows: %s", str(_out["error"])[:200])
            return
        self.rows.update(
            {k: v for k, v in _sink["results"].items() if k in need})
        self._fresh.update(need)

    def read(self, ref):
        """A handle's rows, re-derived against the current surface first."""
        self.ensure_fresh([ref])
        return self.rows.get(ref)

    def record(self, ops: list, results: dict) -> None:
        """A successful program: its results and the ops that re-derive them."""
        self.rows.update(results)
        self._fresh.update(results)
        for op in ops:
            self._producers.setdefault(op["ref"], []).append(len(self._history))
            self._history.append(op)
            self._reads.append(_refs_read(op))

    def mint(self, handle: str, rows: list) -> None:
        """An event handle: rows a flight added, never re-derived."""
        self.rows[handle] = rows
        self._events.add(handle)
        self._fresh.add(handle)

    def alias(self, prev_ops, cur_ops) -> None:
        """A deduped emission: its refs take the rows the first run's refs
        hold now, and its own ops re-derive them from here on."""
        self.record(cur_ops, {c["ref"]: self.rows[p["ref"]]
                              for p, c in zip(prev_ops, cur_ops)
                              if p["ref"] in self.rows})
