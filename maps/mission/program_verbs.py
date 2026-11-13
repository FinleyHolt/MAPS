"""The verbs a library call opens, runs and closes on: grounding its names,
running its op program, and finishing its ack."""

from __future__ import annotations

import difflib

from maps.library.op_compiler import CodeError
from maps.library.result_handles import program_dedup_key, program_reads, run_ops
from maps.mission.acks import Rejections, add_note, live_counts
from maps.mission.surface_state import Progress, Surface
from maps.query.result_tables import BROAD_DUMP_SOFT_ROWS
from maps.query.text_match import FUZZY_THRESHOLD, unmatched_query_tokens

# The dedup hit serves a one-line REFERENCE, never a fresh payload (a full
# re-render of an identical result is loop fuel). Every path that can mint a
# row bumps the surface version, so "nothing had entered the world" is true
# of the moment it fires; the block may fly after it.
_DEDUP_NOTE = (
    "no new information since turn {turn}: nothing had entered the world "
    "between then and this call -- no flight, no look_for(..., at=) -- so this program "
    "returned what it returned then, and it is not re-rendered. Its handles "
    "stay live for later ops. Change the world (fly, look_for(..., at=)) or "
    "ask something different."
)
_DEDUP_REPEAT_NOTE = (
    "still nothing new, {n} time: the world had not moved between that turn "
    "and this call, so no program could answer differently -- only something "
    "entering it can, and a flight folds new detections. Live at this call: "
    "{counts}{cols}."
)


def _ordinal(n: int) -> str:
    """The ordinal with its article: a 1st, a 2nd, a 3rd, an 8th, an 11th,
    an 18th, an 80th."""
    n = int(n)
    if 10 <= n % 100 <= 13:
        nth = f"{n}th"
    else:
        nth = f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"
    # ponytail: 1100th and 1800th take "a" here; no repeat count nears them.
    return ("an " if nth.startswith("8") or n in (11, 18) else "a ") + nth


# How a grounding note says what the call does at a name's best match, by
# the name's why.
_WHY_VERB = {
    "stage": "standing at",
    "move": "flying",
    "map": "mapping around",
    "crops": "looking at",
    "boost": "looking around",
    "facing": "facing",
}


class Programs:
    """The verbs that open, run and close a library call: grounding its names,
    running its op program (once per surface version) and finishing its
    ack."""

    def __init__(self, surface: Surface, rejections: Rejections, progress: Progress):
        self.surface = surface
        self.carry = surface.carry
        self.rejections = rejections
        self.progress = progress
        # Successful self-contained programs by ``program_dedup_key`` ->
        # ``(surface_version, turn, ops)``: a repeat under the same version gets
        # the dedup note. A flight, a boost or a stance bumps the version.
        self._seen: dict[str, tuple] = {}
        # The call in flight: what ground noted and analyze answered, read
        # by finish.
        self._call: dict = {}
        self.new_turn()

    def new_turn(self) -> None:
        """An operator turn starts its dedup streak afresh."""
        self._dedup_key, self._dedup_n = None, 0

    def _run(self, ops, *, answer_ref=""):
        """Run ``ops`` once per surface version: a repeat of a program that
        already ran under this version serves a note, never its rows again."""
        key = program_dedup_key(ops)
        prev = self._seen.get(key) if key is not None else None
        if prev is not None and prev[0] == self.surface.version:
            # A one-line reference, never a payload; the emission's own
            # labels alias the stored handles so nothing downstream can tell
            # it from a re-run.
            self.carry.alias(prev[2], ops)
            if self._dedup_key == key:
                self._dedup_n += 1
            else:
                self._dedup_key, self._dedup_n = key, 1
            if self._dedup_n >= 2:
                c_n, a_n, dcols, _ = self.surface.census(12)
                cols_s = "; columns live: " + ", ".join(dcols) if dcols else ""
                return {
                    "note": _DEDUP_REPEAT_NOTE.format(
                        n=_ordinal(self._dedup_n), counts=live_counts(c_n, a_n), cols=cols_s
                    )
                }
            return {"note": _DEDUP_NOTE.format(turn=prev[1])}
        raw_sink: dict = {}
        try:
            self.carry.ensure_fresh(program_reads(ops))
        except CodeError as exc:
            out = {"error": str(exc)}
        else:
            out = run_ops(
                ops,
                namespace=self.surface.ns,
                raw_sink=raw_sink,
                seed_results=self.carry.rows,
                preflight=self.surface.preflight,
                answer_ref=answer_ref,
            )
        if "error" in out:
            # The failing op keeps one op's errors their own class.
            err = str(out["error"])
            out = {
                **out,
                "error": self.rejections.escalated(
                    err, sig_of=f"{err}|{raw_sink.get('error_op', '')}"
                ),
            }
        else:
            # A successful program ends both streaks: the next same-class
            # error serves its full text again.
            self.rejections.end_streak()
            self._dedup_key, self._dedup_n = None, 0
            if key is not None:
                self._seen[key] = (self.surface.version, self.progress.turn_idx, ops)
            self.carry.record(ops, raw_sink["results"])
            for rows in raw_sink["results"].values():
                for row in rows:
                    if row.get("fid"):
                        self.surface.row_latest[str(row["fid"])] = row
        return out

    def _closest_names(self, what: str) -> str:
        """The 3 nearest named rows to a name that matched nothing."""
        names: dict[str, tuple] = {}
        for f, r in self.surface.rows.items():
            nm = str(r.get("name") or "").strip()
            if nm:
                names.setdefault(nm.lower(), (f, nm))
        close = difflib.get_close_matches((what or "").lower(), list(names), n=3, cutoff=0.55)
        if not close:
            return ""
        return "; closest named rows: " + ", ".join(
            f'"{names[c][1]}" ({names[c][0]})' for c in close
        )

    def _call_minted(self):
        """The handle this call's last flight minted, or None."""
        return (self.carry.minted or [None])[-1]

    def ground(self, names: dict) -> dict:
        """Open a call and ground its names (``{name: why}``) under the
        name-match floor: ``{"ties": {name: top-tie fids}}``, or the first
        refusal."""
        self._call.clear()
        self._call["notes"] = notes = []
        ties: dict = {}
        if not names:
            return {"ties": ties}
        sink: dict = {}
        g_out = run_ops(
            [
                {"ref": f"r{i}", "op": "find", "args": {"query": q.lower()}}
                for i, q in enumerate(names)
            ],
            namespace=self.surface.ns,
            raw_sink=sink,
            preflight=self.surface.preflight,
        )
        if "error" in g_out:
            return self.rejections.error({}, str(g_out["error"]))
        g_res = sink.get("results") or {}
        for i, (q, why) in enumerate(names.items()):
            rows = [r for r in (g_res.get(f"r{i}") or []) if r.get("fid")]
            rows = [
                r
                for r in rows
                if r.get("name_match") is None or float(r["name_match"]) >= FUZZY_THRESHOLD
            ]
            what = q.lower()
            if not rows:
                return self.rejections.error(
                    {},
                    (
                        f'nothing named "{what}" is on '
                        f"this surface{self._closest_names(what)}; say it "
                        "differently, or say a fid off a served row"
                    ),
                )
            # every name keeps only its top name_match tie group
            top = rows[0].get("name_match")
            ties[q] = [str(r["fid"]) for r in rows if r.get("name_match") == top] or [
                str(rows[0]["fid"])
            ]
            if why == "stage" and len(ties[q]) > 1:
                f = ", ".join(ties[q][:4])
                return self.rejections.error(
                    {},
                    (
                        f"a viewpoint is one point; "
                        f'"{what}" has {len(ties[q])} equally good '
                        f"name matches ({f}); say which one — a fid "
                        "works"
                    ),
                )
            # A token no matched row carries gets a note, never a refusal.
            dead = unmatched_query_tokens(q.lower(), rows)
            if dead:
                notes.append(
                    f'no row matching "{what}" carries '
                    + ", ".join(repr(t) for t in dead[:4])
                    + " -- those words did not narrow anything"
                )
            # a covered name skips this note: the pass flies every top tie
            if why not in ("anchor", "cover") and len(rows) > 1:
                notes.append(
                    f'{len(rows)} rows match "{what}"; '
                    f"{_WHY_VERB.get(why, 'using')} the best "
                    f"name match, {rows[0]['fid']} — say a fid to "
                    "choose precisely"
                )
        return {"ties": ties}

    def analyze(self, ops: list, rn_end: int, answer, page) -> dict:
        """Run one compiled program; ``answer`` names the ref whose envelope
        is served, ``page`` the ``(ref, start, the model's name for it)`` a
        more_of window pages."""
        if page:
            # Paging a handle with nothing left to page is refused here.
            ref, lo, h = page
            rows = self.carry.read(ref) or []
            held = len(rows)
            if held and held <= BROAD_DUMP_SOFT_ROWS and not getattr(rows, "ordered", False):
                return self.rejections.error(
                    {},
                    (
                        f"{h} holds {held} row(s), one block, and no order: a bare `{h}` "
                        "line displays its census and len, and an ordering serves its "
                        f'rows ({h}.order(nearest="<fid>")). There is no further block.'
                    ),
                )
            if held and held <= BROAD_DUMP_SOFT_ROWS:
                return self.rejections.error(
                    {},
                    (
                        f"{h} holds {held} row(s), one block, and a bare `{h}` line "
                        "serves all of it: there is no further block."
                    ),
                )
            if held and lo >= held:
                return self.rejections.error(
                    {},
                    (
                        f"all {held} row(s) of {h} are already served -- there "
                        f"is no further block to serve. Narrow it "
                        f'({h}.order(color="<colour>"), '
                        f'{h}.order(nearest="<fid>"), '
                        f'{h}.filter(within="<place>"))'
                        f" or look({h})."
                    ),
                )
        out = self._run(ops, answer_ref=str(answer or ""))
        if "error" in out:
            return {"error": out["error"]}
        self.carry.advance_rn(rn_end)
        # The columns this call ordered by, so finish can say when the
        # ordering separated nothing.
        sb = next((o for o in reversed(ops) if o["op"] == "score_by"), None)
        self._call.update(
            answer=answer,
            page=page,
            rows=self.carry.rows.get(answer),
            order=[str(e["key"]) for e in sb["args"]["emphasis"]] if sb else None,
        )
        return {"result": out}

    def finish(self, ack: dict) -> dict:
        """Close a call nothing refused: the tie note, the grounding notes,
        the streak reset, the scope and paging writes."""
        # An ordering that separated nothing says how many rows tie on
        # every column it ordered by.
        rows = self._call.get("rows") or []
        keys = [
            k
            for k in dict.fromkeys(self._call.get("order") or [])
            if len(rows) >= 2 and rows[0].get(k) is not None
        ]
        tied = [r for r in rows if all(r.get(k) == rows[0][k] for k in keys)]
        if keys and len(tied) >= 2:
            if len(keys) == 1:
                v0 = rows[0][keys[0]]
                t = f"{v0:g}" if isinstance(v0, (int, float)) else str(v0)
                on = f"at {t} on '{keys[0]}'"
            else:
                on = "on every ordering column (" + ", ".join(f"'{k}'" for k in keys) + ")"
            add_note(
                ack,
                f"{len(tied)} of {len(rows)} rows tie {on} -- "
                "this ordering does not separate them, so its rank 0 "
                "is arbitrary among the tie",
                "\n",
            )
        if self._call.get("notes"):
            add_note(ack, "; ".join(self._call["notes"]), "\n")
        self.rejections.end_streak()
        if self._call.get("page"):
            self.carry.paged.update([self._call["page"][:2]])
        # The set in scope: the call's non-empty answer, or what its last
        # flight added.
        if self._call.get("answer") and self._call.get("rows"):
            self.carry.in_scope = self._call["answer"]
        if self._call_minted():
            self.carry.in_scope = self._call_minted()
        return ack
