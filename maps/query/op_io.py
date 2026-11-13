"""What every op reads and returns: the refusal (:class:`ProgramError`), the
result (:class:`Result`), the running program's handles and arguments, an
argument as rows, and the column keys an op attaches."""

from __future__ import annotations

import re

from maps.query import spatial_math as G
from maps.query.result_tables import (
    COMPARE_COL_PREFIXES,
    MODEL_UNSERVED_COLUMNS,
    STANDING_SIDE_PREFIXES,
    column_ref,
)
from maps.xy_token import is_xy


def quote_value(v) -> str:
    """A user value as a refusal quotes it: its repr, cut to 120 chars."""
    s = repr(v)
    return s if len(s) <= 120 else s[:120] + "..."


class ProgramError(Exception):
    pass


# What a refusal of a token that names no surface row says to do.
SAY_A_FID = "Say a fid off a served row."


class Result(list):
    """An op's rows and how :func:`compact_results` renders them, by
    ``kind``: ``broad`` (find, crossings, ends), ``miss`` (an empty find),
    ``cut`` (filter_items, stratum), ``compare`` or ``sorted`` (score_by).
    ``consumed`` marks rows a later op read; the other attributes are render
    inputs its op sets."""

    consumed: bool = False

    def __init__(
        self,
        rows=(),
        kind: str = "broad",
        *,
        ordered=False,
        name_set=False,
        op_cols=(),
        notes=(),
        set_notes=(),
        cut_stats=None,
        empty_condition=None,
    ):
        super().__init__(rows)
        self.kind, self.ordered, self.name_set = kind, ordered, name_set
        self.op_cols, self.notes, self.set_notes = op_cols, notes, set_notes
        self.cut_stats, self.empty_condition = cut_stats, empty_condition


def keeps_order(rows) -> bool:
    """Whether an op's input rows carry a real order its output keeps."""
    return bool(getattr(rows, "ordered", False))


def sanitize_token(lbl) -> str:
    """A token as a column label: lowercase, ``[^a-z0-9]+`` collapsed to
    ``_``. The compiler builds its keys with it too."""
    base = re.sub(r"[^a-z0-9]+", "_", str(lbl).lower()).strip("_")
    return base or "ref"


def _grouped_keys(present) -> str:
    """The keys with each family of more than three per-instance keys
    collapsed ('distance to building_* (100 instances, e.g. ...)'); a key's
    family is its column prefix plus its stem's leading word."""
    fams: dict[str, list] = {}
    for k in present:
        col = column_ref(k, COMPARE_COL_PREFIXES + ("color_",))
        if col is None:
            fams.setdefault(k, []).append(k)
            continue
        pre, stem = col
        base = re.match(r"^([A-Za-z]+)_", stem)
        fams.setdefault(pre + (base.group(1) + "_" if base else stem), []).append(k)
    parts = []
    for fam, members in sorted(fams.items()):
        if len(members) <= 3:
            parts.extend(members)
        else:
            parts.append(f"{fam}* ({len(members)} instances, e.g. {members[0]})")
    return ", ".join(parts)


def present_keys_hint(items, numeric: bool) -> str:
    """For an op on an absent key: the served keys the items DO carry --
    numeric ones (grouped) for a sort or a numeric cut, every scalar one
    for any other cut."""
    kinds = (int, float) if numeric else (int, float, str, bool)
    present = sorted(
        {
            k
            for it in items
            if isinstance(it, dict)
            for k, v in it.items()
            if isinstance(v, kinds) and v != "" and k not in MODEL_UNSERVED_COLUMNS
        }
    )
    if numeric:
        return (
            f"numeric keys present on these items: {_grouped_keys(present)}."
            if present
            else "these items carry NO numeric key."
        )
    return (
        f"keys present on these items: {present}."
        if present
        else "these items carry NO scalar key."
    )


def attach_standing_columns(items, frame, with_distance=False):
    """Copies of the rows with a staged viewpoint's side columns, standing
    at ``from_xy`` looking along ``up_unit``: 'left of'/'right of <h>' and
    'ahead of'/'behind <h>' in metres, the key present naming the side (0
    lands right/ahead). With ``with_distance``, 'distance to <h>' from the
    stand point."""
    h = str(frame["handle"])
    fx, fy = (float(frame["from_xy"][0]), float(frame["from_xy"][1]))
    ux, uy = (float(frame["up_unit"][0]), float(frame["up_unit"][1]))
    rx, ry = G.cw((ux, uy))
    left_k, right_k, ahead_k, behind_k = (f"{p}{h}" for p in STANDING_SIDE_PREFIXES)
    dist_k = f"distance to {h}"
    out = []
    for it in items:
        xy = it.get("xy") if isinstance(it, dict) else None
        if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
            out.append(it)
            continue
        dx, dy = float(xy[0]) - fx, float(xy[1]) - fy
        lat = dx * rx + dy * ry
        lon = dx * ux + dy * uy
        rec = {**it}
        # a re-attach must not leave both members of a pair
        for k in (left_k, right_k, ahead_k, behind_k):
            rec.pop(k, None)
        rec[right_k if lat >= 0 else left_k] = round(abs(lat), 2)
        rec[ahead_k if lon >= 0 else behind_k] = round(abs(lon), 2)
        if with_distance:
            rec[dist_k] = round((dx * dx + dy * dy) ** 0.5, 2)
        out.append(rec)
    return out


class ProgramRun:
    """One ``run_program`` call: the surface, the seed, the results so far
    and the op running."""

    def __init__(self, cands, anchors, seed=None, staged=None, standing=None, watched=()):
        self.cands, self.anchors = cands, anchors
        self.watched = {str(c).lower() for c in watched}
        self.by_fid = {c["fid"]: c for c in cands}
        self.by_fid.update({a["fid"]: a for a in anchors})
        self.seed = seed or {}
        self.staged = staged or {}
        self.standing = self.staged.get(standing) if standing else None
        self.results: dict = {}
        self.op = None

    def is_ref(self, tok) -> bool:
        """A handle: ``rN`` or ``rN[a:b]`` naming a result of this program
        or of an earlier call. A fid never matches."""
        if not isinstance(tok, str):
            return False
        base = tok.partition("[")[0]
        return base in self.results or base in self.seed

    def resolve(self, tok):
        """A handle (:meth:`is_ref`): ``rN`` -> its rows (this program's
        shadowing an earlier call's); ``rN[a:b]`` -> that window of them."""
        base, _, win = tok.partition("[")
        if base in self.results:
            v = self.results[base]
            v.consumed = True
        else:
            v = self.seed[base]
        return _window(v, win) if win else v


class OpArgs(dict):
    """An op's args; an absent required arg raises a ProgramError."""

    def __init__(self, d, cx):
        super().__init__(d)
        self._cx = cx

    def __missing__(self, key):
        raise ProgramError(
            f"op {quote_value(self._cx.op)} is missing required arg {quote_value(key)}"
        )


def empty_window_refusal(part: str, n: int) -> str:
    """The refusal for a window ``part`` that selects none of ``n`` rows."""
    return (
        f"window {part} selected 0 of {n} rows: the set holds {n} row(s) "
        "and the window begins at or past the end of them (or begins at "
        "or after where it stops)."
    )


def window_span(spec: str) -> tuple[int, int]:
    """A window ``"[lo:hi]"`` -> ``(lo, hi)``."""
    lo, hi = spec[1:-1].split(":")
    return int(lo), int(hi)


def _window(cur, win):
    """``rN[a:b]``: those rows, keeping the source's ``kind``, ``ordered``
    and ``op_cols``. Selecting none of a non-empty set raises."""
    lo, hi = window_span("[" + win)
    rows = cur[lo:hi]
    if isinstance(cur, Result):
        rows = Result(
            rows,
            cur.kind,
            ordered=cur.ordered,
            op_cols=cur.op_cols,
            cut_stats=None if rows else cur.cut_stats,
        )
    if not rows and cur:
        raise ProgramError(empty_window_refusal("[" + win, len(cur)))
    return rows


def as_rows(cx, arg):
    """An ``items`` arg -> point-bearing rows: a handle's rows, a fid's row,
    an ``[x, y]`` as an xy-only row, or a list of those (handles flatten).
    A list of rows comes back as itself, keeping its order attributes."""
    v = cx.resolve(arg) if cx.is_ref(arg) else arg
    if is_xy(v) or (isinstance(v, str) and v in cx.by_fid):
        v = [v]
    if all(isinstance(it, dict) and is_xy(it.get("xy")) for it in v):
        return v
    out = []
    for it in v:
        if isinstance(it, dict) and is_xy(it.get("xy")):
            out.append(it)
        elif isinstance(it, str) and it in cx.by_fid:
            out.append(cx.by_fid[it])
        elif is_xy(it):
            out.append({"xy": list(it)})
        elif cx.is_ref(it):
            out.extend(as_rows(cx, it))
        else:
            raise ProgramError(
                f"cannot resolve a point in the list from item {quote_value(it)}: each item must "
                f"be a fid on this surface, an [x, y], or a set. " + SAY_A_FID
            )
    return out


def named_rows(cx, el, need):
    """Every row a crossings/ends ref names (a road is several ways): a fid,
    an rN, a list of them, or {ref, as}. A row off the surface counts when
    it carries ``need``; a string that is neither a fid nor a handle names
    none."""
    if isinstance(el, dict):
        el = el.get("ref")
    out: list = []
    for t in el if isinstance(el, list) else [el]:
        vals = [t]
        if isinstance(t, str) and t not in cx.by_fid:
            vals = cx.resolve(t) if cx.is_ref(t) else []
        for x in vals:
            n = None
            if isinstance(x, str):
                n = cx.by_fid.get(x)
            elif isinstance(x, dict):
                n = cx.by_fid.get(x.get("fid")) or (x if x.get(need) else None)
            if n is not None and n not in out:
                out.append(n)
    return out


def as_said(el):
    """A crossings/ends ref as the model said it: its label, else itself."""
    return el["as"] if isinstance(el, dict) and el.get("as") else el


def refuse_if_empty(cx, el, what: str) -> None:
    """Refuse a crossings/ends ref whose handles hold no rows, saying so."""
    ref = el.get("ref") if isinstance(el, dict) else el
    refs = ref if isinstance(ref, list) else [ref]
    if refs and all(isinstance(t, str) and cx.is_ref(t) and not cx.resolve(t) for t in refs):
        raise ProgramError(
            f"{quote_value(as_said(el))} holds 0 rows here; there is nothing to {what}."
        )


def relate_ran(key, items) -> bool:
    """Whether the compare that writes a ``facing <lab>`` or ``<dir> of
    <lab>`` key ran: its sibling ``distance to <lab>`` is on a row, so the
    key absent everywhere means no row has a word or a side to name."""
    col = column_ref(key, ("facing ", "north of ", "south of ", "east of ", "west of "))
    if col is None:
        return False
    sib = "distance to " + col[1]
    return any(isinstance(it, dict) and sib in it for it in items)


def contact_fact(cx, key) -> str:
    """For an absent 'distance to' / 'facing' a <fid> whose mask contains
    the whole surface window: why neither is stamped. Returns the fact, or
    ''."""
    col = column_ref(key, ("distance to ", "facing "))
    if col is not None:
        ref_fid = col[1]
        ref_row = cx.by_fid.get(ref_fid)
        if isinstance(ref_row, dict) and ref_row.get("covers_window"):
            return (
                f"{ref_fid}'s mask contains the whole surface window, so "
                f"'distance to {ref_fid}'=0 for every row (containment, "
                f"not proximity) and neither the distance nor the "
                f"facing is stamped; 'center distance to "
                f"{ref_fid}'/'bearing from {ref_fid}' attach per-row and DO "
                f"order/filter. "
            )
    return ""
