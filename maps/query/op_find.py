"""The ``find`` op and its match tiers: rows scored against a query by name,
tag and class (``text_match``), the notes on what matched and what did not,
and ``stratum``, one tier of a find as a set."""

from __future__ import annotations

import threading

from maps.query import color_words
from maps.query import text_match as T
from maps.query.op_io import (
    ProgramError,
    Result,
    _as_rows,
    _attach_standing_columns,
    _ordered,
    _quote,
)
from maps.query.result_tables import STANDING_SIDE_PREFIXES, _census_note


class _FindMemo(threading.local):
    """A find's per-query work and word rarity for the surface lists it was
    built on; ``lists`` pins them so their ids cannot be reused. Each thread
    holds its own, so two missions in one process never share one."""

    def __init__(self):
        self.reset(None, None)

    def reset(self, key, lists):
        self.key, self.lists, self.idf = key, lists, None
        self.scored, self.dead, self.census = {}, {}, {}

    def scored_rows(self, query, q_data, rows) -> list:
        """Copies of every row with its ``name_match``, best-first."""
        hit = self.scored.get(query)
        if hit is None:
            if self.idf is None:
                self.idf = T.token_idf(rows)
            hit = [
                {
                    **x,
                    "name_match": round(
                        T.find_score(
                            q_data,
                            x.get("search_text") or "",
                            x.get("name"),
                            x.get("tags"),
                            idf=self.idf,
                        ),
                        3,
                    ),
                }
                for x in rows
            ]
            # A query that is a tag value reaches every row carrying that
            # tag key, at T.TAG_FAMILY_SCORE, below every exact hit.
            fams = T.tag_family_keys(q_data, rows)
            if fams:
                for r in hit:
                    if r["name_match"] < T.TAG_FAMILY_SCORE and (r.get("tags") or {}).keys() & fams:
                        r["name_match"] = T.TAG_FAMILY_SCORE
            hit.sort(key=lambda x: x["name_match"], reverse=True)
            self.scored[query] = hit
        return [dict(r) for r in hit]

    def dead_words(self, query, matched) -> list:
        if query not in self.dead:
            self.dead[query] = T.unmatched_query_tokens(query, matched)
        return self.dead[query]

    def census_note(self, query, rows) -> str:
        if query not in self.census:
            self.census[query] = _census_note(rows, query)
        return self.census[query]


_FIND_MEMO = _FindMemo()


def _find_memo(cands, anchors) -> _FindMemo:
    """The per-surface retrieval memo, emptied when the surface lists move."""
    key = (id(cands) if cands else 0, id(anchors) if anchors else 0)
    if _FIND_MEMO.key != key:
        _FIND_MEMO.reset(key, (cands, anchors))
    return _FIND_MEMO


# Above this many tied ways under one name, a find renders the set's shape.
_NAME_SET_SHAPE_ROWS = 8


def _match_label(r):
    """What a query hit: the row's name, else its first tag pairs."""
    nm = (r.get("name") or "").strip()
    if nm:
        return repr(nm[:32])
    tags = r.get("tags")
    if isinstance(tags, dict) and tags:
        out: list = []
        for k, v in tags.items():
            pair = f"{k}={v}"
            if sum(len(x) + 2 for x in out) + len(pair) > 40:
                break
            out.append(pair)
        if out:
            return ", ".join(out)
    return repr(str(r.get("class") or r.get("fid") or "?")[:32])


def _miss_notes(query, q_data) -> tuple[str, list]:
    """``(empty_condition, notes)`` of a query no row scored past
    T.FUZZY_THRESHOLD on; a palette color word, which matches no row text,
    gets its column named."""
    tokens = sorted({t for cl in q_data for t in cl[1]})
    cond = (
        f"find matched nothing: no row scored >= {T.FUZZY_THRESHOLD} "
        f"against query {_quote(query)} (query tokens: {tokens})."
    )
    for cl in q_data:
        w = str(cl[0] or "").strip()
        if not w or " " in w:
            continue
        canon = color_words.normalize_color(w)
        if canon in color_words.COLOR_ENUM:
            return cond, [
                f"{_quote(w)} is a color word on the closed palette: color "
                f"is not row text -- it rides the color_{canon} column, which "
                f"attaches when the color word is named."
            ]
    return cond, []


def _match_band(scored) -> tuple[int, int]:
    """``(n, band_cut)`` over best-first scored rows: the first ``n`` match,
    and a query that hits a NAME cuts the ``band_cut`` rows under the name
    tier, which share a word, not a referent ("... Road")."""
    n = sum(1 for r in scored if r["name_match"] >= T.FUZZY_THRESHOLD)
    n_named = sum(1 for r in scored if r["name_match"] >= T.NAME_TIER_FLOOR)
    if not n_named:
        return n, 0
    # Only a proper name counts: a detection's class scores 1.0 too.
    n_proper = sum(
        1
        for r in scored
        if r["name_match"] >= T.NAME_TIER_FLOOR and str(r.get("name") or "").strip()
    )
    n_exact = n_named if n_proper else sum(1 for r in scored if r["name_match"] > T.TOKEN_BAG_CEIL)
    return n_exact, n - n_exact


def _is_name_set(matched) -> bool:
    """Whether the matches are one name's tied ways (more than
    _NAME_SET_SHAPE_ROWS, all exact, one non-empty name)."""
    exact = [r for r in matched if r.get("name_match", 0.0) >= T.NAME_TIER_FLOOR]
    names = {str(r.get("name") or "") for r in exact}
    return (
        len(exact) > _NAME_SET_SHAPE_ROWS
        and len(exact) == len(matched)
        and len(names) == 1
        and bool(next(iter(names)))
    )


def _tier(v) -> str:
    """The match tier a ``name_match`` score falls in."""
    if v >= T.NAME_TIER_FLOOR:
        return "exact"
    return "tagged" if v >= T.KV_TIER_BASE else "fuzzy"


def _match_notes(scored, n, band_cut) -> list:
    """What a matched find says: how many matched and whether their order is
    real, the tier split, the band cut and the head."""
    vals = [r["name_match"] for r in scored[:n]]
    rest = len(scored) - n - band_cut
    if vals[0] == vals[-1]:
        order = (
            f"all equally good matches at match {vals[0]}, so this order "
            f"is arbitrary and ordering them by a cue column chooses it"
        )
    else:
        order = f"best-first, match {vals[0]}..{vals[-1]}"
        top_n = sum(1 for x in vals if x == vals[0])
        if top_n > 1:
            order += (
                f"; the first {top_n} tie at match {vals[0]}, so their "
                "order among themselves is arbitrary and ordering them "
                "by a cue column chooses it"
            )
    notes = [
        f"{n} of {len(scored)} rows matched ({order}"
        + (f"; {rest} sub-threshold row(s) NOT in this set" if rest else "")
        + ")."
    ]
    tiers = [_tier(x) for x in vals]
    n_ex, n_tg = tiers.count("exact"), tiers.count("tagged")
    if n_ex and n - n_ex:
        notes.append(
            f"tiers: {n_ex} exact, {n_tg} tagged, {n - n_ex - n_tg} "
            "fuzzy -- .exact / .tagged / .fuzzy on this result hold "
            "each as a set."
        )
    if band_cut:
        notes.append(
            f"matched as a NAME: {band_cut} row(s) under the name "
            f"tier are a different feature and are NOT in this set."
        )
    head = "; ".join(
        f"{r.get('fid')} {_match_label(r)} (match {r['name_match']})" for r in scored[:n][:4]
    )
    lead = "Best" if vals[0] >= T.NAME_TIER_FLOOR else "Head"
    notes.append(f"{lead}: {head}")
    return notes


def _name_cut(rows, want, query, idf) -> tuple[list, str | None]:
    """``name=``: the rows NAMED it, matched as a name, never fuzzily, and
    the note saying how many it kept. None raises."""
    nq = T.fuzzy_prepare_query(want)
    named = [
        r
        for r in rows
        if isinstance(r, dict)
        and T.find_score(nq, r.get("search_text") or "", r.get("name"), r.get("tags"), idf=idf)
        >= T.NAME_TIER_FLOOR
    ]
    if not named:
        raise ProgramError(
            f"nothing found is NAMED {_quote(want)} and matches {_quote(query)}"
            ". A name is matched as a NAME, never fuzzily -- check the "
            f"spelling, or drop the class word and say find({_quote(want)})."
        )
    if len(named) == len(rows):
        return named, None
    return named, (
        f"name={_quote(want)} kept {len(named)} of {len(rows)} by "
        "NAME; the fuzz band does not apply to a name"
    )


def _find(cx, a):
    """Search the surface: a query returns its match set best-first; no
    query browses every row."""
    notes: list = []
    set_notes: list = []  # the notes about the matched rows
    matched = False  # a query ran and matched (the rN is ordered)
    name_set = False  # tied ways of one name: render the shape
    empty_cond = None
    surface = rows = cx.cands + cx.anchors
    query = a.get("query")
    fid_hit = cx.by_fid.get(str(query or "").strip()) if query else None
    if fid_hit is not None:
        # The fuzzy scorer matches text, never ids.
        rows, matched = [fid_hit], True
        notes.append(f"query {str(query).strip()!r} is a live fid -- that row returned directly.")
    elif query:
        q_raw = str(query)
        if ";" in q_raw:
            raise ProgramError(
                f"find query {_quote(q_raw)} contains ';', which is not a "
                f"query operator (the whole string would fuzz-score "
                f"as one text blob). ALTERNATIVES are the literal "
                f"UPPERCASE token OR ('a=b OR c=d', scored as the "
                f"max over them); to require BOTH, put the k=v "
                f"pairs in one query separated by a space ('a=b "
                f"c=d' -- the exact-tag tier scores by pairs "
                f"matched)."
                " OSM multi-value tag VALUES use ';' "
                "(cuisine=pizza;kebab) -- to match one, query that "
                "single value ('cuisine=pizza')."
            )
        q_data = T.fuzzy_prepare_query(q_raw)
        memo = _find_memo(cx.cands, cx.anchors)
        rows = memo.scored_rows(q_raw, q_data, surface)
        if rows and rows[0]["name_match"] < T.FUZZY_THRESHOLD:
            empty_cond, color_note = _miss_notes(q_raw, q_data)
            notes += [memo.census_note(q_raw, surface), *color_note]
            rows = []
        elif rows:
            n, band_cut = _match_band(rows)
            name_set = _is_name_set(rows[:n])
            if not name_set:
                set_notes += _match_notes(rows, n, band_cut)
                notes += set_notes
            rows, matched = rows[:n], True
            # Name the query words no matched row carries.
            dead = memo.dead_words(q_raw, rows)
            if dead:
                notes.append(
                    "no matched row carries "
                    + ", ".join(repr(t) for t in dead[:4])
                    + " -- those words did not narrow anything"
                )
            if n <= 3:
                notes.append(memo.census_note(q_raw, surface))
        want = str(a.get("name") or "").strip()
        if want:
            rows, note = _name_cut(rows, want, q_raw, memo.idf)
            set_notes += [note] if note else []
            notes += [note] if note else []
            matched = True
    op_cols = []
    if cx.standing is not None and rows:
        rows = _attach_standing_columns(rows, cx.standing)
        h = str(cx.standing["handle"])
        op_cols = [f"{p}{h}" for p in STANDING_SIDE_PREFIXES]
    if not rows and query:
        return Result(kind="miss", notes=notes, empty_condition=empty_cond)
    return Result(
        rows, ordered=matched, name_set=name_set, op_cols=op_cols, notes=notes, set_notes=set_notes
    )


def _stratum(cx, a):
    """A find's match tier (exact / tagged / fuzzy) as a set, in input
    order."""
    items = _as_rows(cx, a["items"])
    tier = a["tier"]
    if not any(isinstance(it.get("name_match"), (int, float)) for it in items):
        raise ProgramError(
            "these rows carry no match tiers: .exact/.tagged/.fuzzy come "
            "off a find's own result."
            " A pass's minted rows are all THE thing the "
            "watchlist named; there is nothing to stratify."
        )
    return Result(
        (
            it
            for it in items
            if isinstance(v := it.get("name_match"), (int, float)) and _tier(v) == tier
        ),
        "cut",
        ordered=_ordered(items),
    )
