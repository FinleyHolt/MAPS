"""What a result serves: the model-facing row, the grid its rows render as
under the character budget, the legend, the census of a broad result and the
hints, folded into one envelope per result (:func:`compact_results`)."""

from __future__ import annotations

from collections import Counter

from maps.query import color_words
from maps.query import text_match as T
from maps.render.look_render import LOOK_SET_ROWS


# Row fields the executor resolves by fid and the model never reads.
_MODEL_HIDDEN_ROW_FIELDS = ("search_text", "centerline", "mask",
                            "mask_centroid", "cells")

# Fields folded into the rendered color cell, never served raw.
_RENDER_INPUT_ROW_FIELDS = ("color_shade", "color_flag")

# Below this color_conf the rendered color word is marked uncertain.
_COLOR_CONF_FLOOR = 0.15


def _render_color_cell(row):
    """The row with its color rendered ``[dk|lt ]word[/word2][?]`` (``?``
    for a below-floor conf or ``color_flag='look'``) and the shade/flag
    inputs removed."""
    if not isinstance(row, dict):
        return row
    c = row.get("color")
    has_color = isinstance(c, str) and c.strip()
    if not has_color and not any(k in row for k in _RENDER_INPUT_ROW_FIELDS):
        return row
    out = {k: v for k, v in row.items() if k not in _RENDER_INPUT_ROW_FIELDS}
    if has_color:
        conf = row.get("color_conf")
        uncertain = ((isinstance(conf, (int, float)) and conf < _COLOR_CONF_FLOOR)
                     or row.get("color_flag") == "look")
        shade = row.get("color_shade")
        # A palette mixture's second word is glass or shadow: only a
        # color-name read (shade set) serves it.
        out["color"] = color_words.compact_color_cell(
            c, shade=shade if shade in ("dk", "lt") else None,
            uncertain=uncertain, rich=shade is not None)
    return out


def _model_facing_row(row):
    """A row as the model sees it: hidden fields dropped, color rendered."""
    if isinstance(row, dict) and any(k in row for k in _MODEL_HIDDEN_ROW_FIELDS):
        row = {k: val for k, val in row.items() if k not in _MODEL_HIDDEN_ROW_FIELDS}
    return _render_color_cell(row)


def _num(v):
    """``v`` when it is a number (a bool is not), else None."""
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def join_hints(*parts) -> str:
    """The non-empty hint sentences, one per line."""
    return "\n".join(str(p) for p in parts if p)


# A row list longer than BROAD_DUMP_SOFT_ROWS renders its first
# BROAD_HEAD_ROWS rows (a set crop badges the same rows); the set keeps all.
BROAD_DUMP_SOFT_ROWS = 48
BROAD_HEAD_ROWS = LOOK_SET_ROWS


def _census_counts(rows) -> tuple[Counter, int]:
    """``key=value`` counts over a row set's tags and ``class=<c>``, and how
    many rows carry a name."""
    counts: Counter = Counter()
    named = 0
    for it in rows:
        if not isinstance(it, dict):
            continue
        named += bool((it.get("name") or "").strip())
        if it.get("class"):
            counts[f"class={it['class']}"] += 1
        if isinstance(it.get("tags"), dict):
            counts.update(f"{k}={v}" for k, v in it["tags"].items())
    return counts, named


def _top(counts: dict, n: int) -> dict:
    """The ``n`` largest counts, ties by key."""
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n])


def _query_matched_census(counts: dict, query: str) -> dict:
    """The census entries sharing a token (or a close spelling) with the
    query, top 8: the long tail a global top cut hides."""
    q_tokens = {t for cl in T.fuzzy_prepare_query(query) for t in cl[1]}
    q_long = [t for t in q_tokens if len(t) >= 4]
    matched: dict[str, int] = {}
    for key, n in counts.items():
        toks = set(T.fuzzy_normalize(key).split())
        toks |= {T.singularize(t) for t in toks}
        if toks & q_tokens or any(
                len(ct) >= 4
                # the length gap bounds the edit distance from below
                and abs(len(qt) - len(ct)) <= 0.3 * max(len(qt), len(ct))
                and 1.0 - T.edit_distance(qt, ct) / max(len(qt), len(ct)) >= 0.7
                for qt in q_long for ct in toks):
            matched[key] = n
    return _top(matched, 8)


def _census_note(rows, query: str) -> str:
    """The census note a missed or thin find serves: the tags sharing
    tokens with the query, the top tags and a few names."""
    counts = _census_counts(rows)[0]
    top = _top(counts, 12)
    parts = ", ".join(f"{k}({v})" for k, v in top.items())
    names: list[str] = []
    for it in rows:
        if isinstance(it, dict) and (it.get("name") or "").strip():
            nm = str(it["name"]).strip()
            if nm not in names:
                names.append(nm)
        if len(names) >= 4:
            break
    qm = _query_matched_census(counts, query)
    qparts = ", ".join(f"{k}({v})" for k, v in qm.items())
    note = (f"surface census -- tags sharing tokens with your query: "
            f"{qparts or 'none'}; top tags overall: {parts or 'none'}")
    if names:
        note += "; named e.g. " + ", ".join(repr(n) for n in names)
    return note + "."


def _cut_stands_at(v, body: list, capped: list) -> str | None:
    """For a truncated score_by head: the composite score span it shows and
    the score of the first row past it, so a plateau reads differently from
    a real gap. The composite, since no single key is monotone in a
    multi-key sort."""
    if v.kind != "sorted" or len(capped) >= len(body):
        return None

    def score(row):
        x = _num(row.get("score")) if isinstance(row, dict) else None
        return None if x is None else float(x)

    head = [x for x in map(score, capped) if x is not None]
    if len(head) < 2:
        return None
    nxt = next((x for x in map(score, body[len(capped):])
                if x is not None), None)
    keys = [k for k in v.op_cols if isinstance(k, str)]
    what = " + ".join(f"'{k}'" for k in keys) or "this ordering"
    span = f"{min(head):.6g}..{max(head):.6g}"
    if nxt is None:
        return f"ranked on {what}: this head's score runs {span}"
    return (f"ranked on {what}: this head's score runs {span}, and the next "
            f"row scores {nxt:.6g}")


# The grid's leading columns after ``fid``; attached columns follow in
# first-seen order.
_GRID_COL_PRIORITY = (
    "class", "color", "name", "tags", "xy", "footprint_area_m2",
)

# Repeat the header row every N data rows on a long dump.
_GRID_HEADER_EVERY = 50

# A tags cell renders its first N tags and counts the rest.
_TAG_RENDER_HEAD = 12

# Tag keys the grid does not render: postal and import keys find still
# searches.
_TAG_RENDER_DROP = frozenset({
    "addr_postcode", "addr_city", "oneway", "ref_GB_uprn",
    "tiger_reviewed", "tiger_cfcc", "tiger_source", "tiger_tlid",
    "tiger_separated", "tiger_county", "calcity_object_id", "source_lanes",
})

# Row fields the grid does not render.
_GRID_COL_DROP = frozenset({
    "axis_length_m", "color_conf", "name_match", "score",
})

# The column names the model never sees on any channel: the grid, the
# library's Row, and the key echoes in refusals and notes.
MODEL_UNSERVED_COLUMNS = (
    _GRID_COL_DROP
    | frozenset(_MODEL_HIDDEN_ROW_FIELDS)
    | frozenset(_RENDER_INPUT_ROW_FIELDS))


def _is_gridable(v) -> bool:
    """A value one grid cell renders: a scalar, an xy pair or a flat dict."""
    if isinstance(v, (bool, int, float, str)):
        return True
    if isinstance(v, dict):
        return all(isinstance(x, (bool, int, float, str)) for x in v.values())
    return (isinstance(v, (list, tuple)) and len(v) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in v))


def _fmt_float(v: float) -> str:
    """A stored float in the fewest decimals (4 to 10) that round-trip it,
    trailing zeros trimmed, ``-0.0`` as ``0.0``: values are quantized when
    stamped, so the served number is the stored one."""
    if v == 0.0:
        v = 0.0
    for dp in range(4, 11):
        s = f"{v:.{dp}f}"
        if float(s) == v:
            break
    s = s.rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"


def _grid_cell(v) -> str:
    """One markdown cell: ``-`` when empty, floats by :func:`_fmt_float`, xy
    as ``x, y``, tags as ``k=v; k2=v2``, and ``|`` as ``/``."""
    if v is None or v == "" or (isinstance(v, dict) and not v):
        return "-"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return _fmt_float(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, dict):
        _items = [kv for kv in v.items() if kv[0] not in _TAG_RENDER_DROP]
        if not _items:
            return "-"
        s = "; ".join(f"{k}={val}" for k, val in _items[:_TAG_RENDER_HEAD])
        if len(_items) > _TAG_RENDER_HEAD:
            s += (f"; +{len(_items) - _TAG_RENDER_HEAD} more tags "
                  f"(full set on the node)")
        return s.replace("|", "/")
    if (isinstance(v, (list, tuple)) and len(v) == 2
            and all(isinstance(x, (int, float)) for x in v)):
        return (f"{_fmt_float(float(v[0]))}, {_fmt_float(float(v[1]))}")
    return str(v).replace("|", "/")


def _grid_columns(rows) -> list[str]:
    """The non-empty gridable columns across the model-facing rows: priority
    columns, then the rest in first-seen order."""
    seen: list[str] = []
    seen_set: set = set()
    for row in rows:
        for k, v in row.items():
            if (k == "fid" or k in seen_set or k in _GRID_COL_DROP
                    or v is None or v == ""
                    or (isinstance(v, dict) and not v)):
                continue
            if _is_gridable(v):
                seen_set.add(k)
                seen.append(k)
    prio = [c for c in _GRID_COL_PRIORITY if c in seen_set]
    # A ref's four side columns (one of each pair per row) group where the
    # first appears, in STANDING_SIDE_PREFIXES order.
    grouped: list[str] = []
    for c in seen:
        if c in _GRID_COL_PRIORITY or c in grouped:
            continue
        side = _column_ref(c, STANDING_SIDE_PREFIXES)
        grouped += [c] if side is None else [
            p + side[1] for p in STANDING_SIDE_PREFIXES
            if p + side[1] in seen_set]
    return prio + grouped


def _render_grid_rows(rows, cols) -> str:
    """The model-facing rows as a ``| fid | <cols...> |`` markdown grid, the
    header repeated every _GRID_HEADER_EVERY rows."""
    if cols:
        header = "| fid | " + " | ".join(cols) + " |"
        sep = "|" + " --- |" * (len(cols) + 1)
    else:
        header, sep = "| fid |", "| --- |"
    lines = [header, sep]
    for idx, row in enumerate(rows):
        if idx and idx % _GRID_HEADER_EVERY == 0:
            lines.extend((header, sep))
        cells = [_grid_cell(row.get(k)) for k in cols]
        body = " | ".join(cells)
        lines.append(f"| {row.get('fid') or '?'} | {body} |"
                     if cols else f"| {row.get('fid') or '?'} |")
    return "\n".join(lines)


# The relation column-key prefixes, each followed by its ref's label. The
# side prefixes may name a staged viewpoint handle; the library's rows read
# them, and AUTOATTACH_PREFIXES, to name what writes a missing key.
STANDING_SIDE_PREFIXES = ("left of ", "right of ", "ahead of ", "behind ")
_COMPARE_COL_PREFIXES = ("distance to ", "center distance to ",
                         "bearing from ", "facing ", "contains ",
                         *STANDING_SIDE_PREFIXES,
                         "east of ", "west of ", "north of ", "south of ")
AUTOATTACH_PREFIXES = tuple(
    p for p in _COMPARE_COL_PREFIXES if p != "contains ") + ("color_",)


def _column_ref(key, prefixes) -> "tuple[str, str] | None":
    """``(prefix, ref)`` of a ``<prefix><ref>`` column key, the prefix one
    of ``prefixes``; else None."""
    if isinstance(key, str):
        for p in prefixes:
            if key.startswith(p) and len(key) > len(p):
                return p, key[len(p):]
    return None


# A rendered grid stays under this many chars by eliding columns.
_RENDER_CHAR_BUDGET = 12_000

# Served once per response, under its first grid. xy is surface._row_xy's
# point, distance G.gap_witness, bearings G.bearing.
_TABLE_LEGEND = (
    "reading these tables: xy = [east_m, north_m] world meters, +y is "
    "north (xy is the row's recorded point -- not always its footprint "
    "centroid, and never promised to be a line's midpoint); "
    "'distance to <name>' = meters between nearest edges, 0.0 = "
    "touching; 'facing <name>' = a compass WORD for "
    "which way that row faces that place; '<dir> of <name>' = signed "
    "meters the row lies that way of it, blank where the geometry "
    "declines to say; bearings are compass degrees, 0 = "
    "north, 90 = east, clockwise."
)


def _budget_columns(proj, cols, op_cols) -> tuple[list, int]:
    """``(kept, n_cut)``: ``cols`` of the model-facing rows cut to fit
    ``_RENDER_CHAR_BUDGET``, taken in priority order (``op_cols``, identity,
    the rest) and skipping a column that does not fit; at least one is kept,
    order unchanged."""
    base = 20 * (len(proj) + 2)  # fid cell + pipes per line
    cost = {}
    for c in cols:
        # header and separator overhead, generous so the kept set fits
        w = len(c) + 12
        for row in proj:
            w += len(_grid_cell(row.get(c))) + 3
        cost[c] = w
    if base + sum(cost.values()) <= _RENDER_CHAR_BUDGET:
        return cols, 0
    _named = [c for c in cols if c in set(op_cols or ())]
    _ident = [c for c in cols if c not in _named and c in _GRID_COL_PRIORITY]
    _extra = [c for c in cols if c not in _named and c not in _GRID_COL_PRIORITY]
    kept: set = set()
    total = base
    for c in _named + _ident + _extra:
        if kept and total + cost[c] > _RENDER_CHAR_BUDGET:
            continue
        kept.add(c)
        total += cost[c]
    return [c for c in cols if c in kept], len(cols) - len(kept)


_NO_ORDER_HINT = (
    "no ordering exists on this result, so it renders census + len and no "
    "fids: ordering it by an attached column renders the ordered rows of "
    "THIS handle, and a name or color match returns matched rows directly.")


def _table(head, rows, op_cols, legend) -> "str | None":
    """``rows`` as ``head["rows"]``, a markdown grid; the response's first
    grid takes the legend ``legend`` holds. Returns the elided-columns note,
    or None."""
    proj = [_model_facing_row(r) for r in rows if isinstance(r, dict)]
    cols, n_cut = _budget_columns(proj, _grid_columns(proj), op_cols)
    head["rows"] = _render_grid_rows(proj, cols)
    if legend:
        head["legend"] = legend.pop()
    return (f"{n_cut} column(s) elided from this render to fit; "
            "the set keeps them all") if n_cut else None


def _hint(v) -> str:
    """What an op result's rows are, never how they render: its notes, and
    the line saying why there are none or no order."""
    if v.kind == "compare":
        return join_hints(*v.notes)
    if not v:
        return join_hints(
            v.empty_condition
            or "this call returned no rows: nothing met its condition.",
            *v.notes,
            "this retrieval searched every surface row (prior + runtime)"
            " and matched nothing."
            if v.kind == "miss" else "")
    if not v.ordered:
        return join_hints(*v.notes, _NO_ORDER_HINT)
    return join_hints(*v.notes)


def _envelope(v, legend, cut=None) -> dict:
    """One op result's model-facing envelope (see :func:`compact_results`):
    ``hint`` says what the rows are, ``shown`` how the grid renders them;
    ``cut`` is the stats of the cut that emptied an empty answer."""
    cut = (v.cut_stats if v.kind == "cut" else None) or cut
    if v.kind == "compare":
        head = {"len": len(v)}
    elif not v:
        head = {"len": 0, "items": [], **({"cut_stats": cut} if cut else {})}
    elif not v.ordered:
        counts, named = _census_counts(v)
        head = {"len": len(v), "census": _top(counts, 12), "named": named,
                **({"cut_stats": cut} if cut else {})}
    elif v.name_set:
        # A name's ways tie, so a head would be an arbitrary prefix
        # served as a ranking: the shape renders instead.
        head = {"len": len(v), "name": str(v[0].get("name") or "")}
        set_m = sum(float(r.get("axis_length_m") or 0.0) for r in v)
        if set_m:
            head["length_m"] = round(set_m)
    else:
        body, cover = list(v), None
        if v.kind == "sorted":
            # A window-covering row is 0 m from every on-surface ref and
            # would tie rank 0 of every nearest-first sort.
            covering = [r for r in v if r.get("covers_window")]
            if covering:
                body = [r for r in v if not r.get("covers_window")]
                cf = [str(r.get("fid") or "?") for r in covering]
                cover = (
                    f"{len(covering)} window-covering row(s) "
                    f"({', '.join(cf[:6])}"
                    f"{', ..' if len(cf) > 6 else ''}) hold "
                    f"'distance to' = 0 to "
                    f"every ON-SURFACE ref by containment (their mask "
                    f"contains this whole window; an off-window [x,y] "
                    f"literal ref keeps a real distance) -- omitted from "
                    f"this rendered head; they stay live in this set "
                    f"(full set intact), covers_window marks them "
                    f"wherever they render.")
        capped = (body[:BROAD_HEAD_ROWS] if len(v) > BROAD_DUMP_SOFT_ROWS
                  else body)
        more = None
        if len(capped) < len(body):
            more = join_hints(
                f"showing the first {len(capped)} of {len(v)} row(s); "
                f"the rest stay live in this set -- later ops on it "
                f"read the FULL set, and more_of on it serves the next "
                f"{BROAD_HEAD_ROWS}",
                _cut_stands_at(v, body, capped))
        head = {"len": len(v),
                "fids": [str(r["fid"]) for r in capped if r.get("fid")]}
        shown = join_hints(more, _table(head, capped, v.op_cols, legend),
                           cover)
        if shown:
            head["shown"] = shown
    hint = _hint(v)
    if hint:
        head["hint"] = hint
    if cut:
        head["cut_stats"] = cut   # keeps its place where a branch set it
    return head


def _step_hint(v) -> str:
    """A step's hint as it rides the answer: a step whose rows a later op
    cut, sorted or measured serves only the notes not about those rows."""
    if v.consumed and v:
        return join_hints(*(n for n in v.notes if n not in v.set_notes))
    return _hint(v)


def compact_results(results: dict, *, answer_ref: str = "") -> dict:
    """Each op result as its model-facing envelope (:func:`_envelope`), the
    legend on the first grid; with ``answer_ref``, that one envelope,
    carrying the other ops' hints."""
    legend = [_TABLE_LEGEND]
    if answer_ref not in results:
        out, seen = {}, {}
        for k, v in results.items():
            e = out[k] = _envelope(v, legend)
            if "census" in e:
                sig = repr((sorted(e["census"].items()), e["named"]))
                if sig in seen:
                    del e["census"], e["named"]
                    e["census_ref"] = (
                        f"census identical to {seen[sig]}'s (served once above)")
                else:
                    seen[sig] = str(k)
        return out
    v = results[answer_ref]
    cut = None if v else next((s.cut_stats for s in results.values()
                               if s.kind == "cut" and not s and s.cut_stats),
                              None)
    answer = _envelope(v, legend, cut)
    # The other ops are the call's own steps, refs the model never holds:
    # their hints ride unprefixed, each line once.
    own = str(answer.get("hint") or "").split("\n")
    steps = dict.fromkeys(ln for k, v in results.items() if k != answer_ref
                          if (h := _step_hint(v))
                          for ln in str(h).split("\n"))
    answer["hint"] = join_hints(answer.get("hint"),
                                *(ln for ln in steps if ln not in own))
    if not answer["hint"]:
        del answer["hint"]
    return {answer_ref: answer}
