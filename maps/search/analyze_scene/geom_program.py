"""The analyze executor: runs a compiled op-program over the ``(cands, anchors)`` surface.

:func:`run_program` executes the ``{"ops": [{ref, op, args}, ...]}`` the code
surface compiles, with :mod:`maps.search.analyze_scene.geom_compute`
primitives; every op returns rows, and :func:`compact_results` renders them
for the model. An ``items`` value is a handle -- ``rN`` (a result of this
program or an earlier call) or a window ``rN[a:b]`` -- or a fid list; a ref is
a fid, an ``[x, y]``, or ``{ref, as}`` naming a fid, a point, or a set (a
union).
"""
from __future__ import annotations

import re

import numpy as np

from maps.search.analyze_scene import cell_mesh
from maps.search.analyze_scene import geom_compute as G
from maps.search.analyze_scene import text_match as T
from maps.search.analyze_scene.row_geometry import chains_of, geom_of, mask_center
from maps.search.render.look_render import LOOK_SET_ROWS
from maps.util.xy_token import format_xy_token, is_xy
from maps.search.withhold import withheld


def _quote(v) -> str:
    """A user value as a refusal quotes it: its repr, cut to 120 chars."""
    s = repr(v)
    return s if len(s) <= 120 else s[:120] + "..."


def _is_color_column_key(k) -> bool:
    """A ``color_<word>`` column key (``color_conf`` is a surface field)."""
    return isinstance(k, str) and k.startswith("color_") and k != "color_conf"


def _resolve_color_word(word: str) -> "tuple[str, str | None]":
    """A cue color word on the closed palette: ``(canonical, note)``, the
    note set when the word was normalized (navy -> blue). Off-palette
    raises."""
    w = str(word or "").strip().lower()
    canon = G.normalize_color(w)
    if canon not in G.COLOR_ENUM:
        raise ProgramError(
            f"{_quote(w)} is not a color on the closed palette; color columns "
            f"are color_<word> with word one of: "
            f"{', '.join(G.COLOR_ENUM)} (an open word that maps onto the "
            f"palette is accepted, e.g. navy -> blue).")
    note = (None if canon == w else
            f"color {_quote(w)} maps onto the closed palette as {_quote(canon)}; "
            f"the column is color_{canon}.")
    return canon, note


def _attach_color_column(items, canon):
    """Attach ``color_<canon>`` (the confidence-shrunk color match) to every
    row with a perceived color; a row without one omits the column."""
    col = "color_" + canon
    out = []
    for it in items:
        c = it.get("color") if isinstance(it, dict) else None
        if isinstance(c, str) and c.strip():
            out.append({**it, col: round(
                G.color_match(c, canon, it.get("color_conf")), 3)})
        else:
            out.append(it)
    return out


_COLOR_WORDS = tuple(G.COLOR_FAMILY)


def _color_of(label: str) -> str:
    """The first palette word the label carries as a whole word, else ""."""
    words = set(re.findall(r"[a-z]+", (label or "").lower()))
    return next((w for w in _COLOR_WORDS if w in words), "")


def _modal_attr(n, key):
    """The modal value of ``key`` in the node's ``semantic_attributes``, or None."""
    cc = (n.semantic_attributes or {}).get(key)
    return max(cc.items(), key=lambda kv: kv[1])[0] if cc else None


# Bake bookkeeping some bakes stamp on every row: never served, and never
# searched (as stopwords on every row, ``osm_id`` would donate ``way``).
_UNSERVED_TAGS = frozenset({
    "source", "osm_id", "worldkit_layer", "worldkit_height_m", "source_tag"})


def _row_tags(n) -> dict:
    """A prior row's tags: the node's string-valued OSM tags but _UNSERVED_TAGS."""
    tags = n.tags or {}
    if not isinstance(tags, dict):
        return {}
    return {str(k): v for k, v in tags.items()
            if isinstance(v, str) and str(k).lower() not in _UNSERVED_TAGS}


def _anchor_search_text(n) -> str:
    """A prior row's search text: its name and its tag keys and values."""
    parts = [str(n.name or "")]
    for k, v in _row_tags(n).items():
        parts.append(str(k).replace("_", " "))
        parts.append(str(v).replace("_", " "))
    return " ".join(p for p in parts if p).lower()


def _cand_search_text(n) -> str:
    """A detection row's search text: its class name."""
    return str(n.class_name or "").lower()


class ProgramError(Exception):
    pass


def _color_fields(n) -> dict:
    """``color`` (perceived, else a color word in the class label) plus
    ``color_conf`` / ``color_shade`` / ``color_flag`` where perception set
    them."""
    color = _modal_attr(n, "color")
    fields = {"color": color if color is not None
              else _color_of(str(n.class_name or ""))}
    conf = _modal_attr(n, "color_conf")
    if conf is not None:
        try:
            fields["color_conf"] = round(float(conf), 3)
        except ValueError:
            pass
    shade = _modal_attr(n, "color_shade")
    if shade is not None:
        fields["color_shade"] = shade
    flag = _modal_attr(n, "color_flag")
    if flag:
        fields["color_flag"] = flag
    return fields


def representative_point(chains) -> list:
    """A point ON the geometry, the row's flight target, at 2 dp: a closed
    chain's area centroid when it lies inside the mask, else the arc
    midpoint of the longest chain (a mean over a bent way lies off it)."""
    geom = G.mask_geom(chains)
    if geom["closed"]:
        c = G.loops_area_centroid([ch for ch in chains
                                   if G.is_closed_chain(ch)])
        if c is not None and bool(G.mask_contains(geom, [c])[0]):
            return [round(float(c[0]), 2), round(float(c[1]), 2)]
    p = _polyline_arc_midpoint(max(chains, key=len))
    return [round(float(p[0]), 2), round(float(p[1]), 2)]


# A surface row is a pure function of its node's content and the tile
# geometry, so rows are cached by content (_content_key over the node's whole
# __dict__, so a new MapObject field cannot be invisible to the key): ingest
# returns fresh node objects. Each cache is dropped whole at its cap.
_ROW_CACHE: "dict[tuple, dict | None]" = {}
_ROW_CACHE_MAX = 50_000


class _FindMemo:
    """A find's per-query work and word rarity for the surface lists it was
    built on; ``lists`` pins them so their ids cannot be reused."""

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
            hit = [{**x, "name_match": round(T.find_score(
                       q_data, x.get("search_text") or "", x.get("name"),
                       x.get("tags"), idf=self.idf), 3)}
                   for x in rows]
            # A query that is a tag value reaches every row carrying that
            # tag key, at T.TAG_FAMILY_SCORE, below every exact hit.
            fams = T.tag_family_keys(q_data, rows)
            if fams:
                for r in hit:
                    if (r["name_match"] < T.TAG_FAMILY_SCORE
                            and (r.get("tags") or {}).keys() & fams):
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


def _content_key(v):
    """A hashable summary of ``v`` that changes whenever its content does.
    Arrays and numeric sequences hash their bytes (``cells`` is the bulk of a
    node)."""
    if isinstance(v, (list, tuple)):
        try:
            a = np.asarray(v)
        except ValueError:  # ragged: walk it
            a = None
        if a is None or a.dtype.kind not in "ifb":
            return ("seq", tuple(_content_key(x) for x in v))
        v = a
    if isinstance(v, np.ndarray):
        return ("nd", v.shape, v.dtype.str, hash(v.tobytes()))
    if isinstance(v, dict):
        return ("map", tuple(sorted((str(k), _content_key(x)) for k, x in v.items())))
    return v


def _rows_by_content(kind, nodes, build_row, geom):
    """One row per ``(fid, node)``, reusing unchanged ones. Each is a
    shallow copy, since the caller stamps fields on it; the heavy values
    are shared and read-only."""
    out = []
    for f, n in nodes:
        key = (kind, str(f), geom, tuple(sorted((k, _content_key(v)) for k, v in vars(n).items())))
        if key in _ROW_CACHE:
            row = _ROW_CACHE[key]
        else:
            row = build_row(f, n)
            if len(_ROW_CACHE) >= _ROW_CACHE_MAX:
                _ROW_CACHE.clear()
            _ROW_CACHE[key] = row
        if row is not None:
            out.append(dict(row))
    return out


def build_surface(det_nodes, prior_nodes, origin_xy, cell_m):
    """The ``(cands, anchors)`` surface from detection and prior ``(fid,
    node)`` iterables, both sorted by fid: one row per node, with its mask
    chains and shape scalars when masked and a prior's world centerline."""
    tx, ty = float(origin_xy[0]), float(origin_xy[1])
    cs = float(cell_m)

    def _w(rc):
        return [round(tx + (int(rc[1]) + 0.5) * cs, 2), round(ty + (int(rc[0]) + 0.5) * cs, 2)]

    def _xy(n, mask=None):
        """The row's served point, the ``go_to``/``explore`` flight target: the
        tracker's ``world_xy``, else the mask's :func:`representative_point`,
        else the centroid cell's centre."""
        w = n.world_xy
        if w is not None and len(w) >= 2:
            return [round(float(w[0]), 2), round(float(w[1]), 2)]
        if mask:
            return representative_point(mask)
        c = n.centroid
        return _w(c) if c is not None and len(c) >= 2 else None

    def _cl(n):
        cl = n.centerline
        return [_w(p) for p in cl] if cl else None

    def _cells(n):
        """The node's 2 m mesh as a plain (row, col) int list, so a surface
        round-trips through JSON."""
        c = cell_mesh.cells_of(n)
        return c.tolist() if len(c) else []

    def _axis_q(n):
        # Only a node with a centerline has an axis length.
        if not n.centerline:
            return None
        v = n.axis_length_m
        return round(float(v), 2) if v is not None else None

    def _exact_chains(n) -> list:
        """The node's ``world_contour`` (one ring, way or point) clipped to
        the box of its own cells plus one cell, since the bake clips cells
        to the tile while a contour runs on. A node with no cells is not
        clipped."""
        wc = n.world_contour
        parts = [wc] if wc else []
        chains = [[[float(x), float(y)] for x, y in p] for p in parts if p]
        cl = cell_mesh.cells_of(n)
        if not len(cl) or not chains:
            return chains
        r0, c0 = cl[:, 0].min(), cl[:, 1].min()
        r1, c1 = cl[:, 0].max(), cl[:, 1].max()
        return G.clip_chains_to_box(chains, (
            tx + max(int(c0) - 1, 0) * cs, ty + max(int(r0) - 1, 0) * cs,
            tx + (int(c1) + 2) * cs, ty + (int(r1) + 2) * cs))

    def _mask_fields(n) -> dict:
        # ``mask`` (the contour, else the cell boundary loops), its centroid
        # and the shape scalars; none for a node without either.
        cells = n.cells
        chains = _exact_chains(n)
        # A one-vertex chain (a POI) draws nothing: the cell boundary does.
        def _masked(m) -> dict:
            _c = mask_center(m)
            return {"mask": m, **({"mask_centroid": [round(v, 2) for v in _c]}
                                  if _c is not None else {})}
        if any(len(ch) >= 2 for ch in chains):
            out = _masked(chains)
            shape = G.contour_shape_features(chains)
            if shape:
                out.update(shape)
            return out
        if cells is None or len(cells) == 0:
            return {}
        loops = G.cells_boundary_loops(cells, (tx, ty), cs)
        out = _masked(loops) if loops else {}
        # Below 6 cells the 2 m grid cannot express shape.
        if len(cells) >= 6:
            out.update(G.mask_shape_features(cells, cs))
        return out

    def _cand_row(f, n):
        mf = _mask_fields(n)
        xy = _xy(n, mf.get("mask"))
        if not xy:
            return None
        return {"fid": str(f), "xy": xy, "centerline": None,
                "cells": _cells(n),
                "class": str(n.class_name or ""),
                **_color_fields(n),
                **mf,
                "search_text": _cand_search_text(n)}

    cands = _rows_by_content("c", det_nodes, _cand_row, (tx, ty, cs))
    # A prior row is one way and carries no class: its name and tags say
    # what it is.
    def _anchor_row(f, n):
        mf = _mask_fields(n)
        xy = _xy(n, mf.get("mask"))
        if not xy:
            return None
        return {"fid": str(f), "name": n.name,
                "xy": xy,
                "centerline": _cl(n),
                "cells": _cells(n),
                "axis_length_m": _axis_q(n),
                "tags": _row_tags(n),
                **_color_fields(n),
                **mf,
                "search_text": _anchor_search_text(n)}

    anchors = _rows_by_content("a", prior_nodes, _anchor_row, (tx, ty, cs))
    # A centerline row carries its termini.
    for _ar in anchors:
        _acl = _ar.get("centerline")
        if isinstance(_acl, list) and len(_acl) >= 2:
            _ar["end0_xy"] = [round(float(_acl[0][0]), 2),
                              round(float(_acl[0][1]), 2)]
            _ar["end1_xy"] = [round(float(_acl[-1][0]), 2),
                              round(float(_acl[-1][1]), 2)]
    # Duplicate detections (same rounded xy and class) fold to the one with
    # a color, then the higher color_conf, then the lower fid.
    def _fid_key(f):
        # detected_2 before detected_100; a fid without a numeric tail after.
        s = str(f)
        head, _, tail = s.rpartition("_")
        return ((head, 0, int(tail), "") if tail.isdigit()
                else (s, 1, 0, s))

    _dups: dict = {}
    for c in cands:
        _dups.setdefault((round(float(c["xy"][0]), 1), round(float(c["xy"][1]), 1),
                          c.get("class") or ""), []).append(c)
    # Numeric fid order is mint order.
    cands = sorted((min(g, key=lambda r: (not r.get("color"), -float(r.get("color_conf") or 0.0),
                                          _fid_key(r["fid"])))
                    for g in _dups.values()), key=lambda c: _fid_key(c["fid"]))
    anchors.sort(key=lambda a: _fid_key(a["fid"]))
    # A mask containing every row position is 0 m from every row:
    # ``covers_window`` lets compare state that once instead.
    _all_rows = cands + anchors
    if len(_all_rows) >= 3:
        _xs = [r["xy"][0] for r in _all_rows]
        _ys = [r["xy"][1] for r in _all_rows]
        _wx0, _wx1 = min(_xs), max(_xs)
        _wy0, _wy1 = min(_ys), max(_ys)
        _pts = [r["xy"] for r in _all_rows]
        for r in _all_rows:
            m = r.get("mask")
            if not m:
                continue
            _mx = [p[0] for lp in m for p in lp]
            _my = [p[1] for lp in m for p in lp]
            if (not _mx or min(_mx) > _wx0 or max(_mx) < _wx1
                    or min(_my) > _wy0 or max(_my) < _wy1):
                continue
            if bool(G.mask_contains(G.mask_geom(m), _pts).all()):
                r["covers_window"] = True
    return cands, anchors


def build_surface_from_graph(nodes_iter, origin_xy, cell_m):
    """:func:`build_surface` over a graph's ``(fid, node)`` pairs, split by
    ``node.provenance``."""
    det, prior = [], []
    for fid, node in nodes_iter:
        (det if node.provenance == "detected" else prior).append((fid, node))
    return build_surface(det, prior, origin_xy, cell_m)


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
    if not (isinstance(c, str) and c.strip()):
        if "color_shade" in row or "color_flag" in row:
            row = dict(row)
            row.pop("color_shade", None)
            row.pop("color_flag", None)
        return row
    conf = row.get("color_conf")
    uncertain = ((isinstance(conf, (int, float)) and conf < _COLOR_CONF_FLOOR)
                 or row.get("color_flag") == "look")
    shade = row.get("color_shade")
    # A palette mixture's second word is glass or shadow: only a color-name
    # read (shade set) serves it.
    rich = shade is not None
    out = {**row, "color": G.compact_color_cell(
        c, shade=shade if shade in ("dk", "lt") else None, uncertain=uncertain, rich=rich)}
    out.pop("color_shade", None)
    out.pop("color_flag", None)
    return out


def _model_facing_row(row):
    """A row as the model sees it: hidden fields dropped, color rendered."""
    if isinstance(row, dict) and any(k in row for k in _MODEL_HIDDEN_ROW_FIELDS):
        row = {k: val for k, val in row.items() if k not in _MODEL_HIDDEN_ROW_FIELDS}
    return _render_color_cell(row)


class Result(list):
    """An op's rows and how :func:`compact_results` renders them, by
    ``kind``: ``broad`` (find, crossings, ends), ``miss`` (an empty find),
    ``cut`` (filter_items, stratum), ``compare`` or ``sorted`` (score_by).
    ``consumed`` marks rows a later op read; the other attributes are render
    inputs its op sets."""

    ordered: bool = False
    name_set: bool = False
    consumed: bool = False
    op_cols: list = []
    notes: list = []
    set_notes: list = []
    cut_stats: dict = {}
    empty_condition: "str | None" = None

    def __init__(self, rows=(), kind: str = "broad", **attrs):
        super().__init__(rows)
        self.kind = kind
        vars(self).update(attrs)


def join_hints(*parts) -> str:
    """The non-empty hint sentences, one per line."""
    return "\n".join(str(p) for p in parts if p)


# Above this many tied ways under one name, a find renders the set's shape.
_NAME_SET_SHAPE_ROWS = 8

# A row list longer than _BROAD_DUMP_SOFT_ROWS renders its first
# _BROAD_HEAD_ROWS rows (a set crop badges the same rows); the rN keeps all.
_BROAD_DUMP_SOFT_ROWS = 48
_BROAD_HEAD_ROWS = LOOK_SET_ROWS


def _census_counts(rows) -> tuple[dict, int]:
    """``key=value`` counts over a row set's tags and ``class=<c>``, and how
    many rows carry a name."""
    counts: dict[str, int] = {}
    named = 0
    for it in rows:
        if not isinstance(it, dict):
            continue
        if (it.get("name") or "").strip():
            named += 1
        c = it.get("class")
        if c:
            counts[f"class={c}"] = counts.get(f"class={c}", 0) + 1
        tags = it.get("tags")
        if isinstance(tags, dict):
            for k, v in tags.items():
                key = f"{k}={v}"
                counts[key] = counts.get(key, 0) + 1
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
        if toks & q_tokens:
            matched[key] = n
            continue
        hit = False
        for qt in q_long:
            for ct in toks:
                if len(ct) < 4:
                    continue
                if 1.0 - T.edit_distance(qt, ct) / max(len(qt),
                                                        len(ct)) >= 0.7:
                    hit = True
                    break
            if hit:
                break
        if hit:
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

    def _num(row):
        val = row.get("score") if isinstance(row, dict) else None
        return float(val) if isinstance(val, (int, float)) else None

    head = [x for x in (_num(r) for r in capped) if x is not None]
    if len(head) < 2:
        return None
    nxt = next((x for x in (_num(r) for r in body[len(capped):])
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
# python surface's Row, and the key echoes in refusals and notes.
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
    extras = [c for c in seen if c not in _GRID_COL_PRIORITY]
    # A ref's four side columns (one of each pair per row) group where the
    # first appears, in left/right/ahead/behind order.
    _side = re.compile(r"^(left of|right of|ahead of|behind) (.+)$")
    _order = {"left of": 0, "right of": 1, "ahead of": 2, "behind": 3}
    grouped: list[str] = []
    done: set = set()
    for c in extras:
        if c in done:
            continue
        m = _side.match(c)
        if m is None:
            grouped.append(c)
            done.add(c)
            continue
        ref = m.group(2)
        family = sorted(
            (x for x in extras if (_m2 := _side.match(x)) and
             _m2.group(2) == ref),
            key=lambda x: _order[_side.match(x).group(1)])
        grouped.extend(family)
        done.update(family)
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
# side prefixes may name a staged viewpoint handle; the python surface reads
# them, and _AUTOATTACH_PREFIXES, to name what writes a missing key.
_STANDING_SIDE_PREFIXES = ("left of ", "right of ", "ahead of ", "behind ")
_COMPARE_COL_PREFIXES = ("distance to ", "center distance to ",
                         "bearing from ", "facing ", "contains ",
                         *_STANDING_SIDE_PREFIXES,
                         "east of ", "west of ", "north of ", "south of ")
_AUTOATTACH_PREFIXES = tuple(
    p for p in _COMPARE_COL_PREFIXES if p != "contains ") + ("color_",)

# A wall-clock net: a union measures members x items gaps.
_UNION_MAX_MEMBERS = 400


def _sanitize_token(lbl) -> str:
    """A token as a column label: lowercase, ``[^a-z0-9]+`` collapsed to
    ``_``. The compiler builds its keys with it too."""
    base = re.sub(r"[^a-z0-9]+", "_", str(lbl).lower()).strip("_")
    return base or "ref"


# A rendered grid stays under this many chars by eliding columns.
_RENDER_CHAR_BUDGET = 12_000

# Served once per response, under its first grid. xy is build_surface's
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


def compact_results(results: dict, *, answer_ref: str = "") -> dict:
    """Each op result as its model-facing envelope: an ordered list ``{len,
    fids, rows}`` (a markdown grid, past ``_BROAD_DUMP_SOFT_ROWS`` only the
    first ``_BROAD_HEAD_ROWS``), an unordered one ``{len, census, named}``, an
    empty one ``{len: 0, items: [], hint}``; a compare, always a step a later
    op reads, only ``{len, hint}``. The legend rides the first grid. With ``answer_ref`` the
    response is that one envelope, carrying the other ops' hints. Every rN
    keeps its full rows."""
    legend = [True]

    def table(head, rows, op_cols):
        proj = [_model_facing_row(r) for r in rows if isinstance(r, dict)]
        cols, n_cut = _budget_columns(proj, _grid_columns(proj), op_cols)
        head["rows"] = _render_grid_rows(proj, cols)
        if legend:
            head["legend"] = _TABLE_LEGEND
            legend.clear()
        return (f"{n_cut} column(s) elided from this render to fit; "
                "the rN keeps them all") if n_cut else None

    def env(v):
        notes, kind = v.notes, v.kind
        cut = v.cut_stats if kind == "cut" else None
        if kind == "compare":
            head = {"len": len(v)}
            hint = join_hints(
                "" if v else
                "compare received NO items -- the items ref resolved to an "
                "empty list.",
                *notes)
        elif not v:
            head = {"len": 0, "items": []}
            if cut:
                head["cut_stats"] = cut
            head["hint"] = join_hints(
                v.empty_condition
                or "this op returned NO items: nothing met its condition.",
                *notes,
                "this retrieval searched every surface row"
                + ("" if withheld("prior") else " (prior + runtime)")
                + " and matched nothing."
                if kind == "miss" else "")
            return head
        elif not v.ordered:
            counts, named = _census_counts(v)
            head = {"len": len(v), "census": _top(counts, 12), "named": named}
            if cut:
                head["cut_stats"] = cut
            head["hint"] = join_hints(*notes, _NO_ORDER_HINT)
            return head
        elif v.name_set:
            # A name's ways tie, so a head would be an arbitrary prefix
            # served as a ranking: the shape renders instead.
            head = {"len": len(v), "name": str(v[0].get("name") or "")}
            set_m = sum(float(r.get("axis_length_m") or 0.0) for r in v)
            if set_m:
                head["length_m"] = round(set_m)
            hint = join_hints(*notes)
        else:
            body, cover = list(v), None
            if kind == "sorted":
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
                        f"this rendered head; they stay live in this rN "
                        f"(full set intact), covers_window marks them "
                        f"wherever they render.")
            capped = (body[:_BROAD_HEAD_ROWS] if len(v) > _BROAD_DUMP_SOFT_ROWS
                      else body)
            more = None
            if len(capped) < len(body):
                more = join_hints(
                    f"showing the first {len(capped)} of {len(v)} row(s); "
                    f"the rest stay live in this rN -- later ops on it "
                    f"read the FULL set, and more_of(<rN>) serves the next "
                    f"{_BROAD_HEAD_ROWS}",
                    _cut_stands_at(v, body, capped))
            head = {"len": len(v),
                    "fids": [str(r["fid"]) for r in capped if r.get("fid")]}
            hint = join_hints(more, table(head, capped,
                                          v.op_cols),
                              cover, *notes)
        if hint:
            head["hint"] = hint
        if cut:
            head["cut_stats"] = cut
        return head

    if answer_ref not in results:
        out, seen = {}, {}
        for k, v in results.items():
            e = out[k] = env(v)
            if "census" in e:
                sig = repr(sorted(e["census"].items()))
                if sig in seen:
                    del e["census"], e["named"]
                    e["census_ref"] = (
                        f"census identical to {seen[sig]}'s (served once above)")
                else:
                    seen[sig] = str(k)
        return out
    answer = env(results[answer_ref])
    # The other ops are the call's own steps, refs the model never holds:
    # their hints ride unprefixed, each line once. A step whose rows a later
    # op cut, sorted or measured serves only the notes not about those rows.
    own = str(answer.get("hint") or "").split("\n")

    def _step_hint(v):
        if v.consumed and v:
            return join_hints(*(n for n in v.notes if n not in v.set_notes))
        return env(v).get("hint")
    steps = dict.fromkeys(ln for k, v in results.items() if k != answer_ref
                          if (h := _step_hint(v))
                          for ln in str(h).split("\n"))
    answer["hint"] = join_hints(answer.get("hint"),
                                *(ln for ln in steps if ln not in own))
    if not answer["hint"]:
        del answer["hint"]
    return {answer_ref: answer}


def _grouped_keys(present) -> str:
    """The keys with each family of more than three per-instance keys
    collapsed ('distance to building_* (100 instances, e.g. ...)'); a key's
    family is its longest column prefix plus its stem's leading word."""
    prefixes = sorted(_COMPARE_COL_PREFIXES + ("color_",), key=len, reverse=True)
    fams: dict[str, list] = {}
    for k in present:
        pre = next((p for p in prefixes if k.startswith(p)), None)
        if pre is None:
            fams.setdefault(k, []).append(k)
            continue
        stem = k[len(pre):]
        base = re.match(r"^([A-Za-z]+)_", stem)
        fams.setdefault(pre + (base.group(1) + "_" if base else stem),
                        []).append(k)
    parts = []
    for fam, members in sorted(fams.items()):
        if len(members) <= 3:
            parts.extend(members)
        else:
            parts.append(
                f"{fam}* ({len(members)} instances, e.g. {members[0]})")
    return ", ".join(parts)


def _present_keys_hint(items, numeric: bool) -> str:
    """For an op on an absent key: the served keys the items DO carry --
    numeric ones (grouped) for a sort or a numeric cut, every scalar one
    for any other cut."""
    kinds = (int, float) if numeric else (int, float, str, bool)
    present = sorted({k for it in items if isinstance(it, dict)
                      for k, v in it.items() if isinstance(v, kinds)
                      and v != "" and k not in MODEL_UNSERVED_COLUMNS})
    if numeric:
        return (f"numeric keys present on these items: {_grouped_keys(present)}."
                if present else "these items carry NO numeric key.")
    return (f"keys present on these items: {present}." if present
            else "these items carry NO scalar key.")


def _polyline_arc_midpoint(verts) -> list:
    """The arc-length midpoint of a world-meter polyline."""
    arcs = G.arc_lengths(verts)
    half = arcs[-1] / 2.0
    for i in range(len(verts) - 1):
        if arcs[i + 1] >= half:
            seg = arcs[i + 1] - arcs[i]
            t = (half - arcs[i]) / seg if seg > 0 else 0.0
            return [float(verts[i][0]) + t * (float(verts[i + 1][0]) - float(verts[i][0])),
                    float(verts[i][1]) + t * (float(verts[i + 1][1]) - float(verts[i][1]))]
    return [float(verts[-1][0]), float(verts[-1][1])]


def _attach_standing_columns(items, frame, with_distance=False):
    """Copies of the rows with a staged viewpoint's side columns, standing
    at ``from_xy`` looking along ``up_unit``: 'left of'/'right of <h>' and
    'ahead of'/'behind <h>' in metres, the key present naming the side (0
    lands right/ahead). With ``with_distance``, 'distance to <h>' from the
    stand point."""
    h = str(frame["handle"])
    fx, fy = (float(frame["from_xy"][0]), float(frame["from_xy"][1]))
    ux, uy = (float(frame["up_unit"][0]), float(frame["up_unit"][1]))
    rx, ry = G.cw((ux, uy))
    left_k, right_k = f"left of {h}", f"right of {h}"
    ahead_k, behind_k = f"ahead of {h}", f"behind {h}"
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


class _Run:
    """One ``run_program`` call: the surface, the seed, the results so far
    and the op running."""

    def __init__(self, cands, anchors, seed=None, staged=None, standing=None):
        self.cands, self.anchors = cands, anchors
        self.by_fid = {c["fid"]: c for c in cands}
        self.by_fid.update({a["fid"]: a for a in anchors})
        self.seed = seed or {}
        self.staged = staged or {}
        self.standing = self.staged.get(standing) if standing else None
        self.results: dict = {}
        self.op = None

    def avail(self, kind: str) -> str:
        """The live handles and fids (bounded) a refusal names."""
        def _cap(fids):
            return f"{fids[:12]}" + (f" (+{len(fids) - 12} more)" if len(fids) > 12 else "")
        if kind == "list":
            srcs = ", ".join(str(k) for k in self.results) or (
                "(no result handles yet -- run a find first)")
            return f"list sources: {srcs}; fids present: {_cap(sorted(self.by_fid))}"
        return f"available fids: {_cap(sorted(self.by_fid))}"

    def is_ref(self, tok) -> bool:
        """A handle: ``rN`` or ``rN[a:b]`` naming a result of this program
        or of an earlier call. A fid never matches."""
        if not isinstance(tok, str):
            return False
        base = tok.partition("[")[0]
        return base in self.results or base in self.seed

    def resolve(self, tok):
        """``rN`` -> its rows (this program's shadowing an earlier call's);
        ``rN[a:b]`` -> that window of them."""
        base, _, win = tok.partition("[")
        if base in self.results:
            v = self.results[base]
            v.consumed = True
        elif base in self.seed:
            v = self.seed[base]
        else:
            _known = sorted(set(self.results) | set(self.seed))
            raise ProgramError(
                f"unknown ref {_quote(tok)}: an op defined with ref 'r0' is read back "
                f"as r0. Refs persist across your analyze calls (redefining one "
                f"replaces it). "
                f"valid refs so far: {_known or '(none)'}")
        return _window(v, win, tok) if win else v


class _Args(dict):
    """An op's args; an absent required arg raises a ProgramError."""

    def __init__(self, d, cx):
        super().__init__(d)
        self._cx = cx

    def __missing__(self, key):
        raise ProgramError(
            f"op {_quote(self._cx.op)} is missing required arg {_quote(key)}")


def _window(cur, win, token):
    """``rN[a:b]``: those rows, keeping the source's ``kind``, ``ordered``
    and ``op_cols``. Selecting none of a non-empty set raises."""
    part = "[" + win
    lo, hi = (int(b) for b in win[:-1].split(":"))
    rows = cur[lo:hi]
    if isinstance(cur, Result):
        rows = Result(rows, cur.kind, ordered=cur.ordered, op_cols=cur.op_cols)
    if not rows and cur:
        raise ProgramError(
            f"window {part} in ref {_quote(token)} selected 0 of "
            f"{len(cur)} rows: this ref holds {len(cur)} row(s) and "
            f"the window begins at or past the end of them "
            f"(or begins at or after where it stops).")
    return rows


def _as_rows(cx, arg):
    """An ``items`` arg -> point-bearing rows: a handle's rows, a fid's row,
    an ``[x, y]`` as an xy-only row, or a list of those (handles flatten).
    A list of rows comes back as itself, keeping its order attributes."""
    v = cx.resolve(arg) if cx.is_ref(arg) else arg
    if is_xy(v) or (isinstance(v, str) and v in cx.by_fid):
        v = [v]
    if not isinstance(v, list):
        raise ProgramError(
            f"cannot resolve rows from {_quote(arg)}: items is an rN ref, a "
            f"fid on this surface, an [x,y], or a list of them. "
            f"{cx.avail('list')}")
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
            out.extend(_as_rows(cx, it))
        else:
            raise ProgramError(
                f"cannot resolve a point in the list from item {_quote(it)}: each item must "
                f"be a fid in this surface, a {{'fid':...}} dict, an [x,y], or an rN. "
                f"{cx.avail('point')}")
    return out


def _named_rows(cx, el, need):
    """Every row a crossings/ends ref names (a road is several ways): a fid,
    an rN, a list of them, or {ref, as}. A row off the surface counts when
    it carries ``need``."""
    if isinstance(el, dict):
        el = el.get("ref")
    out: list = []
    for t in (el if isinstance(el, list) else [el]):
        vals = [t]
        if isinstance(t, str) and t not in cx.by_fid:
            vals = cx.resolve(t)
        for x in vals:
            n = None
            if isinstance(x, str):
                n = cx.by_fid.get(x)
            elif isinstance(x, dict):
                n = cx.by_fid.get(x.get("fid")) or (x if x.get(need) else None)
            if n is not None and n not in out:
                out.append(n)
    return out


def _relate_ran(key, items) -> bool:
    """Whether the compare that writes a ``facing <lab>`` key ran: its
    sibling ``distance to <lab>`` is on a row, so a facing absent everywhere
    means every row touches the place."""
    if not (isinstance(key, str) and key.startswith("facing ")):
        return False
    sib = "distance to " + key[len("facing "):]
    return any(isinstance(it, dict) and sib in it for it in items)


def _contact_fact(cx, key) -> str:
    """For an absent 'distance to' / 'facing' a <fid> whose mask contains
    the whole surface window: why neither is stamped. Returns the fact, or
    ''."""
    if isinstance(key, str) and key.startswith(("distance to ", "facing ")):
        _p = (key[len("facing "):] if key.startswith("facing ")
              else key[len("distance to "):])
        _n = cx.by_fid.get(_p)
        if isinstance(_n, dict) and _n.get("covers_window"):
            return (f"{_p}'s mask contains the whole surface window, so "
                    f"'distance to {_p}'=0 for every row (containment, "
                    f"not proximity) and neither the distance nor the "
                    f"facing is stamped; 'center distance to "
                    f"{_p}'/'bearing from {_p}' attach per-row and DO "
                    f"order/filter. ")
    return ""


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
    cond = (f"find matched nothing: no row scored >= {T.FUZZY_THRESHOLD} "
            f"against query {_quote(query)} (query tokens: {tokens}).")
    for cl in q_data:
        w = str(cl[0] or "").strip()
        if not w or " " in w:
            continue
        canon = G.normalize_color(w)
        if canon in G.COLOR_ENUM:
            return cond, [
                f"{_quote(w)} is a color word on the closed palette: color "
                f"is not row text -- it rides the color_{canon} column, which "
                f"attaches when the color word is named."]
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
    n_proper = sum(1 for r in scored if r["name_match"] >= T.NAME_TIER_FLOOR
                   and str(r.get("name") or "").strip())
    n_exact = n_named if n_proper else sum(
        1 for r in scored if r["name_match"] > T.TOKEN_BAG_CEIL)
    return n_exact, n - n_exact


def _is_name_set(matched) -> bool:
    """Whether the matches are one name's tied ways (more than
    _NAME_SET_SHAPE_ROWS, all exact, one non-empty name)."""
    exact = [r for r in matched if r.get("name_match", 0.0) >= T.NAME_TIER_FLOOR]
    names = {str(r.get("name") or "") for r in exact}
    return (len(exact) > _NAME_SET_SHAPE_ROWS and len(exact) == len(matched)
            and len(names) == 1 and bool(next(iter(names))))


def _match_notes(scored, n, band_cut) -> list:
    """What a matched find says: how many matched and whether their order is
    real, the tier split, the band cut and the head."""
    vals = [r["name_match"] for r in scored[:n]]
    rest = len(scored) - n
    if vals[0] == vals[-1]:
        order = (f"all equally good matches at match {vals[0]}, so this order "
                 f"is arbitrary and ordering them by a cue column chooses it")
    else:
        order = f"best-first, match {vals[0]}..{vals[-1]}"
        top_n = sum(1 for x in vals if x == vals[0])
        if top_n > 1:
            order += (f"; the first {top_n} tie at match {vals[0]}, so their "
                      "order among themselves is arbitrary and ordering them "
                      "by a cue column chooses it")
    notes = [f"{n} of {len(scored)} rows matched ({order}"
             + (f"; {rest} sub-threshold row(s) NOT in this rN" if rest else "")
             + ")."]
    n_ex = sum(1 for x in vals if x >= T.NAME_TIER_FLOOR)
    n_tg = sum(1 for x in vals if T.KV_TIER_BASE <= x < T.NAME_TIER_FLOOR)
    if n_ex and n - n_ex:
        notes.append(f"tiers: {n_ex} exact, {n_tg} tagged, {n - n_ex - n_tg} "
                     "fuzzy -- .exact / .tagged / .fuzzy on this result hold "
                     "each as a set.")
    if band_cut:
        notes.append(f"matched as a NAME: {band_cut} row(s) under the name "
                     f"tier are a different feature and are NOT in this rN.")
    head = "; ".join(f"{r.get('fid')} {_match_label(r)} (match {r['name_match']})"
                     for r in scored[:n][:4]
                     if r["name_match"] >= T.FUZZY_THRESHOLD)
    if head:
        lead = "Best" if vals[0] >= T.NAME_TIER_FLOOR else "Head"
        notes.append(f"{lead}: {head}")
    return notes


def _name_cut(rows, want, query, idf) -> tuple[list, str | None]:
    """``name=``: the rows NAMED it, matched as a name, never fuzzily, and
    the note saying how many it kept. None raises."""
    nq = T.fuzzy_prepare_query(want)
    named = [r for r in rows if isinstance(r, dict)
             and T.find_score(nq, r.get("search_text") or "", r.get("name"),
                              r.get("tags"), idf=idf)
             >= T.NAME_TIER_FLOOR]
    if not named:
        raise ProgramError(
            f"nothing found is NAMED {_quote(want)} and matches {_quote(query)}"
            ". A name is matched as a NAME, never fuzzily -- check the "
            f"spelling, or drop the class word and say find({_quote(want)}).")
    if len(named) == len(rows):
        return named, None
    return named, (f"name={_quote(want)} kept {len(named)} of {len(rows)} by "
                   "NAME; the fuzz band does not apply to a name")


def _find(cx, a):
    """Search the surface: a query returns its match set best-first; no
    query browses every row."""
    notes: list = []
    set_notes: list = []  # the notes about the matched rows
    matched = False      # a query ran and matched (the rN is ordered)
    name_set = False     # tied ways of one name: render the shape
    empty_cond = None
    surface = rows = cx.cands + cx.anchors
    query = a.get("query")
    fid_hit = cx.by_fid.get(str(query or "").strip()) if query else None
    if fid_hit is not None:
        # The fuzzy scorer matches text, never ids.
        rows, matched = [fid_hit], True
        notes.append(f"query {str(query).strip()!r} is a live fid -- that "
                     "row returned directly.")
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
                + ("" if withheld("prior") else
                   " OSM multi-value tag VALUES use ';' "
                   "(cuisine=pizza;kebab) -- to match one, query that "
                   "single value ('cuisine=pizza')."))
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
                notes.append("no matched row carries "
                             + ", ".join(repr(t) for t in dead[:4])
                             + " -- those words did not narrow anything")
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
        op_cols = [f"left of {h}", f"right of {h}", f"ahead of {h}",
                   f"behind {h}"]
    if not rows and query:
        return Result(kind="miss", notes=notes, empty_condition=empty_cond)
    return Result(rows, ordered=matched, name_set=name_set, op_cols=op_cols,
                  notes=notes, set_notes=set_notes)


def _x_label(nodes, el):
    names: list = []
    for n in nodes:
        nm = n.get("name")
        if nm and str(nm) not in names:
            names.append(str(nm))
    if names:
        return names[0] if len(names) == 1 else " / ".join(names[:3])
    if isinstance(el, dict) and el.get("as"):
        return str(el["as"])
    return ", ".join(str(n.get("fid")) for n in nodes[:3])


def _crossings(cx, a):
    """The places two features meet: one row per distinct crossing, its fid
    the [x, y] token of its whole-metre point. None raises."""
    sides = []
    for _el in a["refs"]:
        _nodes = _named_rows(cx, _el, "xy")
        _polys = [p for n in _nodes for p in chains_of(n)]
        if not _polys:
            raise ProgramError(
                f"crossings ref {_quote(_el)} names no geometry on this "
                f"surface -- a crossing needs two features that HAVE "
                f"a shape. {cx.avail('point')}")
        sides.append((_x_label(_nodes, _el), _polys,
                      {str(n.get("fid")) for n in _nodes}))
    # A row on both sides would hand back its own corners.
    _shared = sides[0][2] & sides[1][2]
    if _shared:
        _same = "the same feature" if sides[0][2] == sides[1][2] \
            else "the same feature in part"
        raise ProgramError(
            f"{sides[0][0]} and {sides[1][0]} are {_same} "
            f"({', '.join(sorted(_shared)[:4])}), and nothing crosses "
            f"itself -- a feature's own corners are not crossings. "
            f"Name two DIFFERENT features (find each one and read its "
            f"row).")
    (la, fa), (lb, fb) = ((sides[0][0], sides[0][2]),
                          (sides[1][0], sides[1][2]))
    xpts, nearest = G.polyline_crossings(sides[0][1], sides[1][1])
    if not xpts:
        # Sweep a bounded set of same-named sibling pairs for a crossing.
        by_fid = cx.by_fid
        _sa = [c for c in by_fid.values()
               if str(c.get("name")) == str(la)
               and str(c.get("fid")) not in fa][:8]
        _sb = [c for c in by_fid.values()
               if str(c.get("name")) == str(lb)
               and str(c.get("fid")) not in fb][:8]
        _hit = None
        _budget = 24
        for _ra in ([by_fid[f] for f in sorted(fa) if f in by_fid][:8] + _sa):
            for _rb in ([by_fid[f] for f in sorted(fb)
                         if f in by_fid][:8] + _sb):
                if str(_ra.get("fid")) in fa and str(_rb.get("fid")) in fb:
                    continue          # the pair already tested
                if _budget <= 0:
                    break
                _budget -= 1
                _px, _ = G.polyline_crossings(chains_of(_ra), chains_of(_rb))
                if _px:
                    _hx, _hy = (float(round(_px[0][0])),
                                float(round(_px[0][1])))
                    _hit = (str(_ra.get("fid")), str(_rb.get("fid")),
                            format_xy_token(_hx, _hy))
                    break
            if _hit or _budget <= 0:
                break
        _sib = ""
        if _hit:
            _sib = (f" But the NAMES do cross elsewhere: "
                    f"{_hit[0]} and {_hit[1]} (same-named "
                    f"segments) touch at {_hit[2]} -- "
                    f"intersection on those rows returns it.")
        elif _sa or _sb:
            _sib = (f" {len(_sa)} other {la} segment(s) and "
                    f"{len(_sb)} other {lb} segment(s) exist; "
                    f"no swept pair touches either.")
        raise ProgramError(
            f"{la} and {lb} do not meet HERE: the tested rows "
            f"({', '.join(sorted(fa)[:4])} against "
            f"{', '.join(sorted(fb)[:4])}) come no closer than "
            f"{nearest:.1f} m, and a crossing is where they "
            f"TOUCH.{_sib} A point near one of them is not their "
            f"intersection; relate to ONE of them instead "
            f"('... near <feature>').")
    rows = Result()
    # Whole metres: the fid is the coordinate, and the model spells whole
    # numbers only.
    _seen: set = set()
    for _x, _y in xpts:
        _x, _y = float(round(_x)), float(round(_y))
        tok = format_xy_token(_x, _y)
        # Rounding can collapse two crossings onto one fid.
        if tok in _seen:
            continue
        _seen.add(tok)
        rows.append({"fid": tok, "xy": [_x, _y], "name": f"{la} x {lb}"})
    rows.ordered = True
    rows.notes = [
        f"{len(rows)} distinct crossing{'' if len(rows) == 1 else 's'} "
        f"of {la} and {lb}"
        + ("." if len(rows) == 1 else
           " -- each is its OWN row with its OWN fid, so narrow to the "
           "one you mean (order or cut this set) rather than averaging "
           "them; a point between crossings is at none of them.")]
    rows.set_notes = rows.notes
    return rows


def _ends(cx, a):
    """A road's termini (end0_xy / end1_xy) as rows."""
    rows = Result()
    _n_end = 0
    _no_line: list = []
    for _el in a["refs"]:
        _nodes = _named_rows(cx, _el, "centerline")
        if not _nodes:
            raise ProgramError(
                f"ends ref {_quote(_el)} names no row on this surface. "
                f"{cx.avail('point')}")
        for _nd in _nodes:
            _cl = _nd.get("centerline")
            if not (isinstance(_cl, list) and len(_cl) >= 2):
                _no_line.append(
                    f"{_nd.get('fid')}"
                    + (f" ({_nd.get('name')})" if _nd.get("name") else ""))
                continue
            _lab = str(_nd.get("name") or _nd.get("fid"))
            for _which, _pt in (("end0", _cl[0]), ("end1", _cl[-1])):
                _xy = [float(round(_pt[0])), float(round(_pt[1]))]
                rows.append({"fid": f"end_{_n_end}", "xy": _xy,
                             "name": f"{_lab} {_which}"})
                _n_end += 1
    if not rows:
        _named = ", ".join(_no_line[:4]) + (
            f" (+{len(_no_line) - 4} more)" if len(_no_line) > 4 else "")
        raise ProgramError(
            f"none of those rows carries a recorded centerline -- "
            f"{_named} have no line geometry on this surface, so "
            f"no termini exist for them. ends reads a centerline; "
            f"end0_xy/end1_xy on a row is the same fact served as "
            f"columns.")
    rows.ordered = True
    rows.notes = [
        f"{len(rows)} terminus rows; each is its OWN row with its "
        f"own fid, so index the one you mean."]
    rows.set_notes = rows.notes
    return rows


def _score_by(cx, a):
    """The sort: a signed, key-normalized composite over exactly the
    columns the emphasis names."""
    items = _as_rows(cx, a["items"])
    _sb_notes: list = []
    emphasis: dict = {}
    for _el in a["emphasis"]:
        _k = _el["key"]
        if _is_color_column_key(_k):
            # color_navy weights color_blue, noted.
            _canon, _cnote = _resolve_color_word(_k[len("color_"):])
            if _cnote is not None:
                _sb_notes.append(_cnote)
            _k = "color_" + _canon
        emphasis[_k] = float(_el["w"])
    for _k in emphasis:
        # Naming a color column attaches it.
        if _is_color_column_key(_k) and not any(
                it.get(_k) is not None for it in items):
            items = _attach_color_column(items, _k[len("color_"):])
    # A key no row carries would be a silent constant term.
    _dropped_color: list = []
    for _k in list(emphasis):
        if items and not any(isinstance(it.get(_k), (int, float))
                             for it in items):
            if _is_color_column_key(_k):
                # Nothing has a perceived colour before a flight sees it: the
                # term drops, unless it is the only one.
                if len(emphasis) > 1:
                    _dropped_color.append(_k)
                    del emphasis[_k]
                    continue
                raise ProgramError(
                    f"nothing orders by {_quote(_k)} here: no row carries a "
                    f"perceived color, so no color column attaches (rows "
                    f"without a color read omit color_<word> columns). "
                    f"{_present_keys_hint(items, True)}")
            raise ProgramError(
                f"nothing orders by {_quote(_k)} here: no row carries a "
                f"numeric {_quote(_k)}. "
                f"{_contact_fact(cx, _k)}"
                f"{_present_keys_hint(items, True)}")
    for _k in _dropped_color:
        _sb_notes.append(
            f"{_quote(_k)} ranked nothing here: no row on this surface has "
            f"been SEEN yet, so none carries a perceived colour. The "
            f"other keys still rank. A row gets a colour once a "
            f"flight has passed over it.")
    keys = sorted(emphasis)  # deterministic; the sum is order-independent
    r = Result(G.score_by(items, keys, [emphasis[k] for k in keys]), "sorted")
    r.ordered = True
    r.op_cols = list(keys)
    if _sb_notes:
        r.notes = _sb_notes
    return r


def _filter_items(cx, a):
    """The op that drops rows, keeping its input's order: ``le`` on a
    numeric column, ``truthy`` on a present key, ``on_side`` keeps a row its
    cardinal column places on the side or does not place, ``in_or_blank``
    keeps a word column reading one of ``value`` or blank."""
    items = _as_rows(cx, a["items"])
    key, opr, val = a["key"], a["op"], a.get("value")
    _in_ordered = bool(getattr(items, "ordered", False))
    if opr == "le":
        if items and not any(isinstance(it.get(key), (int, float)) for it in items):
            raise ProgramError(
                f"filter_items op {_quote(opr)} on key {_quote(key)}: no item "
                f"carries a numeric {_quote(key)}. "
                f"{_contact_fact(cx, key)}"
                f"{_present_keys_hint(items, True)}")
        # The nearest excluded row, the bound that would admit it and how
        # many rows that bound admits.
        kept: list = []
        excluded: list = []
        for it in items:
            v = it.get(key)
            (kept if v is not None and v <= val else excluded).append(it)

        def _eff(it):
            v = it.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            return v

        stats = {"cut": f"{len(items)}->{len(kept)}"}
        _excl_vals = [(e, it) for it in excluded if (e := _eff(it)) is not None]
        if _excl_vals:
            ne_v, ne_it = min(_excl_vals, key=lambda t: t[0])
            stats["nearest_excluded"] = {
                "fid": str(ne_it.get("fid") or "?"),
                key: round(float(ne_it.get(key)), 3),
            }
            stats["admit_at"] = round(float(ne_v), 3)
            stats["n_at_admit"] = sum(
                1 for it in items
                if (v := _eff(it)) is not None and v <= ne_v)
        r = Result(kept, "cut")
        r.cut_stats = stats
        r.ordered = _in_ordered
        r.op_cols = [key]
        return r
    if (items and not any(key in it for it in items)
            and not _relate_ran(key, items)):
        raise ProgramError(
            f"filter_items op {_quote(opr)} on key {_quote(key)}: no item carries "
            f"{_quote(key)}. "
            f"{_contact_fact(cx, key)}"
            f"{_present_keys_hint(items, False)}")
    if opr == "in_or_blank":
        # The caller passes every word that counts (south: southeast,
        # southwest too).
        _on = [it for it in items if it.get(key) is None or it.get(key) in val]
        _blank = sum(1 for it in _on if it.get(key) is None)
        _want = list(val) if isinstance(val, (list, tuple)) else [val]
        stats = {
            "cut": f"{len(items)}->{len(_on)}",
            "counts_as": _want,
            "reads_one_of_them": len(_on) - _blank,
            "reads_another_word_dropped": len(items) - len(_on),
            "unmeasured_kept": _blank,
        }
        if not (len(_on) - _blank):
            stats["nothing_reads_those_words"] = (
                f"no {'' if withheld('prior') else 'surveyed '}row reads "
                f"any of {_want} in {_quote(key)}; "
                f"the rows kept are the ones the geometry never "
                f"measured (a row TOUCHING the reference has no "
                f"facing word). Widen, or read the column and "
                f"see which words are actually there.")
        r = Result(_on, "cut")
        r.cut_stats = stats
    elif opr == "on_side":
        def _ok(it):
            v = it.get(key)
            return v is None or (isinstance(v, (int, float))
                                 and not isinstance(v, bool) and v > 0)
        _on = [it for it in items if _ok(it)]
        _blank = sum(1 for it in _on if it.get(key) is None)
        stats = {
            "cut": f"{len(items)}->{len(_on)}",
            "on_side": len(_on) - _blank,
            "other_side_dropped": len(items) - len(_on),
            "no_side_kept": _blank,
        }
        if not (len(_on) - _blank):
            stats["nothing_on_that_side"] = (
                ("no row lies on that side" if withheld("prior") else
                 "the prior map holds nothing on that side")
                + " -- what is there has to be SEEN"
                + ("" if withheld("look_for") else
                   ": look_for the class, then explore the place to "
                   "segment it"))
        r = Result(_on, "cut")
        r.cut_stats = stats
    else:  # truthy
        r = Result((it for it in items if it.get(key)), "cut")
    r.ordered = _in_ordered
    r.op_cols = [key]
    return r


def _stratum(cx, a):
    """A find's match tier (exact / tagged / fuzzy) as a set, in input
    order."""
    items = _as_rows(cx, a["items"])
    tier = str(a["tier"]).strip().lower()
    if not any(isinstance(it.get("name_match"), (int, float)) for it in items):
        raise ProgramError(
            "these rows carry no match tiers -- strata come off a "
            "find's own result."
            + ("" if withheld("look_for") else
               " A pass's minted rows are all THE thing the "
               "watchlist named; there is nothing to stratify."))
    _lo, _hi = {"exact": (T.NAME_TIER_FLOOR, None),
                "tagged": (T.KV_TIER_BASE, T.NAME_TIER_FLOOR),
                "fuzzy": (None, T.KV_TIER_BASE)}[tier]

    def _in_tier(it):
        v = it.get("name_match")
        if not isinstance(v, (int, float)):
            return False
        return (_lo is None or v >= _lo) and (_hi is None or v < _hi)
    r = Result((it for it in items if _in_tier(it)), "cut")
    r.ordered = bool(getattr(items, "ordered", False))
    return r


def _union_members(cx, ref_el, as_label) -> list:
    """The member rows of a union ref ``[fid | rN | [x, y], ...]``: a fid's
    row, a handle's rows (a row off the surface, a crossing, rides its own
    xy) and an ``[x, y]`` as an xy-only row."""
    members: list = []
    for x in ref_el:
        if is_xy(x):
            members.append({"xy": [float(x[0]), float(x[1])]})
            continue
        if cx.is_ref(x):
            got = [cx.by_fid.get(r.get("fid"), r) for r in cx.resolve(x)
                   if isinstance(r, dict)
                   and (r.get("fid") in cx.by_fid or is_xy(r.get("xy")))]
            lbl = as_label or _sanitize_token(x)
            if not got:
                raise ProgramError(
                    f"{lbl} is empty, so 'distance to {lbl}' is undefined on "
                    f"every row and the ordering would be meaningless."
                    + ("" if withheld("look_for") else
                       " If these are things the map does not carry (a "
                       "pool, solar panels, a shed, a car), nothing is "
                       "segmented until look_for names the class and a "
                       "flight goes over the place -- arm it, fly, then "
                       "find them again."))
            if len(got) > _UNION_MAX_MEMBERS:
                raise ProgramError(
                    f"{lbl} holds {len(got)} rows and a union folds at most "
                    f"{_UNION_MAX_MEMBERS}: measuring to the nearest of that "
                    f"many is a whole-map scan per row. Narrow it first "
                    f"(within=, a cardinal, or a tighter query), then "
                    f"measure to what is left.")
            members.extend(got)
            continue
        node = cx.by_fid.get(x)
        if node is None:
            raise ProgramError(
                f"union ref member {_quote(x)} is not a fid on this surface. "
                f"{cx.avail('point')}")
        members.append(node)
    return members


def _row_geom_center(cx, node):
    """``(geom, centre)`` of a row: its mask, else its point. Geometry is
    re-read by fid, since op output rows carry no mask."""
    return geom_of(cx.by_fid.get(node.get("fid")) or node)


def _union_center(cx, members):
    """``(member geoms, centre)`` of a union: the centre is the members'
    area-weighted centroid, else the mean of their centres, else None."""
    geoms: list = []
    wx = wy = wa = 0.0
    ucs: list = []
    for node in members:
        g, c = _row_geom_center(cx, node)
        geoms.append(g)
        if c is not None:
            ucs.append(c)
            a_m = G.loops_area(node.get("mask") or [])
            wx += a_m * float(c[0])
            wy += a_m * float(c[1])
            wa += a_m
    if wa > 0:
        return geoms, [wx / wa, wy / wa]
    if ucs:
        return geoms, [sum(float(c[0]) for c in ucs) / len(ucs),
                       sum(float(c[1]) for c in ucs) / len(ucs)]
    return geoms, None


def _unique_label(label, used: set, fid=None) -> str:
    """``label``, or on a collision ``label_<fid>`` (a row ref) and then
    ``label_<n>`` counting from 2."""
    lab, k = label, 2
    while lab in used:
        lab = f"{label}_{fid}" if fid is not None and k == 2 else f"{label}_{k}"
        k += 1
    used.add(lab)
    return lab


def _compare(cx, a):
    """Attach each ref's columns to every item row, in input order:
    'distance to' (footprint gap, 0.0 on contact), 'facing' (the compass
    word of the gap vector, absent on contact), 'center distance to' and
    'bearing from' (centroid to centroid), and a signed '<dir> of' column
    per requested cardinal. A ref is a surface row, an [x, y], a staged
    viewpoint, or ``{ref: [fid | rN | [x, y], ...]}``, a union measured as
    one place."""
    items = _as_rows(cx, a["items"])
    _in_ordered = bool(getattr(items, "ordered", False))

    notes: list = []
    used_labels: set = set()
    standing: list = []         # staged viewpoint handles, in ref order

    # The cardinals asked for per ref (named by fid or label).
    _card_req: dict = {}
    for _el in (a.get("cardinals") or []):
        _card_req.setdefault(str(_el.get("ref")), []).append(_el["dir"])

    # One entry per ref: its label, geometry and cardinals.
    refs: list[dict] = []
    for el in a.get("refs") or []:
        as_label = None
        ref_el = el
        if isinstance(el, dict):
            if el.get("as"):
                as_label = _sanitize_token(el["as"])
            ref_el = el["ref"]
        if (isinstance(el, dict) and isinstance(ref_el, (list, tuple))
                and ref_el and not is_xy(ref_el)
                and all(isinstance(x, str) or is_xy(x) for x in ref_el)):
            kind, payload = "union", _union_members(cx, ref_el, as_label)
            lab = _unique_label(as_label or "ref", used_labels)
            geom, center = _union_center(cx, payload)
        elif is_xy(ref_el):
            kind, payload = "point", [float(ref_el[0]), float(ref_el[1])]
            lab = _unique_label(as_label or "xy", used_labels)
            geom, center = G.mask_geom([[payload]]), payload
        else:
            if not isinstance(ref_el, str):
                raise ProgramError(
                    f"compare refs element {_quote(ref_el)} must be a bare fid string, "
                    f"an rN result, an [x,y] literal, or {{ref:..., as:...}}.")
            payload = cx.by_fid.get(ref_el)
            if payload is None and ref_el in cx.staged:
                # A staged viewpoint attaches its side columns.
                if ref_el not in standing:
                    standing.append(ref_el)
                continue
            if payload is None:
                raise ProgramError(
                    f"compare ref {_quote(ref_el)} is not a fid on this "
                    f"surface. {cx.avail('point')}")
            kind = "row"
            lab = _unique_label(as_label or _sanitize_token(payload["fid"]),
                                used_labels, payload["fid"])
            geom, center = _row_geom_center(cx, payload)
        _fid = payload.get("fid") if kind == "row" else None
        want = next((_card_req[_k] for _k in (_fid, lab)
                     if _k and _k in _card_req), None)
        _pts = G.points_from_geoms(geom if isinstance(geom, list)
                                   else [geom]) if want else None
        # A window-covering ref's gap columns would be 0 on every row: one
        # note says so instead.
        covers = bool(kind == "row" and payload.get("covers_window"))
        if covers:
            notes.append(
                f"ref {lab} ('{payload.get('name') or payload['fid']}'): its "
                f"mask contains the whole surface window, so 'distance to "
                f"{lab}'=0 for EVERY row (containment, not proximity) and "
                f"'facing {lab}' has no direction -- those two columns are "
                f"not stamped; 'center distance to {lab}'/'bearing from "
                f"{lab}' (centroid vector) ride per-row.")
        refs.append({"kind": kind, "payload": payload, "lab": lab,
                     "geom": geom, "center": center, "covers": covers,
                     "cards": (want, _pts) if _pts is not None else None,
                     "bb": G.stack_bboxes(geom) if kind == "union" else None,
                     "contains": []})

    out_rows: list = []
    for i, it in enumerate(items):
        row = {k: v for k, v in it.items() if k not in _MODEL_HIDDEN_ROW_FIELDS}
        # Items resolve like refs, so a ref compared to itself is exactly 0.
        item_geom, it_c = _row_geom_center(cx, it)
        for ref in refs:
            kind, payload, lab = ref["kind"], ref["payload"], ref["lab"]
            if ref["cards"] is not None and it_c is not None:
                _dirs, _rpts = ref["cards"]
                for _d in _dirs:
                    _v = G.cardinal_offset(_rpts, it_c, _d)
                    # Absent, never 0.0, when there is no side to name.
                    if _v is not None:
                        row[f"{_d} of {lab}"] = round(_v, 1)
            ref_geom, center = ref["geom"], ref["center"]
            if ref["covers"]:
                d = None
            elif kind == "union":
                d, wr, wi = G.set_gap(ref_geom, ref["bb"], item_geom)
            else:
                d, wr, wi = G.gap_witness(ref_geom, item_geom)
            if d is not None:
                dr = round(d, 2)
                row[f"distance to {lab}"] = dr
                if dr > 0.0 and wr is not None:
                    # The witness pair runs ref->item; a row faces item->ref.
                    row[f"facing {lab}"] = G.cardinal(wi, wr)
                elif (dr == 0.0
                      and not isinstance(ref_geom, list)
                      and item_geom["closed"]
                      and len(item_geom["A"])
                      and len(ref_geom["V"])
                      and not (kind == "row" and payload.get("fid")
                               and str(payload.get("fid"))
                               == str(row.get("fid")))
                      and bool(G.mask_contains(item_geom, ref_geom["V"]).all())):
                    # every ref vertex inside the row's own mask
                    row[f"contains {lab}"] = True
                    ref["contains"].append(str(row.get("fid") or f"item[{i}]"))
            if center is not None and it_c is not None:
                dcr = round(G.dist(center, it_c), 2)
                row[f"center distance to {lab}"] = dcr
                if dcr > 0.0:
                    row[f"bearing from {lab}"] = round(G.bearing(center, it_c), 1)
        out_rows.append(row)
    # Rows whose own mask contains a ref, per ref, named in one note; it
    # states a fact of each row it names, so it outlives a later cut or sort.
    for ref in refs:
        _cf, lab = ref["contains"], ref["lab"]
        if _cf:
            _cfs = f"{_cf[:8]}" + (
                f" (+{len(_cf) - 8} more)" if len(_cf) > 8 else "")
            notes.append(
                f"ref {lab}: row(s) {_cfs} read 'distance to {lab}'=0 because "
                f"their OWN mask CONTAINS {lab} (every vertex of the ref sits "
                f"inside the row's footprint -- containment, not adjacency); "
                f"each such row carries 'contains {lab}'=true. distance reads "
                f"0 on ANY mask contact, so a 0 in a nearest-first read spans "
                f"touching neighbors AND containers.")
    for _h in standing:
        out_rows = _attach_standing_columns(
            out_rows, cx.staged[_h], with_distance=True)
        notes.append(
            f"ref {_h} is your staged viewpoint: its side columns ('left "
            f"of/right of/ahead of/behind {_h}') + 'distance to {_h}' "
            f"(meters from where you stand) ride every row.")
    r = Result(out_rows, "compare")
    r.notes = notes
    r.ordered = _in_ordered
    return r


OPS = {
    "find": _find,
    "crossings": _crossings,
    "ends": _ends,
    "compare": _compare,
    "filter_items": _filter_items,
    "score_by": _score_by,
    "stratum": _stratum,
}


def run_program(cands, anchors, program, seed_results=None,
                staged=None, standing=None):
    """Run ``{"ops": [{ref, op, args}, ...]}`` (dependency order) over the
    surface rows; returns ``{"results": {ref: rows}}``. A handle defined here
    shadows its ``seed_results`` entry; ``staged`` maps stand() handles to
    ``{"handle", "from_xy", "up_unit"}`` and ``standing`` names the one whose
    side columns ride every find."""
    cx = _Run(cands, anchors, seed_results, staged, standing)
    ops = program["ops"]
    for op in ops:
        cx.op = op["op"]
        # A null arg is an absent one: a handle bound to a row without a fid
        # compiles to items=None.
        try:
            cx.results[op["ref"]] = OPS[op["op"]](cx, _Args(
                {k: v for k, v in op["args"].items() if v is not None}, cx))
        except ProgramError as exc:
            exc.op = cx.op  # the caller's repeated-error class
            raise
    return {"results": cx.results}
