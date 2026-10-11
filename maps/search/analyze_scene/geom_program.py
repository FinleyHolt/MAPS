"""The analyze executor: runs a compiled op-program over the ``(cands, anchors)`` surface.

``code_surface.compile_steps`` turns the model's run-block calls into a small
program of ops, and :func:`run_program` executes it with ONLY
:mod:`maps.search.analyze_scene.geom_compute` primitives and a few list
helpers. Every op returns rows; :func:`compact_results` renders them for the
model. Program shape (the compiled dialect)::

  {"ops": [{"ref": "r1", "op": "find", "args": {"query": "car"}},
           {"ref": "r2", "op": "compare",
            "args": {"items": "r1", "refs": [{"ref": "prior_3", "as": "lot"}]}},
           {"ref": "r3", "op": "score_by",
            "args": {"items": "r2", "emphasis": [{"key": "distance to lot", "w": -1}]}}]}

An ``items`` value is a handle -- ``rN`` (a result of this program or an
earlier call) or a window ``rN[a:b]`` -- or a fid list; a ref is a fid, an
``[x, y]``, or ``{ref, as}`` naming a fid, a point, or a set (a union).
"""
from __future__ import annotations

import copy
import re

from maps.search.analyze_scene import cardinal_side
from maps.search.analyze_scene import cell_mesh
from maps.search.analyze_scene import geom_compute as G
from maps.search.analyze_scene.cardinal_side import CARDINALS
from maps.search.analyze_scene import text_match as T
from maps.search.render.look_render import LOOK_SET_ROWS
from maps.util.node_fingerprint import fingerprint_node
from maps.util.xy_token import format_xy_token, is_xy
from maps.search.withhold import withheld

# Column names that are not columns -> the redirect the error teaches.
_RENAMED_KEYS = {
    "color_score": ("the named color columns color_<word> (word on the "
                    "closed palette, e.g. color_blue); naming one anywhere "
                    "a column key is accepted attaches it"),
}


def _reject_renamed_key(key):
    if key in _RENAMED_KEYS:
        raise ProgramError(
            f"{key!r} is not a column; it was replaced by "
            f"{_RENAMED_KEYS[key]}.")


# --- the named color-column family: color_<word> ---------------------------
# The parameter rides in the key name: naming a color column anywhere a
# column key is accepted attaches it. The closed palette is G.COLOR_ENUM;
# G.normalize_color maps an open cue word onto it (navy -> blue), noted on
# the result envelope.


def _is_color_column_key(k) -> bool:
    """A key requesting a color column: the ``color_`` prefix. ``color_conf``
    is a base surface field (perception's read confidence), not a column
    request, so it is excluded."""
    return isinstance(k, str) and k.startswith("color_") and k != "color_conf"


def _resolve_color_word(word: str) -> "tuple[str, str | None]":
    """Resolve a cue color word onto the closed palette. Returns
    ``(canonical, note)`` where note states the normalization when the typed
    word differs; raises a loud ProgramError off-palette."""
    w = str(word or "").strip().lower()
    canon = G.normalize_color(w)
    if canon not in G.COLOR_ENUM:
        raise ProgramError(
            f"{w!r} is not a color on the closed palette; color columns "
            f"are color_<word> with word one of: "
            f"{', '.join(G.COLOR_ENUM)} (an open word that maps onto the "
            f"palette is accepted, e.g. navy -> blue).")
    note = (None if canon == w else
            f"color {w!r} maps onto the closed palette as {canon!r}; "
            f"the column is color_{canon}.")
    return canon, note


def _attach_color_column(items, canon):
    """Attach ``color_<canon>`` = the conf-shrunk, family-soft cue match on
    every color-carrying row; rows without a perceived color omit the column
    (the existing missing-value idiom — score_by treats it as the neutral).
    Multiple color columns coexist on the same rows."""
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


# Per-op signatures appended to an op's ProgramError (``; signature: ...``).
# The analyze path cuts each error at that marker, but the marker still
# decides where the repeated-error escalation's appended facts are cut, and
# a grounding find's error is served whole, so the suffix bytes stay.
_OP_SIGNATURE = {
    "stratum": ("stratum{items, tier} -- a find's own match tier served as "
                "a set: tier is exact (the rows that ARE the queried "
                "thing), tagged (the tag-carriers), or fuzzy (the "
                "word-sharers). A view: input order kept, nothing else "
                "dropped, the full set stays live behind its handle."),
    "find": ("find([query], [color], [provenance]) -- SEARCH the world-state: "
             "query=fuzzy free text over each row's name + tags (priors) or "
             "detector class (detections); returns THE MATCH SET best-first "
             "(len = the match count; sub-threshold rows stay out -- a "
             "query-less find browses the full surface). Match quality is "
             "tiered-lexical, best-first: an exact name, then the full "
             "MULTI-WORD query phrase word-bounded inside the name, then "
             "an exact tag hit (a key=value pair, or the whole query equal "
             "to one tag key/value), then token/typo fuzz -- a single-word "
             "query scores names only through the fuzz band (the word "
             "inside a longer proper name is a mention, not a hit), so its "
             "exact tag hit outranks any name mention. A NAME that hits "
             "answers alone: the lesser tiers are cut, and the set of its "
             "ways IS the street -- a VIEW of them, like dict.keys(): it "
             "re-reads the world, so len() grows as flights fold ground "
             "in, and it does not subscript, because ties have no first. "
             "list() freezes it. A key=value "
             "query matches tag keys AND values; the literal UPPERCASE token OR "
             "separates alternatives (scored as the max over them). "
             "color=perceptual "
             "(attaches the named color column color_<word> on every "
             "color-carrying row, drops none), "
             "provenance=any|prior|runtime"),
    "crossings": ("crossings(refs) -- CONSTRUCT the places two features MEET. "
                  "refs is EXACTLY TWO elements, each a bare fid, an 'rN', a "
                  "[fid,...] list (one road is routinely several ways -- pass "
                  "them all), or {ref:..., as:'label'}. Returns ONE ROW PER "
                  "DISTINCT CROSSING, each carrying its own fid (its [x, y] "
                  "token) and its own xy, ordered west-to-east; chain "
                  "compare/score_by/filter_items on them like any rows. Two "
                  "features MEET where their shapes TOUCH, and two touches "
                  "at two places are two crossings. Never a "
                  "centroid over several -- a centre between crossings is at "
                  "none of them. If they never meet the op ERRORS with their "
                  "measured nearest approach"),
    "compare": ("compare(items, [refs], [frame]) -- items a candidate LIST (a "
                "narrowed rN); at least one of refs/frame present. "
                "refs: a LIST whose elements are each a bare fid, an 'rN' that "
                "resolved to ONE row, an [x,y] literal, {ref:..., as:'label'}, "
                "or {ref:[fid,...], as:'label'} = the union of the named "
                "masks as ONE ref ('distance to <label>' = the gap to the "
                "nearest member; center/bearing from the union's "
                "area-weighted centroid). "
                "ATTACHES each ref's SHAPE-natural columns onto every item "
                "row (input fields PASS THROUGH). A POINT/POLYGON ref, four "
                "columns: 'distance to <ref>' = the "
                "FOOTPRINT GAP item-mask to ref-mask (0.0 = overlap OR touch -- "
                "contact, not necessarily 'inside'; a footprint-less mask "
                "measures from its centroid); 'facing <ref>' = a compass "
                "WORD (north, southeast) for the direction item->ref along "
                "that min-gap vector -- which way the row FACES the ref, "
                "already inverted, no arithmetic to do (ABSENT on contact -- "
                "no local direction exists); 'center distance to "
                "<ref>' = center-to-center; 'bearing from <ref>' = centroid->"
                "centroid compass bearing ref->item (defined on contact too; "
                "absent only when the centroids coincide). A ref that is "
                "SEVERAL DISJOINT PLACES under one name serves the two gap "
                "columns and NEITHER center column -- a centre over two ways "
                "at opposite ends of the map is at neither of them. "
                "A LINE ref "
                "(a fid with a stored centerline, or an [[x,y],...] polyline "
                "literal), TWO columns: 'along <ref>' (arc m from its "
                "end0_xy end) + 'distance to <ref>' (unsigned lateral m off "
                "that chart) -- its own chart, not hull-gap columns. A ref whose "
                "mask contains the whole surface window serves the ONE "
                "containment fact instead of stamping zero distance on every "
                "row (center distance / bearing still attach per-row). "
                "Bearings in compass degrees, 0=N, clockwise. Column label per "
                "ref: a bare fid keeps that fid ('distance to prior_3'); an "
                "rN resolving to ONE surface row uses that row's FID (never "
                "r0); a bare "
                "[x,y] literal labels the family 'xy' ('distance to xy') "
                "and {ref:[x,y], as:'label'} names those "
                "columns 'distance to <label>'/... instead (TWO bare [x,y] "
                "literals in "
                "one compare is an error -- label each); an rN holding "
                "constructed geometry (a fitted axis, a corner point) keeps "
                "its handle token (r0 -> 'distance to r0'); your as:'label' "
                "always "
                "overrides. "
                "frame takes EXACTLY ONE form: reference:'auto' (no declared view -- "
                "the pool's own axis), {along:<line fid|[[x,y],...]>, [toward:<fid|"
                "[x,y]|compass>]} (THE CHART: travel the line; 'along <ref>' + the "
                "side columns follow the travel), or {toward:<fid|[x,y]|compass>} "
                "(the viewer look: stand at the pool center, look at toward). "
                "as:'label' names the columns."
                " Every geometry slot is CLASS-BLIND; toward also takes "
                "north|south|east|west. The chart attaches 'along <ref>' "
                "(arc meters, exact on bends) + 'left of/right of <ref>' "
                "(lateral meters on that side of the travel; toward "
                "declares it, else undeclared) + 'distance to <ref>' "
                "(either side); the viewer look attaches each ref's own "
                "left/right/ahead/behind of the view center. "
                "Columns (label rule as refs): 'left of/right of/ahead "
                "of/behind <ref>' meters — the column PRESENT names the "
                "side (0.0 on-axis lands right/ahead). Read the "
                "one-line frame receipt (the view "
                "check) before counting. An ATTACH, not a sort (input order "
                "preserved)"),
    "score_by": ("score_by(items, emphasis) -- items a LIST of dicts; THE sort: "
                 "orders by EXACTLY the columns you weight. emphasis is a "
                 "REQUIRED non-empty list [{key:<column>, w:<signed number>}] "
                 "naming EVERY discriminator that should count and its sign "
                 "(NEGATIVE=lower-raw-is-better e.g. a distance, POSITIVE="
                 "higher-raw-is-better e.g. a color_<word> column); nothing is "
                 "auto-added. The composite is the w-weighted sum of each "
                 "key's normalized value (spatial keys min-max to [0,1] over "
                 "the rows carrying them; color_<word>/name_match centered on "
                 "their 0.5 neutral; a missing/flat key contributes its "
                 "neutral) and rank-0 is the LARGEST composite -- w>0 ranks "
                 "larger raw values first, w<0 ranks smaller raw values "
                 "first. Single-key sort: emphasis=[{key:<k>, w:-1|1}]; "
                 "abs:true on an element reads |value| (a SIGNED column's "
                 "magnitude: sort 'distance to <ref>' -- nearest-the-"
                 "axis-first is {key:'distance to <ref>', w:-1})."),
    "filter_items": "filter_items(items, key, [op], [value]) -- items a LIST, key a SCALAR field",
}

# _color_of reads a color token from a class label; _color_fields prefers the
# perceived color on semantic_attributes and falls back to it. A row without
# a perceived color carries no color_<word> column.
_COLOR_WORDS = ("white", "black", "gray", "silver", "red", "blue", "green",
                "yellow", "orange", "brown", "tan", "beige", "gold", "maroon", "purple")


def _color_of(label: str) -> str:
    for w in _COLOR_WORDS:
        if w in (label or "").lower():
            return w
    return ""


def _modal_attr(n, key):
    """The modal value of ``key`` on the node's ``semantic_attributes``
    ({value: count} buckets), or None."""
    cc = (getattr(n, "semantic_attributes", None) or {}).get(key)
    return max(cc.items(), key=lambda kv: kv[1])[0] if cc else None


# Tag keys withheld from the model-facing tags (search text + rendered tags):
# the GT-label ``description`` tags (the catalog instructions are sampled
# from) and the bake's ``source`` provenance (the fid and the find
# ``provenance`` arg already carry it).
_SEARCH_TAG_DENYLIST = frozenset({"description", "descriptions", "source"})


def _row_tags(n) -> dict:
    """A prior row's model-facing tags: the node's raw OSM tags verbatim
    (string values), minus only the bookkeeping keys (_SEARCH_TAG_DENYLIST)."""
    tags = getattr(n, "tags", None) or {}
    if not isinstance(tags, dict):
        return {}
    return {str(k): v for k, v in tags.items()
            if isinstance(v, str) and str(k).lower() not in _SEARCH_TAG_DENYLIST}


# Bake-provenance keys, withheld from the searchable free text only (the row
# keeps every tag; the exact tag tiers and ``tag_family_keys`` read them).
# They are bookkeeping stamped on every row and act as fuzzy-match stopwords
# (``osm_id: way/...`` donates ``way`` to every row). All four are also in
# ``_TAG_RENDER_DROP``.
_SEARCH_PROVENANCE_DROP = frozenset({
    "osm_id", "worldkit_layer", "worldkit_height_m", "source_tag"})


def _anchor_search_text(n) -> str:
    """Searchable free text for a PRIOR row: name + raw OSM tag keys and
    values, minus the bake-provenance keys (_SEARCH_PROVENANCE_DROP) and the
    denylisted tags. A 'parking' query reaches amenity=parking,
    service=parking_aisle and parking=surface rows through keys as well as
    values. Coordinates never enter it."""
    parts = [str(getattr(n, "name", "") or "")]
    for k, v in _row_tags(n).items():
        if str(k) in _SEARCH_PROVENANCE_DROP:
            continue
        parts.append(str(k).replace("_", " "))
        parts.append(str(v).replace("_", " "))
    return " ".join(p for p in parts if p).lower()


def _cand_search_text(n) -> str:
    """Searchable free text for a DETECTION row: class name only. Color is
    matched through the ``color`` field, not free text."""
    return str(getattr(n, "class_name", "") or "").lower()


class ProgramError(Exception):
    pass



def _color_fields(n) -> dict:
    """``{"color": ...}`` plus ``color_conf``/``color_shade``/``color_flag``
    only when perception set them (omitted, never None, otherwise). The color
    is the perceived one, else a color token in the class label."""
    color = _modal_attr(n, "color")
    fields = {"color": color if color is not None
              else _color_of(str(getattr(n, "class_name", "") or ""))}
    conf = _modal_attr(n, "color_conf")
    if conf is not None:
        try:
            fields["color_conf"] = round(float(conf), 3)
        except ValueError:
            pass
    # _mask_noise_color folds shade/flag into the ``col`` cell.
    shade = _modal_attr(n, "color_shade")
    if shade is not None:
        fields["color_shade"] = shade
    flag = _modal_attr(n, "color_flag")
    if flag:
        fields["color_flag"] = flag
    return fields


def _mask_center(mask):
    """The centre a mask may honestly claim as its position — or ``None``.

    ONE place has a centre; several places do not. ``center distance to
    <ref>`` and ``bearing from <ref>`` are served off this point, so a mask
    of several disjoint pieces (a name that resolves to ways far apart)
    serves no centre. ``distance to <ref>``
    is the gap to the nearest piece and still reads.
    """
    return (G.loops_area_centroid(mask)
            if mask and G.chain_components(mask) == 1 else None)


def representative_point(chains) -> list:
    """A point ON the geometry -- what a row may honestly serve as a POSITION.

    The other half of :func:`_mask_center`'s law. That one answers "where is
    the centre of this thing" and refuses when there are several things;
    this one answers "somewhere this thing IS", which every row can answer
    and which ``act.move`` needs to be able to fly to.

    A vertex or cell mean is not on a way bent around a block: it sits in
    the middle of the block, which is the one place the way is not. So a
    closed chain takes its area centroid WHEN that lands inside the mask
    (the ordinary building -- a filled footprint does contain its own
    centroid, and nothing moves for it), and otherwise the boundary's arc
    midpoint; an open chain takes the arc midpoint outright. Several chains
    resolve on the longest, which is a piece of the feature and therefore a
    place it is.

    Rounded to the 2 dp every other served point is spelled at.
    """
    if not any(chains):
        return [0.0, 0.0]
    geom = G.mask_geom(chains)
    if geom["closed"]:
        c = G.loops_area_centroid([ch for ch in chains
                                   if G.is_closed_chain(ch)])
        if c is not None and bool(G.mask_contains(geom, [c])[0]):
            return [round(float(c[0]), 2), round(float(c[1]), 2)]
    longest = max(chains, key=len)
    if len(longest) < 2:
        return [round(float(longest[0][0]), 2), round(float(longest[0][1]), 2)]
    p = _polyline_arc_midpoint(longest)
    return [round(float(p[0]), 2), round(float(p[1]), 2)]


# A surface row is a pure function of its node's content and the tile
# geometry, so an unchanged node is not rebuilt. The key hashes the node's
# content (fingerprint_node), not its identity: ingest returns fresh objects,
# and a hand-kept dirty list could go stale.
_ROW_CACHE: "dict[tuple, dict | None]" = {}
# Steady state is one entry per node per tile. The cap is a runaway backstop;
# on overflow the cache is dropped whole and the next rebuild refills it.
_ROW_CACHE_MAX = 50_000

# A row's mask geometry (G.mask_geom + centre), built once per mask list.
# Keyed on the list's identity, pinned by the entry so the id cannot be
# reused; the row cache shares one mask list per unchanged node. Dropped
# whole at the cap.
_MASK_GEOM: dict = {}
_MASK_GEOM_MAX = 50_000

# The retrieval is memoised per surface: a find's scored rows are a pure
# function of (surface, provenance, query, colour), and a program asks the
# same query repeatedly. The memo keys on the identity of the two surface
# lists (replaced whole on every rebuild) and empties when either changes.
# It hands out copies: a result never aliases another result's dicts.
_FIND_MEMO: dict = {"lists": None, "idf": {}, "scored": {}, "dead": {},
              "census": {}}


def _find_memo(cands, anchors) -> dict:
    """The per-surface retrieval memo, emptied when the surface lists move."""
    key = (id(cands) if cands else 0, id(anchors) if anchors else 0)
    lists = _FIND_MEMO["lists"]
    if lists is None or lists[0] != key:
        _FIND_MEMO.update(lists=(key, cands, anchors), idf={}, scored={},
                          dead={}, census={})
    return _FIND_MEMO


def _mask_geom_center(m):
    """``(geom, centre)`` of one mask list, built once."""
    hit = _MASK_GEOM.get(id(m))
    if hit is not None and hit[0] is m:
        return hit[1], hit[2]
    if len(_MASK_GEOM) >= _MASK_GEOM_MAX:
        _MASK_GEOM.clear()
    g, c = G.mask_geom(m), _mask_center(m)
    _MASK_GEOM[id(m)] = (m, g, c)
    return g, c


def _rows_by_content(kind, nodes, build_row, geom):
    """Build one row per ``(fid, node)``, reusing unchanged ones.

    Returns fresh dicts: the caller stamps ``covers_window`` / ``end0_xy`` on
    rows, and the cached original must stay the pre-stamp row. The copy is
    shallow, so the heavy values (cells, mask loops) are shared rather than
    duplicated -- they are read-only everywhere in the tree.
    """
    out = []
    for f, n in nodes:
        key = (kind, str(f), geom, fingerprint_node(n))
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
    """THE single (cands, anchors) surface builder — called by BOTH the offline
    enrich (det/prior nodes from a bundle's tile graph) and the live analyze_scene
    tool (det/prior nodes from the runtime graph). One function ⇒ no teacher/student
    drift in xy source, centerline policy, class/color, rounding, or order.

    ``det_nodes`` / ``prior_nodes`` are iterables of ``(fid, node)``; a node exposes
    ``centroid`` (grid row,col), ``centerline`` (grid polyline or None), ``class_name``,
    ``name``. xy PREFERS the node's continuous ``world_xy`` (the tracker's
    sub-cell position, 2dp), then a point on the node's own geometry, then the
    cell centre of its centroid. Detections carry NO centerline; priors carry
    their baked centerline (cell-center world, 2dp). Both lists are sorted by
    fid.

    UNIFORM FOOTPRINT: every node with a realised mask — a detected car AND a
    prebake building — gets the SAME mask-derived fields via ``_mask_fields``:
    the hidden EXACT ``mask`` loops + ``mask_centroid`` (the compare op's gap /
    center substrate — the ingested mask verbatim, every fragment and hole,
    no decimation) plus two VISIBLE opt-in shape scalars
    (``footprint_area_m2``/``aspect_ratio``). Nodes without a mask stay bare.
    """
    tx, ty = float(origin_xy[0]), float(origin_xy[1])
    cs = float(cell_m)

    def _w(rc):
        return [round(tx + (int(rc[1]) + 0.5) * cs, 2), round(ty + (int(rc[0]) + 0.5) * cs, 2)]

    def _xy(n, mask=None):
        """The row's served point -- the ``act.move`` FLIGHT TARGET.

        A measured position wins: a detection's ``world_xy`` is where the
        tracker put the object, sub-cell. Otherwise the point is
        representative of the row's own geometry
        (:func:`representative_point`), never a mean over it: a mean is off
        a loop road, off a bent way, and off every piece of a name that is
        several places, and a flight target has to be where the feature is.
        """
        w = getattr(n, "world_xy", None)
        if w is not None and len(w) >= 2:
            return [round(float(w[0]), 2), round(float(w[1]), 2)]
        if mask:
            return representative_point(mask)
        c = getattr(n, "centroid", None)
        return _w(c) if c is not None and len(c) >= 2 else None

    def _cl(n):
        cl = getattr(n, "centerline", None)
        return [_w(p) for p in cl] if cl else None

    def _cells(n):
        """The node's own 2 m mesh, as a plain (row, col) int list.

        The primary geometry: every spatial op measures on this; the mask
        loops and the centerline are reductions of it. A list rather than an
        array so a surface round-trips through JSON unchanged. ``tolist``
        rather than a comprehension: this is the hottest line of the rebuild.
        """
        c = cell_mesh.cells_of(n)
        return c.tolist() if len(c) else []

    def _axis_q(n, key):
        # axis_length_m is meaningful only for a node that HAS a baked
        # centerline (a linear feature); None when there is no centerline.
        if not getattr(n, "centerline", None):
            return None
        v = getattr(n, key, None)
        return round(float(v), 2) if v is not None else None

    def _exact_chains(n) -> list:
        """The node's EXACT vertex chains, CUT to the ground this tile
        actually covers.

        A map node carries one ``world_contour``: a ring, a way or a single
        point, stating its own closure. A row is ONE way, so that contour is
        the whole of what the row describes, and the set a name resolves to
        carries one row -- and one contour -- per way.

        The clip: the bake rasterises the full geometry and clips the CELLS
        to its grid, so a feature's cells stop at the tile edge while its
        contour runs on. The box the node's own cells occupy is therefore the
        clip: a no-op for a feature wholly inside, the tile edge for one that
        leaves.

        One cell of margin, because a cell contains the chain that rasterised
        it and its centre is half a cell inside its own edge. The margin stops
        at the tile origin: cell index -1 is off the map.

        A node with no cells is not clipped: there is nothing to say which
        ground it covers, and guessing would delete geometry on no evidence.
        """
        wc = getattr(n, "world_contour", None)
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
        # The mask-derived fields, identical for a detection and a prior:
        #  * ``mask`` + ``mask_centroid``: the exact geometry as world-meter
        #    vertex chains (the node's contour, else its full multi-loop cell
        #    boundary) and its area centroid -- the compare op's gap and
        #    centre substrate, hidden from the model (_MODEL_HIDDEN_ROW_FIELDS).
        #  * ``footprint_area_m2`` / ``aspect_ratio``: visible opt-in shape
        #    scalars. An open chain encloses nothing and gets none.
        # A node without a mask gets no mask fields.
        cells = getattr(n, "cells", None)
        chains = _exact_chains(n)
        # A one-vertex chain (an OSM POI) is not a drawable shape: it falls
        # through to the cell boundary below, which draws. Two-vertex chains
        # (single-segment ways) stay on their exact geometry, per cell_mesh's
        # policy that distance and intersection run on the exact chains.
        if any(len(ch) >= 2 for ch in chains):
            out = {"mask": chains}
            _c = _mask_center(chains)
            if _c is not None:
                out["mask_centroid"] = [round(v, 2) for v in _c]
            shape = G.contour_shape_features(chains)
            if shape:
                out.update(shape)
            return out
        if cells is None or len(cells) == 0:
            return {}
        out = {}
        loops = G.cells_boundary_loops(cells, (tx, ty), cs)
        if loops:
            out["mask"] = loops
            _c = _mask_center(loops)
            if _c is not None:
                out["mask_centroid"] = [round(v, 2) for v in _c]
        # Shape scalars only when the raster can express shape: below ~6
        # cells the 2 m grid quantizes area to {4..16} m2 and aspect to {1,2}.
        if len(cells) >= 6:
            shape = G.mask_shape_features(cells, cs)
            if shape:
                out.update(shape)
        return out

    # ``_mask_fields`` runs once per node and its ``mask`` is handed to
    # ``_xy``, so the served point is representative of the geometry served.
    def _cand_row(f, n):
        mf = _mask_fields(n)
        xy = _xy(n, mf.get("mask"))
        if not xy:
            return None
        return {"fid": str(f), "xy": xy, "centerline": None,
                "cells": _cells(n),
                "class": str(getattr(n, "class_name", "") or ""),
                **_color_fields(n),
                **mf,
                "search_text": _cand_search_text(n)}

    cands = _rows_by_content("c", det_nodes, _cand_row, (tx, ty, cs))
    # PRIOR rows carry no class field: a prior's semantics are its name + raw
    # OSM tags. One row per way: a named road is routinely several ways and
    # the set is what a name resolves to, so a row's cells, centerline and
    # position describe one way and its fid resolves to that way in the
    # runtime graph.
    def _anchor_row(f, n):
        mf = _mask_fields(n)
        xy = _xy(n, mf.get("mask"))
        if not xy:
            return None
        return {"fid": str(f), "name": getattr(n, "name", None),
                "xy": xy,
                "centerline": _cl(n),
                "cells": _cells(n),
                "axis_length_m": _axis_q(n, "axis_length_m"),
                "tags": _row_tags(n),
                **_color_fields(n),
                **mf,
                "search_text": _anchor_search_text(n)}

    anchors = _rows_by_content("a", prior_nodes, _anchor_row, (tx, ty, cs))
    # A centerline row renders its termini: end0_xy is the along=0 end of
    # every chart over it, end1_xy the far end.
    for _ar in anchors:
        _acl = _ar.get("centerline")
        if isinstance(_acl, list) and len(_acl) >= 2:
            _ar["end0_xy"] = [round(float(_acl[0][0]), 2),
                              round(float(_acl[0][1]), 2)]
            _ar["end1_xy"] = [round(float(_acl[-1][0]), 2),
                              round(float(_acl[-1][1]), 2)]
    # Fold pure duplicate cand rows (identical rounded xy and class), keeping
    # the best-attributed one: color present, then higher color_conf, then
    # the lowest fid numerically. Distinct classes at one xy are not folded.
    def _fid_key(f):
        # detected_2 before detected_100: the numeric tail sorts as a
        # NUMBER; a fid with no numeric tail sorts after, by string.
        s = str(f)
        head, _, tail = s.rpartition("_")
        return ((head, 0, int(tail), "") if tail.isdigit()
                else (s, 1, 0, s))

    _best: dict = {}
    for c in cands:
        k = (round(float(c["xy"][0]), 1), round(float(c["xy"][1]), 1),
             c.get("class") or "")
        prev = _best.get(k)
        if prev is None:
            _best[k] = c
            continue
        def _rank(row):
            return (1 if row.get("color") else 0,
                    float(row.get("color_conf") or 0.0))
        if _rank(c) > _rank(prev) or (
                _rank(c) == _rank(prev)
                and _fid_key(c["fid"]) < _fid_key(prev["fid"])):
            _best[k] = c
    if len(_best) != len(cands):
        cands = list(_best.values())
    # Natural fid order (detected_2 before detected_100): numeric order is
    # mint order, which for an explore pass is stop order.
    cands.sort(key=lambda c: _fid_key(c["fid"]))
    anchors.sort(key=lambda a: _fid_key(a["fid"]))
    # Window-covering mask (geometric, tag-blind): a mask containing every
    # surface row position makes the gap to it 0 for every row. The visible
    # ``covers_window`` fact stamped here makes compare serve one containment
    # fact instead; a bbox prefilter keeps the exact test to the few
    # window-spanning masks.
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
    """Split a flat ``(fid, node)`` iterable into detections (cands) and
    priors (anchors) by ``node.provenance`` and run :func:`build_surface`:
    the one graph -> surface path."""
    det, prior, seen = [], [], set()
    for fid, node in nodes_iter:
        fid = str(fid)
        if fid in seen:
            continue
        seen.add(fid)
        if getattr(node, "provenance", "prebaked") == "detected":
            det.append((fid, node))
            continue
        prior.append((fid, node))
    return build_surface(det, prior, origin_xy, cell_m)


# Row fields resolved server-side by fid that no model decision needs
# verbatim: ``search_text`` (the fuzzy-match haystack), ``centerline`` (a
# full polyline, hundreds of tokens on a road), ``mask``/``mask_centroid``
# (the exact footprint compare resolves) and ``cells`` (the bake's 2 m mesh;
# distance and intersection run on the exact ``mask`` chains).
_MODEL_HIDDEN_ROW_FIELDS = ("search_text", "centerline", "mask",
                            "mask_centroid", "cells")

# Render-input fields: folded into the compact color cell
# (:func:`_mask_noise_color`), never served as raw columns or named in a
# present-columns echo.
_RENDER_INPUT_ROW_FIELDS = ("color_shade", "color_flag")


# Below this color_conf a color word is noise, and the model literal-matches
# color words, so the model-facing render marks it uncertain. Render only:
# the raw rows keep the real color, so the color columns are unchanged.
_COLOR_CONF_FLOOR = 0.15


def _mask_noise_color(row):
    """Render the color cell as a token-minimal ``col`` tag:
    ``[dk|lt ]word[/word2][?]``. A trailing ``?`` marks an uncertain read
    (below-floor color_conf, or ``color_flag='look'``); a ``color_shade`` of
    dk/lt prefixes. Render only: the raw row keeps the mixture string.
    Returns the row unchanged when it carries no color."""
    if not isinstance(row, dict):
        return row
    c = row.get("color")
    if not (isinstance(c, str) and c.strip()):
        # shade/flag are render inputs to the `col` tag; with no color word
        # they are still stripped (never served raw).
        if "color_shade" in row or "color_flag" in row:
            row = dict(row)
            row.pop("color_shade", None)
            row.pop("color_flag", None)
        return row
    conf = row.get("color_conf")
    uncertain = ((isinstance(conf, (int, float)) and conf < _COLOR_CONF_FLOOR)
                 or row.get("color_flag") == "look")
    shade = row.get("color_shade")
    # /word2 only when color_shade is set (a color-name posterior); the
    # palette-mode mixture's second component is glass/shadow noise.
    rich = shade is not None
    out = {**row, "color": G.compact_color_cell(
        c, shade=shade if shade in ("dk", "lt") else None, uncertain=uncertain, rich=rich)}
    # shade/flag are render INPUTS folded into the `col` tag above — never shown raw.
    out.pop("color_shade", None)
    out.pop("color_flag", None)
    return out


def _model_facing_row(row):
    """Project a surface row for the model: strip server-resolvable fields and
    mask a below-floor color word to '?' (noise looks like noise).

    The commit binds on ``fid`` and every op resolves geometry by fid, so the
    hidden fields are pure context weight. Idempotent where absent."""
    if isinstance(row, dict) and any(k in row for k in _MODEL_HIDDEN_ROW_FIELDS):
        row = {k: val for k, val in row.items() if k not in _MODEL_HIDDEN_ROW_FIELDS}
    return _mask_noise_color(row)


class _BroadRows(list):
    """Marker for a retrieval (``find``) result list.

    Behaves as a plain list everywhere server-side (rN chaining, geometry
    ops); :func:`compact_results` applies the ONE broad-result envelope law
    to it like every other row list. ``ordered`` is True on a query-matched
    find (best-first is a real ordering, so its rows render); a query-less
    browse stays unordered and renders census + len only."""

    empty_condition: "str | None" = None
    # A tied one-name set renders its shape (len + name), never rows or fids.
    name_set: bool = False
    # POSITIVE ordering marker: True when this list's order is real
    # (query-matched best-first). compact_results renders rows only when an
    # ordering exists.
    ordered: bool = False
    # Columns this result's own op attached (render-budget priority).
    op_cols: list = []


# THE broad-result envelope law: any row-list result longer than
# _BROAD_DUMP_SOFT_ROWS renders as census + the first _BROAD_HEAD_ROWS rows
# in the result's order, with the full set live in the rN handle. Response
# shape only, never a candidate cap: the commit reads the RAW results. The
# head equals LOOK_SET_ROWS, so a set crop badges exactly the rows the head
# shows.
def join_hints(*parts) -> str:
    """Hints joined as SENTENCES, one per line -- the ONE hint shape.

    Several ops contribute facts to one envelope; a line each keeps them
    separable. Empty parts drop out.
    """
    return "\n".join(str(p) for p in parts if p)


# Above this many tied ways under ONE name, a find's initial reply is the
# shape (how many ways, how long), not a table of arbitrarily ordered rows.
_NAME_SET_SHAPE_ROWS = 8

_BROAD_DUMP_SOFT_ROWS = 48
_BROAD_HEAD_ROWS = LOOK_SET_ROWS



def _census_counts(rows) -> tuple[dict, int]:
    """FULL tag ``key=value`` counts of a row set (priors carry raw OSM tags;
    detections counted as ``class=<c>``) + how many rows carry a name."""
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
                if str(k) == "osm_id":
                    continue  # per-row unique id, not vocabulary
                key = f"{k}={v}"
                counts[key] = counts.get(key, 0) + 1
    return counts, named


def _surface_census(rows, top_n: int = 12) -> tuple[dict, int]:
    """FACTS census of a row set: the top-``top_n`` tag ``key=value`` counts.
    The world-graph discovery affordance — what this world speaks; counts
    only, no vocabulary advice."""
    counts, named = _census_counts(rows)
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    return dict(top), named


def _query_matched_census(counts: dict, query: str, top_n: int = 8) -> dict:
    """QUERY-CONDITIONED slice of a full census: the tag ``key=value`` entries
    whose key or value shares a token (or an edit-distance-close token) with
    the query, by count. Pure LEXICAL retrieval over the tag vocabulary — it
    surfaces the long-tail tags a global top-N cut hides exactly for
    divergent-vocabulary queries (counts only, no advice)."""
    q_tokens = {t for cl in T.fuzzy_prepare_query(query) for t in cl[1]}
    q_long = [t for t in q_tokens if len(t) >= 4]
    matched: dict[str, int] = {}
    for key, n in counts.items():
        toks = set(T._fuzzy_normalize(key).split())
        toks |= {T._singularize(t) for t in toks}
        if toks & q_tokens:
            matched[key] = n
            continue
        hit = False
        for qt in q_long:
            for ct in toks:
                if len(ct) < 4:
                    continue
                if 1.0 - T._edit_distance(qt, ct) / max(len(qt),
                                                        len(ct)) >= 0.7:
                    hit = True
                    break
            if hit:
                break
        if hit:
            matched[key] = n
    top = sorted(matched.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    return dict(top)


def _census_note(rows, *, name_samples: int = 4, query: str | None = None) -> str:
    """One-line surface census note: on a query miss, FIRST the tags sharing
    tokens/fuzz with the query (the long-tail entries a global top-N hides),
    then the global top counts + name samples. Rides empty/no-match
    retrievals so the re-query is one read, not a guess."""
    counts, _named = _census_counts(rows)
    top = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12])
    parts = ", ".join(f"{k}({v})" for k, v in top.items())
    names: list[str] = []
    for it in rows:
        if isinstance(it, dict) and (it.get("name") or "").strip():
            nm = str(it["name"]).strip()
            if nm not in names:
                names.append(nm)
        if len(names) >= name_samples:
            break
    note = "surface census -- "
    if query:
        qm = _query_matched_census(counts, query)
        qparts = ", ".join(f"{k}({v})" for k, v in qm.items())
        note += (f"tags sharing tokens with your query: {qparts or 'none'}; "
                 f"top tags overall: {parts or 'none'}")
    else:
        note += f"top tags: {parts or 'none'}"
    if names:
        note += "; named e.g. " + ", ".join(repr(n) for n in names)
    return note + "."


class _EmptyFindMiss(list):
    """Marker: a ``find`` whose retrieval matched NOTHING on a surface that
    includes the prior map rows (a query miss, or an explicit
    ``provenance:'prior'`` search). Always empty and chaining-safe;
    :func:`compact_results` states WHICH rows were searched (``searched``)
    on top of the empty-result hint (the truthful ``empty_condition`` when
    a query miss set one), so a dead ground attempt is distinguishable from
    an empty runtime-detection query."""

    empty_condition: "str | None" = None
    # A tied one-name set renders its shape (len + name), never rows or fids.
    name_set: bool = False
    # The truthful scope of the miss, set where provenance is known.
    searched: str = "the surface rows"


class _FilterCutRows(list):
    """Marker for every ``filter_items`` result (a cut over its input).

    Behaves as a plain list server-side (rN chaining, geometry ops);
    :func:`compact_results` additionally renders ``cut_stats`` on a NUMERIC
    cut — the one-row deterministic cut line: ``{"cut": "27->4",
    "nearest_excluded": {"fid": ..., <key>: <value>}, "admit_at": <value>}``.
    Information-only: the harness never relaxes or re-runs the cut itself; the
    MODEL reads the nearest-excluded row + the admit-at threshold and decides
    whether to widen. On an EMPTY result the same stats ride the payload
    (facts only, radical arm). ``ordered`` propagates from the filter's
    INPUT: a filter preserves order, so a cut over an ordered set stays
    renderable."""

    cut_stats: dict = {}
    ordered: bool = False
    op_cols: list = []


class _CompareRows(list):
    """Marker for a ``compare`` op result: a plain list of evidence rows
    server-side (chainable — ``score_by{items:"rN", emphasis:[{key:"d_...", w:-1}]}`` works
    directly), rendered by :func:`compact_results` as the SAME union-header
    markdown grid every other non-empty list result gets — identity columns
    + every column this compare attached, ONE joined table (audit: served as
    two shapes, the blocks were never joined; the model re-derived from xy
    and injected arithmetic errors) — prefixed by the per-ref ``refs``
    header (labels, ref_fid, per-ref ``cols`` echo, ``name_match`` grounding
    trust) and the verbatim ``columns`` echo (the one broad-result
    census+head law applies past ``_BROAD_DUMP_SOFT_ROWS``).
    ``notes`` carries the loud empty-set / no-flanker notes; ``columns`` is
    the VERBATIM ordered list of every column key this compare attached
    (echoed as the result header's ``columns`` so the model reads real names
    instead of guessing d_<...>); ``ordered`` propagates from the compare's
    INPUT (compare preserves input order)."""

    refs: list = []
    notes: list = []
    columns: list = []
    # Propagates from the compare's INPUT (compare preserves input order).
    ordered: bool = False


def _cut_stands_at(v, body: list, capped: list) -> str | None:
    """Where the ORDERING COLUMN stood at the head's cut, both sides of it.

    A head of 24 rows out of thousands says nothing about rank 25. If the
    sort separated, the next row sits far from the last one shown; if not,
    the head is an arbitrary slice of a plateau. The value just past the cut
    is the one thing the served head cannot show. A fact, never a verdict.

    It reports the composite ``score``, not a column: a multi-key sort is
    ordered by the sum, and no single column is monotone in the head. Only a
    ``score_by`` result (``_CueSortedRows``) is an ordering; a find's or a
    filter's ``op_cols`` is a column-budget list, not a sort.

    The next row comes from ``body`` (window-covering rows already dropped),
    the sequence the head was cut from.
    """
    if not isinstance(v, _CueSortedRows) or len(capped) >= len(body):
        return None

    def _num(row):
        val = row.get("score") if isinstance(row, dict) else None
        return float(val) if isinstance(val, (int, float)) else None

    head = [x for x in (_num(r) for r in capped) if x is not None]
    if len(head) < 2:
        return None
    nxt = next((x for x in (_num(r) for r in body[len(capped):])
                if x is not None), None)
    keys = [k for k in (getattr(v, "op_cols", None) or []) if isinstance(k, str)]
    what = " + ".join(f"'{k}'" for k in keys) or "this ordering"
    span = f"{min(head):.6g}..{max(head):.6g}"
    if nxt is None:
        return f"ranked on {what}: this head's score runs {span}"
    return (f"ranked on {what}: this head's score runs {span}, and the next "
            f"row scores {nxt:.6g}")


class _CueSortedRows(list):
    """Marker: this list's ORDER was produced by the ordering op (score_by)
    over cue-derived keys. Behaves as a plain list everywhere else."""

    cue_sorted = True
    op_cols: list = []


# Model-facing column order for the working-set grid: identity first, then the
# opt-in shape scalars, then whatever compare/other ops attached (first-seen).
# ``idx`` + ``fid`` are the two leading columns the renderer emits itself and
# are NOT listed here.
_GRID_COL_PRIORITY = (
    "class", "color", "color_conf", "name", "tags", "xy",
    "footprint_area_m2", "aspect_ratio", "axis_length_m",
)

# Repeat the header row every N data rows on a long dump.
_GRID_HEADER_EVERY = 50

# Render bound on a tag-fat row's tags cell: the first N tags render, the
# rest fold to a '+K more tags' marker. The node keeps every tag and find
# still searches all of them.
_TAG_RENDER_HEAD = 12

# Tag keys the renderer drops: ids, postal bookkeeping and import constants
# the model does not read. A render cut only: the node keeps every tag and
# ``search_text`` is built from the full set, so a find on them still
# matches.
_TAG_RENDER_DROP = frozenset({
    "osm_id", "source_tag", "addr_postcode", "addr_city", "oneway",
    "ref_GB_uprn",
    # Import bookkeeping and constants (tiger_county and worldkit_height_m
    # barely vary). tiger_name_base / tiger_name_type stay: street names are
    # not readable from nadir pixels.
    "tiger_reviewed", "tiger_cfcc", "tiger_source", "tiger_tlid",
    "tiger_separated", "tiger_county", "worldkit_layer",
    "worldkit_height_m", "calcity_object_id", "source_lanes",
})

# Columns the grid does not render (the raw fields stay on the rows):
# name_match and score (never read to pick a row; score duplicates in-row
# values on color sorts), aspect_ratio (a min-rect scalar is a lossy proxy
# for shape; the crop answers shape), axis_length_m (inconsistent with the
# endpoints) and color_conf.
_GRID_COL_DROP = frozenset({
    "aspect_ratio", "axis_length_m", "color_conf",
    "name_match", "score",
})

# The one list of column names the model never sees, on any channel: the
# grid, the Row repr on the python surface, and the census echoes in error
# and dedup notes.
MODEL_UNSERVED_COLUMNS = (
    _GRID_COL_DROP
    | frozenset(_MODEL_HIDDEN_ROW_FIELDS)
    | frozenset(_RENDER_INPUT_ROW_FIELDS))


def _is_gridable(v) -> bool:
    """A value the grid renders as ONE cell: a scalar (str/int/float/bool), an
    xy-like 2-number list, or a flat tags dict (rendered compact ``k=v``).
    Hidden list fields (mask/centerline) are stripped
    by :func:`_model_facing_row` before a row reaches the grid, so ``xy`` is the
    only list that survives here."""
    if isinstance(v, (bool, int, float, str)):
        return True
    if isinstance(v, dict):
        return all(isinstance(x, (bool, int, float, str)) for x in v.values())
    return (isinstance(v, (list, tuple)) and len(v) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in v))


def _fmt_float(v: float) -> str:
    """THE one float render rule: serve the STORED value verbatim (trailing
    zeros trimmed, ``-0.0`` normalized to ``0.0``, always at least one
    decimal). Every compute site quantizes ONCE when it stamps the value
    (d_/dc_ at 2 dp, bearings at 1 dp, name_match/color at 3 dp, score at
    ADAPTIVE 4..10 dp — geom_compute._SCORE_MAX_DP), so the SERVED number IS
    the number every law/invariant keyed on — a 0.04 m gap can never render
    as ``0.0`` next to a served ``brg_gap_`` (the absence law keys on the
    same stored value), and distinct stored composite scores always render
    distinct (the auditability fact; the render widens past 4 dp exactly
    when 4 dp would lose the stored value, capped at 10 dp)."""
    if v == 0.0:
        v = 0.0  # normalize -0.0 (a signed zero is a render artifact)
    s = f"{v:.4f}"
    if float(s) != v:
        # ADAPTIVE widening: the fewest decimals that round-trip the stored
        # value (a >4 dp adaptive score stamp), bounded at the stamp cap.
        for dp in range(5, 11):
            s = f"{v:.{dp}f}"
            if float(s) == v:
                break
    s = s.rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"


def _grid_cell(v) -> str:
    """One markdown cell. Absent/empty -> ``-``; floats -> the stored value
    verbatim (:func:`_fmt_float` — the ONE rounding rule); xy -> ``x, y``;
    a tags dict -> compact ``k=v; k2=v2``; a stray ``|`` in a string becomes
    ``/`` so a value never column-breaks the pipe table."""
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
    """The ordered union of non-fid gridable columns present (non-empty) across
    model-facing rows: priority columns first, then first-seen extras."""
    seen: list[str] = []
    seen_set: set = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        row = _model_facing_row(row)
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
    # Side columns are presence pairs (one member per row), so a first-seen
    # union interleaves them with foreign columns. Group each ref's four side
    # columns at the first position any appears, in the fixed order
    # left/right/ahead/behind.
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


def _render_grid_rows(rows, cols=None) -> str:
    """Union-header markdown grid — the ONE working-set surface serialization
    (the main ``{len, items}`` list results AND ``compare`` evidence).

    ``| fid | <cols...> |`` -- every column at least one row carries. Absent
    cells ``-``, the header repeated every _GRID_HEADER_EVERY data rows. Rows
    pass through :func:`_model_facing_row`. There is no index column:
    position is what ``rN[i]`` / ``rN[a:b]`` addresses."""
    proj = [_model_facing_row(r) if isinstance(r, dict) else r for r in rows]
    if cols is None:
        cols = _grid_columns(proj)
    if cols:
        header = "| fid | " + " | ".join(cols) + " |"
        sep = "|" + " --- |" * (len(cols) + 1)
    else:
        header, sep = "| fid |", "| --- |"
    lines = [header, sep]
    idx = 0
    for row in proj:
        if not isinstance(row, dict):
            continue
        if idx and idx % _GRID_HEADER_EVERY == 0:
            lines.extend((header, sep))
        cells = [_grid_cell(row.get(k)) for k in cols]
        body = " | ".join(cells)
        lines.append(f"| {row.get('fid') or '?'} | {body} |"
                     if cols else f"| {row.get('fid') or '?'} |")
        idx += 1
    return "\n".join(lines)


# The relation-column prefixes compare attaches (suffixed with the ref
# label).
_COMPARE_COL_PREFIXES = ("distance to ", "center distance to ",
                         "bearing from ", "facing ",
                         "contains ", "along ", "left of ", "right of ",
                         "ahead of ", "behind ",
                         "east of ", "west of ", "north of ", "south of ")

# The map-fixed cardinal columns. One per DIRECTION asked for, never all
# four: the direction rides the key exactly as the reference does, so a cue
# saying "east of Orchid Drive" costs one column and not a compass rose.
# Signed metres, positive when the row IS in that direction; ABSENT when the
# reference has no side to name there (it straddles the row's band, or none
# of it lies abreast). See maps/search/analyze_scene/cardinal_side.py.
_CARDINAL_COL_PREFIXES = ("east of ", "west of ", "north of ", "south of ")


# A union's own bound, a wall-clock net: a union renders one column set
# however many members it folds, but an rN member can fold a whole class,
# and members x items gap calls is the cost.
_UNION_MAX_MEMBERS = 400

def _sanitize_token(lbl, fallback="ref") -> str:
    """The one column-key sanitizer: a typed token (fid, ``rN`` base, name,
    ``as:`` label) -> a ``[^a-z0-9]+``-collapsed lowercase base, shared by
    the compare column labels and the compiler's keys."""
    base = re.sub(r"[^a-z0-9]+", "_", str(lbl).lower()).strip("_")
    return base or fallback




# THE one render budget: a result's rendered grid stays under this many
# chars. Over it, columns the program's own ops named come first, then
# identity columns, then first-seen extras; the tail elides with a count
# note. Render only: every column lives on the full set in the rN.
_RENDER_CHAR_BUDGET = 12_000

# THE one-line table legend, served once per response under the first
# envelope that renders rows. Every clause is code-derived (test_table_legend):
#   xy       build_surface._xy: a detection's measured world_xy, else a
#            point ON the row's own geometry (representative_point); the
#            same point act.move flies.
#   distance G.gap_witness min edge gap, 0.0 on contact; line refs:
#            _poly_chart_one unsigned lateral
#   along    arc m from the end0_xy end
#   bearing  G.bearing: 0 = north = +y, 90 = east = +x, clockwise
_TABLE_LEGEND = (
    "reading these tables: xy = [east_m, north_m] world meters, +y is "
    "north (xy is the row's recorded point -- not always its footprint "
    "centroid, and never promised to be a line's midpoint); "
    "'distance to <name>' = meters between nearest edges, 0.0 = "
    "touching, and against a LINE it is the unsigned offset off the "
    "line, the same either side; 'facing <name>' = a compass WORD for "
    "which way that row faces that place; '<dir> of <name>' = signed "
    "meters the row lies that way of it, blank where the geometry "
    "declines to say; 'along <name>' = meters along the "
    "line from its end0_xy end; bearings are compass degrees, 0 = "
    "north, 90 = east, clockwise."
)


def _budget_columns(rows_r, cols, op_cols) -> tuple[list, int]:
    """Trim ``cols`` (original order kept) so the rendered grid of ``rows_r``
    fits ``_RENDER_CHAR_BUDGET``: keep every priority column that fits, in
    priority order (program-named ``op_cols`` first, then
    ``_GRID_COL_PRIORITY`` identity, then first-seen extras; always at least
    one column) — one oversized column skips, it does not wall off the
    cheaper columns behind it. Returns ``(kept_cols, n_cut)``."""
    proj = [_model_facing_row(r) for r in rows_r if isinstance(r, dict)]
    base = 20 * (len(proj) + 2)  # fid cell + pipes per line
    cost = {}
    for c in cols:
        # header cell + separator overhead, deliberately GENEROUS so the
        # kept prefix always renders under the budget.
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


def compact_results(results: dict, *, widen_form: str = "",
                    answer_ref: str = "") -> dict:
    """Project each op result into its model-facing envelope. Applied
    IDENTICALLY by the live tool return and the offline CLI trace so the
    student reads the shape it was trained on. The live envelopes:

      - grid  : a non-empty ORDERED list (score_by output; a query-matched
                find; a filter over an ordered input) -> ``{len, fids, rows}``
                (rows is the union-header markdown grid STRING — its header
                row carries every column name; ``+cut_stats`` on a numeric
                filter).
                Past ``_BROAD_DUMP_SOFT_ROWS`` the ONE broad-result law
                applies: ``{len, census, named, fids, rows, hint}`` — census
                over the WHOLE set + the first ``_BROAD_HEAD_ROWS`` rows (and
                fids) in the result's order; the hint states exactly how many
                rows are held live in the rN and how to slice/chain it. The
                rendered columns fit ``_RENDER_CHAR_BUDGET`` (program-named
                columns first, then identity; the elided tail is counted in
                the hint).
      - census: a non-empty UNORDERED list (query-less browse; a filter over
                one) -> ``{len, census, named, hint}`` — NO row head: a
                fid-lexicographic head reads as a candidate list. Rows render
                once an ordering exists (chain score_by / a query find); the
                full set stays live in the rN.
      - empty : an empty list -> ``{len:0, items:[], hint}`` (``+cut_stats`` on
                a numeric cut). The hint is the op's TRUTHFUL condition (a
                query miss states its threshold; a filter states its
                predicate) — never a claim about the world. This is the ONLY
                envelope still carrying ``items``.
      - compare: a ``compare`` result -> the same grid envelope +
                ``refs``/``columns`` header: ``{refs, len, columns, fids,
                rows, hint}`` under the SAME broad-result law (census + head
                past the knee).

    Served ONCE per response (never twice in one reply):
      - a census identical to one already served: later envelopes carry
        ``census_ref`` naming the ref that holds it.

    Rows are projected through :func:`_model_facing_row` (retrieval-internal
    fields dropped). Response SHAPE only: the raw rows all stay live in
    program state (rN chaining), and the commit path reads the RAW results —
    the head cap is never a candidate cap. The first rows-carrying envelope
    of a response carries ``legend`` — the one-line column-semantics key.

    ``answer_ref`` names the op whose result IS the answer, and the response
    is then that ONE envelope: a sentence asked one question, so it gets one
    table. The other ops' hints ride the answer, each attributed to the ref
    that made it — the facts survive, the re-rendered working does not.
    Response shape as ever: every rN stays chainable and no row is dropped
    from any of them.

    The legend rides every response, once, under its first rows-carrying
    envelope.
    """
    _legend_out = [False]

    def _table(head, rows, op_cols):
        """The grid of ``rows`` under the render budget into
        ``head["rows"]``, and the legend on the response's first table (it
        reads a TABLE, and it is served once per response). Returns the
        elided-columns note, or None."""
        cols, n_cut = _budget_columns(rows, _grid_columns(rows), op_cols)
        head["rows"] = _render_grid_rows(rows, cols)
        if not _legend_out[0]:
            head["legend"] = _TABLE_LEGEND
            _legend_out[0] = True
        return (f"{n_cut} column(s) elided from this render to fit; "
                "the rN keeps them all") if n_cut else None

    def _c(v):
        if isinstance(v, _CompareRows):
            # compare renders the SAME union-header grid every list result
            # gets, prefixed by the refs header; the joined table is the
            # point. ``fids`` keeps the result fid-bearing for the
            # terminal walk (rank order preserved) without duplicating the
            # rows.
            head: dict = {"refs": list(getattr(v, "refs", []) or []), "len": len(v)}
            # The header lists every attached column key verbatim, so the
            # model reads real names instead of guessing them.
            _cols_echo = list(getattr(v, "columns", []) or [])
            if _cols_echo:
                head["columns"] = _cols_echo
            hints: list[str] = []
            if len(v) == 0:
                hints.append(
                    "compare received NO items -- the items ref resolved to "
                    "an empty list."
                )
            else:
                _rrows = (list(v)[:_BROAD_HEAD_ROWS]
                          if len(v) > _BROAD_DUMP_SOFT_ROWS else v)
                # No `fids` array beside a grid: its first column carries
                # them. A census-only head carries it below.
                _elided = _table(head, _rrows, getattr(v, "columns", None))
                if _elided:
                    hints.append(_elided)
            hints.extend(getattr(v, "notes", []) or [])
            if hints:
                head["hint"] = join_hints(*hints)
            return head
        if isinstance(v, list):
            if len(v) == 0:
                # An empty list is a valid result. State the op's truthful
                # condition when one was recorded (a query miss names its
                # threshold), never a claim about the world; a numeric cut's
                # ``cut_stats`` carries the threshold facts.
                _empty_hint = (getattr(v, "empty_condition", None)
                               or "this op returned NO items: nothing met "
                                  "its condition.")
                _empty_hint = join_hints(
                    _empty_hint, *(getattr(v, "notes", None) or []))
                if isinstance(v, _EmptyFindMiss):
                    # A retrieval that matched nothing over rows that include
                    # the prior map features is a distinct mechanical fact
                    # (the named ref never grounded) — state WHAT was
                    # searched, truthfully scoped by provenance.
                    _empty_hint = join_hints(
                        _empty_hint,
                        f"this retrieval searched {v.searched} and "
                        "matched nothing.")
                _cut = getattr(v, "cut_stats", None) \
                    if isinstance(v, _FilterCutRows) else None
                if _cut:
                    return {"len": 0, "items": [], "cut_stats": _cut,
                            "hint": _empty_hint}
                return {"len": 0, "items": [], "hint": _empty_hint}
            # The union-header grid is the one serialization for a non-empty
            # list, under the broad-result law (census + head past
            # _BROAD_DUMP_SOFT_ROWS; the full-set fid array is never served;
            # the commit path reads the RAW results).
            #
            # Window-covering rows are omitted from the rendered head of a
            # score_by result and named in one note: their mask contains the
            # whole window, so they hold distance 0 to every ref and would
            # tie rank 0 of every closest-first sort. The rN keeps them.
            _ordered = bool(getattr(v, "cue_sorted", False)
                            or getattr(v, "ordered", False))
            if not _ordered:
                # No real ordering: census + len, no row head (a
                # fid-lexicographic head reads as a candidate list).
                head = {"len": len(v)}
                census, named = _surface_census(v)
                head.update({"census": census, "named": named})
                _cut = getattr(v, "cut_stats", None) \
                    if isinstance(v, _FilterCutRows) else None
                if _cut:
                    head["cut_stats"] = _cut
                # The route to rows, stated as the fact in neither channel's
                # spelling (it renders for the sentence surface too).
                head["hint"] = join_hints(
                    *(getattr(v, "notes", None) or []),
                    "no ordering exists on this result, so it renders "
                    "census + len and no fids: ordering it by an attached "
                    "column renders the ordered rows of THIS handle, and a "
                    "name or color match returns matched rows directly.")
                return head
            _body = list(v)
            _cover_note = None
            if isinstance(v, _CueSortedRows):
                _covering = [it for it in v if isinstance(it, dict)
                             and it.get("covers_window")]
                if _covering:
                    _body = [it for it in v
                             if not (isinstance(it, dict)
                                     and it.get("covers_window"))]
                    _cfids = [str(it.get("fid") or "?") for it in _covering]
                    _cover_note = (
                        f"{len(_covering)} window-covering row(s) "
                        f"({', '.join(_cfids[:6])}"
                        f"{', ..' if len(_cfids) > 6 else ''}) hold "
                        f"'distance to' = 0 to "
                        f"every ON-SURFACE ref by containment (their mask "
                        f"contains this whole window; an off-window [x,y] "
                        f"literal ref keeps a real distance) -- omitted from "
                        f"this rendered head; they stay live in this rN "
                        f"(full set intact), covers_window marks them "
                        f"wherever they render.")
            # A name's set renders as its shape: every row ties, so a head
            # would be an arbitrary prefix served as a ranking. The rN keeps
            # every row.
            if getattr(v, "name_set", False):
                capped = []
            else:
                capped = list(_body[:_BROAD_HEAD_ROWS]
                              if len(v) > _BROAD_DUMP_SOFT_ROWS else _body)
            # A truncated head says so. Whether the order means anything is
            # stated by the producing op (find's best-first or tie note),
            # never re-derived here.
            _head_note = None
            if getattr(v, "name_set", False):
                pass
            elif len(capped) < len(_body):
                # Channel-specific: the caller passes the widening form its
                # channel can say (``widen_form``); none gets the fact alone.
                _head_note = (
                    f"showing the first {len(capped)} of {len(v)} row(s); "
                    f"the rest stay live in this rN -- later ops on it "
                    f"read the FULL set"
                    + (f", and {widen_form} serves the next "
                       f"{_BROAD_HEAD_ROWS}" if widen_form else ""))
                _head_note = join_hints(_head_note,
                                        _cut_stands_at(v, _body, capped))
            _cut = getattr(v, "cut_stats", None) \
                if isinstance(v, _FilterCutRows) else None
            head = {"len": len(v)}
            if getattr(v, "name_set", False):
                # The shape as data: the name and the ways' total length.
                head["name"] = str(_body[0].get("name") or "")
                _set_m = sum(float(r.get("axis_length_m") or 0.0)
                             for r in _body)
                if _set_m:
                    head["length_m"] = round(_set_m)
            if not getattr(v, "name_set", False):
                head["fids"] = [str(it["fid"]) for it in capped
                                if isinstance(it, dict) and it.get("fid")]
            if not getattr(v, "name_set", False):
                _head_note = join_hints(_head_note, _table(
                    head, capped, getattr(v, "op_cols", None)))
            head["hint"] = join_hints(head.get("hint"), _head_note,
                                      _cover_note)
            if not head["hint"]:
                head.pop("hint")
            if _cut:
                # The one-row deterministic cut line on a non-empty numeric
                # cut. Info only.
                head["cut_stats"] = _cut
            head["hint"] = join_hints(
                head.get("hint"), *(getattr(v, "notes", None) or []))
            if not head["hint"]:
                head.pop("hint")
            return head

    # Serve any identical census once per response. The answer ref goes
    # first, so it keeps its own census rather than a pointer.
    order = ([answer_ref] + [k for k in results if k != answer_ref]
             if answer_ref in results else list(results))
    out: dict = {}
    _census_seen: dict[str, str] = {}
    for k in order:
        env = _c(results[k])
        if isinstance(env, dict) and "census" in env:
            _sig = repr(sorted(env["census"].items()))
            _first = _census_seen.get(_sig)
            if _first is not None:
                env.pop("census", None)
                env.pop("named", None)
                env["census_ref"] = (
                    f"census identical to {_first}'s (served once above)")
            else:
                _census_seen[_sig] = str(k)
        out[k] = env
    if answer_ref not in out:
        return out
    # One sentence, one table: the answer ref's envelope alone. Each other
    # op's hint rides the answer, attributed to its ref; every rN stays
    # chainable and the commit path reads the RAW results.
    answer = out[answer_ref]
    if isinstance(answer, dict):
        answer["hint"] = join_hints(
            answer.get("hint"),
            *(f"{k}: {e['hint']}" for k, e in out.items()
              if k != answer_ref and isinstance(e, dict) and e.get("hint")))
        if not answer["hint"]:
            answer.pop("hint")
    return {answer_ref: answer}


def _grouped_keys(present) -> str:
    """Key list -> a readable summary: singleton keys verbatim, per-instance
    families collapsed ('distance to building_* (100 instances)'), so an
    error naming hundreds of per-fid columns does not bury the one key the
    model needs. Each key joins exactly one family, its longest matching
    prefix from :data:`_COMPARE_COL_PREFIXES` + ``color_``.
    """
    prefixes = sorted(_COMPARE_COL_PREFIXES + ("color_",), key=len, reverse=True)
    fams: dict[str, list] = {}
    for k in present:
        pre = next((p for p in prefixes if k.startswith(p)), None)
        if pre is None:
            fams.setdefault(k, []).append(k)
            continue
        # Family = prefix + the instance stem's leading token, so
        # 'distance to building_1'/'..._2' collapse but 'distance to road_5'
        # stays its own family.
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


def _present_numeric_keys_hint(items) -> str:
    """Live-state tail for a score_by/filter on an absent key:
    the numeric keys the items ACTUALLY carry, facts only (the remedy for
    the MISSING key is :func:`_key_remedy`, the ONE family->provenance
    source)."""
    present = sorted({k for it in items if isinstance(it, dict)
                      for k, v in it.items() if isinstance(v, (int, float))
                      and k not in _RENDER_INPUT_ROW_FIELDS})
    if present:
        return f"numeric keys present on these items: {_grouped_keys(present)}."
    return "these items carry NO numeric key."


def _present_keys_hint(items) -> str:
    """Live-state tail for a filter naming an ABSENT column: every scalar key
    the items ACTUALLY carry (numeric AND string/bool -- an eq cut is legal on
    a string column like class), facts only (the remedy for the MISSING key
    is :func:`_key_remedy`)."""
    present = sorted({k for it in items if isinstance(it, dict)
                      for k, v in it.items()
                      if isinstance(v, (int, float, str, bool)) and v != ""
                      and k not in _MODEL_HIDDEN_ROW_FIELDS
                      and k not in _RENDER_INPUT_ROW_FIELDS})
    if present:
        return f"keys present on these items: {present}."
    return "these items carry NO scalar key."


def _key_remedy(key) -> str:
    """The producing fact for an ABSENT relation column, rendered beside the
    present-keys census. Returns a sentence ending in a space, or ''."""
    k = str(key)
    if not re.fullmatch(r"[A-Za-z][\w. ]{0,60}", k):
        return ""  # not a column-shaped token: no worked example applies
    for pre in ("facing ", "distance to "):
        if not k.startswith(pre):
            continue
        param = k[len(pre):]
        if re.fullmatch(r"[0-9][0-9_.]*", param):
            # The model may never have written this label: a point named in
            # one call mints it, and a later ordering can name the column.
            return (f"{param!r} is not a fid: a number pair names a POINT, "
                    f"and a point carries columns only under the label it is "
                    f"given when it is related to rows -- the column is then "
                    f"{pre}<label>. ")
        # The say path reaches here through `in <place>`, so the remedy names
        # the fact (the column exists once the ref is related to these rows),
        # not an op.
        return (f"a '{pre}<ref>' column exists once <ref> has been related "
                "to these rows, and <ref> must be a fid on this surface. ")
    return ""


def _polyline_arc_midpoint(verts) -> list:
    """The arc-length midpoint of a world-meter polyline."""
    arcs = G._arc_lengths(verts)
    half = arcs[-1] / 2.0
    for i in range(len(verts) - 1):
        if arcs[i + 1] >= half:
            seg = arcs[i + 1] - arcs[i]
            t = (half - arcs[i]) / seg if seg > 0 else 0.0
            return [float(verts[i][0]) + t * (float(verts[i + 1][0]) - float(verts[i][0])),
                    float(verts[i][1]) + t * (float(verts[i + 1][1]) - float(verts[i][1]))]
    return [float(verts[-1][0]), float(verts[-1][1])]



# The staged-viewpoint side columns: the four phrase prefixes whose
# parameter may name a staged handle (act.stage's vN). Distance/bearing
# prefixes are NOT here -- a staged handle resolves as a plain point ref
# on those families (the compare path), so nothing is re-implemented.
_STANDING_SIDE_PREFIXES = ("left of ", "right of ", "ahead of ", "behind ")

# Every key naming one of these prefixes is a computation request: an op that
# names it gets the column computed (``_auto_attach``, in run_program). Module
# scope so the python surface reads the same tuple.
_AUTOATTACH_PREFIXES = ("facing ", "bearing from ",
                        "center distance to ", "distance to ",
                        "along ", "left of ", "right of ",
                        "ahead of ", "behind ",
                        "east of ", "west of ", "north of ", "south of ",
                        "color_")


def _attach_standing_columns(items, frame, with_distance=False):
    """Attach a staged viewpoint's presence-pair side columns to every
    xy-bearing row -- on COPIES, never the shared surface rows: 'left of
    <h>'/'right of <h>' (lateral meters) and 'ahead of <h>'/'behind <h>'
    (longitudinal meters), computed by standing at ``from_xy`` and looking
    along ``up_unit`` (right = cw(up), the one handedness law). Presence IS
    the side; 0.0 on-axis lands right/ahead.
    A row without an xy passes through untouched."""
    h = str(frame.get("handle") or "v")
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
        # a re-attach must not leave BOTH members of a presence-pair behind
        for k in (left_k, right_k, ahead_k, behind_k):
            rec.pop(k, None)
        # Fixed pair order: the grid's column order derives from first-seen
        # insertion.
        rec[right_k if lat >= 0 else left_k] = round(abs(lat), 2)
        rec[ahead_k if lon >= 0 else behind_k] = round(abs(lon), 2)
        if with_distance:
            # straight-line meters from the STAND POINT to the row's
            # recorded position (centroid) -- the stance is a viewer, not
            # a mask, so there is no footprint-gap variant to confuse.
            rec[dist_k] = round((dx * dx + dy * dy) ** 0.5, 2)
        out.append(rec)
    return out


def union_gap(geoms, bbs, item_geom):
    """``(gap, ref_witness, item_witness)`` from ``item_geom`` to the nearest
    of ``geoms`` -- the exact min over the members, the witness from the
    argmin member.

    ``bbs`` is ``G.stack_bboxes(geoms)``, built once per reference by the
    caller. A member whose bounding-box separation already exceeds the best
    gap in hand cannot beat it and is skipped. The scan is seeded from the
    nearest box, an upper bound straight away, but the record is still taken
    in member order with a strict test, so a tie keeps the earliest member's
    witness. The seed member's measurement is reused when the pass reaches
    it. A zero gap ends the scan: nothing beats contact.
    """
    bounds = G.bbox_bounds(bbs, item_geom)
    ub = None
    seed_i, seed = -1, None
    if len(geoms) > 1 and len(bounds):
        seed_i = int(bounds.argmin())
        seed = G.gap_witness(geoms[seed_i], item_geom)
        ub = seed[0]
    d = wr = wi = None
    for gi, g in enumerate(geoms):
        if d is not None and d <= 0.0:
            break
        cut = d if d is not None else ub
        if cut is not None and bounds[gi] > cut + G.BOUND_SLACK_M:
            continue
        _d, _wr, _wi = seed if gi == seed_i else G.gap_witness(g, item_geom)
        if d is None or _d < d:
            d, wr, wi = _d, _wr, _wi
    return d, wr, wi


class _Run:
    """One ``run_program`` call: the surface, the results so far, the seed,
    and the op dispatching (its signature rides that op's errors)."""

    def __init__(self, cands, anchors, seed=None, staged=None, standing=None):
        self.cands, self.anchors = cands, anchors
        self.by_fid = {c["fid"]: c for c in cands}
        self.by_fid.update({a["fid"]: a for a in anchors})
        self.seed = seed or {}
        self.staged = staged or {}
        self.standing = self.staged.get(standing) if standing else None
        self.results: dict = {}
        self.op = None

    def child(self) -> "_Run":
        """A fresh one-op run over the same surface: the compare an
        auto-attach runs reads no handle."""
        c = copy.copy(self)
        c.seed, c.results, c.op = {}, {}, "compare"
        return c

    def sig(self) -> str:
        sig = _OP_SIGNATURE.get(self.op)
        return f"; signature: {sig}" if sig else ""

    def avail(self, kind: str) -> str:
        """The run's live fid census (bounded) for an error, keyed by the
        arg kind that failed."""
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
        """``rN`` -> its rows (this program's result shadows an earlier
        call's); ``rN[a:b]`` -> that window of them."""
        base, _, win = tok.partition("[")
        if base in self.results:
            v = self.results[base]
        elif base in self.seed:
            v = self.seed[base]
        else:
            _known = sorted(set(self.results) | set(self.seed))
            raise ProgramError(
                f"unknown ref {tok!r}: an op defined with ref 'r0' is read back "
                f"as r0. Refs persist across your analyze calls (redefining one "
                f"replaces it). "
                f"valid refs so far: {_known or '(none)'}")
        return _window(v, win, tok) if win else v


class _Args(dict):
    """An op's args: a required arg that is absent raises a ProgramError
    naming the op and the arg."""

    def __init__(self, d, cx):
        super().__init__(d)
        self._cx = cx

    def __missing__(self, key):
        raise ProgramError(
            f"op {self._cx.op!r} is missing required arg {key!r}{self._cx.sig()}")


def _window(cur, win, token):
    """``rN[a:b]``: the rows, keeping the source's class and order attributes
    (whole-set descriptors -- notes, cut_stats, refs, columns -- stay
    behind). A window that selects 0 of a non-empty set fails loud."""
    part = "[" + win
    lo, hi = (int(b) for b in win[:-1].split(":"))
    if not isinstance(cur, list):
        cur = [cur]
    rows = cur[lo:hi]
    if type(cur) is not list:
        w = type(cur)(rows)
        for attr in ("ordered", "op_cols"):
            if attr in vars(cur):
                setattr(w, attr, getattr(cur, attr))
        rows = w
    if not rows and cur:
        raise ProgramError(
            f"window {part} in ref {token!r} selected 0 of "
            f"{len(cur)} rows: this ref holds {len(cur)} row(s) and "
            f"the window begins at or past the end of them "
            f"(or begins at or after where it stops).")
    return rows


def _as_list(cx, arg):
    """An ``items`` arg -> a row list: a handle's rows, or a list whose
    handles flatten to their rows and whose fids resolve to theirs."""
    v = cx.resolve(arg) if cx.is_ref(arg) else arg
    if isinstance(v, list):
        if not any(isinstance(x, str) for x in v):
            return v
        out: list = []
        for x in v:
            if cx.is_ref(x):
                sub = cx.resolve(x)
                out.extend(sub if isinstance(sub, list) else [sub])
            elif isinstance(x, str) and x in cx.by_fid:
                out.append(cx.by_fid[x])
            elif isinstance(x, str):
                raise ProgramError(
                    f"cannot resolve {x!r} in list {arg!r}: not a fid in this "
                    f"surface or an rN ref. {cx.avail('list')}{cx.sig()}")
            else:
                out.append(x)
        return out
    raise ProgramError(
        f"cannot resolve a LIST from {arg!r}: wants a list -- an rN ref to a "
        f"find/compare/filter_items result -- NOT a single fid. "
        f"{cx.avail('list')}{cx.sig()}")


def _as_points(cx, arg):
    """compare's ``items`` -> point-bearing rows: a handle's rows, fids
    through ``by_fid``, an ``[x, y]`` as an xy-only row."""
    v = cx.resolve(arg) if cx.is_ref(arg) else arg
    if is_xy(v):
        v = [v]
    if isinstance(v, str) and v in cx.by_fid:
        v = [cx.by_fid[v]]
    if not isinstance(v, list):
        raise ProgramError(
            f"cannot resolve a POINT-LIST from {arg!r}: wants a list of points -- an "
            f"rN ref to a find/compare result, or a list of "
            f"fids/[x,y] -- NOT a single fid. {cx.avail('list')}{cx.sig()}")
    out = []
    for it in v:
        if isinstance(it, dict) and is_xy(it.get("xy")):
            out.append(it)
        elif isinstance(it, str) and it in cx.by_fid:
            out.append(cx.by_fid[it])
        elif is_xy(it):
            out.append({"xy": list(it)})
        elif cx.is_ref(it):
            out.extend(_as_points(cx, cx.resolve(it)))
        else:
            raise ProgramError(
                f"cannot resolve a point in the list from item {it!r}: each item must "
                f"be a fid in this surface, a {{'fid':...}} dict, an [x,y], or an rN. "
                f"{cx.avail('point')}{cx.sig()}")
    return out


def _named_rows(cx, el, need):
    """One crossings/ends ref -> EVERY row it names (a road is routinely
    several ways): a fid, an rN, a list of them, or {ref:..., as:...}. A
    row off the surface counts when it carries ``need``."""
    if isinstance(el, dict):
        el = el.get("ref")
    out: list = []
    for t in (el if isinstance(el, list) else [el]):
        vals = [t]
        if isinstance(t, str) and t not in cx.by_fid:
            v = cx.resolve(t)
            vals = v if isinstance(v, list) else [v]
        for x in vals:
            n = None
            if isinstance(x, str):
                n = cx.by_fid.get(x)
            elif isinstance(x, dict):
                n = cx.by_fid.get(x.get("fid")) or (x if x.get(need) else None)
            if n is not None and n not in out:
                out.append(n)
    return out


# ---- auto-attach columns (the parameter rides in the key name) ----
# Naming a relation column '<prefix> <ref>' or a color column color_<word>
# where a column key is accepted is the computation request: when <ref> is a
# live surface fid (or a staged handle) a one-op compare attaches it and the
# op proceeds. A key that resolves to nothing falls to the op's own
# absent-key guard.

def _attach_compare(cx, items, args):
    """A one-op compare over ``items`` in a fresh run; its rows, else the
    items unchanged."""
    out = _compare(cx.child(), {"items": items, **args})
    return list(out) if out else items


def _auto_attach(cx, items, keys, notes=None):
    if not (isinstance(items, list) and items):
        return items
    refs: list = []
    charts: list = []
    colors: list = []
    # (direction, ref) pairs; the DIRECTION rides the prefix and the
    # reference the suffix, so one key names the whole column.
    cardinals: list = []
    for k in keys:
        if not isinstance(k, str) or k == "color_conf" or any(
                isinstance(it, dict) and it.get(k) is not None for it in items):
            continue  # not a column request, or already attached
        pre = next((p for p in _AUTOATTACH_PREFIXES if k.startswith(p)), None)
        if pre is None:
            continue
        param = k[len(pre):]
        if pre == "color_":
            canon, note = _resolve_color_word(param)
            if canon not in colors:
                colors.append(canon)
            if note is not None and notes is not None:
                notes.append(note)
        elif pre in _CARDINAL_COL_PREFIXES:
            # A map-fixed cardinal needs only the reference's geometry, so
            # ANY ref on the surface attaches one, including the merged
            # multi-way features whose centerline is None.
            if param in cx.by_fid or param in cx.staged:
                _cd = (pre.strip()[:-3].strip(), param)
                if _cd not in cardinals:
                    cardinals.append(_cd)
                if param not in refs:
                    refs.append(param)
        else:
            # a line-carrying fid charts its 'distance to'; any other ref on
            # the surface gets the relation columns
            _n = cx.by_fid.get(param)
            _cl = _n.get("centerline") if isinstance(_n, dict) else None
            if (pre == "distance to " and isinstance(_cl, list)
                    and len(_cl) >= 2):
                if param not in charts:
                    charts.append(param)
            elif ((param in cx.by_fid or param in cx.staged)
                  and param not in refs):
                refs.append(param)
    for canon in colors:
        items = _attach_color_column(items, canon)
    for param in charts:
        items = _attach_compare(cx, items, {"frame": {"along": param}})
    if not refs:
        return items
    _cargs: dict = {"refs": refs}
    if cardinals:
        _cargs["cardinals"] = [{"dir": d, "ref": r} for d, r in cardinals]
    return _attach_compare(cx, items, _cargs)


def _relate_ran(key, items) -> bool:
    """Did the compare that WRITES ``key`` actually run on these rows?

    A relation column has siblings: relating rows to a place stamps
    ``distance to <lab>`` beside ``facing <lab>``. So a ``facing <lab>``
    missing on every row while ``distance to <lab>`` is present is the
    geometry declining to name a direction (every row TOUCHES the place,
    which has no min-gap vector), not the model guessing a column name.
    Only ``in_or_blank`` needs the distinction, and it needs one that works
    for an ``as`` LABEL -- ``_contact_fact`` reads ``by_fid`` and answers ""
    for a labelled ref.
    """
    if not (isinstance(key, str) and key.startswith("facing ")):
        return False
    sib = "distance to " + key[len("facing "):]
    return any(isinstance(it, dict) and sib in it for it in items)


def _contact_fact(cx, key, items) -> str:
    """The degenerate-column facts for the absent-key guards, where the
    compare ran and the column is legitimately absent: (a) 'distance to'
    / 'facing' a <fid> whose mask contains the whole surface window; (b)
    'facing <fid>' where 'distance to <fid>' is 0 on every row (contact:
    no min-gap direction). Returns the fact, or ''."""
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
    if not (isinstance(key, str) and key.startswith("facing ")):
        return ""
    param = key[len("facing "):]
    if param not in cx.by_fid:
        return ""
    dkey = "distance to " + param
    vals = [it.get(dkey) for it in items if isinstance(it, dict)]
    if vals and all(isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and float(v) == 0.0 for v in vals):
        return (f"{key} is absent where '{dkey}'=0 (contact -- no "
                f"min-gap direction exists); 'bearing from {param}' "
                f"(centroid bearing) is defined on contact too. ")
    return ""


def _match_label(r):
    """What a query hit, as a name: the row's own name, else whole
    identifying tag pairs (never osm_id)."""
    nm = (r.get("name") or "").strip()
    if nm:
        return repr(nm[:32])
    tags = r.get("tags")
    if isinstance(tags, dict) and tags:
        out: list = []
        for k, v in tags.items():
            if str(k) == "osm_id":
                continue
            pair = f"{k}={v}"
            if sum(len(x) + 2 for x in out) + len(pair) > 40:
                break
            out.append(pair)
        if out:
            return ", ".join(out)
    return repr(str(r.get("class") or r.get("fid") or "?")[:32])


def _find(cx, a):
    """Search the surface. A query returns the match set (best-first, len =
    the match count); a query-less find browses the selected surface."""
    _q_notes: list = []
    _q_matched = False   # a query ran and matched (the rN is ordered)
    _name_set = False    # tied ways of ONE name: render the shape
    _find_cols: list = []  # columns this find attached (budget prio)
    _empty_cond = None  # the truthful no-match condition, set on a query miss
    prov = str(a.get("provenance") or "any").strip().lower()
    if prov == "prior":
        rows = list(cx.anchors)              # baked map priors only
    elif prov == "runtime":
        rows = list(cx.cands)                # in-flight detections only
    elif prov == "any":
        rows = cx.cands + cx.anchors         # the whole surface
    else:
        raise ProgramError(
            f"unknown find provenance {prov!r}; legal: any | prior "
            f"(baked map features) | runtime (in-flight detections)."
            f"{cx.sig()}")
    _fid_hit = (cx.by_fid.get(str(a.get("query") or "").strip())
                if a.get("query") else None)
    if _fid_hit is not None:
        # A query naming a live fid is retrieval of that row (the fuzzy
        # scorer matches text, never ids).
        rows = [_fid_hit]
        _q_matched = True
        _q_notes.append(
            "query " + repr(str(a.get("query")).strip())
            + " is a live fid -- that row returned directly.")
    elif a.get("query"):
        # Fuzzy free text over the walled search_text + name. The result is
        # the match set, best-first; the sub-threshold tail stays out (its
        # size is stated in the note). Every returned row carries name_match
        # (the grid renders none).
        _q_raw = str(a["query"])
        if ";" in _q_raw:
            # ';' is not a query operator: loud, naming the idioms.
            raise ProgramError(
                f"find query {_q_raw!r} contains ';', which is not a "
                f"query operator (the whole string would fuzz-score "
                f"as one text blob). ALTERNATIVES are the literal "
                f"UPPERCASE token OR ('a=b OR c=d', scored as the "
                f"max over them); to require BOTH, put the k=v "
                f"pairs in one query separated by a space ('a=b "
                f"c=d' -- the exact-tag tier scores by pairs "
                f"matched). OSM multi-value tag VALUES use ';' "
                f"(cuisine=pizza;kebab) -- to match one, query that "
                f"single value ('cuisine=pizza').{cx.sig()}")
        q_data = T.fuzzy_prepare_query(_q_raw)
        _searched_rows = list(rows)
        _memo = _find_memo(cx.cands, cx.anchors)
        _mkey = (prov, _q_raw)
        _hit = _memo["scored"].get(_mkey)

        def _memo_census():
            # the surface census a miss or a thin match serves: edit
            # distances over the whole vocabulary, once
            got = _memo["census"].get(_mkey)
            if got is None:
                got = _memo["census"][_mkey] = _census_note(
                    _searched_rows, query=_q_raw)
            return got
        # The fuzz band's denominator is a CORPUS fact, minted over the same
        # rows being scored (the sibling of `tag_family_keys` below): how
        # rare a query word is here, and whether any row could cover it.
        _idf = _memo["idf"].get(prov)
        if _idf is None and _hit is None:
            _idf = _memo["idf"][prov] = T.token_idf(_searched_rows)
        if _hit is None:
            scored = [{**x, "name_match": round(
                          T.find_score(q_data, x.get("search_text") or "",
                                       x.get("name"), x.get("tags"),
                                       idf=_idf), 3)}
                      for x in rows]
            # Widen to the query's own tag family: a query that is a value
            # of some tag key reaches every row carrying that key, at
            # T.TAG_FAMILY_SCORE, ranked below every exact hit. Never a drop:
            # no row that matched before stops matching.
            _fams = T.tag_family_keys(q_data, _searched_rows)
            if _fams:
                for _r in scored:
                    if (_r["name_match"] < T.TAG_FAMILY_SCORE
                            and (_r.get("tags") or {}).keys() & _fams):
                        _r["name_match"] = T.TAG_FAMILY_SCORE
            scored.sort(key=lambda x: x["name_match"], reverse=True)
            _memo["scored"][_mkey] = _hit = scored
        scored = [dict(_r) for _r in _hit]
        if scored and scored[0]["name_match"] < T.FUZZY_THRESHOLD:
            # Nothing matched: the no-match affordance, its truthful
            # condition (the scoring threshold) and the query-conditioned
            # census, so the re-query is one read.
            scored = []
            _q_tokens = sorted({t for _cl in q_data for t in _cl[1]})
            _empty_cond = (
                f"find matched nothing: no row scored >= "
                f"{T.FUZZY_THRESHOLD} against query {_q_raw!r} "
                f"(query tokens: {_q_tokens}).")
            _q_notes.append(_memo_census())
            # A query word that is a palette color matches no row text: the
            # miss names the column that carries it.
            for _cl in q_data:
                _w = str(_cl[0] or "").strip()
                if not _w or " " in _w:
                    continue
                _canon = G.normalize_color(_w)
                if _canon in G.COLOR_ENUM:
                    _q_notes.append(
                        f"{_w!r} is a color word on the closed "
                        f"palette: color is not row text -- it "
                        f"rides the color_{_canon} column, which "
                        f"attaches when the color word is named.")
                    break
        elif scored:
            _n_match = sum(1 for r in scored
                           if r["name_match"] >= T.FUZZY_THRESHOLD)
            # When the query hits a NAME, everything under the name tier
            # shares a word, not a referent (find("Neuralia Road") would also
            # return every other "... Road"), so the band under the name tier
            # is cut. A class word hits no name tier and keeps its whole
            # fuzzy band.
            _n_named = sum(
                1 for r in scored
                if r["name_match"] >= T._NAME_PHRASE_TIER_BASE)
            # A proper noun only: a detection whose CLASS is the query also
            # scores 1.0, and cutting there would delete the priors that are
            # the candidates. Only a name is a referent.
            _n_proper = sum(
                1 for r in scored
                if r["name_match"] >= T.NAME_TIER_FLOOR
                and str(r.get("name") or "").strip())
            _n_exact = sum(1 for r in scored
                           if r["name_match"] >= T.NAME_TIER_FLOOR) \
                if _n_proper else sum(
                    1 for r in scored
                    if r["name_match"] > T._TOKEN_BAG_CEIL)
            _band_cut = (_n_match - _n_exact) if _n_named else 0
            if _n_named:
                _n_match = _n_exact
            # The head names what is in this rN: off the match set, never off
            # ``scored`` (rows the name cut removed).
            _head_note = "; ".join(
                f"{r.get('fid')} {_match_label(r)} "
                f"(match {r['name_match']})"
                for r in scored[:_n_match][:4]
                if r["name_match"] >= T.FUZZY_THRESHOLD)
            # The match set is the result: state its score range and the size
            # of the sub-threshold tail kept out. "best-first" is claimed only
            # when the scores separate; a tie says so.
            _m_vals = [r["name_match"] for r in scored[:_n_match]]
            # A name's ways are a set, not a ranking: every row ties at the
            # name tier, so the initial reply is the shape and the rN holds
            # every row. The one name must BE a name: unnamed rows collapse
            # to a single "" and 60 anonymous ways read as one street.
            _ex_rows = [r for r in scored[:_n_match]
                        if r.get("name_match", 0.0) >= T.NAME_TIER_FLOOR]
            _ex_names = {str(r.get("name") or "") for r in _ex_rows}
            _name_set = (len(_ex_rows) > _NAME_SET_SHAPE_ROWS
                         and len(_ex_rows) == _n_match
                         and len(_ex_names) == 1
                         and bool(next(iter(_ex_names))))
            _rest = [r["name_match"] for r in scored[_n_match:]]
            _rest_frag = (f"; {len(_rest)} sub-threshold row(s) "
                          f"NOT in this rN" if _rest else "")
            if _m_vals[0] == _m_vals[-1]:
                _order_frag = (
                    f"all equally good matches at match "
                    f"{_m_vals[0]}, so this order is arbitrary and "
                    f"ordering them by a cue column chooses it")
            else:
                _order_frag = (f"best-first, match {_m_vals[0]}.."
                               f"{_m_vals[-1]}")
                # A mixed pool with a tied head: "best-first" is true of the
                # pool while the first k rows tie; said when k >= 2.
                _top_n = sum(1 for _x in _m_vals if _x == _m_vals[0])
                if _top_n > 1:
                    _order_frag += (
                        f"; the first {_top_n} tie at match "
                        f"{_m_vals[0]}, so their order among "
                        "themselves is arbitrary and ordering "
                        "them by a cue column chooses it")
            if not _name_set:
                _q_notes.append(
                    f"{_n_match} of {len(scored)} rows matched "
                    f"({_order_frag}{_rest_frag}).")
            # A MIXED POOL STATES ITS TIER CENSUS as structure the model can
            # hold: the strata are live attribute views.
            _n_ex = sum(1 for x in _m_vals if x >= T.NAME_TIER_FLOOR)
            _n_tg = sum(1 for x in _m_vals
                        if T._KV_TIER_BASE <= x < T.NAME_TIER_FLOOR)
            _n_fz = _n_match - _n_ex - _n_tg
            if _n_ex and (_n_tg or _n_fz):
                _q_notes.append(
                    f"tiers: {_n_ex} exact, {_n_tg} tagged, "
                    f"{_n_fz} fuzzy -- .exact / .tagged / .fuzzy "
                    "on this result hold each as a set.")
            if _band_cut and not _name_set:
                _q_notes.append(
                    f"matched as a NAME: {_band_cut} row(s) under "
                    f"the name tier are a different feature and are "
                    f"NOT in this rN.")
            if _head_note and not _name_set:
                # "Best" only on a name-tier head; a class/tag head is just
                # the head.
                _lead = ("Best" if _m_vals and _m_vals[0] >= T.NAME_TIER_FLOOR
                         else "Head")
                _q_notes.append(f"{_lead}: {_head_note}")
            # A query token no matched row carries: the match stands, and the
            # unmatched words are named.
            _dead = _memo["dead"].get(_mkey)
            if _dead is None:
                _dead = _memo["dead"][_mkey] = T.unmatched_query_tokens(
                    _q_raw, scored[:_n_match])
            if _dead:
                _q_notes.append(
                    "no matched row carries "
                    + ", ".join(repr(t) for t in _dead[:4])
                    + " -- those words did not narrow anything")
            if _n_match <= 3:
                # LOW-match: the surface census keeps the re-query
                # vocabulary one read away.
                _q_notes.append(_memo_census())
            _q_matched = True
        # THE narrowing: the returned row set IS the match set.
        rows = scored[:_n_match] if _q_matched else scored
        # ``name=``: the model says which of the two it meant, a class word
        # plus a proper noun. The NAME gates (matched as a name, never
        # fuzzily); the class is accepted because the model writes it.
        _want = str(a.get("name") or "").strip()
        if _want:
            _nq = T.fuzzy_prepare_query(_want)
            _named = [r for r in rows
                      if isinstance(r, dict)
                      and T.find_score(_nq, r.get("search_text") or "",
                                       r.get("name"), r.get("tags"),
                                       idf=_idf)
                      >= T._NAME_PHRASE_TIER_BASE]
            if not _named:
                # LOUD, never a silent fall back to the fuzz band this gate
                # exists to exclude.
                raise ProgramError(
                    f"nothing found is NAMED {_want!r}"
                    + (f" and matches {_q_raw!r}" if _q_raw else "")
                    + ". A name is matched as a NAME, never fuzzily "
                    "-- check the spelling, or drop the class word "
                    f"and say find({_want!r}).{cx.sig()}")
            if len(_named) < len(rows):
                _q_notes.append(
                    f"name={_want!r} kept {len(_named)} of "
                    f"{len(rows)} by NAME; the fuzz band does not "
                    "apply to a name")
            rows = _named
            _q_matched = True
    if cx.standing is not None and rows:
        # The standing viewpoint's side columns ride every find row (the
        # stage receipt states this fact). Presence-pair, ordering/reading
        # evidence only -- never a gate.
        rows = _attach_standing_columns(rows, cx.standing)
        _h = str(cx.standing.get("handle") or "v")
        _find_cols.extend(
            (f"left of {_h}", f"right of {_h}",
             f"ahead of {_h}", f"behind {_h}"))
    if not rows and prov != "runtime" and (a.get("query") or prov == "prior"):
        # A ground attempt (a named query, or provenance:'prior') matched
        # nothing: the marker states the searched scope; the query notes ride
        # it as on any empty result.
        rows = _EmptyFindMiss()
        rows.searched = (
            "the prior map rows (provenance:'prior')" if prov == "prior"
            else "every surface row (prior + runtime)")
        if _q_notes:
            rows.notes = list(_q_notes)
        if _empty_cond:
            rows.empty_condition = _empty_cond
        return rows
    # A query-matched find is ORDERED (best-first is real) and its rows
    # render; a query-less browse is unordered and renders census + len only.
    # Either way the rN holds the full returned set for chaining.
    rows = _BroadRows(rows)
    rows.ordered = _q_matched
    # Ordered stays TRUE on a name's set: the rN is indexable and pageable
    # like any find. Only the initial RENDER is the shape.
    rows.name_set = bool(_name_set)
    if _find_cols:
        rows.op_cols = list(_find_cols)
    if _q_notes:
        rows.notes = list(_q_notes)
    if _empty_cond:
        rows.empty_condition = _empty_cond
    return rows


def _x_polys(node):
    """A row's exact geometry as vertex chains (its mask), else its point."""
    m = node.get("mask")
    if m:
        return list(m)
    xy = node.get("xy")
    return [[xy]] if xy else []


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
    """CONSTRUCT the places two features meet: one row per distinct
    crossing, each with its own fid (its [x, y] token) and xy; never a
    centroid over several. Zero crossings is a loud refusal."""
    sides = []
    for _el in a["refs"]:
        _nodes = _named_rows(cx, _el, "xy")
        _polys = [p for n in _nodes for p in _x_polys(n)]
        if not _polys:
            raise ProgramError(
                f"crossings ref {_el!r} names no geometry on this "
                f"surface -- a crossing needs two features that HAVE "
                f"a shape. {cx.avail('point')}{cx.sig()}")
        sides.append((_x_label(_nodes, _el), _polys,
                      {str(n.get("fid")) for n in _nodes}))
    # Nothing crosses itself: a row on both sides would hand back its own
    # corners as crossings.
    _shared = sides[0][2] & sides[1][2]
    if _shared:
        _same = "the same feature" if sides[0][2] == sides[1][2] \
            else "the same feature in part"
        raise ProgramError(
            f"{sides[0][0]} and {sides[1][0]} are {_same} "
            f"({', '.join(sorted(_shared)[:4])}), and nothing crosses "
            f"itself -- a feature's own corners are not crossings. "
            f"Name two DIFFERENT features (find each one and read its "
            f"row).{cx.sig()}")
    (la, fa), (lb, fb) = ((sides[0][0], sides[0][2]),
                          (sides[1][0], sides[1][2]))
    xpts, nearest = G.polyline_crossings(sides[0][1], sides[1][1])
    if not xpts:
        # A row-pair fact, never a street-level verdict: sweep a bounded set
        # of same-named sibling pairs for a crossing.
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
                _px, _ = G.polyline_crossings(_x_polys(_ra), _x_polys(_rb))
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
            f"('... near <feature>').{cx.sig()}")
    rows = _BroadRows()
    # Whole metres: this row's fid is its coordinate and the model has to be
    # able to say it (the language spells whole numbers only). The exact
    # intersection stays in the geometry.
    _seen: set = set()
    for _x, _y in xpts:
        _x, _y = float(round(_x)), float(round(_y))
        tok = format_xy_token(_x, _y)
        # Rounding can collapse two crossings onto one address; two rows
        # sharing a fid would be worse than one row.
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
    return rows


def _ends(cx, a):
    """A road's termini as rows: end0_xy/end1_xy served as rows a cue like
    "the end of Walker Court" can act on."""
    rows = _BroadRows()
    _n_end = 0
    _no_line: list = []
    for _el in a["refs"]:
        _nodes = _named_rows(cx, _el, "centerline")
        if not _nodes:
            raise ProgramError(
                f"ends ref {_el!r} names no row on this surface. "
                f"{cx.avail('point')}{cx.sig()}")
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
        # Loud, never an empty bind; a row fact, never a class verdict.
        _named = ", ".join(_no_line[:4]) + (
            f" (+{len(_no_line) - 4} more)" if len(_no_line) > 4 else "")
        raise ProgramError(
            f"none of those rows carries a recorded centerline -- "
            f"{_named} have no line geometry on this surface, so "
            f"no termini exist for them. ends reads a centerline; "
            f"end0_xy/end1_xy on a row is the same fact served as "
            f"columns.{cx.sig()}")
    rows.ordered = True
    rows.notes = [
        f"{len(rows)} terminus rows; each is its OWN row with its "
        f"own fid, so index the one you mean."]
    return rows


def _score_by(cx, a):
    """THE sort: a sign-aware, key-normalized composite over EXACTLY the
    columns the emphasis names -- nothing is auto-added."""
    items = _as_list(cx, a["items"])
    _sb_notes: list = []
    emphasis: dict = {}
    for _el in a["emphasis"]:
        _reject_renamed_key(_el["key"])
        _k = _el["key"]
        if _is_color_column_key(_k):
            # THE WORD RIDES IN THE KEY: the emphasis reads the canonical
            # column the attach mints (color_navy weights color_blue; noted).
            _canon, _cnote = _resolve_color_word(_k[len("color_"):])
            if _cnote is not None:
                _sb_notes.append(_cnote)
            _k = "color_" + _canon
        emphasis[_k] = float(_el["w"])
    items = _auto_attach(cx, items, list(emphasis), notes=_sb_notes)
    # An emphasis key no row carries (after the auto-attach) would be a
    # silent constant term: loud.
    _dropped_color: list = []
    for _k in list(emphasis):
        if items and not any(isinstance(it.get(_k), (int, float))
                             for it in items if isinstance(it, dict)):
            if _is_color_column_key(_k):
                # An unseen colour drops its term when the sort has other
                # terms: nothing carries a perceived colour before a view
                # passes over it. Alone it still raises: a sort with no term
                # is not an ordering.
                if len(emphasis) > 1:
                    _dropped_color.append(_k)
                    del emphasis[_k]
                    continue
                raise ProgramError(
                    f"nothing orders by {_k!r} here: no row carries a "
                    f"perceived color, so no color column attaches (rows "
                    f"without a color read omit color_<word> columns). "
                    f"{_present_numeric_keys_hint(items)}")
            raise ProgramError(
                f"nothing orders by {_k!r} here: no row carries a "
                f"numeric {_k!r}. "
                f"{_contact_fact(cx, _k, items) or _key_remedy(_k)}"
                f"{_present_numeric_keys_hint(items)}")
    for _k in _dropped_color:
        # NEVER SILENT: the rows are ranked by fewer things than asked for.
        _sb_notes.append(
            f"{_k!r} ranked nothing here: no row on this surface has "
            f"been SEEN yet, so none carries a perceived colour. The "
            f"other keys still rank. A row gets a colour once a "
            f"flight has passed over it.")
    keys = sorted(emphasis)  # deterministic; the sum is order-independent
    r = _CueSortedRows(G.score_by(items, keys, [emphasis[k] for k in keys]))
    r.op_cols = list(keys)
    if _sb_notes:
        r.notes = _sb_notes
    return r


def _filter_items(cx, a):
    """The only op that drops rows. ``le`` cuts a numeric column; ``truthy``
    keeps rows carrying the key; ``on_side`` keeps a row its signed cardinal
    column does not place (absent: the reference straddles the row, or none
    of it lies abreast) and drops one it places on the other side;
    ``in_or_blank`` keeps a word column reading one of ``value`` or blank."""
    items = _as_list(cx, a["items"])
    key, opr, val = a["key"], a["op"], a.get("value")
    # A filter PRESERVES its input's order: a cut over an ordered set stays
    # ordered (and renderable); a cut over a browse stays not.
    _in_ordered = bool(getattr(items, "cue_sorted", False)
                       or getattr(items, "ordered", False))
    items = _auto_attach(cx, items, [key])
    if opr == "le":
        if items and not any(isinstance(it.get(key), (int, float)) for it in items):
            raise ProgramError(
                f"filter_items op {opr!r} on key {key!r}: no item "
                f"carries a numeric {key!r}. "
                f"{_contact_fact(cx, key, items) or _key_remedy(key)}"
                f"{_present_numeric_keys_hint(items)}")
    elif (items
          and not any(isinstance(it, dict) and key in it for it in items)
          and not _relate_ran(key, items)):
        # Loud with the missing key's own producing idiom, unless the relate
        # ran (a 'facing <ref>' absent on every row because every row touches
        # <ref> is a geometric fact; in_or_blank keeps them).
        raise ProgramError(
            f"filter_items op {opr!r} on key {key!r}: no item carries "
            f"{key!r}. "
            f"{_contact_fact(cx, key, items) or _key_remedy(key)}"
            f"{_present_keys_hint(items)}")
    if opr == "in_or_blank":
        # The three-way statement for a word column: rows reading a counted
        # word, rows reading another (dropped), and rows never measured
        # (kept). A compass word composes ("facing south" counts southeast
        # and southwest), so the caller passes the set that counts.
        _on = [it for it in items if it.get(key) is None or it.get(key) in val]
        _blank = sum(1 for it in _on
                     if (it.get(key) if isinstance(it, dict) else None) is None)
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
                f"no surveyed row reads any of {_want} in {key!r}; "
                f"the rows kept are the ones the geometry never "
                f"measured (a row TOUCHING the reference has no "
                f"facing word). Widen, or read the column and "
                f"see which words are actually there.")
        r = _FilterCutRows(_on)
        r.cut_stats = stats
    elif opr == "on_side":
        # The side cut states its three-way outcome: on the side, measurably
        # on the other side (dropped), unplaced (kept).
        def _ok(it):
            v = it.get(key)
            return v is None or (isinstance(v, (int, float))
                                 and not isinstance(v, bool) and v > 0)
        _on = [it for it in items if _ok(it)]
        _blank = sum(1 for it in _on
                     if not isinstance((it.get(key) if isinstance(it, dict)
                                        else None), (int, float)))
        stats = {
            "cut": f"{len(items)}->{len(_on)}",
            "on_side": len(_on) - _blank,
            "other_side_dropped": len(items) - len(_on),
            "no_side_kept": _blank,
        }
        if not (len(_on) - _blank):
            # Nothing SURVEYED is on that side, so what is there has to be
            # seen rather than looked up.
            stats["nothing_on_that_side"] = (
                "the prior map holds nothing on that side -- what is "
                "there has to be SEEN"
                + ("" if withheld("look_for") else
                   ": look_for the class, then explore the place to "
                   "segment it"))
        r = _FilterCutRows(_on)
        r.cut_stats = stats
    elif opr == "le":
        # Cut stats: how much was cut, the nearest excluded row (the smallest
        # value over the bound), the inclusive threshold that would admit it
        # and how many rows that admits. Information only.
        kept: list = []
        excluded: list = []
        for it in items:
            v = it.get(key)
            (kept if v is not None and v <= val else excluded).append(it)

        def _eff(it):
            v = it.get(key) if isinstance(it, dict) else None
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            return v

        stats = {"cut": f"{len(items)}->{len(kept)}"}
        _excl_vals = [(e, it) for it in excluded if (e := _eff(it)) is not None]
        if _excl_vals:
            ne_v, ne_it = min(_excl_vals, key=lambda t: t[0])
            stats["nearest_excluded"] = {
                "fid": str((ne_it.get("fid") if isinstance(ne_it, dict)
                            else "") or "?"),
                key: round(float(ne_it.get(key)), 3),
            }
            stats["admit_at"] = round(float(ne_v), 3)
            stats["n_at_admit"] = sum(
                1 for it in items
                if (v := _eff(it)) is not None and v <= ne_v)
        r = _FilterCutRows(kept)
        r.cut_stats = stats
    else:  # truthy
        r = _FilterCutRows(it for it in items if it.get(key))
    r.ordered = _in_ordered
    r.op_cols = [key]
    return r


def _stratum(cx, a):
    """A find's match tier as a set (r1.exact / r1.tagged / r1.fuzzy). A
    view, never a re-retrieval: input order kept, nothing else dropped, the
    full set live behind its own handle."""
    items = _as_list(cx, a["items"])
    tier = str(a["tier"]).strip().lower()
    if not any(isinstance(it, dict)
               and isinstance(it.get("name_match"), (int, float))
               for it in items):
        raise ProgramError(
            "these rows carry no match tiers -- strata come off a "
            "find's own result."
            + ("" if withheld("look_for") else
               " A pass's minted rows are all THE thing the "
               "watchlist named; there is nothing to stratify."))
    _lo, _hi = {"exact": (T.NAME_TIER_FLOOR, None),
                "tagged": (T._KV_TIER_BASE, T.NAME_TIER_FLOOR),
                "fuzzy": (None, T._KV_TIER_BASE)}[tier]

    def _in_tier(it):
        v = it.get("name_match") if isinstance(it, dict) else None
        if not isinstance(v, (int, float)):
            return False
        return (_lo is None or v >= _lo) and (_hi is None or v < _hi)
    r = _FilterCutRows(it for it in items if _in_tier(it))
    r.ordered = bool(getattr(items, "cue_sorted", False)
                     or getattr(items, "ordered", False))
    return r


def _compare_specs(cx, refs):
    """compare refs -> ``(specs, staged handles)``. A spec is ``(kind,
    payload, label)``: 'row' (payload the surface row; label the ``as``
    label, else derived from its fid), 'point' (an [x, y]), 'union' (the
    member rows -- ``{ref:[fid|rN,...], as}``, the named masks as ONE ref:
    'distance to <label>' is the min gap over the members, the centre
    columns read the union's area-weighted centroid)."""
    specs: list = []
    standing: list = []
    for el in refs:
        as_label = None
        ref_el = el
        if isinstance(el, dict):
            if el.get("as"):
                as_label = _sanitize_token(el["as"])
            ref_el = el["ref"]
            if (isinstance(ref_el, (list, tuple)) and ref_el
                    and all(isinstance(x, str) for x in ref_el)):
                _members = []
                for x in ref_el:
                    if cx.is_ref(x):
                        # A constructed or fid-less row (a crossing, an
                        # [x, y] item) is not in by_fid and rides on its own xy.
                        _got = [cx.by_fid.get(r.get("fid"), r)
                                for r in cx.resolve(x)
                                if isinstance(r, dict)
                                and (r.get("fid") in cx.by_fid
                                     or is_xy(r.get("xy")))]
                        _lbl = as_label or _sanitize_token(x)
                        if not _got:
                            # Name the handle the model wrote and why it is
                            # usually empty: nothing is segmented until
                            # look_for names the class and a flight goes over
                            # the place.
                            raise ProgramError(
                                f"{_lbl} is empty, so 'distance to "
                                f"{_lbl}' is undefined on every row "
                                f"and the ordering would be "
                                f"meaningless."
                                + ("" if withheld("look_for") else
                                   " If these are things the map "
                                   "does not carry (a pool, solar "
                                   "panels, a shed, a car), nothing "
                                   "is segmented until look_for "
                                   "names the class and a flight "
                                   "goes over the place -- arm it, "
                                   "fly, then find them again.")
                                + f"{cx.sig()}")
                        # A cap, a wall-clock net: an rN member can fold a
                        # whole class, and members x items gap calls is the
                        # cost.
                        if len(_got) > _UNION_MAX_MEMBERS:
                            raise ProgramError(
                                f"{_lbl} "
                                f"holds {len(_got)} rows and a union "
                                f"folds at most "
                                f"{_UNION_MAX_MEMBERS}: measuring to "
                                f"the nearest of that many is a "
                                f"whole-map scan per row. Narrow it "
                                f"first (within=, a cardinal, or a "
                                f"tighter query), then measure to "
                                f"what is left.{cx.sig()}")
                        _members.extend(_got)
                        continue
                    _mnode = cx.by_fid.get(x)
                    if _mnode is None:
                        raise ProgramError(
                            f"union ref member {x!r} is not a fid on "
                            f"this surface. {cx.avail('point')}"
                            f"{cx.sig()}")
                    _members.append(_mnode)
                specs.append(("union", _members, as_label))
                continue
        if is_xy(ref_el):
            specs.append(("point", [float(ref_el[0]), float(ref_el[1])],
                          as_label or "xy"))
            continue
        if not isinstance(ref_el, str):
            raise ProgramError(
                f"compare refs element {ref_el!r} must be a bare fid string, "
                f"an rN result, an [x,y] literal, or {{ref:..., as:...}}."
                f"{cx.sig()}")
        node = cx.by_fid.get(ref_el)
        if node is None and ref_el in cx.staged:
            # A staged viewpoint handle attaches the stance's own columns,
            # not the point family.
            if ref_el not in standing:
                standing.append(ref_el)
            continue
        if node is None:
            raise ProgramError(
                f"compare ref {ref_el!r} is not a fid on this "
                f"surface. {cx.avail('point')}{cx.sig()}")
        specs.append(("row", node, as_label))
    return specs, standing


def _compare(cx, a):
    """THE relation attach: evidence columns onto every item row, the row's
    fields passing through. Per ref: 'distance to' (footprint gap, 0.0 on
    contact), 'facing' (compass word of the min-gap vector, absent on
    contact), 'center distance to' and 'bearing from' (centroid to
    centroid), and per requested direction a signed cardinal column. Every
    number comes from a geom_compute primitive. An attach, not a sort: input
    order preserved.

    ``frame:{along:<fid>}`` (the auto-attach's chart of a line ref) attaches
    'along <fid>' (arc metres from its end0) and 'distance to <fid>' (the
    unsigned lateral offset) instead.
    """
    items = _as_points(cx, a["items"])
    # Order propagation: compare preserves input order, so its result is
    # ordered iff its input was (read off the resolved source).
    _src = cx.resolve(a["items"]) if cx.is_ref(a["items"]) else a["items"]
    _in_ordered = bool(getattr(_src, "cue_sorted", False)
                       or getattr(_src, "ordered", False))
    specs, _standing_ref_handles = _compare_specs(cx, a.get("refs") or [])

    # --- per-ref column labels + headers ------------------------------
    notes: list = []
    used_labels: set = set()
    headers: list = []
    col_specs: list = []   # (kind, payload, label)
    for kind, payload, label_base in specs:
        label = label_base or (_sanitize_token(payload["fid"])
                               if kind == "row" else "ref")
        lab = label
        _k = 2
        while lab in used_labels:  # collision -> fid suffix, then counter
            if kind == "row" and _k == 2:
                lab = f"{label}_{payload['fid']}"
            else:
                lab = f"{label}_{_k}"
            _k += 1
        used_labels.add(lab)
        if kind == "row":
            hdr = {"ref_fid": payload["fid"],
                   "ref_name": str(payload.get("name") or payload.get("class")
                                   or payload["fid"])}
            # ref ATTRIBUTE confidence: a color-described detection used as an
            # anchor ('the white car') is only as trustworthy as its perceived
            # color.
            if payload.get("color_conf") is not None:
                hdr["ref_color"] = payload.get("color")
                hdr["ref_color_conf"] = payload["color_conf"]
        elif kind == "point":
            hdr = {"ref_name": lab, "kind": "point",
                   "xy": [round(payload[0], 2), round(payload[1], 2)]}
        else:
            hdr = {"ref_name": lab, "kind": "union",
                   "members": [n.get("fid") or [round(n["xy"][0], 2),
                                                round(n["xy"][1], 2)]
                               for n in payload]}
        headers.append(hdr)
        col_specs.append((kind, payload, lab))

    # --- the chart of a line ref (auto-attach only) --------------------
    frame_join: dict = {}
    flab = None
    if a.get("frame") is not None and items:
        _fid = a["frame"]["along"]
        flab = _sanitize_token(_fid)
        structured, f_receipt = G.frame_polyline(
            items, cx.by_fid[_fid]["centerline"])
        if f_receipt.get("error"):
            raise ProgramError(
                f"compare frame: {f_receipt['error']}{cx.sig()}")
        for rec in structured:
            key = rec.get("fid") or (round(rec["xy"][0], 2),
                                     round(rec["xy"][1], 2))
            frame_join[key] = rec

    # --- row assembly: PASSTHROUGH + columns ---------------------------
    # Every attached key is recorded VERBATIM -- per ref (that header's
    # ``cols``) and op-wide (the result's ``columns`` echo) -- so the model
    # reads real column names instead of guessing them.
    ref_col_keys: list[dict] = [{} for _ in col_specs]
    all_col_keys: dict = {}

    def _put(row, k, val, ci=None):
        row[k] = val
        all_col_keys[k] = None
        if ci is not None:
            ref_col_keys[ci][k] = None

    # Per-ref geometry, resolved once: ``(geom, center)``. The gap side is
    # the exact realised mask, else its point; ``center`` is the mask's area
    # centroid. A union holds its member geoms plus the area-weighted
    # centroid. Refs and items re-resolve geometry BY FID: op output rows
    # strip the mask fields, so a chained row would otherwise degrade to its
    # point. A fid-less row (a literal, constructed geometry) keeps its own.
    def _row_geom_center(node):
        if isinstance(node, dict):
            node = cx.by_fid.get(node.get("fid")) or node
        _cxy = node.get("xy") if is_xy(node.get("xy")) else None
        m = node.get("mask")
        if m:
            g, c = _mask_geom_center(m)
            return (g, node.get("mask_centroid") or c)
        return (G.mask_geom([[_cxy]] if _cxy else []), _cxy)

    col_geoms: list = []
    for kind, payload, _lab in col_specs:
        if kind == "row":
            col_geoms.append(_row_geom_center(payload))
        elif kind == "union":
            _geoms: list = []
            _wx = _wy = _wa = 0.0
            _ucs: list = []
            for node in payload:
                g, c = _row_geom_center(node)
                _geoms.append(g)
                if c is not None:
                    _ucs.append(c)
                    a_m = G.loops_area(node.get("mask") or [])
                    _wx += a_m * float(c[0])
                    _wy += a_m * float(c[1])
                    _wa += a_m
            if _wa > 0:
                u_c = [_wx / _wa, _wy / _wa]
            elif _ucs:
                u_c = [sum(float(c[0]) for c in _ucs) / len(_ucs),
                       sum(float(c[1]) for c in _ucs) / len(_ucs)]
            else:
                u_c = None
            col_geoms.append((_geoms, u_c))
        else:  # constructed point
            col_geoms.append((G.mask_geom([[payload]]), payload))

    # --- the map-fixed cardinal columns -------------------------------
    # The directions asked for, per column, built once: the reference's
    # point cloud is fixed for the whole pass. A cardinals element names its
    # ref by fid or by label.
    _card_req: dict = {}
    for _el in (a.get("cardinals") or []):
        _card_req.setdefault(str(_el.get("ref")), []).append(_el["dir"])
    col_cards: list = []
    for ci, (kind, payload, lab) in enumerate(col_specs):
        want = []
        if _card_req:
            _fid = payload.get("fid") if kind == "row" else None
            for _k in (_fid, lab):
                if _k and _k in _card_req:
                    want = _card_req[_k]
                    break
        if not want:
            col_cards.append(None)
            continue
        _g = col_geoms[ci][0]
        _pts = cardinal_side.points_from_geoms(_g if isinstance(_g, list) else [_g])
        col_cards.append((want, _pts) if _pts is not None else None)

    # Window-covering ref (stamped at build_surface): its gap columns would
    # be a constant 0, so the envelope carries one containment fact instead;
    # the centre columns stay per-row.
    col_covers: list = []
    for (kind, payload, lab) in col_specs:
        covers = bool(kind == "row" and payload.get("covers_window"))
        col_covers.append(covers)
        if covers:
            notes.append(
                f"ref {lab} "
                f"('{payload.get('name') or payload['fid']}'): its "
                f"mask contains the whole surface window, so "
                f"'distance to {lab}'=0 for EVERY row (containment, not "
                f"proximity) and 'facing {lab}' has no "
                f"direction -- "
                f"those two columns are not stamped; "
                f"'center distance to {lab}'/"
                f"'bearing from {lab}' (centroid vector) ride per-row.")

    # Ref-in-mask containment (per ref index): rows whose own mask contains
    # the ref read distance 0; the row carries a 'contains <ref>' cell and
    # one note names them. Sorts unchanged.
    contains_rows: dict[int, list[str]] = {}
    out_rows: list = []
    # Member boxes are a property of the REFERENCE, so they are stacked once
    # per column and reused down the items.
    col_member_bb: dict = {}
    for i, it in enumerate(items):
        row = {k: v for k, v in it.items() if k not in _MODEL_HIDDEN_ROW_FIELDS}
        # Gap intent (distance, facing) uses both sides' exact masks; position
        # intent (centre columns) uses the area centroid. The item resolves
        # through the same ``_row_geom_center`` as the refs, so a ref compared
        # to itself is exactly 0.
        item_geom, it_c = _row_geom_center(it)
        for ci, (kind, payload, lab) in enumerate(col_specs):
            if col_cards[ci] is not None and it_c is not None:
                _dirs, _rpts = col_cards[ci]
                for _d in _dirs:
                    _v = cardinal_side.cardinal_offset(_rpts, it_c, _d)
                    # ABSENT, never 0.0, when there is no side to name: a zero
                    # would read as "exactly on the line" and sort like a real
                    # measurement.
                    if _v is not None:
                        _put(row, f"{_d} of {lab}", round(_v, 1), ci)
            ref_geom, center = col_geoms[ci]
            if col_covers[ci]:
                d = None
            elif kind == "union":
                if ci not in col_member_bb:
                    col_member_bb[ci] = G.stack_bboxes(ref_geom)
                d, wr, wi = union_gap(ref_geom, col_member_bb[ci], item_geom)
            else:
                d, wr, wi = G.gap_witness(ref_geom, item_geom)
            if d is not None:
                dr = round(d, 2)
                _put(row, f"distance to {lab}", dr, ci)
                if dr > 0.0 and wr is not None:
                    # Which way this row faces the ref, as a word: the witness
                    # pair runs ref->item and a thing faces the place it
                    # fronts, so the word is item->ref.
                    _put(row, f"facing {lab}", G.cardinal(wi, wr), ci)
                elif (dr == 0.0
                      and not isinstance(ref_geom, list)
                      and item_geom["closed"]
                      and len(item_geom["A"])
                      and len(ref_geom["V"])
                      and not (kind == "row" and payload.get("fid")
                               and str(payload.get("fid"))
                               == str(row.get("fid")))
                      and bool(G.mask_contains(item_geom, ref_geom["V"]).all())):
                    # the row's OWN mask CONTAINS the ref (every ref vertex
                    # inside it): distance 0 by containment, not adjacency.
                    _put(row, f"contains {lab}", True, ci)
                    contains_rows.setdefault(ci, []).append(
                        str(row.get("fid") or f"item[{i}]"))
            if center is not None and it_c is not None:
                dcr = round(G.dist(center, it_c), 2)
                _put(row, f"center distance to {lab}", dcr, ci)
                if dcr > 0.0:
                    # centroid->centroid bearing: defined on contact too,
                    # ABSENT only when the centroids coincide.
                    _put(row, f"bearing from {lab}",
                         round(G.bearing(center, it_c), 1), ci)
        if frame_join:
            rec = frame_join.get(it.get("fid") or (round(it["xy"][0], 2),
                                                   round(it["xy"][1], 2)))
            if rec is not None:
                _put(row, f"along {flab}", rec.get("along"))
                if rec.get("off") is not None:
                    _put(row, f"distance to {flab}",
                         round(abs(float(rec["off"])), 2))
        out_rows.append(row)
    for ci, (kind, payload, lab) in enumerate(col_specs):
        _cf = contains_rows.get(ci)
        if _cf:
            _cfs = f"{_cf[:8]}" + (
                f" (+{len(_cf) - 8} more)" if len(_cf) > 8 else "")
            notes.append(
                f"ref {lab}: row(s) {_cfs} read 'distance to {lab}'=0 because "
                f"their OWN mask CONTAINS {lab} (every vertex of "
                f"the ref sits inside the row's footprint -- "
                f"containment, not adjacency); each such row "
                f"carries 'contains {lab}'=true. distance reads 0 on ANY "
                f"mask contact, so a 0 in a nearest-first read "
                f"spans touching neighbors AND containers.")
    for hdr, keys in zip(headers, ref_col_keys):
        if keys:
            hdr["cols"] = list(keys)
    for _h in _standing_ref_handles:
        out_rows = _attach_standing_columns(
            out_rows, cx.staged[_h], with_distance=True)
        notes.append(
            f"ref {_h} is your staged viewpoint: its side columns "
            f"('left of/right of/ahead of/behind {_h}') + "
            f"'distance to {_h}' (meters from where you stand) "
            f"ride every row.")
    r = _CompareRows(out_rows)
    r.refs = headers
    r.columns = list(all_col_keys)
    r.notes = notes
    r.ordered = _in_ordered
    return r


# The op vocabulary: find retrieves; crossings/ends construct rows; compare
# attaches relation columns; filter_items cuts; score_by orders; stratum views
# a find's match tier. Everything is a list of rows.
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
                geometry_sink=None, staged=None, standing=None):
    """Execute a compiled op-program over an explicit surface.

    ``program`` is ``{"ops": [{ref, op, args}, ...]}`` in dependency order,
    as ``code_surface.compile_steps`` emits it. ``cands`` / ``anchors`` are
    the surface rows (:func:`build_surface`). ``seed_results`` is a
    read-only {ref: raw_result} map of earlier calls' results: a handle not
    defined in THIS program resolves from it; one defined here shadows it.
    ``staged`` (the act.stage viewpoints) is a read-only ``{handle:
    {"handle", "from_xy", "up_unit"}}`` map: a staged handle as a compare ref
    attaches that viewpoint's side columns, and ``standing`` names the handle
    whose side columns ride every find result. ``geometry_sink`` is accepted
    and ignored.

    Returns ``{results}``; wrap with :func:`compact_results` for the
    model-facing payload.
    """
    cx = _Run(cands, anchors, seed_results, staged, standing)
    ops = program["ops"]
    _seen: set = set()
    for op in ops:
        if op["ref"] in _seen:
            raise ProgramError(
                f"handle {op['ref']!r} is defined by two ops -- one handle names ONE "
                f"op (a program never redefines a handle). Give the second op a "
                f"fresh ref (rN) and reference each by its own name.")
        _seen.add(op["ref"])
    for op in ops:
        cx.op = op["op"]
        # A null arg is an absent one: a handle bound to a row without a fid
        # compiles to items=None.
        cx.results[op["ref"]] = OPS[op["op"]](cx, _Args(
            {k: v for k, v in op["args"].items() if v is not None}, cx))
    return {"results": cx.results}
