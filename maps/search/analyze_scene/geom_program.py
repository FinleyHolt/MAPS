"""Structured op-program executor — the ONE geometry surface, shared by teacher + student.

The reasoner emits a typed op-program (a small DAG of composable ops) and this
module executes it over an explicit ``(cands, anchors)`` surface using ONLY
:mod:`maps.search.analyze_scene.geom_compute` primitives + a few list helpers. There is no
free-Python channel: the op vocabulary IS the surface, the N/S frame flip is
pinned inside ``geom_compute``, and every op returns RAW geometry the caller must
still interpret (it never returns "the answer").

This is the SINGLE source of the executor: the live ``analyze_scene`` tool
(surface from the runtime graph) and the offline synth ``distill/gen/geom_exec``
(surface from a bundle) both import ``run_program`` from here, so the served
surface is byte-identical to the trained surface.

Program shape (JSON) — EXPLORE, ONE composed ``compare``, then the analysis
ordering (programs ANALYZE only; the DECISION is the ``act{select}`` terminal
of the three-verb surface — it is NOT an op):
  {"ops": [ {"ref":"r0","op":"find","args":{"query":"car","provenance":"runtime"}},
            {"ref":"r1","op":"compare","args":{"items":"r0","refs":["prior_3"],
                                               "frame":{"reference":"auto"}}},
            {"ref":"r2","op":"score_by","args":{"items":"r1","emphasis":[{"key":"distance to prior_3","w":-1}]}} ]}

An arg value resolves as: ``"rN"`` (a prior op result) | a feature id (a node
-> its xy or centerline, by op context) | ``[x,y]`` literal | a number/string
literal. (The executor additionally ABSORBS the bare set words
``"cands"``/``"anchors"`` as the injected provenance lists — input tolerance
only; no served surface teaches them.)
"""
from __future__ import annotations

import os
import re

from maps.search.analyze_scene import cardinal_side
from maps.search.analyze_scene import cell_mesh
from maps.search.analyze_scene import geom_compute as G
from maps.search.analyze_scene.cardinal_side import CARDINALS
from maps.search.analyze_scene import text_match as T
from maps.search.render.look_render import LOOK_SET_ROWS
from maps.util.node_fingerprint import fingerprint_node
from maps.util.xy_token import format_xy_token
from maps.search.withhold import withheld

# The ONLY geometry/analysis vocabulary the structured surface exposes — the
# four-op retrieve->relate->filter->order surface. ``select`` is the ``act{select}``
# terminal of the three-verb surface, not an op; programs ANALYZE only.
# EXPLORE (the one ``find`` op) learns what exists; the ref CONSTRUCTORS
# (the frame aspect) fixes viewer frames + arrangement structure; the SET
# FOUR verbs, one contract: everything is a LIST OF ROWS; find makes lists,
# compare annotates lists (per-ref relation columns + the frame aspect's
# arrangement columns — all delegate-only over geom_compute primitives),
# filter_items narrows lists, score_by orders lists; brackets read them
# (rN[k] indexes, rN[a:b] windows — READ is reference grammar, not an op).
OP_VOCAB = (
    "find",          # RETRIEVE: THE retrieval op (provenance selects the surface)
    "crossings",     # CONSTRUCT: the places two features meet (one row each)
    "ends",          # CONSTRUCT: a linear feature's two TERMINI (one row each)
    "compare",       # RELATE: the relation/arrangement attach (refs + frame aspects)
    "filter_items",  # FILTER: the only op that drops rows
    "score_by",      # ORDER: the only sort (one key or a fused composite)
    "stratum",       # VIEW: a find's own match tier as a set (exact/tagged/fuzzy)
)

# Deleted op names -> the redirect the retry needs. One canonical idiom per
# meaning: every arrangement read is a compare{frame} column; every sort is
# score_by; every ordinal read is a column (world arrangement) or an index
# (your own ordering). The error TEACHES the replacement so the retry is
# mechanical, never a dead end.
_DELETED_OPS = {
    "rank_by": ("score_by{items, emphasis:[{key:<k>, w:-1|1}]} (NEGATIVE w = "
                "lower-raw-first). score_by orders by EXACTLY the columns "
                "emphasis weights -- a single-key emphasis IS the one-key "
                "order."),
    "pick": ("index the register instead: rN[2] is the 3rd row of YOUR "
             "ordering (0-based; negatives + windows rN[a:b] work). A WORLD "
             "ordinal ('7th from the right') is a sort + index: score_by on "
             "the axis column ('along <road>' or a side column), then rN[6]."),
    "partition_rows": ("row/band columns are deleted; an axis read is a "
                       "coordinate read: compare frame:{toward:<f>} (the "
                       "viewer look) or frame:{along:<road>, [toward]} (the "
                       "chart) "
                       "attaches the view columns ('left of/right of/ahead of/behind <f>', or the chart set on a road) and "
                       "score_by/filter_items order/cut them."),
    "cluster_cands": ("density-group columns are deleted; nearness is a "
                      "column read: compare refs (a fid or [x,y]) then "
                      "score_by/filter_items on 'distance to <ref>'."),
    "order_along": ("along a ROAD, chart the pool on it: compare{items, "
                    "refs:[<road>]} attaches 'along <road>' (meters "
                    "along the centerline -- exact on bends) + the side "
                    "columns + 'distance to <road>'; score_by"
                    "{emphasis:[{key:'along <road>', w:-1}]} orders from "
                    "the near end."),
    "between": ("'between A and B' is a two-distance read: compare{items, "
                "refs:[<A>, <B>]} attaches 'distance to <A>' + 'distance "
                "to <B>'; score_by{emphasis:[{key:'distance to <A>', w:-1}, "
                "{key:'distance to <B>', w:-1}]} orders the gap-sum "
                "smallest-first, and filter_items le on each distance cuts "
                "to the span."),
    "axis_crossing": ("crossings{refs:[<roadA>, <roadB>]} returns the "
                      "crossing itself -- ONE row per distinct place they "
                      "meet, each with its own fid and xy. It is a real "
                      "point, not an ordering: two small line distances rank "
                      "rows that already exist and can never name a spot "
                      "where none sits."),
    "row_centerline_fit": ("compare{items, frame:{reference:'auto'}} "
                           "derives the pool's own axis and attaches its view "
                           "columns; order/cut those."),
}

# Renamed columns -> the redirect a reader needs. Loud, never a silent
# dead-filter. The convention is the BOTH-ENDS 1-based pair: direction lives
# in the NAME, with no sign/base to invert.
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
# THE PARAMETER RIDES IN THE KEY NAME (the d_<ref> idiom): naming a color
# column anywhere a column key is accepted IS the computation request — the
# emphasis/key lazy attach resolves the word and attaches the column, no
# op-ordering. The closed palette is G.COLOR_ENUM (exactly the vocabulary the
# M6 extractor emits); G.normalize_color maps an open cue word onto it
# (navy -> blue), noted factually on the result envelope.


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


# The served frame forms — the ONE statement every frame teaching splices in
# (prompts._COMPARE_FRAME offers exactly {reference, toward, along, as}; the
# import-time guard in prompts.py holds the compare signature to that schema).
_FRAME_FORMS = (
    "frame takes EXACTLY ONE form: reference:'auto' (no declared view -- "
    "the pool's own axis), {along:<line fid|[[x,y],...]>, [toward:<fid|"
    "[x,y]|compass>]} (THE CHART: travel the line; 'along <ref>' + the "
    "side columns follow the travel), or {toward:<fid|[x,y]|compass>} "
    "(the viewer look: stand at the pool center, look at toward). "
    "as:'label' names the columns.")

# Per-op signatures appended to a ProgramError so a retry knows the op's contract —
# its arg names AND the load-bearing arg-KIND (a single POINT vs a CENTERLINE vs a
# candidate LIST), the distinction the model conflates. Generic / abstract: op names
# from OP_VOCAB + generic ``rN`` placeholders only, NO episode vocabulary (never
# teach to the test). Surfaced by the resolvers + ``_Args.__missing__`` via the live op.
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
                + _FRAME_FORMS +
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

# The legal ``filter_items`` comparison ops, single-sourced so the typed args
# schema (prompts.ANALYZE_SCENE_PROGRAM_SCHEMA) can NEVER offer a value
# ``run_program`` rejects (same contract as ``OP_VOCAB`` for the op enum).
FILTER_OPS = ("eq", "ne", "lt", "le", "gt", "ge", "truthy", "on_side",
              "in_or_blank")

# ``on_side`` exists because VIOLATING and UNPLACEABLE are different facts and
# every other numeric op collapses them. A signed cardinal column is positive
# where the row IS that way, negative where it measurably is NOT, and ABSENT
# where the geometry declines to name a side at all (the reference straddles
# the row, or none of it lies abreast). ``gt 0`` would drop the absent rows
# with the violating ones -- and on the live case that motivated this, all 19
# real parking rows in the bake ABSTAINED against the school while 163
# fuzzy-matched rows carried a measured violation. Cutting on ``gt`` would
# have thrown away exactly the rows still in play.
#
# Keep a row when the column does not place it, drop it when the column
# places it on the other side. The cut is earned: the relation is right on
# 150 of 154 addressable corpus cues.

# Generic color basis: _color_of extracts a color TOKEN from a class label
# (legacy color-qualified concepts). PARITY NOTE: the surface ``color`` field is
# produced by _node_color, which reads PERCEIVED color off semantic_attributes
# when present (live runtime graph + the captured-live geom_surface) and falls
# back to _color_of(class) otherwise. So color is NOT byte-parity on the offline
# REBUILD-from-tile path (offline tile nodes carry no semantic_attributes ->
# color='' -> find keeps ALL such rows, colorless); only
# the captured-live geom_surface is color-parity-safe. The LIVE eval is
# unaffected (perception supplies color); populating offline perceived color is
# a distillation-parity follow-up. The descriptions WALL and the name/class/tag
# search_text parity are unaffected by this. The color_<word> columns are attached
# as sortable per-row fields inside run_program (the find color branch + the
# emphasis-key lazy attach; NOT at build_surface time), so the gated build_surface
# surface is unchanged; on the offline tile-REBUILD path color='' -> the column
# is omitted (a colorless row is never pulled by a color emphasis).
_COLOR_WORDS = ("white", "black", "gray", "gray", "silver", "red", "blue", "green",
                "yellow", "orange", "brown", "tan", "beige", "gold", "maroon", "purple")


def _color_of(label: str) -> str:
    for w in _COLOR_WORDS:
        if w in (label or "").lower():
            return w
    return ""


def _node_color(n) -> str:
    """Perception-derived color for a node. Reads the enriched ``color`` attribute
    off ``semantic_attributes`` ({key: {value: count}}; top-count value wins) -- the
    structured attribute the perception color-enrichment populates over the BARE-NOUN
    mask. Falls back to a color token in the class_name label (legacy color-qualified
    concepts / prior nodes). This is NOT cue-language processing: it labels what the
    object LOOKS like (like the SAM class), and the agent matches the instruction's
    color word to it by reasoning (color is also surfaced per-candidate in the
    selection listing for the fuzzy cases -- silver vs gray -- a filter can't settle)."""
    sa = getattr(n, "semantic_attributes", None) or {}
    cc = sa.get("color") or {}
    if isinstance(cc, dict) and cc:
        return str(max(cc.items(), key=lambda kv: kv[1])[0])
    if isinstance(cc, str) and cc:
        return cc
    return _color_of(str(getattr(n, "class_name", "") or ""))


def _node_color_conf(n):
    """Perception's CONFIDENCE in this node's perceived ``color`` in [0,1], or None.

    Reads the ``color_conf`` bucket off ``semantic_attributes`` (stored by
    perception._set_color_attr as a stringified float, accumulated as {value:count}
    like every attribute) and returns the modal bucket parsed to float -- with the
    measured single-observation reality (one color vote per car node) this IS the
    per-detection confidence. Returns None when ABSENT (the offline tile-REBUILD graph
    carries no semantic_attributes) so the surface row OMITS color_conf and a color
    column falls back to its conf=None family-soft number -- byte-parity with the
    pre-change offline-rebuild path (the same parity boundary _node_color already
    lives on)."""
    sa = getattr(n, "semantic_attributes", None) or {}
    cc = sa.get("color_conf") or sa.get("color_conf") or {}
    raw = None
    if isinstance(cc, dict) and cc:
        raw = max(cc.items(), key=lambda kv: kv[1])[0]
    elif isinstance(cc, (str, int, float)) and cc != "":
        raw = cc
    if raw is None:
        return None
    try:
        return round(float(raw), 3)
    except (TypeError, ValueError):
        return None


def _node_color_shade(n):
    """M6's value-axis shade tag ``dk``|``mid``|``lt`` off ``semantic_attributes``, or
    None when absent (offline tile-rebuild rows, or a legacy single-word color). Drives
    the compact ``col`` cell's dk/lt prefix + enables the ``/word2`` mixture component
    (a color-name posterior; the palette mixture's 2nd component is glass/shadow
    noise, so its absence correctly suppresses ``/word2``)."""
    sa = getattr(n, "semantic_attributes", None) or {}
    cc = sa.get("color_shade") or {}
    if isinstance(cc, dict) and cc:
        return str(max(cc.items(), key=lambda kv: kv[1])[0])
    if isinstance(cc, str) and cc:
        return cc
    return None


def _node_color_flag(n):
    """M6's uncertainty flag off ``semantic_attributes`` — ``'look'`` when the chroma
    sits in the ambiguous achromatic/chromatic band (resolve with a viewpoint), else
    ``''``/None for a confident read (which ``_color_fields`` drops via ``if flag:`` so
    only ``'look'`` ever reaches a row). Renders the compact ``col`` cell's ``?``."""
    sa = getattr(n, "semantic_attributes", None) or {}
    cc = sa.get("color_flag") or {}
    if isinstance(cc, dict) and cc:
        return str(max(cc.items(), key=lambda kv: kv[1])[0])
    if isinstance(cc, str) and cc:
        return cc
    return None


# The tag keys withheld from the model-facing tag surface (search blob + the
# row's rendered tags). TWO families, both bookkeeping — never semantics:
#  * the GT-label wall: an OSM/CityRefer ``description`` tag must never feed
#    the agent's free-text search — it is the catalog the instruction was
#    sampled from.
#  * the bake's own ``source`` provenance channel (osm / cityrefer_*): the
#    row's provenance already rides the fid + the find ``provenance`` arg; as
#    a tag it is one fact repeated on every row.
# Every other tag rides verbatim.
_SEARCH_TAG_DENYLIST = frozenset({"description", "descriptions", "source"})


def _row_tags(n) -> dict:
    """A prior row's model-facing tags: the node's raw OSM tags verbatim
    (string values), minus only the bookkeeping keys (_SEARCH_TAG_DENYLIST)."""
    tags = getattr(n, "tags", None) or {}
    if not isinstance(tags, dict):
        return {}
    return {str(k): v for k, v in tags.items()
            if isinstance(v, str) and str(k).lower() not in _SEARCH_TAG_DENYLIST}


# The BAKE-PROVENANCE keys, withheld from the searchable free-text (and from
# the free-text ONLY -- the node keeps every tag, ``_row_tags`` returns them
# all, and the exact tag tiers and ``tag_family_keys`` read them unchanged).
# These are not map vocabulary; they are bookkeeping the bake stamped on
# every row, and they were the fuzzy haystack's dominant near-stopwords:
# ``way`` is a token of every ``osm_id: way/1136853905`` (the normalizer
# splits on '/'), so ``find("hugo way")`` covered one of two words on 4,733 of
# 5,257 rows; ``worldkit_layer=roads`` donated ``road`` to all 810 of its
# rows. Eleven of the top thirteen near-stopword tokens came from these four
# keys, and ZERO of 53,711 real query uses contains a '/'.
#
# All four are ALREADY on ``_TAG_RENDER_DROP`` -- the tree had decided the
# model should not READ them; they stayed in the haystack by omission. The
# full 16-key drop set is deliberately NOT used: measured at 80.3% vs 79.8%
# precision, which does not pay for the extra blast radius.
_SEARCH_PROVENANCE_DROP = frozenset({
    "osm_id", "worldkit_layer", "worldkit_height_m", "source_tag"})


def _anchor_search_text(n) -> str:
    """Walled searchable free-text for a PRIOR row: name + raw OSM tag keys
    AND values, minus the bake-provenance keys (_SEARCH_PROVENANCE_DROP).

    The free-text the ``find`` op matches a cue's named landmark/type against
    (a 'parking' query reaches amenity=parking, service=parking_aisle and
    parking=surface rows through the keys as well as the values). Prior nodes
    are MapObject on BOTH the live (runtime graph) and offline
    (``_build_node`` tile) sides with identical OSM ``tags``, so this text is
    teacher/student parity-safe. NEVER reads ``MapObject.descriptions`` (the
    CityRefer GT-label catalog) or coordinates — that is the descriptions
    wall; a ``description`` tag is withheld too as defense-in-depth."""
    parts = [str(getattr(n, "name", "") or "")]
    for k, v in _row_tags(n).items():
        if str(k) in _SEARCH_PROVENANCE_DROP:
            continue
        parts.append(str(k).replace("_", " "))
        parts.append(str(v).replace("_", " "))
    return " ".join(p for p in parts if p).lower()


def _cand_search_text(n) -> str:
    """Walled searchable free-text for a DETECTION row: class name ONLY.

    A detection's runtime ``tags``/``name`` are asymmetric (live SAM
    label/source_tier vs offline none), so only ``class_name`` — symmetric on
    both sides — is parity-safe here. Color is matched via the first-class
    ``color`` field (``find{color}``), not free text."""
    return str(getattr(n, "class_name", "") or "").lower()


class ProgramError(Exception):
    pass


class _Args(dict):
    """An op's args dict that raises a CLEAN ``ProgramError`` (named op + key)
    when a REQUIRED arg accessed via ``a["k"]`` is missing — so the agent reads
    an actionable message and retries, instead of a cryptic ``KeyError``.
    Optional args via ``a.get("k")`` are unaffected."""

    def __init__(self, d, op_name):
        super().__init__(d)
        self._op = op_name

    def __missing__(self, key):
        sig = _OP_SIGNATURE.get(self._op)
        suffix = f"; signature: {sig}" if sig else ""
        raise ProgramError(
            f"op {self._op!r} is missing required arg {key!r}{suffix}")


def _color_fields(n) -> dict:
    """``{"color": ...}`` plus ``"color_conf"`` ONLY when perception populated it.
    color_conf is OMITTED (never None/null) when absent so an offline tile-rebuild row
    is BYTE-IDENTICAL to the pre-change shape (no spurious key) -- keeping the
    offline-rebuild surface parity intact; present, it is a float in [0,1] the model
    reads as the reliability of the perceived color."""
    fields = {"color": _node_color(n)}
    conf = _node_color_conf(n)
    if conf is not None:
        fields["color_conf"] = conf
    # M6's two extra reads — present ONLY on the live path (semantic_attributes),
    # absent (omitted, never None) on offline tile-rebuild rows so their shape stays
    # byte-identical. _mask_noise_color folds them into the compact ``col`` cell and
    # strips them, so they never reach the model as raw fields.
    shade = _node_color_shade(n)
    if shade is not None:
        fields["color_shade"] = shade
    flag = _node_color_flag(n)
    if flag:
        fields["color_flag"] = flag
    return fields


def _mask_center(mask):
    """The centre a mask may honestly claim as its position — or ``None``.

    ONE place has a centre; several places do not. ``center distance to
    <ref>`` and ``bearing from <ref>`` are SERVED columns read straight off
    this point, and it also orients ``frame{side:}``, so a centre over
    several disjoint pieces is a confident-sounding answer at none of them —
    the law the crossings op already states about crossing places.

    A name resolves to the union of its ways (:func:`_merge_named_features`),
    and 117 of 5,257 calcity anchor rows come back as several places under
    one name: "98th Street" is two ways 869 m apart and its mean is on
    neither. Those rows now serve the honest absence — the columns are simply
    not there — while ``distance to <ref>``, which is the gap to the NEAREST
    piece and correct however the map split the feature, keeps reading. No
    per-piece centre either: the merge exists so that no op is ever handed a
    piece of a named feature.
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
    flat = [p for ch in chains for p in ch]
    if not flat:
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


# A surface row is a PURE FUNCTION of its node's content and the tile
# geometry, so an unchanged node need not be rebuilt. The rebuild runs once
# per flight over every node on the graph, and an explore pass runs it once
# per stop: calcity::6#6 rebuilt all 6,188 Cal City rows 262 times, and its
# graph gained NOTHING on any of those stops (`cands=0`, `evidence=0`).
# Measured on that bake: 2.66 s to rebuild, 0.18 s to prove nothing changed.
#
# CONTENT, NOT IDENTITY, and not a dirty-list either. Every node the ingest
# fold touches comes back over the wire as a fresh object, so an identity check
# misses on exactly the nodes that matter; and a hand-maintained dirty set
# would have to be right about every mutation path in the tree forever,
# silently serving a stale row the first time one was forgotten. Hashing what
# the row is built FROM cannot go stale: a node that changed anywhere in its
# ``__dict__`` gets a different key and is rebuilt.
_ROW_CACHE: "dict[tuple, dict | None]" = {}
# Steady state is one entry per node per tile (6,188 on Cal City). The cap is
# a runaway backstop for churning detections, not a tuning knob; on overflow
# the cache is dropped whole rather than evicted one at a time, because the
# next rebuild refills exactly what it needs and a partial sweep does not.
_ROW_CACHE_MAX = 50_000

# A row's mask geometry is built once per mask list. ``G.mask_geom`` turns
# the chains into the vectorised arrays every gap reads, and the compare op
# rebuilt it for every item on every call: 3,692 buildings per
# ``find("building", near=...)``, over the same lists every time. The key
# is the mask list's identity, pinned by the entry so the id cannot be
# reused; the row cache above shares one mask list per unchanged node across
# rebuilds, so the memo survives a flight for every node it did not touch.
# Dropped whole at the cap, like the row cache.
_MASK_GEOM: dict = {}
_MASK_GEOM_MAX = 50_000

# THE RETRIEVAL IS MEMOISED PER SURFACE. A find scores every row of the
# surface against its query -- 7,405 rows, 0.3 s on the Cal City bake --
# and a mission asks the same query again and again: ``for d in cars:
# find("building", near=d, dwithin=40)`` scores "building" once per car.
# The rows are a pure function of (surface, provenance, query, colour), and
# the surface is the namespace's two row lists, replaced whole on every
# rebuild; so the memo keys on the lists' identity, pinned by the entry,
# and empties itself the moment either list is a different object. It
# hands out COPIES of the scored rows: a result must never alias another
# result's dicts.
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
    sub-cell position, 2dp) and falls back to the cell-CENTER of the (int)
    centroid — ONE rule on both sides (teacher/student parity is the RULE,
    not the precision: offline tile nodes carry no world_xy and read
    cell-center exactly as before). The 2 m cell grid quantizes arrangement
    structure away (a 2.4 m in-row spacing is unresolvable at 2 m; measured:
    it erased a real lot's lattice and mega-banded the frame), so the fine
    position is the evidence the arrangement columns stand on. Detections
    carry NO centerline (points; the frame aspect derives a row axis from the
    pool); priors carry their baked centerline (cell-center world, 2dp). Both
    lists sorted by fid so pick{index} ties resolve identically.

    UNIFORM FOOTPRINT: every node with a realised mask — a detected car AND a
    prebake building — gets the SAME mask-derived fields via ``_mask_fields``:
    the hidden EXACT ``mask`` loops + ``mask_centroid`` (the compare op's gap /
    center substrate — the ingested mask verbatim, every fragment and hole,
    no decimation) plus two VISIBLE opt-in shape scalars
    (``footprint_area_m2``/``aspect_ratio``). Nodes
    without a mask (a centroid-only detection mint) stay bare, exactly as before.
    """
    tx, ty = float(origin_xy[0]), float(origin_xy[1])
    cs = float(cell_m)

    def _w(rc):
        return [round(tx + (int(rc[1]) + 0.5) * cs, 2), round(ty + (int(rc[0]) + 0.5) * cs, 2)]

    def _xy(n, mask=None):
        """The row's served point -- the ``act.move`` FLIGHT TARGET.

        A MEASURED position wins: a detection's ``world_xy`` is where the
        tracker put the object, sub-cell, and the arrangement columns stand
        on that precision. Failing one, the point is REPRESENTATIVE of the
        row's own geometry (:func:`representative_point`) and never a mean
        over it: a mean is off a loop road, off a bent way, and off every
        piece of a name that is several places. 130 of the 5,257 calcity
        anchor rows served a point more than 20 m from their own geometry,
        the worst 1,123.1 m out, and flying one of those goes where the
        feature is not.

        This is not a claim about WHICH place a several-place name means --
        ``mask_centroid`` still refuses to serve a centre for those
        (:func:`_mask_center`) and ``distance to`` still measures to the
        nearest piece. It is the weaker and true statement that a flight
        target has to be somewhere the feature is.
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

        THE primary geometry: every spatial op measures on this, and the mask
        loops and the centerline are reductions of it kept for rendering and
        for the terminus columns. A list rather than an array so a surface
        round-trips through JSON unchanged (the offline teacher reads the same
        bytes the live tool serves).

        ``tolist`` and NOT a comprehension: this is the hottest function in
        the whole rebuild, and the rebuild runs once per flight. The Cal City
        graph carries 3.9 M cells over its 6,188 nodes, and converting them
        one Python ``int()`` at a time cost 4.0 s of the 5.1 s build.
        ``cells_of`` already returns an ``(n, 2)`` int array, so ``tolist``
        builds the same nested list of native ints in C -- 0.45 s, and
        byte-identical on all 6,188 rows.
        """
        c = cell_mesh.cells_of(n)
        return c.tolist() if len(c) else []

    def _axis_q(n, key):
        # axis_length_m is meaningful only for a node that HAS a baked
        # centerline (a linear feature); None when there is no centerline.
        # (axis_residual_m was retired from the served surface: zero
        # operations across every audited run.)
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

        THE CLIP. The bake rasterises the full OSM geometry and then clips the
        CELLS to its grid, so a feature's cells stop at the tile edge while
        its contour runs on — 72 of 353 named calcity features carry vertices
        outside the tile. The node's OWN cells are therefore the record of
        which of its geometry this tile kept, and the box they occupy is the
        clip: for a feature wholly inside it is a no-op, and for one that
        leaves it is the tile edge. That keeps the rule inside the module
        that owns it (``cell_mesh``: "the cells are what a feature COVERS")
        rather than threading tile bounds through every caller of
        ``build_surface``.

        One cell of margin, because a cell CONTAINS the chain that rasterised
        it and its centre is half a cell inside its own edge; without the
        margin a vertex on the far side of its own cell clips away. The
        margin stops at the tile ORIGIN: cell index -1 is off the map, and
        growing into it serves a point outside the declared bounds. An
        unnamed stream crossing Cal City's south edge kept a contour vertex
        at y -0.91 that way and became the row's representative point.

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
        # UNIFORM mask-derived footprint substrate + shape descriptors, computed
        # identically for a detected car and a prebake building from the node's
        # ingested mask. Two kinds of field:
        #  * ``mask`` + ``mask_centroid`` — the EXACT mask realisation (a list
        #    of world-meter vertex chains, each stating its own closure: the
        #    exact geometry the node was given, one chain per part — or, for a
        #    node with none, the full multi-loop cell boundary, every fragment
        #    and hole, lossless) and its TRUE area centroid. The compare op's
        #    gap (d_/brg_gap_) and center (dc_/brg_) substrate. MODEL-HIDDEN
        #    (see _MODEL_HIDDEN_ROW_FIELDS): ops resolve them server-side; the
        #    model never pays the vertex tokens.
        #  * ``footprint_area_m2`` / ``aspect_ratio`` —
        #    two VISIBLE opt-in shape scalars (min-area-rect; see
        #    G.mask_shape_features) the model MAY compare/score_by/filter on to
        #    tell a sedan from a van from a truck. They are additive columns
        #    like axis_length_m — never forced, never gated. No axis heading
        #    rides here: orientation is measured against the feature a thing
        #    faces, never read off its own bounding rectangle.
        # A node without a mask gets NO mask fields (a centroid-only mint stays
        # bare) -> compare falls back to centroid distance for that ref.
        #
        # WORLD_CONTOUR substrate: the node's exact geometry REPLACES the 2 m
        # cell-quantized loops as the mask and backs the shape scalars
        # REGARDLESS of cell count — the 6-cell S2 gate below guards only
        # CELL-derived shape (2 m raster quantization garbage); a
        # contour-backed 2-cell car gets its real
        # footprint_area_m2/aspect_ratio. An OPEN chain
        # encloses nothing, so it gets no shape scalars at all
        # (G.contour_shape_features) — a road's area column was the area of
        # the line buffer that rasterized it.
        cells = getattr(n, "cells", None)
        chains = _exact_chains(n)
        # A CHAIN OF ONE VERTEX IS NOT A SHAPE. This gate tested chains for
        # NON-EMPTINESS, so an OSM POI node -- whose whole contour is the
        # point it sits at -- served a one-vertex mask and stopped here.
        # Both renderers need three vertices to draw a ring (_proj_poly),
        # so that mask draws NOTHING: 518 of 5,257 calcity anchors carry a
        # footprint no picture can ever show, prior_6005 "Rite Aid" among
        # them. The cells branch below builds a real 5-vertex loop off the
        # same node's raster, so the outline already existed -- this gate
        # was the only thing between it and the frame.
        #
        # >= 2 AND NOT >= 3, deliberately. The 224 two-vertex rows are
        # single-segment WAYS, and routing those to the raster moves one
        # served point 1,417 m (prior_4956 "78th Street": a 259-cell
        # multi-loop boundary puts representative_point on a different
        # component than the way's own midpoint) and contradicts
        # cell_mesh's stated policy that distance and intersection run on
        # the EXACT geometry. Every row this admits is exactly ONE cell,
        # which bounds the served-xy shift at half a cell diagonal --
        # measured 0.80 m median, 1.38 m worst over all 518.
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
        # Shape scalars only when the raster can EXPRESS shape (gt_mi
        # first-dozen audit S2): below ~6 cells the 2m grid quantizes
        # area to {4..16} m2, aspect to {1,2} and orientation to {0,90} —
        # fabricated axis-aligned headings that invite false inference.
        # The mask (the gap substrate) stays; only the descriptors
        # are withheld until a finer source (SAM pixel mask) computes them.
        if len(cells) >= 6:
            shape = G.mask_shape_features(cells, cs)
            if shape:
                out.update(shape)
        return out

    # ``_mask_fields`` is evaluated ONCE per node and its ``mask`` is handed
    # to ``_xy``: the served point is representative of the very geometry
    # this row serves, and cannot drift from it by being derived twice.
    def _cand_row(f, n):
        # ``_mask_fields`` is evaluated ONCE per node and its ``mask`` is
        # handed to ``_xy`` -- see the note above.
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
    # PRIOR rows carry NO class field: a prior's semantics are its name + raw
    # OSM tags (rendered compact k=v); only detections carry ``class`` (the
    # detector's measurement).
    # ONE ROW PER WAY. A named road is routinely several OSM ways (75 rows
    # carry "California City Boulevard" on the calcity bake), and every one of
    # them is served: the SET is what a name resolves to, and ``.exact``
    # names it. A row is therefore always one way, so its cells, its
    # centerline and its position all describe the same thing, and the fid a
    # row carries resolves to that way in the runtime graph too -- which a
    # folded row did not, because ``explore`` walks
    # ``SearchLoop.node_coverage_stops`` off the graph node and got one
    # member's cells for the whole street.
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
    # A centerline row RENDERS its termini: end0_xy is the along_=0 end of
    # every chart over it, end1_xy the far end — so a terminus/corner cue is
    # derivable from the row itself (4/5 audited spatial failures hand-derived
    # crossings they could not act on).
    for _ar in anchors:
        _acl = _ar.get("centerline")
        if isinstance(_acl, list) and len(_acl) >= 2:
            _ar["end0_xy"] = [round(float(_acl[0][0]), 2),
                              round(float(_acl[0][1]), 2)]
            _ar["end1_xy"] = [round(float(_acl[-1][0]), 2),
                              round(float(_acl[-1][1]), 2)]
    # Fold PURE duplicate cand rows — identical (xy, class) — keeping the
    # best-attributed one (color present > absent, then higher color_conf,
    # then lowest fid, NUMERICALLY: the string compare let "detected_11"
    # beat "detected_3", evicting the fid a handle already held).
    # Duplicate coincident rows inflate ordinal/row counts (verified in 3
    # of the 8 lost ordinal episodes). Distinct classes at one xy (the
    # deliberate class-gated color layers) are NOT folded. The set-look
    # note still states an eviction by name — an evicted row and a row
    # too far to frame are two different facts and only one is true.
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
    # NATURAL fid order, never string order. Lexicographic collation put
    # every detected_1xx ahead of detected_2, so once a pass minted ~110
    # rows its EARLIEST stops' finds fell past every served head -- the
    # 8-crop catch, the 24-row broad head, the model's own r1[0:20] --
    # and 140#138's GT lot (minted at stop 2, index 37 of 114) sat on
    # the surface all episode, unreachable by position. Numeric order is
    # mint order, which for a pass is STOP order: the heads read along
    # the chain, and the catch's "in pass order" is true again.
    cands.sort(key=lambda c: _fid_key(c["fid"]))
    anchors.sort(key=lambda a: _fid_key(a["fid"]))
    # WINDOW-COVERING MASK FACT (geometric, tag-blind): a mask that CONTAINS
    # every surface row position (a tile-covering admin/site relation) makes
    # the gap TO it degenerate (d_=0 for every row). Stamp the VISIBLE
    # ``covers_window`` fact on the row: the compare op serves the one
    # containment fact instead of the constant columns, and the row's own
    # render carries it wherever it appears as an item. Bbox prefilter keeps
    # the exact containment test to the handful of window-spanning masks.
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
    """Split a flat ``(fid, node)`` iterable into detections (cands) vs priors
    (anchors) by ``node.provenance`` and run :func:`build_surface`. The ONE
    graph→surface path: the live ``analyze_scene`` tool (agentic, over the
    runtime + prebake graphs) and the preflight Stage-2a round (over
    ``ctx.graph``) both call this, so a preflight program and an in-flight
    program see the same surface contract."""
    det, prior, seen = [], [], set()
    # TARGET-TYPE WALL (config B, benchmarks/citynav/PRIOR_CONTRACT.md): an
    # unnamed baked CityRefer prior of the episode's target annotation type is
    # the annotation's own target geometry — dropping it here (the ONE
    # graph→surface seam) keeps the target reachable only through live
    # perception. The wall keys on the row's OWN ``object_type`` tag (the
    # CityRefer annotation type riding the bake). Set per-episode by
    # run_benchmark; absent = no wall (offline tools, tests).
    walled = os.environ.get("SEARCH_WALLED_UNNAMED_TYPE") or None
    for fid, node in nodes_iter:
        fid = str(fid)
        if fid in seen:
            continue
        seen.add(fid)
        if getattr(node, "provenance", "prebaked") == "detected":
            det.append((fid, node))
            continue
        _tags = getattr(node, "tags", None) or {}
        if (
            walled
            and _tags.get("object_type") == walled
            and _tags.get("source") == "cityrefer_unnamed"
        ):
            continue
        prior.append((fid, node))
    return build_surface(det, prior, origin_xy, cell_m)


# Fields resolvable SERVER-SIDE by fid that no model-side decision needs verbatim.
# ``search_text`` is the fuzzy-match haystack (descriptions walled out).
# ``centerline`` is a full polyline (+~5-6 tokens/vertex; a road can be +300-1800
# tokens PER ROW) recomputed by every geometry op from ``by_fid``
# — the model chains ops on fids, it never needs the vertices in-prompt. The
# scalar shape field (axis_length_m) stays on the row.
# ``mask`` / ``mask_centroid`` are the EXACT footprint realisation (multi-loop
# boundary of the ingested mask + its true area centroid) the ``compare`` op
# resolves server-side (gap_witness); the model reads the d_*/dc_* columns,
# never the vertices.
# ``cells`` is the bake's own 2 m mesh: what a feature COVERS. It is NOT what
# the spatial ops measure — distance and intersection run on the EXACT chains
# each node was given (``gap_witness`` / ``polyline_crossings`` over
# ``mask``), because a cell-set intersection agreed with them on only 85.0% of
# the 397 road pairs the model asked for. Thousands of pairs per row, resolved
# server-side by fid.
_MODEL_HIDDEN_ROW_FIELDS = ("search_text", "centerline", "mask",
                            "mask_centroid", "cells")

# Render-INPUT fields: folded into the compact color cell at render time
# (:func:`_mask_noise_color` — the dk/lt prefix and the ``?``), never served
# as raw columns. Every "columns present" echo excludes them too (one rule:
# every served key is documented or hidden; these are hidden-by-folding —
# they leaked through the present-columns lists, measured 12+5 mentions).
_RENDER_INPUT_ROW_FIELDS = ("color_shade", "color_flag")


# Below this perception confidence a color LABEL is noise — and the model
# LITERAL-MATCHES color words (measured: a decoy "white/white/red" pattern at
# conf 0.004 outweighs anchor distance when it is the only literal match
# rendered). The color-column math (color_match) is already conf-shrunk; the WORD is not. Below
# the floor the MODEL-FACING render prints "?" so noise looks like noise; absent
# conf (offline rebuild rows) keeps the word. RENDER-ONLY: the raw rows the ops
# chain on keep the real color, so the color columns are byte-
# unchanged. Hosted here so EVERY model-facing row — compact_results items,
# compare KV records, a select's echoed row — masks identically on the live and
# offline paths (cached bundles are stale against this render — regenerate before
# any distillation).
_COLOR_CONF_FLOOR = 0.15


def _mask_noise_color(row):
    """Render the color cell as a token-minimal ``col`` tag: ``[dk|lt ]word[/word2][?]``
    (~1-2 tokens vs the ~12-token mixture). A trailing ``?`` = the read is UNCERTAIN
    (below-floor color_conf OR an extractor ``color_flag='look'``) -- the SAME 'noise looks
    like noise' signal as the legacy masked ``?``, which the agent resolves with a viewpoint;
    an M6 ``color_shade`` (dk/lt) prefixes only when present. Pure render transform: the RAW
    row keeps the mixture string, so the color columns / dedup / _parse_color_mixture are
    untouched. Returns the row unchanged when it carries no color."""
    if not isinstance(row, dict):
        return row
    c = row.get("color")
    if not (isinstance(c, str) and c.strip()):
        # shade/flag are render INPUTS to the `col` tag — with no color
        # word there is nothing to fold them into, but they must still
        # never reach the model as raw fields (a leaked color_shade on a
        # colorless row is an undocumented served column).
        if "color_shade" in row or "color_flag" in row:
            row = dict(row)
            row.pop("color_shade", None)
            row.pop("color_flag", None)
        return row
    conf = row.get("color_conf")
    uncertain = ((isinstance(conf, (int, float)) and conf < _COLOR_CONF_FLOOR)
                 or row.get("color_flag") == "look")
    shade = row.get("color_shade")
    # /word2 is meaningful only from a color-name posterior (the M6 extractor sets color_shade);
    # the palette-MODE mixture's 2nd component is glass/shadow noise, so suppress it there.
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
    hidden fields are pure context weight in the model-facing payload (the
    dominant term of the dense-scene prompt blow-up). Idempotent where absent."""
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
    # A tied one-name set: the initial render is the SHAPE (len + the note
    # naming the street and its size), never rows or fids. The handle is a
    # normal ordered find -- slice it, page it, explore it.
    name_set: bool = False
    # POSITIVE ordering marker: True when this list's order is real
    # (query-matched best-first). Consumers: compact_results (rows render
    # only when an ordering exists) and the same-named-tie index guard.
    ordered: bool = False
    # Columns this result's own op attached (render-budget priority).
    op_cols: list = []


# THE one broad-result envelope law: ANY row-list result (find / score_by /
# filter_items / compare) longer than _BROAD_DUMP_SOFT_ROWS renders as
# census + the first _BROAD_HEAD_ROWS rows IN THE RESULT'S ORDER (a ranked
# head IS the ranking's top), with the full set live in the rN handle
# (slice/chain it). Response SHAPE only — never a candidate cap: the raw
# rows all stay in program state, and the commit reads the RAW results
# (reasoner_loop), so no row is ever dropped from the set. The head is bounded
# (tag-heavy OSM rows measure ~0.6KB each rendered) where the uncapped dump
# measured 113-148KB.
#
# The head IS the vision budget: a set crop badges the first LOOK_SET_ROWS
# rows of the handle, so a shorter head is a head whose tail the model never
# reads and vision is shown anyway. It read 12 against a 24-row crop, and the
# 12 it hid were worth 9.9 points -- containment@12 69.2% vs containment@24
# 79.1% over the same ordering (containment_probe, 325-episode facing
# family). Measured cost of the wider head on a `find house near <road>.`
# return over the calcity bake: +1,418 chars on the ordered table.
def join_hints(*parts) -> str:
    """Hints joined as SENTENCES, one per line -- the ONE hint shape.

    Several ops contribute facts to one envelope and they were glued with a
    literal " ALSO: " into a single run-on: 1,413 results over the audited
    corpus served two independent facts as one sentence, and a reader has to
    unpick which clause belongs to which. A line each is the whole fix.
    Empty parts drop out, so a caller never has to test before appending.
    """
    return "\n".join(str(p) for p in parts if p)


# Above this many tied ways under ONE name, a find's initial reply is the
# SHAPE (how many ways, how long) and not a table. Below it the table is
# cheap and a reader can hold it, so it renders: the point is not to hide
# the set, it is that an arbitrary prefix of 64 tied rows costs more than
# the sentence saying what the street IS, and reads as a ranking besides.
_NAME_SET_SHAPE_ROWS = 8

_BROAD_DUMP_SOFT_ROWS = 48
_BROAD_HEAD_ROWS = LOOK_SET_ROWS


# Above this many same-named row-ref instances, a compare's DEFAULT grid swaps
# that name's wide per-instance column block for a census + three cells per
# candidate reduced from the EXISTING d_<instance> columns (near_/by_/spread_).
# Render shape only — the raw rows and every per-instance column stay live on
# the rN (chain score_by/filter_items for full rows), mirroring the
# _BROAD_DUMP_SOFT_ROWS census idiom. Never a cap: no instance is dropped from
# the set, and the guard net (_FANOUT_TOKEN_GUARD) is unchanged.
_ANCHOR_FANOUT_CENSUS_INSTANCES = 6


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
    # A tied one-name set: the initial render is the SHAPE (len + the note
    # naming the street and its size), never rows or fids. The handle is a
    # normal ordered find -- slice it, page it, explore it.
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
    ``terminal_warn`` fires the out-loud attach-not-a-sort warning when compare
    was the program's LAST op; ``notes`` carries the loud empty-set /
    no-flanker notes the v2 aggregates emit; ``columns`` is the VERBATIM
    ordered list of every column key this compare attached (echoed as the
    result header's ``columns`` so the model reads real names instead of
    guessing d_<...>); ``ordered`` propagates from the compare's INPUT
    (compare preserves input order)."""

    refs: list = []
    terminal_warn: bool = False
    notes: list = []
    columns: list = []
    # Propagates from the compare's INPUT (compare preserves input order).
    ordered: bool = False


def _cut_stands_at(v, body: list, capped: list) -> str | None:
    """Where the ORDERING COLUMN stood at the head's cut, both sides of it.

    A head of 24 rows out of 3,414 says nothing about whether the answer is
    at rank 25. If the sort separated, row 25 sits far from row 24 and the
    head really is "the near ones". If it did not -- 24 houses on one street
    at 25.0..41.2 m and the next at 41.4 -- the head is an arbitrary slice
    of a plateau and rank 0 is a coin flip. The rows are all in hand, so the
    fact is free, and it is the one thing the served head cannot show: every
    row's own column value is rendered, and the value of the row just past
    the cut is not.

    Measured motive: the episodes that never got a crop onto the target
    queried into a match set of median 3,080 rows against 146 for the ones
    that did, and their correct row sat at rank 0 one time in eight against
    six in ten.

    A FACT and never a verdict -- it names what was ranked and three
    numbers, says nothing about what they mean, and drops nothing.

    IT REPORTS THE COMPOSITE ``score``, NOT A COLUMN, and that is the whole
    correctness of it. A multi-key sort is ordered by the SUM; no single
    column is monotone in the head, so quoting one says something false
    about the cut. Reading ``op_cols[0]`` was doubly wrong: ``score_by``
    stores ``sorted(emphasis)``, so on the legend's own
    ``near=<place>, color=<word>`` shape the alphabetically-first key is
    ``color_white`` and the row just past the cut carries its BEST possible
    value -- the note then read as "the cut discarded a row better than
    everything shown" on a head that had separated cleanly.

    Only a result ``score_by`` itself produced (``_CueSortedRows``) is an
    ordering. A ``find`` and a ``filter_items`` also stamp ``op_cols`` --
    it is a column-BUDGET priority list there, not a sort -- and firing on
    a query find put this note in the same hint as find's own "all equally
    good matches, so this order is arbitrary", one asserting a clean
    ordering and the other denying it.

    The next row comes from ``body``, the sequence the head was cut from,
    not from the raw result: window-covering rows are dropped before the
    cut, so ``v[len(capped)]`` is a row already on the page or one
    deliberately omitted.
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
    """Marker: this list's ORDER was produced by an ORDERING op (
    score_by) over cue-derived keys — indexing its head is a resolution, not
    an arbitrary pick, so the same-named-tie guard does not apply. Behaves as
    a plain list everywhere else."""

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

# Repeat the header row every N DATA rows on a long dump so a reader deep in the
# table never loses the column names (SOTA markdown-table best practice).
_GRID_HEADER_EVERY = 50

# RENDER-SHAPE bound on a tag-fat row's tags cell: the first N tags render,
# the rest fold to a factual '+K more tags' marker. A multilingual name_*
# flood (measured 6KB on one admin row) is render weight, never information
# the sort needs — the NODE keeps every tag and find/query still searches
# ALL tags (search_text is built from the full set).
_TAG_RENDER_HEAD = 12

# Tag keys the RENDERER drops. Measured over 2,700 stored episodes as the share
# naming the key anywhere in a say sentence OR in the model's own reasoning
# text -- a gate that overcounts reads (substring match) and so fails toward
# keeping: osm_id 0.5%, source_tag 0.7%, addr_postcode 0.3%, addr_city 0.4%,
# oneway 1.2%, ref_GB_uprn 0.0%. Together ~8% of every served byte.
#
# This is a RENDER cut, not a data cut. The node keeps every tag and
# ``search_text`` is still built from the full set, so a find on a postcode or
# an osm id still matches -- the model just is not made to read the id of every
# row to reach the ones it does use (addr_street 14.8%, building 84.0%,
# surface 22.9%, highway 4.8%, addr_housenumber 2.4%, all kept).
_TAG_RENDER_DROP = frozenset({
    "osm_id", "source_tag", "addr_postcode", "addr_city", "oneway",
    "ref_GB_uprn",
    # IMPORT BOOKKEEPING AND CONSTANTS. Measured over 16,110 reasoning
    # responses on the calcity_ends run: tiger_reviewed 2 mentions,
    # tiger_cfcc 21, tiger_source/tiger_tlid/tiger_separated 0,
    # worldkit_layer 43 (and its value restates the class the highway= /
    # building= tag already carries), calcity_object_id 21, source_lanes 0.
    # tiger_county is "Kern, CA" on every row of the block and
    # worldkit_height_m is 4.0 on 41,522 of 42,318 cells — a column that
    # never varies carries no information, so no value-level read of it is
    # even possible. ~950K chars of the corpus.
    # tiger_name_base / tiger_name_type STAY: street names are not readable
    # from nadir pixels and the model uses them to disambiguate.
    "tiger_reviewed", "tiger_cfcc", "tiger_source", "tiger_tlid",
    "tiger_separated", "tiger_county", "worldkit_layer",
    "worldkit_height_m", "calcity_object_id", "source_lanes",
})

# Columns the renderer drops, by the COMPREHENSION rule -- what the model
# actually DOES with each column at value level, not how often it names it
# (wf_e09c2b2e: 8 readers over 96
# stratified episodes of 2d0e85a:saydialect_1000, verifier-corrected). The
# mention instrument (``served_surface --emit use``) inverted the two most
# consequential calls -- values are quoted, names are not.
# RESTORED (the mention instrument was wrong):
#   footprint_area_m2  5/8 load-bearing; the only size-adjective sort
#                      column and the only degenerate-mask detector;
#                      decided commits; quoted by VALUE ("28,075 m2"),
#                      never by name (0.7% mention).
#   end1_xy            8/8 load-bearing consumed AS A PAIR with end0_xy
#                      (direction, extent, ends, corners; ~1.2% mention).
# CUT (raw fields all stay on the rows):
#   name_match         7 not-observed / 1 distracting; zero
#                      reads-to-pick-a-row in 8/8 packs; anti-correlated
#                      on 2 committed rows.
#   score              inert 1/1/4/2; never decided a commit as a value
#                      (one counterexample, 533); on colour blocks it is a
#                      rendered duplicate of an in-row value (literally
#                      color_<word> - 0.5 there).
# KEPT CUT (unchanged from 0df1803d, audit-confirmed):
#   aspect_ratio       inert (the first diet's cut).
#   axis_length_m      inert 1/2/5; in-row inconsistent with the
#                      endpoints (78.19 vs a 68.0 chord).
#   color_conf         occasional 1/4/3; trust gate in the looker
#                      stratum, hazard in miss strata -- the cut stands
#                      WITH the looker-stratum caveat named.
# center distance to <ref> / facing <ref> are DEFERRED, not
# kept: they attach in compare, and cutting an attach reaches the
# round-era remedy strings and the OP_VOCAB claims -- a quarantine
# decision, not a render cut.
# aspect_ratio is an OWNER DECISION, not just an audit cut: a min-rect
# scalar is a lossy proxy for shape, so that is a looking question -- the
# crop channel answers it. The field stays on nodes; the mission surface
# neither renders nor names it. Its axis-heading sibling is not cut here
# but DELETED at the producer: orientation is a relation to the feature a
# thing faces (the compare's 'facing <ref>'), and an undirected
# [0, 180) rectangle heading reads the same for a north-facing and a
# south-facing house on one street.
_GRID_COL_DROP = frozenset({
    "aspect_ratio", "axis_length_m", "color_conf",
    "name_match", "score",
})

# The ONE list of column names the model never sees, on ANY channel.
# THREE paths render or name columns -- the grid, the Row repr on the
# python surface, and the live-surface census echoed into error and
# dedup notes -- and each grew its own filter, so a cut enforced on one
# leaked out the others. Measured on calcity_conj_93_72fe9987, down the
# Row path alone: aspect_ratio 1,315 occurrences over 49 of 93 episodes,
# color_conf 1,158 over 43, score 431 over 45. The census echoes leaked
# the same names a second way, as bare "Columns computed on rows so far:
# [...]" lists -- no values, but naming a column is how the model learns
# to ask for it. A name goes in ONE place now.
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
    a tags dict -> compact ``k=v; k2=v2``; any stray ``|`` in a
    string is neutralised to ``/`` so a value
    can NEVER column-break the pipe table (defense in depth atop the
    color-mixture delimiter fix — the only field that ever carried ``|``)."""
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
    # STANCE-PAIR CONTIGUITY: side columns are presence-pairs (one member
    # per row), so first-seen union interleaves them with foreign columns
    # in data-dependent order (523 layouts measured across one fleet; the
    # pair split across three compare columns). Group each handle's four
    # side columns at the FIRST position any of them appears, in the one
    # fixed order left/right/ahead/behind.
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
    cells ``-``, floats 1 dp, the header repeated every ~_GRID_HEADER_EVERY
    data rows on long dumps. The noise-color ``?`` mask + hidden-field strip
    (:func:`_model_facing_row`) apply per row. Header-once (no repeated JSON key
    labels) realises ~-38% tok/row vs the ``{len, items}`` dicts it replaces.

    There is no ``idx`` column. It restated the row's own position in a table
    the reader walks top-to-bottom, and position is what the ``rN[i]`` /
    ``rN[a:b]`` reference grammar addresses -- that form needs no printed
    index and keeps working. Measured over 2,700 stored episodes: ``idx``
    named in 0.6% of episodes' prose, emitted in a say sentence 0 times, and
    ``rN[k]`` itself emitted 0 times."""
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


# Compare v2's per-ref relation-column prefixes (dynamic names — suffixed with
# the ref/set label) + the fixed absolute-view column names. compare_columns
# captures exactly these so the pick-step evidence table folds the model's OWN
# relation evidence in without re-listing every dynamic name. ``between_a`` /
# ``between_b`` / ``span_m`` / ``perp_offset_m`` are flat row fields the static
# evidence-field tuple already renders, so they are excluded here (no dup line).
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

# THE served column-name pattern (the Key sub-language): a literal
# alternation of the phrase-prefix families + the fixed column names, then
# a bounded suffix — canary-safe constructs only (no $-escape, no negated
# class). Owned by the executor because these ARE its columns; every
# schema/lexicon reader imports it from here. The literal is kept verbatim
# (alternation order is served bytes); the assert holds it to the prefix
# families above.
KEY_PATTERN = (
    "^(distance to |center distance to |bearing from |"
    "facing |along |left of |right of |ahead of |"
    "behind |contains |east of |west of |north of |south of |"
    "color_|score|name_match|"
    "footprint_area_m2|aspect_ratio|"
    "axis_length_m)[a-zA-Z0-9_]{0,40}$"
)
assert all(p in KEY_PATTERN for p in _COMPARE_COL_PREFIXES)
# LOUD token-overflow safety net for a multi-instance compare ref (NOT a
# per-instance cap). A ref that resolves to several instances fans out into one
# column set PER instance, at ANY count -- a fragmented road or a duplicate-named
# landmark keeps all its instances so the model can disambiguate. The ONLY bound
# is this net: if the fan-out's rendered size (instances x candidates, in cells)
# would be a genuinely crazy amount of tokens, it fails loud and STEERS (narrow /
# sort / index one / pass-as-items). It fires ~never once the model narrows on
# the result summary, and it catches the self-compare (refs = the candidate set).
# ~8000 cells ~= tens of k tokens rendered; genuine landmark fan-outs sit far below.
_FANOUT_TOKEN_GUARD = 8000

# The UNION's own bound, and it is a WALL-CLOCK net rather than a token one:
# a union renders ONE column set however many members it folds, so nothing
# here is about render size. Before a union member could be an rN it was
# bounded by what a model would hand-type; now `near=find("house")` folds
# every house on the tile, which is the members x items shape the union
# comment records as having "ran 885 s without returning". The bbox prune
# should hold it, but a wall-clock cliff discovered inside a 500-episode
# fleet run is the expensive place to discover it. A real cue's set -- the
# pools on a street, the crossings of two roads -- sits far below this.
_UNION_MAX_MEMBERS = 400


def _is_compare_column(k: str) -> bool:
    # one C call, not a generator over ten prefixes: this runs for every
    # key of every row of every compare result on every call
    return k.startswith(_COMPARE_COL_PREFIXES)


def compare_columns(results: dict) -> dict:
    """``{fid: {column: value}}`` over every ``compare`` result in a RAW (pre-
    ``compact_results``) results dict — the pick-step evidence table folds the
    model's OWN comparison in by fid (no harness backfill; a row the model never
    compared shows nothing). Captures every relation column (d_/brg_ — the
    brg_ prefix covers brg_gap_ — /dc_/along_/off_/vx_/vy_). Later compare
    ops win per column on overlap."""
    out: dict = {}
    for v in (results or {}).values():
        if not isinstance(v, _CompareRows):
            continue
        for row in v:
            if not (isinstance(row, dict) and row.get("fid")):
                continue
            cols = {k: val for k, val in row.items()
                    if _is_compare_column(k)
                    and isinstance(val, (int, float, bool, str)) and val != ""}
            if cols:
                out.setdefault(str(row["fid"]), {}).update(cols)
    return out


def _sanitize_token(lbl, fallback="ref") -> str:
    """The ONE column/cell-key sanitizer: a typed token (fid, ``rN`` base,
    name, ``as:`` label) -> ``[^a-z0-9]+``-collapsed lowercase base. Single-
    sourced so the compare column base (``d_road_1``), the aggregate labels
    (``vx_road_1``), and the fan-out census cell keys (``near_road``) all
    derive a base the SAME way — one naming law, no dialect drift."""
    base = re.sub(r"[^a-z0-9]+", "_", str(lbl).lower()).strip("_")
    return base or fallback


def _anchor_fanout_census(v):
    """RENDER-ONLY census of a wide same-named-instance ref fan-out.

    Groups a compare's row-ref headers by ``ref_name``; a name with
    ``>= _ANCHOR_FANOUT_CENSUS_INSTANCES`` instances has its per-instance
    columns dropped from the DEFAULT grid and replaced, per candidate, by
    three cells reduced from the EXISTING ``d_<instance>`` columns (no
    geometry recompute). The cell base is the group's shared ref-name run
    through the ONE ``_sanitize_token`` law (the same base the compare
    columns derive), echoed on the census header as ``cell_suffix``:
    ``near_<cell>`` (min d — distance to the nearest instance),
    ``by_<cell>`` (that instance's fid — the "which instance" signal),
    ``spread_<cell>`` (max−min of the candidate's instance distances — how
    much the instance choice matters for it). The census header names every
    instance fid + column verbatim (``instance_fids`` / ``instance_cols`` /
    ``cell_suffix``) so a chained score_by/filter_items reaches any of
    them. Raw rows are untouched (the rN keeps every column); returns
    ``(render_rows, censuses)`` — ``(v, [])`` when no group crosses the knee.
    """
    groups: dict[str, list[dict]] = {}
    for h in getattr(v, "refs", []) or []:
        if isinstance(h, dict) and h.get("ref_fid") and h.get("cols"):
            groups.setdefault(str(h.get("ref_name") or ""), []).append(h)
    censuses: list[dict] = []
    drop: set[str] = set()
    reduced: list[tuple[str, dict[str, str]]] = []  # (cell key, fid -> d col)
    used_keys: set[str] = set()
    for name, hs in groups.items():
        if not name or len(hs) < _ANCHOR_FANOUT_CENSUS_INSTANCES:
            continue
        d_by_fid: dict[str, str] = {}
        cols: list[str] = []
        for h in hs:
            cols.extend(h["cols"])
            dcol = next((c for c in h["cols"]
                         if c.startswith("distance to ")), None)
            if dcol:
                d_by_fid[str(h["ref_fid"])] = dcol
        if len(d_by_fid) < _ANCHOR_FANOUT_CENSUS_INSTANCES:
            continue
        # Same naming law as the compare columns: the cell key is the group's
        # shared ref-name sanitized through the ONE ``_sanitize_token`` base
        # (the per-instance columns it replaces were d_<fid>; this group cell
        # keys on the name they share).
        key = _sanitize_token(name)
        _k, _n = key, 2
        while _k in used_keys:  # distinct names, same sanitized key -> counter
            _k = f"{key}_{_n}"  # (mirrors the header label collision rule)
            _n += 1
        used_keys.add(_k)
        key = _k
        censuses.append({
            "ref_name": name,
            "n_instances": len(hs),
            "instance_fids": [str(h["ref_fid"]) for h in hs],
            "instance_cols": cols,
            "cell_suffix": key,
        })
        drop.update(cols)
        reduced.append((key, d_by_fid))
    if not censuses:
        return v, []
    render_rows: list = []
    for it in v:
        if not isinstance(it, dict):
            render_rows.append(it)
            continue
        row = {k: val for k, val in it.items() if k not in drop}
        for key, d_by_fid in reduced:
            pairs = []
            for fid, col in d_by_fid.items():
                val = it.get(col)
                if isinstance(val, (int, float)):
                    pairs.append((float(val), fid))
            if pairs:
                lo, lo_fid = min(pairs)
                hi = max(p[0] for p in pairs)
                # near_ IS the stored d_ value -- serve it verbatim (the one
                # _fmt_float render law); a second 1-dp round here collapsed a
                # 0.04 m gap to 0.0 beside the row's own 2-dp d_ column. spread_
                # is a NEW derived value, stamped ONCE at the d_ family's 2 dp.
                row[f"near_{key}"] = lo
                row[f"by_{key}"] = lo_fid
                row[f"spread_{key}"] = round(hi - lo, 2)
        render_rows.append(row)
    return render_rows, censuses


# THE one render budget: a result's rendered rows-x-columns grid stays under
# this many chars (~3k tokens). Priority when over: columns the program's own
# ops named first, then base identity columns, then first-seen extras — the
# tail elides with a count note. Shapes the RENDER only: every column lives
# on the full set in the rN (the F485 bound case: one uncapped compare
# rendered 96,775 chars for two already-known numbers).
_RENDER_CHAR_BUDGET = 12_000

# THE one-line table legend ("Beyond
# columns" item 1: the axis-sign question alone derailed ~4 episodes per
# pack in two packs; "one legend line would recover three columns").
# Served ONCE per response, under the FIRST envelope that renders rows.
# EVERY clause is code-derived and pinned by test_table_legend:
#   xy       build_surface._xy: a detection's measured world_xy, else a
#            point ON the row's own geometry (representative_point) --
#            NOT an endpoint midpoint, NOT always the mask centroid.
#            THE SAME point act.move flies (search_loop
#            ._world_xy_for_node_locked reads it back), so a row states one
#            position and the drone goes to it.
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
    base = 20 * (len(proj) + 2)  # idx+fid cells + pipes per line
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
                past the knee; ``anchor_census`` for wide same-named ref
                fan-outs).
      - row   : a single dict (a ``pick``) -> the bare ``_model_facing_row``.
      - scalar: a number/string -> passed through.

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

    THE LEGEND RIDES EVERY RESPONSE, once, under its first rows-carrying
    envelope. A once-per-mission countdown was tried and REGRESSED SR
    -0.042 (pysurface4, efb4a7a3): the axis-sign and xy-is-a-recorded-
    point inferences broke in exactly the episodes whose tables arrived
    legend-less. The per-response copy is load-bearing, not repetition.
    """
    _legend_out = [False]

    def _c(v):
        if isinstance(v, _CompareRows):
            # compare renders the SAME union-header grid every list result
            # gets, prefixed by the refs header; the joined table is the
            # point. ``fids`` keeps the result fid-bearing for the
            # terminal walk (rank order preserved) without duplicating the
            # rows.
            head: dict = {"refs": list(getattr(v, "refs", []) or []), "len": len(v)}
            # The header lists EVERY attached column key verbatim, so the model
            # reads real names (d_r0, vx_road, ...) instead of guessing them
            # (chains died on guessed absent columns).
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
                # Same-named-instance fan-out past the knee: census + reduced
                # near_/by_/spread_ cells in the DEFAULT grid (render shape
                # only — raw rows, columns echo and the rN keep every
                # per-instance column).
                _rrows, _censuses = _anchor_fanout_census(v)
                if _censuses:
                    head["anchor_census"] = _censuses
                if len(v) > _BROAD_DUMP_SOFT_ROWS:
                    _rrows = list(_rrows)[:_BROAD_HEAD_ROWS]
                # NO `fids` ARRAY BESIDE A GRID. It was the same fids, in
                # the same order, as the grid's own first column, served
                # twice — 1.30M chars over the corpus, dead by construction
                # rather than by measurement. A census-only head (no rows)
                # still carries it below, where it is the only fid source.
                _cols = _grid_columns(_rrows)
                _cols, _n_cut = _budget_columns(
                    _rrows, _cols, getattr(v, "columns", None))
                if _n_cut:
                    hints.append(f"{_n_cut} column(s) elided from this "
                                 "render to fit; the rN keeps them all")
                head["rows"] = _render_grid_rows(_rrows, _cols)
                if not _legend_out[0]:
                    head["legend"] = _TABLE_LEGEND
                    _legend_out[0] = True
            hints.extend(getattr(v, "notes", []) or [])
            if hints:
                head["hint"] = join_hints(*hints)
            return head
        if isinstance(v, list):
            if len(v) == 0:
                # An empty list is a VALID result (no item met the op's
                # condition). State the mechanical fact — the
                # op's TRUTHFUL condition when one was recorded (a query miss
                # names its threshold + tokens), never a claim about the
                # world. On a numeric cut the deterministic ``cut_stats``
                # payload (cut / nearest_excluded / admit_at / n_at_admit)
                # carries the threshold facts -- no widen-and-retry prose.
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
            # Union-header markdown grid (header-once, ~-38% tok/row) is the
            # SOLE model-facing serialization for a non-empty list result,
            # under the ONE broad-result law: past _BROAD_DUMP_SOFT_ROWS the
            # envelope is census + the first _BROAD_DUMP_SOFT_ROWS rows/fids
            # in the result's order (a ranked head IS the ranking's top), and
            # the hint states how many rows the rN holds and how to slice.
            # The full-set fid array is deliberately NOT served (measured
            # 113-148KB single messages); the commit path reads the RAW
            # results, so nothing is dropped from the candidate set.
            #
            # WINDOW-COVERING rows out of a SORT's rendered head (render
            # shape, never a filter): a mask that contains the whole window
            # holds d_=0 to every ref by containment, so such rows tie
            # rank-0 of every closest-first sort while never being
            # candidate referents (measured: admin boundaries owned ranks
            # 0-2 of ranked views in 3 episodes and burned 12 iteration-0
            # rounds). They are OMITTED from the rendered head of an
            # ordering-op (score_by) result and NAMED in one truthful note;
            # the full set (their rows included, covers_window fact riding
            # them) stays live in the rN — find/query/census are untouched.
            _ordered = bool(getattr(v, "cue_sorted", False)
                            or getattr(v, "ordered", False))
            if not _ordered:
                # NO real ordering exists -> census + len, NO row head: a
                # fid-lexicographic head under a result billing reads as a
                # candidate list (render shape only — the full set lives in
                # the rN; the commit path reads the RAW results).
                head = {"len": len(v)}
                census, named = _surface_census(v)
                head.update({"census": census, "named": named})
                _cut = getattr(v, "cut_stats", None) \
                    if isinstance(v, _FilterCutRows) else None
                if _cut:
                    head["cut_stats"] = _cut
                head["hint"] = join_hints(
                    head.get("hint"), *(getattr(v, "notes", None) or []))
                # The route to rows, stated where the dead end appears
                # (measured: 521 census heads over 97/101 episodes, with
                # repeat loops trying to guess the way in). Stated as the
                # FACT and in neither channel's spelling: this renders for
                # the sentence surface AND for the frozen round-era one, so
                # an op-JSON idiom here teaches a register the sentence
                # channel cannot emit.
                _rows_route = (
                    "no ordering exists on this result, so it renders "
                    "census + len and no fids: ordering it by an attached "
                    "column renders the ordered rows of THIS handle, and a "
                    "name or color match returns matched rows directly.")
                head["hint"] = join_hints(head.get("hint"), _rows_route)
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
            # A NAME'S SET RENDERS AS ITS SHAPE. Every row ties, so a head
            # is an arbitrary prefix served as though it were a ranking --
            # and the fids alone cost more than the sentence that says what
            # the street IS. The rN keeps every row for a slice or explore.
            if getattr(v, "name_set", False):
                capped = []
            else:
                capped = list(_body[:_BROAD_HEAD_ROWS]
                              if len(v) > _BROAD_DUMP_SOFT_ROWS else _body)
            # A truncated head must say so: an unstated prefix reads as the
            # candidate set (measured: models hand-copy it into select).
            # WHETHER the order means anything is the producing op's fact and
            # is stated there -- `find` says either "best-first from a down
            # to b" or "all equally well, so this order is arbitrary", once,
            # from the scores it actually computed. This render used to
            # re-derive the same fact from a uniform name_match column and
            # append it, so a result could carry "(best-first, match 0.8)"
            # and "all 3414 rows are equally good matches, so this head's
            # order is arbitrary" in ONE hint.
            _head_note = None
            if getattr(v, "name_set", False):
                pass
            elif len(capped) < len(_body):
                # CHANNEL-SPECIFIC, and it has to be: this hint once
                # advertised rN[a:b] to a channel whose language cannot spell
                # a numeral, and the sentence lanes measured 36 attempts and
                # 36 errors at it. The caller passes the form ITS channel can
                # say (``widen_form``); a caller that passes none gets the
                # fact without a form, never another dead end.
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
            _cols = _grid_columns(capped)
            head = {"len": len(v)}
            if getattr(v, "name_set", False):
                # The shape as DATA. It speaks Python and holds the handle:
                # a sentence naming explore() and rN[a:b] is workflow legend,
                # and the ways' own total length is the one fact len cannot
                # carry.
                head["name"] = str(_body[0].get("name") or "")
                _set_m = sum(float(r.get("axis_length_m") or 0.0)
                             for r in _body)
                if _set_m:
                    head["length_m"] = round(_set_m)
            if not getattr(v, "name_set", False):
                head["fids"] = [str(it["fid"]) for it in capped
                                if isinstance(it, dict) and it.get("fid")]
            _cols, _n_cut = _budget_columns(
                capped, _cols, getattr(v, "op_cols", None))
            if _n_cut:
                _head_note = join_hints(
                    _head_note,
                    f"{_n_cut} column(s) elided from this render to fit; "
                    "the rN keeps them all")
            if not getattr(v, "name_set", False):
                head["rows"] = _render_grid_rows(capped, _cols)
            # The legend reads a TABLE, and it is served once per response:
            # attaching it to a head that renders no rows spends the slot,
            # so the next result's real table loses its column key.
            if not _legend_out[0] and "rows" in head:
                head["legend"] = _TABLE_LEGEND
                _legend_out[0] = True
            head["hint"] = join_hints(head.get("hint"), _head_note,
                                      _cover_note)
            if not head["hint"]:
                head.pop("hint")
            if _cut:
                # Iter-4 cut-stats: the one-row deterministic cut line on a
                # NON-EMPTY numeric cut (how much was cut + the nearest row
                # the threshold excluded + the admit-at value). Info-only.
                head["cut_stats"] = _cut
            head["hint"] = join_hints(
                head.get("hint"), *(getattr(v, "notes", None) or []))
            if not head["hint"]:
                head.pop("hint")
            return head
        if isinstance(v, dict):
            # A single surface row (pick) carries search_text too.
            return _model_facing_row(v)
        return v

    # Serve any identical census ONCE per response: a program whose ops each
    # summarize the same broad set would otherwise render the same census
    # block N times in one message (measured: 3 identical blocks in one
    # 32.7 KB message).
    #
    # ANSWER FIRST, so the ref that survives the reduction below is the one
    # that keeps its own census rather than a pointer to a dropped ref's.
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
    # ONE SENTENCE, ONE TABLE. `find house near <road>.` is one question and
    # it compiled to find -> compare -> score_by; all three results were
    # served, only the last was ordered, and it came last. 749 of the 3,120
    # successful geometry returns on calcity_full_replyrepair served three
    # tables and 57 served four -- the same rows re-rendered twice as the
    # working before the answer.
    #
    # The working's FACTS survive: each dropped op's hint rides the answer,
    # attributed to the ref that made it, which is also what keeps an order
    # claim honest ("r0: ... this order is arbitrary" is about r0, whose
    # order is still live in the handle, and not about the answer's). Every
    # rN stays chainable -- nothing leaves program state, and the commit
    # path reads the RAW results.
    answer = out[answer_ref]
    if isinstance(answer, dict):
        answer["hint"] = join_hints(
            answer.get("hint"),
            *(f"{k}: {e['hint']}" for k, e in out.items()
              if k != answer_ref and isinstance(e, dict) and e.get("hint")))
        if not answer["hint"]:
            answer.pop("hint")
    return {answer_ref: answer}


def _unwrap_listish(v):
    """A list result echoed back in the legacy ``{len, items}`` shape -> its
    items. Parity-safe: a raw ``{len, items}`` dict is never a valid list arg
    (as_list/as_points hard-error on any dict), so this only turns a former
    error into the obvious value. Post the markdown-grid cutover a NON-EMPTY
    list result serializes as ``{len, fids, rows}`` (its rows a grid
    STRING, not reconstructable row dicts) — such a pasted dict is NOT unwrapped
    here and instead fails loud, directing the model to chain the ``rN`` ref
    (which resolves against the RAW rows). Only an EMPTY result still carries
    ``items: []``."""
    if isinstance(v, dict) and "len" in v and isinstance(v.get("items"), list):
        return v["items"]
    return v


def _grouped_keys(present) -> str:
    """Key list -> a readable summary: singleton keys verbatim, per-instance
    families collapsed ('distance to building_* (100 instances)') — a fan-out
    compare against a multi-row ref attaches hundreds of per-fid columns, and
    an error that DUMPS them all buries the one key the model needs (measured:
    a 400-name dump the model had to fish 'distance to road_54' out of). Each
    key is assigned to exactly ONE family — its longest matching prefix (the
    same law the auto-attach prefix extraction states) — so no key prints
    twice ('facing ' must win over 'bearing from ', which a
    startswith re-scan double-listed).

    The family is the COLUMN PREFIX, read from the one vocabulary
    (:data:`_COMPARE_COL_PREFIXES` + ``color_``). This matched on
    ``^([a-z]+_)([a-z]+_)?`` when the columns were spelled ``d_building_1``;
    the rename to ``distance to building_1`` left no underscore where the
    regex needed one, so every key fell through to its own family and the
    400-name dump came straight back.
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
    """The producing idiom for an ABSENT column, keyed to the MISSING key's
    family — the ONE family->provenance remedy source, rendered into every
    absent-key guard alongside the present-keys census (a one-size d_-family
    default misdirected 7 of 8 measured absent-key errors). Returns a
    sentence ending in a space, or '' for a key with no known family (the
    census still shows what exists).

    TRUTHFUL AFFORDANCES ONLY: every example this channel serves must
    EXECUTE if followed. A color example is echoed only after resolving
    onto the closed palette (an off-palette echo like color_lt_white is a
    remedy the palette gate then rejects — measured, 10624); a token that
    is not column-shaped (prose, whitespace) gets no interpolated example;
    and the attach sentence names the ONE place naming attaches (a score_by
    emphasis / filter_items key) — naming a key anywhere else (e.g. a
    select basis) does NOT attach it."""
    k = str(key)
    if not re.fullmatch(r"[A-Za-z][\w. ]{0,60}", k):
        return ""  # not a column-shaped token: no worked example applies
    if k == "name_match":
        return ("name_match comes from find{query:'<name>'} (the fuzzy name "
                "search annotates it on every row). ")
    if k == "score":
        return "score is attached by score_by (the composite sort). "
    if k.startswith(("ahead of ", "behind ", "left of ", "right of ")):
        return ("this column comes from compare frame:{toward:<f>} (the "
                "viewer look) or the standing viewpoint (act.stage); on a "
                "line, frame {along:<line>, [toward]}. ")
    for pre in ("facing ", "bearing from ",
                "center distance to ", "distance to ", "along ",
                "color_"):
        if not k.startswith(pre):
            continue
        param = k[len(pre):]
        if pre == "color_":
            # Echo an example ONLY for a key the palette gate would accept:
            # resolve the word (shade prefixes lt/dk are display-only, so
            # strip one before resolving); off-palette words get the palette
            # list, never an example that errors when followed.
            _w = re.sub(r"^(lt|dk)[_ ]", "", param.strip().lower())
            _canon = G.normalize_color(_w)
            if _canon in G.COLOR_ENUM:
                _ck = f"color_{_canon}"
                return (f"a color column attaches when NAMED in a score_by "
                        f"emphasis -- e.g. emphasis "
                        f"[{{key:'{_ck}', w:1}}] -- given rows carry a "
                        f"perceived color (color orders, never filters)"
                        + (f" (shade prefixes like '{param}' are "
                           f"display-only; the column is {_ck})"
                           if _canon != param.strip().lower() else "")
                        + ". ")
            return (f"{param!r} is not on the closed color palette "
                    f"({', '.join(G.COLOR_ENUM)}); color columns are "
                    f"color_<palette word>, named in a score_by emphasis "
                    f"(color orders, never filters). ")
        if re.fullmatch(r"[0-9][0-9_.]*", param):
            # NOT "reads like a coordinate, not a fid": the model may never
            # have written this label. A point named in one call mints it, and
            # a later ordering can name that column over rows the first call
            # never related to it. Accusing the model of a type error it did
            # not make, then prescribing a JSON call, was wrong twice.
            return (f"{param!r} is not a fid: a number pair names a POINT, "
                    f"and a point carries columns only under the label it is "
                    f"given when it is related to rows -- the column is then "
                    f"{pre}<label>. ")
        # The say path reaches here through `in <place>` (the one filter it
        # spells), so the remedy names the FACT -- the column exists once
        # the ref is related to these rows -- and not the op that would
        # attach it. Naming score_by/filter_items taught a JSON register to
        # a channel whose only move is a sentence, and 'along' / the side
        # columns are productions the strip priced out.
        return (f"a '{pre}<ref>' column exists once <ref> has been related "
                "to these rows, and <ref> must be a fid on this surface. ")
    return ""


def _polyline_arc_midpoint(verts) -> list:
    """The arc-length midpoint of a world-meter polyline — the CENTER a
    constructed-axis compare ref reports dc_/brg_ from (a row ref uses its
    baked centroid ``xy``)."""
    arcs = G._arc_lengths(verts)
    half = arcs[-1] / 2.0
    for i in range(len(verts) - 1):
        if arcs[i + 1] >= half:
            seg = arcs[i + 1] - arcs[i]
            t = (half - arcs[i]) / seg if seg > 0 else 0.0
            return [float(verts[i][0]) + t * (float(verts[i + 1][0]) - float(verts[i][0])),
                    float(verts[i][1]) + t * (float(verts[i + 1][1]) - float(verts[i][1]))]
    return [float(verts[-1][0]), float(verts[-1][1])]


def _side_pair(row_put, lab, lateral, longitudinal=None, ci=None,
               distance=False, sides=True):
    """Presence-pair side columns AS PHRASES: the column present names the
    side ("left of X": 7.0) and the other side is simply absent — a signed
    vx/off lies about half its values, and a phrase key is the model's own
    training-data shape (no invented underscore grammar). 0.0 (on the
    axis) lands on the right/ahead member. ``lateral`` + = right;
    ``longitudinal`` + = ahead; ``distance`` also emits "distance to X" =
    |lateral| (the either-side proximity sort, replacing the abs idiom).

    ``sides=False`` keeps the distance column and drops the LATERAL pair, for
    a frame the ENGINE inferred rather than one the model declared.

    This branch had already measured the cost of getting that side wrong, in
    the note that used to describe these columns: on a 694-episode ordinal
    stratum, render-window surfacing is 62.0% for "find building near
    <road>.", 83.9% with the CORRECT side cut and 46.0% with a coin-flip one.
    A wrong cut is worse than no cut, and the inference is a coin flip: ``lateral_convention`` scored every axis this
    engine can compute at chance -- road_travel_intrinsic 0.461,
    road_local_tangent 0.473, anchor_own_axis 0.479, north_up 0.442, against
    sham_random_axis 0.519 -- on 926 resolved mentions, with cluster
    intervals, block-flip significance and an anchor shuffle.

    Fixing the convention was priced before being built and does not pay:
    ``relation_replay``'s street-viewer arm (the conventions that ARE real,
    0.642 / 0.752) re-orders the same items at top-1 0.095, against 0.110 for
    the map-fixed axis and 0.217 for plain proximity, n=825. A side call is
    binary and halves a pool; an ORDERING must rank INSIDE the half it keeps,
    and the side word says nothing about that. Worse, a spatial relation is
    the one thing the surface lets the model FILTER on, so a coin-flip side
    call drops the target outright.

    A stance the MODEL declared is a different claim and keeps its laterals:
    ``frame_type == "viewer"`` here, and ``_attach_standing_columns`` (which
    writes the staged-viewpoint pair directly, not through this function).
    Neither was measured by the readers above, so neither is cut on their
    evidence. The engine stops inventing an axis; it reports one when
    somebody stood somewhere.
    """
    if lateral is not None:
        v = float(lateral)
        if sides:
            if v >= 0:
                row_put(f"right of {lab}", round(v, 2), ci)
            else:
                row_put(f"left of {lab}", round(-v, 2), ci)
        if distance:
            row_put(f"distance to {lab}", round(abs(v), 2), ci)
    if longitudinal is not None:
        v = float(longitudinal)
        if v >= 0:
            row_put(f"ahead of {lab}", round(v, 2), ci)
        else:
            row_put(f"behind {lab}", round(-v, 2), ci)



def _geo_pt(p, nd=2):
    """A world point coerced to plain rounded floats (witness pairs and
    frame vectors can arrive as numpy scalars, which don't JSON-serialize)."""
    return [round(float(p[0]), nd), round(float(p[1]), nd)]


# The staged-viewpoint side columns: the four phrase prefixes whose
# parameter may name a staged handle (act.stage's vN). Distance/bearing
# prefixes are NOT here -- a staged handle resolves as a plain point ref
# on those families (the compare path), so nothing is re-implemented.
_STANDING_SIDE_PREFIXES = ("left of ", "right of ", "ahead of ", "behind ")

# EVERY column key naming a key is a computation REQUEST: an op that names
# one of these gets the column computed for it (``_auto_attach``, inside
# run_program). Module scope so the PYTHON surface can read the same tuple
# -- a Row asked for one of these keys and unattached can name the call
# that writes it instead of raising KeyError over a column list that does
# not contain what was asked for.
# ``brg_gap_`` MUST precede ``brg_`` (longest-prefix-first extraction).
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
    the side; 0.0 on-axis lands right/ahead (the _side_pair convention).
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
        # FIXED pair order (left/right, then ahead/behind): the grid's
        # column order derives from first-seen insertion, and a varying
        # order measured as 8 distinct layouts across one run.
        rec[right_k if lat >= 0 else left_k] = round(abs(lat), 2)
        rec[ahead_k if lon >= 0 else behind_k] = round(abs(lon), 2)
        if with_distance:
            # straight-line meters from the STAND POINT to the row's
            # recorded position (centroid) -- the stance is a viewer, not
            # a mask, so there is no footprint-gap variant to confuse.
            rec[dist_k] = round((dx * dx + dy * dy) ** 0.5, 2)
        out.append(rec)
    return out


def run_program(cands, anchors, program, seed_results=None,
                geometry_sink=None, staged=None, standing=None,
                standing_anchor=None):
    """Execute a typed op-program over an explicit surface. The SHARED core.

    ``staged`` (the act.stage viewpoints): a read-only ``{handle: {"handle",
    "from_xy": [x, y], "up_unit": [ux, uy]}}`` map. Every staged handle
    resolves as a named POINT ref anywhere a fid is accepted (compare refs,
    auto-attach keys -- 'distance to v1' / 'bearing from v1'), and the
    side-phrase keys 'left of/right of/ahead of/behind <handle>' attach that
    viewpoint's presence-pair columns. ``standing`` names the handle whose
    side columns ride every find result automatically.
    ``standing_anchor`` (``{label, ref}``; ref = fid | [fid,...] | [x, y])
    attaches 'distance to <label>' -- and 'along <label>' when the ref
    charts -- to every find result, the same law as ``standing``. Default
    None = the offline teacher's byte-identical behavior.

    ``cands`` / ``anchors`` are lists of ``{fid, xy, centerline, class, color, ...}``
    dicts (built identically offline and live). Returns ``{results}``; wrap
    with :func:`compact_results` for the model-facing payload.

    ``geometry_sink`` (dashboard-side audit plumbing, never model-facing —
    the ``raw_sink`` pattern): when given, receives under ``sink["ops"]``
    one entry per executed ``compare`` with the resolved geometry the op
    computed and does not return (chart polyline + end0 + travel, frame
    basis/standpoint, linear-ref polylines, ref centroids, min-gap witness
    pairs). Default None = zero behavioral change; the returned results
    and every receipt stay byte-identical either way.

    ``seed_results`` (the rN cross-call amnesia is the round's top friction,
    8/20 episodes): a read-only
    {ref: raw_result} map of PRIOR calls' results this round. Resolution
    falls back to it when a ref is not defined in THIS program, so a
    later analyze_scene call can reference an earlier call's ``r2``;
    a ref redefined here shadows the seed. Seeded refs are resolvable
    but never re-rendered (the returned ``results`` holds only THIS
    program's ops). Default None = the offline teacher's byte-identical
    single-call behavior.
    """
    seed = seed_results or {}
    by_fid = {c["fid"]: c for c in cands}
    by_fid.update({a["fid"]: a for a in anchors})
    staged = staged or {}
    _standing_frame = staged.get(standing) if standing else None
    results: dict = {}
    # Resolver hardening (measured: 57% of hard-errors are rN-ref/shape frictions
    # on chains Qwen formed correctly). Each tolerant branch below fires only
    # where a bare resolve would otherwise raise, so a currently-succeeding
    # program is unaffected.
    # The op currently dispatching. The resolver closures append THIS op's signature
    # to their error message via ``_sig_suffix`` (the closures don't otherwise know
    # the op name). A dict holder, not the loop-local ``name``, so a resolver called
    # outside the op loop degrades to no-signature rather than a call-time NameError
    # (see feedback_deadcode_callsite_nameerror).
    _cur_op = {"op": None}

    def _sig_suffix() -> str:
        sig = _OP_SIGNATURE.get(_cur_op["op"])
        return f"; signature: {sig}" if sig else ""

    def _avail(kind: str) -> str:
        """The run's LIVE available-fid census (bounded) for an error message, keyed
        by the arg-KIND that failed. Surfacing the run's OWN state — not a memorised
        answer — so this is not teach-to-the-test."""
        def _cap(fids):
            return f"{fids[:12]}" + (f" (+{len(fids) - 12} more)" if len(fids) > 12 else "")
        if kind == "list":
            srcs = ", ".join(str(k) for k in results) or (
                "(no result handles yet -- run a find first)")
            return f"list sources: {srcs}; fids present: {_cap(sorted(by_fid))}"
        return f"available fids: {_cap(sorted(by_fid))}"

    def _walk_path(base, path, token):
        """Walk a dotted / indexed access path into a prior result, e.g.
        ``members[0]`` on a prior list result. Supports ``.field`` on
        dicts, ``[i]`` indexing, and ``[a:b]`` / ``[:n]`` / ``[n:]`` WINDOWING on
        lists (the ``r0[:5]`` top-N slice / ``r8[0]`` one-element pulls the model
        reaches for). Empty path returns ``base`` unchanged."""
        cur = base
        # tokenize "rows[0:5].members" -> ['rows', '[0:5]', 'members']; a bracket
        # carries an int index [i] or a Python window [a:b]/[:n]/[n:].
        for part in re.findall(r"[^.\[\]]+|\[[^\]]*\]", path):
            if part.startswith("["):
                inner = part[1:-1]
                if ":" in inner:
                    # agentic WINDOWING of a prior list (r0[:5] top-5) -> the
                    # sub-list; a slice on a single row means "that row, windowed".
                    if not isinstance(cur, list):
                        cur = [cur]
                    src_n = len(cur)
                    try:
                        # TypeError catches a >3-part window ([1:2:3:4] -> slice()
                        # with too many args); ValueError catches non-int bounds.
                        _win = cur[slice(*[int(b) if b.strip() else None
                                           for b in inner.split(":")])]
                    except (ValueError, TypeError):
                        raise ProgramError(
                            f"bad slice {part} in ref {token!r}: "
                            f"use [start:stop] with integer bounds")
                    if type(cur) is not list:
                        # A window KEEPS its source's class and order-describing
                        # attrs: list.__getitem__(slice) returns a bare list,
                        # and the dropped cue_sorted/ordered bit downgraded
                        # every windowed handle to the census envelope
                        # (`more of` served rows 0 times in 451 flown calls,
                        # and the round era's r0[:5] lost its order the same
                        # way). Whole-set descriptors (notes, cut_stats, refs,
                        # columns) stay behind on purpose -- they describe the
                        # full result, not this window.
                        _w = type(cur)(_win)
                        for _a in ("ordered", "op_cols"):
                            if _a in vars(cur):
                                setattr(_w, _a, getattr(cur, _a))
                        _win = _w
                    cur = _win
                    if not cur and src_n:
                        # a window that selected 0 of a NON-empty pool -- a start
                        # index past the end or start >= stop (the [5:]/[:5]
                        # direction slip). Loud, not a silent pool-drop: the
                        # integer-index branch already fails loud on the analogous
                        # single-row [i] miss. (An already-empty pool windows to
                        # empty silently -- src_n == 0 -- which is correct.)
                        # The FACT, in neither channel's spelling: this text
                        # renders for the sentence surface AND the frozen
                        # round-era one, and the sentence channel cannot emit
                        # a slice at all — the window is the compiler's, never
                        # the sentence's.
                        raise ProgramError(
                            f"window {part} in ref {token!r} selected 0 of "
                            f"{src_n} rows: this ref holds {src_n} row(s) and "
                            f"the window begins at or past the end of them "
                            f"(or begins at or after where it stops).")
                    continue
                try:
                    idx = int(inner)
                except ValueError:
                    # '[road_88]' -- the model indexed by a FID, not a position.
                    # A clean loud error (not an int() crash); narrow by fid with
                    # filter_items instead.
                    raise ProgramError(
                        f"bad index {part} in ref {token!r}: an index is a POSITION "
                        f"([0], [-1]) or a window ([:n]); to pick a fid, filter_items "
                        f"on it. Got {inner!r}.")
                if not isinstance(cur, list):
                    # '[0]'/'[-1]' on a SINGLE row (a one-row result the
                    # model models as a list) -> the row itself; any other
                    # index into a non-list is a genuine miss.
                    if idx in (0, -1):
                        continue
                    raise ProgramError(
                        f"bad index {part} in ref {token!r}: not a list of that size")
                if not (-len(cur) <= idx < len(cur)):
                    raise ProgramError(
                        f"bad index {part} in ref {token!r}: not a list of that size")
                # SAME-NAMED-TIE guard: a name is not a place. Indexing is a
                # resolution ONLY when the list's ORDER is real (score_by's
                # cue_sorted, or an ordered query-matched find — whose own
                # tie note already stated any within-tie arbitrariness) or
                # the tie is otherwise broken (a unique composite ``score`` /
                # ``name_match``). On retrieval/surface order, a position that
                # lands among same-named tied rows picks arbitrarily — which
                # instance the cue means is decided by the cue's OTHER refs
                # (order the set with score_by, then [0] IS the head).
                _picked = cur[idx]
                if isinstance(_picked, dict) and not (
                        getattr(cur, "cue_sorted", False)
                        or getattr(cur, "ordered", False)):
                    _pname = _picked.get("name")
                    if _pname not in (None, ""):
                        _pmatch = _picked.get("name_match")
                        _pscore = _picked.get("score")
                        _tied = sum(
                            1 for _r in cur
                            if isinstance(_r, dict)
                            and _r.get("name") == _pname
                            and _r.get("name_match") == _pmatch
                            and _r.get("score") == _pscore
                        )
                        if _tied >= 2:
                            raise ProgramError(
                                f"index {part} in ref {token!r} lands on one of "
                                f"{_tied} rows all named {_pname!r} with the SAME "
                                f"name_match -- a name is not a place, so the "
                                f"position picks a tied same-named instance "
                                f"arbitrarily. Resolve WHICH instance the cue "
                                f"means by its OTHER refs (compare against them + "
                                f"score_by, or filter_items after a sort), "
                                f"then index the ordered result.{_sig_suffix()}")
                cur = cur[idx]
            else:
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                elif isinstance(cur, list) and part == "items":
                    # '.items' on the RAW list IS the list itself -- the model
                    # echoes the {len,items} VIEW compact_results renders it.
                    # Parity-safe: '.items' on a list was always a hard error.
                    pass
                elif isinstance(cur, list) and part == "len":
                    cur = len(cur)
                else:
                    raise ProgramError(
                        f"bad field '.{part}' in ref {token!r}: "
                        f"{('keys ' + str(sorted(cur))) if isinstance(cur, dict) else 'not a dict'}")
        return cur

    def _resolve_ref(token):
        """Resolve a reference to a prior op result. The canonical form is the
        op's ``ref`` value BARE (an op with ``ref:"r0"`` reads back as ``"r0"``
        — see the tool spec). Tolerant of the malformed forms the model emits
        when the binding syntax reads ambiguously: a leading ``$`` (the old
        sigil — stripped and forwarded silently, one meaning), ``ref:r0`` /
        ``ref_r0`` (the literal word ``ref`` prepended to the bound name) and a
        bare ``ref`` / ``prev`` / ``last`` (the immediately-preceding result).
        Also supports a dotted / indexed access PATH onto the result —
        ``r2.members[0]`` walks into a prior op's list result to pull a
        chosen element. The spec teaches
        ``r0``; this just keeps a minor syntactic slip from bricking the whole
        program (a live bench measured 86% of programs hard-erroring on exactly
        this), and the error below names the fix when it truly can't bind.
        """
        # de-sigil: a leading '$' is a tolerated slip for the bare form.
        key = token[1:] if token.startswith("$") else token
        # split off any access path after the leading ref name
        m = re.match(r"^([^.\[\]]+)(.*)$", key)
        base_key, path = (m.group(1), m.group(2)) if m else (key, "")
        if base_key in results:
            return _walk_path(results[base_key], path, token)
        for pre in ("ref:", "ref_", "ref-", "ref."):
            if base_key.startswith(pre) and base_key[len(pre):] in results:
                return _walk_path(results[base_key[len(pre):]], path, token)
        # Surface-list ABSORPTION (input tolerance only — no served surface
        # teaches these words): a bare 'cands'/'anchors' (or a sigiled
        # '$cands' slip, already de-sigiled above) resolves to the injected
        # provenance list rather than erroring (measured when taught:
        # 125/227 episodes hit "unknown ref cands").
        if base_key == "cands":
            return _walk_path(cands, path, token)
        if base_key == "anchors":
            return _walk_path(anchors, path, token)
        # Order-free execution (the topological sort in run_program) resolves
        # every in-program handle BEFORE it is read, so a forward reference is
        # legal. Reaching here means base_key's producing op did not run -- it
        # errored, or a reference cycle the sort flagged -- so its result is
        # genuinely unavailable. (An in-program handle always means the op this
        # program defines, never a same-named prior-call seed.)
        if base_key in _program_ref_names:
            if base_key in seed:
                raise ProgramError(
                    f"{token!r} reads an op THIS program defines, and that "
                    f"name hides the {base_key!r} an earlier call left live "
                    f"(an in-program ref always means the op beside it). "
                    f"This program's {base_key!r} produced no result: it "
                    f"errored, or a reference cycle blocked it. Give this "
                    f"program's op a ref name no earlier call used, and "
                    f"{base_key!r} reads the earlier handle again.")
            raise ProgramError(
                f"{token!r} names an op in this program whose result is not "
                f"available: its producing op failed, or a reference cycle "
                f"prevented resolution.")
        # Cross-call persistence (S4): a ref from a PRIOR call this round
        # resolves from the seed; a ref redefined in THIS program shadowed
        # it above.
        if base_key in seed:
            return _walk_path(seed[base_key], path, token)
        for pre in ("ref:", "ref_", "ref-", "ref."):
            if base_key.startswith(pre) and base_key[len(pre):] in seed:
                return _walk_path(seed[base_key[len(pre):]], path, token)
        if base_key.lower() in ("ref", "prev", "last", "previous") and results:
            return _walk_path(list(results.values())[-1], path, token)
        _known = sorted(set(results) | set(seed))
        raise ProgramError(
            f"unknown ref {token!r}: an op defined with ref 'r0' is read back "
            f"as r0. Refs persist across your analyze calls (redefining one "
            f"replaces it). "
            f"valid refs so far: {_known or '(none)'}")

    def _node_xy(v):
        """A point ``[x, y]`` from a node dict / fid string / ``[x, y]`` literal.
        Never ``float()`` a raw fid — resolve it through ``by_fid`` first."""
        if isinstance(v, dict):
            xy = v.get("xy")
            if isinstance(xy, (list, tuple)) and len(xy) == 2:
                return list(xy)
            if v.get("fid") in by_fid:
                return by_fid[v["fid"]]["xy"]
        if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(z, (int, float)) for z in v):
            return list(v)
        if isinstance(v, str) and v in by_fid:
            return by_fid[v]["xy"]
        if isinstance(v, str) and v in staged:
            # a staged viewpoint handle IS a place: its stand point.
            return list(staged[v]["from_xy"])
        return None

    _RESERVED_REFS = ("cands", "anchors", "last", "prev", "previous", "ref")

    def _ref_base(token):
        """The bare ref NAME of a token: strip a tolerated leading ``$``, then
        the access path (``r2.members[0]`` -> ``r2``). ``cands``/``anchors``
        and the ``last``/``prev`` family stay whole."""
        t = token[1:] if token.startswith("$") else token
        m = re.match(r"^([^.\[\]]+)", t)
        return m.group(1) if m else t

    def _is_ref_token(arg):
        """A BARE token is a handle iff its base exactly matches a known result
        name (this program's ``results``, a prior-call ``seed``, or a name this
        program defines — so the top-to-bottom guard in ``_resolve_ref`` still
        fires) OR a reserved word (``cands``/``anchors``/``last``/``prev``/
        ``previous``/``ref``). A leading ``$`` is tolerated (typo absorption).
        A real fid never matches: result names are grammar-locked to ``rN`` and
        the reserved words are not fid prefixes, so a bare fid falls through to
        fid handling."""
        if not isinstance(arg, str):
            return False
        base = _ref_base(arg)
        if base in results or base in seed or base in _program_ref_names:
            return True
        if base.lower() in _RESERVED_REFS:
            return True
        # the ref:/ref_ typo-prefix slips (already tolerated in _resolve_ref)
        for pre in ("ref:", "ref_", "ref-", "ref."):
            if base.startswith(pre) and base[len(pre):] in results:
                return True
            if base.startswith(pre) and base[len(pre):] in seed:
                return True
        # a leftover leading '$' on anything else is still a ref-intent slip
        return arg.startswith("$")

    def _raw(arg):
        # A ref token (bare rN / cands / anchors / last / ... , or a $-slip)
        # resolves through the ONE resolver (cands/anchors resolve there too);
        # anything else is a literal that passes through.
        if _is_ref_token(arg):
            return _resolve_ref(arg)
        return arg

    def as_point(arg):
        xy = _node_xy(_raw(arg))
        if xy is None and isinstance(arg, str) and arg in by_fid:
            xy = by_fid[arg]["xy"]
        if xy is None:
            # an rN ref that resolved to a SINGLETON list (a find narrowed
            # to one) where ONE point was wanted -- unwrap it. Only in the
            # point/centerline coercers: as_list/as_points keep list semantics.
            u = _unwrap_listish(_raw(arg))
            if isinstance(u, list) and len(u) == 1:
                xy = _node_xy(u[0])
        if xy is None:
            raise ProgramError(
                f"cannot resolve a POINT from {arg!r}: this arg wants ONE node "
                f"(a single fid, an rN ref to a one-node result, or [x,y]). A name "
                f"is not a place: if it is a landmark with several instances, "
                f"resolve WHICH one first -- score the set against the cue's other "
                f"refs (compare + score_by) and pass rN[0] (the head of YOUR scored "
                f"set), or narrow it (filter_items); or pass it as a compare "
                f"ref to measure candidates against each instance. "
                f"{_avail('point')}{_sig_suffix()}")
        return xy

    def as_list(arg):
        v = _unwrap_listish(_raw(arg))
        if isinstance(v, list):
            # A list LITERAL the model wrote (items:['r0'] / ['d0','d1']) carries
            # unresolved tokens: a nested rN must flatten to its rows and a bare
            # fid must resolve to its row dict -- otherwise a downstream .get() on
            # the raw string crashes (was an unhandled AttributeError). Mirror of
            # the as_points nested-rN fan-out. A list of already-resolved dicts
            # (the normal `items:"r0"` path) has no strings and passes through.
            if any(isinstance(x, str) for x in v):
                out: list = []
                for x in v:
                    if _is_ref_token(x):
                        sub = _unwrap_listish(_resolve_ref(x))
                        out.extend(sub if isinstance(sub, list) else [sub])
                    elif isinstance(x, str) and x in by_fid:
                        out.append(by_fid[x])
                    elif isinstance(x, str):
                        # an unknown / typo'd fid string -- passing it through raw
                        # would slip past the op's absent-key guard (a live sibling
                        # dict carries the key) and crash a later .get() on the
                        # string. Reject loud, mirroring the as_points loop.
                        raise ProgramError(
                            f"cannot resolve {x!r} in list {arg!r}: not a fid in this "
                            f"surface or an rN ref. {_avail('list')}{_sig_suffix()}")
                    else:
                        out.append(x)
                return out
            return v
        if isinstance(v, dict) and (v.get("fid") is not None or v.get("xy") is not None):
            # a single resolved ROW (r0[0], a pick) where a list-op wants a list
            # -> a one-item list (do-what-it-means; mirror of the as_points
            # single-row wrap). Downstream filter/rank/score is a list either way.
            return [v]
        raise ProgramError(
            f"cannot resolve a LIST from {arg!r}: wants a list -- an rN ref to a "
            f"find/compare/filter_items result -- NOT a single fid. "
            f"{_avail('list')}{_sig_suffix()}")

    def as_points(arg):
        """A list of point-bearing dicts. Resolves fid strings / ``{fid}`` dicts
        through ``by_fid`` (carrying fid + class through for downstream labeling)
        and wraps bare ``[x, y]`` literals — but NEVER passes a raw fid on to
        geom_compute (which would ``float()`` the fid string and crash)."""
        v = _unwrap_listish(_raw(arg))
        if isinstance(v, list) and len(v) == 2 and all(isinstance(z, (int, float)) for z in v):
            v = [v]  # a bare [x,y] where a point-LIST is wanted -> a one-point list
                     # (parity-safe: two bare numbers always hard-errored in the loop)
        if isinstance(v, dict) and (
                v.get("fid") in by_fid
                or (isinstance(v.get("xy"), (list, tuple)) and len(v.get("xy")) == 2)):
            v = [v]  # a single resolved row (r0[0]) where a point-LIST is wanted
                     # -> a one-item list (do what the model means; downstream is a list)
        if isinstance(v, str) and v in by_fid:
            v = [by_fid[v]]  # a bare fid (between.a:'building_129') where a
                             # point-LIST is wanted -> its row (mirror of as_list)
        if not isinstance(v, list):
            raise ProgramError(
                f"cannot resolve a POINT-LIST from {arg!r}: wants a list of points -- an "
                f"rN ref to a find/compare result, or a list of "
                f"fids/[x,y] -- NOT a single fid. {_avail('list')}{_sig_suffix()}")
        out = []
        for it in v:
            if isinstance(it, dict) and isinstance(it.get("xy"), (list, tuple)) and len(it["xy"]) == 2:
                out.append(it)
            elif isinstance(it, str) and it in by_fid:
                out.append(by_fid[it])
            elif isinstance(it, dict) and it.get("fid") in by_fid:
                merged = {**by_fid[it["fid"]], **it}
                if merged.get("xy") is None:
                    # a null-xy overlay (a pasted row echo with xy elided)
                    # must not erase the resolved coords -- downstream
                    # consumers silently skip xy-less dicts.
                    merged["xy"] = by_fid[it["fid"]].get("xy")
                out.append(merged)
            elif isinstance(it, (list, tuple)) and len(it) == 2 and all(isinstance(z, (int, float)) for z in it):
                out.append({"xy": list(it)})
            elif _is_ref_token(it):
                # an rN nested INSIDE the list literal (pts:['r1', ...]) -> resolve
                # and flatten its rows through this same resolver (the point-list
                # mirror of the compare multi-row fan-out); a genuinely unresolvable
                # ref still raises inside _resolve_ref.
                out.extend(as_points(_resolve_ref(it)))
            else:
                raise ProgramError(
                    f"cannot resolve a point in the list from item {it!r}: each item must "
                    f"be a fid in this surface, a {{'fid':...}} dict, an [x,y], or an rN. "
                    f"{_avail('point')}{_sig_suffix()}")
        return out

    # ---- auto-attach columns (the parameter rides in the key name) ----
    # Naming a per-ref relation column d_/brg_gap_/brg_/dc_<ref>, a chart
    # column along_/off_<ref> (a centerline-carrying <ref>), or a color
    # column color_<word> anywhere a column key is accepted IS the
    # computation request (measured before the attach: ~219 hits/110 eps of
    # pure attach-first ceremony — the model reasons the geometry correctly
    # and is blocked on ceremony). When <ref> is a LIVE surface fid the SAME
    # nested single-op compare runs transparently — refs:[<fid>] for the
    # relation columns, frame:{reference:<fid>} for the along_/off_ chart —
    # and the op proceeds (full reuse, nothing reimplemented); a color_<word>
    # key attaches the named color column the same way (an off-palette word
    # is a loud error listing the palette). vx_/vy_ and the arrangement
    # columns stay compare-only: a side/toward frame does not ride the key.
    # The op's own absent-key guard stays as the fallback for a genuinely
    # unresolvable / guessed key (compare attaches nothing there, so the loud
    # error still fires).
    # The tuple itself is at module scope, beside the other two prefix
    # tables, so the Python surface reads the same one.

    def _auto_attach(items, keys, notes=None):
        if not (isinstance(items, list) and items):
            return items
        refs: list = []
        charts: list = []
        colors: list = []
        standings: list = []
        # (direction, ref) pairs; the DIRECTION rides the prefix and the
        # reference the suffix, so one key names the whole column.
        cardinals: list = []
        for k in keys:
            # already-attached short-circuit: ANY present value means the
            # attach ran -> byte-parity, skip. (brg_gap_ is absent-on-contact,
            # so an all-contact pool re-runs the compare -- a no-op -- and the
            # absent-key guard's CONTACT branch then states the fact:
            # brg_gap_<ref> absent where d_<ref>=0; brg_<ref> still defined.)
            if not isinstance(k, str) or any(
                    isinstance(it, dict) and it.get(k) is not None for it in items):
                continue  # not a column key, or already attached
            if k == "color_conf":
                continue  # a base surface field, never a column request
            for pre in _AUTOATTACH_PREFIXES:
                if k.startswith(pre):
                    param = k[len(pre):]
                    if pre in _STANDING_SIDE_PREFIXES and param in staged:
                        # a side key naming a staged viewpoint handle: the
                        # attach is that viewpoint's presence-pair columns.
                        if param not in standings:
                            standings.append(param)
                        break
                    if pre == "color_":
                        canon, note = _resolve_color_word(param)
                        if canon not in colors:
                            colors.append(canon)
                        if note is not None and notes is not None:
                            notes.append(note)
                    elif pre in _CARDINAL_COL_PREFIXES:
                        # A map-fixed cardinal needs only the reference's
                        # geometry -- no centerline, no stance, no viewer --
                        # so ANY ref on the surface attaches one, including
                        # the merged multi-way features whose centerline is
                        # None and which therefore chart nothing.
                        if param in by_fid or param in staged:
                            _cd = (pre.strip()[:-3].strip(), param)
                            if _cd not in cardinals:
                                cardinals.append(_cd)
                            if param not in refs:
                                refs.append(param)
                    elif pre in ("ahead of ", "behind "):
                        # frame-only columns: only a staged handle attaches
                        # them from a key (handled above); anything else
                        # falls to the op's own absent-key guard.
                        pass
                    elif pre in ("along ", "left of ", "right of "):
                        # the no-side polyline chart takes nothing but the
                        # reference, so the parameter fully rides the key —
                        # but only a fid CARRYING a stored centerline charts;
                        # anything else falls through to the loud guard.
                        _n = by_fid.get(param)
                        _cl = (_n.get("centerline")
                               if isinstance(_n, dict) else None)
                        if (isinstance(_cl, list) and len(_cl) >= 2
                                and param not in charts):
                            charts.append(param)
                    elif pre == "distance to ":
                        # gap column on any ref; a line-carrying fid charts
                        _n = by_fid.get(param)
                        _cl = (_n.get("centerline")
                               if isinstance(_n, dict) else None)
                        if isinstance(_cl, list) and len(_cl) >= 2:
                            if param not in charts:
                                charts.append(param)
                        elif ((param in by_fid or param in staged)
                              and param not in refs):
                            refs.append(param)
                    elif ((param in by_fid or param in staged)
                          and param not in refs):
                        refs.append(param)
                    break
        for canon in colors:
            items = _attach_color_column(items, canon)
        for h in standings:
            items = _attach_standing_columns(items, staged[h])
        for param in charts:
            sub = run_program(
                cands, anchors,
                {"ops": [{"ref": "_autoattach", "op": "compare",
                          "args": {"items": items,
                                   "frame": {"along": param}}}]},
                geometry_sink=geometry_sink, staged=staged, standing=standing)
            out = sub["results"].get("_autoattach")
            if isinstance(out, list) and out:
                items = list(out)
        if not refs:
            return items
        _cargs = {"items": items, "refs": refs}
        if cardinals:
            _cargs["cardinals"] = [{"dir": d, "ref": r} for d, r in cardinals]
        sub = run_program(cands, anchors,
                          {"ops": [{"ref": "_autoattach", "op": "compare",
                                    "args": _cargs}]},
                          geometry_sink=geometry_sink, staged=staged,
                          standing=standing)
        out = sub["results"].get("_autoattach")
        return list(out) if isinstance(out, list) and out else items

    def _relate_ran(key, items) -> bool:
        """Did the compare that WRITES ``key`` actually run on these rows?

        A relation column has siblings: relating rows to a place stamps
        ``distance to <lab>`` beside ``facing <lab>``. So a ``facing
        <lab>`` missing on every row while ``distance to <lab>`` is
        present is the geometry declining to name a direction (every row
        TOUCHES the place, which has no min-gap vector), not the model
        guessing a column name. Only ``in_or_blank`` needs the
        distinction, and it needs one that works for an ``as`` LABEL --
        ``_contact_fact`` reads ``by_fid`` and answers "" for a labelled
        ref.
        """
        if not (isinstance(key, str) and key.startswith("facing ")):
            return False
        sib = "distance to " + key[len("facing "):]
        return any(isinstance(it, dict) and sib in it for it in items)

    def _contact_fact(key, items) -> str:
        """The degenerate-column FACTS for the absent-key guards — the cases
        where the compare RAN and the column is legitimately absent, so a
        'not a fid on this surface' remedy would state the wrong wrongness:
        (a) a d_/brg_gap_<fid> whose <fid> mask contains the whole surface
        window (the containment fact; gap-to-it is a degenerate constant);
        (b) brg_gap_<fid> where d_<fid> is present-and-zero on every row
        (contact — no min-gap direction exists). Returns the fact + the
        defined siblings, or ''."""
        if isinstance(key, str) and key.startswith(
                ("distance to ", "facing ")):
            _p = (key[len("facing "):]
                  if key.startswith("facing ")
                  else key[len("distance to "):])
            _n = by_fid.get(_p)
            if isinstance(_n, dict) and _n.get("covers_window"):
                return (f"{_p}'s mask contains the whole surface window, so "
                        f"'distance to {_p}'=0 for every row (containment, "
                        f"not proximity) and neither the distance nor the "
                        f"facing is stamped; 'center distance to "
                        f"{_p}'/'bearing from {_p}' attach per-row and DO "
                        f"order/filter. ")
        if not (isinstance(key, str)
                and key.startswith("facing ")):
            return ""
        param = key[len("facing "):]
        if param not in by_fid:
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

    _prog_ops = list(program.get("ops", []) or [])
    # Every ref THIS program defines — drives the top-to-bottom read rule in
    # ``_resolve_ref`` (an rN ref may name only an op above it or an un-shadowed
    # prior-call result; one way to read a name, no stale-read ambiguity).
    _program_ref_names = {o.get("ref") for o in _prog_ops if isinstance(o, dict)}
    # ``select`` is not an op; the decision is ``act.select`` under the three-
    # verb surface. A program that carries a select op (a model drifting to a
    # select-as-op grammar) gets the actionable redirect, not a bad-op.
    if any(isinstance(o, dict) and o.get("op") == "select" for o in _prog_ops):
        raise ProgramError(
            "select is not a program op (programs ANALYZE only); the decision "
            "is the separate `act` TOOL's select field "
            "{fid, basis, runner_up (null ok), confirmed?}.")

    # --- FULL ORDER FREEDOM: execute in dependency (topological) order --------
    # A program is order-free: any op may reference a handle any OTHER op in the
    # SAME program produces, regardless of emission position. We topologically
    # sort by handle-dependency so every consumed handle is produced BEFORE it is
    # read -- a forward reference is a resolved edge, not an error.
    # The ops are PURE dataflow (each builds NEW rows via {**it, ...}, never
    # mutating the shared surface), so ANY topological order yields the IDENTICAL
    # result -- this is faithful evaluation of the program's DAG, not a
    # meaning-changing reshuffle. The only order-sensitive shapes are LOUD errors
    # (never silent guesses): a handle DEFINED TWICE (ambiguous), or a reference
    # CYCLE (unresolvable). The agent's EMITTED-last op stays the terminal.
    _emitted_terminal = _prog_ops[-1] if _prog_ops else None
    _handle_seen: set = set()
    for _o in _prog_ops:
        _h = _o.get("ref") if isinstance(_o, dict) else None
        if _h is None:
            continue
        if _h in _handle_seen:
            raise ProgramError(
                f"handle {_h!r} is defined by two ops -- one handle names ONE "
                f"op (a program never redefines a handle). Give the second op a "
                f"fresh ref (rN) and reference each by its own name.")
        _handle_seen.add(_h)

    def _consumed_handles(_o):
        """In-program handles this op READS anywhere in its keys (args + a
        mis-placed top-level frame), minus its own ref -- a handle is a token
        whose base names ANOTHER op this program defines."""
        _found: set = set()

        def _walk(v):
            if isinstance(v, str):
                if _ref_base(v) in _program_ref_names:
                    _found.add(_ref_base(v))
            elif isinstance(v, dict):
                for _vv in v.values():
                    _walk(_vv)
            elif isinstance(v, (list, tuple)):
                for _vv in v:
                    _walk(_vv)

        if isinstance(_o, dict):
            for _k, _v in _o.items():
                if _k not in ("ref", "op"):
                    _walk(_v)
        _found.discard(_o.get("ref") if isinstance(_o, dict) else None)
        return _found

    # Stable Kahn sort: among ops whose deps are all produced, keep EMISSION
    # order -- a fully-ordered program is left EXACTLY as written and the
    # reorder is deterministic (served == teacher byte-parity). A handle a
    # program never defines is not an edge here (it falls to _resolve_ref's
    # unknown-ref path), so only a genuine mutual reference trips the cycle.
    _dict_ops = [_o for _o in _prog_ops if isinstance(_o, dict)]
    _nondict_ops = [_o for _o in _prog_ops if not isinstance(_o, dict)]
    _need = {id(_o): _consumed_handles(_o) for _o in _dict_ops}
    _emit_i = {id(_o): _i for _i, _o in enumerate(_dict_ops)}
    _produced: set = set()
    _pending = list(_dict_ops)
    _ordered: list = []
    while _pending:
        _ready = [_o for _o in _pending if _need[id(_o)] <= _produced]
        if not _ready:
            _cyc = sorted(str(_o.get("ref")) for _o in _pending)
            raise ProgramError(
                f"reference cycle among {_cyc}: these ops reference one "
                f"another so no order resolves them -- an op cannot read a "
                f"handle that (directly or through others) reads its own "
                f"result. Break the cycle.")
        _ready.sort(key=lambda _o: _emit_i[id(_o)])
        _pick = _ready[0]
        _ordered.append(_pick)
        _pending.remove(_pick)
        _produced.add(_pick.get("ref"))
    _prog_ops = _ordered + _nondict_ops

    # The effective-args echo: one line per op, executor truth (defaults
    # applied), served beside ``results`` so a decode-dropped key is visible.
    for op in _prog_ops:
        if not isinstance(op, dict):
            # Keep the loud error surface (a silent skip would hide a malformed
            # emission from the model); the grammar makes this an impossible
            # token live, so this guards offline/direct callers.
            raise ProgramError(
                f"each op must be an object {{ref, op, args}}; got "
                f"{type(op).__name__}: {str(op)[:80]!r}")
        ref = op.get("ref")
        name = op.get("op")
        if name not in OP_VOCAB:
            if name in _DELETED_OPS:
                raise ProgramError(
                    f"{name!r} is not an op. {_DELETED_OPS[name]}")
            raise ProgramError(f"bad op {name!r}: not a known op; vocab={OP_VOCAB}")
        if not ref:
            # The op name is VALID -- it just lacks its 'ref' field. State the
            # mechanical fact, so the model does not read a "bad op X (ref=None)"
            # message as X being an invalid op (it is in the vocab) and abandon it.
            raise ProgramError(
                f"op {name!r} lacks its required 'ref' field (the op name is "
                f"valid; a later op reads the result back by that name, e.g. "
                f"items:'r0').")
        # REJECT unknown top-level keys instead of silently dropping them. An op
        # is exactly {op, ref, args}; a 'frame' emitted at
        # the top level would vanish without a trace (a verbatim-retry loop: the
        # model re-emits the same op because nothing told it WHY the key had no
        # effect). Echo the corrected form so the retry is mechanical.
        _extra = [k for k in op if k not in ("op", "ref", "args")]
        # A stray top-level key that is THIS op's own documented optional arg
        # (compare's view/between/neighbors, and any key the op's args already
        # accept) is folded into args instead of rejected — the intent is
        # unambiguous, and rejecting it is the worst-chaining error class
        # (29 errors / 12 eps; 41% of next turns errored again). Op-name keys
        # still get the chain advice; genuinely unknown keys still fail loud.
        if _extra and name == "compare":
            _foldable = [k for k in _extra if k in ("frame",)]
            if _foldable:
                _args = dict(op.get("args") or {})
                for k in _foldable:
                    _args.setdefault(k, op[k])
                op = {**op, "args": _args}
                for k in _foldable:
                    op.pop(k, None)
                _extra = [k for k in _extra if k not in _foldable]
        if _extra:
            _k0 = _extra[0]
            if _k0 in OP_VOCAB:
                # The stray key is ITSELF an op name (e.g. rank_by attached to
                # a compare) -- nesting it into args would be wrong advice
                # (measured: a round collapse followed it). Chain it.
                raise ProgramError(
                    f"op {name!r} carries unknown top-level key(s) {_extra} -- an "
                    f"op is exactly {{op, ref, args}}. {_k0!r} is its own op: "
                    f"chain it as the NEXT op, e.g. "
                    f"{{op:'{_k0}', ref:'r<next>', args:{{items:'{ref}', ...}}}}.")
            raise ProgramError(
                f"op {name!r} carries unknown top-level key(s) {_extra} -- an op "
                f"is exactly {{op, ref, args}}, so those keys were NOT applied. "
                f"The corrected form: {{op:'{name}', ref:'{ref}', "
                f"args:{{..., {_k0}:{{...}}}}}}.")
        # A required arg accessed as ``a["k"]`` that the model omitted raises a
        # CLEAN, actionable ProgramError (the agent reads it and retries) rather
        # than a cryptic KeyError; optional args keep using ``a.get("k")``.
        # NULL-STRIP (the required-form schema): a null value is the served
        # "not this one" sentinel -- the executor sees exactly the args the
        # model filled, and every absent-arg path behaves as before.
        a = _Args({k: v for k, v in (op.get("args", {}) or {}).items()
                   if v is not None}, name)
        _cur_op["op"] = name  # so a resolver's ProgramError appends THIS op's signature
        if name == "find":
            # SEARCH / EXPLORE the whole world-state surface — the absorbed
            # search_features capability, surface-only (no graph/ctx, so the
            # offline teacher runs it identically). Filters AND-compose; all
            # optional. A query returns THE MATCH SET (best-first, len = the
            # match count); a query-less find browses the full selected
            # surface. Geometry ops chain on the rN either way.
            _q_notes: list = []
            _q_matched = False   # a query ran and matched (the rN is ordered)
            _name_set = False    # tied ways of ONE name: render the shape
            _find_cols: list = []  # columns this find attached (budget prio)
            _empty_cond = None  # the truthful no-match condition, set on a query miss
            _color_col = None   # set when the color arg attaches a column
            prov = str(a.get("provenance") or "any").strip().lower()
            if prov in ("prior", "priors", "prebaked"):
                rows = list(anchors)              # baked map priors only
            elif prov in ("runtime", "detected", "detection", "detections"):
                rows = list(cands)                # in-flight detections only
            elif prov in ("any", "all", "both"):
                rows = cands + anchors            # the whole surface
            else:
                # Fail LOUD on an unknown provenance instead of silently
                # widening to the whole surface. The tool schema's enum
                # promises any|prior|runtime, but the executor owns its own
                # contract: ``run_program`` also serves callers with no
                # grammar over them at all — the dashboard lab runs
                # human-authored programs straight through it.
                raise ProgramError(
                    f"unknown find provenance {prov!r}; legal: any | prior "
                    f"(baked map features) | runtime (in-flight detections)."
                    f"{_sig_suffix()}")
            if a.get("class") is not None:
                # Not an arg — retrieval is ONE idiom. Loud redirect naming
                # query (never a silent drop of the narrowing the model asked
                # for).
                raise ProgramError(
                    f"find takes no 'class' arg; retrieval is find{{query}}: "
                    f"fuzzy free text over each row's name + tags (priors) "
                    f"or detector class (detections). Re-emit with "
                    f"query:{str(a['class'])!r}.{_sig_suffix()}")
            if a.get("color"):
                # ATTACH the named color column color_<word> (conf-weighted
                # cue match) on every color-carrying row and DROP NONE --
                # color FEEDS the sort, it never GATES the pool (CLAUDE.md
                # contract). FAMILY-soft + color_conf-shrunk: low-conf -> ~0.5
                # neutral, so an unreliable color is muted, never a hard
                # drop; a colorless row omits the column (score_by's neutral).
                # There is NO hard color cut: filter_items on any color_* key
                # is REJECTED (it dropped correctly-detected GTs); color is
                # rank-only via score_by. See the filter_items handler.
                qc, _cnote = _resolve_color_word(str(a["color"]))
                if _cnote is not None:
                    _q_notes.append(_cnote)
                rows = _attach_color_column(rows, qc)
                _color_col = "color_" + qc
                _find_cols.append(_color_col)
            _fid_hit = (by_fid.get(str(a.get("query") or "").strip())
                        if a.get("query") else None)
            if _fid_hit is not None:
                # a query naming a live fid IS retrieval of that row --
                # the fuzzy scorer matches text, never ids, and the old
                # miss taught nothing (framestage_1000: the model looked
                # rows up by fid and was told "no such object exists").
                rows = [_fid_hit]
                _q_matched = True
                _q_notes.append(
                    "query " + repr(str(a.get("query")).strip())
                    + " is a live fid -- that row returned directly.")
            elif a.get("query"):
                # Fuzzy free-text over the walled search_text + name (typo / OCR /
                # abbreviation tolerant, strictly lexical). The result IS the
                # MATCH SET, best-first: len = the match count, slices and
                # downstream ops operate on matches, and the sub-threshold
                # tail stays out (its size is stated in the result note; a
                # query-less find browses it). Every RETURNED row carries
                # name_match raw (a NAME matched, never WHICH instance the
                # cue means); the grid renders no such column.
                _q_raw = str(a["query"])
                if ";" in _q_raw:
                    # ';' is not a query operator — silently fuzz-scoring the
                    # compound as ONE text blob buried real matches (measured
                    # at 0.63 with no remedy). Loud, naming the supported
                    # idioms.
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
                        f"single value ('cuisine=pizza').{_sig_suffix()}")
                q_data = T.fuzzy_prepare_query(_q_raw)
                _searched_rows = list(rows)
                _memo = _find_memo(cands, anchors)
                _mkey = (prov, _q_raw, _color_col)
                _hit = _memo["scored"].get(_mkey)

                def _memo_census():
                    # the surface census a miss or a thin match serves:
                    # edit distances over the whole vocabulary, once
                    got = _memo["census"].get(_mkey)
                    if got is None:
                        got = _memo["census"][_mkey] = _census_note(
                            _searched_rows, query=_q_raw)
                    return got
                # The fuzz band's denominator is a CORPUS fact, minted over
                # the same rows being scored (the sibling of `tag_family_keys`
                # below): how rare a query word is here, and whether any row
                # could cover it at all.
                _idf = _memo["idf"].get(prov)
                if _idf is None and _hit is None:
                    _idf = _memo["idf"][prov] = T.token_idf(_searched_rows)
                scored = [{**x, "name_match": round(
                              T.find_score(q_data, x.get("search_text") or "",
                                           x.get("name"), x.get("tags"),
                                           idf=_idf), 3)}
                          for x in rows] if _hit is None else None
                # WIDEN to the query's own tag FAMILY. An exact tag VALUE hit
                # is a corpus fact the per-row scorer cannot see, so it is
                # applied here, over the searched set: a query that IS a value
                # of some tag key reaches every row carrying that key, at
                # T.TAG_FAMILY_SCORE -- retrieved, and ranked BELOW every
                # exact hit. `find house.` goes 41 -> 3,414 rows with the 41
                # exact `building=house` hits still at 0-40 of the find order.
                # Never a drop: no row that matched before stops matching.
                #
                # Priced on the 325-episode facing family (containment_probe,
                # 2026-08-16): the widening ALONE is worth 0.0 points, and
                # WITH the distance order a `near <anchor>` clause compiles
                # to it is +76.3 [+70.6, +82.0] containment@24, 161W/0L,
                # exact McNemar p=6.8e-49 (2.8% -> 79.1% of the 211-episode
                # reachable pool, 1.8% -> 51.4% of all 325). The two halves
                # are one change: the order alone is +0.5 and the widening
                # alone is 0.0, because 41 rows never bind a 24-row budget.
                if _hit is None:
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
                    # The query matched NOTHING real: the honest answer to a
                    # NAMED query is the no-match affordance (disambiguate via
                    # the cue's other refs), not a noise-ordered dump. The
                    # empty hint states the TRUTHFUL condition (the scoring
                    # threshold — never a claim about the world), and the
                    # QUERY-CONDITIONED census (tags sharing tokens with the
                    # query, then the global top) rides the miss so the
                    # re-query is one read.
                    scored = []
                    _q_tokens = sorted({t for _cl in q_data for t in _cl[1]})
                    _empty_cond = (
                        f"find matched nothing: no row scored >= "
                        f"{T.FUZZY_THRESHOLD} against query {_q_raw!r} "
                        f"(query tokens: {_q_tokens}).")
                    _q_notes.append(_memo_census())
                    # Producing-idiom parity for the color family: a query
                    # clause that IS a palette color word matches no row
                    # lexically because color is not row text -- the miss
                    # names the column that carries it (the same parity
                    # every absent-key error already serves). The column, not
                    # the op that attaches it: this note reaches the sentence
                    # channel, whose only move is a sentence.
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
                    # The MATCHED HEAD must survive the census summarizer: the
                    # model asked "who is named X" -- a {len, census} census
                    # that hides the two real matches manufactured a false
                    # "lexical search is broken" belief on the first live
                    # flight (17 wasted rounds). Name the best matches in a
                    # note the census renders.
                    _n_match = sum(1 for r in scored
                                   if r["name_match"] >= T.FUZZY_THRESHOLD)
                    # IF THE QUERY NAMED SOMETHING, THE BAND IS A DIFFERENT
                    # THING. A name tier (>= ``_NAME_PHRASE_TIER_BASE``)
                    # means the query matched a NAME; everything under the
                    # token-bag ceiling then shares a WORD and not a
                    # referent. ``find("Neuralia Road")`` served its exact
                    # rows plus 31 at 0.48 -- Mendiburu Road, Randsburg
                    # Mojave Road, Kenniston Road -- every one of them on
                    # the word "road".
                    #
                    # DERIVED, NOT DECLARED. The query's own result says
                    # which kind it was, so the model never has to say.
                    # It was asked to: ``find(<class>, name=<name>)`` exists
                    # and matches a name "as a NAME, never fuzzily" -- and
                    # over 500 episodes it was written 17 times against 682
                    # bare proper-noun finds, 40 to 1. A declaration the
                    # surface can compute is one it should compute.
                    #
                    # THE CUT IS EVERYTHING UNDER THE NAME TIER. A proper
                    # noun that HITS is answered by the thing it names, and
                    # a lesser tier under it is a different KIND of row, not
                    # a worse match for the same one: `find("California City
                    # Boulevard")` returned its 75 ways plus Ace Hardware,
                    # Subway, Coyote Cafe and 7 more, every one of them a
                    # business carrying `addr_street=California City
                    # Boulevard`. They ride the same rN as the road and
                    # dilute the head that answers "which street".
                    #
                    # A row reached this way is reachable a better way: a
                    # named business is its OWN exact hit on its own name,
                    # and `.tagged` still holds the tier as a set.
                    #
                    # DEGRADE ONLY WHEN THERE IS NOTHING TO DEGRADE FROM.
                    # The cut is conditioned on a name tier EXISTING, so a
                    # class word is untouched: `find("pool")` hits no exact
                    # tier and keeps its whole fuzzy band, which carries all
                    # 17 episodes where fuzz adds the ground truth (16W/0L
                    # paired, p = 3.05e-5).
                    #
                    # DERIVED, NOT DECLARED -- the query's own result says
                    # which kind it was, so the model never has to.
                    _n_named = sum(
                        1 for r in scored
                        if r["name_match"] >= T._NAME_PHRASE_TIER_BASE)
                    # A PROPER NOUN, not any exact hit. A detection whose
                    # CLASS is the query scores 1.0 and reaches the name
                    # tier too -- ``find("parking")`` over three parking
                    # detections -- and cutting the band there would delete
                    # the ``amenity=parking`` priors that ARE the candidates
                    # for "the parking lot". Only a hit on a row's NAME
                    # answers alone, because only a name is a referent.
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

                    def _match_label(r):
                        # what the query HIT, as a NAME: the row's own name
                        # when it has one, else the tags that identify it.
                        # A prefix of the match haystack is not a name --
                        # `prior_1000 'osm_id=way/1136811785; building=yes;
                        # sou'` cut off mid-word and led with the row's
                        # unique id, which is the one tag that identifies
                        # nothing (_census_counts drops it for the same
                        # reason). Whole pairs only, and the id is not one.
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
                        return repr(str(r.get("class")
                                        or r.get("fid") or "?")[:32])

                    # OFF THE MATCH SET, NOT OFF ``scored``. The head names
                    # what is IN this rN; sliced from ``scored`` it went on
                    # advertising rows the name cut had just removed --
                    # find("Neuralia Road") said "Best: ... 'Mendiburu Road'
                    # (match 0.452)" about a row no op on that handle could
                    # reach. A served list of things you cannot have is the
                    # same defect as a served count that silently shrank.
                    _head_note = "; ".join(
                        f"{r.get('fid')} {_match_label(r)} "
                        f"(match {r['name_match']})"
                        for r in scored[:_n_match][:4]
                        if r["name_match"] >= T.FUZZY_THRESHOLD)
                    # SERVED TRUTH: the match set IS the result — state the
                    # matched head's score range and the size of the
                    # sub-threshold tail that stayed OUT of this rN (a
                    # query-less find browses it).
                    # "best-first" is a CLAIM about this order, and it was
                    # false on 34.2% of the results that made it: when every
                    # matched row scores the same, the order left is fid
                    # order and nothing about it is best. The claim is now
                    # made only when the scores actually separate, and the
                    # tie says so instead -- once, here, where the retrieval
                    # knows it. The render used to say it a second time in
                    # its own words, and a result carrying both read
                    # "(best-first, match 0.8)" and "all 3414 rows are
                    # equally good matches, so this head's order is
                    # arbitrary" in one hint.
                    _m_vals = [r["name_match"] for r in scored[:_n_match]]
                    # A NAME'S WAYS ARE A SET, NOT A RANKING. Every row ties
                    # at the name tier, so a rendered table is N rows in an
                    # order that means nothing and the model reads the first
                    # as the answer (calcity::284#282 took ``street_88[0]``
                    # and swept 1 of 10 segments). The initial reply is the
                    # SHAPE -- how much road, how many ways -- and the rN
                    # holds every row for a slice or an explore. Saying the
                    # order is arbitrary, then rendering it anyway, spends
                    # tokens teaching the model to distrust what it is being
                    # shown.
                    _ex_rows = [r for r in scored[:_n_match]
                                if r.get("name_match", 0.0)
                                >= T.NAME_TIER_FLOOR]
                    _ex_names = {str(r.get("name") or "") for r in _ex_rows}
                    # The one name must BE a name: unnamed rows collapse to
                    # a single "" and 60 anonymous ways read as one street.
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
                        # A MIXED POOL WITH A TIED HEAD reads as decided
                        # when it is not: "best-first" is true of the
                        # POOL while the first k rows tie among
                        # themselves (140#138's refly: six detections at
                        # 1.0 over priors at 0.8, [0] taken as if the
                        # rank had chosen). Said only when the tie is
                        # real (k >= 2).
                        _top_n = sum(1 for _x in _m_vals
                                     if _x == _m_vals[0])
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
                    # A MIXED POOL STATES ITS TIER CENSUS as structure the
                    # model can hold: the strata are live attribute views,
                    # so "the ones that ARE the thing" is a thing in hand
                    # rather than a float column the grid does not render.
                    _n_ex = sum(1 for x in _m_vals
                                if x >= T.NAME_TIER_FLOOR)
                    _n_tg = sum(1 for x in _m_vals
                                if T._KV_TIER_BASE <= x
                                < T.NAME_TIER_FLOOR)
                    _n_fz = _n_match - _n_ex - _n_tg
                    if _n_ex and (_n_tg or _n_fz):
                        _q_notes.append(
                            f"tiers: {_n_ex} exact, {_n_tg} tagged, "
                            f"{_n_fz} fuzzy -- .exact / .tagged / .fuzzy "
                            "on this result hold each as a set.")
                    # A NAME'S WAYS ARE A SET, NOT A RANKING. Every row here
                    # ties at the name tier, which this envelope already
                    # calls arbitrary -- so a rendered table is 64 rows in
                    # an order that means nothing, and the model reads the
                    # first one as the answer (calcity::284#282 took
                    # ``street_88[0]`` and swept 1 of 10 segments). The SHAPE
                    # is what the model needs: how much road this is, and
                    # that one explore covers all of it.
                    if _band_cut and not _name_set:
                        # SAY THAT THE CUT HAPPENED. The rows are gone from
                        # the rN, and a count that silently shrank is the
                        # shape this repo keeps getting wrong -- a set that
                        # looks complete and is not. Naming it also teaches
                        # the name/class split by showing it, which is what
                        # a `name=` kwarg asked the model to know in advance.
                        _q_notes.append(
                            f"matched as a NAME: {_band_cut} row(s) under "
                            f"the name tier are a different feature and are "
                            f"NOT in this rN.")
                    if _head_note and not _name_set:
                        # "Best" is an ENDORSEMENT, and on a tag-tier head
                        # it endorsed 0.5-0.8 matches the model then flew
                        # (140#138 flew a parking_aisle 212 m off the cue's
                        # street from exactly this banner). The word now
                        # rides only a NAME-tier head; a class/tag head is
                        # just the head, its numbers already visible.
                        _lead = ("Best" if _m_vals
                                 and _m_vals[0] >= T.NAME_TIER_FLOOR
                                 else "Head")
                        _q_notes.append(f"{_lead}: {_head_note}")
                    # A token NO matched row carries: the match stands (a
                    # real query often names an attribute no row spells),
                    # but it is said out loud, because the same shape is
                    # how a glued token binds silently -- pilot-7 measured
                    # "find car withcolor blue." returning 520 rows with
                    # no error and no note.
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
                        # LOW-match: the surface census (computed PRE-slice
                        # over the searched rows) keeps the re-query
                        # vocabulary one read away — the rescue the zero-
                        # match path already serves.
                        _q_notes.append(_memo_census())
                    _q_matched = True
                # THE narrowing: the returned row set IS the match set.
                rows = scored[:_n_match] if _q_matched else scored
                # ``name=`` -- THE MODEL SAYS WHICH OF THE TWO IT MEANT.
                # A query is either a CLASS ("house") or a PROPER NOUN
                # ("Icaarus Court"), and the surface used to guess by
                # retrieving every tier at once and trusting the ordering.
                # That is exactly where it lost: the exact hit does rank
                # first on a bare find, and then `near=<anchor>` re-sorts by
                # distance and buries it under 65-98 tied fuzz rows.
                #
                # The model already draws the line unprompted --
                # find(<class>, name=<name>) was the LARGEST error family in
                # the run, 135 blocks over 61 episodes scoring 13/61 against
                # 92/194, and the surface answered "find() has no name=
                # argument". It was refused because a silently-dropped kwarg
                # would be the model's whole constraint vanishing, which is
                # an argument for implementing it.
                #
                # Measured, the CLASS half does no disambiguating work: of
                # 353 names in this bake exactly 1 is borne by more than one
                # tag family, and over 1,009 cues naming a feature exactly 1
                # spans families. So the class is accepted because the model
                # writes it, and the NAME is what gates. Do not "optimise"
                # the class term expecting it to narrow.
                _want = str(a.get("name") or "").strip()
                if _want:
                    _nq = T.fuzzy_prepare_query(_want)
                    _named = [r for r in rows
                              if isinstance(r, dict)
                              and T.find_score(_nq,
                                               r.get("search_text") or "",
                                               r.get("name"), r.get("tags"),
                                               idf=_idf)
                              >= T._NAME_PHRASE_TIER_BASE]
                    if not _named:
                        # LOUD, never a silent fall back to the fuzz band
                        # this gate exists to exclude.
                        raise ProgramError(
                            f"nothing found is NAMED {_want!r}"
                            + (f" and matches {_q_raw!r}" if _q_raw else "")
                            + ". A name is matched as a NAME, never fuzzily "
                            "-- check the spelling, or drop the class word "
                            f"and say find({_want!r}).{_sig_suffix()}")
                    if len(_named) < len(rows):
                        _q_notes.append(
                            f"name={_want!r} kept {len(_named)} of "
                            f"{len(rows)} by NAME; the fuzz band does not "
                            "apply to a name")
                    rows = _named
                    _q_matched = True
            if _color_col and not a.get("query") and isinstance(rows, list):
                # A color-carrying browse ORDERS by its own attached column —
                # unordered it renders census-only and reads as a null while
                # the ranking signal sits silent in the handle (measured: the
                # model abandoned the cue on exactly this shape). ORDERING
                # only: every row, colorless included, stays in the rN.
                rows = sorted(
                    rows,
                    key=lambda x: (x.get(_color_col) is not None,
                                   x.get(_color_col) or 0.0),
                    reverse=True)
                _q_matched = True
                _q_notes.append(
                    f"ordered by {_color_col} (highest first; colorless "
                    f"rows last; ordering only -- nothing dropped, all "
                    f"{len(rows)} rows stay in this rN)")
            if _standing_frame is not None and isinstance(rows, list) and rows:
                # The standing viewpoint's side columns ride every find row
                # (the stage receipt states this fact). Presence-pair,
                # ordering/reading evidence only -- never a gate.
                rows = _attach_standing_columns(rows, _standing_frame)
                _h = str(_standing_frame.get("handle") or "v")
                _find_cols.extend(
                    (f"left of {_h}", f"right of {_h}",
                     f"ahead of {_h}", f"behind {_h}"))
            if standing_anchor and isinstance(rows, list) and rows:
                # The standing ANCHOR's proximity column rides every find row,
                # the same way the standing viewpoint's side columns do above.
                # Ordering/reading evidence only -- never a gate. Skipped
                # without error when the anchor's fid left this surface
                # (detection eviction: build_surface fold), or when the column
                # already rides (a program whose own compare attached it).
                _sa_lab = str(standing_anchor.get("label") or "")
                _sa_ref = standing_anchor.get("ref")
                # NOTE: _is_xy_literal is nested inside the COMPARE dispatch
                # branch and is NOT in scope here -- the xy check is
                # inlined (2 lines), not imported.
                _sa_ok = (
                    (isinstance(_sa_ref, (list, tuple)) and len(_sa_ref) == 2
                     and all(isinstance(z, (int, float))
                             and not isinstance(z, bool) for z in _sa_ref))
                    or (isinstance(_sa_ref, str) and _sa_ref in by_fid)
                    or (isinstance(_sa_ref, list) and _sa_ref
                        and all(isinstance(x, str) and x in by_fid
                                for x in _sa_ref)))
                if _sa_lab and _sa_ok and not any(
                        isinstance(it, dict)
                        and it.get(f"distance to {_sa_lab}") is not None
                        for it in rows):
                    _sub = run_program(
                        cands, anchors,
                        {"ops": [{"ref": "_anchor", "op": "compare",
                                  "args": {"items": rows,
                                           "refs": [{"ref": _sa_ref,
                                                     "as": _sa_lab}]}}]},
                        geometry_sink=geometry_sink, staged=staged,
                        standing=standing)
                    _arows = _sub["results"].get("_anchor")
                    if isinstance(_arows, list) and _arows:
                        rows = list(_arows)
                        _find_cols.append(f"distance to {_sa_lab}")
                        if any(isinstance(it, dict)
                               and it.get(f"along {_sa_lab}") is not None
                               for it in rows):
                            _find_cols.append(f"along {_sa_lab}")
            if (not rows
                    and prov not in ("runtime", "detected", "detection",
                                     "detections")
                    and (a.get("query")
                         or prov in ("prior", "priors", "prebaked"))):
                # A ground attempt (a named query, or an explicit
                # provenance:'prior' search) matched NOTHING -- mark it so
                # the compacted hint states the searched-scope fact. The
                # query notes (the class census, a color normalization)
                # ride the marker like any other empty result -- the SAME
                # miss on the runtime rows keeps them, so this path must
                # too.
                rows = _EmptyFindMiss()
                rows.searched = (
                    "the prior map rows (provenance:'prior')"
                    if prov in ("prior", "priors", "prebaked")
                    else "every surface row (prior + runtime)"
                )
                if _q_notes:
                    rows.notes = list(_q_notes)
                if _empty_cond:
                    rows.empty_condition = _empty_cond
            else:
                # A query-matched find is ORDERED (best-first is real) and its
                # rows render; a query-less browse is unordered and renders
                # census + len only (rows render once an ordering exists).
                # Either way the rN holds the full returned set for chaining.
                rows = _BroadRows(rows)
                rows.ordered = _q_matched
                # Ordered stays TRUE: the rN is indexable and pageable like
                # any find. Only the initial RENDER is the shape.
                rows.name_set = bool(_name_set)
                if _find_cols:
                    rows.op_cols = list(_find_cols)
                if _q_notes:
                    rows.notes = list(_q_notes)
                if _empty_cond:
                    rows.empty_condition = _empty_cond
            r = rows
        elif name == "crossings":
            # CONSTRUCT: the places two named features MEET. Every other op
            # ORDERS or CUTS rows that already exist; a road crossing is a
            # point where NO row sits, so it is the one op that mints them.
            #
            # ONE ROW PER DISTINCT CROSSING, each with its own fid (its own
            # [x, y] token — the tree's uniform fid-or-point address, so the
            # row flies / stages / maps / anchors with no new plumbing) and
            # its own xy. NEVER a centroid over several: a single centre over
            # three crossings lands off all three and reads as a confident
            # answer. Zero crossings is a LOUD refusal below, never an empty
            # bind and never a fallback to something near-ish.
            refs_raw = a["refs"]
            if not isinstance(refs_raw, list) or len(refs_raw) != 2:
                raise ProgramError(
                    f"crossings takes exactly two refs -- the two features "
                    f"whose meeting places you want; got "
                    f"{len(refs_raw) if isinstance(refs_raw, list) else type(refs_raw).__name__}"
                    f".{_sig_suffix()}")

            def _x_nodes(el):
                """One crossings ref -> its surface rows. A road is routinely
                several ways (179 of 457 named roads in the bake carry more
                than one row), so a side is EVERY row the ref names."""
                if isinstance(el, dict):
                    el = el.get("ref")
                toks = el if isinstance(el, list) else [el]
                out: list = []
                for t in toks:
                    vals = [t]
                    if isinstance(t, str) and t not in by_fid:
                        v = _unwrap_listish(_resolve_ref(t))
                        vals = v if isinstance(v, list) else [v]
                    for x in vals:
                        n = None
                        if isinstance(x, str):
                            n = by_fid.get(x)
                        elif isinstance(x, dict):
                            n = by_fid.get(x.get("fid")) or (
                                x if x.get("xy") else None)
                        if n is not None and n not in out:
                            out.append(n)
                return out

            def _x_polys(node):
                """A row's EXACT geometry as vertex chains, else its point.

                It used to prefer the ``centerline`` -- a PCA reduction of the
                2 m cell raster -- so a BUILDING was intersected by its own
                derived principal axis, and a road by a reconstruction of the
                line that made it. The mask IS the geometry the feature was
                given, and it crosses whatever it really crosses."""
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

            sides = []
            for _i, _el in enumerate(refs_raw):
                _nodes = _x_nodes(_el)
                _polys = [p for n in _nodes for p in _x_polys(n)]
                if not _polys:
                    raise ProgramError(
                        f"crossings ref {_el!r} names no geometry on this "
                        f"surface -- a crossing needs two features that HAVE "
                        f"a shape. {_avail('point')}{_sig_suffix()}")
                sides.append((_x_label(_nodes, _el), _polys,
                              {str(n.get("fid")) for n in _nodes}))
            # NOTHING CROSSES ITSELF. Every segment touches its own
            # neighbours, so a row appearing on both sides hands back the
            # polyline's own CORNERS as crossings -- a confident answer with
            # no error, exactly the class this op exists to refuse. Reachable
            # by naming one road twice, and (the real case) by two name
            # phrases fuzzy-grounding onto the same row.
            _shared = sides[0][2] & sides[1][2]
            if _shared:
                _same = "the same feature" if sides[0][2] == sides[1][2] \
                    else "the same feature in part"
                raise ProgramError(
                    f"{sides[0][0]} and {sides[1][0]} are {_same} "
                    f"({', '.join(sorted(_shared)[:4])}), and nothing crosses "
                    f"itself -- a feature's own corners are not crossings. "
                    f"Name two DIFFERENT features (find each one and read its "
                    f"row).{_sig_suffix()}")
            (la, fa), (lb, fb) = ((sides[0][0], sides[0][2]),
                                  (sides[1][0], sides[1][2]))
            pa, pb = sides[0][1], sides[1][1]
            xpts, nearest = G.polyline_crossings(pa, pb)
            if not xpts:
                # A ROW-PAIR fact, never a street-level verdict: a named
                # street is many segments, and the tested pair not touching
                # says nothing about its same-named siblings. One episode
                # looped segments, was told "Airway Boulevard and Lime
                # Avenue do not meet" on the FIRST non-touching pair, and
                # never reached the sibling that crosses (22-block miss).
                # Bounded sweep: do any same-named sibling pairs cross?
                _sa = [c for c in by_fid.values()
                       if str(c.get("name")) == str(la)
                       and str(c.get("fid")) not in fa][:8]
                _sb = [c for c in by_fid.values()
                       if str(c.get("name")) == str(lb)
                       and str(c.get("fid")) not in fb][:8]
                _hit = None
                _budget = 24
                for _ra in ([by_fid[f] for f in sorted(fa)
                             if f in by_fid][:8] + _sa):
                    for _rb in ([by_fid[f] for f in sorted(fb)
                                 if f in by_fid][:8] + _sb):
                        if str(_ra.get("fid")) in fa \
                                and str(_rb.get("fid")) in fb:
                            continue          # the pair already tested
                        if _budget <= 0:
                            break
                        _budget -= 1
                        _px, _ = G.polyline_crossings(
                            _x_polys(_ra), _x_polys(_rb))
                        if _px:
                            _hx, _hy = (float(round(_px[0][0])),
                                        float(round(_px[0][1])))
                            _hit = (str(_ra.get("fid")),
                                    str(_rb.get("fid")),
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
                    f"('... near <feature>').{_sig_suffix()}")
            rows = _BroadRows()
            # WHOLE METRES, because this row's fid IS its coordinate and the
            # model has to be able to SAY it. The language spells digits only
            # as whole tokens (no invented ordinals or coordinates), so a
            # 2 dp mint made the crossing referent unaddressable in every
            # form it has: the fid verbatim is bracketed, the xy is decimal,
            # and `rN` is scope-only -- `go to r0.` does not parse. Measured
            # on calcity_v3base: 111 `parse: decimal coordinate` refusals
            # against 16 before the crossings op started minting rows, 17 of
            # 17 sampled echoing a number the surface had just served, and
            # 14 of the 15 episodes that reached a crossing refused acting
            # on it. The bake is a 2 m grid and SR is a 20 m radius, so the
            # rounding is far inside both; the exact intersection stays in
            # the geometry, this is the ADDRESS.
            _seen: set = set()
            for _x, _y in xpts:
                _x, _y = float(round(_x)), float(round(_y))
                tok = format_xy_token(_x, _y)
                # Rounding can collapse two crossings a metre apart onto one
                # address, and two rows sharing a fid is a worse lie than one
                # row: a fid is how every later sentence names this place.
                # Two junctions <1 m apart on a 2 m bake are one junction.
                if tok in _seen:
                    continue
                _seen.add(tok)
                rows.append({"fid": tok, "xy": [_x, _y],
                             "name": f"{la} x {lb}"})
            rows.ordered = True
            rows.notes = [
                f"{len(rows)} distinct crossing{'' if len(rows) == 1 else 's'} "
                f"of {la} and {lb}"
                + ("." if len(rows) == 1 else
                   " -- each is its OWN row with its OWN fid, so narrow to the "
                   "one you mean (order or cut this set) rather than averaging "
                   "them; a point between crossings is at none of them.")]
            r = rows
        elif name == "ends":
            # A ROAD'S TERMINI, as rows. The centerline's two ends are
            # already on every linear row as end0_xy/end1_xy, and a cue
            # naming one -- "the cul-de-sac at the end of Walker Court",
            # 106 of 1,023 calcity cues say "end of" and 73 say
            # "cul-de-sac" -- had no way to reach it: the value was SERVED
            # and the only thing the model could do with it was type it
            # back as a coordinate, which the moving calls refuse.
            # Measured on 25713026: 44 episodes hit that refusal, scoring
            # 13/44, and on calcity::814#808 end1_xy IS the ground truth
            # to within a metre.
            _e_refs = a.get("refs")
            if not isinstance(_e_refs, list) or not _e_refs:
                raise ProgramError(
                    f"ends takes a LIST of refs -- the linear features whose "
                    f"termini you want.{_sig_suffix()}")
            def _e_nodes(el):
                """One ends ref -> its surface rows. A road is routinely
                several ways, so a terminus set is EVERY row the ref names."""
                if isinstance(el, dict):
                    el = el.get("ref")
                toks = el if isinstance(el, list) else [el]
                out_n: list = []
                for t in toks:
                    vals = [t]
                    if isinstance(t, str) and t not in by_fid:
                        v = _unwrap_listish(_resolve_ref(t))
                        vals = v if isinstance(v, list) else [v]
                    for x in vals:
                        nd = None
                        if isinstance(x, str):
                            nd = by_fid.get(x)
                        elif isinstance(x, dict):
                            nd = by_fid.get(x.get("fid")) or (
                                x if x.get("centerline") else None)
                        if nd is not None and nd not in out_n:
                            out_n.append(nd)
                return out_n

            rows = _BroadRows()
            _n_end = 0
            _no_line: list = []
            for _el in _e_refs:
                _nodes = _e_nodes(_el)
                if not _nodes:
                    raise ProgramError(
                        f"ends ref {_el!r} names no row on this surface. "
                        f"{_avail('point')}{_sig_suffix()}")
                for _nd in _nodes:
                    _cl = _nd.get("centerline")
                    if not (isinstance(_cl, list) and len(_cl) >= 2):
                        _no_line.append(
                            f"{_nd.get('fid')}"
                            + (f" ({_nd.get('name')})"
                               if _nd.get("name") else ""))
                        continue
                    _lab = str(_nd.get("name") or _nd.get("fid"))
                    for _which, _pt in (("end0", _cl[0]), ("end1", _cl[-1])):
                        _xy = [float(round(_pt[0])), float(round(_pt[1]))]
                        rows.append({"fid": f"end_{_n_end}", "xy": _xy,
                                     "name": f"{_lab} {_which}"})
                        _n_end += 1
            if not rows:
                # LOUD, never an empty bind -- and a ROW fact, never a
                # class verdict: the model was told "a road has [a
                # centerline] and a footprint does not" about rows that
                # WERE roads, whose centerlines this surface simply does
                # not record.
                _named = ", ".join(_no_line[:4]) + (
                    f" (+{len(_no_line) - 4} more)"
                    if len(_no_line) > 4 else "")
                raise ProgramError(
                    f"none of those rows carries a recorded centerline -- "
                    f"{_named} have no line geometry on this surface, so "
                    f"no termini exist for them. ends reads a centerline; "
                    f"end0_xy/end1_xy on a row is the same fact served as "
                    f"columns.{_sig_suffix()}")
            rows.ordered = True
            rows.notes = [
                f"{len(rows)} terminus rows; each is its OWN row with its "
                f"own fid, so index the one you mean."]
            r = rows
        elif name == "score_by":
            # THE sort (fully agent-weighted). Sign-aware, KEY-AWARE-normalized
            # composite over EXACTLY the columns the agent's `emphasis` names --
            # nothing is auto-added, so the ordering is precisely the criteria
            # the agent stated (a single-key emphasis IS the one-key order).
            items = as_list(a["items"])
            # CLEAN-CUTOVER fail-loud (adversarial-verify gap): keys/weights are DELETED.
            # The live serve grammar (additionalProperties:False) blocks them, but a legacy
            # program or a frozen resume source hits the un-enforced offline/replay
            # path -- error instead of silently re-shaping the sort it asked for.
            if a.get("keys") is not None or a.get("weights") is not None:
                raise ProgramError(
                    "score_by takes no keys/weights -- it orders by EXACTLY the columns "
                    "you weight via emphasis:[{key:'<column>', w:<signed number>}]; name "
                    "every discriminator that should count and its sign.")
            emphasis_raw = a.get("emphasis")
            if not isinstance(emphasis_raw, list) or not emphasis_raw:
                raise ProgramError(
                    "score_by needs a NON-EMPTY emphasis list [{key:'<column>', w:<signed "
                    "number>}] -- it orders by EXACTLY the columns you weight (nothing is "
                    "auto-added), so the sort has no criteria until you name them. w's "
                    "SIGN is the direction (NEGATIVE=lower-raw-better e.g. a 'distance to <ref>', "
                    "POSITIVE=higher-raw-better e.g. a color_<word> column). "
                    f"{_present_numeric_keys_hint(items)}")
            _sb_notes: list = []
            emphasis: dict = {}
            for _el in emphasis_raw:
                if isinstance(_el, dict) and _el.get("abs") is not None:
                    raise ProgramError(
                        "emphasis 'abs' is retired: side columns are "
                        "presence pairs (the column present names the "
                        "side) and the either-side magnitude is its own "
                        "column -- sort 'distance to <ref>' instead.")
                if (not isinstance(_el, dict) or not isinstance(_el.get("key"), str)
                        or not isinstance(_el.get("w"), (int, float))
                        or isinstance(_el.get("w"), bool)):
                    raise ProgramError(
                        "score_by emphasis element must be {key:'<column>', w:<signed "
                        "number>} -- w's SIGN is the direction (NEGATIVE="
                        f"lower-raw-better e.g. a 'distance to <ref>', POSITIVE=higher-raw-better "
                        f"e.g. a color_<word> column); got {_el!r}.")
                _reject_renamed_key(_el["key"])
                _k = _el["key"]
                if _is_color_column_key(_k):
                    # THE WORD RIDES IN THE KEY: resolve it onto the closed
                    # palette here so the emphasis reads the canonical column
                    # the attach mints (color_navy weights color_blue; noted).
                    _canon, _cnote = _resolve_color_word(_k[len("color_"):])
                    if _cnote is not None:
                        _sb_notes.append(_cnote)
                    _k = "color_" + _canon
                emphasis[_k] = float(_el["w"])
            items = _auto_attach(items, list(emphasis), notes=_sb_notes)
            # An EMPHASIS key NO row carries would be a SILENT constant term (score_by
            # treats a missing key as the neutral midpoint) -- a typo'd/never-attached
            # column would just vanish from the sort. This fires AFTER the auto-attach
            # above, i.e. only when the key genuinely resolved to nothing.
            # The FACT, in neither channel's spelling: this raises for the
            # sentence surface AND the frozen round-era one, and `score_by
            # emphasis key` is a call the sentence channel cannot form. The
            # model read it as proof its own sentence was malformed ("the tool
            # is interpreting my input `the Brown one.` literally ... because I
            # forgot proper syntax?") and repeated that sentence verbatim after
            # 334 of 503 serves. The COLUMN stays named: it is a row header the
            # model already reads, it disambiguates a multi-key emphasis, and
            # the !r is load-bearing -- _err_class_sig collapses quoted tokens,
            # so an unquoted key would split one loop class into many.
            _dropped_color: list = []
            for _k in list(emphasis):
                if items and not any(isinstance(it.get(_k), (int, float))
                                     for it in items if isinstance(it, dict)):
                    if _is_color_column_key(_k):
                        # AN UNSEEN COLOUR IS NOT A BROKEN SORT WHEN THE
                        # SORT HAS OTHER TERMS. Nothing carries a perceived
                        # colour until a view has passed over it, so on a
                        # pre-takeoff surface this key is absent BY
                        # CONSTRUCTION -- and raising kills the whole block
                        # (``reasoner_loop`` breaks the segment on the first
                        # error), taking down the ``look_for`` that would
                        # have armed the segmenter. The model is then unable
                        # to ever see the thing whose colour it asked about,
                        # from one legal call. So the term is DROPPED, said
                        # out loud, and the remaining keys still rank.
                        #
                        # ALONE it is still a raise: a sort with no term
                        # left is not an ordering, and serving one as though
                        # it were is the silent failure this tree keeps
                        # paying for.
                        if len(emphasis) > 1:
                            _dropped_color.append(_k)
                            del emphasis[_k]
                            continue
                        raise ProgramError(
                            f"nothing orders by {_k!r} here: no row carries a "
                            f"perceived color, so no color column attaches (rows "
                            f"without a color read omit color_<word> columns). "
                            f"{_present_numeric_keys_hint(items)}")
                    # remedy keyed to the MISSING key's family (the contact
                    # fact first — the compare RAN there), + the live census.
                    raise ProgramError(
                        f"nothing orders by {_k!r} here: no row carries a "
                        f"numeric {_k!r}. "
                        f"{_contact_fact(_k, items) or _key_remedy(_k)}"
                        f"{_present_numeric_keys_hint(items)}")
            for _k in _dropped_color:
                # NEVER SILENT. The rows are ranked by fewer things than the
                # model asked for, and it has to know which -- and that the
                # remedy is to look, not to re-spell.
                _sb_notes.append(
                    f"{_k!r} ranked nothing here: no row on this surface has "
                    f"been SEEN yet, so none carries a perceived colour. The "
                    f"other keys still rank. A row gets a colour once a "
                    f"flight has passed over it.")
            keys = sorted(emphasis)  # deterministic; the sum is order-independent
            weights = [emphasis[k] for k in keys]
            for _k, _w in zip(keys, weights):
                # match-scored columns are higher-is-better BY MEANING; a
                # negative w on one sorts worst-match-first. A fact, stated.
                if (_k == "name_match" or _is_color_column_key(_k)) and _w < 0:
                    _sb_notes.append(
                        f"emphasis w on '{_k}' INVERTS its natural direction "
                        f"(higher-raw-is-better): rank-0 is now the WORST "
                        f"{_k} match. If you meant best-first, flip the "
                        f"sign.")
                # PARITY with the match-key note above, for the distance
                # family: a distance/chart column is lower-raw-nearest by
                # meaning, so a POSITIVE w sorts it worst-first. A stated
                # fact, never a rejection — 'far from X' is legitimate intent.
                if _k.startswith(("distance to ", "center distance to ")) \
                        and _w > 0:
                    _sb_notes.append(
                        f"emphasis w on '{_k}' INVERTS its natural direction "
                        f"(lower-raw-is-nearer): rank-0 is now the LARGEST "
                        f"{_k} (farthest/highest-first -- 'far from X' is a "
                        f"legitimate read). If you meant nearest-first, flip "
                        f"the sign.")
                elif _k.startswith("along ") and _w > 0:
                    _sb_notes.append(
                        f"emphasis w:+ on '{_k}' puts rank-0 at the LARGEST "
                        f"arc coordinate (farthest along the axis from its 0 "
                        f"end); w:- ranks the 0-end first. 'along' is an axis "
                        f"coordinate, not a distance to the ref.")
            if any(k.startswith(("bearing from ", "facing "))
                   for k in emphasis):
                # a stated FACT about the linear fuse over a circular scale —
                # never a harness re-interpretation of the agent's weights
                # (no circular-mean scoring).
                _sb_notes.append(
                    "bearings are compass-circular: 359 and 1 are 2 degrees "
                    "apart but rank at opposite ends of a linear sort; "
                    "direction cuts are filter_items ranges or the side "
                    "columns in a {toward} frame.")
            _sorted_rows = G.score_by(items, keys, weights)
            r = _CueSortedRows(_sorted_rows)
            r.op_cols = list(keys)
            for k in list(keys):
                m = re.match(r"^(?:left of|right of|ahead of|behind) (.+)$",
                             str(k))
                if m:
                    # a side key names its whole presence-pair family: the
                    # sibling columns must survive the budget beside it.
                    r.op_cols.extend(
                        f"{p} {m.group(1)}"
                        for p in ("left of", "right of", "ahead of", "behind")
                        if f"{p} {m.group(1)}" not in r.op_cols)
            if _sb_notes:
                r.notes = _sb_notes
        elif name == "filter_items":
            items = as_list(a["items"]); key = a["key"]; opr = a.get("op", "eq"); val = a.get("value")
            # A filter PRESERVES its input's order: a cut over an ordered set
            # stays ordered (and renderable); a cut over a browse stays not.
            _in_ordered = bool(getattr(items, "cue_sorted", False)
                               or getattr(items, "ordered", False))
            # FILTER = CLASS + SPATIAL ONLY. Color is an ORDERING signal, never a
            # filter: a hard color cut drops a correctly-detected target whose
            # perceived color reads a hair off the cue (nadir chroma-collapse makes
            # the read unreliable). Reject a cut on ANY color key (the color/
            # color_conf fields and every color_<word> column) and route the agent
            # to the ordering path, which deprioritises by color without dropping
            # the GT.
            if key == "color" or (
                    isinstance(key, str) and key.startswith("color_")):
                raise ProgramError(
                    "color is an ordering signal, not a filter. Color rides "
                    f"the named columns color_<word> (closed palette: "
                    f"{', '.join(G.COLOR_ENUM)}); weight one in score_by's "
                    "emphasis (with your relation columns) to order the "
                    "cue-matching rows to the top without dropping any.")
            _reject_renamed_key(key)
            items = _auto_attach(items, [key])
            _LEGAL_FILTER_OPS = FILTER_OPS
            if opr not in _LEGAL_FILTER_OPS:
                # Previously fell through to .get(opr, False) -> a SILENT dead-filter
                # that zeroed the result and hid real matches (e.g. the on-street /
                # near-row abs_le filter the teacher reaches for). Fail loud instead.
                raise ProgramError(
                    f"unknown filter_items op {opr!r}; legal: {list(_LEGAL_FILTER_OPS)}")
            # ``on_side`` compares against the column's own SIGN, so like
            # ``truthy`` it carries no value of its own.
            if opr not in ("truthy", "on_side") and val is None:
                # Every comparison op takes an explicit 'value'; with it
                # omitted, lt/le/gt/ge would TypeError against None and the
                # default eq would run v == None — a silent dead-filter
                # (len:0 with no error, exactly the class the absent-key
                # guards kill). Fail loud, naming the op and the missing arg.
                raise ProgramError(
                    f"filter_items op {opr!r} takes an explicit 'value' to "
                    f"compare {key!r} against, and none was given. A "
                    f"key-presence test is op:'truthy' (no value)."
                    f"{_sig_suffix()}")
            _NUMERIC_OPS = ("lt", "le", "gt", "ge")
            if opr in _NUMERIC_OPS and not (isinstance(val, (int, float))
                                            and not isinstance(val, bool)):
                raise ProgramError(
                    f"filter_items op {opr!r} needs a NUMERIC 'value'; got "
                    f"{type(val).__name__} {val!r}.{_sig_suffix()}")
            if opr in _NUMERIC_OPS:
                bad = next((it.get(key) for it in items
                            if it.get(key) is not None
                            and not isinstance(it.get(key), (int, float))), None)
                if bad is not None:
                    raise ProgramError(
                        f"filter_items op {opr!r} needs a SCALAR key, but {key!r} holds a "
                        f"non-scalar ({type(bad).__name__}, e.g. an xy pair). Scalar "
                        f"fields: a compare column ('distance to <ref>' / "
                        f"'along <ref>' / a side column) or footprint_area_m2. "
                        f"A compass word has no ordering -- 'facing <ref>' "
                        f"answers eq/truthy, not gt/le. Shape is still a "
                        f"looking question: look at the rows for that.")
                # Absent-key guard: a numeric op on a key NO item carries would
                # be a SILENT dead-filter (v=None -> every predicate False ->
                # len:0), e.g. filter_items{key:'d_x'} on a bare find list (d_*
                # is attached by `compare`, not find) -> the agent gets 0
                # results, no error, and falls back to hand-math. Fail loud + actionable.
                if items and not any(isinstance(it.get(key), (int, float)) for it in items):
                    raise ProgramError(
                        f"filter_items op {opr!r} on key {key!r}: no item "
                        f"carries a numeric {key!r}. "
                        f"{_contact_fact(key, items) or _key_remedy(key)}"
                        f"{_present_numeric_keys_hint(items)}")
            elif (items
                  and not any(isinstance(it, dict) and key in it for it in items)
                  and not _relate_ran(key, items)):
                # Absent-key guard for the equality/truthy ops: eq/ne/truthy on
                # a column NO item carries is a SILENT dead-filter (eq -> len:0,
                # ne/truthy -> a meaningless pass-all/none) -- the model GUESSED
                # a column name, or the column is honestly absent everywhere
                # (brg_gap_ on an all-contact pool). Fail loud with the
                # missing key's OWN producing idiom + the live census.
                #
                # UNLESS THE RELATE RAN. A `facing <ref>` absent on EVERY row
                # because every row TOUCHES <ref> is a geometric fact, not a
                # guessed name, and `in_or_blank` answers it correctly by
                # keeping them all. `_contact_fact` already separates the two
                # cases but only for a bare-fid ref; the sibling column is the
                # discriminator that also works for an `as` label.
                raise ProgramError(
                    f"filter_items op {opr!r} on key {key!r}: no item carries "
                    f"{key!r}. "
                    f"{_contact_fact(key, items) or _key_remedy(key)}"
                    f"{_present_keys_hint(items)}")

            def _ok(it):
                # LAZY per-op predicates: a dict LITERAL would evaluate every
                # branch eagerly, so op 'truthy' (value omitted -> val=None)
                # would raise TypeError ('True < None') whenever ANY item
                # carried the key. Only the selected predicate runs.
                v = it.get(key)
                av = abs(v) if isinstance(v, (int, float)) else None
                return {"eq": lambda: v == val, "ne": lambda: v != val,
                        "lt": lambda: v is not None and v < val,
                        "le": lambda: v is not None and v <= val,
                        "gt": lambda: v is not None and v > val,
                        "ge": lambda: v is not None and v >= val,
                        "truthy": lambda: bool(v),
                        # absent = unplaced, KEPT; negative = measurably the
                        # other side, dropped. See FILTER_OPS.
                        "on_side": lambda: v is None or (
                            isinstance(v, (int, float))
                            and not isinstance(v, bool) and v > 0),
                        # `on_side` for a WORD column, and the reason it
                        # takes a LIST rather than a value: a compass word
                        # COMPOSES, so "facing south" is satisfied by
                        # south, southeast and southwest, and the caller
                        # passes the set that counts. The op layer stays a
                        # comparison layer -- which words compose is the
                        # compiler's question, beside COMPASS_UNIT.
                        #
                        # And the blank is KEPT: `facing <ref>` is absent
                        # exactly at contact (no min-gap vector to take a
                        # bearing from), and a house touching the road it
                        # fronts is the COMMON case for "the south-facing
                        # house on Viburnum Avenue". Plain equality
                        # compares None to the word and drops it, refuting
                        # a row that was never measured -- unmeasured is
                        # never refuted, on a word column for the same
                        # reason it is not on a signed one.
                        "in_or_blank": lambda: v is None or v in val,
                        }[opr]()
            if opr == "in_or_blank":
                # THE SAME THREE-WAY STATEMENT the side cut makes, for a
                # word column: rows that read the word, rows that read a
                # DIFFERENT word (dropped), and rows the geometry never
                # measured (kept). Without it a facing cut that kept only
                # blanks would look like a cut that found matches.
                _on = [it for it in items if _ok(it)]
                _blank = sum(1 for it in _on
                             if (it.get(key) if isinstance(it, dict)
                                 else None) is None)
                _want = list(val) if isinstance(val, (list, tuple)) else [val]
                _eq_stats = {
                    "cut": f"{len(items)}->{len(_on)}",
                    "counts_as": _want,
                    "reads_one_of_them": len(_on) - _blank,
                    "reads_another_word_dropped": len(items) - len(_on),
                    "unmeasured_kept": _blank,
                }
                if not (len(_on) - _blank):
                    _eq_stats["nothing_reads_those_words"] = (
                        f"no surveyed row reads any of {_want} in {key!r}; "
                        f"the rows kept are the ones the geometry never "
                        f"measured (a row TOUCHING the reference has no "
                        f"facing word). Widen, or read the column and "
                        f"see which words are actually there.")
                r = _FilterCutRows(_on)
                r.cut_stats = _eq_stats
            elif opr == "on_side":
                # THE SIDE CUT STATES ITSELF. The three-way outcome is the
                # whole point of this op and no other cut has it: rows ON the
                # named side, rows measurably on the OTHER side (dropped),
                # and rows the geometry could not place (kept, unmeasured).
                #
                # This lives HERE and not in the reasoner loop because the
                # loop only ever sees the SURVIVORS. A note written there
                # could never fire -- survivors are positive by construction,
                # so it skipped; and when nothing survived the list was empty,
                # so it skipped again. Measured: 0 firings in 194 episodes,
                # exactly the present-but-always-empty failure this repo has
                # a rule about. The count only exists at the moment of the
                # cut.
                _on = [it for it in items if _ok(it)]
                _blank = sum(1 for it in _on
                             if not isinstance((it.get(key)
                                                if isinstance(it, dict)
                                                else None), (int, float)))
                _side_stats = {
                    "cut": f"{len(items)}->{len(_on)}",
                    "on_side": len(_on) - _blank,
                    "other_side_dropped": len(items) - len(_on),
                    "no_side_kept": _blank,
                }
                if not (len(_on) - _blank):
                    # The fact that changes what the model should do next:
                    # nothing SURVEYED is on that side, so what is there has
                    # to be seen rather than looked up.
                    _side_stats["nothing_on_that_side"] = (
                        "the prior map holds nothing on that side -- what is "
                        "there has to be SEEN"
                        + ("" if withheld("look_for") else
                           ": look_for the class, then explore the place to "
                           "segment it"))
                r = _FilterCutRows(_on)
                r.cut_stats = _side_stats
            elif opr in _NUMERIC_OPS:
                # Cut-stats: ONE deterministic info line on
                # every NUMERIC cut -- how much was cut, the nearest-EXCLUDED
                # row, and the (inclusive) threshold that would admit it, plus
                # how many rows that relaxation admits. INFORMATION-ONLY: the
                # MODEL reads it and decides whether to widen / rank instead;
                # the harness never relaxes a cut itself (no soft-admit band,
                # no election). Rendered by compact_results as ``cut_stats``
                # (and folded into the loud hint on an EMPTY result).
                kept: list = []
                excluded: list = []
                for it in items:
                    (kept if _ok(it) else excluded).append(it)
                _absv = opr.startswith("abs_")

                def _eff(it):
                    v = it.get(key) if isinstance(it, dict) else None
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        return None
                    return abs(v) if _absv else v

                stats: dict = {"cut": f"{len(items)}->{len(kept)}"}
                _excl_vals = [(e, it) for it in excluded
                              if (e := _eff(it)) is not None]
                if _excl_vals:
                    # "Nearest" = the excluded row closest to admission: the
                    # smallest value for an upper-bound cut (lt/le), the
                    # largest for a lower-bound cut (gt/ge).
                    _lower_near = opr in ("lt", "le")
                    ne_v, ne_it = (min if _lower_near else max)(
                        _excl_vals, key=lambda t: t[0])
                    stats["nearest_excluded"] = {
                        "fid": str((ne_it.get("fid") if isinstance(ne_it, dict)
                                    else "") or "?"),
                        key: round(float(ne_it.get(key)), 3),
                    }
                    stats["admit_at"] = round(float(ne_v), 3)
                    stats["n_at_admit"] = sum(
                        1 for it in items
                        if (v := _eff(it)) is not None
                        and (v <= ne_v if _lower_near else v >= ne_v))
                r = _FilterCutRows(kept)
                r.cut_stats = stats
            else:
                r = _FilterCutRows(it for it in items if _ok(it))
            r.ordered = _in_ordered
            r.op_cols = [key]
            _m = re.match(r"^(?:left of|right of|ahead of|behind) (.+)$",
                          str(key))
            if _m:
                r.op_cols.extend(
                    f"{p} {_m.group(1)}"
                    for p in ("left of", "right of", "ahead of", "behind")
                    if f"{p} {_m.group(1)}" != key)
        elif name == "stratum":
            # THE MATCH TIERS AS SETS. A query find scores every row into
            # the retrieval's own tiers (text_match: name tier, exact-tag
            # tier, the word-share bag below), and until this op the tiers
            # were a float column the grid does not render: the 140#138
            # endgame took [0] of a pool where six detections at 1.0 sat
            # over tag priors at 0.8, with no way to SAY "the ones that
            # ARE the thing". r1.exact / r1.tagged / r1.fuzzy compile
            # here. A VIEW, never a re-retrieval: input order preserved,
            # nothing else dropped, the full set stays live behind its
            # own handle.
            items = as_list(a["items"])
            tier = str(a["tier"]).strip().lower()
            if tier not in ("exact", "tagged", "fuzzy"):
                raise ProgramError(
                    f"unknown stratum {tier!r}; the tiers are exact / "
                    "tagged / fuzzy")
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
                v = (it.get("name_match")
                     if isinstance(it, dict) else None)
                if not isinstance(v, (int, float)):
                    return False
                return ((_lo is None or v >= _lo)
                        and (_hi is None or v < _hi))
            r = _FilterCutRows(it for it in items if _in_tier(it))
            r.ordered = bool(getattr(items, "cue_sorted", False)
                             or getattr(items, "ordered", False))
        elif name == "compare":
            # compare — THE relation/arrangement attach. ATTACHES evidence
            # columns onto every item row, PASSING the input row's fields
            # THROUGH (kill-the-whitelist: a score_by key survives to the
            # downstream sort). Per ref — the SAME four columns for EVERY ref
            # whatever its mask shape (no linearity type-switch): d_<ref>
            # (footprint gap, 0.0 on touch/overlap), brg_gap_<ref> (compass
            # bearing of that min-gap vector ref->item; ABSENT on contact,
            # where no local direction exists), dc_<ref> (centroid-to-centroid
            # distance), brg_<ref> (centroid->centroid compass bearing, absent
            # only when the centroids coincide). frame={reference:'auto' |
            # along[,toward] | toward} is THE view aspect: G.frame fixes the
            # view in code (viewer/auto; 'relative' survives for offline/lab
            # programs) or G.frame_polyline charts the pool along a
            # road/polyline — view coords become column
            # reads. DELEGATE-ONLY: every number comes
            # from a geom_compute primitive; this branch never touches raw
            # coordinates. An ATTACH, not a sort: input rows + order
            # preserved; a compare TERMINAL gets the out-loud warning
            # (terminate in score_by).
            items = as_points(a["items"])
            # Audit-geometry capture (geometry_sink only): filled at the
            # sites below where the op resolves geometry it does not return.
            _geo = None if geometry_sink is None else {
                "ref": ref, "op": "compare", "chart": None, "frame": None,
                "linear_refs": [], "ref_centers": [], "gaps": []}
            # Order propagation: compare preserves input order, so its result
            # is ordered iff its input was (as_points builds a fresh list, so
            # read the marker off the resolved source).
            _cmp_src = _unwrap_listish(_raw(a["items"]))
            _in_ordered = bool(getattr(_cmp_src, "cue_sorted", False)
                               or getattr(_cmp_src, "ordered", False))
            # Deleted aspects redirect loudly to the ONE arrangement aspect.
            for _dead in ("view", "between", "neighbors"):
                if a.get(_dead) is not None:
                    raise ProgramError(
                        f"compare {_dead!r} is not an aspect. "
                        + _DELETED_OPS.get(_dead, _FRAME_FORMS)
                        + _sig_suffix())
            refs_arg = a.get("refs")  # optional: the frame aspect needs none
            if not isinstance(refs_arg, list) or not refs_arg:
                # A refs-less compare is legal whenever it carries the frame
                # aspect (it computes its own columns from its OWN args); refs
                # is only required for the per-ref relation columns.
                if a.get("frame") is not None:
                    refs_arg = []
                else:
                    raise ProgramError(
                        "compare needs a non-empty 'refs' LIST (each element a "
                        "bare fid, an rN single row, an [x,y] literal, or "
                        "{ref:..., as:'label'}) OR the frame aspect -- it "
                        "needs no refs. " + _FRAME_FORMS + _sig_suffix())
            # name_match is QUERY-FIND provenance only (the fuzzy NAME-match
            # grounding trust): a composite `score` from score_by/other chains
            # must not masquerade as it. A name is not a place -- name_match
            # says the NAME matched, never WHICH instance the cue means.
            _query_find_refs = {
                o.get("ref") for o in _prog_ops
                if isinstance(o, dict) and o.get("op") == "find"
                and isinstance(o.get("args"), dict) and o["args"].get("query")
            }

            def _ref_base_of_token(token):
                """The bare ref base of a token -- sigil-tolerant (a leading
                ``$`` is a slip) and path-stripped (``r2.members[0]`` -> ``r2``,
                bare ``road_3`` -> ``road_3``)."""
                t = str(token)
                t = t[1:] if t.startswith("$") else t
                m = re.match(r"^([^.\[\]]+)", t)
                return m.group(1) if m else t

            def _is_query_find_token(token):
                return _ref_base_of_token(token) in _query_find_refs

            def _sanitize_label(lbl, fallback="ref"):
                # the ONE naming-law base (module-level _sanitize_token)
                return _sanitize_token(lbl, fallback)

            def _node_label(node):
                """The node-derived column label: the FID. Deriving from the
                display NAME made the model guess wrong ~12x/6 eps
                (refs:["parking_4"] produced d_parking_lot_4); the FID is
                (a) exactly what the model types, (b) identical whether the
                ref was a bare fid or an rN that resolved to it (dialect
                stability preserved), (c) round-stable. The display name
                still shows in the result HEADER (ref_name)."""
                return _sanitize_label(node["fid"])

            def _set_label(arg_token, fallback):
                """A short TOKEN-derived label for a SET/axis arg
                (between/neighbors/view): 'r1' -> 'r1', 'cands' ->
                'cands', a bare fid -> the fid, an inline list -> the
                fallback. Sigil-tolerant on a '$' slip."""
                if isinstance(arg_token, str):
                    return _sanitize_label(_ref_base_of_token(arg_token), fallback)
                return fallback

            def _is_xy_literal(v):
                return (isinstance(v, (list, tuple)) and len(v) == 2
                        and all(isinstance(z, (int, float)) for z in v))

            def _arg_row(arg):
                """The ONE surface row an aggregate arg (view reference /
                between a|b / neighbors pool) resolves to, else None: a
                bare fid, a {fid} dict, a one-element fid list, or an rN
                whose value unwraps to a single by_fid row. A multi-row
                set, constructed geometry, or [x,y] literal has no node."""
                v = arg
                if _is_ref_token(v):
                    try:
                        v = _resolve_ref(v)
                    except ProgramError:
                        return None
                # unwrap the compacted {len, items} echo for INLINE literals
                # too, not just ref-resolved values -- the value coercers
                # (as_point/as_points) unwrap it, so the label
                # must follow the same resolution or a pasted one-row echo
                # computes from the node yet labels from the fallback token.
                v = _unwrap_listish(v)
                if isinstance(v, list) and len(v) == 1 and not _is_xy_literal(v):
                    v = v[0]
                if isinstance(v, str):
                    # a resolved value / a bare fid -- never a live ref name here
                    return by_fid.get(v)
                if isinstance(v, dict) and isinstance(v.get("fid"), str):
                    return by_fid.get(v["fid"])
                return None

            def _set_label_node(arg_token, fallback):
                """(label, node) for an aggregate arg -- the FID-label rule
                EXTENDED to view/between/neighbors: an arg
                resolving to ONE surface row labels from the resolved NODE
                (_node_label -- the SAME column whether passed as a bare
                fid or an rN, so vx_road_1 never becomes vx_r0 the next
                round); anything else (a multi-row set, constructed
                geometry, an inline literal) keeps the token-derived
                label, node=None."""
                node = _arg_row(arg_token)
                if node is not None:
                    return _node_label(node), node
                return _set_label(arg_token, fallback), None

            # --- element-wise ref resolution ---------------------------------
            # Each spec is (kind, payload, label_base, node|None, name_match):
            #   kind 'row'   -> payload = the canonical surface row
            #   kind 'point' -> payload = [x, y] (a constructed literal)
            #   kind 'line'  -> payload = centerline polyline (constructed axis)
            ref_specs: list = []
            _standing_ref_handles: list = []
            # Bare (label-less) [x,y] literal refs all label 'xy': one is the
            # documented _xy family; a second would silently collide on the
            # same column names. Loud ambiguity error demanding as: labels.
            _bare_xy_refs = 0
            for el in refs_arg:
                as_label = None
                ref_el = el
                if isinstance(el, dict):
                    if "ref" not in el:
                        raise ProgramError(
                            "compare refs object element must be {ref: <fid|rN|"
                            f"[x,y]>, as: 'label'}}; got keys {sorted(el)}.{_sig_suffix()}")
                    if el.get("as"):
                        as_label = _sanitize_label(el["as"])
                    ref_el = el["ref"]
                    if (isinstance(ref_el, (list, tuple)) and ref_el
                            and all(isinstance(x, str) for x in ref_el)):
                        # UNION ref {ref:[fid|rN,...], as:'label'}: the union
                        # of the named masks as ONE ref -- d_<label> is the
                        # exact min gap over the members (witness from the
                        # argmin member, so brg_gap_<label> is that member's
                        # local direction), dc_/brg_<label> read from the
                        # union's area-weighted centroid.
                        #
                        # A MEMBER MAY BE AN rN, and that is what makes the
                        # union reachable at all. Spelling it needs the fids,
                        # and a set a program MINTED does not have any until
                        # it runs -- so "rank these houses by their nearest
                        # pool" was expressible only by hand-typing every
                        # pool fid. Fanning the rN out instead measures each
                        # instance separately and dies on the token guard the
                        # moment either side is large: 3,414 houses against
                        # 40 pools is 136,560 cells. One label can only mean
                        # one column, and the column a set folds to is the
                        # gap to its NEAREST member.
                        if not as_label:
                            raise ProgramError(
                                "a union ref {ref:[fid|rN,...]} needs its "
                                "as:'label' -- the ONE column base its "
                                f"distance/bearing columns carry."
                                f"{_sig_suffix()}")
                        _members = []
                        for x in ref_el:
                            if _is_ref_token(x):
                                _rows = _unwrap_listish(_resolve_ref(x))
                                if isinstance(_rows, dict):
                                    _rows = [_rows]
                                # A CONSTRUCTED row (a crossing, a terminus)
                                # is not in by_fid, and it is still a place:
                                # it rides on its own xy, exactly as the
                                # single-ref path's point branch does. The
                                # by_fid-only filter served "inter is empty"
                                # while len(inter)==1 printed true.
                                _got = [by_fid.get(r["fid"], r)
                                        for r in _rows
                                        if isinstance(r, dict)
                                        and (r.get("fid") in by_fid
                                             or _is_xy_literal(r.get("xy")))]
                                if not _got:
                                    # NAME THE HANDLE THE MODEL WROTE, not
                                    # the executor ref it compiled to, and
                                    # name why it is usually empty. The
                                    # common case by far is a detection
                                    # class asked for before anything has
                                    # looked: nothing is segmented until
                                    # look_for names the class AND a flight
                                    # goes over the place, so find("pool")
                                    # on turn 1 is empty by construction and
                                    # a sort against it could not mean
                                    # anything.
                                    _lbl = as_label or _sanitize_label(x)
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
                                        + f"{_sig_suffix()}")
                                # A CAP, because the old guard is gone. A
                                # union used to need hand-typed fids, which
                                # bounded it by what a model would spell; an
                                # rN member bounds it by nothing, and the
                                # union comment below records a members x
                                # items case that "ran 885 s without
                                # returning". The bbox prune should hold, but
                                # a wall-clock cliff inside a 500-episode
                                # fleet run is not the place to find out.
                                if len(_got) > _UNION_MAX_MEMBERS:
                                    raise ProgramError(
                                        f"{as_label or _sanitize_label(x)} "
                                        f"holds {len(_got)} rows and a union "
                                        f"folds at most "
                                        f"{_UNION_MAX_MEMBERS}: measuring to "
                                        f"the nearest of that many is a "
                                        f"whole-map scan per row. Narrow it "
                                        f"first (inside=, a cardinal, or a "
                                        f"tighter query), then measure to "
                                        f"what is left.{_sig_suffix()}")
                                _members.extend(_got)
                                continue
                            _mnode = by_fid.get(x)
                            if _mnode is None:
                                raise ProgramError(
                                    f"union ref member {x!r} is not a fid on "
                                    f"this surface. {_avail('point')}"
                                    f"{_sig_suffix()}")
                            _members.append(_mnode)
                        ref_specs.append(("union", _members, as_label,
                                          None, None))
                        continue
                if isinstance(ref_el, str):
                    # a point literal the model wrapped in a JSON string
                    # ('[347.2, 519.1]') -> coerce into the [x,y] branch below.
                    _xy = re.match(r"^\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*"
                                   r"(-?\d+(?:\.\d+)?)\s*\]\s*$", ref_el)
                    if _xy:
                        ref_el = [float(_xy.group(1)), float(_xy.group(2))]
                if _is_xy_literal(ref_el):
                    if not as_label:
                        _bare_xy_refs += 1
                        if _bare_xy_refs > 1:
                            raise ProgramError(
                                "two bare [x,y] literal refs in one compare "
                                "are ambiguous: each would attach the SAME "
                                "'distance to xy'/'facing xy'/"
                                "'center distance to xy'/'bearing from xy' "
                                "columns. Give "
                                "each literal its own label -- {ref:[x,y], "
                                "as:'label'} names its columns 'distance to "
                                "<label>'/'facing <label>'/"
                                "'center distance to <label>'/'bearing from "
                                "<label>'."
                                + _sig_suffix())
                    ref_specs.append(("point", [float(ref_el[0]), float(ref_el[1])],
                                      as_label or "xy", None, None))
                    continue
                if not isinstance(ref_el, str):
                    raise ProgramError(
                        f"compare refs element {ref_el!r} must be a bare fid string, "
                        f"an rN result, an [x,y] literal, or {{ref:..., as:...}}."
                        f"{_sig_suffix()}")
                if not _is_ref_token(ref_el):
                    # a bare fid (never an rN result name -- those route to the
                    # resolver below): a live surface row, else the loud miss.
                    node = by_fid.get(ref_el)
                    if node is None and ref_el in staged:
                        # a staged viewpoint handle: the natural "score these
                        # rows against my stance" call -- attach the stance's
                        # OWN columns ('left of/right of/ahead of/behind vN'
                        # + 'distance to vN'), not the point family
                        # (framestage_1000: the model asked this way and got
                        # bearings it could not use).
                        if ref_el not in _standing_ref_handles:
                            _standing_ref_handles.append(ref_el)
                        continue
                    if node is None:
                        raise ProgramError(
                            f"compare ref {ref_el!r} is not a fid on this "
                            f"surface. {_avail('point')}{_sig_suffix()}")
                    ref_specs.append(("row", node, as_label, node, None))
                    continue
                v = _unwrap_listish(_resolve_ref(ref_el))
                if isinstance(v, list) and len(v) == 1 and not _is_xy_literal(v):
                    v = v[0]
                # STABLE column naming: a ref that resolves to a SURFACE ROW is
                # labeled from the resolved node's FID — the SAME column whether the
                # model passed the bare fid or an rN that resolved to it, and
                # exactly the token the model types (d_road_1, never
                # d_earl_street and never d_r0). Constructed geometry (an
                # [x,y] literal, a fitted axis) has no node and keeps the
                # token-derived label (r0 -> d_r0). An as:'label' always
                # overrides.
                tok_base = _sanitize_label(_ref_base_of_token(ref_el))
                if isinstance(v, str) and v in by_fid:
                    ref_specs.append(("row", by_fid[v], as_label,
                                      by_fid[v], None))
                    continue
                if isinstance(v, dict) and v.get("fid") in by_fid:
                    ms = v.get("name_match")
                    ref_specs.append((
                        "row", by_fid[v["fid"]], as_label,
                        by_fid[v["fid"]],
                        ms if (isinstance(ms, (int, float))
                               and _is_query_find_token(ref_el)) else None,
                    ))
                    continue
                if isinstance(v, dict) and isinstance(v.get("centerline"), list) \
                        and len(v["centerline"]) >= 2:
                    # a constructed axis (row_centerline_fit result)
                    ref_specs.append(("line", v["centerline"],
                                      as_label or tok_base, None, None))
                    continue
                if isinstance(v, dict) and _is_xy_literal(v.get("xy")):
                    ref_specs.append(("point", [float(v["xy"][0]), float(v["xy"][1])],
                                      as_label or tok_base, None, None))
                    continue
                if _is_xy_literal(v):
                    ref_specs.append(("point", [float(v[0]), float(v[1])],
                                      as_label or tok_base, None, None))
                    continue
                # MULTI-INSTANCE LANDMARK -> one column set PER instance (NO cap).
                # A ref that resolves to several surface rows (a named landmark's
                # instances -- a duplicate name, a fragmented road) fans out into
                # one column set PER instance, reusing the per-ref machinery below:
                # every candidate is measured (distance + direction) against EACH
                # instance, and rank_by/emphasize resolves which one the target
                # sits by -- the disambiguation signal a single collapsed distance
                # would destroy. No election; the model holds the instances and the
                # ordering decides. The ONLY bound is a LOUD token-overflow safety
                # net: a fan-out whose rendered size (instances x candidates) would
                # be a crazy amount of tokens fails loud and STEERS (narrow / sort /
                # index one / pass-as-items). It fires ~never once the model narrows
                # on the result summary, and catches the self-compare (refs = cands).
                if (isinstance(v, list) and 2 <= len(v)
                        and all(isinstance(x, dict) and x.get("fid") in by_fid for x in v)):
                    # The ATTACH cost (instances x items columns on the raw
                    # rows, which chain into carry state and the compaction
                    # budget) — the rendered grid is separately bounded by
                    # the census+head envelope law.
                    _cells = len(v) * max(1, len(items))
                    if _cells > _FANOUT_TOKEN_GUARD:
                        raise ProgramError(
                            f"compare ref {ref_el!r} fans out to {len(v)} instances x "
                            f"{len(items)} candidates (~{_cells} cells) -- too large to "
                            f"render. Narrow the ref (filter_items) or the candidates "
                            f"first, then sort; or index a specific instance "
                            f"('{ref_el}[i]'). If these ARE your candidates, pass them "
                            f"as items, not refs.{_sig_suffix()}")
                    _q = _is_query_find_token(ref_el)
                    for _it in v:
                        _ms = _it.get("name_match")
                        ref_specs.append(("row", by_fid[_it["fid"]], None, by_fid[_it["fid"]],
                                          _ms if (_q and isinstance(_ms, (int, float))) else None))
                    continue
                if (isinstance(v, list) and len(v) >= 2
                        and all(isinstance(x, dict)
                                and _is_xy_literal(x.get("xy"))
                                and x.get("fid") not in by_fid for x in v)):
                    # CONSTRUCTED points (a crossings result with several
                    # crossings). The generic error below is loud but names
                    # filter_items and 'rN[i]' -- op vocabulary the sentence
                    # channel cannot say, which is the measured register
                    # break. Say the same refusal in plain terms instead;
                    # none of the places is dropped, so the narrowing is an
                    # ordinary next question over that handle.
                    _pl = str(v[0].get("name") or "").strip()
                    raise ProgramError(
                        f"{ref_el!r} is {len(v)} places"
                        + (f" ({_pl})" if _pl else "")
                        + f", and a relation reads ONE -- their centre is at "
                          f"none of them. All {len(v)} are live in {ref_el!r}: "
                          f"ask about them on their own, narrow to the one "
                          f"you mean, then relate to that.{_sig_suffix()}")
                shape = (f"{len(v)} rows" if isinstance(v, list)
                         else type(v).__name__)
                raise ProgramError(
                    f"compare ref {ref_el!r} did not resolve to surface row(s), "
                    f"an [x,y] point, or an axis; it "
                    f"resolved to {shape}. Narrow it (filter_items) or index "
                    f"one element ('{ref_el}[i]').{_sig_suffix()}")

            # --- per-ref column labels + headers ------------------------------
            notes: list = []
            used_labels: set = set()
            headers: list = []
            col_specs: list = []   # (kind, payload, label)
            for kind, payload, label_base, node, ms in ref_specs:
                if kind == "row":
                    label = label_base or _node_label(node)
                else:
                    label = label_base or "ref"
                lab = label
                _k = 2
                while lab in used_labels:  # collision -> fid suffix, then counter
                    if kind == "row" and _k == 2:
                        lab = f"{label}_{node['fid']}"
                    else:
                        lab = f"{label}_{_k}"
                    _k += 1
                used_labels.add(lab)
                if kind == "row":
                    hdr = {"ref_fid": node["fid"],
                           "ref_name": str(node.get("name") or node.get("class")
                                           or node["fid"])}
                    if ms is not None:
                        hdr["name_match"] = ms
                    # ref ATTRIBUTE confidence: a color-described detection used
                    # as an anchor ('the white car') is only as trustworthy as its
                    # perceived color -- echo it so a color_conf-0.3 ref reads as
                    # the weak hop it is.
                    if node.get("color_conf") is not None:
                        hdr["ref_color"] = node.get("color")
                        hdr["ref_color_conf"] = node["color_conf"]
                elif kind == "point":
                    hdr = {"ref_name": lab, "kind": "point",
                           "xy": [round(payload[0], 2), round(payload[1], 2)]}
                elif kind == "union":
                    hdr = {"ref_name": lab, "kind": "union",
                           "members": [n["fid"] for n in payload]}
                else:
                    hdr = {"ref_name": lab, "kind": "axis"}
                headers.append(hdr)
                col_specs.append((kind, payload, lab))

            # --- the frame aspect: view/chart coords ---------------------------
            # (resolved once, joined onto rows below; skipped on empty items)
            # The DECLARED form orients counting and provides the metric view
            # coords (G.frame / G.frame_polyline); no structure pass exists.
            frame_arg = a.get("frame")
            frame_join: dict = {}
            frame_cols: list = []
            frame_basis: tuple | None = None
            frame_poly: list | None = None
            frame_arcs: list | None = None
            # Did the MODEL declare where it stands, or did the engine infer
            # an axis? Only a declared stance ("viewer": from -> toward) still
            # earns the left/right pair -- see _side_pair's ``sides``.
            frame_declared: bool = False
            if frame_arg is not None and items:
                if not isinstance(frame_arg, dict):
                    raise ProgramError(
                        "compare 'frame' must be an object. "
                        + _FRAME_FORMS + _sig_suffix())
                f_ref = frame_arg.get("reference")
                f_toward = frame_arg.get("toward")
                f_from = frame_arg.get("from")
                if f_from is not None and f_ref is not None:
                    # 'from' is a VIEWER-form key; with 'reference' present it was
                    # silently dropped (read only inside the toward branch) and the
                    # call misdirected to the areal 'needs side' error. Name the
                    # form the caller half-assembled instead.
                    raise ProgramError(
                        "'from' belongs to the VIEWER form {toward, [from]} "
                        "(stand at 'from', look at 'toward'); you gave "
                        "'reference' with 'from'. For 'X viewed from Y' use "
                        "{toward:X, from:Y}. " + _FRAME_FORMS + _sig_suffix())
                f_along = frame_arg.get("along")
                if f_along is not None and (f_ref is not None
                                            or frame_arg.get("side")):
                    raise ProgramError(
                        "frame 'along' is the CHART form on its own: "
                        "{along:<line>, [toward]} travels the line "
                        "(the side columns follow the travel; toward declares "
                        "the direction). 'reference' and 'side' are not "
                        "read here." + _sig_suffix())
                if f_along is None:
                    if f_ref is not None and f_toward is not None:
                        raise ProgramError(
                            "the chart along a line is {along:<line>, "
                            "[toward]}; 'reference' does not combine with "
                            "'toward'. " + _FRAME_FORMS + _sig_suffix())
                    if (f_ref is None) == (f_toward is None):
                        raise ProgramError(
                            "compare " + _FRAME_FORMS + _sig_suffix())
                flab = (_sanitize_label(frame_arg["as"])
                        if frame_arg.get("as") else None)
                if frame_arg.get("scale_m") is not None:
                    raise ProgramError(
                        "frame 'scale_m' is retired: the derived-scale "
                        "arrangement columns it tuned are deleted; the frame "
                        "coordinates ('ahead of/behind/left of/right of "
                        "<ref>', 'along <ref>') need no scale."
                        + _sig_suffix())
                fr = None
                _geo_view = None
                view_frag = ""
                weak_side = False
                if f_along is not None:
                    # THE CHART: travel along the line; 'toward' declares
                    # the direction (+off = right of the declared travel),
                    # else the stored end0->end1 rides and the receipt
                    # SAYS it is undeclared.
                    _lab0, a_node = _set_label_node(f_along, "along")
                    poly = None
                    if (isinstance(f_along, (list, tuple)) and len(f_along) >= 2
                            and all(isinstance(q, (list, tuple)) and len(q) == 2
                                    and all(isinstance(z, (int, float)) for z in q)
                                    for q in f_along)):
                        poly = [[float(q[0]), float(q[1])] for q in f_along]
                        _lab0 = "line"
                        # A literal polyline IS a declaration: the model wrote
                        # the vertices, so it chose the travel direction. Only
                        # a BAKED centerline's end0->end1 is arbitrary.
                        frame_declared = True
                    elif a_node is not None:
                        _cl = a_node.get("centerline")
                        if isinstance(_cl, list) and len(_cl) >= 2:
                            poly = _cl
                    if poly is None:
                        raise ProgramError(
                            "frame 'along' needs a centerline-carrying fid "
                            "(a road) or an [[x,y],...] polyline literal to "
                            "travel." + _sig_suffix())
                    flab = flab or _lab0
                    if f_toward is not None:
                        _dcomp = (isinstance(f_toward, str)
                                  and f_toward.strip().lower() in
                                  ("north", "south", "east", "west"))
                        _dtw = (f_toward.strip().lower() if _dcomp
                                else as_point(f_toward))
                        poly, _ = G.orient_polyline_toward(poly, _dtw)
                        _dl = (_dtw if _dcomp
                               else _set_label_node(f_toward, "toward")[0])
                        _dir_frag = f", travel TOWARD {_dl}"
                        # The model said which way it is going, so left/right
                        # of that travel is a claim it made and keeps.
                        frame_declared = True
                    else:
                        # Travel falls back to the stored end0->end1 order,
                        # which is an arbitrary property of the bake -- this
                        # is road_travel_intrinsic, measured at 0.461 against
                        # a 0.470-0.535 chance band. No side columns.
                        _dir_frag = (", travel direction UNDECLARED "
                                     "(toward:<ref> declares it)")
                    structured, f_receipt = G.frame_polyline(items, poly)
                    if f_receipt.get("error"):
                        raise ProgramError(
                            f"compare frame: {f_receipt['error']}"
                            f"{_sig_suffix()}")
                    frame_poly = poly
                    frame_arcs = G._arc_lengths(poly)
                    frame_cols = [f"along {flab}"]
                    _a0 = frame_poly[0]
                    if _geo is not None:
                        _geo["chart"] = {
                            "label": flab,
                            "pts": [_geo_pt(q) for q in frame_poly],
                            "end0_xy": _geo_pt(_a0),
                            "travel": {
                                "declared": f_toward is not None,
                                "toward_label": (_dl if f_toward is not None
                                                 else None)}}
                    view_frag = (
                        f"{flab}: 'along {flab}' = meters along it "
                        f"(0 = its ({_a0[0]:.0f},{_a0[1]:.0f}) end"
                        f"{_dir_frag}), "
                        f"'left of/right of {flab}' = lateral meters on "
                        f"that side of the travel; 'distance to {flab}' = "
                        f"either side)")
                elif f_toward is not None and f_ref is None:
                    # VIEWER form: any mask fills either slot identically
                    # (class-blind); toward may be a compass word. A slot
                    # whose node carries a CENTERLINE resolves to the
                    # nearest centerline point to its counterpart — a
                    # describer "on Main St" stands where the road meets
                    # the look, never at the road's area centroid (blocks
                    # away on a long road). Pool-item positions stay area
                    # centroids (POSITION intent, geom_compute MASK-INTENT
                    # BASIS).
                    if frame_arg.get("side"):
                        raise ProgramError(
                            "the viewer frame fixes orientation by the look "
                            "ray -- 'side' is not read; the columns already "
                            "carry meters on the side the NAME says. "
                            + _FRAME_FORMS + _sig_suffix())
                    _compass = (isinstance(f_toward, str) and f_toward.strip()
                                .lower() in ("north", "south", "east", "west"))

                    def _cline(node):
                        cl = (node or {}).get("centerline") if isinstance(
                            node, dict) else None
                        return cl if (isinstance(cl, list)
                                      and len(cl) >= 2) else None

                    _tl, _t_node = ((f_toward.strip().lower(), None) if _compass
                                    else _set_label_node(f_toward, "toward"))
                    _fl, _f_node = ((_set_label_node(f_from, "from"))
                                    if f_from is not None else (None, None))
                    _fxs = [it["xy"][0] for it in items]
                    _fys = [it["xy"][1] for it in items]
                    _pool_c = [sum(_fxs) / len(_fxs), sum(_fys) / len(_fys)]
                    _vantage = False
                    if f_from is not None:
                        _fcl = _cline(_f_node)
                        if _fcl is not None:
                            _cp = (_pool_c if _compass
                                   else as_point(f_toward))
                            fp = G.polyline_nearest_point(_fcl, _cp)
                            _vantage = True
                        else:
                            fp = as_point(f_from)
                    else:
                        # vantage defaults to the pool centroid (stated in
                        # the receipt); a compass frame needs no vantage.
                        fp = _pool_c
                    if _compass:
                        tw = f_toward.strip().lower()
                    else:
                        _tcl = _cline(_t_node)
                        tw = (G.polyline_nearest_point(_tcl, fp)
                              if _tcl is not None else as_point(f_toward))
                        _vantage = _vantage or _tcl is not None
                    fr = G.frame(items, (fp, tw), "viewer")
                    if _geo is not None:
                        _geo_view = {
                            "standpoint_xy": _geo_pt(fp),
                            "look_xy": None if _compass else _geo_pt(tw),
                            "compass": tw if _compass else None,
                            "toward_label": _tl,
                            "from_label": _fl}
                    if flab is None:
                        flab = f"{_fl}_{_tl}" if _fl else _tl
                    _span = (fr.get("frame") or {}).get("span_m")
                    view_frag = (f"looking at {_tl} from "
                                 f"{_fl or 'the pool center'}"
                                 + (f"; span {_span}m" if _span is not None
                                    else "")
                                 + (" (line slots stand at the nearest "
                                    "centerline point)" if _vantage else ""))
                elif isinstance(f_ref, str) and f_ref.strip().lower() == "auto":
                    if f_toward is not None:
                        raise ProgramError(
                            "frame reference:'auto' derives the axis from "
                            "the pool itself -- 'toward' is not read. "
                            + _FRAME_FORMS + _sig_suffix())
                    if frame_arg.get("side"):
                        raise ProgramError(
                            "frame reference:'auto' derives the axis from the "
                            "pool itself -- 'side' is not read. "
                            + _FRAME_FORMS + _sig_suffix())
                    fr = G.frame(items, None, "auto")
                    flab = flab or "auto"
                    view_frag = "no declared view (+vx = the pool's own axis, east-ward)"
                else:
                    # reference form -- dispatch by CALL SHAPE + DATA KIND:
                    # an explicit 'side' is the DECLARED VIEW about the
                    # reference's position (the mask's TRUE area centroid for
                    # any areal reference -- a building/lot/road footprint --
                    # else its point); NO side + a fid CARRYING a stored
                    # centerline or an [[x,y],...] literal is the CURVILINEAR
                    # chart along it. 'side' on a reference whose only
                    # geometry is a polyline (no areal footprint, no point)
                    # is undefined and errors loud, valid forms first.
                    _lab0, f_node = _set_label_node(f_ref, "frame")
                    poly = None
                    if (isinstance(f_ref, (list, tuple)) and len(f_ref) >= 2
                            and all(isinstance(q, (list, tuple)) and len(q) == 2
                                    and all(isinstance(z, (int, float)) for z in q)
                                    for q in f_ref)):
                        poly = [[float(q[0]), float(q[1])] for q in f_ref]
                        _lab0 = "line"
                    elif f_node is not None:
                        _cl = f_node.get("centerline")
                        if isinstance(_cl, list) and len(_cl) >= 2:
                            poly = _cl
                    f_side = frame_arg.get("side")
                    if f_side:
                        _fm = (f_node or {}).get("mask") if isinstance(
                            f_node, dict) else None
                        if _fm:
                            refpt = (f_node.get("mask_centroid")
                                     or _mask_center(_fm))
                            if refpt is None:
                                raise ProgramError(
                                    f"frame{{side:}} orients about the "
                                    f"reference's CENTRE, and {_lab0} is "
                                    f"several disjoint places under one name "
                                    f"-- a centre over them is at none of "
                                    f"them, so no side about it exists. Frame "
                                    f"about an [x, y] you pick, or use "
                                    f"frame{{along: ...}} to chart along a "
                                    f"line." + _sig_suffix())
                        elif poly is not None:
                            raise ProgramError(
                                f"valid forms for {_lab0}: frame{{along: "
                                f"{_lab0!r}, [toward]}} charts ALONG "
                                f"it ('along <ref>' = meters along it, exact "
                                f"on bends; 'left of/right of <ref>' = lateral "
                                f"meters on that side of travel); "
                                f"reference:'auto' derives the pool's own "
                                f"axis; {{toward:<fid|[x,y]|compass>}} is "
                                f"the viewer look. "
                                f"{_lab0} realises only a polyline (no areal "
                                f"footprint), so 'side' about it is "
                                f"undefined and was not read."
                                + _sig_suffix())
                        else:
                            refpt = as_point(f_ref)
                        flab = flab or _lab0
                        fr = G.frame(items, refpt, "relative", f_side)
                        if not fr.get("error"):
                            # Premortem disclosure: a reference whose footprint
                            # holds most of the pool, or whose point sits
                            # INSIDE the pool's spread, orients a side weakly
                            # (a receipt fact, never a gate).
                            _m = (f_node or {}).get("mask") if isinstance(
                                f_node, dict) else None
                            if _m:
                                _in = int(G.mask_contains(
                                    G.mask_geom(_m),
                                    [it["xy"] for it in items]).sum())
                                weak_side = 2 * _in > len(items)
                            if not weak_side and len(items) >= 3:
                                _hull = G._convex_hull(
                                    [it["xy"] for it in items])
                                weak_side = G.point_in_polygon(refpt, _hull)
                    elif poly is not None:
                        # a line with no side is the CHART -- its name is
                        # 'along' (the arg says what it does); no alias.
                        raise ProgramError(
                            f"{_lab0} is a line: the chart along it is "
                            f"frame {{along:{_lab0!r}, [toward]}} "
                            "('left of'/'right of' follow the travel; toward "
                            "declares the direction). A line takes no other "
                            "frame form." + _sig_suffix())
                    else:
                        raise ProgramError(
                            "a bare areal/point 'reference' declares no "
                            "served view; 'side' is not a key of the served "
                            "schema. " + _FRAME_FORMS + _sig_suffix())
                if fr is not None:
                    if fr.get("error"):
                        raise ProgramError(
                            f"compare frame: {fr['error']}{_sig_suffix()}")
                    _fmeta = fr.get("frame") or {}
                    # A DECLARED orientation: "viewer" (stand at from, look at
                    # toward) or "relative" WITH a side the model named ("put
                    # the shed at the bottom of my view"). "relative" without a
                    # side, and "auto" (the pool's own measured axis), are the
                    # engine choosing -- anchor_own_axis, measured 0.479.
                    frame_declared = (
                        _fmeta.get("frame_type") == "viewer"
                        or (_fmeta.get("frame_type") == "relative"
                            and bool(frame_arg.get("side"))))
                    _ru, _uu = _fmeta.get("right_unit"), _fmeta.get("up_unit")
                    # the framed pts ARE the join rows (op_x/op_y view
                    # coords); no structure pass exists any more.
                    structured = fr.get("pts") or []
                    frame_basis = (_ru, _uu, _fmeta.get("origin"))
                    if _geo is not None:
                        _og = _fmeta.get("origin")
                        _geo["frame"] = {
                            "type": _fmeta.get("frame_type"),
                            "label": flab,
                            "origin": (_geo_pt(_og)
                                       if _og is not None else None),
                            "right_unit": (_geo_pt(_ru, nd=4)
                                           if _ru is not None else None),
                            "up_unit": (_geo_pt(_uu, nd=4)
                                        if _uu is not None else None),
                            "span_m": _fmeta.get("span_m"),
                            "side": frame_arg.get("side"),
                            **(_geo_view or {})}
                    frame_cols = []
                    if _fmeta.get("frame_type") == "relative":
                        # the self-check echo: the declared reference read
                        # back in the view it fixed, as WORDS.
                        rox, roy = fr.get("reference_op") or (0.0, 0.0)
                        if abs(rox) >= abs(roy):
                            _w = "RIGHT" if rox >= 0 else "LEFT"
                            view_frag = f"{flab} {abs(rox):.0f} m to your {_w}"
                        else:
                            _w = "AHEAD" if roy >= 0 else "BEHIND"
                            view_frag = f"{flab} {abs(roy):.0f} m {_w}"
                for rec in structured:
                    key = rec.get("fid") or (round(rec["xy"][0], 2),
                                             round(rec["xy"][1], 2))
                    frame_join[key] = rec
            # --- row assembly: PASSTHROUGH + columns ---------------------------
            # Column-key tracking: every attached key is
            # recorded VERBATIM -- per ref (echoed as that header's ``cols``)
            # and program-wide (the result's ``columns`` echo) -- so the model
            # reads real column names instead of guessing them.
            ref_col_keys: list[dict] = [{} for _ in col_specs]
            all_col_keys: dict = {}

            def _put(row, k, val, ci=None):
                row[k] = val
                all_col_keys[k] = None
                if ci is not None:
                    ref_col_keys[ci][k] = None

            # Per-ref MASK geometry, resolved ONCE: ``(geom, center)``.
            # The gap side is the ref's EXACT realised mask (G.mask_geom over
            # its ``mask`` chains — the geometry it was given, one chain per
            # part, each stating its own closure) when present, else its point
            # (a point's position IS its whole footprint). ``center`` is
            # the mask's TRUE area centroid for dc_/brg_ (derived from the
            # mask itself — unconditional whenever any geometry exists; a
            # constructed axis reads its arc midpoint). A UNION ref holds a
            # LIST of member geoms plus the union's area-weighted centroid.
            # THE ONE GEOMETRY-RESOLUTION LAW (refs AND items route through
            # here): a row's geometry is re-resolved BY FID from the
            # authoritative node registry (``by_fid``) at compute time. Op
            # OUTPUT rows strip the server-resolvable mask fields
            # (_MODEL_HIDDEN_ROW_FIELDS), so a chained / sliced / re-labeled
            # row would otherwise silently degrade to its xy point — the
            # measured contact-flip / self-distance defect. Re-resolving here
            # makes every chained op byte-identical to a fresh compare on the
            # same fids; a fid-less row (an [x,y] literal, constructed
            # geometry) keeps its own fields.
            def _row_geom_center(node):
                if isinstance(node, dict):
                    node = by_fid.get(node.get("fid")) or node
                _cxy = (node.get("xy")
                        if _is_xy_literal(node.get("xy")) else None)
                m = node.get("mask")
                if m:
                    g, c = _mask_geom_center(m)
                    return (g, node.get("mask_centroid") or c)
                return (G.mask_geom([[_cxy]] if _cxy else []),
                        _cxy)

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
                elif kind == "line":
                    col_geoms.append(
                        (G.mask_geom([payload]),
                         _polyline_arc_midpoint(payload)))
                else:  # constructed point
                    col_geoms.append(
                        (G.mask_geom([[payload]]), payload))
            if _geo is not None:
                for (_ks, _ps, _lc), (_gm, _cc) in zip(col_specs, col_geoms):
                    if _cc is not None:
                        _geo["ref_centers"].append(
                            {"label": _lc, "center_xy": _geo_pt(_cc)})

            # LINEAR REF -> ITS OWN CHART: a ref that IS a line (a stored
            # centerline ≥2 verts — mask presence irrelevant, so masked and
            # unmasked road priors behave identically — or an [[x,y],...]
            # polyline literal) attaches TWO columns, 'along <ref>' (arc
            # meters from its end0_xy end) + 'distance to <ref>' (the chart's
            # UNSIGNED lateral), INSTEAD of the hull-gap columns: a 946 m
            # road's gap tie-block at 0.0 is contact, not location (measured:
            # it misled repeatedly while 'along' carried 100 legitimate ops).
            # The chart's SIGNED lateral stays unserved — see _side_pair.
            # Point/polygon refs keep their four gap/centroid columns.
            col_lin: list = []
            # per line ref, its polyline and that polyline's cumulative arc
            # lengths — the chart every row on this ref projects onto.
            for kind, payload, lab in col_specs:
                # ONLY a DECLARED chart (frame:{along: ...}) charts. A row ref
                # measures its gap, whatever shape it is: distance is nearest
                # point to nearest point and a road has exactly one of those,
                # like a building does. Substituting a lateral offset for the
                # gap because the ref was elongated made `distance to <road>`
                # a different quantity than `distance to <building>` under one
                # column name -- 11.24 m where the footprints are 3.66 m
                # apart, against a 20 m success radius, on the closest
                # house/road pair the calcity bake has. Roads carry real
                # footprints (median 57 cells of the 2 m mesh), so the mask
                # gap has their width in it and the centerline never could.
                _lin = payload if kind == "line" else None
                if _lin is not None:
                    col_lin.append((_lin, G._arc_lengths(_lin)))
                    if _geo is not None:
                        _geo["linear_refs"].append(
                            {"label": lab, "pts": [_geo_pt(q) for q in _lin]})
                    # Says only what attaches. It claimed a lateral pair that
                    # _side_pair drops (sides=False below): measured over
                    # saydialect_1000, 308/308 note-carrying grids carried
                    # 'along' and 'distance to' and ZERO carried a side, while
                    # 348 notes over 227 episodes said the pair attached. Both
                    # words are also productions the sentence language dropped,
                    # and the model tried them -- four sentences, all refused.
                    # The old parenthetical named the counterfactual hull-gap
                    # column with the name of a column that IS served and does
                    # NOT behave that way.
                    notes.append(
                        f"ref {lab} is a LINE (stored centerline): two "
                        f"chart columns attach instead of a footprint-gap "
                        f"column. 'along {lab}' is arc m from its end0_xy "
                        f"end, un-clamped past both ends (negative before "
                        f"end0_xy). 'distance to {lab}' is unsigned lateral "
                        f"m off that chart, the same either side of the "
                        f"line. A footprint gap would read 0.0 for every row "
                        f"touching the line anywhere along its length -- "
                        f"contact, not location.")
                else:
                    col_lin.append(None)

            # WINDOW-COVERING REF (geometric fact, stamped at build_surface):
            # a ref whose mask contains every surface row position makes the
            # per-row gap columns DEGENERATE-CONSTANT (d_=0 for every row,
            # brg_gap_ direction-less at contact). Serve the ONE containment
            # fact in the envelope instead of stamping meaningless zeros
            # (the flat-sort/contact idiom); dc_/brg_ (the centroid vector)
            # stay per-row — they are informative.
            # --- the map-fixed cardinal columns -------------------------------
            # Which DIRECTIONS were asked for, per column. Built once here
            # rather than per row: the reference's point cloud is fixed for
            # the whole pass, and a 64-way road densifies to ~9k points that
            # would otherwise be rebuilt for every candidate.
            _card_req: dict = {}
            for _el in (a.get("cardinals") or []):
                if not isinstance(_el, dict):
                    raise ProgramError(
                        f"cardinals element must be {{dir:.., ref:..}}; got "
                        f"{type(_el).__name__}.{_sig_suffix()}")
                _d = str(_el.get("dir") or "").strip().lower()
                if _d not in CARDINALS:
                    raise ProgramError(
                        f"cardinal direction must be one of "
                        f"{sorted(CARDINALS)}; got {_el.get('dir')!r}. A "
                        f"diagonal has no single axis to measure across, so "
                        f"it is not offered.{_sig_suffix()}")
                _card_req.setdefault(str(_el.get("ref")), []).append(_d)
            col_cards: list = []
            for ci, (kind, payload, lab) in enumerate(col_specs):
                want = []
                if _card_req:
                    _fid = (payload.get("fid") if kind == "row"
                            and isinstance(payload, dict) else None)
                    for _k in (_fid, lab):
                        if _k and _k in _card_req:
                            want = _card_req[_k]
                            break
                if not want:
                    col_cards.append(None)
                    continue
                _g = col_geoms[ci][0]
                _pts = cardinal_side.points_from_geoms(
                    _g if isinstance(_g, list) else [_g])
                col_cards.append((want, _pts) if _pts is not None else None)

            col_covers: list = []
            for (kind, payload, lab) in col_specs:
                covers = bool(kind == "row" and isinstance(payload, dict)
                              and payload.get("covers_window"))
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

            # REF-IN-MASK CONTAINMENT (per ref index): rows whose OWN mask
            # CONTAINS the ref -- the sub-window-but-huge container case
            # (boundary/admin polygons km2-scale: a political-ward relation
            # CONTAINS the cue's landmark, reads d_=0, and sits atop every
            # nearest-first sort as if adjacent). covers_window handles only
            # masks containing the WHOLE surface; this is its per-ref twin:
            # the d_=0 row renders the containment fact (a contains_<ref>
            # cell + one note). Sorts unchanged, nothing dropped.
            contains_rows: dict[int, list[str]] = {}
            out_rows: list = []
            # Member boxes are a property of the REFERENCE, not of the item,
            # so they are stacked once per column and reused down the items.
            col_member_bb: dict[int, "object"] = {}
            for i, it in enumerate(items):
                # intrinsic passthrough: every input field survives (row_index/
                # row_size/cluster_*/color_<word>/along_*/... feed the downstream
                # sort); only the server-resolvable fields stay hidden.
                row = {k: v for k, v in it.items()
                       if k not in _MODEL_HIDDEN_ROW_FIELDS}
                # MASK-INTENT BASIS (geom_compute design contract): d_ +
                # brg_gap_ carry GAP/CONTAINMENT intent -> both sides use
                # their EXACT realised mask (mask-to-mask min + its witness
                # pair; a point-mask object degrades to its point
                # consistently). dc_/brg_ carry POSITION intent -> the mask's
                # TRUE area centroid is its position summary. The item side
                # resolves through the SAME ``_row_geom_center`` law as the
                # refs (geometry BY FID from the node registry), so item and
                # ref read identical geometry for the same fid — a ref
                # compared to ITSELF is exactly d=0 / dc=0.
                item_geom, it_c = _row_geom_center(it)
                for ci, (kind, payload, lab) in enumerate(col_specs):
                    # A cardinal is independent of the gap/chart split below:
                    # it needs the reference's extent and the row's position
                    # and nothing else, so it attaches to a charting linear
                    # ref and a merged multi-way one alike.
                    if col_cards[ci] is not None and it_c is not None:
                        _dirs, _rpts = col_cards[ci]
                        for _d in _dirs:
                            _v = cardinal_side.cardinal_offset(_rpts, it_c, _d)
                            # ABSENT, never 0.0, when there is no side to
                            # name: the executor's missing-value idiom, which
                            # score_by already treats as the neutral. A zero
                            # would read as "exactly on the line" and sort
                            # like a real measurement.
                            if _v is not None:
                                _put(row, f"{_d} of {lab}", round(_v, 1), ci)
                    if col_lin[ci] is not None:
                        # linear ref: its own chart, not hull-gap columns.
                        _lp, _la = col_lin[ci]
                        _pr = (G._poly_chart_one(_lp, _la, it_c)
                               if it_c is not None else None)
                        if _pr is not None:
                            _put(row, f"along {lab}", _pr[0], ci)
                            # sides=False: travel on a BAKED centerline is the
                            # stored end0->end1 order, an arbitrary property
                            # of the bake, measured at 0.461 against a
                            # 0.470-0.535 chance band. Only the unsigned
                            # magnitude serves, so an undeclared road chart
                            # names no side and no compass for one.
                            # ci rides the kwarg: _side_pair calls
                            # row_put(key, val, ci) with its OWN ci, so a
                            # lambda default here is overwritten by the
                            # third positional -- that dropped `distance to
                            # <lab>` from every line ref's cols echo
                            # (308/308 grids on saydialect_1000) while the
                            # program-wide columns echo carried it.
                            _side_pair(lambda k, v, c, _r=row:
                                       _put(_r, k, v, c), lab, _pr[1],
                                       ci=ci, distance=True, sides=False)
                        continue
                    ref_geom, center = col_geoms[ci]
                    if col_covers[ci]:
                        # window-covering ref: the gap columns are a
                        # degenerate constant — the envelope carries the ONE
                        # containment fact; dc_/brg_ still attach below.
                        d = None
                    elif kind == "union":
                        # exact min gap over the members; the witness pair
                        # comes from the argmin member (its LOCAL direction).
                        #
                        # THE SAME MIN, FEWER CALLS. A reference standing for
                        # a whole class is a union of every member: `find
                        # building in building.` is 3,452 items against 3,452
                        # members, and calling gap_witness on all 11.9 M pairs
                        # is why that one sentence ran 885 s without
                        # returning. gap_witness never reads below the two
                        # bounding boxes' separation, so a member whose bound
                        # already exceeds the best gap in hand cannot beat it,
                        # and nothing beats 0. Members are still walked in
                        # their own order and the test is still strict, so the
                        # witness a tie keeps is the same one.
                        _bbs = col_member_bb.get(ci)
                        if _bbs is None:
                            _bbs = col_member_bb[ci] = G.stack_bboxes(ref_geom)
                        _bounds = G.bbox_bounds(_bbs, item_geom)
                        # SEED FROM THE NEAREST BOX. Walking members in their
                        # own order means an unlucky far first member leaves
                        # the cut loose for a long stretch. Measuring the
                        # closest box first gives an upper bound on the answer
                        # straight away, and a member whose box already
                        # exceeds it cannot be the minimum -- so it is skipped
                        # without ever being the recorded one. The witness is
                        # still taken from the ordered pass below, so the
                        # member a tie keeps does not move.
                        _ub = None
                        _seed_i, _seed = -1, None
                        if len(ref_geom) > 1 and len(_bounds):
                            _seed_i = int(_bounds.argmin())
                            _seed = G.gap_witness(ref_geom[_seed_i], item_geom)
                            _ub = _seed[0]
                        d = wr = wi = None
                        for _gi, _g in enumerate(ref_geom):
                            if d is not None and d <= 0.0:
                                break
                            _cut = d if d is not None else _ub
                            if (_cut is not None
                                    and _bounds[_gi] > _cut + G.BOUND_SLACK_M):
                                continue
                            # The seed member is usually the answer, and the
                            # pass reached it a second time: the same pair
                            # gives the same triple, so it is not re-measured.
                            _d, _wr, _wi = (_seed if _gi == _seed_i
                                            else G.gap_witness(_g, item_geom))
                            if d is None or _d < d:
                                d, wr, wi = _d, _wr, _wi
                    else:
                        d, wr, wi = G.gap_witness(ref_geom, item_geom)
                    if d is not None:
                        dr = round(d, 2)
                        _put(row, f"distance to {lab}", dr, ci)
                        if dr > 0.0 and wr is not None:
                            # the min-gap vector's compass bearing, ref
                            # witness -> item witness. ABSENT exactly when
                            # d_<ref>=0: on contact no local direction
                            # exists (brg_ still reads).
                            # WHICH WAY THIS ROW FACES THE REF, as a word.
                            # The witness pair runs ref->item; a thing faces
                            # the place it fronts, so the served direction is
                            # item->ref. That reversal used to be a formula
                            # in the legend, `(b + 180) % 360`, and the one
                            # episode that read the degrees column applied it
                            # zero times and called an facing of 0
                            # "on the north side" (calcity::480#478, a
                            # baseline hit the arm lost). A word cannot be
                            # read backwards.
                            _put(row, f"facing {lab}",
                                 G.cardinal(wi, wr), ci)
                            if _geo is not None:
                                _geo["gaps"].append(
                                    {"label": lab, "fid": row.get("fid"),
                                     "a_xy": _geo_pt(wr),
                                     "b_xy": _geo_pt(wi), "gap_m": dr})
                        elif (dr == 0.0
                              and not isinstance(ref_geom, list)
                              and item_geom["closed"]
                              and len(item_geom["A"])
                              and len(ref_geom["V"])
                              and not (kind == "row"
                                       and isinstance(payload, dict)
                                       and payload.get("fid")
                                       and str(payload.get("fid"))
                                       == str(row.get("fid")))
                              and bool(G.mask_contains(
                                  item_geom, ref_geom["V"]).all())):
                            # the row's OWN mask CONTAINS the ref (every ref
                            # vertex inside it): d_=0 by containment, not
                            # adjacency — the visible fact rides the row.
                            _put(row, f"contains {lab}", True, ci)
                            contains_rows.setdefault(ci, []).append(
                                str(row.get("fid") or f"item[{i}]"))
                    if center is not None and it_c is not None:
                        dcr = round(G.dist(center, it_c), 2)
                        _put(row, f"center distance to {lab}", dcr, ci)
                        if dcr > 0.0:
                            # centroid->centroid bearing: defined on contact
                            # too, ABSENT only when the centroids coincide
                            # (dc_<ref>=0 -- no direction exists; the
                            # self-compare case).
                            _put(row, f"bearing from {lab}",
                                 round(G.bearing(center, it_c), 1), ci)
                if frame_join:
                    key = it.get("fid") or (round(it["xy"][0], 2),
                                            round(it["xy"][1], 2))
                    rec = frame_join.get(key)
                    if rec is not None:
                        def _rput(k, v, ci=None, _row=row):
                            _put(_row, k, v, ci)
                        if frame_poly is not None:
                            _put(row, f"along {flab}", rec.get("along"))
                            _side_pair(_rput, flab, rec.get("off"),
                                       distance=True, sides=frame_declared)
                        else:
                            _side_pair(_rput, flab, rec.get("op_x"),
                                       rec.get("op_y"),
                                       sides=frame_declared)
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
            # Header echo: each header entry lists ITS columns verbatim.
            # With a frame present, each point-like ref ALSO projects into the
            # SAME frame -- ref_vx/ref_vy in a view frame, ref_along/ref_off
            # in a polyline chart -- so "left of X" is vx_<row> vs ref_vx (the
            # same class-blind projection the items got).
            for hdr, keys, spec in zip(headers, ref_col_keys, col_specs):
                if keys:
                    hdr["cols"] = list(keys)
                if (frame_basis and frame_basis[0] is not None) or frame_poly:
                    _kind_s, _payload, _lab_s = spec
                    _rxy = None
                    if _kind_s == "row" and isinstance(_payload, dict):
                        _rxy = _payload.get("xy")
                    elif _kind_s == "point":
                        _rxy = _payload
                    def _hput(k, v, ci=None, _h=hdr):
                        _h[k] = v
                    if _rxy is not None and frame_poly is not None:
                        _pr = G._poly_chart_one(frame_poly, frame_arcs, _rxy)
                        if _pr is not None:
                            hdr["along"] = _pr[0]
                            _side_pair(_hput, "the line", _pr[1])
                    elif _rxy is not None:
                        (_ru, _uu, _og) = frame_basis
                        _dx = float(_rxy[0]) - float(_og[0])
                        _dy = float(_rxy[1]) - float(_og[1])
                        _side_pair(_hput, "view center",
                                   _dx * _ru[0] + _dy * _ru[1],
                                   _dx * _uu[0] + _dy * _uu[1])
            for _h in _standing_ref_handles:
                out_rows = _attach_standing_columns(
                    out_rows, staged[_h], with_distance=True)
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
            r.terminal_warn = op is _emitted_terminal
            if _geo is not None:
                geometry_sink.setdefault("ops", []).append(_geo)
        results[ref] = r
    return {"results": results}
