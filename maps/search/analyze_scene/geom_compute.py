"""Frame-correct geometry helpers — the math behind the ``analyze_scene`` ops.

This module is the **single source of truth** for the spatial primitives the
reasoner needs to disambiguate same-class instances ("the 4th car from the
corner", "the white car between A and B", "on the road left of the church").
The ``analyze_scene`` op-program executor
(:mod:`maps.search.analyze_scene.geom_program`) calls these primitives to run each op.

Design contract (load-bearing):

- **Pure + data-in-args.** No graph, no I/O, ``math`` and ``numpy`` only
  (numpy vectorises the exact-mask gap substrate). Points are ``[x, y]``
  world meters; polylines are ``[[x, y], ...]`` world meters.

- **ONE frame: x = east, y = NORTH.** World coordinates are
  easting/northing (row 0 of every repo grid is the SOUTH edge). Cardinal
  words, bearings, signed sides and left/right are all computed in this
  frame: compass words map through the one :data:`COMPASS_UNIT` table, and
  viewer handedness is a proper rotation of it (right = cw(forward) in a
  y-up frame). Grid-vs-world is an affine map with positive scale, so
  world-xy results equal grid ``(col, row)`` results in meters.

- **Composable primitives, never prompt-specific solvers.** The executor
  composes ``gap_witness`` / ``bearing`` into the
  relation the instruction states.
"""

from __future__ import annotations

import math

import numpy as np

from maps.util.polyline import clip_t

# 8-point compass wedges over compass bearing (0=N, clockwise).
_COMPASS8 = (
    (22.5, "north"), (67.5, "northeast"), (112.5, "east"), (157.5, "southeast"),
    (202.5, "south"), (247.5, "southwest"), (292.5, "west"), (337.5, "northwest"),
)

# THE compass-word unit vectors, world frame (x east, y north): eight winds,
# the same words :func:`_compass8` emits. Every consumer (the crop rotation
# in reasoner_loop, the band rule in cardinal_side) reads this one table.
_SQRT_HALF = math.sqrt(0.5)
COMPASS_UNIT = {
    "north": (0.0, 1.0), "northeast": (_SQRT_HALF, _SQRT_HALF),
    "east": (1.0, 0.0), "southeast": (_SQRT_HALF, -_SQRT_HALF),
    "south": (0.0, -1.0), "southwest": (-_SQRT_HALF, -_SQRT_HALF),
    "west": (-1.0, 0.0), "northwest": (-_SQRT_HALF, _SQRT_HALF),
}

# The four axis-aligned winds, declared as a subset of the table above. The
# band rule needs one orthogonal axis to band on and a diagonal has none
# (:mod:`cardinal_side`).
CARDINALS = frozenset({"north", "south", "east", "west"})


def cw(v):
    """Clockwise quarter-turn in the y-up world frame, the handedness law
    every left/right consumer shares. A viewer's RIGHT is ``cw(forward)``:
    looking north (0,1) -> right = east (1,0)."""
    return (v[1], -v[0])


def _xy(p):
    """Coerce a point to ``(x, y)`` floats. Accepts ``[x, y]`` or ``{'xy': ...}``."""
    if isinstance(p, dict):
        p = p.get("xy")
    return float(p[0]), float(p[1])


def _compass8(bearing_deg: float) -> str:
    b = float(bearing_deg) % 360.0
    for hi, name in _COMPASS8:
        if b < hi:
            return name
    return "north"


def dist(a, b) -> float:
    """Euclidean distance in meters (frame-independent)."""
    ax, ay = _xy(a)
    bx, by = _xy(b)
    return math.hypot(bx - ax, by - ay)


def bearing(a, b) -> float:
    """Compass bearing (0=N, clockwise, degrees) of the ray ``a -> b``.

    World frame is ``(east, north)`` directly, so ``atan2(dx, dy)``: due
    north (+y) is 0, due east (+x) is 90.
    """
    ax, ay = _xy(a)
    bx, by = _xy(b)
    return math.degrees(math.atan2(bx - ax, by - ay)) % 360.0


def cardinal(a, b) -> str:
    """8-point compass direction of ``a -> b`` ('north', 'northeast', ...)."""
    return _compass8(bearing(a, b))


# --- fuzzy color matching (perceptual, FAMILY-based) ------------------------
# Perception's dominant color slip is within a family (a gray car reads
# "silver"/"white" under glare or "black" under shadow; a red car reads
# "maroon"), so matching is by family, never by Lab distance, which would
# drop exactly those. Color stays a SOFT signal; geometry is the hard
# discriminator.
_COLOR_FAMILY = {
    "white": "achromatic", "silver": "achromatic", "gray": "achromatic", "black": "achromatic",
    "red": "red", "maroon": "red",
    "orange": "warm", "brown": "warm", "tan": "warm", "beige": "warm", "gold": "warm",
    "yellow": "yellow", "green": "green", "blue": "blue", "purple": "purple",
}


def _color_word_raw(a: str, b: str) -> float:
    """Word-vs-word family-soft raw score (inputs already spell-normalized +
    lowercased): 1.0 same word; 0.8 same family; 0.0 cross-family; 0.5 when
    either side is absent; unknown words fall back to substring containment."""
    if not a or not b:
        return 0.5
    if a == b:
        return 1.0
    fa, fb = _COLOR_FAMILY.get(a), _COLOR_FAMILY.get(b)
    if fa is not None and fb is not None:
        return 0.8 if fa == fb else 0.0
    return 1.0 if (a in b or b in a) else 0.0  # unknown words: substring fallback


def _parse_color_mixture(s: str) -> "list[tuple[str, float]] | None":
    """Parse a color-MIXTURE string ("red 0.45 / dark 0.30") into
    ``[(word, fraction), ...]``; ``None`` when ``s`` is not a mixture (a plain
    single word). The delimiter is ' / ' (it never breaks the markdown grid);
    ' | ' parses too. Each part must be exactly ``<word> <fraction in
    [0,1]>``."""
    parts = [p.strip() for p in (s or "").replace("|", "/").split("/")]
    comps: list[tuple[str, float]] = []
    for part in parts:
        toks = part.split()
        if len(toks) != 2 or not toks[0].isalpha():
            return None
        try:
            frac = float(toks[1])
        except ValueError:
            return None
        if not 0.0 <= frac <= 1.0:
            return None
        word = toks[0].lower()
        comps.append((word, frac))
    return comps or None


def color_match(cand_color: str, cue_color: str, conf: float | None = None) -> float:
    """FAMILY-based soft color match, optionally CONFIDENCE-shrunk: the value
    of a ``color_<word>`` column. 1.0 = same word; 0.8 = same perceptual
    family (achromatic white/silver/gray/black, red/maroon, the warm group);
    0.0 = different family; absent color -> 0.5.

    ``cand_color`` may be a color MIXTURE ("red 0.45 / black 0.30"): each
    component's family-soft raw is shrunk toward the 0.5 neutral by its pixel
    fraction (``0.5 + frac*(raw - 0.5)``) and the best component wins, so a
    matching 0.45-fraction "red" scores 0.725.

    ``conf`` is perception's per-detection color confidence in [0,1]. When
    given, the score shrinks toward 0.5: ``0.5 + conf*(raw - 0.5)``, so a
    low-confidence color neither vetoes nor carries much signal. ``conf=None``
    returns the family-soft score unchanged."""
    b = (cue_color or "").strip().lower()
    mix = _parse_color_mixture(cand_color or "")
    if mix is not None:
        if not b:
            raw = 0.5
        else:
            raw = max(0.5 + frac * (_color_word_raw(w, b) - 0.5) for w, frac in mix)
    else:
        a = (cand_color or "").strip().lower()
        raw = _color_word_raw(a, b)
    if conf is None:
        return raw
    c = 0.0 if conf < 0.0 else (1.0 if conf > 1.0 else float(conf))
    return 0.5 + c * (raw - 0.5)


# --- color INTERFACE: closed enum + open->closed map + token-minimal surface cell ----------
# ONE closed vocabulary the agent searches with (the find ``color`` arg), the
# extractor emits, and the surface shows; ``normalize_color`` maps an open
# word onto it (cue "burgundy" -> red, "navy" -> blue).
COLOR_ENUM = ["white", "gray", "black", "red", "blue", "green", "yellow", "orange", "brown", "purple"]

_COLOR_SYNONYMS = {
    "silver": "gray", "charcoal": "gray", "slate": "gray",
    "maroon": "red", "crimson": "red", "scarlet": "red", "burgundy": "red", "pink": "red",
    "navy": "blue", "cyan": "blue", "teal": "blue", "turquoise": "blue", "aqua": "blue",
    "violet": "purple", "magenta": "purple", "lavender": "purple",
    "gold": "yellow", "golden": "yellow", "amber": "yellow",
    "cream": "white", "beige": "brown", "tan": "brown", "bronze": "brown", "copper": "brown",
    "olive": "green", "lime": "green",
}


def normalize_color(word: str) -> str:
    """Map a (possibly open-world) cue color onto ``COLOR_ENUM``; identity on
    an enum member; any other word comes back as itself, for the caller to
    refuse or score."""
    w = (word or "").strip().lower()
    if w in COLOR_ENUM:
        return w
    return _COLOR_SYNONYMS.get(w, w)


# The named color-column family the agent sorts on: ``color_<word>`` over
# ``COLOR_ENUM``. Each column's value is ``color_match(row.color, word,
# row.color_conf)``; rows without a perceived color omit the column. The set
# drives the calibrated-unit treatment in :func:`score_by` and the evidence
# carry in the frame charts.
COLOR_COLUMNS = frozenset("color_" + w for w in COLOR_ENUM)


def compact_color_cell(color_str: str, shade: "str | None" = None, uncertain: bool = False,
                       rich: bool = True) -> str:
    """RENDER-ONLY: a mixture string ("red 0.45 / black 0.30") + optional
    shade/uncertain -> a token-minimal color tag ``[dk|lt ]word[/word2][?]``.
    ``word2`` shows only when ``rich`` (a color-name posterior's second
    choice) AND it crosses achromatic<->chromatic or a hue family. A trailing
    ``?`` marks an UNCERTAIN read. Never call on the raw row:
    ``color_match`` needs the mixture intact."""
    s = (color_str or "").strip()
    if not s or s == "?":
        return "?"
    mix = _parse_color_mixture(s)
    if mix is None:  # a plain single word (not a mixture)
        w = s.lower()
        return f"{w}?" if uncertain else w
    word = mix[0][0]
    second = None
    if rich and len(mix) > 1 and mix[1][1] >= 0.20 and mix[1][0] != word:
        w2 = mix[1][0]
        w_ach = _COLOR_FAMILY.get(word) == "achromatic"
        w2_ach = _COLOR_FAMILY.get(w2) == "achromatic"
        if (w_ach != w2_ach) or (not w_ach and not w2_ach
                                 and _COLOR_FAMILY.get(word) != _COLOR_FAMILY.get(w2)):
            second = w2
    head = f"{shade} " if shade in ("dk", "lt") else ""
    body = f"{word}/{second}" if second else word
    return f"{head}{body}?" if uncertain else f"{head}{body}"


def _arc_lengths(poly_en):
    out = [0.0]
    for i in range(len(poly_en) - 1):
        out.append(out[-1] + math.hypot(poly_en[i + 1][0] - poly_en[i][0],
                                        poly_en[i + 1][1] - poly_en[i][1]))
    return out


# Keys already calibrated to [0,1] with a 0.5 NEUTRAL (the color_<word>
# columns, name_match): CENTERED (v-0.5), never min-max'd, so the neutral and
# the color_conf shrink survive into the composite.
def _is_calibrated_unit_key(k) -> bool:
    return k == "name_match" or k in COLOR_COLUMNS


# Cap on the adaptive composite-score stamp (geom_program._fmt_float's render
# bound matches it): raw composites still colliding at this many decimals are
# float-noise apart and stamp equal, a TRUE tie.
_SCORE_MAX_DP = 10


def score_by(items, keys, weights=None, abs_keys=()) -> list[dict]:
    """Fuse several scalar keys into ONE composite, sign-aware score; return items
    ordered BEST-FIRST (rank 0 = highest). The multi-key terminal sort; the caller
    (geom_program.score_by) passes EXACTLY the keys+weights the agent's emphasis
    names -- nothing is auto-added here. score = sum(weight_k * term_k):

      - a CALIBRATED unit key (a color_<word> column / name_match): term =
        (v - 0.5) -- CENTERED on its 0.5 neutral, so a neutral 0.5 -> 0 (no
        pull) and the color_conf shrink stays muted; NEVER min-max'd.
      - any other (unbounded SPATIAL) key (dist_m/arc_m/perp_offset_m): MIN-MAX
        normalized to [0,1] over the rows that CARRY a numeric value, so a key in
        meters fuses with the [-0.5,0.5] color term without its magnitude dominating.
      - a MISSING spatial key on a row, or a DEGENERATE (constant, max==min)
        spatial key: the NEUTRAL 0.5 midpoint -- no pull, and a row missing a
        key is never dropped, only un-boosted.
      - a MISSING ``name_match``: strictly WORSE than any measured value under
        either weight sign (term -0.6 for w>=0, +0.6 for w<0 -- outside the
        measured [-0.5, 0.5] band). Retrieval SCORED that row and it fell
        below threshold, so it never outranks a row whose read merely
        scored low.
      - a MISSING ``color_<word>``: the NEUTRAL 0. Absence there means the
        row has not been LOOKED at (a prior carries a perceived colour only
        after a view passes over it), not that it fails the cue -- and
        unknown belongs between a match and a mismatch.

    A weight's SIGN is its direction: NEGATIVE = lower-raw-is-better (a d_<ref>),
    POSITIVE = higher-raw-is-better (a color_<word> column); weights default to +1.0 each.
    A key in ``abs_keys`` reads |value|, the magnitude term for a SIGNED
    column (off_/vx_/vy_), so w:-1 there means nearest-the-axis-first. Each
    returned item gets ``score`` stamped at ADAPTIVE precision: the fewest
    decimals (floor 4, cap ``_SCORE_MAX_DP``) at which every pair of adjacent
    DISTINCT raw composites stays distinct, so a skew-compressed distribution
    never rounds into a false tie. A stable sort keeps input (fid-sorted)
    order on ties. The absent-key / required-emphasis guards live in
    geom_program.score_by."""
    rows = list(items or [])
    ws = [1.0] * len(keys) if not weights else [float(w) for w in weights]
    _abs = frozenset(abs_keys or ())

    def _raw(it, k):
        v = it.get(k) if isinstance(it, dict) else None
        if (k in _abs and isinstance(v, (int, float))
                and not isinstance(v, bool)):
            return abs(v)
        return v

    spans = []
    for k in keys:
        if _is_calibrated_unit_key(k):
            spans.append(None)
            continue
        vals = [_raw(it, k) for it in rows
                if isinstance(it, dict) and isinstance(it.get(k), (int, float))]
        spans.append((min(vals), max(vals)) if vals else (0.0, 0.0))

    def _term(it, k, span, w):
        v = _raw(it, k)
        present = isinstance(v, (int, float))
        if _is_calibrated_unit_key(k):
            if present:
                return v - 0.5
            # A missing ``name_match`` is a real negative (retrieval scored
            # the row below threshold): outside the [-0.5, 0.5] band on the
            # losing side of the weight's sign. A missing ``color_<word>``
            # means nothing has looked at the row yet (a prior gains a
            # perceived color only once a view passes over it), so unknown
            # sits between a match and a mismatch, at the neutral.
            if k in COLOR_COLUMNS:
                return 0.0
            return -0.6 if w >= 0 else 0.6
        lo, hi = span
        if not present or hi <= lo:
            return 0.5
        return (v - lo) / (hi - lo)

    out = []
    raws = []
    for it in rows:
        raws.append(sum(w * _term(it, k, span, w)
                        for w, k, span in zip(ws, keys, spans)))
        out.append(dict(it) if isinstance(it, dict) else {"value": it})
    # ADAPTIVE-precision stamp (see the docstring).
    dp = 4
    distinct = sorted(set(raws))
    while dp < _SCORE_MAX_DP and any(
            round(a, dp) == round(b, dp)
            for a, b in zip(distinct, distinct[1:])):
        dp += 1
    for nit, s in zip(out, raws):
        # + 0.0 turns a signed -0.0 into 0.0.
        nit["score"] = round(s, dp) + 0.0
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


# --- exact-mask geometry (the `compare` op's distance substrate) --------------
# Relation columns are computed against the EXACT mask as ingested: a
# rasterized footprint realises as its full multi-loop cell boundary
# (`cells_boundary_loops`), a ``world_contour`` as its polygon, a polyline-only
# node as its polyline, a point-only node as its point. No simplification or
# loop selection: d = 0.0 inside/on the footprint, else exact meters to it.


def point_in_polygon(pt, poly) -> bool:
    """Even-odd ray cast: True iff ``pt`` lies inside the CLOSED polygon ``poly``
    (a list of [x, y] vertices; the closing edge last->first is implicit). A
    point exactly on a horizontal edge follows the half-open rule."""
    if not poly or len(poly) < 3:
        return False
    x, y = _xy(pt)
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xin:
                inside = not inside
    return inside


# --- MASK-INTENT BASIS ---------------------------------------------------------
# Each primitive consumes the mask functional that matches its INTENT:
#
#   GAP / CONTAINMENT (d_<ref>, brg_gap_<ref>) -> exact mask-to-mask min
#       distance, 0.0 on containment/contact/crossing (brg_gap_ is then
#       absent: no local direction exists).
#   POSITION (dc_/brg_/along_/off_/vx_/vy_/frame ordinals/groups) -> the
#       mask's AREA CENTROID (:func:`loops_area_centroid`); parked cars are
#       ~5 m apart center-to-center but ~0.5 m edge-to-edge.
#   AREA (footprint_area_m2) -> mask_shape_features.
#
# An object with no realised mask is a POINT-mask, and every functional
# degrades to the point.

# Memory bound on the vectorised (points x segments) blocks: batching only,
# every pair is still computed exactly.
_PAIR_CHUNK = 4_000_000


def loops_area_centroid(loops):
    """TRUE area centroid ``[x, y]`` of a mask realised as boundary loops —
    the signed shoelace centroid summed across ALL loops, so hole loops
    (which the directed cell trace winds opposite to outer loops) subtract
    exactly. A single loop is orientation-independent (the sign cancels).
    ONLY closed chains contribute: the ``% n`` wrap would give an open way a
    phantom closing edge. With no closed chain (a way encloses nothing) this
    falls back to the vertex mean."""
    sa = sx = sy = 0.0
    for lp in (loops or []):
        if not is_closed_chain(lp):
            continue
        n = len(lp)
        for i in range(n):
            x1, y1 = float(lp[i][0]), float(lp[i][1])
            x2, y2 = float(lp[(i + 1) % n][0]), float(lp[(i + 1) % n][1])
            cr = x1 * y2 - x2 * y1
            sa += cr
            sx += (x1 + x2) * cr
            sy += (y1 + y2) * cr
    if abs(sa) < 1e-12:
        pts = [p for lp in (loops or []) for p in lp]
        if not pts:
            return None
        return [sum(float(p[0]) for p in pts) / len(pts),
                sum(float(p[1]) for p in pts) / len(pts)]
    return [sx / (3.0 * sa), sy / (3.0 * sa)]


def loops_area(loops):
    """Total enclosed area (m^2) of a mask realised as boundary loops — the
    absolute signed shoelace sum across ALL loops (hole loops wind opposite
    to outer loops and subtract exactly). 0.0 for a point or polyline: open
    chains are skipped, since the ``% n`` wrap would close a way
    implicitly."""
    sa = 0.0
    for lp in (loops or []):
        if not is_closed_chain(lp):
            continue
        n = len(lp)
        for i in range(n):
            x1, y1 = float(lp[i][0]), float(lp[i][1])
            x2, y2 = float(lp[(i + 1) % n][0]), float(lp[(i + 1) % n][1])
            sa += x1 * y2 - x2 * y1
    return abs(sa) / 2.0


def is_closed_chain(pts) -> bool:
    """A chain ENCLOSES iff it returns to its first vertex.

    The one rule that tells a ring from a way. Every producer emits the
    closing vertex (the bake's ``_world_contour_from_geometry``,
    :func:`cells_boundary_loops`, and the perception contour paths)."""
    return (len(pts) >= 4
            and float(pts[0][0]) == float(pts[-1][0])
            and float(pts[0][1]) == float(pts[-1][1]))


def clip_chains_to_box(chains, box) -> list:
    """``chains`` cut to ``box`` = ``(x0, y0, x1, y1)``, one chain per run
    that stays inside.

    A bake clips a feature's CELLS to its grid while ``world_contour`` runs
    past the tile edge, so chains are cut to the tile before anything
    measures against them.

    A segment crossing the boundary is CUT at it, never dropped whole. A chain
    that exits and re-enters comes back as two. Closure is not preserved: a
    ring the boundary cuts is no longer a ring, and stitching it along the
    tile edge would claim a wall nobody surveyed.
    """
    x0, y0, x1, y1 = (float(box[0]), float(box[1]),
                      float(box[2]), float(box[3]))

    def inside(p):
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    def at(a, b, t):
        # The endpoints come back verbatim: interpolation is not
        # bit-identical, and :func:`chain_component_labels` partitions on
        # exact vertex identity.
        if t <= 0.0:
            return [a[0], a[1]]
        if t >= 1.0:
            return [b[0], b[1]]
        # Clamped: a cut is on the boundary, and the interpolation can land a
        # few ULPs past it.
        return [min(max(a[0] + (b[0] - a[0]) * t, x0), x1),
                min(max(a[1] + (b[1] - a[1]) * t, y0), y1)]

    out: list = []
    for ch in chains or ():
        pts = [[float(p[0]), float(p[1])] for p in ch]
        if len(pts) == 1:
            if inside(pts[0]):
                out.append(pts)
            continue
        run: list = []
        for a, b in zip(pts, pts[1:]):
            seg = clip_t(a, b, (x0, y0, x1, y1))
            if seg is None:
                if run:
                    out.append(run)
                    run = []
                continue
            t0, t1 = seg
            pa, pb = at(a, b, t0), at(a, b, t1)
            if not run:
                run = [pa]
            elif run[-1] != pa:
                # The previous segment left the box before this one entered.
                out.append(run)
                run = [pa]
            if pb != run[-1]:
                run.append(pb)
            if t1 < 1.0:
                out.append(run)
                run = []
        if run:
            out.append(run)
    return [c for c in out if c]


def chain_component_labels(chains) -> list:
    """WHICH place each chain of a mask realisation belongs to.

    The partition :func:`chain_components` counts. A label is the index of
    the lowest-numbered chain in its component, so the labelling depends on
    nothing but the input order.

    Two chains are the same place when they SHARE A VERTEX, or when one
    ENCLOSES the other, which makes a hole loop part of the ring around it
    rather than a place of its own. Whatever is left over is genuinely
    disjoint: two fragments of a speckled mask, a way the tile boundary cut
    into two in-tile pieces.

    Exact vertex identity is the whole rule: no tolerance, no join radius.
    """
    chains = [ch for ch in (chains or []) if ch]
    n = len(chains)
    if n < 2:
        return list(range(n))
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    seen: dict = {}
    for i, ch in enumerate(chains):
        for p in ch:
            k = (float(p[0]), float(p[1]))
            if k in seen:
                union(i, seen[k])
            else:
                seen[k] = i
    if len({find(i) for i in range(n)}) > 1:
        closed = [i for i in range(n) if is_closed_chain(chains[i])]
        for i in closed:
            for j in range(n):
                if j != i and find(j) != find(i) \
                        and point_in_polygon(chains[j][0], chains[i]):
                    union(i, j)
    return [find(i) for i in range(n)]


def chain_components(chains) -> int:
    """How many separate PLACES a mask realisation is
    (:func:`chain_component_labels`). A centre is defined over one place
    and over no more."""
    return len(set(chain_component_labels(chains)))


def mask_geom(chains):
    """Precompute the vectorised arrays of one EXACT mask realisation:
    ``{"V": (N,2) vertices, "A"/"B": (S,2) segment endpoints, "CA"/"CB": the
    subset of those lying on CLOSED chains, "closed"}``.

    ``chains`` is a list of vertex chains ``[[x, y], ...]``: the multi-loop
    cell boundary (:func:`cells_boundary_loops`), a feature's ``world_contour``
    (one chain per part), or a single point as ``[[x, y]]``. Each chain states
    its OWN closure (:func:`is_closed_chain`) and the closing edge is IN the
    chain, so segments are plain consecutive pairs.

    ``CA``/``CB`` carry the closed chains' edges alone (even-odd containment
    is defined over those); ``A``/``B`` carry every edge (distance is defined
    over all). Build ONCE per object per op; :func:`gap_witness` consumes two
    of these."""
    V: list = []
    A: list = []
    B: list = []
    CA: list = []
    CB: list = []
    for ch in (chains or []):
        pts = [(float(p[0]), float(p[1])) for p in ch]
        if not pts:
            continue
        V.extend(pts)
        closed = is_closed_chain(pts)
        for i in range(len(pts) - 1):
            A.append(pts[i])
            B.append(pts[i + 1])
            if closed:
                CA.append(pts[i])
                CB.append(pts[i + 1])
    return {"V": np.asarray(V, dtype=float).reshape(-1, 2),
            "A": np.asarray(A, dtype=float).reshape(-1, 2),
            "B": np.asarray(B, dtype=float).reshape(-1, 2),
            "CA": np.asarray(CA, dtype=float).reshape(-1, 2),
            "CB": np.asarray(CB, dtype=float).reshape(-1, 2),
            "closed": bool(CA)}


def geom_bbox(g):
    """``(minx, miny, maxx, maxy)`` of a :func:`mask_geom`, or ``None`` when it
    has no vertices. Cached on the geom, which is built once per object per op
    and then compared against every item."""
    bb = g.get("BB")
    if bb is None and "BB" not in g:
        V = g.get("V")
        if V is None or len(V) == 0:
            bb = None
        else:
            bb = (float(V[:, 0].min()), float(V[:, 1].min()),
                  float(V[:, 0].max()), float(V[:, 1].max()))
        g["BB"] = bb
    return bb


def bbox_gap(ga, gb) -> float:
    """A LOWER BOUND on :func:`gap_witness`'s distance: the axis-aligned
    separation of the two bounding boxes, 0.0 where they overlap.

    Never greater than the true gap, so a caller taking a MIN over many
    candidates may skip any whose bound already exceeds a gap it holds -- the
    minimum it ends with is the same minimum, not an estimate of one. Returns
    0.0 (prunes nothing) for anything that is not a vertex-bearing geom.
    """
    if not isinstance(ga, dict) or not isinstance(gb, dict):
        return 0.0
    a, b = geom_bbox(ga), geom_bbox(gb)
    if a is None or b is None:
        return 0.0
    dx = max(0.0, a[0] - b[2], b[0] - a[2])
    dy = max(0.0, a[1] - b[3], b[1] - a[3])
    return math.hypot(dx, dy)


#: Slack on a bound comparison, in metres. A box bound never exceeds the gap
#: it bounds mathematically, but the two arithmetic routes can disagree by an
#: ulp; a micron of slack can only ever evaluate MORE members, never fewer.
BOUND_SLACK_M = 1e-6


def stack_bboxes(geoms) -> np.ndarray:
    """``(M, 4)`` of every member's bounding box, ``nan`` where one has none;
    built once per reference column and compared against every item."""
    out = np.full((len(geoms), 4), np.nan, dtype=float)
    for i, g in enumerate(geoms):
        bb = geom_bbox(g) if isinstance(g, dict) else None
        if bb is not None:
            out[i] = bb
    return out


def bbox_bounds(stacked: np.ndarray, g) -> np.ndarray:
    """:func:`bbox_gap` from every row of ``stacked`` to ``g``, vectorised.

    Same lower bound, one numpy pass instead of a call per member. A member
    whose box is unknown reads 0.0 and so prunes nothing, matching
    :func:`bbox_gap`'s own refusal to guess.
    """
    if len(stacked) == 0:
        return np.zeros(0, dtype=float)
    bb = geom_bbox(g) if isinstance(g, dict) else None
    if bb is None:
        return np.zeros(len(stacked), dtype=float)
    dx = np.maximum(0.0, np.maximum(stacked[:, 0] - bb[2],
                                    bb[0] - stacked[:, 2]))
    dy = np.maximum(0.0, np.maximum(stacked[:, 1] - bb[3],
                                    bb[1] - stacked[:, 3]))
    return np.nan_to_num(np.hypot(dx, dy), nan=0.0)


def _inside_mask_arr(P, A, B):
    """Even-odd membership (bool array) of points ``P`` (N,2) against the
    closed-loop edges ``A``->``B`` (S,2). Crossing parity over ALL loops
    together IS exact mask membership (fragments add, holes subtract).

    ONLY THE EDGES THAT CAN CROSS. An edge contributes to a point's parity
    only when ``ylo <= py < yhi``, so testing points in y-order lets a chunk
    spanning ``[lo, hi]`` drop every edge with ``ylo > hi`` or ``yhi <= lo``;
    the parity is the same one (``test_inside_mask_prune_is_exact``).
    """
    out = np.zeros(len(P), dtype=bool)
    if len(P) == 0 or len(A) == 0:
        return out
    # Only the points inside the loops' box: a point outside it crosses an
    # even number of edges.
    inbox = ((P[:, 0] >= min(float(A[:, 0].min()), float(B[:, 0].min())))
             & (P[:, 0] <= max(float(A[:, 0].max()), float(B[:, 0].max())))
             & (P[:, 1] >= min(float(A[:, 1].min()), float(B[:, 1].min())))
             & (P[:, 1] <= max(float(A[:, 1].max()), float(B[:, 1].max()))))
    if not inbox.any():
        return out
    ylo = np.minimum(A[:, 1], B[:, 1])
    yhi = np.maximum(A[:, 1], B[:, 1])
    order = np.flatnonzero(inbox)
    order = order[np.argsort(P[order, 1], kind="stable")]
    step = max(1, int(_PAIR_CHUNK // max(1, len(A))))
    for s in range(0, len(order), step):
        idx = order[s:s + step]
        p = P[idx]
        lo, hi = float(p[:, 1].min()), float(p[:, 1].max())
        sel = (ylo <= hi) & (yhi > lo)
        if not sel.any():
            continue
        a, b = A[sel], B[sel]
        Ax, Ay = a[:, 0][None, :], a[:, 1][None, :]
        Bx, By = b[:, 0][None, :], b[:, 1][None, :]
        px, py = p[:, 0][:, None], p[:, 1][:, None]
        cond = (Ay > py) != (By > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (Bx - Ax) * (py - Ay) / (By - Ay) + Ax
        cross = cond & (px < xin)
        out[idx] = (cross.sum(axis=1) & 1).astype(bool)
    return out


def mask_contains(geom, pts):
    """Vectorised exact-mask containment: bool array over ``pts``, tested
    against the CLOSED chains of a :func:`mask_geom`. A geom with no closed
    chain encloses nothing and contains nothing."""
    P = np.asarray([[float(p[0]), float(p[1])] for p in pts],
                   dtype=float).reshape(-1, 2)
    return _inside_mask_arr(P, geom["CA"], geom["CB"])


# Past this many (point, segment) pairs the argmin is looked for on the pairs
# that can hold it (``_min_pt_to_segs_pruned``).
_PRUNE_PAIRS = 100_000


def _min_pt_to_segs(P, A, B):
    """``(d, i, foot)`` — the exact argmin distance from points ``P`` (N,2)
    to segments ``A``->``B`` (S,2): winning point index + its foot point."""
    if len(P) == 0 or len(A) == 0:
        return float("inf"), -1, None
    if len(P) * len(A) > _PRUNE_PAIRS:
        return _min_pt_to_segs_pruned(P, A, B)
    return _min_pt_to_segs_all(P, A, B)


def _min_pt_to_segs_pruned(P, A, B):
    """The full scan's triple, found on the pairs that can hold it.

    A pair's distance is never below the separation of the point from the
    segments' box, nor below the separation of the segment from the
    points' box. One exact distance from the best-bounded segment to every
    point, and one from the best-bounded point to every segment, gives an
    upper bound the minimum cannot exceed; every point and every segment
    whose bound clears it is dropped, in order, and the full scan runs on
    what is left. The minimal pairs all survive the cut and keep their
    order, so the first of them, the scan's witness, is the same pair,
    computed by the same arithmetic. ``BOUND_SLACK_M`` errs toward keeping.
    """
    sx0, sx1 = np.minimum(A[:, 0], B[:, 0]), np.maximum(A[:, 0], B[:, 0])
    sy0, sy1 = np.minimum(A[:, 1], B[:, 1]), np.maximum(A[:, 1], B[:, 1])
    px0, px1 = float(P[:, 0].min()), float(P[:, 0].max())
    py0, py1 = float(P[:, 1].min()), float(P[:, 1].max())
    bx0, bx1 = float(sx0.min()), float(sx1.max())
    by0, by1 = float(sy0.min()), float(sy1.max())
    bp = np.hypot(np.maximum(0.0, np.maximum(bx0 - P[:, 0], P[:, 0] - bx1)),
                  np.maximum(0.0, np.maximum(by0 - P[:, 1], P[:, 1] - by1)))
    bs = np.hypot(np.maximum(0.0, np.maximum(px0 - sx1, sx0 - px1)),
                  np.maximum(0.0, np.maximum(py0 - sy1, sy0 - py1)))
    s0, p0 = int(bs.argmin()), int(bp.argmin())
    ub = min(_min_pt_to_segs_all(P, A[s0:s0 + 1], B[s0:s0 + 1])[0],
             _min_pt_to_segs_all(P[p0:p0 + 1], A, B)[0])
    keep_p = np.flatnonzero(bp <= ub + BOUND_SLACK_M)
    keep_s = np.flatnonzero(bs <= ub + BOUND_SLACK_M)
    d, i, foot = _min_pt_to_segs_all(P[keep_p], A[keep_s], B[keep_s])
    return d, (int(keep_p[i]) if i >= 0 else -1), foot


def _min_pt_to_segs_all(P, A, B):
    """Every pair, in order: the scan ``_min_pt_to_segs`` is defined by."""
    if len(P) == 0 or len(A) == 0:
        return float("inf"), -1, None
    E = B - A
    L2 = np.einsum("ij,ij->i", E, E)
    L2s = np.where(L2 > 0.0, L2, 1.0)
    best_d2, best_i, best_foot = float("inf"), -1, None
    step = max(1, int(_PAIR_CHUNK // max(1, len(A))))
    for s in range(0, len(P), step):
        p = P[s:s + step]
        W = p[:, None, :] - A[None, :, :]
        t = np.einsum("nsj,sj->ns", W, E) / L2s[None, :]
        t = np.clip(np.where(L2[None, :] > 0.0, t, 0.0), 0.0, 1.0)
        F = A[None, :, :] + t[..., None] * E[None, :, :]
        D = p[:, None, :] - F
        d2 = np.einsum("nsj,nsj->ns", D, D)
        i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
        if float(d2[i, j]) < best_d2:
            best_d2 = float(d2[i, j])
            best_i = s + i
            best_foot = (float(F[i, j, 0]), float(F[i, j, 1]))
    return math.sqrt(best_d2), best_i, best_foot


def _overlaps_box(A, B, C, D):
    """Which segments ``A``->``B`` overlap the bounding box of the segments
    ``C``->``D`` (inclusive on the boundary, so a touch is never dropped)."""
    xmin = min(float(C[:, 0].min()), float(D[:, 0].min()))
    xmax = max(float(C[:, 0].max()), float(D[:, 0].max()))
    ymin = min(float(C[:, 1].min()), float(D[:, 1].min()))
    ymax = max(float(C[:, 1].max()), float(D[:, 1].max()))
    return ((np.maximum(A[:, 0], B[:, 0]) >= xmin)
            & (np.minimum(A[:, 0], B[:, 0]) <= xmax)
            & (np.maximum(A[:, 1], B[:, 1]) >= ymin)
            & (np.minimum(A[:, 1], B[:, 1]) <= ymax))


def _segments_cross(A1, B1, A2, B2):
    """The contact point of a PROPERLY crossing segment pair between two
    segment sets (interiors intersect), or None. Touch/collinear contact
    needs no special case — the vertex-to-segment min already reads 0 there;
    this catches exactly the crossing-without-vertex-contact overlap (a
    plus-sign footprint pair, an open polyline threading a footprint)."""
    if len(A1) == 0 or len(A2) == 0:
        return None
    # Only the segments whose boxes meet: each side keeps, in order, the
    # segments overlapping the other side's box, so the first hit in
    # row-major order is the same pair.
    k1 = _overlaps_box(A1, B1, A2, B2)
    k2 = _overlaps_box(A2, B2, A1, B1)
    if not (k1.all() and k2.all()):
        if not k1.any() or not k2.any():
            return None
        A1, B1, A2, B2 = A1[k1], B1[k1], A2[k2], B2[k2]
    r = B1 - A1
    s = B2 - A2
    step = max(1, int(_PAIR_CHUNK // max(1, len(A2))))
    for k in range(0, len(A1), step):
        a1, b1, rr = A1[k:k + step], B1[k:k + step], r[k:k + step]
        W1 = A2[None, :, :] - a1[:, None, :]
        W2 = B2[None, :, :] - a1[:, None, :]
        o1 = rr[:, None, 0] * W1[..., 1] - rr[:, None, 1] * W1[..., 0]
        o2 = rr[:, None, 0] * W2[..., 1] - rr[:, None, 1] * W2[..., 0]
        W3 = a1[:, None, :] - A2[None, :, :]
        W4 = b1[:, None, :] - A2[None, :, :]
        o3 = s[None, :, 0] * W3[..., 1] - s[None, :, 1] * W3[..., 0]
        o4 = s[None, :, 0] * W4[..., 1] - s[None, :, 1] * W4[..., 0]
        hit = (o1 * o2 < 0.0) & (o3 * o4 < 0.0)
        if hit.any():
            i, j = np.unravel_index(int(np.argmax(hit)), hit.shape)
            rxs = rr[i, 0] * s[j, 1] - rr[i, 1] * s[j, 0]
            t = (W1[i, j, 0] * s[j, 1] - W1[i, j, 1] * s[j, 0]) / rxs
            return (float(a1[i, 0] + t * rr[i, 0]),
                    float(a1[i, 1] + t * rr[i, 1]))
    return None


def gap_witness(ga, gb):
    """GAP intent WITH the argmin witness pair: the EXACT mask-to-mask MIN
    distance between two :func:`mask_geom` realisations + the two points that
    achieve it, ``(gap_m, witness_a, witness_b)``. 0.0 on containment, touch,
    OR boundary crossing (the witnesses then sit at the contact point — the
    caller reads gap 0 as "no local direction"). Exact: the min distance
    between disjoint vertex chains is achieved vertex-to-segment, so the
    symmetric vectorised vertex-to-chain argmin is an exact witness, and the
    proper-crossing test catches the interiors-intersect overlap no vertex
    membership sees. Empty either side -> ``(inf, None, None)``."""
    Va, Vb = ga["V"], gb["V"]
    if len(Va) == 0 or len(Vb) == 0:
        return float("inf"), None, None
    # Containment and crossing both need the two boxes to overlap, and
    # ``bbox_gap`` is 0.0 exactly on overlap (touching included).
    if bbox_gap(ga, gb) <= 0.0:
        if len(gb["CA"]):
            ins = _inside_mask_arr(Va, gb["CA"], gb["CB"])
            if ins.any():
                k = int(np.argmax(ins))
                p = (float(Va[k, 0]), float(Va[k, 1]))
                return 0.0, p, p
        if len(ga["CA"]):
            ins = _inside_mask_arr(Vb, ga["CA"], ga["CB"])
            if ins.any():
                k = int(np.argmax(ins))
                p = (float(Vb[k, 0]), float(Vb[k, 1]))
                return 0.0, p, p
        x = _segments_cross(ga["A"], ga["B"], gb["A"], gb["B"])
        if x is not None:
            return 0.0, x, x
    best = (float("inf"), None, None)
    d, i, foot = _min_pt_to_segs(Va, gb["A"], gb["B"])
    if d < best[0]:
        best = (d, (float(Va[i, 0]), float(Va[i, 1])), foot)
    d, i, foot = _min_pt_to_segs(Vb, ga["A"], ga["B"])
    if d < best[0]:
        best = (d, foot, (float(Vb[i, 0]), float(Vb[i, 1])))
    if not math.isfinite(best[0]):
        # neither side carries a segment (point-mask vs point-mask)
        D = Va[:, None, :] - Vb[None, :, :]
        d2 = np.einsum("nmj,nmj->nm", D, D)
        i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
        best = (math.sqrt(float(d2[i, j])),
                (float(Va[i, 0]), float(Va[i, 1])),
                (float(Vb[j, 0]), float(Vb[j, 1])))
    return best


def _segment_arrays(polys):
    """Bundle of ``[[x, y], ...]`` polylines -> its ``(A, B)`` segment
    endpoint arrays. A one-vertex polyline is a zero-length segment, so a
    point feature meets a line exactly as its position does."""
    A: list = []
    B: list = []
    for p in polys or []:
        P = np.asarray([[float(q[0]), float(q[1])] for q in p],
                       dtype=float).reshape(-1, 2)
        # a non-finite vertex would poison every orientation cross-product
        P = P[np.isfinite(P).all(axis=1)]
        if len(P) >= 2:
            A.append(P[:-1])
            B.append(P[1:])
        elif len(P) == 1:
            A.append(P)
            B.append(P)
    if not A:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(A), np.concatenate(B)


def _pt_to_segs_grid(P, A, B):
    """``(d, F)`` for EVERY (point, segment) pair: distances ``(N, M)`` and
    foot points ``(N, M, 2)`` from points ``P`` to segments ``A``->``B``."""
    E = B - A
    L2 = np.einsum("mj,mj->m", E, E)
    L2s = np.where(L2 > 0.0, L2, 1.0)
    W = P[:, None, :] - A[None, :, :]
    t = np.einsum("nmj,mj->nm", W, E) / L2s[None, :]
    t = np.clip(np.where(L2[None, :] > 0.0, t, 0.0), 0.0, 1.0)
    F = A[None, :, :] + t[..., None] * E[None, :, :]
    D = P[:, None, :] - F
    return np.sqrt(np.einsum("nmj,nmj->nm", D, D)), F


def polyline_crossings(polys_a, polys_b):
    """``(points, nearest_m)`` — every DISTINCT place two BUNDLES of exact
    vertex chains MEET, and their nearest approach.

    A bundle is a list of ``[[x, y], ...]`` chains because ONE road is
    routinely several ways. They MEET where they TOUCH: distance exactly
    zero, at a shared vertex, a vertex on the other's segment, or a proper
    crossing of two segment interiors. Nothing is dilated or clustered:
    roads that meet share an OSM node, so their contact is exactly 0.0.

    ``points`` holds one entry PER CROSSING, deduplicated at the 2 dp the
    geometry is stored to (the resolution of each crossing row's ``[x, y]``
    fid). It is never collapsed to a centroid: a single centre over three
    crossings lands off all three. Empty ``points`` with a finite
    ``nearest_m`` is the caller's refusal material.
    """
    A1, B1 = _segment_arrays(polys_a)
    A2, B2 = _segment_arrays(polys_b)
    if len(A1) == 0 or len(A2) == 0:
        return [], float("inf")
    r2 = B2 - A2
    hits: list = []
    nearest = float("inf")
    step = max(1, int(_PAIR_CHUNK // max(1, len(A2))))
    for k in range(0, len(A1), step):
        a1, b1 = A1[k:k + step], B1[k:k + step]
        r1 = b1 - a1
        d1, F1 = _pt_to_segs_grid(a1, A2, B2)
        d2, F2 = _pt_to_segs_grid(b1, A2, B2)
        d3, F3 = _pt_to_segs_grid(A2, a1, b1)
        d4, F4 = _pt_to_segs_grid(B2, a1, b1)
        d = np.stack([d1, d2, d3.T, d4.T])
        w = np.stack([
            (a1[:, None, :] + F1) * 0.5,
            (b1[:, None, :] + F2) * 0.5,
            (A2[None, :, :] + F3.transpose(1, 0, 2)) * 0.5,
            (B2[None, :, :] + F4.transpose(1, 0, 2)) * 0.5,
        ])
        pick = np.argmin(d, axis=0)
        dmin = np.take_along_axis(d, pick[None], 0)[0]
        wmin = np.take_along_axis(w, pick[None, ..., None], 0)[0]
        # PROPER crossing (interiors intersect): distance 0 at the exact
        # contact, whatever the vertex spacing.
        W1 = A2[None, :, :] - a1[:, None, :]
        W2 = B2[None, :, :] - a1[:, None, :]
        o1 = r1[:, None, 0] * W1[..., 1] - r1[:, None, 1] * W1[..., 0]
        o2 = r1[:, None, 0] * W2[..., 1] - r1[:, None, 1] * W2[..., 0]
        W3 = a1[:, None, :] - A2[None, :, :]
        W4 = b1[:, None, :] - A2[None, :, :]
        o3 = r2[None, :, 0] * W3[..., 1] - r2[None, :, 1] * W3[..., 0]
        o4 = r2[None, :, 0] * W4[..., 1] - r2[None, :, 1] * W4[..., 0]
        X = (o1 * o2 < 0.0) & (o3 * o4 < 0.0)
        if X.any():
            rxs = (r1[:, None, 0] * r2[None, :, 1]
                   - r1[:, None, 1] * r2[None, :, 0])
            t = ((W1[..., 0] * r2[None, :, 1] - W1[..., 1] * r2[None, :, 0])
                 / np.where(rxs != 0.0, rxs, 1.0))
            P = a1[:, None, :] + t[..., None] * r1[:, None, :]
            dmin = np.where(X, 0.0, dmin)
            wmin = np.where(X[..., None], P, wmin)
        nearest = min(nearest, float(dmin.min()))
        keep = dmin <= 0.0
        if keep.any():
            hits.append(wmin[keep])
    if not hits:
        return [], nearest
    pts = np.round(np.concatenate(hits), 2)
    return ([list(p) for p in sorted({(float(x), float(y)) for x, y in pts})],
            nearest)


def _collapse_collinear(corners):
    """Drop interior vertices of straight runs on a closed rectilinear loop."""
    n = len(corners)
    if n < 3:
        return list(corners)
    out = []
    for i in range(n):
        pr, pc = corners[i - 1]
        r, c = corners[i]
        nr, nc = corners[(i + 1) % n]
        # keep the vertex iff the incoming and outgoing directions differ
        if (r - pr, c - pc) != (nr - r, nc - c):
            out.append((r, c))
    return out or [corners[0]]


def cells_boundary_loops(cells, origin_xy, cell_m):
    """Trace the EXACT boundary of a rasterized footprint: every closed loop
    (the outer boundary of every fragment and every hole) with collinear runs
    collapsed, which is lossless.

    ``cells`` is an iterable of (row, col) tile-grid cells (numpy ``(N,2)`` or
    a list of pairs). Returns a list of CLOSED loops, each a list of [x, y]
    world meters ending on the vertex it started from
    (:func:`is_closed_chain`), or None when no cells. The directed trace winds
    hole loops opposite to outer loops, so even-odd membership over all loops
    reproduces exact cell-union membership and :func:`loops_area_centroid`'s
    signed sum subtracts holes exactly."""
    cs = {(int(rc[0]), int(rc[1])) for rc in (cells if cells is not None else [])}
    if not cs:
        return None
    # Directed boundary edges on cell-corner coords, consistently oriented so
    # loops chain head-to-tail: for each cell, each side with no 4-neighbor
    # contributes one corner->corner edge (clockwise in (row, col) space).
    edges: dict[tuple, list] = {}
    for (r, c) in cs:
        if (r - 1, c) not in cs:
            edges.setdefault((r, c), []).append((r, c + 1))
        if (r, c + 1) not in cs:
            edges.setdefault((r, c + 1), []).append((r + 1, c + 1))
        if (r + 1, c) not in cs:
            edges.setdefault((r + 1, c + 1), []).append((r + 1, c))
        if (r, c - 1) not in cs:
            edges.setdefault((r + 1, c), []).append((r, c))
    loops: list = []
    while edges:
        start = min(edges)  # deterministic loop seed
        loop = [start]
        cur = edges[start].pop()
        if not edges[start]:
            del edges[start]
        while cur != start and cur in edges:
            loop.append(cur)
            nxt = edges[cur].pop()
            if not edges[cur]:
                del edges[cur]
            cur = nxt
        if cur != start:
            # open chain (pinched corner artifact) — skip, keep tracing others
            continue
        lp = _collapse_collinear(loop)
        if len(lp) >= 3:
            loops.append(lp)
    if not loops:
        return None
    tx, ty = float(origin_xy[0]), float(origin_xy[1])
    m = float(cell_m)
    return [[[tx + c * m, ty + r * m] for (r, c) in (*lp, lp[0])]
            for lp in loops]


def mask_shape_features(cells, cell_m):
    """``{"footprint_area_m2": #unique cells x cell_m**2}`` of a rasterized
    footprint (``(row, col)`` cells), or ``None`` when it has no cells."""
    cs = {(int(rc[0]), int(rc[1])) for rc in (cells if cells is not None else [])}
    if not cs:
        return None
    m = float(cell_m)
    return {"footprint_area_m2": round(len(cs) * m * m, 1)}


def contour_shape_features(chains):
    """``{"footprint_area_m2": ...}`` summed over a feature's CLOSED world
    chains (:func:`loops_area`), or ``None`` when no closed chain encloses
    area: a way's "area" would be the accident of closing it."""
    rings = [ch for ch in (chains or []) if is_closed_chain(ch)]
    area = loops_area(rings)
    if area <= 1e-9:
        return None
    return {"footprint_area_m2": round(area, 2)}
