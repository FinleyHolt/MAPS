"""Frame-correct geometry helpers — the math behind the ``analyze_scene`` ops.

This module is the **single source of truth** for the spatial primitives the
reasoner needs to disambiguate same-class instances ("the 4th car from the
corner", "the white car between A and B", "on the road left of the church").
The ``analyze_scene`` op-program executor
(:mod:`maps.search.analyze_scene.geom_program`) calls these primitives to run each op,
and the offline distillation teacher harness imports the SAME executor — so
the live student and the teacher compute over *identical* geometry.

Design contract (load-bearing):

- **Pure + data-in-args.** No graph, no I/O, ``import math``/``numpy``
  only (numpy vectorises the exact-mask gap substrate). Every input
  is plain data: points are ``[x, y]`` world meters; centerlines are
  ``[[x, y], ...]`` world-meter polylines (the caller converts the baked grid
  centerline to world xy before injection). The agent never re-derives geometry
  from a mask.

- **ONE frame: x = east, y = NORTH.** World coordinates are
  easting/northing (the per-city ``benchmarks/citynav/georef`` affine from
  EPSG:27700 is an orientation-preserving similarity, det>0, ~1 m residual;
  row 0 of every repo grid is the SOUTH edge). Cardinal words, bearings,
  signed sides and annotator-facing left/right are ALL computed in this
  same frame — compass words map through the one :data:`COMPASS_UNIT`
  table, and viewer/screen handedness is a proper rotation of it (right =
  cw(forward) in a y-up frame). There is no second frame anywhere in the
  engine; do not introduce one. (Grid-vs-world is an affine map with
  positive scale, so world-xy results equal grid ``(col, row)`` results in
  meters — this is what the sign-locked frame tests assert.)

- **Composable primitives only — never prompt-specific solvers.** A solver for
  "Nth car from the corner of A and B" would just move the 8-tool bloat into the
  library. The agent composes ``frame`` / ``frame_polyline``
  / ``bearing`` into the relation the instruction actually states.

- **View is orientation only.** The pool's own axis is measured once
  (:func:`_row_axis_direction` for the auto frame; the polyline chart's
  arc-length / lateral offset when the reference is a road/polyline). The
  DECLARED frame (``frame``: relative / viewer / auto) never bands anything:
  it only signs the metric view coords (which end is left) — there is no
  structure pass. TOPOLOGICAL containment lives in the world-state graph
  tools, not here.
"""

from __future__ import annotations

import math

import numpy as np

# 8-point compass wedges over compass bearing (0=N, clockwise). The
# canonical wedge table for the tool boundary.
_COMPASS8 = (
    (22.5, "north"), (67.5, "northeast"), (112.5, "east"), (157.5, "southeast"),
    (202.5, "south"), (247.5, "southwest"), (292.5, "west"), (337.5, "northwest"),
)

# THE compass-word unit vectors, world frame (x east, y north). EIGHT winds,
# the same words :func:`_compass8` emits — a column that says "southeast"
# beside an operation that cannot hear the word is how the 8-out/4-in
# asymmetry started, and it cost `stand(facing="southeast")` a silent fall
# through to a lookup for a place named "southeast". Every consumer (frame
# ops here, the crop rotation in reasoner_loop, the band rule in
# cardinal_side) reads this one table — a second hand-written copy is how a
# silent mirror starts, and there were five.
_SQRT_HALF = math.sqrt(0.5)
COMPASS_UNIT = {
    "north": (0.0, 1.0), "northeast": (_SQRT_HALF, _SQRT_HALF),
    "east": (1.0, 0.0), "southeast": (_SQRT_HALF, -_SQRT_HALF),
    "south": (0.0, -1.0), "southwest": (-_SQRT_HALF, -_SQRT_HALF),
    "west": (-1.0, 0.0), "northwest": (-_SQRT_HALF, _SQRT_HALF),
}

# The four axis-aligned winds, DECLARED as a subset rather than copied. An
# op that takes four names THIS to say which four, so widening the table
# above can never silently widen an operation. Two declare it: the band rule
# needs one orthogonal axis to band on and a diagonal has none
# (:mod:`cardinal_side`), and the sentence language's `looking` slot is four
# winds because its grammar says so.
CARDINALS = frozenset({"north", "south", "east", "west"})


def ccw(v):
    """Counter-clockwise quarter-turn in the y-up world frame — THE
    handedness law, shared by every left/right consumer (frame ops here,
    the crop view-up table in reasoner_loop). In a y-up aerial view a
    viewer's RIGHT is ``cw(forward)``: looking north (0,1) -> right =
    east (1,0)."""
    return (-v[1], v[0])


def cw(v):
    """Clockwise quarter-turn in the y-up world frame (see :func:`ccw`)."""
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
# Color matching is by perceptual FAMILY, not Lab distance. Reason (adversarially
# measured): perception assigns ONE color by nearest-palette over the median masked
# RGB, and the dominant slip is WITHIN a family -- a gray car reads "silver"/"white"
# under glare or "black" under shadow (the same car voted {gray:2,white:1} in tests);
# a red car reads "maroon". A Lab-distance threshold drops exactly these (gray<->white
# dE 41, red<->maroon dE 62) and so HARD-DROPS the GT -- the lean-pool / binding-gap
# failure. Family grouping keeps every within-family slip (GT-safe) and separates
# only across families (gray car vs RED cue). Color stays a SOFT pre-filter the
# agent reasons over; GEOMETRY (between/order) is the hard discriminator.
_COLOR_SPELL = {"gray": "gray"}  # spelling only
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
    """Parse a C3b deterministic color-MIXTURE string ("red 0.45 / dark 0.30")
    into ``[(word, fraction), ...]``; ``None`` when ``s`` is not a mixture (a
    plain single word keeps the legacy word path byte-identical). The delimiter
    is ' / ' (the current producer, chosen so the mixture never column-breaks
    the model-facing markdown-grid surface); legacy ' | ' strings are ALSO
    accepted (both normalized below) so any cached/older bundle still parses.
    Neither a color word (alpha) nor a fraction contains '/' or '|', so the
    normalization is unambiguous. Each part must be exactly ``<word> <fraction
    in [0,1]>``. Pure string parsing (this module's no-new-imports contract: no
    ``re``)."""
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
        comps.append((_COLOR_SPELL.get(word, word), frac))
    return comps or None


def color_match(cand_color: str, cue_color: str, conf: float | None = None) -> float:
    """FAMILY-based soft color match, optionally CONFIDENCE-shrunk — the value
    of a ``color_<word>`` column. 1.0 = same word;
    0.8 = same perceptual family (achromatic white/silver/gray/black, red/maroon, the
    warm group) so a perceived "silver"/"white" still matches cue "gray" -- a naming
    slip can NEVER drop the GT; 0.0 = different family; absent/unknown color -> 0.5.

    ``cand_color`` may also be a C3b deterministic color MIXTURE
    ("red 0.45 / black 0.30", the real-SAM extraction; legacy ' | ' also
    parses): the cue is scored
    against EVERY component word with the component's pixel fraction as its
    weight -- each component's family-soft raw is shrunk toward the 0.5
    neutral by its fraction (``0.5 + frac*(raw - 0.5)``, the same shrink form
    as ``conf``) and the best component wins. A matching 0.45-fraction "red"
    scores 0.725 (> the 0.5 absent floor); a mismatching mixture sinks below
    it in proportion to its dominance. Plain single-word candidates never
    enter this path, so today's word behavior is byte-identical.

    ``conf`` is perception's per-detection color confidence in [0,1] (RGB margin x a
    per-color reliability prior). When given, the family-soft score is shrunk toward
    the 0.5 neutral floor: ``0.5 + conf*(raw - 0.5)``. A LOW-conf color collapses
    toward 0.5 -- it neither VETOES the GT (a low-conf cross-family mislabel stays >=
    the 0.4 find/select threshold, never hard-dropped) nor contributes much signal in
    either direction; a HIGH-conf color keeps ~today's discriminating power. ``conf=
    None`` (the offline tile-REBUILD surface carries no perceived color/confidence)
    returns the family-soft score UNCHANGED -- BYTE-IDENTICAL to the pre-confidence
    behavior (offline-rebuild parity, gated by verify_surface_commit_parity.py).
    Color stays a SOFT signal the agent reasons over; geometry is the hard
    discriminator."""
    b = _COLOR_SPELL.get((cue_color or "").strip().lower(), (cue_color or "").strip().lower())
    mix = _parse_color_mixture(cand_color or "")
    if mix is not None:
        if not b:
            raw = 0.5
        else:
            raw = max(0.5 + frac * (_color_word_raw(w, b) - 0.5) for w, frac in mix)
    else:
        a = _COLOR_SPELL.get((cand_color or "").strip().lower(), (cand_color or "").strip().lower())
        raw = _color_word_raw(a, b)
    if conf is None:
        return raw
    c = 0.0 if conf < 0.0 else (1.0 if conf > 1.0 else float(conf))
    return 0.5 + c * (raw - 0.5)


# --- color INTERFACE: closed enum + open->closed map + token-minimal surface cell ----------
# ONE closed vocabulary the agent searches with (the find ``color`` arg), the extractor
# emits, and the surface shows. Under vLLM 0.24 STRICT the enum is a hard grammar constraint on the
# tool call; ``normalize_color`` is the executor-side safety net (and the map the agent applies at
# reason-time: e.g. cue "burgundy"->red, "navy"->blue).
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
    """Map a (possibly open-world) cue color onto ``COLOR_ENUM``; identity if already an enum
    member; falls back to the raw word (the caller decides whether an off-enum word is a loud
    error — the color-column resolver — or flows into ``color_match``'s family/substring path)."""
    w = _COLOR_SPELL.get((word or "").strip().lower(), (word or "").strip().lower())
    if w in COLOR_ENUM:
        return w
    return _COLOR_SYNONYMS.get(w, w)


# The named color-column family the agent sorts on: ``color_<word>`` with word
# on the closed palette (== ``COLOR_ENUM``, which is exactly the vocabulary the
# M6 extractor emits after its vdW->vocab fold — perception/color.py
# ``_VDW_TO_VOCAB``). Each column's value is ``color_match(row.color, word,
# row.color_conf)``; rows without a perceived color omit the column. The set
# drives the calibrated-unit treatment in :func:`score_by` and the sortable-
# evidence carry in the frame charts.
COLOR_COLUMNS = frozenset("color_" + w for w in COLOR_ENUM)


def compact_color_cell(color_str: str, shade: "str | None" = None, uncertain: bool = False,
                       rich: bool = True) -> str:
    """RENDER-ONLY: a mixture string ("red 0.45 / black 0.30") + optional shade/uncertain ->
    a token-minimal color tag ``[dk|lt ]word[/word2][?]`` (~1-2 tokens vs the ~12 of the mixture).
    ``word2`` shows only when ``rich`` (a color-name posterior 2nd, i.e. the M6 extractor -- the
    palette-MODE mixture's 2nd component is glass/shadow noise, not a color alternative) AND it
    crosses achromatic<->chromatic or a hue family (the recovery / two-tone signal). A trailing
    ``?`` means the read is UNCERTAIN -- the same signal as the legacy masked ``?`` -- and the agent
    resolves it with a viewpoint (the confirmed:false + viewpoints guidance). NEVER call on the raw
    row: ``color_match`` / ``_parse_color_mixture`` need the mixture intact. Pure string ops."""
    s = (color_str or "").strip()
    if not s or s == "?":
        return "?"
    mix = _parse_color_mixture(s)
    if mix is None:  # a plain single word (not a mixture)
        w = _COLOR_SPELL.get(s.lower(), s.lower())
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


# Keys already calibrated to [0,1] with a 0.5 NEUTRAL (the color_<word> column
# family, name_match from the query find): CENTERED (v-0.5), never min-max'd, so the
# 0.5-neutral floor + the color_conf shrink-toward-0.5 survive into the composite
# (min-max would re-spread them to the full range and erase both -- the
# gate-via-sort demotion this prevents).
def _is_calibrated_unit_key(k) -> bool:
    return k == "name_match" or k in COLOR_COLUMNS


# Sanity cap on the adaptive composite-score stamp (and the render bound in
# geom_program._fmt_float matches it): raw composites still colliding at this
# many decimals are float-noise apart -- stamped equal, a TRUE tie.
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
    A key in ``abs_keys`` reads |value| — the magnitude term for a SIGNED
    column (off_/vx_/vy_), so w:-1 there means nearest-the-axis-first. Each
    returned item gets ``score`` stamped at ADAPTIVE precision: the fewest
    decimals (floor 4, cap ``_SCORE_MAX_DP``) at which every pair of adjacent
    DISTINCT raw composites stays distinct -- a skew-compressed distribution
    (one outlier min-maxing the rest into a hair's width) must never round
    distinct scores into a false all-zero tie (arm-C7 F3, 598-row 0.0 tie);
    the tie annotation downstream keys on stored scores, so it fires only on
    TRUE ties. A stable sort keeps input (fid-sorted)
    order on ties. Pure + data-in-args; the absent-key / required-emphasis guards
    live in geom_program.score_by (fail-loud BEFORE this is called)."""
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
            # ABSENCE IS NOT THE SAME FACT ON BOTH CALIBRATED KEYS.
            #
            # A missing ``name_match`` means the retrieval SCORED this row
            # and it fell below threshold: a real negative, which must not
            # outrank a row whose read merely scored low. It sits strictly
            # outside the measured [-0.5, 0.5] band on the LOSING side of
            # the weight's sign.
            #
            # A missing ``color_<word>`` means NOTHING HAS LOOKED AT IT YET.
            # A prior acquires a perceived colour only once a view passes
            # over it (``search_loop.upsert_prior_percept``), and 0 of the
            # 5,581 cold anchors on the calcity bake carry one. Scoring that
            # as worse-than-any-measured ranked a SEEN GREY house above
            # every UNSEEN house under ``color="white"`` (0.27 against
            # -0.6) -- a coverage ordering wearing a colour's name, on a
            # bench where the target is usually the unseen one. Unknown
            # belongs BETWEEN a match and a mismatch, so it is the neutral.
            #
            # That is also what the producer has always documented --
            # ``geom_program._attach_color_column``: "rows without a
            # perceived color omit the column (the existing missing-value
            # idiom -- score_by treats it as the neutral)". The two halves
            # disagreed; the consumer now honours the contract instead of
            # the promise being quietly false.
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
    # ADAPTIVE-precision stamp: the fewest decimals (floor 4, cap
    # _SCORE_MAX_DP) at which adjacent DISTINCT raw composites stay distinct
    # once rounded. Raw sums that still collide at the cap (float-noise
    # hair-widths) stamp equal -- a TRUE tie.
    dp = 4
    distinct = sorted(set(raws))
    while dp < _SCORE_MAX_DP and any(
            round(a, dp) == round(b, dp)
            for a, b in zip(distinct, distinct[1:])):
        dp += 1
    for nit, s in zip(out, raws):
        # + 0.0 normalizes a signed -0.0 (round of a tiny negative sum) so a
        # zero score is ONE value everywhere it is stored, compared, served.
        nit["score"] = round(s, dp) + 0.0
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


# Axial concentration above which the NN row axis is taken as measured fact
# rather than re-scored against PCA. 1.0 is a perfectly aligned local lattice.
_NN_AXIS_TRUST: float = 0.7


def _row_partition_tightness(P, u_world, min_row_gap_m=4.0, gap_factor=2.5):
    """How TIGHT the rows are when ``P`` (world xy) is split across the direction
    ``u_world``. Projects every point to its across-axis offset, gap-splits into rows,
    then returns the size-weighted mean
    within-row perpendicular RMS spread (meters) — LOW means the axis carves clean,
    tight parking rows; HIGH means it cuts diagonally across rows. This is the
    data-driven score used to choose between the nearest-neighbor axis and the PCA
    axis (and to resolve the NN axis's 90 deg ambiguity): pick whichever candidate
    direction yields the tighter rows, not a hand-tuned heuristic. Frame-independent
    (a metric spread), so it runs in world xy directly."""
    ux, uy = u_world
    nrm = math.hypot(ux, uy) or 1.0
    ux, uy = ux / nrm, uy / nrm
    cx = sum(p[0] for p in P) / len(P)
    cy = sum(p[1] for p in P) / len(P)
    perps = sorted((p[0] - cx) * (-uy) + (p[1] - cy) * ux for p in P)
    if len(perps) < 2:
        return 0.0
    gaps = [perps[i + 1] - perps[i] for i in range(len(perps) - 1)]
    med = sorted(gaps)[len(gaps) // 2] or 1.0
    thr = max(float(min_row_gap_m), float(gap_factor) * med)
    rows = [[perps[0]]]
    for i in range(1, len(perps)):
        if perps[i] - perps[i - 1] > thr:
            rows.append([])
        rows[-1].append(perps[i])
    tot = wt = 0.0
    for r in rows:
        if len(r) < 2:
            continue
        mu = sum(r) / len(r)
        # BESSEL. Dividing by len(r) is the biased (max-likelihood) spread, low
        # by sqrt((n-1)/n) -- 0.816 for a band of 3, 0.894 for a band of 5. A
        # partition into MORE, SMALLER bands therefore scores ~9% tighter for
        # identical underlying jitter, so this metric systematically rewarded
        # splitting more finely and picked the 5-column partition over the
        # 3-row one on a jittered 3x5 lot. n-1 removes the bias so the score
        # compares partitions rather than their band sizes.
        rms = math.sqrt(sum((v - mu) ** 2 for v in r) / (len(r) - 1))
        tot += rms * len(r)
        wt += len(r)
    return tot / wt if wt else 1e9


def _row_axis_direction(P, nn_radius_m=9.0):
    """Robust row-running unit direction for a (possibly multi-row, near-square)
    cluster, in WORLD xy. Returns ``(u_world, confidence, source)`` — ``u_world`` is a
    unit ``(ex, ey)`` along the recovered row direction, ``source`` in
    {``"nn"``, ``"pca"``}.

    A single global PCA over a near-square 2D parking block is ill-conditioned: its
    eigenvalues are nearly equal so the major eigenvector lands on a row-CROSSING
    diagonal (measured 18-147 deg off the true row direction on real lots). We instead
    estimate orientation from LOCAL structure — each point's bearing to its nearest
    neighbor WITHIN ``nn_radius_m`` (a tight in-row neighbor, since cars in a row are
    the closest spacing), combined as AXIAL data (period 180 deg: angle-doubled,
    distance-weighted circular mean). The resultant length is a [0,1] confidence.

    Both the NN axis and a whole-cluster PCA axis are then SCORED by
    :func:`_row_partition_tightness` (the size-weighted within-row spread the actual
    gap-split partition would produce), and the direction giving the TIGHTER rows wins.
    This is a data-driven choice, not a hand heuristic: it simultaneously (a) resolves
    the NN axis's 90 deg ambiguity (row-spacing < car-spacing can make the nearest
    neighbor the ACROSS-row car) and (b) falls back to PCA when the lot has no clean
    local lattice for NN to lock onto — exactly when PCA's major axis is the better
    bet. The world frame IS (east, north), so the axis is computed and
    returned in world xy directly."""
    XY = [_xy(p) for p in P]
    n = len(XY)
    C = S = W = 0.0
    used = 0
    for i in range(n):
        best = None
        for j in range(n):
            if i == j:
                continue
            dx, dy = XY[j][0] - XY[i][0], XY[j][1] - XY[i][1]
            d = math.hypot(dx, dy)
            if d < 1e-6:
                continue
            if best is None or d < best[0]:
                best = (d, dx, dy)
        if best is None or best[0] > nn_radius_m:
            continue
        d, dx, dy = best
        th = math.atan2(dx, dy)            # compass bearing: atan2(d_east, d_north)
        w = 1.0 / d                        # tighter pairs (more trustworthy) weigh more
        C += w * math.cos(2.0 * th)
        S += w * math.sin(2.0 * th)
        W += w
        used += 1
    pca = _pca_direction(P)
    conf = 0.0
    cand = []
    if W > 0 and used >= max(3, n // 6):
        phi = 0.5 * math.atan2(S, C)
        conf = math.hypot(C, S) / W
        u_world = (math.sin(phi), math.cos(phi))  # unit dir matching atan2(dx, dy)
        cand.append((u_world, "nn"))
    cand.append((pca, "pca"))
    # Pick the direction whose gap-split partition gives the tightest rows --
    # but SEED FROM NN, not PCA. On a regular lot both partitions are perfectly
    # tight and score EXACTLY 0.0, so the strict `<` never fires and whichever
    # direction seeds `best` wins by default. Seeding with PCA silently banded
    # every axis-aligned lot deeper than it is wide 90 deg off (3x5, 4x6, 5x10
    # all counted ACROSS their rows), because PCA takes the larger-variance
    # axis and that is the across-row one there. NN measures the pool's LOCAL
    # lattice; PCA is the documented fallback for when there is no lattice to
    # measure. A tie must therefore fall to NN.
    best_u, best_src = cand[0]
    best_score = _row_partition_tightness(P, best_u)
    # A CONFIDENT NN axis is not up for re-litigation. On a regular lot the two
    # partitions are genuinely equally tight -- a grid has clean structure in
    # BOTH directions -- so the spread metric decides by coin flip, and it was
    # overriding a conf=1.0 local-lattice measurement about half the time. What
    # actually distinguishes a row from a column is SPACING (5 m along vs 12 m
    # across), which a perpendicular-spread score discards by construction.
    # conf is exactly the "is there a clean local lattice" measure the docstring
    # says PCA is the fallback for, so gate on it.
    if not (best_src == "nn" and conf >= _NN_AXIS_TRUST):
        for u, src in cand[1:]:
            sc = _row_partition_tightness(P, u)
            if sc < best_score - 1e-6:
                best_u, best_src, best_score = u, src, sc
    nrm = math.hypot(*best_u) or 1.0
    # Report the NN orientation concentration as the confidence ONLY when the NN axis
    # was actually chosen; a PCA-chosen axis has no local-lattice concentration, so 0
    # (so the beacon never claims confidence the chosen axis doesn't have).
    out_conf = conf if best_src == "nn" else 0.0
    return (best_u[0] / nrm, best_u[1] / nrm), round(out_conf, 3), best_src


def _pca_direction(P):
    """Closed-form principal eigenvector of the point cluster's covariance, in WORLD
    xy (frame-invariant). Used as the fallback axis when local NN structure is too
    sparse to estimate a row direction. Returns a unit ``(ex, ey)`` in world xy."""
    n = len(P)
    mx = sum(p[0] for p in P) / n
    my = sum(p[1] for p in P) / n
    sxx = sum((p[0] - mx) ** 2 for p in P) / n
    syy = sum((p[1] - my) ** 2 for p in P) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in P) / n
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    lam = tr / 2.0 + math.sqrt(max(0.0, tr * tr / 4.0 - det))
    if abs(sxy) > 1e-12:
        ex, ey = lam - syy, sxy
    elif sxx >= syy:
        ex, ey = 1.0, 0.0
    else:
        ex, ey = 0.0, 1.0
    nrm = math.hypot(ex, ey) or 1.0
    return ex / nrm, ey / nrm


def _u(v):
    n = math.hypot(v[0], v[1]) or 1.0
    return (v[0] / n, v[1] / n)


# Side word -> which screen edge the NAMED reference sits on. 'front/near' read as
# 'bottom' and 'back/far/behind' as 'top' (the reference defines the near/far edge).
_FRAME_SIDE = {
    "right": "right", "left": "left",
    "top": "top", "above": "top", "back": "top", "behind": "top", "rear": "top", "far": "top",
    "bottom": "bottom", "below": "bottom", "front": "bottom", "near": "bottom", "nearest": "bottom",
}


def frame(pts, reference, frame_type="relative", side=None) -> dict:
    """Reproject candidates into the DECLARED view (ISO-Space OLINK / Levinson
    2003) so a view-relative cue becomes a coordinate look-up. ORIENTATION
    ONLY: the declared view signs the counting columns; there is no structure
    pass, so a declared frame changes signs and coordinates, never any
    banding.

    - ``"relative"`` — orient the view so the NAMED ``reference`` (a landmark
      POINT ``[x, y]``) sits at ``side`` (right|left|top|bottom; aliases
      above/below/front/back/near/far). ``op_x`` = +RIGHT, ``op_y`` = +UP in
      that view. A POLYLINE reference is the curvilinear chart
      (:func:`frame_polyline`) — never this proper-rotation view.
    - ``"viewer"`` — ``reference = (from_xy, toward)``: stand at ``from``,
      look at ``toward`` (a point, or a compass word).
    - ``"auto"`` — the pool's own measured axis, east-default sign.

    Returns ``{frame:{...}, reference_op?, pts:[{fid?, xy, op_x, op_y, class?,
    color?}]}``.
    """
    ft = str(frame_type or "relative").strip().lower()
    if ft == "auto":
        return _frame_auto(pts)
    if ft == "viewer":
        return _frame_viewer(pts, reference)
    if ft != "relative":
        return {"error": f"frame_type {frame_type!r} not understood; use "
                         "relative|viewer|auto", "pts": []}
    if side is None or str(side).strip() == "":
        return {"error": "frame_type='relative' needs a side word "
                         "(right|left|top|bottom)", "pts": []}
    if reference is None:
        return {"error": "frame_type='relative' needs a reference point "
                         "([x, y], or a landmark fid resolved to one)", "pts": []}
    return _frame_relative(pts, reference, side)


def _frame_auto(pts) -> dict:
    """The pool's OWN frame: +right = the measured row axis
    (:func:`_row_axis_direction` — NN axial statistics scored against PCA by
    partition tightness), sign chosen so +right points east-ward (the
    deterministic default view, DISCLOSED in the frame dict; a cue that
    declares a view uses frame_type='relative' instead). Origin = pool
    centroid; ``op_x`` = +right, ``op_y`` = +up = ccw(right)."""
    items = list(pts or [])
    xys = [_xy(it) for it in items]
    if not xys:
        return {"error": "frame needs a non-empty pts list", "pts": []}
    cx = sum(p[0] for p in xys) / len(xys)
    cy = sum(p[1] for p in xys) / len(xys)
    if len(xys) == 1:
        u, conf, src = (1.0, 0.0), 0.0, "single-point"
    else:
        u, conf, src = _row_axis_direction(xys)
    # Deterministic default handedness: +right = the axis direction nearer
    # east ((1,0) in world xy); exact tie -> +y. Stated, never silent.
    if u[0] < 0 or (u[0] == 0 and u[1] < 0):
        u = (-u[0], -u[1])
    right_unit = u
    up_unit = (-u[1], u[0])
    out = _project_frame_rows(items, xys, (cx, cy), right_unit, up_unit)
    return {
        "frame": {
            "right_unit": [round(right_unit[0], 3), round(right_unit[1], 3)],
            "up_unit": [round(up_unit[0], 3), round(up_unit[1], 3)],
            "origin": [round(cx, 2), round(cy, 2)],
            "axis_source": f"auto ({src}, conf {conf:.2f})",
            "side": "east-default (+right = measured pool axis, east-ward)",
            "frame_type": "auto",
        },
        "pts": out,
    }


def _frame_viewer(pts, reference) -> dict:
    """The VIEWER frame: ``reference = (from_xy, toward)`` — stand at ``from``,
    look at ``toward``; +vy (up) is the LOOK direction, +vx (right) is the
    viewer's right hand (cw of forward in the aerial view). ``toward`` may be
    a point OR a compass word ('north'|'south'|'east'|'west' — the absolute
    frame; ``from`` then only sets the stated vantage). Every geometry is a
    MASK CENTROID — class plays no part; a car, a building, an [x,y] literal
    fill either slot identically. Origin = the pool centroid (coords stay
    spread-centered for reading; the ray fixes ORIENTATION only, disclosed).

    "left of X looking at Y from Z" = frame{from:Z, toward:Y}, then vx_ of
    the candidate vs vx_ of X (refs project into the same frame).
    "between A and B" = frame{from:A, toward:B}: |vx_| small AND vy_ within
    the stated span. "north of X" = frame{toward:'north'} + the X ref row.
    """
    items = list(pts or [])
    xys = [_xy(it) for it in items]
    if not xys:
        return {"error": "frame needs a non-empty pts list", "pts": []}
    try:
        from_xy, toward = reference
    except (TypeError, ValueError):
        return {"error": "viewer frame needs (from_xy, toward)", "pts": []}
    span = None
    if isinstance(toward, str):
        word = toward.strip().lower()
        if word not in COMPASS_UNIT:
            return {"error": f"viewer 'toward' word {toward!r} not understood; "
                             f"use one of {'|'.join(COMPASS_UNIT)} or a point",
                    "pts": []}
        up_unit = COMPASS_UNIT[word]
        src = f"compass ({word})"
        # no meaningful vantage: coords stay pool-centered for reading spread
        cx = sum(p[0] for p in xys) / len(xys)
        cy = sum(p[1] for p in xys) / len(xys)
        origin_src = "pool centroid"
    else:
        fx, fy = (float(from_xy[0]), float(from_xy[1]))
        tx, ty = (float(toward[0]), float(toward[1]))
        span = math.hypot(tx - fx, ty - fy)
        if span < 1e-6:
            return {"error": "viewer 'from' and 'toward' coincide -- no look "
                             "direction", "pts": []}
        up_unit = ((tx - fx) / span, (ty - fy) / span)
        src = "look ray (from->toward)"
        # origin = the VIEWER: vy_ = meters ahead along the look, vx_ =
        # lateral offset from the ray ("between from and toward" = vy_ in
        # [0, span_m] AND |vx_| small).
        cx, cy = fx, fy
        origin_src = "the viewer (from)"
    # Viewer's RIGHT in the y-up (north-up) aerial view is the CLOCKWISE
    # quarter-turn of forward: looking north (0,1) -> right = east (1,0);
    # looking east (1,0) -> right = south (0,-1).
    right_unit = cw(up_unit)
    out = _project_frame_rows(items, xys, (cx, cy), right_unit, up_unit)
    frame_meta = {
        "right_unit": [round(right_unit[0], 3), round(right_unit[1], 3)],
        "up_unit": [round(up_unit[0], 3), round(up_unit[1], 3)],
        "origin": [round(cx, 2), round(cy, 2)],
        "axis_source": f"viewer {src}; origin = {origin_src}",
        "side": "+vy = the look direction (forward); +vx = the viewer's right",
        "frame_type": "viewer",
    }
    if span is not None:
        frame_meta["span_m"] = round(span, 1)
    return {"frame": frame_meta, "pts": out}


def _project_frame_rows(items, xys, origin, right_unit, up_unit) -> list:
    """Shared row projection for the frame family: op_x/op_y per item with
    fid/class/color + prior sortable evidence carried through (the same
    carry contract as _frame_relative)."""
    cx, cy = origin
    out = []
    for it, xy in zip(items, xys):
        dx, dy = xy[0] - cx, xy[1] - cy
        rec = {"xy": [round(xy[0], 2), round(xy[1], 2)],
               "op_x": round(dx * right_unit[0] + dy * right_unit[1], 2),
               "op_y": round(dx * up_unit[0] + dy * up_unit[1], 2)}
        if isinstance(it, dict):
            if it.get("fid") is not None:
                rec["fid"] = it["fid"]
            if it.get("class"):
                rec["class"] = it["class"]
            if it.get("color"):
                rec["color"] = it["color"]
            if it.get("color_conf") is not None:
                rec["color_conf"] = it["color_conf"]
            for _k in ("dist_m", *(k2 for k2 in it if k2 in COLOR_COLUMNS)):
                if it.get(_k) is not None:
                    rec[_k] = it[_k]
        out.append(rec)
    out.sort(key=lambda r: (-r["op_y"], -r["op_x"]))
    return out


def _frame_relative(pts, reference, side) -> dict:
    """Reorient candidates into the operator's view the instruction NAMES, so the
    relational read becomes a trivial coordinate look-up.

    The model NAMES the orienting reference (a road's centerline, or a landmark's
    point — via ``centerline_of`` / ``xy_of`` / ``anchors``) and the literal
    ``side`` word that fixes the view ('with <X> to the right/left', 'viewed from
    above/below', 'the near/front row', or a compass word). The executor returns
    each candidate in SCREEN coordinates: ``op_x`` = +RIGHT, ``op_y`` = +UP. The
    model then reads the relation straight off — it never rotates anything (the
    measured failure mode): 'bottom / front / near row' = the band at MIN ``op_y``;
    'top / back / far row' = MAX ``op_y``; 'Nth from the right' = the Nth by
    DESCENDING ``op_x`` (from the left = ascending ``op_x``).

    ``reference`` is a single point ``[x, y]`` (a landmark centroid — the view
    orients along the pool->landmark direction; a POLYLINE reference belongs to
    :func:`frame_polyline`, the curvilinear chart). ``pts`` is the candidate
    set (``[x, y]`` or cand dicts; ``fid`` / ``class`` / ``color`` carried
    through). The orientation is fixed once here, in code, in the same pinned
    frame as every other op; 'right/up' are the operator's SCREEN axes for the
    named viewpoint, NOT the map default — so a position read in this frame
    lands on the row and the end the instruction means.

    Returns ``{frame:{right_unit, up_unit, origin, axis_source, side},
    reference_op, pts:[{fid?, xy, op_x, op_y, class?, color?}]}``. Do the row
    pick / ordinal over the returned ``op_x`` / ``op_y`` — do NOT re-rotate.
    """
    items = list(pts or [])
    xys = [_xy(it) for it in items]
    if not xys:
        return {"error": "frame needs a non-empty pts list", "pts": []}
    cx = sum(p[0] for p in xys) / len(xys)
    cy = sum(p[1] for p in xys) / len(xys)
    side_key = _FRAME_SIDE.get(str(side or "").strip().lower())
    if side_key is None:
        return {"error": f"side {side!r} not understood; use right|left|top|bottom "
                         "(aliases: above/below/front/back/near/far)", "pts": []}
    refpt = (float(reference[0]), float(reference[1]))
    _d = (refpt[0] - cx, refpt[1] - cy)
    # A reference AT the pool center carries no direction: fall back to the
    # deterministic east axis (the caller's receipt discloses the weak side).
    ref_dir = _u(_d) if math.hypot(*_d) > 1e-9 else (1.0, 0.0)
    axis_source = "normal"
    # Screen basis in world coordinates (x = east, y = north). In this y-up
    # frame ``_ccw`` / ``_cw`` ARE the visual counter-/clockwise
    # quarter-turns, so the basis below is a PROPER ROTATION of the aerial
    # view — never a reflection. GT-verified against rendered-frame
    # adjudication (the textbook-mirror candidate is REFUTED against
    # GT+instruction), 3-way blind verified. For 'reference to the right'
    # the +right axis points TOWARD the reference and up = ccw(right); for
    # top/bottom the reference fixes the +up axis and right = cw(up).
    # Sign-locked by maps/search/tests/test_frame_aspect.py.
    if side_key in ("right", "left"):
        right_unit = ref_dir if side_key == "right" else (-ref_dir[0], -ref_dir[1])
        up_unit = ccw(right_unit)
    else:  # top / bottom: the reference fixes the up axis
        up_unit = ref_dir if side_key == "top" else (-ref_dir[0], -ref_dir[1])
        right_unit = cw(up_unit)

    def _op(p):
        dx, dy = p[0] - cx, p[1] - cy
        return (round(dx * right_unit[0] + dy * right_unit[1], 2),
                round(dx * up_unit[0] + dy * up_unit[1], 2))

    out = []
    for it, xy in zip(items, xys):
        ox, oy = _op(xy)
        rec = {"xy": [round(xy[0], 2), round(xy[1], 2)], "op_x": ox, "op_y": oy}
        if isinstance(it, dict):
            if it.get("fid") is not None:
                rec["fid"] = it["fid"]
            if it.get("class"):
                rec["class"] = it["class"]
            if it.get("color"):
                rec["color"] = it["color"]
            if it.get("color_conf"):
                rec["color_conf"] = it["color_conf"]
            # L3: carry the sortable evidence a prior op attached (a color
            # column's 0.0 is a valid cross-family score -> is-not-None) so
            # color + distance co-arrive at a downstream score_by over the
            # framed rows.
            # op_x/op_y deliberately NOT carried: frame computes them FRESH for
            # THIS reference/side; a stale pair from a prior frame would clobber
            # the new screen coords AND corrupt the reading-order sort below.
            for _k in ("dist_m", *(k2 for k2 in it if k2 in COLOR_COLUMNS)):
                if it.get(_k) is not None:
                    rec[_k] = it[_k]
        out.append(rec)
    out.sort(key=lambda r: (-r["op_y"], -r["op_x"]))
    rox, roy = _op(refpt)
    return {
        "frame": {
            "right_unit": [round(right_unit[0], 3), round(right_unit[1], 3)],
            "up_unit": [round(up_unit[0], 3), round(up_unit[1], 3)],
            "origin": [round(cx, 2), round(cy, 2)],
            "axis_source": axis_source,
            "side": side_key,
            "frame_type": "relative",
        },
        "reference_op": [rox, roy],
        "pts": out,
    }


def _poly_chart_one(poly, arc_v, pt):
    """Project one point onto a world-meter polyline:
    ``(along_m, off_m, foot_xy)``. ``foot_xy`` is the point on the chart
    ``off`` was signed against — the ONE basis a compass reading of that
    sign may use, and NOT :func:`polyline_nearest_point`'s foot, which
    clamps at the ends where this one extends (measured over 2,229 side
    columns of the current bake: reading the compass off the clamped foot
    scores mean purity 0.700 / 4.7% exact against 0.904 / 38.9% here,
    because a row past a short line's end reads as ahead, not beside).
    ``off`` is SIGNED with + = the RIGHT of
    travel — the same handedness as :func:`_frame_viewer` (right =
    cw(forward) in the y-up world frame), computed here as
    ``cross(tangent, to_point)``: cross > 0 puts the point on the ccw =
    LEFT side, so right-of-travel is the cross <= 0 side.
    UN-CLAMPED at the ends: a point whose projection falls beyond
    an end reports its TRUE arc coordinate on the end segment's infinite
    extension (``along`` < 0 before the start, > the total length past the
    end) and its lateral offset from that extension — never pinned to the
    endpoint, so there is no endpoint artifact to flag."""
    px, py = _xy(pt)
    best = None
    n_edges = len(poly) - 1
    for i in range(n_edges):
        ax, ay = float(poly[i][0]), float(poly[i][1])
        ex, ey = float(poly[i + 1][0]) - ax, float(poly[i + 1][1]) - ay
        seg2 = ex * ex + ey * ey
        if seg2 <= 0:
            continue
        t = ((px - ax) * ex + (py - ay) * ey) / seg2
        tc = max(0.0, min(1.0, t))
        fx, fy = ax + tc * ex, ay + tc * ey
        perp = math.hypot(px - fx, py - fy)  # nearest-segment argmin: clamped
        if best is None or perp < best[0]:
            # the chart coordinate extends past the terminal segments' ends
            te = t if ((i == 0 and t < 0.0)
                       or (i == n_edges - 1 and t > 1.0)) else tc
            gx, gy = ax + te * ex, ay + te * ey
            dx, dy = px - gx, py - gy
            cross = ex * dy - ey * dx  # + = ccw(tangent) side = viewer-LEFT
            off = math.hypot(dx, dy) if cross <= 0 else -math.hypot(dx, dy)
            arc = arc_v[i] + te * math.sqrt(seg2)
            best = (perp, arc, off, [gx, gy])
    if best is None:
        return None
    _perp, arc, off, foot = best
    return round(arc, 2), round(off, 2), foot


def polyline_nearest_point(poly, pt):
    """The nearest point ON the polyline to ``pt`` (exact per-segment
    projection, no chord shortcut). The VANTAGE resolver for a
    line-carrying frame slot: a describer "on Main St" stands at the
    centerline point nearest what they look at, never at the road's area
    centroid (blocks away on a long road). Pool-item positions are NOT
    resolved here — POSITION intent stays the mask's area centroid (the
    MASK-INTENT BASIS below)."""
    px, py = _xy(pt)
    best, best_d2 = None, None
    for i in range(len(poly) - 1):
        ax, ay = float(poly[i][0]), float(poly[i][1])
        bx, by = float(poly[i + 1][0]), float(poly[i + 1][1])
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy
        t = 0.0 if L2 <= 0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
        qx, qy = ax + t * vx, ay + t * vy
        d2 = (px - qx) ** 2 + (py - qy) ** 2
        if best_d2 is None or d2 < best_d2:
            best, best_d2 = [qx, qy], d2
    return best


def orient_polyline_toward(poly, toward):
    """Polyline orientation for a DECLARED travel direction — the chart's
    ``+off = right of travel`` sign follows it. ``toward`` is a point (the
    referent travelled toward: reverse iff end0 sits nearer it than the
    last vertex — travel runs toward the end nearest the referent) or a
    compass word (:data:`COMPASS_UNIT`, world frame: reverse iff the net
    end0->end-1 displacement opposes it). Returns ``(poly, reversed)``;
    exact ties keep the stored orientation."""
    P = [[float(q[0]), float(q[1])] for q in poly]
    if isinstance(toward, str):
        u = COMPASS_UNIT.get(toward.strip().lower())
        if u is None:
            return P, False
        net = (P[-1][0] - P[0][0], P[-1][1] - P[0][1])
        rev = net[0] * u[0] + net[1] * u[1] < 0
    else:
        tx, ty = _xy(toward)
        d0 = math.hypot(P[0][0] - tx, P[0][1] - ty)
        d1 = math.hypot(P[-1][0] - tx, P[-1][1] - ty)
        rev = d0 < d1
    return (P[::-1], True) if rev else (P, False)


def frame_polyline(pts, poly):
    """The CURVILINEAR chart: project the pool ALONG a polyline (a road
    centerline, or an ``[[x, y], ...]`` literal) — arc-length + signed
    lateral offset, exact on bends (the chord appears NOWHERE in position
    math).

    Every input row gains ``along`` (meters along the polyline — UN-CLAMPED:
    a point past an end reads its true arc coordinate, < 0 or > the total
    length) and ``off`` (signed lateral meters, + = right of travel —
    RAW-WORLD handedness, see :func:`_poly_chart_one`). Returns
    ``(rows, receipt)``; a degenerate polyline returns ``([], {"error":
    ...})`` for the caller to raise."""
    items = list(pts or [])
    receipt: dict = {"n": len(items)}
    if not items:
        receipt["error"] = "frame needs a non-empty pts list"
        return [], receipt
    P2 = [[float(q[0]), float(q[1])] for q in poly or []]
    if len(P2) < 2:
        receipt["error"] = "a polyline reference needs >= 2 [x,y] vertices"
        return [], receipt
    for i in range(len(P2) - 1):
        if math.hypot(P2[i + 1][0] - P2[i][0], P2[i + 1][1] - P2[i][1]) < 1e-9:
            receipt["error"] = ("polyline vertices {} and {} coincide -- no "
                                "travel direction there".format(i, i + 1))
            return [], receipt
    arc_v = _arc_lengths(P2)

    rows: list = []
    for it in items:
        xy = _xy(it)
        along, off, _foot = _poly_chart_one(P2, arc_v, xy)
        rec = {"xy": [round(xy[0], 2), round(xy[1], 2)],
               "along": along, "off": off}
        if isinstance(it, dict):
            if it.get("fid") is not None:
                rec["fid"] = it["fid"]
            if it.get("class"):
                rec["class"] = it["class"]
            if it.get("color"):
                rec["color"] = it["color"]
            if it.get("color_conf") is not None:
                rec["color_conf"] = it["color_conf"]
            for _k in ("dist_m", *(k2 for k2 in it if k2 in COLOR_COLUMNS)):
                if it.get(_k) is not None:
                    rec[_k] = it[_k]
        rows.append(rec)
    return rows, receipt


# --- exact-mask geometry (the `compare` op's distance substrate) --------------
# Every surface object IS a labeled mask, and the relation columns are computed
# against the EXACT mask as ingested: a rasterized footprint realises as its
# full multi-loop cell boundary (`cells_boundary_loops` — EVERY fragment and
# hole loop, collinear runs collapsed, which is lossless), a fine
# ``world_contour`` realises as its full polygon, a polyline-only node as its
# polyline, a point-only node as its point. NO simplification, NO decimation,
# NO loop selection anywhere in this path — d = 0.0 INSIDE/ON the footprint
# else exact meters to it (the "in the parking lot of X must read d=0"
# semantics the compare op is built on). numpy-vectorised; deterministic.


def point_in_polygon(pt, poly) -> bool:
    """Even-odd ray cast: True iff ``pt`` lies inside the CLOSED polygon ``poly``
    (a list of [x, y] vertices; the closing edge last->first is implicit).
    Frame-independent (pure planar geometry). A point exactly on a horizontal
    edge follows the standard half-open rule (deterministic, never crashes)."""
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


# --- MASK-INTENT BASIS (the design contract) ----------------------------------
# Every surface object IS a labeled mask (uniform substrate — detected
# candidates carry the same exact mask realisation the prebake does). Each
# primitive consumes the mask FUNCTIONAL that matches its INTENT — the
# mask-vs-centroid choice is per-primitive semantics, never a blanket rule:
#
#   GAP / CONTAINMENT intent (d_<ref>, and its brg_gap_<ref> min-gap-vector
#       bearing) -> exact-mask-to-exact-mask min distance, 0.0 on containment/
#       contact/crossing (where brg_gap_ is honestly absent — no local
#       direction exists). A car "at the lot" is 0.0 the moment its footprint
#       touches; a building candidate's gap to a road is its wall's gap, not
#       its roof-centroid's.
#   POSITION intent (dc_/brg_/along_/off_/vx_/vy_/frame ordinals/groups)
#       -> the mask's TRUE AREA CENTROID (its position summary;
#       :func:`loops_area_centroid`). Boundary-nearest here would DESTROY the
#       signal: parked-car ordinal spacing is ~5m center-to-center but ~0.5m
#       edge-to-edge (measured), and a bearing needs one point per object.
#   SHAPE intent (footprint_area_m2 / aspect_ratio)
#       -> mask_shape_features (min-area-rect over cells).
#
# An object with no realised mask stays a POINT-mask (its centroid is its
# whole footprint) — every functional degrades to the point consistently.
# The ONLY variation across nodes is genuine data-kind availability
# (mask-cells vs contour vs polyline-only vs point-only), each kind used
# EXACTLY — never a fidelity downgrade, never a shape/linearity test.

# Pairwise-kernel memory bound: vectorised (points x segments) blocks are
# chunked to this many pairs. Batching only — every pair is still computed
# exactly; this is a memory ceiling, not an approximation.
_PAIR_CHUNK = 4_000_000


def loops_area_centroid(loops):
    """TRUE area centroid ``[x, y]`` of a mask realised as boundary loops —
    the signed shoelace centroid summed across ALL loops, so hole loops
    (which the directed cell trace winds opposite to outer loops) subtract
    exactly. A single loop is orientation-independent (the sign cancels).
    ONLY closed chains contribute. The shoelace wraps with ``% n``, which on
    a ring re-walks the duplicated closing vertex for a zero term but on an
    OPEN way manufactures a phantom closing edge: its implicit area clears
    the 1e-12 guard by a hair and ``sx / (3*sa)`` explodes. Measured the day
    ways gained exact geometry: 883 of 5,257 calcity anchor rows landed a
    centroid more than 20 m off their own vertices, 46 outside the tile, the
    worst 23,161 m away -- and ``center distance to <ref>`` / ``bearing from
    <ref>`` are served columns, so the model read it. A way encloses nothing,
    so it has no area centroid; with no closed chain this falls back to the
    vertex mean, which is a position summary a way does have."""
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
    to outer loops and subtract exactly). 0.0 for an absent / degenerate
    realisation (a point or polyline encloses nothing) -- enforced by SKIPPING
    open chains rather than trusting the shoelace to cancel, which it does
    not: the ``% n`` wrap closes a way implicitly, and a LineString waterway
    reported 4,983,953 m2 of enclosed area. 514 open-chain rows carried a
    non-zero one, weighting a union ref's centroid at
    ``geom_program._union_center``."""
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

    The one rule that tells a ring from a way, read off the geometry rather
    than declared by whoever happens to be holding it. Every producer emits
    the closing vertex — the bake
    (``benchmarks.citynav.bake.city._world_contour_from_geometry``),
    :func:`cells_boundary_loops`, and the two perception contour paths — so a
    node, a way and a ring are ONE representation and nothing has to ask
    which kind of feature it is looking at."""
    return (len(pts) >= 4
            and float(pts[0][0]) == float(pts[-1][0])
            and float(pts[0][1]) == float(pts[-1][1]))


def clip_chains_to_box(chains, box) -> list:
    """``chains`` cut to ``box`` = ``(x0, y0, x1, y1)``, one chain per run
    that stays inside.

    THE MAP IS THE WORLD. A bake rasterises a feature's full OSM geometry and
    then clips the CELLS to its grid, so ``cells`` stop at the tile edge while
    ``world_contour`` runs on: 72 of 353 named calcity features carry vertices
    outside the tile, "Neuralia Road" reaching y 23,745 on a tile 6,297 m
    tall. Every reader of the chains then measures against ground the drone
    cannot reach — and :func:`representative_point` resolves several chains
    "on the longest", so that road's served position was the arc midpoint of
    a 19 km way, at y 14,198 off the north edge of the map.

    A segment crossing the boundary is CUT at it, never dropped whole:
    dropping loses the in-tile half of every arterial that leaves. A chain
    that exits and re-enters comes back as two.

    Closure is not preserved and must not be faked — a ring the boundary cuts
    is no longer a ring, and stitching it shut along the tile edge would claim
    a wall the survey never recorded.
    """
    x0, y0, x1, y1 = (float(box[0]), float(box[1]),
                      float(box[2]), float(box[3]))

    def inside(p):
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    def cut(a, b):
        """Liang-Barsky: the ``(t_enter, t_exit)`` of a->b inside the box."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]),
                     (-dy, a[1] - y0), (dy, y1 - a[1])):
            if p == 0:
                if q < 0:
                    return None
                continue
            t = q / p
            if p < 0:
                if t > t1:
                    return None
                t0 = max(t0, t)
            else:
                if t < t0:
                    return None
                t1 = min(t1, t)
        return (t0, t1) if t0 <= t1 else None

    def at(a, b, t):
        # THE ENDPOINTS COME BACK VERBATIM. ``a[0] + (b[0]-a[0]) * 1.0`` is
        # not bit-identical to ``b[0]``, and the whole partition rule one
        # module over is exact vertex identity with no tolerance
        # (:func:`chain_component_labels`) — so interpolating an untouched
        # segment's own endpoints silently cut every named feature into
        # "separate places" that touch at 0.0 m. Measured before this
        # guard: 51 rows lost their centre, 37 of them with their pieces
        # within 10 m and a MEDIAN gap of 0.0 m.
        if t <= 0.0:
            return [a[0], a[1]]
        if t >= 1.0:
            return [b[0], b[1]]
        # CLAMPED, because a cut is ON the boundary by construction and the
        # interpolation lands a few ULPs past it -- ``-2.0000000000000004``
        # for a box edge at -2.0. A vertex outside the box the clip just
        # promised to enforce is the kind of thing every later reader has to
        # carry a tolerance for.
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
            seg = cut(a, b)
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

    The partition :func:`chain_components` counts. ONE implementation, so a
    caller that needs the pieces — a name resolving to its components — and a
    caller that needs only how many — a centre is defined over one place and
    over no more — can never disagree about where a boundary falls. A label
    is the index of the lowest-numbered chain in its component, so the
    labelling depends on nothing but the input order.

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
    """How many separate PLACES a mask realisation is.

    Two chains are the same place when they SHARE A VERTEX, or when one
    ENCLOSES the other, which makes a hole loop part of the ring around it
    rather than a place of its own. Whatever is left over is genuinely
    disjoint: two fragments of a speckled mask, a way the tile boundary cut
    into two in-tile pieces.

    A CENTRE is defined over one of these and over no more. The crossings op
    states the same law about the places two roads meet — "a single centre
    over three crossings lands off all three and reads as a confident answer"
    — and a merged feature is the same shape of lie: on the calcity bake 117
    of 5,257 anchor rows are several places under one name, and 64 of the 135
    rows whose centre sits off their own geometry are among them ("98th
    Street", two ways 869 m apart, centred on neither).

    Exact vertex identity is the whole rule: no tolerance, no join radius.
    Endpoint-only sharing gives the same count on all but 2 of those 5,257
    rows, so nothing rides on interior-vertex touches either."""
    return len(set(chain_component_labels(chains)))


def mask_geom(chains):
    """Precompute the vectorised arrays of one EXACT mask realisation:
    ``{"V": (N,2) vertices, "A"/"B": (S,2) segment endpoints, "CA"/"CB": the
    subset of those lying on CLOSED chains, "closed"}``.

    ``chains`` is a list of vertex chains ``[[x, y], ...]`` — the multi-loop
    exact cell boundary (:func:`cells_boundary_loops`), a feature's exact
    ``world_contour`` (one chain per part it was given), or a single point as
    ``[[x, y]]``. Each chain states its OWN closure
    (:func:`is_closed_chain`), so a side may mix a ring and a way honestly,
    and the closing edge is IN the chain — segments are plain consecutive
    pairs with no wrap-around index.

    ``CA``/``CB`` carry the closed chains' edges alone, because even-odd
    containment is defined over those only; ``A``/``B`` carry every edge,
    because distance is defined over all of them. Build ONCE per object per
    op; :func:`gap_witness` consumes two of these."""
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


#: Slack on a bound comparison, in metres. A bounding-box bound is
#: mathematically never greater than the gap it bounds, but the two are reached
#: by different routes -- hypot over the box corners against the einsum in
#: :func:`_min_pt_to_segs` -- and they can disagree by an ulp: measured at
#: 39.95481255057674 against 39.95481255057673, which was enough to skip the
#: member holding the true minimum and return no gap at all. A micron sits far
#: below the millimetre this module reports to (``_FLOOR_EPS_M``) and
#: enormously above double rounding, and erring by this much can only ever
#: evaluate MORE members, never fewer.
BOUND_SLACK_M = 1e-6


def stack_bboxes(geoms) -> np.ndarray:
    """``(M, 4)`` of every member's bounding box, ``nan`` where one has none.

    Built ONCE per reference column and then compared against every item, which
    is the whole point: the per-member Python call this replaces was 5.8 M
    calls and 13% of the wall on a class-sized reference.
    """
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

    ONLY THE EDGES THAT CAN CROSS. ``cond`` is true exactly when one endpoint
    is strictly above the ray and the other is not, i.e. when
    ``ylo <= py < yhi``; every other edge contributes a hard zero to the
    parity. Testing points in y-order therefore lets a chunk spanning
    ``[lo, hi]`` drop every edge with ``ylo > hi`` or ``yhi <= lo`` -- those
    could not have crossed any point in the chunk, so the parity is the same
    one, not an approximation of it.

    Without that drop the cost is ``len(P) x len(A)`` whatever the geometry,
    and a reference standing for a whole class ("in building" resolves every
    building on the map) turns one sentence into hundreds of millions of
    point-edge tests: `find building in building.` ran 885 s and had not
    returned, which is why depth-2 gt_trace never finished.
    ``test_inside_mask_prune_is_exact`` pins the identity against the
    unpruned parity on geometry built to straddle the chunk edges.
    """
    out = np.zeros(len(P), dtype=bool)
    if len(P) == 0 or len(A) == 0:
        return out
    # ONLY THE POINTS INSIDE THE LOOPS' BOX. A point outside it crosses an
    # even number of edges -- none to its right, or every loop whole -- so
    # its parity is zero without the count. The 30k vertices of a union
    # tested against one car keep the handful that lie in the car's box.
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


# Past this many (point, segment) pairs the argmin is looked for on the
# pairs that can hold it (see ``_min_pt_to_segs_pruned``); under it the
# bound arithmetic costs more than the pairs it would drop.
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
    order, so the first of them -- the scan's witness -- is the same pair,
    computed by the same arithmetic. ``BOUND_SLACK_M`` errs toward keeping.
    A car measured against a 3,691-building union as one place scans a few
    hundred of its 3.6 M pairs.
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
    # ONLY THE SEGMENTS WHOSE BOXES MEET. A segment lying wholly outside the
    # other set's bounding box crosses nothing in it, so each side keeps the
    # segments that overlap the other side's box, in their own order. The
    # first hit in row-major order is the same pair -- the pairs dropped were
    # never hits -- and the point computed from it the same point. A car
    # against a 30k-segment union keeps a few dozen segments a side.
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
    # Containment and crossing both need the two boxes to overlap: a vertex
    # of one side inside the other lies in both boxes, and two segments can
    # only cross at a point that lies in both. ``bbox_gap`` is 0.0 exactly
    # on overlap (touching included), so the three tests run when they could
    # return anything and are skipped when they could not -- which for a
    # car against 3,692 buildings is nearly every pair, and three of the
    # five array blocks each pair paid for.
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
        # into a warning storm and report the nearest approach as "inf m"
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
    routinely several ways (179 of 457 named roads in the bake carry more
    than one row), so each side is a bundle, never a line. They MEET where
    they TOUCH: distance exactly zero — a shared vertex, a vertex on the
    other's segment, or a proper crossing of two segment interiors (which
    counts at 0 whatever the vertex spacing, and is what catches an X
    between two long segments that no endpoint is near).

    Nothing is dilated and nothing is clustered. Two scales used to live
    here, 2.0 m of touch and 15.0 m of apartness, and both existed to
    recover contact from a 2 m raster reconstruction of the geometry: the
    tile persisted no vertices for a road, so its centerline was PCA'd out
    of buffered cells and two roads that met never quite touched. With the
    vertices the map actually gives, roads that meet SHARE AN OSM NODE, so
    their contact is 0.0 rather than nearly so — measured over the 305
    named line features of the calcity bundle: exact-zero recalls all 1,124
    touching pairs with zero false contacts, 995 of them at a single point,
    and not one pair shares a corridor for the apartness scale to re-merge.

    ``points`` holds one entry PER CROSSING, deduplicated at the 2 dp the
    geometry is stored to — which is also the resolution of the ``[x, y]``
    fid each crossing row carries, so two rows can never share a fid. It is
    never collapsed to a centroid: a single centre over three crossings
    lands off all three and reads as a confident answer. Empty ``points``
    with a finite ``nearest_m`` is the caller's refusal material.
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
        # contact, whatever the vertex spacing. Without it two long segments
        # meeting mid-span read as their endpoint separation and vanish.
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
    """Trace the EXACT boundary of a rasterized footprint: EVERY closed loop —
    the outer boundary of every fragment AND every hole loop — with collinear
    runs collapsed (lossless). NO decimation, NO loop selection: the loops ARE
    the ingested mask.

    ``cells`` is an iterable of (row, col) tile-grid cells (numpy ``(N,2)`` or
    a list of pairs). Returns a list of CLOSED loops, each a list of [x, y]
    world meters ending on the vertex it started from — the closure is stated
    by the chain, never assumed by a reader (:func:`is_closed_chain`) — or
    None when no cells. The
    directed trace winds hole loops opposite to outer loops, so even-odd
    membership over ALL loops together (:func:`mask_contains` /
    :func:`gap_witness`) reproduces exact cell-union membership, and
    :func:`loops_area_centroid`'s signed sum subtracts holes exactly.
    Deterministic."""
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


# --- mask-derived shape descriptors (area / aspect / orientation) -------------
# Computed UNIFORMLY from a node's rasterized `cells`, for BOTH a detected car
# and a prebake building, so the surface exposes the same shape columns on every
# node type. `mask_shape_features` returns
# three opt-in discriminators the model may compare/score_by/filter on:
# footprint area, min-area-rectangle aspect ratio, and long-axis orientation.
# Pure stdlib, deterministic (no RNG); frame math matches
# maps.sws.spatial.compute_orientation and the (east,north) cardinal frame.


def _convex_hull(points):
    """Monotone-chain convex hull of 2D points ``(a, b)``. Returns the hull
    vertices in CCW order with NO repeated endpoint; collinear interior points
    are dropped. Deterministic, O(n log n). Fewer than 3 unique points return
    the sorted unique points unchanged."""
    pts = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(pts) <= 2:
        return pts

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _min_area_rect(hull):
    """Rotating-calipers minimum-area bounding rectangle of a CONVEX hull (list
    of ``(a, b)`` vertices in CCW order). Returns ``(long_side, short_side)``.
    The min-area rectangle has one side collinear with a hull edge (classic
    theorem), so trying every edge is exact. Degenerate hulls (0-1 points)
    return a zero-extent rect.

    The long side's DIRECTION is deliberately not returned. It was, and it
    became ``orientation_deg``: an undirected [0, 180) heading that reads the
    same for a north-facing and a south-facing house on one street. Which way
    a thing faces is a relation to the feature it faces."""
    n = len(hull)
    if n < 2:
        return 0.0, 0.0
    best = None  # (area, long_side, short_side, long_dir)
    for i in range(n):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        elen = math.hypot(ex, ey)
        if elen < 1e-9:
            continue
        ux, uy = ex / elen, ey / elen      # edge direction (unit)
        vx, vy = -uy, ux                    # edge normal (unit)
        min_u = min_v = math.inf
        max_u = max_v = -math.inf
        for (px, py) in hull:
            du = px * ux + py * uy
            dv = px * vx + py * vy
            if du < min_u:
                min_u = du
            if du > max_u:
                max_u = du
            if dv < min_v:
                min_v = dv
            if dv > max_v:
                max_v = dv
        wu, wv = max_u - min_u, max_v - min_v
        area = wu * wv
        if best is None or area < best[0]:
            best = ((area, wu, wv, (ux, uy)) if wu >= wv
                    else (area, wv, wu, (vx, vy)))
    if best is None:
        return 0.0, 0.0
    _area, long_s, short_s, _u = best
    return long_s, short_s


def mask_shape_features(cells, cell_m):
    """Mask-derived shape descriptors from a rasterized footprint, UNIFORM
    across every node type.

    ``cells`` is an iterable of ``(row, col)`` tile-grid cells (numpy ``(N, 2)``
    or a list of pairs) at ``cell_m`` meters per cell. Returns a dict of two
    opt-in row columns, or ``None`` when the footprint is empty:

      * ``footprint_area_m2`` — occupied area = (#unique cells) x cell_m**2 (the
        SAME formula the prebake builder stamps into ``area_m2``).
      * ``aspect_ratio`` — minimum-area rectangle LONG / SHORT side, ``>= 1.0``
        (``~1`` for a square/round blob, large for an elongated van / truck /
        road). A single-cell footprint is 1x1 -> ``1.0``.

    ORIENTATION IS A RELATION, so no axis heading rides here. Which way a
    thing faces is measured against the feature it faces — the compare's
    ``facing <ref>`` — and a min-area rectangle cannot answer it:
    its heading folds to ``[0, 180)``, so a north-facing and a south-facing
    house on one street read the SAME number.

    The rectangle is fit over cell CORNERS (each cell is a unit square) so a
    1-cell or 1-wide footprint has a well-defined extent instead of a
    zero-width centroid. Pure stdlib, deterministic."""
    cs = {(int(rc[0]), int(rc[1])) for rc in (cells if cells is not None else [])}
    if not cs:
        return None
    m = float(cell_m)
    area = round(len(cs) * m * m, 1)
    corners: set = set()
    for (r, c) in cs:
        corners.add((r, c))
        corners.add((r, c + 1))
        corners.add((r + 1, c))
        corners.add((r + 1, c + 1))
    long_s, short_s = _min_area_rect(_convex_hull(corners))
    aspect = round(long_s / short_s, 2) if short_s > 1e-9 else 1.0
    return {"footprint_area_m2": area, "aspect_ratio": aspect}


def contour_shape_features(chains):
    """Shape descriptors from a feature's exact world-meter chains
    (``[[[x_east, y_north], ...], ...]`` — one per part it was given, the same
    shape :func:`mask_geom` takes) — the native-precision counterpart of
    :func:`mask_shape_features`, for nodes whose 2 m cell raster cannot
    express shape (the S2 gate: below ~6 cells orientation quantizes to
    {0, 90} and aspect to {1, 2}).

    Only the CLOSED chains describe a footprint, because only they enclose:
    area sums across them (a hole loop winds opposite and subtracts exactly,
    as in :func:`loops_area`) and the rectangle fits their vertices. A feature
    with no closed chain gets ``None`` — a way's shoelace "area" is the
    accident of closing it, and the cell fallback's answer is the accident of
    the line buffer that rasterized it (Wisteria Street's 368.0 m2 is 92 cells
    x 4 m2 of a 1.0 m buffer). Withholding the column is honest; both
    alternatives are a fabricated number on a column the model scores and
    filters on.

    Returns the SAME two opt-in columns, or ``None`` when degenerate
    (no closed chain, or zero area / zero-width min-area rectangle):

      * ``footprint_area_m2`` — shoelace polygon area (2 dp: the fine source
        is sub-cell accurate, unlike the 0.1-m2-rounded cell count).
      * ``aspect_ratio`` — minimum-area rectangle LONG / SHORT side, >= 1.0
        (convex hull + edge-direction sweep via :func:`_min_area_rect` —
        exact, O(n^2) over <= 16 hull vertices).

    Pure stdlib, deterministic."""
    rings = [[(float(v[0]), float(v[1])) for v in ch] for ch in (chains or [])]
    rings = [ch for ch in rings if is_closed_chain(ch)]
    if not rings:
        return None
    s = 0.0
    for pts in rings:
        for i in range(len(pts) - 1):
            (x1, y1), (x2, y2) = pts[i], pts[i + 1]
            s += x1 * y2 - x2 * y1
    area = abs(s) / 2.0
    long_s, short_s = _min_area_rect(
        _convex_hull([p for ch in rings for p in ch]))
    if area <= 1e-9 or short_s <= 1e-9:
        return None
    return {"footprint_area_m2": round(area, 2),
            "aspect_ratio": round(long_s / short_s, 2)}
