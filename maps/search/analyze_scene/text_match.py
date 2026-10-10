"""Pure text-matching kernel for the ``analyze_scene`` ``find`` op — search by name.

This is the SEARCH/EXPLORE half of the single ``analyze_scene`` tool: fuzzy
free-text matching (token coverage, plural and abbreviation tolerant) over
each row's ``search_text`` — a prior's name + raw OSM tag keys and values, a
detection's detector class. Matching is strictly LEXICAL (token overlap,
plural stripping, abbreviation expansion); there is
no semantic English-noun -> OSM-value mapping here. The one CORPUS-level
widening lives in :func:`tag_family_keys`, which the ``find`` branch applies
over the searched set: a query that is an exact tag VALUE somewhere reaches
that tag's KEY family, below every exact tier. Where a query's vocabulary
diverges from the tag vocabulary in any other way, the agent-side remedy is
the tag census (served on misses and in the mission packet's tile overview),
not a lookup table.

Scoring is TIERED so lexical EXACTNESS is visible in the ordering (a
bag-of-tokens hit can cover every query token without the query phrase
existing anywhere on the row — 'Portugal Street' covers {portugal, street}
on an addr_street='Portugal Place' row and must not read as a perfect
match):

  1.0            exact name (normalized equality), or the whole normalized
                 query equal to the row's whole search text (a detection
                 class hit);
  [0.90, 0.99]   the full MULTI-WORD query phrase word-bounded inside the
                 row's NAME ('Downing Street' inside 'Little Downing
                 Street'). A single-word query never enters this tier: a
                 generic head noun ('building') word-bounded inside a proper
                 name ('Baker Building') is an English mention, not a
                 semantic hit — it scores names only through the fuzz band
                 below, so its exact tag hit (0.80-0.89) outranks it;
  [0.80, 0.89]   an exact TAG hit — the query's ``key=value`` pairs match
                 the row's tags exactly after normalization, or the whole
                 query equals ONE tag key or ONE tag value (the tag-family
                 token hit: 'parking' on an amenity=parking row);
  (0, 0.75]      token-bag / abbreviation fuzz (capped at
                 ``_TOKEN_BAG_CEIL``) — including any query token(s) merely
                 mentioned INSIDE a longer proper name. Token COVERAGE is the
                 whole band: the substring and edit-distance paths were
                 measured to add zero ground truth over the corpus and were
                 deleted (see :func:`fuzzy_match_score`). Coverage is IDF-
                 WEIGHTED over the words some row could cover, so the
                 corpus's commonest word cannot carry a match alone.

Query grammar (ONE canonical form each):
  * ``key=value`` — '=' is a token separator, so the pair also token-matches
    tag keys AND values; when the row carries exactly that tag the k=v tier
    applies. The value starts after the '=' and the bare (non-``=``) tokens
    that follow the pair extend it, any extension matching exactly — so a
    MULTI-WORD tag value matches whole ('addr_street=Castle Street' matches
    the row tag addr_street=Castle Street, not just 'castle').
  * the literal standalone token ``OR`` (uppercase) separates alternatives;
    the score is the MAX over the OR-clauses.

It is the pure, stdlib-only counterpart to the geometry
kernel in :mod:`maps.search.analyze_scene.geom_compute`, and like it is imported by BOTH
the live executor (:mod:`maps.search.analyze_scene.geom_program`) and the offline
distillation teacher (via ``geom_program``) — so the served search surface
is byte-identical to the trained one.

Design contract (load-bearing):

- **Pure + data-in-args.** No graph, no I/O, no ``ctx``; ``re`` and ``math``
  only. Every matcher takes plain strings / prepared query tuples (and, for the
  fuzz band, the corpus token weights :func:`token_idf` mints from the searched
  rows) and returns scores. This is what lets ``find`` run inside
  ``run_program(cands, anchors, program)`` with no
  signature change and no heavy-dependency import into the teacher.
- **The descriptions WALL lives upstream.** This module only scores whatever
  ``search_text`` the surface builder put on a row. The builder
  (:func:`geom_program.build_surface`) is responsible for NEVER putting the
  CityRefer ``MapObject.descriptions`` (the GT-label catalog) into that text —
  see the wall note there. Matching an operator's instruction against
  the catalog's descriptions would turn search into a GT oracle.

The scoring functions (:func:`fuzzy_prepare_query` / :func:`fuzzy_match_score`)
are byte-identical to the ``search_features`` fuzzy ranker, so the ``find`` op
resolves names identically to the offline/teacher grounding path.
"""

from __future__ import annotations

import math
import re

# --- lexical abbreviation expansion -----------------------------------------

_STREET_ABBREVS: dict[str, str] = {
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "rd": "road",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "cir": "circle",
    "hwy": "highway",
    "pkwy": "parkway",
}

_GENERAL_ABBREVS: dict[str, str] = {
    "hlth": "health",
    "ctr": "center",
    "cntr": "center",
    "cntrl": "central",
    "hosp": "hospital",
    "med": "medical",
    "comm": "community",
    "pharm": "pharmacy",
    "univ": "university",
    "elem": "elementary",
    "lib": "library",
    "govt": "government",
    "dept": "department",
    "stn": "station",
    "sta": "station",
    "apt": "apartment",
    "bldg": "building",
    "pk": "park",
    "mt": "mount",
    "mtn": "mountain",
    "intl": "international",
    "natl": "national",
    "urbn": "urban",
    "gdn": "garden",
    "gdns": "gardens",
    "mem": "memorial",
    "acad": "academy",
    "sch": "school",
}

_FUZZY_ABBREVS: dict[str, str] = {**_GENERAL_ABBREVS, **_STREET_ABBREVS}

FUZZY_THRESHOLD = 0.45

# Tier bands (see the module docstring): the token-bag/typo-fuzz path is
# capped BELOW the exact tiers so a saturated bag hit can never outrank an
# exact name or an exact key=value tag hit.
_TOKEN_BAG_CEIL = 0.75
_KV_TIER_BASE = 0.80
_NAME_PHRASE_TIER_BASE = 0.90
# The boundary the render's head label gates on: at or above it the head
# matched as a NAME; below it, as a tag/class/bag hit. Public because
# "Best:" is an endorsement and only a name-tier head has earned it.
NAME_TIER_FLOOR = _NAME_PHRASE_TIER_BASE

# What reaching a query's own tag FAMILY is worth (see
# :func:`tag_family_keys`). Above FUZZY_THRESHOLD so the family is
# RETRIEVED, below every exact tier so the exact hits keep the head of the
# find order. Anything strictly inside those two bounds gives the same
# retrieved set in the same order.
TAG_FAMILY_SCORE = 0.5


def tag_family_keys(q_data: tuple, rows) -> frozenset:
    """The tag KEYS a query is an exact VALUE of, anywhere on ``rows``.

    A CORPUS fact, which is why it does not live in :func:`find_score`: that
    sees one row at a time, and whether ``house`` is a value of ``building``
    somewhere on the surface is not answerable from a ``building=yes`` row.

    ``house`` is the value of ``building=house`` on 41 of 5,257 calcity rows
    while 3,345 ``building=yes`` rows score 0.0, so the commonest noun a cue
    writes retrieves 1.2% of the buildings and a downstream 24-row budget
    never binds. Reaching the KEY family at :data:`TAG_FAMILY_SCORE` puts
    the other 3,373 in the match set BELOW the exact hits — a widening, and
    never a drop: no row that matched before stops matching.
    """
    want = {cl[0] for cl in q_data if cl[0]}
    if not want:
        return frozenset()
    out: set = set()
    norm: dict = {}
    for row in rows:
        for k, v in ((row.get("tags") if isinstance(row, dict) else None)
                     or {}).items():
            sv = str(v)
            if sv not in norm:
                norm[sv] = _fuzzy_normalize(sv)
            if norm[sv] in want:
                out.add(str(k))
    return frozenset(out)


def token_idf(rows) -> dict[str, float]:
    """``idf(t) = ln(1 + N/(1+df(t)))`` over the searched rows' vocabulary.

    A CORPUS fact, for the same reason :func:`tag_family_keys` is one: whether
    ``way`` is a near-stopword of this bake is not answerable from the one row
    :func:`fuzzy_match_score` sees. Same shape as that function — rows in
    args, pure, stdlib — so it stays callable from the teacher and the
    analysis readers with no dependency.

    ``df`` counts ROWS, over :func:`_fuzzy_expand_tokens` of the row's
    ``search_text`` and its ``name`` — the two texts the fuzz band actually
    scores against, so the weights describe the haystack being searched and
    not some other one.

    A token absent from the returned map has ``df == 0``: it is UNSATISFIABLE,
    no row can ever cover it. That is the distinction the fuzz band's
    denominator turns on, and it is why this returns the map rather than the
    weights alone. Every present token has a strictly positive weight (df <= N
    gives idf >= ln 2), so a zero lookup means absent and nothing else.
    """
    df: dict[str, int] = {}
    n = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        n += 1
        for t in _fuzzy_expand_tokens(
                f"{row.get('search_text') or ''} {row.get('name') or ''}"):
            df[t] = df.get(t, 0) + 1
    return {t: math.log(1 + n / (1 + d)) for t, d in df.items()}


def _fuzzy_normalize(text: str) -> str:
    """Lowercase, strip punctuation, expand abbreviations. ``=`` joins the
    token-separator set so a ``key=value`` query tokenizes to [key, value]
    and matches tag keys AND values (the census's own taught idiom)."""
    text = text.lower()
    text = re.sub(r"[_\-/=]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(_FUZZY_ABBREVS.get(t, t) for t in text.split())


def _singularize(word: str) -> str:
    """Simple English plural suffix stripping — the ONE best guess."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "ses", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        return word[:-1]
    return word


def _singular_forms(word: str) -> set[str]:
    """EVERY plausible singular of a word, because one guess is not enough.

    ``-ses`` is genuinely ambiguous: "buses" is "bus"+es and "houses" is
    "house"+s, and no suffix rule separates them. :func:`_singularize` picks
    the first, so ``houses`` became ``hous`` and never matched a
    ``building=house`` row on the token path at all -- the edit-distance
    fuzz was silently covering for it (``hous``/``house`` is one edit), and
    deleting that path is what exposed it. Emitting BOTH forms costs one set
    entry and a spurious form matches nothing.
    """
    out = {word, _singularize(word)}
    if word.endswith("ses") and len(word) > 3:
        out.add(word[:-1])          # houses -> house, beside buses -> bus
    return out


def _fuzzy_expand_tokens(text: str) -> set[str]:
    """Token set after normalization + singularization."""
    tokens = set(_fuzzy_normalize(text).split())
    return {f for t in tokens for f in _singular_forms(t)}


def _word_forms(text: str) -> tuple:
    """One frozenset of spellings PER WORD of the text.

    The coverage ratio asks how many of the query's WORDS were found, so the
    forms of one word have to stay grouped: flattening them made the
    denominator grow with the number of spellings and a plural query harder
    to satisfy than a singular one.
    """
    return tuple(frozenset(_singular_forms(t))
                 for t in dict.fromkeys(_fuzzy_normalize(text).split()))


# The perceived-colour palette. Mirrors ``geom_compute.COLOR_ENUM``, which
# this STDLIB-PURE module cannot import (geom_compute pulls numpy, and the
# analysis readers re-parse on a bare interpreter).
COLORS = frozenset((
    "white", "gray", "black", "red", "blue", "green", "yellow", "orange",
    "brown", "purple"))


def _row_match_text(row) -> str:
    """Everything a row SPELLS, as one blob: its name, its walled
    search_text, and its tags flattened to ``k v`` pairs. Perceived colour
    is deliberately absent — it is not text (see :func:`_colors_read`).
    """
    return (f"{row.get('name') or ''} {row.get('search_text') or ''} "
            + " ".join(f"{k} {v}"
                       for k, v in (row.get("tags") or {}).items()))


def _colors_read(rows) -> set:
    """The PALETTE words these rows were perceived as, off their ``color``
    field: the raw mixture ("gray 0.55 / black 0.30") or the rendered cell
    ("dk blue/black", "gray?"), read as words so a shade prefix, a second
    component and the uncertainty mark all fall out.

    ``color`` is the signal because the ``color_<word>`` columns exist only
    where a query NAMED a word — the grounding path names none, so a column
    test would see nothing on half the call sites — and because
    ``color_conf`` is omitted on offline tile-rebuild rows that still carry
    a colour.
    """
    return COLORS & set(re.findall(r"[a-z]+", " ".join(
        r["color"].lower() for r in rows
        if isinstance(r, dict) and isinstance(r.get("color"), str))))


def unmatched_query_tokens(query: str, rows, min_len: int = 3):
    """Query tokens carried by NO row in the match set, longest first.

    A junk token DILUTES ``find_score``'s token-bag ratio and never
    disqualifies, so a query the model never meant to write still binds:
    "car withcolor blue" strips ``blue`` into find{color}, leaving
    {car, withcolor}, whose ratio 1/2 = 0.5 clears FUZZY_THRESHOLD 0.45
    on every car row. Pilot-7 measured exactly that shape returning 520
    rows with no error and no note -- and the same shape carries every
    residual glue token the mask still admits ("onBeche", "atalt",
    "aremagenta"). Naming the token nothing carries is the loud signal;
    the match itself is untouched, because a real query often names an
    attribute no row spells.

    ``rows`` are the matched row dicts, not their texts, because a palette
    word the rows were PERCEIVED as is not dead vocabulary. The note reads
    the row's TEXT and a perceived colour is not in it -- it is on the
    ``color_<word>`` columns -- so pilot-9 served "no matched row carries
    'gray'" over a 465-row grid reading gray, lt gray, dk black (10 of 57
    token-claims flagged a palette word, against 0 of 191 on pilot-8, after
    857f5022 stopped stripping the colour out of a head noun phrase). The
    model OBEYS the note 90% of the time, so a false claim teaches it to
    drop a word that was working.

    Suppression is per WORD, not per row set: "no matched row carries
    'purple'" over rows that all read red is true in both channels and is
    the most useful thing the note can say, and a palette word inside a
    proper name that missed ("Green Street", 24 of 1784 tile names carry
    one) has to stay loud or a wrong bind goes silent. It is exact-word,
    never the prefix relation ``is_carried`` uses, so "greenhouse" is still
    glue.
    """
    qt = {t for t in _fuzzy_expand_tokens(query) if len(t) >= min_len}
    if not qt:
        return []
    rows = list(rows)
    read = _colors_read(rows)
    if read:
        # _singularize is the identity on every palette word, so this one
        # test covers the plural spelling too ("grays" -> "gray").
        qt = {t for t in qt if _singularize(t) not in read}
        if not qt:
            return []
    carried: set = set()
    for row in rows:
        text = _row_match_text(row)
        if text:
            carried |= _fuzzy_expand_tokens(text)

    def is_carried(tok: str) -> bool:
        # PREFIX relation, either direction: the singularizer is lossy
        # ("houses" -> "hous"), so a literal miss flagged every plural of
        # a word the rows plainly carry -- pilot-8 measured 14 of 40
        # flags as exactly that (cars, houses, buildings). A shared
        # prefix is the cheap two-way test, and it leaves the glue this
        # exists to catch untouched ("oncastle" shares no prefix with
        # "castle").
        return any(tok.startswith(c) or c.startswith(tok) for c in carried)

    # One word per prefix family: the expansion yields both "lots" and
    # "lot", and naming the same word twice reads as two problems.
    kept: list = []
    for tok in sorted((t for t in qt if not is_carried(t)), key=len):
        if not any(tok.startswith(k) or k.startswith(tok) for k in kept):
            kept.append(tok)
    return sorted(kept, key=lambda t: (-len(t), t))


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance.

    RETRIEVAL no longer uses this. The typo path scored 0 of the 17 corpus
    episodes where fuzz adds the ground truth, while reaching 48
    ``crossing_markings`` rows for ``parking``, ``highway=track`` for
    ``truck`` and ``leisure=pitch`` for ``patch``. What survives is the
    MISS-TIME TAG CENSUS (:func:`geom_program._query_matched_census`), which
    lists near tag vocabulary as COUNTS and retrieves nothing -- the
    agent-side remedy this module's docstring names for a
    divergent-vocabulary query. Telling a model that wrote ``truck`` that
    the bake says ``highway=track`` is the useful half; serving it the track
    rows as matches was the harmful one.
    """
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _extract_kv_pairs(clause: str) -> tuple:
    """The ``key=value`` pair GROUPS of one raw query clause, normalized.

    A pair starts at a whitespace token containing ``=`` with text on both
    sides (``landuse=grass``); its VALUE is that token's tail plus each
    extension by the bare (non-``=``) tokens that follow it, up to the next
    pair. Each group is a tuple of candidate ``(key, value)`` forms and
    counts as matched when ANY form equals a row tag — so a MULTI-WORD tag
    value matches whole: 'addr_street=Castle Street building=yes' yields
    the groups (addr street, castle)/(addr street, castle street) and
    (building, yes). Measured (armC7 gate B4, episode 7872): the
    whitespace-truncated value 'castle' matched no row tag, so the
    discriminating pair contributed nothing and rows carrying BOTH tags
    scored byte-identically to rows carrying only building=yes. Purely
    lexical — the groups feed the exact-tag tier's coverage fraction."""
    toks = clause.split()
    groups = []
    for i, tok in enumerate(toks):
        if "=" not in tok or tok.startswith("=") or tok.endswith("="):
            continue
        k, v = tok.split("=", 1)
        kn, vn = _fuzzy_normalize(k), _fuzzy_normalize(v)
        if not (kn and vn):
            continue
        forms = [(kn, vn)]
        ext = v
        for nxt in toks[i + 1:]:
            if "=" in nxt:
                break
            ext = f"{ext} {nxt}"
            en = _fuzzy_normalize(ext)
            if en and (kn, en) != forms[-1]:
                forms.append((kn, en))
        groups.append(tuple(forms))
    return tuple(groups)


def fuzzy_prepare_query(query: str) -> tuple:
    """Pre-normalize a query once for re-use across many candidate scorings.

    The literal standalone token ``OR`` (uppercase) separates alternatives —
    the ONE alternation form; scores take the MAX over clauses. Returns a
    tuple of clause tuples ``(q_norm, qt, qw, kv_pairs)``:
      - ``q_norm``: normalized clause string
      - ``qt``: expanded token set (with singularised variants), for MATCHING
      - ``qw``: one frozenset of spellings PER WORD, so coverage counts
        WORDS and not spellings
      - ``kv_pairs``: the clause's normalized ``key=value`` pair GROUPS
        (each a tuple of value-extension candidate forms; see
        :func:`_extract_kv_pairs`)

    ``qt_long`` is gone with the edit-distance phase it existed for.

    ``qw`` is separate from ``qt`` deliberately. A singular variant
    is an ALTERNATE SPELLING of a word, not a second word, and dividing by
    the expanded set made a plural query harder to satisfy than a singular
    one: "sports field" expands to {sports, sport, field}, so matching
    "field" scored 1/3 = 0.33 and fell under the 0.45 floor while "pool"
    scored 1/1. The substring path used to mask this by matching the name
    anyway; with it gone the denominator has to be right. Measured: this is
    what keeps ``find("sports field")`` returning its 16 rows.
    """
    clauses = [c for c in re.split(r"\s+OR\s+", query or "") if c.strip()] \
        or [query or ""]
    out = []
    for c in clauses:
        out.append((_fuzzy_normalize(c), _fuzzy_expand_tokens(c),
                    _word_forms(c), _extract_kv_pairs(c)))
    return tuple(out)


def fuzzy_match_score(clause: tuple, candidate: str, idf: dict) -> float:
    """Score a candidate string against ONE prepared clause
    ``(q_norm, qt, kv_pairs)`` of ``fuzzy_prepare_query``, weighting coverage
    by the corpus token weights ``idf`` (:func:`token_idf` over the searched
    rows).

    1.0 = exact (normalized equality); the ONE partial path, token coverage,
    lives at or below ``_TOKEN_BAG_CEIL``, so a fuzz hit is never 'exact'
    and never enters the k=v / name-phrase tier bands ``find_score``
    documents.

    TOKEN COVERAGE IS THE WHOLE FUZZ BAND, and it earns its place: it asks
    how much of the QUERY was found, so ``pool`` scores 0.75 against a row
    reading ``leisure swimming pool building yes``. ``find pool.`` returns
    zero rows without it on a bake holding three ``leisure=swimming_pool``
    ones, and ``pool`` is the third-commonest query real flights write (120
    uses over 2,700 stored missions). Measured over the corpus, this path
    carries ALL 17 of the episodes where fuzz adds the ground truth
    (2.01% [1.18, 3.08], 16W/0L paired, p=3.05e-5).

    COVERAGE IS IDF-WEIGHTED, OVER SATISFIABLE WORDS ONLY. Counting words
    equally let the corpus's commonest word carry a match on its own:
    ``hugo way`` covered one of two words on 4,733 of 5,257 rows and served
    65-98 tied fuzz rows ahead of the one the query named, so 49.0% of
    named-feature query uses put the queried row outside a 24-row crop once
    ``near=<anchor>`` re-sorted by distance and threw the exact tier's
    ordering away. Weighting by ``idf`` prices ``hugo`` far above ``way`` and
    the tie collapses.

    A word NO row can cover is dropped from the denominator, and that rule is
    load-bearing rather than a refinement: plain IDF takes ``parking lot`` to
    ZERO rows, because ``lot`` has zero df, therefore maximum idf, and it
    vetoes the word that does match. Dropping unsatisfiable words fixes that
    -- but ON ITS OWN it re-opens the flood, because any absent rare word
    leaves the common one as the whole denominator (``smith way`` back to
    4,733). It is safe only because the surface builder already withheld the
    bake's provenance keys from ``search_text``
    (``geom_program._SEARCH_PROVENANCE_DROP``), which is what makes ``way``
    and ``road`` ordinary words again. The two halves are ONE change; neither
    works alone, and that was measured as a 2x2.

    A word's weight is the idf of its most-covering spelling (the lowest idf
    among the forms present, i.e. the highest df) -- the same form the
    satisfiability test turns on, so a word cannot be priced by a spelling
    that satisfies nothing.

    TWO PATHS WERE DELETED because they were measured to add nothing:

    * **edit distance / typo** contributed 0 of those 17 and produced the
      junk: ``parking`` reached 48 ``crossing_markings`` rows at 2 edits in
      8 characters, ``truck`` reached ``highway=track``, ``patch`` reached
      ``leisure=pitch``, ``home`` reached ``Hume Avenue``. Re-running every
      ground-truth number with only this path removed was byte-identical to
      shipped.
    * **substring** contributed 0 of those 17. It also scored a short query
      by the HAYSTACK's length -- ``pool`` scored 0.44, under the 0.45
      floor, purely because the row carried a second tag -- which is the
      regression the token bag was added to fix. Nothing now depends on it.

    Measured on the calcity surface (5,257 prior rows, served retrieval),
    as the 2x2 of provenance-drop x weighting. The two halves fix DIFFERENT
    populations, which is why shipping either alone leaves a flood standing:

        query          none     idf    drop    both
        hugo way       4733       1       7       7
        smith way      4733    4733       7       7
        neuralia road   810       1       5       5
        85th street      99       1      99       1
        sports field     17       1      17       1
        road            810     810     270     270
        parking lot      75      75      75      75

    The drop alone cannot touch ``85th street`` or ``sports field``: nothing
    provenance-borne is inflating those, an ordinary word is. The weighting
    alone cannot touch ``smith way`` or ``road``: ``smith`` is absent from
    this bake, so it leaves the denominator and ``way`` -- still on all 5,257
    rows via ``osm_id`` -- becomes the whole of it.

    ``pool`` holds at its 3 ``leisure=swimming_pool`` rows, ``swimming pool``
    at 34 and ``house`` at 3,414: a single satisfiable word is its own whole
    denominator, and the corpus widening is untouched.
    """
    q_norm, qt, qw = clause[0], clause[1], clause[2]
    c = _fuzzy_normalize(candidate)
    if not q_norm or not c:
        return 0.0
    if q_norm == c:
        return 1.0
    if not qt:
        return 0.0
    ct = {f for t in c.split() for f in _singular_forms(t)}
    # How much of the query's IDF MASS was found -- a word counts once however
    # many spellings of it exist (which is what keeps a plural query exactly
    # as satisfiable as its singular), weighted by how rare it is, and only
    # if some row could have covered it at all.
    num = den = 0.0
    for forms in qw:
        w = min((idf[f] for f in forms if f in idf), default=0.0)
        if not w:                       # df == 0 everywhere: unsatisfiable
            continue
        den += w
        if forms & ct:
            num += w
    if not num or not den:
        return 0.0
    return min(num / den, _TOKEN_BAG_CEIL)


def find_score(q_data: tuple, search_text: str, name: str | None = None,
               tags: dict | None = None, *, idf: dict) -> float:
    """TIERED lexical score of a prepared query (``fuzzy_prepare_query``)
    over a row's ``search_text``, its ``name``, and its raw ``tags``.

    ``idf`` is the corpus token weighting (:func:`token_idf` over the SAME
    rows being scored) the fuzz band divides by. It is required and
    keyword-only: it is a property of the haystack, not of the row, and a
    default would silently score one corpus's rows against another's
    weights -- or against none.

    Per clause (max over OR-clauses): exact name equality -> 1.0; the full
    MULTI-WORD clause phrase word-bounded inside the name ->
    0.90+0.09*(length ratio) — a single-word clause never enters the
    name-phrase tier (a generic head noun inside a longer proper name is a
    lexical mention, not a semantic hit; it reaches names only through the
    fuzz band, so its exact tag hit outranks it); the clause's ``key=value``
    pair groups exactly matching row tags (any value-extension form — a
    multi-word tag value matches whole) ->
    0.80+0.09*(pairs matched / pairs given) — the COVERAGE fraction, so a
    row matching both pairs of a two-pair query outranks a row matching
    one; the whole clause exactly equal
    to ONE tag key or ONE tag value (a TAG-FAMILY token hit: 'parking' on an
    amenity=parking row) -> 0.80 — an exact lexical hit on the tag
    vocabulary, tiered WITH the k=v band so it outranks any proper-name
    SUBSTRING / token-bag fuzz (both capped at 0.75); else the capped
    token-bag / typo fuzz over search_text and name. Exactness is thereby
    visible in the ordering — a saturated bag hit ranks below any exact
    tier."""
    name_norm = _fuzzy_normalize(name) if name else ""
    tag_pairs = None  # normalized lazily, once, only when a clause has k=v
    tag_tokens = None  # normalized whole keys/values, lazily, once
    best = 0.0
    for clause in q_data:
        q_norm, kv = clause[0], clause[3]
        s = 0.0
        if name_norm and q_norm:
            if q_norm == name_norm:
                s = 1.0
            elif (len(q_norm.split()) >= 2
                    and re.search(r"\b" + re.escape(q_norm) + r"\b",
                                  name_norm)):
                s = _NAME_PHRASE_TIER_BASE + 0.09 * (len(q_norm)
                                                     / len(name_norm))
        if s < 1.0 and kv and tags:
            if tag_pairs is None:
                tag_pairs = {
                    (_fuzzy_normalize(str(k)), _fuzzy_normalize(str(v)))
                    for k, v in tags.items()}
            # each GROUP counts when ANY of its value-extension forms equals
            # a row tag (multi-word values match whole); the tier scores the
            # COVERAGE fraction — a row matching both pairs of a two-pair
            # query outranks a row matching one.
            matched = sum(
                1 for grp in kv if any(p in tag_pairs for p in grp))
            if matched:
                s = max(s, _KV_TIER_BASE + 0.09 * (matched / len(kv)))
        if s < _KV_TIER_BASE and q_norm and tags:
            # TAG-FAMILY token hit: the whole normalized clause equals a
            # whole tag key or value. Strictly lexical whole-token equality
            # — a token INSIDE a multi-word proper-name value ('parking'
            # inside 'Parking End Cottages') does NOT qualify and stays in
            # the fuzz band below.
            if tag_tokens is None:
                tag_tokens = set()
                for k, v in tags.items():
                    tag_tokens.add(_fuzzy_normalize(str(k)))
                    tag_tokens.add(_fuzzy_normalize(str(v)))
            if q_norm in tag_tokens:
                s = max(s, _KV_TIER_BASE)
        if s < _KV_TIER_BASE:
            f = fuzzy_match_score(clause, search_text or "", idf)
            if name:
                f = max(f, fuzzy_match_score(clause, name, idf))
            s = max(s, f)
        best = max(best, s)
    return best
