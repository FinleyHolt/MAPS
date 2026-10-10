"""Pure text-matching kernel for the ``find`` op: search by name.

Fuzzy free-text matching (token coverage, plural and abbreviation tolerant)
over each row's ``search_text``: a prior's name + raw OSM tag keys and
values, a detection's detector class. Matching is strictly LEXICAL; there
is no semantic English-noun -> OSM-value mapping. The one CORPUS-level
widening is :func:`tag_family_keys`: a query that is an exact tag VALUE
somewhere reaches that tag's KEY family, below every exact tier. Where a
query's vocabulary diverges from the tags in any other way, the remedy is
the tag census served on misses, not a lookup table.

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
                 ``_TOKEN_BAG_CEIL``), including query tokens merely
                 mentioned INSIDE a longer proper name. Token COVERAGE is the
                 whole band, IDF-WEIGHTED over the words some row could
                 cover, so the corpus's commonest word cannot carry a match
                 alone.

Query grammar (ONE canonical form each):
  * ``key=value`` — '=' is a token separator, so the pair also token-matches
    tag keys AND values; when the row carries exactly that tag the k=v tier
    applies. The value starts after the '=' and the bare (non-``=``) tokens
    that follow the pair extend it, any extension matching exactly — so a
    MULTI-WORD tag value matches whole ('addr_street=Castle Street' matches
    the row tag addr_street=Castle Street, not just 'castle').
  * the literal standalone token ``OR`` (uppercase) separates alternatives;
    the score is the MAX over the OR-clauses.

It is the stdlib-only counterpart to the geometry kernel
(:mod:`maps.search.analyze_scene.geom_compute`), imported by the executor
(:mod:`maps.search.analyze_scene.geom_program`).

- **Pure + data-in-args.** No graph, no I/O; ``re`` and ``math`` only. Every
  matcher takes plain strings / prepared query tuples (and, for the fuzz
  band, the corpus token weights :func:`token_idf` mints from the searched
  rows) and returns scores.
- **The description WALL lives upstream.** This module scores whatever
  ``search_text`` the surface builder put on a row; the builder keeps the
  CityRefer ``description`` tags (the catalog the instruction was sampled
  from) out of it (``geom_program._SEARCH_TAG_DENYLIST``).
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
# matched as a NAME; below it, as a tag/class/bag hit.
NAME_TIER_FLOOR = _NAME_PHRASE_TIER_BASE

# What reaching a query's own tag FAMILY is worth (:func:`tag_family_keys`):
# above FUZZY_THRESHOLD so the family is retrieved, below every exact tier so
# the exact hits keep the head of the find order.
TAG_FAMILY_SCORE = 0.5


def tag_family_keys(q_data: tuple, rows) -> frozenset:
    """The tag KEYS a query is an exact VALUE of, anywhere on ``rows``.

    A CORPUS fact, so it does not live in :func:`find_score`, which sees one
    row at a time. Reaching the KEY family at :data:`TAG_FAMILY_SCORE` puts
    the family's other rows (``building=yes`` for a ``house`` query) in the
    match set BELOW the exact hits: a widening, never a drop.
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

    A CORPUS fact, like :func:`tag_family_keys`. ``df`` counts ROWS, over
    :func:`_fuzzy_expand_tokens` of the row's ``search_text`` and ``name``,
    the two texts the fuzz band scores against.

    A token absent from the returned map has ``df == 0``: UNSATISFIABLE, no
    row can cover it, which is what the fuzz band's denominator turns on.
    Every present token has a strictly positive weight (idf >= ln 2), so a
    zero lookup means absent and nothing else.
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
    """Lowercase, strip punctuation, expand abbreviations. ``=`` is a token
    separator, so a ``key=value`` query tokenizes to [key, value] and matches
    tag keys AND values."""
    text = text.lower()
    text = re.sub(r"[_\-/=]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(_FUZZY_ABBREVS.get(t, t) for t in text.split())


def _singularize(word: str) -> str:
    """Simple English plural suffix stripping: the one best guess."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "ses", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        return word[:-1]
    return word


def _singular_forms(word: str) -> set[str]:
    """EVERY plausible singular of a word, because one guess is not enough.

    ``-ses`` is ambiguous ("buses" is "bus"+es, "houses" is "house"+s), so
    both forms are emitted; a spurious form matches nothing.
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
    forms of one word stay grouped and a plural query is exactly as
    satisfiable as its singular.
    """
    return tuple(frozenset(_singular_forms(t))
                 for t in dict.fromkeys(_fuzzy_normalize(text).split()))


# The perceived-colour palette: ``geom_compute.COLOR_ENUM``, which this
# stdlib-only module cannot import (geom_compute pulls numpy).
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
    where a query named a word.
    """
    return COLORS & set(re.findall(r"[a-z]+", " ".join(
        r["color"].lower() for r in rows
        if isinstance(r, dict) and isinstance(r.get("color"), str))))


def unmatched_query_tokens(query: str, rows, min_len: int = 3):
    """Query tokens carried by NO row in the match set, longest first.

    A junk token DILUTES ``find_score``'s token-bag ratio and never
    disqualifies, so "car withcolor" still binds every car row at 1/2.
    Naming the token nothing carries is the loud signal; the match itself is
    untouched, because a real query often names an attribute no row spells.

    ``rows`` are the matched row dicts, not their texts, because a palette
    word the rows were PERCEIVED as (their ``color``) is not dead vocabulary.
    Suppression is per exact WORD: a palette word no matched row reads
    ("purple" over red rows) stays loud, and "greenhouse" is still glue.
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
        # ("houses" -> "hous"), and glue ("oncastle") shares no prefix with
        # the word it hides ("castle").
        return any(tok.startswith(c) or c.startswith(tok) for c in carried)

    # One word per prefix family: the expansion yields both "lots" and
    # "lot", and naming the same word twice reads as two problems.
    kept: list = []
    for tok in sorted((t for t in qt if not is_carried(t)), key=len):
        if not any(tok.startswith(k) or k.startswith(tok) for k in kept):
            kept.append(tok)
    return sorted(kept, key=lambda t: (-len(t), t))


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance, for the miss-time tag census
    (:func:`geom_program._query_matched_census`), which lists near tag
    vocabulary as counts and retrieves nothing.
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
    counts as matched when ANY form equals a row tag, so a MULTI-WORD tag
    value matches whole: 'addr_street=Castle Street building=yes' yields
    the groups (addr street, castle)/(addr street, castle street) and
    (building, yes). The groups feed the exact-tag tier's coverage
    fraction."""
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

    The literal standalone token ``OR`` (uppercase) separates alternatives;
    scores take the MAX over clauses. Returns a tuple of clause tuples
    ``(q_norm, qt, qw, kv_pairs)``:
      - ``q_norm``: normalized clause string
      - ``qt``: expanded token set (with singularised variants), for MATCHING
      - ``qw``: one frozenset of spellings PER WORD, so coverage counts
        WORDS and not spellings
      - ``kv_pairs``: the clause's normalized ``key=value`` pair GROUPS
        (each a tuple of value-extension candidate forms; see
        :func:`_extract_kv_pairs`)

    ``qw`` is separate from ``qt`` because a singular variant is an alternate
    SPELLING, not a second word: "sports field" is two words, not three.
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

    TOKEN COVERAGE IS THE WHOLE FUZZ BAND: it asks how much of the QUERY was
    found, so ``pool`` scores 0.75 against a row reading ``leisure swimming
    pool building yes``.

    COVERAGE IS IDF-WEIGHTED, OVER SATISFIABLE WORDS ONLY. Weighting by
    ``idf`` keeps the corpus's commonest word (``way`` in ``hugo way``) from
    carrying a match on its own. A word NO row can cover is dropped from the
    denominator, or its maximal idf would veto the word that does match
    (``parking lot`` with ``lot`` absent). Dropping it is safe only because
    the surface builder withholds the bake's provenance keys from
    ``search_text`` (``geom_program._SEARCH_PROVENANCE_DROP``), which makes
    ``way`` and ``road`` ordinary words; the two halves work only together.

    A word's weight is the idf of its most-covering spelling (the lowest idf
    among the forms present), the same form the satisfiability test turns
    on.
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
    # How much of the query's IDF MASS was found: a word counts once however
    # many spellings it has, weighted by its rarity, and only if some row
    # could cover it.
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
    rows being scored), required and keyword-only: it is a property of the
    haystack, not of the row.

    Per clause (max over OR-clauses): exact name equality -> 1.0; the full
    MULTI-WORD clause phrase word-bounded inside the name ->
    0.90+0.09*(length ratio) (a single-word clause never enters this tier);
    the clause's ``key=value`` pair groups exactly matching row tags ->
    0.80+0.09*(pairs matched / pairs given); the whole clause exactly equal
    to ONE tag key or ONE tag value (a TAG-FAMILY token hit: 'parking' on an
    amenity=parking row) -> 0.80; else the capped token-bag fuzz over
    search_text and name. A saturated bag hit ranks below any exact tier."""
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
            # a row tag; the tier scores the coverage fraction.
            matched = sum(
                1 for grp in kv if any(p in tag_pairs for p in grp))
            if matched:
                s = max(s, _KV_TIER_BASE + 0.09 * (matched / len(kv)))
        if s < _KV_TIER_BASE and q_norm and tags:
            # TAG-FAMILY token hit: the whole normalized clause equals a
            # whole tag key or value ('parking' inside 'Parking End
            # Cottages' does not qualify).
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
