"""The lexical ``find`` matcher: tiered scores of a query over each row's
``search_text``, ``name`` and tags.

  1.0           the whole query equals the name, or the whole search text;
  [0.90, 0.99]  a multi-word query phrase word-bounded inside the name;
  [0.80, 0.89]  the query's ``key=value`` pairs match row tags exactly, or the
                whole query equals one tag key or value;
  (0, 0.75]     IDF-weighted coverage of the query's words, capped below
                every exact tier.

A ``key=value`` value extends over the bare words after it, so a multi-word
tag value matches whole; a standalone ``OR`` separates alternative clauses,
scored by their max.
"""

from __future__ import annotations

import functools
import math
import re

from maps.query.spatial_math import COLOR_FAMILY

# The abbreviations query and row words expand to, before matching.
_FUZZY_ABBREVS: dict[str, str] = {
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

FUZZY_THRESHOLD = 0.45

# The tier bands of the module docstring.
TOKEN_BAG_CEIL = 0.75
KV_TIER_BASE = 0.80
# At or above: the row matched as a NAME.
NAME_TIER_FLOOR = 0.90

# A tag-family hit (:func:`tag_family_keys`): retrieved, below every exact tier.
TAG_FAMILY_SCORE = 0.5


def tag_family_keys(q_data: tuple, rows) -> frozenset:
    """The tag keys a query clause is an exact value of anywhere on
    ``rows``: their other rows (``building=yes`` for ``house``) match at
    :data:`TAG_FAMILY_SCORE`."""
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
                norm[sv] = fuzzy_normalize(sv)
            if norm[sv] in want:
                out.add(str(k))
    return frozenset(out)


def token_idf(rows) -> dict[str, float]:
    """``idf(t) = ln(1 + N/(1+df(t)))``, ``df`` counting rows over their
    ``search_text`` and ``name`` tokens; a token no row carries is absent."""
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


@functools.lru_cache(maxsize=1 << 18)
def fuzzy_normalize(text: str) -> str:
    """Lowercase, punctuation out, ``_-/=`` as spaces, abbreviations expanded."""
    text = text.lower()
    text = re.sub(r"[_\-/=]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(_FUZZY_ABBREVS.get(t, t) for t in text.split())


def singularize(word: str) -> str:
    """One best-guess English singular."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "ses", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        return word[:-1]
    return word


def _singular_forms(word: str) -> set[str]:
    """Every plausible singular: ``-ses`` gives both "bus" and "house"."""
    out = {word, singularize(word)}
    if word.endswith("ses") and len(word) > 3:
        out.add(word[:-1])          # houses -> house, beside buses -> bus
    return out


def _fuzzy_expand_tokens(text: str) -> set[str]:
    """Token set after normalization + singularization."""
    tokens = set(fuzzy_normalize(text).split())
    return {f for t in tokens for f in _singular_forms(t)}


def _word_forms(text: str) -> tuple:
    """One frozenset of spellings per word, so coverage counts words."""
    return tuple(frozenset(_singular_forms(t))
                 for t in dict.fromkeys(fuzzy_normalize(text).split()))


COLORS = frozenset(COLOR_FAMILY)


def _row_match_text(row) -> str:
    """A row's name, search_text and ``k v`` tag pairs as one string."""
    return (f"{row.get('name') or ''} {row.get('search_text') or ''} "
            + " ".join(f"{k} {v}"
                       for k, v in (row.get("tags") or {}).items()))


def _colors_read(rows) -> set:
    """The palette words in the rows' ``color`` fields, mixture or rendered
    cell alike."""
    return COLORS & set(re.findall(r"[a-z]+", " ".join(
        r["color"].lower() for r in rows
        if isinstance(r, dict) and isinstance(r.get("color"), str))))


def unmatched_query_tokens(query: str, rows):
    """Query tokens (at least 3 long) no matched row carries,
    longest first, one per prefix family; a palette word the rows were
    perceived as counts as carried."""
    qt = {t for t in _fuzzy_expand_tokens(query) if len(t) >= 3}
    if not qt:
        return []
    rows = list(rows)
    read = _colors_read(rows)
    if read:
        qt = {t for t in qt if singularize(t) not in read}
        if not qt:
            return []
    carried: set = set()
    for row in rows:
        text = _row_match_text(row)
        if text:
            carried |= _fuzzy_expand_tokens(text)

    def is_carried(tok: str) -> bool:
        # A prefix either way: the singular is lossy ("houses" -> "hous").
        return any(tok.startswith(c) or c.startswith(tok) for c in carried)

    kept: list = []
    for tok in sorted((t for t in qt if not is_carried(t)), key=len):
        if not any(tok.startswith(k) or k.startswith(tok) for k in kept):
            kept.append(tok)
    return sorted(kept, key=lambda t: (-len(t), t))


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance."""
    if len(a) < len(b):
        return edit_distance(b, a)
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
    """One group per ``key=value`` token of a clause: its normalized
    ``(key, value)`` forms, the value extended by each following bare word
    (``addr_street=Castle Street`` -> castle, castle street)."""
    toks = clause.split()
    groups = []
    for i, tok in enumerate(toks):
        if "=" not in tok or tok.startswith("=") or tok.endswith("="):
            continue
        k, v = tok.split("=", 1)
        kn, vn = fuzzy_normalize(k), fuzzy_normalize(v)
        if not (kn and vn):
            continue
        forms = [(kn, vn)]
        ext = v
        for nxt in toks[i + 1:]:
            if "=" in nxt:
                break
            ext = f"{ext} {nxt}"
            en = fuzzy_normalize(ext)
            if en and (kn, en) != forms[-1]:
                forms.append((kn, en))
        groups.append(tuple(forms))
    return tuple(groups)


def fuzzy_prepare_query(query: str) -> tuple:
    """The query's ``OR`` clauses, each ``(normalized text, token set with
    singulars, spellings per word, key=value groups)``."""
    clauses = [c for c in re.split(r"\s+OR\s+", query or "") if c.strip()] \
        or [query or ""]
    out = []
    for c in clauses:
        out.append((fuzzy_normalize(c), _fuzzy_expand_tokens(c),
                    _word_forms(c), _extract_kv_pairs(c)))
    return tuple(out)


@functools.lru_cache(maxsize=1 << 18)
def _candidate_forms(c: str) -> frozenset:
    """Every singular spelling of a normalized candidate's words."""
    return frozenset(f for t in c.split() for f in _singular_forms(t))


def fuzzy_match_score(clause: tuple, candidate: str, idf: dict) -> float:
    """1.0 when the clause equals ``candidate``; else the share of the
    clause's idf mass the candidate covers, capped at ``TOKEN_BAG_CEIL``. A
    word weighs the idf of its commonest spelling, and a word no row carries
    leaves the denominator."""
    q_norm, qw = clause[0], clause[2]
    c = fuzzy_normalize(candidate)
    if not q_norm or not c:
        return 0.0
    if q_norm == c:
        return 1.0
    ct = _candidate_forms(c)
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
    return min(num / den, TOKEN_BAG_CEIL)


def find_score(q_data: tuple, search_text: str, name: str | None = None,
               tags: dict | None = None, *, idf: dict) -> float:
    """The tiered score (module docstring) of a prepared query over one row,
    max over its clauses; ``idf`` is :func:`token_idf` of the rows scored."""
    name_norm = fuzzy_normalize(name) if name else ""
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
                s = NAME_TIER_FLOOR + 0.09 * (len(q_norm) / len(name_norm))
        if s < 1.0 and kv and tags:
            if tag_pairs is None:
                tag_pairs = {
                    (fuzzy_normalize(str(k)), fuzzy_normalize(str(v)))
                    for k, v in tags.items()}
            matched = sum(
                1 for grp in kv if any(p in tag_pairs for p in grp))
            if matched:
                s = max(s, KV_TIER_BASE + 0.09 * (matched / len(kv)))
        if s < KV_TIER_BASE and q_norm and tags:
            if tag_tokens is None:
                tag_tokens = set()
                for k, v in tags.items():
                    tag_tokens.add(fuzzy_normalize(str(k)))
                    tag_tokens.add(fuzzy_normalize(str(v)))
            if q_norm in tag_tokens:
                s = max(s, KV_TIER_BASE)
        if s < KV_TIER_BASE:
            f = fuzzy_match_score(clause, search_text or "", idf)
            if name:
                f = max(f, fuzzy_match_score(clause, name, idf))
            s = max(s, f)
        best = max(best, s)
    return best
