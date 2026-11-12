"""Colour as the op program reads it: the palette words and their families,
a colour word matched against a row's perceived colour, and the colour
columns a result serves."""

from __future__ import annotations


# Colour matches by FAMILY: perception's usual slip stays inside one (gray
# reads silver under glare, red reads maroon).
# Palette word -> family; key order is the order a label's colour is read in
# (surface._color_of).
COLOR_FAMILY = {
    "white": "achromatic", "black": "achromatic", "gray": "achromatic", "silver": "achromatic",
    "red": "red", "blue": "blue", "green": "green", "yellow": "yellow",
    "orange": "warm", "brown": "warm", "tan": "warm", "beige": "warm", "gold": "yellow",
    "maroon": "red", "purple": "purple",
}


def _color_word_raw(a: str, b: str) -> float:
    """1.0 same word, 0.8 same family, 0.0 another family, 0.5 when either is
    absent; a word outside the families matches by substring."""
    if not a or not b:
        return 0.5
    if a == b:
        return 1.0
    fa, fb = COLOR_FAMILY.get(a), COLOR_FAMILY.get(b)
    if fa is not None and fb is not None:
        return 0.8 if fa == fb else 0.0
    return 1.0 if (a in b or b in a) else 0.0  # unknown words: substring fallback


def _parse_color_mixture(s: str) -> "list[tuple[str, float]] | None":
    """``[(word, fraction), ...]`` of a mixture ("red 0.45 / black 0.30",
    ``|`` also separates), or ``None`` when ``s`` is not one."""
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
    """The value of a ``color_<word>`` column: :func:`_color_word_raw`, each
    mixture component shrunk toward 0.5 by its fraction and the best kept,
    then shrunk toward 0.5 by ``conf`` when given."""
    b = (cue_color or "").strip().lower()
    mix = _parse_color_mixture(cand_color or "")
    if mix is not None:
        raw = max(0.5 + frac * (_color_word_raw(w, b) - 0.5) for w, frac in mix)
    else:
        a = (cand_color or "").strip().lower()
        raw = _color_word_raw(a, b)
    if conf is None:
        return raw
    c = 0.0 if conf < 0.0 else (1.0 if conf > 1.0 else float(conf))
    return 0.5 + c * (raw - 0.5)


# The closed palette the surface serves; :func:`normalize_color` maps an open
# word onto it.
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
    """``word`` on :data:`COLOR_ENUM` by synonym, or itself for the caller to
    refuse."""
    w = (word or "").strip().lower()
    if w in COLOR_ENUM:
        return w
    return _COLOR_SYNONYMS.get(w, w)


# ``color_<word>`` = ``color_match(row.color, word, row.color_conf)``; a row
# with no perceived colour omits it.
COLOR_COLUMNS = frozenset("color_" + w for w in COLOR_ENUM)


def compact_color_cell(color_str: str, *, shade: "str | None", uncertain: bool,
                       rich: bool) -> str:
    """The rendered colour cell ``[dk|lt ]word[/word2][?]`` of a mixture
    (``shade`` is ``dk``, ``lt`` or None): ``word2`` when ``rich`` and the
    second component crosses a family, ``?`` on an uncertain read."""
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
        if COLOR_FAMILY.get(word) != COLOR_FAMILY.get(w2):
            second = w2
    head = f"{shade} " if shade else ""
    body = f"{word}/{second}" if second else word
    return f"{head}{body}?" if uncertain else f"{head}{body}"
