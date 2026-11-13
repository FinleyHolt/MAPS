"""The ops that cut and sort a result: ``filter_items`` drops rows and
``score_by`` orders them, by distance or by a colour column."""

from __future__ import annotations

from maps.query import color_words
from maps.query import spatial_math as G
from maps.query.op_io import (
    ProgramError,
    Result,
    as_rows,
    contact_fact,
    keeps_order,
    present_keys_hint,
    quote_value,
    relate_ran,
)
from maps.query.result_tables import number_or_none


def is_color_column_key(k) -> bool:
    """A ``color_<word>`` column key (``color_conf`` is a surface field)."""
    return isinstance(k, str) and k.startswith("color_") and k != "color_conf"


def _resolve_color_word(word: str) -> tuple[str, str | None]:
    """A cue color word on the closed palette: ``(canonical, note)``, the
    note set when the word was normalized (navy -> blue). Off-palette
    raises."""
    w = str(word or "").strip().lower()
    canon = color_words.normalize_color(w)
    if canon not in color_words.COLOR_ENUM:
        raise ProgramError(
            f"{quote_value(w)} is not a color on the closed palette; color columns "
            f"are color_<word> with word one of: "
            f"{', '.join(color_words.COLOR_ENUM)} (an open word that maps onto the "
            f"palette is accepted, e.g. navy -> blue)."
        )
    note = (
        None
        if canon == w
        else f"color {quote_value(w)} maps onto the closed palette as {quote_value(canon)}; "
        f"the column is color_{canon}."
    )
    return canon, note


def _attach_color_column(items, canon):
    """Attach ``color_<canon>`` (the confidence-shrunk color match) to every
    row with a perceived color; a row without one omits the column."""
    col = "color_" + canon
    out = []
    for it in items:
        c = it.get("color") if isinstance(it, dict) else None
        if isinstance(c, str) and c.strip():
            out.append(
                {**it, col: round(color_words.color_match(c, canon, it.get("color_conf")), 3)}
            )
        else:
            out.append(it)
    return out


def run_score_by(cx, a):
    """The sort: a signed, key-normalized composite over exactly the
    columns the emphasis names."""
    items = as_rows(cx, a["items"])
    notes: list = []
    emphasis: dict = {}
    for el in a["emphasis"]:
        key = el["key"]
        if is_color_column_key(key):
            # color_navy weights color_blue, noted.
            canon, cnote = _resolve_color_word(key[len("color_") :])
            if cnote is not None:
                notes.append(cnote)
            key = "color_" + canon
        emphasis[key] = float(el["w"])
    for key in emphasis:
        # Naming a color column attaches it.
        if is_color_column_key(key) and not any(it.get(key) is not None for it in items):
            items = _attach_color_column(items, key[len("color_") :])
    # A key no row carries would be a silent constant term.
    dropped_color: list = []
    for key in list(emphasis):
        if items and not any(isinstance(it.get(key), (int, float)) for it in items):
            if is_color_column_key(key):
                # Nothing has a perceived colour before a flight sees it: the
                # term drops, unless it is the only one.
                if len(emphasis) > 1:
                    dropped_color.append(key)
                    del emphasis[key]
                    continue
                raise ProgramError(
                    f"nothing orders by {quote_value(key)} here: no row carries a "
                    f"perceived color, so no color column attaches (rows "
                    f"without a color read omit color_<word> columns). "
                    f"{present_keys_hint(items, True)}"
                )
            raise ProgramError(
                f"nothing orders by {quote_value(key)} here: no row carries a "
                f"numeric {quote_value(key)}. "
                f"{contact_fact(cx, key)}"
                f"{present_keys_hint(items, True)}"
            )
    for key in dropped_color:
        notes.append(
            f"{quote_value(key)} ranked nothing here: no row in this set carries "
            "a perceived colour yet (a row gets one once a flight has "
            "passed over it); the other keys still rank."
        )
    keys = sorted(emphasis)  # deterministic; the sum is order-independent
    return Result(
        G.score_by(items, keys, [emphasis[k] for k in keys]),
        "sorted",
        ordered=True,
        op_cols=keys,
        notes=notes,
    )


def run_filter_items(cx, a):
    """The op that drops rows, keeping its input's order: ``le`` on a
    numeric column, ``truthy`` on a present key, ``on_side`` keeps a row its
    cardinal column places on the side or does not place, ``in_or_blank``
    keeps a word column reading one of ``value`` or blank."""
    items = as_rows(cx, a["items"])
    key, opr, val = a["key"], a["op"], a.get("value")
    if opr == "le":
        if items and not any(isinstance(it.get(key), (int, float)) for it in items):
            raise ProgramError(
                f"no item carries a numeric {quote_value(key)}. "
                f"{contact_fact(cx, key)}"
                f"{present_keys_hint(items, True)}"
            )
        # The nearest excluded row, the bound that would admit it and how
        # many rows that bound admits.
        kept: list = []
        excluded: list = []
        for it in items:
            v = it.get(key)
            (kept if v is not None and v <= val else excluded).append(it)

        stats = {"cut": f"{len(items)}->{len(kept)}"}
        excl_vals = [(e, it) for it in excluded if (e := number_or_none(it.get(key))) is not None]
        if excl_vals:
            ne_v, ne_it = min(excl_vals, key=lambda t: t[0])
            stats["nearest_excluded"] = {
                "fid": str(ne_it.get("fid") or "?"),
                key: round(float(ne_it.get(key)), 3),
            }
            stats["admit_at"] = round(float(ne_v), 3)
            stats["n_at_admit"] = sum(
                1 for it in items if (v := number_or_none(it.get(key))) is not None and v <= ne_v
            )
        return Result(kept, "cut", ordered=keeps_order(items), op_cols=[key], cut_stats=stats)
    if items and not any(key in it for it in items) and not relate_ran(key, items):
        raise ProgramError(
            f"no item carries {quote_value(key)}. "
            f"{contact_fact(cx, key)}"
            f"{present_keys_hint(items, False)}"
        )
    if opr == "in_or_blank":
        # The caller passes every word that counts (south: southeast,
        # southwest too).
        on = [it for it in items if it.get(key) is None or it.get(key) in val]
        blank = sum(1 for it in on if it.get(key) is None)
        want = list(val)
        stats = {
            "cut": f"{len(items)}->{len(on)}",
            "counts_as": want,
            "reads_one_of_them": len(on) - blank,
            "reads_another_word_dropped": len(items) - len(on),
            "unmeasured_kept": blank,
        }
        if not (len(on) - blank):
            stats["nothing_reads_those_words"] = (
                f"no surveyed row reads "
                f"any of {want} in {quote_value(key)}; "
                f"the rows kept are the ones the geometry never "
                f"measured (a row TOUCHING the reference has no "
                f"facing word). Widen, or read the column and "
                f"see which words are actually there."
            )
    elif opr == "on_side":
        on = [it for it in items if it.get(key) is None or (number_or_none(it.get(key)) or 0) > 0]
        blank = sum(1 for it in on if it.get(key) is None)
        stats = {
            "cut": f"{len(items)}->{len(on)}",
            "on_side": len(on) - blank,
            "other_side_dropped": len(items) - len(on),
            "no_side_kept": blank,
        }
        if not (len(on) - blank):
            dropped = [it for it in items if it not in on]
            seen = any(str(it.get("fid") or "").startswith("detected_") for it in dropped)
            watched = any(str(it.get("class") or "").lower() in cx.watched for it in dropped)
            stats["nothing_on_that_side"] = "nothing in the world state lies on that side" + (
                ""
                if seen or watched
                else " -- what is there has to be SEEN: look_for the class, then "
                "explore the place to segment it"
            )
    else:  # truthy
        on, stats = [it for it in items if it.get(key)], None
    # A cut that keeps every way of a name's tied set is still that set.
    return Result(
        on,
        "cut",
        ordered=keeps_order(items),
        op_cols=[key],
        cut_stats=stats,
        name_set=getattr(items, "name_set", False) and len(on) == len(items),
    )
