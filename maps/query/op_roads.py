"""The road ops: ``crossings``, where two roads cross, and ``ends``, where a
road ends."""

from __future__ import annotations

from itertools import islice

from maps.query import spatial_math as G
from maps.query.op_io import (
    ProgramError, Result, _SAY_A_FID, _as_said, _named_rows, _quote,
    _refuse_if_empty)
from maps.query.row_geometry import chains_of
from maps.xy_token import format_xy_token


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


def _sibling_crossing(by_fid, la, fa, lb, fb) -> str:
    """What a crossing that did not meet adds: where two same-named
    segments touch, from a sweep of at most 24 untested pairs (each side's
    first 8 tested rows and first 8 siblings), else how many siblings it
    swept, else nothing."""
    def side(lab, fids):
        tested = [by_fid[f] for f in sorted(fids) if f in by_fid][:8]
        sibs = [c for c in by_fid.values() if str(c.get("name")) == str(lab)
                and str(c.get("fid")) not in fids][:8]
        return tested, sibs
    (ta, sa), (tb, sb) = side(la, fa), side(lb, fb)
    pairs = ((ra, rb) for ra in ta + sa for rb in tb + sb
             if not (str(ra.get("fid")) in fa and str(rb.get("fid")) in fb))
    for ra, rb in islice(pairs, 24):
        px, _ = G.polyline_crossings(chains_of(ra), chains_of(rb))
        if px:
            tok = format_xy_token(float(round(px[0][0])),
                                  float(round(px[0][1])))
            return (f" But the NAMES do cross elsewhere: {ra.get('fid')} and "
                    f"{rb.get('fid')} (same-named segments) touch at {tok} "
                    f"-- intersection on those rows returns it.")
    if sa or sb:
        return (f" {len(sa)} other {la} segment(s) and {len(sb)} other {lb} "
                f"segment(s) exist; no swept pair touches either.")
    return ""


def _crossings(cx, a):
    """The places two features meet: one row per distinct crossing, its fid
    the [x, y] token of its whole-metre point. None raises."""
    sides, members = [], []
    for el in a["refs"]:
        nodes = _named_rows(cx, el, "xy")
        members.append((nodes, el))
        polys = [p for n in nodes for p in chains_of(n)]
        if not polys:
            _refuse_if_empty(cx, el, "cross")
            raise ProgramError(
                f"{_quote(_as_said(el))} names no geometry on this surface "
                f"-- a crossing needs two features that HAVE a shape. "
                + _SAY_A_FID)
        sides.append((_x_label(nodes, el), polys,
                      {str(n.get("fid")) for n in nodes}))
    # A row on both sides would hand back its own corners.
    shared = sides[0][2] & sides[1][2]
    if shared:
        same = "the same feature" if sides[0][2] == sides[1][2] \
            else "the same feature in part"
        raise ProgramError(
            f"{sides[0][0]} and {sides[1][0]} are {same} "
            f"({', '.join(sorted(shared)[:4])}), and nothing crosses "
            f"itself -- a feature's own corners are not crossings. "
            f"Name two DIFFERENT features (find each one and read its "
            f"row).")
    (la, fa), (lb, fb) = ((sides[0][0], sides[0][2]),
                          (sides[1][0], sides[1][2]))
    xpts, nearest = G.polyline_crossings(sides[0][1], sides[1][1])
    if not xpts:
        sib = _sibling_crossing(cx.by_fid, la, fa, lb, fb)
        raise ProgramError(
            f"{la} and {lb} do not meet HERE: the tested rows "
            f"({', '.join(sorted(fa)[:4])} against "
            f"{', '.join(sorted(fb)[:4])}) come no closer than "
            f"{nearest:.1f} m, and a crossing is where they "
            f"TOUCH.{sib} A point near one of them is not their "
            f"intersection; relate to ONE of them instead "
            f"('... near <feature>').")
    def namer(i):
        """A crossing's name on side ``i``: the side's label when its rows
        share one name, else the label of the rows the crossing is on. Each
        row's crossings come from the call above with that side cut to the
        row, so every crossing is on a row of each side."""
        nodes, el = members[i]
        if len({n.get("name") for n in nodes}) <= 1:
            return lambda p: sides[i][0]
        on = [(n, {tuple(q) for q in G.polyline_crossings(
            *([chains_of(n), sides[1][1]] if i == 0 else
              [sides[0][1], chains_of(n)]))[0]}) for n in nodes]
        return lambda p: _x_label([n for n, pts in on if tuple(p) in pts], el)

    name_a, name_b = namer(0), namer(1)
    rows = []
    # Whole metres: the fid is the coordinate, and the model spells whole
    # numbers only.
    seen: set = set()
    for p in xpts:
        x, y = float(round(p[0])), float(round(p[1]))
        tok = format_xy_token(x, y)
        # Rounding can collapse two crossings onto one fid.
        if tok in seen:
            continue
        seen.add(tok)
        rows.append({"fid": tok, "xy": [x, y], "name": f"{name_a(p)} x {name_b(p)}"})
    notes = [
        f"{len(rows)} distinct crossing{'' if len(rows) == 1 else 's'} "
        f"of {la} and {lb}"
        + ("." if len(rows) == 1 else
           " -- each is its OWN row with its OWN fid, so narrow to the "
           "one you mean (order or cut this set) rather than averaging "
           "them; a point between crossings is at none of them.")]
    return Result(rows, ordered=True, notes=notes, set_notes=notes)


def _ends(cx, a):
    """A road's termini (end0_xy / end1_xy) as rows."""
    rows = []
    n_end = 0
    no_line: list = []
    for el in a["refs"]:
        nodes = _named_rows(cx, el, "centerline")
        if not nodes:
            _refuse_if_empty(cx, el, "take the ends of")
            raise ProgramError(
                f"{_quote(_as_said(el))} names no row on this surface. "
                + _SAY_A_FID)
        for nd in nodes:
            cl = nd.get("centerline")
            if not (isinstance(cl, list) and len(cl) >= 2):
                no_line.append(
                    f"{nd.get('fid')}"
                    + (f" ({nd.get('name')})" if nd.get("name") else ""))
                continue
            lab = str(nd.get("name") or nd.get("fid"))
            for which, pt in (("end0", cl[0]), ("end1", cl[-1])):
                xy = [float(round(pt[0])), float(round(pt[1]))]
                rows.append({"fid": f"end_{n_end}", "xy": xy,
                             "name": f"{lab} {which}"})
                n_end += 1
    if not rows:
        named = ", ".join(no_line[:4]) + (
            f" (+{len(no_line) - 4} more)" if len(no_line) > 4 else "")
        raise ProgramError(
            f"none of those rows carries a recorded centerline -- "
            f"{named} have no line geometry on this surface, so "
            f"no termini exist for them. ends reads a centerline; "
            f"end0_xy/end1_xy on a row is the same fact served as "
            f"columns.")
    notes = [
        f"{len(rows)} terminus rows; each is its OWN row with its "
        f"own fid, so index the one you mean."]
    return Result(rows, ordered=True, notes=notes, set_notes=notes)
