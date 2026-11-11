"""An explore's stops: the walk over one place's footprint, or one pass
chained over a set's members, and the note saying how much they reach."""

from __future__ import annotations

import math

from maps.xy_token import format_xy_token, parse_xy_token


def spread(seq: list, k: int) -> list:
    """``k`` (clamped to 1 .. ``len(seq)``) items of ``seq`` at an even stride,
    both ends kept; one item is the middle one, since an end would see half
    the shape."""
    k = max(1, min(k, len(seq)))
    if k == 1:
        return [seq[(len(seq) - 1) // 2]]
    stride = (len(seq) - 1) / (k - 1)
    return [seq[int(round(i * stride))] for i in range(k)]


def _coverage_stops(tok: str, spec, spacing):
    """The engine's stops over one fid's footprint; None for a point,
    a fid with no cells, or a run without an observation extent."""
    if (parse_xy_token(tok) is None and spacing is not None
            and spec.coverage_stops is not None):
        return spec.coverage_stops(tok, spacing)
    return None


def _uncovered_segments(name: str, rows: dict, covered: set) -> str:
    """`` '<name>' has N more segment(s) no pass has covered (...).``,
    or "" when every segment bearing the name is covered."""
    sibs = sorted(f for f, r in rows.items()
                  if r.get("name") == name and f not in covered)
    if not sibs:
        return ""
    head = ", ".join(sibs[:6])
    more = f", +{len(sibs) - 6} more" if len(sibs) > 6 else ""
    return (f" {name!r} has {len(sibs)} more segment(s) no pass has "
            f"covered ({head}{more}).")


def cover_items(accepted: list, said, *, spec, spacing, rows: dict,
                covered: set, pose_xy):
    """One explore -> ``(move items, note, subject, segments)``: the stops
    over the subject's footprint in shape order and nothing after them,
    and a note saying how much of the shape they reach. A set is one
    pass over the union (``_cover_set_items``)."""
    if len(accepted) > 1:
        return _cover_set_items(accepted, said, spec=spec,
                                spacing=spacing, rows=rows,
                                covered=covered, pose_xy=pose_xy)
    item = accepted[0]
    tok = str(item.get("target_fid") or "")
    stops = _coverage_stops(tok, spec, spacing)
    note = ""
    if stops:
        walk = sum(math.dist(a, b) for a, b in zip(stops, stops[1:]))
        reached = min(1.0, len(stops) * spacing / walk) if walk else 1.0
        if len(stops) == 1:
            note = (f"{tok} fits inside one observation; covered it "
                    "from the middle in one stop.")
        elif reached >= 1.0:
            note = (f"covered {walk:.0f} m of {tok} in "
                    f"{len(stops)} stops.")
        else:
            note = (f"covered {reached:.0%} of {tok}'s "
                    f"{walk:.0f} m in {len(stops)} stops.")
        items = [{"target_fid": format_xy_token(float(p[0]),
                                                float(p[1]))}
                 for p in stops]
    else:
        # A point, a fid with no cells, or no observation extent: the
        # pass is the one stop it is.
        note = (f"{tok} has no footprint to cover, so this pass is "
                "the one stop it is.")
        items = [dict(item)]
    # A named feature is often many segments: the note names those no
    # pass has covered.
    covered.add(str(tok))
    name = (rows.get(str(tok)) or {}).get("name")
    if name:
        note += _uncovered_segments(name, rows, covered)
    return items, note, tok, 1


def _cover_set_items(accepted: list, said, *, spec, spacing,
                     rows: dict, covered: set, pose_xy):
    """The multi-member half of ``cover_items``: each member's own walk,
    chained greedily by nearest endpoint from the live pose (a member
    reversed when its far end is nearer), dropping a stop within half a
    spacing of one already kept; a chain longer than the flight budget
    keeps that many stops (``spread``)."""
    # ponytail: O(n*m) greedy chaining, not a TSP; order the members
    # optimally if a pass's flown length ever matters.
    members: list[dict] = []
    for it in accepted:
        tok = str(it.get("target_fid") or "")
        pt = parse_xy_token(tok)
        stops = _coverage_stops(tok, spec, spacing)
        if stops:
            pts = [[float(p[0]), float(p[1])] for p in stops]
            members.append({
                "tok": tok, "pts": pts, "footprint": True,
                "items": [{"target_fid": format_xy_token(p[0], p[1])}
                          for p in pts],
                "walk": sum(math.dist(a, b)
                            for a, b in zip(pts, pts[1:]))})
        else:
            # one stop; its recorded position anchors the chain.
            xy = ((rows.get(tok) or {}).get("xy")
                  if pt is None else list(pt))
            p = ([float(xy[0]), float(xy[1])]
                 if isinstance(xy, (list, tuple)) and len(xy) == 2
                 else None)
            members.append({"tok": tok, "pts": [p] if p else [],
                            "footprint": False,
                            "items": [dict(it)], "walk": 0.0})
    # greedy nearest-endpoint chain, seeded from where the drone is
    cur = pose_xy
    cur = [float(cur[0]), float(cur[1])] if cur else None
    chained: list[dict] = []
    rest = list(members)
    while rest:
        if cur is None:
            nxt = rest.pop(0)
        else:
            def gap(m):
                if not m["pts"]:
                    return float("inf")
                return min(math.dist(cur, m["pts"][0]),
                           math.dist(cur, m["pts"][-1]))
            nxt = min(rest, key=gap)
            rest.remove(nxt)
            if (len(nxt["pts"]) > 1
                    and math.dist(cur, nxt["pts"][-1])
                    < math.dist(cur, nxt["pts"][0])):
                nxt["pts"].reverse()
                nxt["items"].reverse()
        chained.append(nxt)
        if nxt["pts"]:
            cur = nxt["pts"][-1]
    # flatten in chain order, dropping stops a kept stop already covers
    kept_pts: list = []
    flat: list[tuple[dict, dict]] = []
    for m in chained:
        for i, mv in enumerate(m["items"]):
            p = m["pts"][i] if i < len(m["pts"]) else None
            if (p is not None and spacing is not None
                    and any(math.dist(p, q) < spacing / 2.0
                            for q in kept_pts)):
                continue
            flat.append((m, mv))
            if p is not None:
                kept_pts.append(p)
    if len(flat) > spec.flight_budget:
        flat = spread(flat, spec.flight_budget)
    for m in chained:
        m["kept"] = sum(1 for f, _ in flat if f is m)
    total = sum(m["walk"] for m in chained)
    n_stops = len(flat)
    reached = (min(1.0, n_stops * spacing / total)
               if total and spacing else 1.0)
    subject = str(said) if said else f"{len(chained)} places"
    if not any(m["footprint"] for m in chained):
        note = (f"{subject} has no footprint to cover, so this pass is "
                + ("its one stop." if n_stops == 1
                   else f"its {n_stops} stops."))
    elif reached >= 1.0:
        note = (
            (f"covered {len(chained)} places in {n_stops} stops; each "
             f"fits inside one observation.")
            if not total else
            (f"covered {total:.0f} m of {subject} across "
             f"{len(chained)} segments in {n_stops} stops."))
    else:
        note = (f"covered {reached:.0%} of {subject}'s "
                f"{total:.0f} m.")

    def pct(m):
        if not m["walk"] or spacing is None:
            return 1.0
        return min(1.0, m["kept"] * spacing / m["walk"])
    mm = ", ".join(f"{m['tok']} {pct(m):.0%}" for m in chained[:8])
    mm_more = (f", +{len(chained) - 8} more"
               if len(chained) > 8 else "")
    note = f"{note} ({mm}{mm_more})"
    for m in chained:
        covered.add(m["tok"])
    names = {(rows.get(m["tok"]) or {}).get("name")
             for m in chained}
    names.discard(None)
    names.discard("")
    for name in sorted(names):
        note += _uncovered_segments(name, rows, covered)
    return ([mv for _, mv in flat], note, subject, len(chained))
