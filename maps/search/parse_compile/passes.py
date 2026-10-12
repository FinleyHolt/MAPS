"""An explore's stops: the walk over one place's footprint, or one pass
chained over a set's members, and the note saying how much they reach."""

from __future__ import annotations

import math

from maps.util.xy_token import format_xy_token, parse_xy_token


def _coverage_stops(_tok: str, spec, spacing):
    """The engine's stops over one fid's footprint; None for a point,
    a fid with no cells, or a run without an observation extent."""
    if (parse_xy_token(_tok) is None and spacing is not None
            and spec.coverage_stops is not None):
        return spec.coverage_stops(_tok, spacing)
    return None


def _uncovered_segments(_nm: str, rows: dict, covered: set) -> str:
    """`` '<name>' has N more segment(s) no pass has covered (...).``,
    or "" when every segment bearing the name is covered."""
    _sibs = sorted(f for f, r in rows.items()
                   if r.get("name") == _nm and f not in covered)
    if not _sibs:
        return ""
    _head = ", ".join(_sibs[:6])
    _more = f", +{len(_sibs) - 6} more" if len(_sibs) > 6 else ""
    return (f" {_nm!r} has {len(_sibs)} more segment(s) no pass has "
            f"covered ({_head}{_more}).")


def cover_items(_accepted_items: list, _said, *, spec, spacing, rows: dict,
                covered: set, pose_xy):
    """One explore -> ``(move items, note, subject, segments)``: the stops
    over the subject's footprint in shape order and nothing after them,
    and a note saying how much of the shape they reach. A set is one
    pass over the union (``_cover_set_items``)."""
    if len(_accepted_items) > 1:
        return _cover_set_items(_accepted_items, _said, spec=spec,
                                spacing=spacing, rows=rows,
                                covered=covered, pose_xy=pose_xy)
    _item = _accepted_items[0]
    _tok = str(_item.get("target_fid") or "")
    _span = spacing
    _stops = _coverage_stops(_tok, spec, spacing)
    _note = ""
    if _stops:
        _walk = sum(math.dist(a, b) for a, b in zip(_stops, _stops[1:]))
        _reached = min(1.0, len(_stops) * _span / _walk) if _walk else 1.0
        if len(_stops) == 1:
            _note = (f"{_tok} fits inside one observation; covered it "
                     "from the middle in one stop.")
        elif _reached >= 1.0:
            _note = (f"covered {_walk:.0f} m of {_tok} in "
                     f"{len(_stops)} stops.")
        else:
            _note = (f"covered {_reached:.0%} of {_tok}'s "
                     f"{_walk:.0f} m in {len(_stops)} stops.")
        _items = [{"target_fid": format_xy_token(float(p[0]),
                                                 float(p[1]))}
                  for p in _stops]
    else:
        # A point, a fid with no cells, or no observation extent: the
        # pass is the one stop it is.
        _note = (f"{_tok} has no footprint to cover, so this pass is "
                 "the one stop it is.")
        _items = [dict(_item)]
    # A named feature is often many segments: the note names those no
    # pass has covered.
    covered.add(str(_tok))
    _nm = (rows.get(str(_tok)) or {}).get("name")
    if _nm:
        _note += _uncovered_segments(_nm, rows, covered)
    return _items, _note, _tok, 1


def _cover_set_items(_accepted_items: list, _said, *, spec, spacing,
                     rows: dict, covered: set, pose_xy):
    """The multi-member half of ``cover_items``: each member's own walk,
    chained greedily by nearest endpoint from the live pose (a member
    reversed when its far end is nearer), dropping a stop within half a
    spacing of one already kept; a chain longer than the flight budget
    keeps that many evenly spaced stops, its first among them."""
    # ponytail: O(n*m) greedy chaining, not a TSP; order the members
    # optimally if a pass's flown length ever matters.
    _span = spacing
    _members: list[dict] = []
    for _it in _accepted_items:
        _tok = str(_it.get("target_fid") or "")
        _pt = parse_xy_token(_tok)
        _stops = _coverage_stops(_tok, spec, spacing)
        if _stops:
            _pts = [[float(p[0]), float(p[1])] for p in _stops]
            _members.append({
                "tok": _tok, "pts": _pts,
                "items": [{"target_fid": format_xy_token(p[0], p[1])}
                          for p in _pts],
                "walk": sum(math.dist(a, b)
                            for a, b in zip(_pts, _pts[1:]))})
        else:
            # one stop; its recorded position anchors the chain.
            _xy = ((rows.get(_tok) or {}).get("xy")
                   if _pt is None else list(_pt))
            _p = ([float(_xy[0]), float(_xy[1])]
                  if isinstance(_xy, (list, tuple)) and len(_xy) == 2
                  else None)
            _members.append({"tok": _tok, "pts": [_p] if _p else [],
                             "items": [dict(_it)], "walk": 0.0})
    # greedy nearest-endpoint chain, seeded from where the drone is
    _cur = pose_xy
    _cur = [float(_cur[0]), float(_cur[1])] if _cur else None
    _chained: list[dict] = []
    _rest = list(_members)
    while _rest:
        if _cur is None:
            _next = _rest.pop(0)
        else:
            def _gap(_m):
                if not _m["pts"]:
                    return float("inf")
                return min(math.dist(_cur, _m["pts"][0]),
                           math.dist(_cur, _m["pts"][-1]))
            _next = min(_rest, key=_gap)
            _rest.remove(_next)
            if (len(_next["pts"]) > 1
                    and math.dist(_cur, _next["pts"][-1])
                    < math.dist(_cur, _next["pts"][0])):
                _next["pts"].reverse()
                _next["items"].reverse()
        _chained.append(_next)
        if _next["pts"]:
            _cur = _next["pts"][-1]
    # flatten in chain order, dropping stops a kept stop already covers
    _kept_pts: list = []
    _flat: list[tuple[dict, dict]] = []
    for _m in _chained:
        for _i, _mv in enumerate(_m["items"]):
            _p = _m["pts"][_i] if _i < len(_m["pts"]) else None
            if (_p is not None and _span is not None
                    and any(math.dist(_p, _q) < _span / 2.0
                            for _q in _kept_pts)):
                continue
            _flat.append((_m, _mv))
            if _p is not None:
                _kept_pts.append(_p)
    _n = spec.flight_budget
    if len(_flat) > _n:
        _stride = (len(_flat) - 1) / max(_n - 1, 1)
        _keep = sorted({int(round(_i * _stride)) for _i in range(_n)})
        _flat = [_flat[_i] for _i in _keep]
    for _m in _chained:
        _m["kept"] = sum(1 for _f, _ in _flat if _f is _m)
    _total = sum(_m["walk"] for _m in _chained)
    _n_stops = len(_flat)
    _reached = (min(1.0, _n_stops * _span / _total)
                if _total and _span else 1.0)
    _subject = str(_said) if _said else f"{len(_chained)} places"
    if _reached >= 1.0:
        _note = (
            (f"covered {len(_chained)} places in {_n_stops} stops; each "
             f"fits inside one observation.")
            if not _total else
            (f"covered {_total:.0f} m of {_subject} across "
             f"{len(_chained)} segments in {_n_stops} stops."))
    else:
        _note = (f"covered {_reached:.0%} of {_subject}'s "
                 f"{_total:.0f} m.")

    def _pct(_m):
        if not _m["walk"] or _span is None:
            return 1.0
        return min(1.0, _m["kept"] * _span / _m["walk"])
    _mm = ", ".join(f"{_m['tok']} {_pct(_m):.0%}" for _m in _chained[:8])
    _mm_more = (f", +{len(_chained) - 8} more"
                if len(_chained) > 8 else "")
    _note = f"{_note} ({_mm}{_mm_more})"
    for _m in _chained:
        covered.add(_m["tok"])
    _names = {(rows.get(_m["tok"]) or {}).get("name")
              for _m in _chained}
    _names.discard(None)
    _names.discard("")
    for _nm in sorted(_names):
        _note += _uncovered_segments(_nm, rows, covered)
    return ([_mv for _, _mv in _flat], _note, _subject, len(_chained))
