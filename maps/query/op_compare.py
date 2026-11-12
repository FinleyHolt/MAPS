"""The ``compare`` op: each ref's relation columns (distance, bearing, side,
containment) attached to every item row, a set measured as one place."""

from __future__ import annotations

from maps.query import spatial_math as G
from maps.query.op_io import (
    _SAY_A_FID,
    ProgramError,
    Result,
    _as_rows,
    _attach_standing_columns,
    _ordered,
    _quote,
    sanitize_token,
)
from maps.query.result_tables import _MODEL_HIDDEN_ROW_FIELDS
from maps.query.row_geometry import geom_of, place_geom
from maps.xy_token import is_xy

# A wall-clock net: a union measures members x items gaps.
_UNION_MAX_MEMBERS = 400


def _union_members(cx, ref_el, as_label) -> list:
    """The member rows of a union ref ``[fid | rN | [x, y], ...]``: a fid's
    row, a handle's rows (a row off the surface, a crossing, rides its own
    xy) and an ``[x, y]`` as an xy-only row."""
    members: list = []
    for x in ref_el:
        if is_xy(x):
            members.append({"xy": [float(x[0]), float(x[1])]})
            continue
        if cx.is_ref(x):
            got = [
                cx.by_fid.get(r.get("fid"), r)
                for r in cx.resolve(x)
                if isinstance(r, dict) and (r.get("fid") in cx.by_fid or is_xy(r.get("xy")))
            ]
            lbl = as_label or sanitize_token(x)
            if not got:
                raise ProgramError(
                    f"{lbl} is empty, so 'distance to {lbl}' is undefined on "
                    f"every row and nothing can be cut or ordered by it."
                    " If these are things the map does not carry (a "
                    "pool, solar panels, a shed, a car), nothing is "
                    "segmented until look_for names the class and a "
                    "flight goes over the place -- arm it, fly, then "
                    "find them again."
                )
            if len(got) > _UNION_MAX_MEMBERS:
                raise ProgramError(
                    f"{lbl} holds {len(got)} rows and a union folds at most "
                    f"{_UNION_MAX_MEMBERS}: measuring to the nearest of that "
                    f"many is a whole-map scan per row. Narrow it first "
                    f"(within=, a cardinal, or a tighter query), then "
                    f"measure to what is left."
                )
            members.extend(got)
            continue
        node = cx.by_fid.get(x)
        if node is None:
            raise ProgramError(f"{_quote(x)} is not a fid on this surface. " + _SAY_A_FID)
        members.append(node)
    return members


def _union_center(members):
    """``(member geoms, centre)`` of a union: the centre is the members'
    area-weighted centroid, else the mean of their centres, else None."""
    geoms: list = []
    wx = wy = wa = 0.0
    ucs: list = []
    for node in members:
        g, c = geom_of(node)
        geoms.append(g)
        if c is not None:
            ucs.append(c)
            a_m = G.loops_area(node.get("mask") or [])
            wx += a_m * float(c[0])
            wy += a_m * float(c[1])
            wa += a_m
    if wa > 0:
        return geoms, [wx / wa, wy / wa]
    if ucs:
        return geoms, [
            sum(float(c[0]) for c in ucs) / len(ucs),
            sum(float(c[1]) for c in ucs) / len(ucs),
        ]
    return geoms, None


def _unique_label(label, used: set, fid=None) -> str:
    """``label``, or on a collision ``label_<fid>`` (a row ref) and then
    ``label_<n>`` counting from 2."""
    lab, k = label, 2
    while lab in used:
        lab = f"{label}_{fid}" if fid is not None and k == 2 else f"{label}_{k}"
        k += 1
    used.add(lab)
    return lab


def _compare(cx, a):
    """Attach each ref's relation columns (:data:`_TABLE_LEGEND`) to every
    item row, in input order. A ref is a surface row, an [x, y], a staged
    viewpoint or a union ``{ref: [fid | rN | [x, y], ...]}`` measured as one
    place."""
    items = _as_rows(cx, a["items"])
    notes: list = []
    used_labels: set = set()
    standing: list = []  # staged viewpoint handles, in ref order

    # The cardinals asked for per ref (named by fid or label).
    card_req: dict = {}
    for card in a.get("cardinals") or []:
        card_req.setdefault(str(card.get("ref")), []).append(card["dir"])

    # One entry per ref: its label, geometry and cardinals.
    refs: list[dict] = []
    for el in a.get("refs") or []:
        as_label = None
        ref_el = el
        if isinstance(el, dict):
            if el.get("as"):
                as_label = sanitize_token(el["as"])
            ref_el = el["ref"]
        if (
            isinstance(el, dict)
            and isinstance(ref_el, (list, tuple))
            and ref_el
            and not is_xy(ref_el)
            and all(isinstance(x, str) or is_xy(x) for x in ref_el)
        ):
            kind, payload = "union", _union_members(cx, ref_el, as_label)
            lab = _unique_label(as_label or "ref", used_labels)
            geom, center = _union_center(payload)
        elif is_xy(ref_el):
            kind, payload = "point", [float(ref_el[0]), float(ref_el[1])]
            lab = _unique_label(as_label or "xy", used_labels)
            geom, center = G.mask_geom([[payload]]), payload
        else:
            if not isinstance(ref_el, str):
                raise ProgramError(
                    f"{_quote(ref_el)} is not a place: a place is a fid, a set, a row or an [x, y]."
                )
            payload = cx.by_fid.get(ref_el)
            if payload is None and ref_el in cx.staged:
                # A staged viewpoint attaches its side columns.
                if ref_el not in standing:
                    standing.append(ref_el)
                continue
            if payload is None:
                raise ProgramError(f"{_quote(ref_el)} is not a fid on this surface. " + _SAY_A_FID)
            kind = "row"
            lab = _unique_label(
                as_label or sanitize_token(payload["fid"]), used_labels, payload["fid"]
            )
            geom, center = geom_of(payload)
        fid = payload.get("fid") if kind == "row" else None
        want = next((card_req[k] for k in (fid, lab) if k and k in card_req), None)
        pts = G.points_from_geoms(geom if isinstance(geom, list) else [geom]) if want else None
        # A window-covering ref's gap columns would be 0 on every row: one
        # note says so instead.
        covers = bool(kind == "row" and payload.get("covers_window"))
        if covers:
            notes.append(
                f"ref {lab} ('{payload.get('name') or payload['fid']}'): its "
                f"mask contains the whole surface window, so 'distance to "
                f"{lab}'=0 for EVERY row (containment, not proximity) and "
                f"'facing {lab}' has no direction -- those two columns are "
                f"not stamped; 'center distance to {lab}'/'bearing from "
                f"{lab}' (centroid vector) ride per-row."
            )
        refs.append(
            {
                "kind": kind,
                "payload": payload,
                "lab": lab,
                "geom": geom,
                "center": center,
                "covers": covers,
                "cards": (want, pts) if pts is not None else None,
                "bb": G.stack_bboxes(geom) if kind == "union" else None,
                "contains": [],
            }
        )

    out_rows: list = []
    for i, it in enumerate(items):
        row = {k: v for k, v in it.items() if k not in _MODEL_HIDDEN_ROW_FIELDS}
        # Items resolve like refs, so a ref compared to itself is exactly 0.
        item_geom, it_c = place_geom(cx.by_fid.get(it.get("fid")), it)
        for ref in refs:
            kind, payload, lab = ref["kind"], ref["payload"], ref["lab"]
            if ref["cards"] is not None and it_c is not None:
                sides, rpts = ref["cards"]
                for side in sides:
                    off = G.cardinal_offset(rpts, it_c, side)
                    if off is not None:
                        row[f"{side} of {lab}"] = round(off, 1)
            ref_geom, center = ref["geom"], ref["center"]
            if ref["covers"]:
                d = None
            elif kind == "union":
                d, wr, wi = G.set_gap(ref_geom, ref["bb"], item_geom)
            else:
                d, wr, wi = G.gap_witness(ref_geom, item_geom)
            if d is not None:
                dr = round(d, 2)
                row[f"distance to {lab}"] = dr
                if dr > 0.0 and wr is not None:
                    # The witness pair runs ref->item; a row faces item->ref.
                    row[f"facing {lab}"] = G.cardinal(wi, wr)
                elif (
                    dr == 0.0
                    and not isinstance(ref_geom, list)
                    and item_geom["closed"]
                    and len(item_geom["A"])
                    and len(ref_geom["V"])
                    and not (
                        kind == "row"
                        and payload.get("fid")
                        and str(payload.get("fid")) == str(row.get("fid"))
                    )
                    and bool(G.mask_contains(item_geom, ref_geom["V"]).all())
                ):
                    # every ref vertex inside the row's own mask
                    row[f"contains {lab}"] = True
                    ref["contains"].append(str(row.get("fid") or f"item[{i}]"))
            if center is not None and it_c is not None:
                dcr = round(G.dist(center, it_c), 2)
                row[f"center distance to {lab}"] = dcr
                if dcr > 0.0:
                    row[f"bearing from {lab}"] = round(G.bearing(center, it_c), 1)
        out_rows.append(row)
    # Rows whose own mask contains a ref, per ref, named in one note; it
    # states a fact of each row it names, so it outlives a later cut or sort.
    for ref in refs:
        contains, lab = ref["contains"], ref["lab"]
        if contains:
            named = f"{contains[:8]}" + (
                f" (+{len(contains) - 8} more)" if len(contains) > 8 else ""
            )
            notes.append(
                f"ref {lab}: row(s) {named} read 'distance to {lab}'=0 because "
                f"their OWN mask CONTAINS {lab} (every vertex of the ref sits "
                f"inside the row's footprint -- containment, not adjacency); "
                f"each such row carries 'contains {lab}'=true. distance reads "
                f"0 on ANY mask contact, so a 0 in a nearest-first read spans "
                f"touching neighbors AND containers."
            )
    for h in standing:
        out_rows = _attach_standing_columns(out_rows, cx.staged[h], with_distance=True)
        notes.append(
            f"ref {h} is your staged viewpoint: its side columns ('left "
            f"of/right of/ahead of/behind {h}') + 'distance to {h}' "
            f"(meters from where you stand) ride every row."
        )
    return Result(out_rows, "compare", ordered=_ordered(items), notes=notes)
