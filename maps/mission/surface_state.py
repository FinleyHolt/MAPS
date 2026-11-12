"""The surface one mission reads: the ``{cands, anchors, cell_m}`` rows of
its world-state graph, the result handles over them, the place a token
names, and the viewpoint ``stand`` adopts."""

from __future__ import annotations

from dataclasses import dataclass

from maps.library.op_compiler import END_RE
from maps.library.result_handles import Carry
from maps.mission.acks import Rejections
from maps.mission.picture_requests import map_features, rounded_xy
from maps.query.result_tables import MODEL_UNSERVED_COLUMNS
from maps.query.spatial_math import COMPASS_UNIT
from maps.query.surface import build_surface_from_graph
from maps.render.look_render import look_xy, up_bearing_deg


@dataclass
class Progress:
    """Where the conversation stands, shared by the loop and its verbs."""

    #: The log tag of the operator turn (``m0``, ``m1``, ...).
    rid_tag: str = "h?"
    #: The reasoner turn within the operator turn.
    turn_idx: int = 0
    #: The mission's flights so far, over every operator turn.
    flights: int = 0
    #: The operator turn's flights: the backstop's count.
    turn_flights: int = 0


def surface_from_ctx(ctx) -> dict:
    """The ``{cands, anchors, cell_m}`` surface of a :class:`QueryContext`'s
    graph."""
    cell_m = float(ctx.cell_size_m)
    cands, anchors = build_surface_from_graph(ctx.graph.nodes.items(), ctx.tile_origin_xy, cell_m)
    return {"cands": cands, "anchors": anchors, "cell_m": cell_m}


class Surface:
    """The surface a mission reads: its rows by fid, re-derived when the world
    steps, the result handles over them, the place a token names and the
    standing viewpoint."""

    def __init__(self, ctx, namespace: dict, name_of_ref, *, preflight: bool = False):
        self.ctx = ctx
        #: ``{cands, anchors, cell_m}``, plus ``staged`` and ``standing``
        #: once a stance is adopted: what the op program runs over.
        self.ns = namespace
        #: Surface rows by fid, rebuilt in place after every flight and boost.
        self.rows: dict[str, dict] = {}
        self._rebuild_rows()
        #: Each fid's latest result-enriched row, for the rejection census.
        self.row_latest: dict[str, dict] = {}
        #: The surface version, bumped by a flight, a boost and a stance:
        #: what a program's result, the dedup and the map features key on.
        self.version = 0
        #: The result handles, re-derived once per surface version.
        self.carry = Carry(namespace, name=name_of_ref, surface_version=lambda: self.version)
        #: True until the first flight: before it no row carries a seen colour.
        self.preflight = preflight
        #: The mission's flown stops, ``{stop_id: [x, y]}``: each a sayable place.
        self.stops: dict[str, list] = {}
        #: The staged viewpoints by vN handle; the latest is STANDING (its side
        #: columns ride every find row, its facing orients what renders).
        self.viewpoints: dict[str, dict] = {}
        self.standing: str | None = None
        # The map's flattened features, built once per surface version.
        self._map_feats_version = None
        self._map_feats: list[dict] = []

    def world_moved(self) -> None:
        """The world stepped: refresh both surface views and bump the
        version."""
        self.ns.update(surface_from_ctx(self.ctx))
        self._rebuild_rows()
        self.version += 1

    def _rebuild_rows(self) -> None:
        # cands win on fid collision.
        self.rows.clear()
        for key in ("cands", "anchors"):
            for r in self.ns.get(key) or []:
                self.rows.setdefault(str(r["fid"]), r)

    def census(self, cap: int) -> tuple[int, int, list[str], str]:
        """Live counts: detections, priors, the first ``cap`` computed
        columns and the " (+N more)" past them."""
        c_n = len(self.ns.get("cands") or [])
        a_n = len(self.ns.get("anchors") or [])
        cols = sorted(
            {
                k
                for row in self.row_latest.values()
                for k, v in row.items()
                if k not in MODEL_UNSERVED_COLUMNS and v not in (None, "")
            }
        )
        more = f" (+{len(cols) - cap} more)" if len(cols) > cap else ""
        return c_n, a_n, cols[:cap], more

    def stop_xy(self, tok: str):
        """A flown stop id's recorded point, or None."""
        return self.stops.get(tok.strip().lower())

    def point(self, tok) -> list[float] | None:
        """THE place a token names, as [x, y] at 2 dp: a live surface fid, a
        flown stop id, a staged handle, or the canonical '[x, y]' token; None
        otherwise."""
        if isinstance(tok, str):
            row = self.rows.get(tok.strip())
            if row is not None:
                return look_xy(row)
            sp = self.stop_xy(tok)
            if sp is not None:
                return rounded_xy(sp)
            end = self.terminus(tok)
            if end is not None:
                return look_xy(end)
            sv = self.viewpoints.get(tok.strip().lower())
            if sv is not None:
                return rounded_xy(sv["from_xy"])
            return rounded_xy(tok)
        return None

    def adopt_viewpoint(self, at_xy: list, up: list) -> dict:
        """Stage a viewpoint at ``at_xy`` whose image-up is the unit vector
        ``up``: it becomes STANDING and the surface version bumps (a program
        can answer differently under it)."""
        h = f"v{len(self.viewpoints) + 1}"
        self.viewpoints[h] = {
            "handle": h,
            "from_xy": at_xy,
            "up_unit": [round(up[0], 4), round(up[1], 4)],
            "bearing_deg": round(up_bearing_deg(up)) % 360,
        }
        self.standing = h
        self.ns["staged"] = self.viewpoints
        self.ns["standing"] = h
        self.version += 1
        return self.viewpoints[h]

    def standing_frame(self) -> tuple[list | None, str]:
        """The STANDING viewpoint's image-up and the caption note stating it
        (never a silent rotation); ``(None, "")`` with no stance."""
        if self.standing is None:
            return None, ""
        sv = self.viewpoints[self.standing]
        return (
            list(sv["up_unit"]),
            f"image-up = {sv['handle']}'s facing (bearing {sv['bearing_deg']})",
        )

    def map_features(self) -> list[dict]:
        if self._map_feats_version != self.version:
            self._map_feats_version = self.version
            self._map_feats = map_features(self.ctx)
        return self._map_feats

    def terminus(self, tok: str) -> dict | None:
        """The carried row a typed terminus fid (``end_N``) names, or None."""
        if not END_RE.fullmatch(tok.strip()):
            return None
        # ponytail: end_N restarts per ends_of call, so the newest carried
        # row holding it wins; unique terminus fids if two ends_of sets must
        # be named by fid.
        return next(
            (
                r
                for rows in reversed(self.carry.rows.values())
                for r in rows or ()
                if r.get("fid") == tok.strip()
            ),
            None,
        )


class Stance:
    """The verb behind ``stand``: a viewpoint the surface adopts."""

    def __init__(self, surface: Surface, rejections: Rejections):
        self.surface = surface
        self.rejections = rejections

    def stand(self, at_tok: str, facing_tok: str) -> dict:
        """stand(): mint a vN handle that becomes STANDING."""

        def refuse(why: str) -> dict:
            return self.rejections.error({}, f"the stance did not bind: {why}")

        at_xy = self.surface.point(at_tok)
        if at_xy is None:
            return refuse(f"at {at_tok!r} is not a live surface fid or an [x, y] world point")
        compass = facing_tok.strip().lower()
        if compass in COMPASS_UNIT:
            up = list(COMPASS_UNIT[compass])
        else:
            f_xy = self.surface.point(facing_tok)
            if f_xy is None:
                return refuse(
                    f"facing {facing_tok!r} is not a live surface "
                    f"fid, an [x, y] point, or one of "
                    f"{'|'.join(COMPASS_UNIT)}"
                )
            dx = f_xy[0] - at_xy[0]
            dy = f_xy[1] - at_xy[1]
            n = (dx * dx + dy * dy) ** 0.5
            if n < 1e-6:
                return refuse("at and facing coincide -- no look direction")
            up = [dx / n, dy / n]
        staged = self.surface.adopt_viewpoint(at_xy, up)
        h = staged["handle"]
        return {
            "stance": {
                "staged": [
                    {"handle": h, "at_xy": at_xy, "facing_bearing_deg": staged["bearing_deg"]}
                ],
                "note": (
                    f"standing viewpoint = {h}: 'left of {h}'/'right of "
                    f"{h}'/'ahead of {h}'/'behind {h}' ride every find "
                    f"row; {h} is a point ref ('distance to {h}'); "
                    "crops and maps render its facing up."
                ),
            }
        }
