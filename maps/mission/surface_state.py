"""The surface one mission reads: the ``{cands, anchors, cell_m}`` rows of
its world-state graph, the place a token names, and the viewpoint
``stand`` adopts."""

from __future__ import annotations

from maps.library.op_compiler import END_RE
from maps.mission.picture_requests import map_features, rounded_xy
from maps.query.spatial_math import COMPASS_UNIT
from maps.query.surface import build_surface_from_graph
from maps.render.look_render import look_xy, up_bearing_deg


def _surface_from_ctx(ctx) -> dict:
    """The ``{cands, anchors, cell_m}`` surface of a :class:`QueryContext`'s
    graph."""
    cell_m = float(ctx.cell_size_m)
    cands, anchors = build_surface_from_graph(
        ctx.graph.nodes.items(), ctx.tile_origin_xy, cell_m
    )
    return {"cands": cands, "anchors": anchors, "cell_m": cell_m}


class _SurfaceState:
    """The surface a mission reads: its rows by fid, re-derived when the world
    steps, the place a token names and the standing viewpoint."""

    def _world_moved(self) -> None:
        """The world stepped: refresh both surface views and bump the
        version."""
        self.ns.update(_surface_from_ctx(self.ctx))
        self._rebuild_ns_rows()
        self.carry.version += 1

    def _rebuild_ns_rows(self) -> None:
        # cands win on fid collision.
        self._ns_rows.clear()
        for key in ("cands", "anchors"):
            for r in self.ns.get(key) or []:
                self._ns_rows.setdefault(str(r["fid"]), r)

    def _stop_xy(self, tok: str):
        """A flown stop id's recorded point, or None."""
        return self._plan_stops.get(tok.strip().lower())

    def _point(self, tok) -> list[float] | None:
        """THE place a token names, as [x, y] at 2 dp: a live surface fid, a
        flown stop id, a staged handle, or the canonical '[x, y]' token; None
        otherwise."""
        if isinstance(tok, str):
            row = self._ns_rows.get(tok.strip())
            if row is not None:
                return look_xy(row)
            sp = self._stop_xy(tok)
            if sp is not None:
                return rounded_xy(sp)
            end = self._terminus(tok)
            if end is not None:
                return look_xy(end)
            sv = self.staged_viewpoints.get(tok.strip().lower())
            if sv is not None:
                return rounded_xy(sv["from_xy"])
            return rounded_xy(tok)
        return None

    def _stand(self, at_tok: str, facing_tok: str) -> dict:
        """stand(): mint a vN handle that becomes STANDING and bump the
        surface version (a program can answer differently under it)."""
        def refuse(why: str) -> dict:
            return self._plan_err({}, f"the stance did not bind: {why}")

        at_xy = self._point(at_tok)
        if at_xy is None:
            return refuse(f"at {at_tok!r} is not a live surface fid or an "
                          f"[x, y] world point")
        compass = facing_tok.strip().lower()
        if compass in COMPASS_UNIT:
            up = list(COMPASS_UNIT[compass])
        else:
            f_xy = self._point(facing_tok)
            if f_xy is None:
                return refuse(f"facing {facing_tok!r} is not a live surface "
                              f"fid, an [x, y] point, or one of "
                              f"{'|'.join(COMPASS_UNIT)}")
            dx = f_xy[0] - at_xy[0]
            dy = f_xy[1] - at_xy[1]
            n = (dx * dx + dy * dy) ** 0.5
            if n < 1e-6:
                return refuse("at and facing coincide -- no look direction")
            up = [dx / n, dy / n]
        h = f"v{len(self.staged_viewpoints) + 1}"
        brg = round(up_bearing_deg(up)) % 360
        self.staged_viewpoints[h] = {
            "handle": h,
            "from_xy": at_xy,
            "up_unit": [round(up[0], 4), round(up[1], 4)],
            "bearing_deg": brg,
        }
        self.standing_handle = h
        self.ns["staged"] = self.staged_viewpoints
        self.ns["standing"] = h
        self.carry.version += 1
        return {"stance": {
            "staged": [{"handle": h, "at_xy": at_xy,
                        "facing_bearing_deg": brg}],
            "note": (f"standing viewpoint = {h}: 'left of {h}'/'right of "
                     f"{h}'/'ahead of {h}'/'behind {h}' ride every find "
                     f"row; {h} is a point ref ('distance to {h}'); "
                     "crops and maps render its facing up."),
        }}

    def _standing_frame(self) -> tuple[list | None, str]:
        """The STANDING viewpoint's image-up and the caption note stating it
        (never a silent rotation); ``(None, "")`` with no stance."""
        if self.standing_handle is None:
            return None, ""
        sv = self.staged_viewpoints[self.standing_handle]
        return (list(sv["up_unit"]),
                f"image-up = {sv['handle']}'s facing "
                f"(bearing {sv['bearing_deg']})")

    def _map_features_cached(self) -> list[dict]:
        if self._map_feats_version != self.carry.version:
            self._map_feats_version = self.carry.version
            self._map_feats = map_features(self.ctx)
        return self._map_feats

    def _terminus(self, tok: str) -> dict | None:
        """The carried row a typed terminus fid (``end_N``) names, or None."""
        if not END_RE.fullmatch(tok.strip()):
            return None
        # ponytail: end_N restarts per ends_of call, so the newest carried
        # row holding it wins; unique terminus fids if two ends_of sets must
        # be named by fid.
        return next((r for rows in reversed(self.carry.rows.values())
                     for r in rows or () if r.get("fid") == tok.strip()), None)
