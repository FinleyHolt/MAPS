"""The verbs that fly: ``go_to``'s one flight and ``explore``'s pass, under
the flight and wall-clock backstops."""

from __future__ import annotations

import logging
import time

from maps.library.op_compiler import HANDLE_RE, STOP_RE, row_token
from maps.mission.acks import Rejections, add_note, summarise_exc
from maps.mission.explore_stops import cover_items
from maps.mission.picture_requests import LOOK_MAX_CROPS
from maps.mission.picture_verbs import Pictures
from maps.mission.spec import MissionSpec
from maps.mission.surface_state import Progress, Surface
from maps.render.look_render import look_xy
from maps.xy_token import format_xy_token, parse_xy_token


log = logging.getLogger(__name__)


def _served_flight(report: dict) -> dict:
    """A flight's report as the model reads it: never where the aircraft
    was (``pose_xy`` and ``alt_m`` stay for the engine)."""
    return {k: v for k, v in report.items() if k not in ("pose_xy", "alt_m")}


# How many of a pass's new rows ride back as crops (its catch): half the
# image budget, so the model's own looks keep room.
_CATCH_CROPS = LOOK_MAX_CROPS // 2


class Flights:
    """The verbs that move the aircraft: one flight (``go_to``) or one pass of
    flights (``explore``), the surface re-derived after each and the rows
    it added minted as a handle."""

    def __init__(self, spec: MissionSpec, surface: Surface, rejections: Rejections,
                 pictures: Pictures, progress: Progress, *, obs_span_m: float | None):
        self.spec = spec
        self.surface = surface
        self.rejections = rejections
        self.pictures = pictures
        self.progress = progress
        # The fids explore passes have covered this mission.
        self._covered_fids: set = set()
        # A pass's stop spacing: half the footprint's long side, under the
        # short side for any sensor below 2:1, so stops overlap on both axes.
        self._cover_spacing = (obs_span_m / 2.0
                               if obs_span_m and obs_span_m > 0 else None)

    def _mint_rows(self, fids_before: set, what: str) -> tuple[str | None, str | None]:
        """The rows a flight or a pass added (the surface's rows not in
        ``fids_before``) -> their new rN handle and the note naming it;
        ``(None, None)`` when nothing was added."""
        new_rows = [r for f, r in self.surface.rows.items() if f not in fids_before]
        if not new_rows:
            return None, None
        h = self.surface.carry.mint(new_rows)
        return h, (f"{h} — the {len(new_rows)} rows this {what} added; "
                   f"the handle is bound, so {h}, {h}[0] and "
                   f"{h}.order(...) all reach them")

    def _pick_next(self, queue: list, cover: str | None) -> int:
        """The queue index to fly next: in a pass, a stop the recording
        already holds flies first; otherwise the head."""
        if cover is None or len(queue) < 2 or self.spec.held_now is None:
            return 0
        try:
            held = self.spec.held_now(
                [(str(it.get("target_fid") or ""), self.surface.point(it.get("target_fid")))
                 for it in queue])
        except Exception as exc:  # noqa: BLE001 - the order is a convenience, never a fault
            log.warning("%s pass: held_now failed: %s", self.progress.rid_tag, summarise_exc(exc))
            return 0
        pick = next((i for i, h in enumerate(held) if h), 0)
        if pick:
            log.info("%s pass: %s taken first, the recording holds it",
                     self.progress.rid_tag, queue[pick].get("target_fid"))
        return pick

    def _snap_to_row(self, item: dict) -> str | None:
        """A point target within 1 m of a row flies as that row's fid; the
        point it was asked as, or None when nothing snapped."""
        pt = parse_xy_token(item["target_fid"])
        if pt is None:
            return None
        best, bd = None, 1.0
        for bf, brow in self.surface.rows.items():
            # look_xy rounds by at most 0.005 m: a raw gap of 1.01 m or more
            # on either axis cannot snap
            xy = brow.get("xy")
            if (not isinstance(xy, (list, tuple)) or len(xy) != 2
                    or abs(float(xy[0]) - pt[0]) >= 1.01
                    or abs(float(xy[1]) - pt[1]) >= 1.01):
                continue
            bxy = look_xy(brow)
            if bxy is None:
                continue
            d = ((bxy[0] - pt[0]) ** 2 + (bxy[1] - pt[1]) ** 2) ** 0.5
            if d < bd:
                best, bd = bf, d
        if best is None:
            return None
        snap_from = item["target_fid"]
        item["target_fid"] = best
        return snap_from

    def _after_flight(self, report: dict, fids_before: set,
                      cover: str | None) -> None:
        """A flight that flew: count it, record its stop, re-derive the
        surface, and (outside a pass) mint the rows it added."""
        self.progress.turn_flights += 1
        self.progress.flights += 1
        pxy = report["pose_xy"]
        self.surface.stops[str(report["stop_id"]).strip().lower()] = [
            float(pxy[0]), float(pxy[1])]
        self.surface.world_moved()
        # The flight's new rows (the surface diff) mint a handle; a pass
        # mints once, around the whole of itself.
        if cover is None:
            _handle, note = self._mint_rows(fids_before, "flight")
            if note is not None:
                report["new_rows"] = note
        self.surface.preflight = False

    def _fly_moves(self, accepted: list, item_errors: list,
                   cover: str | None = None) -> list[dict]:
        """Fly each entry in order and return the flight reports (a stop the
        backstop cut lands in ``item_errors``); after each flight the surface
        re-derives and the rows it added mint an rN handle. ``cover`` (an
        explore) makes the loop ONE pass: no per-stop view, no per-stop
        handle."""
        reports: list[dict] = []
        clock_cut = False
        queue = list(accepted)
        while queue:
            item = queue.pop(self._pick_next(queue, cover))
            if cover is not None:
                # The rest of the pass rides every stop, for the crew.
                item["pass_ahead"] = [
                    [float(xy[0]), float(xy[1])] for xy in
                    (self.surface.point(it.get("target_fid")) for it in queue) if xy]
            cut = self.progress.turn_flights >= self.spec.flight_budget
            if cut:
                log.warning("%s flight backstop hit (%d flights)",
                            self.progress.rid_tag, self.spec.flight_budget)
            elif (self.spec.deadline_epoch is not None
                    and time.time() >= self.spec.deadline_epoch):
                # The wall-clock backstop stops the flying only; the model
                # keeps its turns and its own commit.
                # ponytail: checked between stops, so one slow stop still
                # overruns by its own duration.
                if not clock_cut:
                    log.warning(
                        "%s wall-clock backstop hit (%d flights, %.0f s past "
                        "the deadline)", self.progress.rid_tag, self.progress.turn_flights,
                        time.time() - self.spec.deadline_epoch)
                cut = clock_cut = True
            if cut:
                item_errors.append({"target_fid": item["target_fid"],
                                     "error": "not flown."})
                continue
            snap_from = self._snap_to_row(item)
            fids_before = set(self.surface.rows)
            report = self.spec.fly(dict(item))
            if snap_from:
                add_note(report, f"{snap_from} is {item['target_fid']}"
                                  "'s own position -- flown as that fid.")
            if "error" not in report:
                # A pass renders no per-stop view.
                if cover is None:
                    self.pictures.flight_view(item, report)
                self._after_flight(report, fids_before, cover)
            reports.append(report)
        return reports

    def _fly_token(self, tok: str) -> tuple[str, str | None]:
        """A typed move token -> the token a flight takes: a flown stop id
        its recorded point, a terminus fid its row's point; an unflown stop
        is refused. ``(token, error)``."""
        sp = self.surface.stop_xy(tok)
        if sp is not None:
            return format_xy_token(sp[0], sp[1]), None
        row = self.surface.terminus(tok)
        if row is not None:
            return row_token(row), None
        if STOP_RE.fullmatch(tok.strip().lower()):
            known = ", ".join(sorted(self.surface.stops)) or "none yet"
            return tok, (f"{tok} has not been flown; the flown stops are: "
                         f"{known}")
        return tok, None

    def go_to(self, tok: str) -> dict:
        """go_to(): one flight to a place."""
        tok, stop_err = self._fly_token(tok)
        if stop_err:
            return self.rejections.error({}, stop_err)
        rejected: list = []
        flown = [_served_flight(r) for r in
                 self._fly_moves([{"target_fid": tok.strip()}], rejected)]
        mv: dict = {"flown": flown}
        if rejected:
            mv["rejected"] = rejected
        if all("error" in r for r in flown):
            # nothing flew: the failure is the result
            msg = next((str(r["error"]) for r in flown + rejected
                        if r.get("error")), "nothing flew")
            return self.rejections.error({"flown": mv}, msg)
        return {"flown": mv}

    def explore(self, members: list, said_to: str) -> dict:
        """explore(): one pass of flights over a place or a set, whose
        members arrive as a name's top tie group (a list), a handle's rows
        or a token."""
        raw_members: list = []
        for t in members:
            if isinstance(t, list):
                raw_members.extend(t)
                continue
            t, stop_err = self._fly_token(t)
            if stop_err:
                return self.rejections.error({}, stop_err)
            if HANDLE_RE.fullmatch(t):
                raw_members.extend(row_token(r) for r in (
                    self.surface.carry.read(t) or []) if r.get("fid"))
                continue
            raw_members.append(t)
        accepted = [{"target_fid": t.strip()}
                     for t in dict.fromkeys(raw_members)]
        if not accepted:
            return self.rejections.error({}, "nothing to cover")
        # The pass's members count as covered once it flies.
        covered = set(self._covered_fids)
        accepted, cov_note, subject, segs = cover_items(
            accepted, said_to, spec=self.spec,
            spacing=self._cover_spacing, rows=self.surface.rows,
            covered=covered, pose_xy=self.surface.ctx.pose_xy)
        if not accepted:
            return self.rejections.error({}, cov_note)
        # One ack for the whole pass; the standing answer is its last stop.
        before, rejected = set(self.surface.rows), []
        reports = self._fly_moves(accepted, rejected, cover=subject)
        ok = [r for r in reports if "error" not in r]
        errs = [str(r["error"]) for r in reports if r.get("error")]
        if not ok:
            err = errs[0] if errs else f"nothing of {subject} could be flown"
            return self.rejections.error({"explored": {"error": err}}, err)
        self._covered_fids = covered
        mv: dict = {
            "explored": subject, "stops": len(ok),
            "answer_if_you_reply_now": (
                f"the last stop of the pass over {subject}; go_to() the row "
                f"you mean before you reply"),
            "new_detections": sum(int(r.get("new_detections") or 0)
                                  for r in ok)}
        h, note = self._mint_rows(before, "pass")
        if note is not None:
            mv["new_rows"] = note
        if errs:
            mv["not_flown"] = errs[:4]
        if rejected:
            mv["rejected"] = rejected
        if segs > 1:
            mv["segments"] = segs
        # A truncated pass reads as truncated.
        if mv["stops"] < len(accepted):
            cov_note = (f"the pass ended after "
                         f"{mv['stops']} of its stops.")
        if cov_note:
            add_note(mv, cov_note, before=True)
        # THE PASS SHOWS ITS CATCH: one crop per row it added, pass order.
        minted = self.surface.carry.rows.get(h) or [] if h else []
        rows = [r for r in minted if r.get("fid")]
        catch = rows[:_CATCH_CROPS]
        if catch:
            cack = self.pictures.crops_of(
                [str(r["fid"]) for r in catch],
                preamble=("The pass's catch, one crop per "
                          "row it added, in pass order: "))
            if not cack.get("error"):
                mv["catch"] = {k: cack[k] for k in ("delivered", "not_drawn")
                               if k in cack}
                if len(catch) < len(minted):
                    mv["catch"]["shown"] = (
                        f"first {len(catch)} of "
                        f"{len(minted)}"
                        f" -- look({h}[a:b]) pages the rest")
        return {"explored": mv}
