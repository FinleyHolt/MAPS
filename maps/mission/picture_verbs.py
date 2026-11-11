"""The verbs that show the model a picture: ``look``, ``show_map`` and the
view a flight shows, each recorded by sha before it is served."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from maps.library.op_compiler import CodeError, row_token
from maps.mission.acks import _add_note, _summarise_exc
from maps.mission.picture_requests import (
    crop_requests, look_vision_text, map_spec, map_vision_text)
from maps.render.look_render import (
    LOOK_MAX_BADGES as _MAP_MAX_MARKS, look_label, look_outline, look_xy)
from maps.withhold import withheld


log = logging.getLogger(__name__)


def _deliver_frames(jpegs: list[bytes], sink: list[str]) -> None:
    """Record every image about to reach the model: its sha256 (the wire
    log's ``image_ref.sha256``) into ``sink`` and the bytes into
    ``<workdir>/frames/<sha256>.jpg`` when a mission is recording. Never
    raises: a full disk must not end the turn."""
    workdir = os.environ.get("SEARCH_COMPILE_WORKDIR", "").strip()
    for jpeg in jpegs:
        if not jpeg:
            continue
        sha = hashlib.sha256(jpeg).hexdigest()
        sink.append(sha)
        if not workdir:
            continue
        try:
            d = Path(workdir) / "frames"
            d.mkdir(parents=True, exist_ok=True)
            dst = d / f"{sha}.jpg"
            if dst.exists():
                continue
            tmp = d / f".{sha}.jpg.tmp"
            tmp.write_bytes(jpeg)
            os.replace(tmp, dst)
        except OSError:
            log.warning("frame %s not written under %s", sha[:12], workdir,
                        exc_info=True)


# What a look and a pass's catch say under the ``images`` withhold.
_PICTURE_WITHHELD_OF = ("the {}'s caption follows in the next message; "
                        "its picture is not served on this run")


class _PictureVerbs:
    """The verbs that show a picture: a look's crops, a map, and the view a
    flight shows, queued for the message after the turn's tool results."""

    def _flight_view(self, item: dict, report: dict) -> None:
        """THE FLIGHT SHOWS WHAT IT SAW: the target's own look, centred on
        the row it asked for (a stop flown again centres on the stop's
        point), queued for the next message and named in the report."""
        tfid = str(item.get("target_fid") or "")
        trow = self._ns_rows.get(tfid)
        txy = look_xy(trow)
        try:
            fv = self.look_renderer([{
                "center_fid": tfid if txy else "",
                "center_xy": (list(txy) if txy
                              else list(report["pose_xy"])),
                "center_label": (
                    look_label(tfid, trow, is_target=True)
                    if txy else (tfid or "the stop")),
                "center_boundary": look_outline(trow) if txy else None,
                "marks": [],
            }])
        except Exception as exc:  # noqa: BLE001 - never crash a flight
            log.warning("%s flight view render failed: %s",
                        self.rid_tag, _summarise_exc(exc))
            fv = []
        fj = [c["jpeg"] for c in fv if c.get("jpeg")]
        if not fj:
            return
        fv_text = look_vision_text(fv)
        shown = not withheld("images")
        fv_note = ("the view follows." if shown else
                    _PICTURE_WITHHELD_OF.format("view"))
        if shown:
            _deliver_frames(fj, self._block_frames)
        self.pending_vision.append((fv_text, fj if shown else []))
        _add_note(report, fv_note)

    def _ship(self, ack: dict, kind: str, jpegs: list, vis_text: str) -> bool:
        """Queue one look's or map's pictures (or, under the images
        withhold, its caption alone) for the next message; ``ack`` states
        what follows. True when pictures shipped."""
        ack["delivered"] = len(jpegs)
        if jpegs and withheld("images"):
            ack["delivered"] = 0
            ack["note"] = _PICTURE_WITHHELD_OF.format(kind)
            self.pending_vision.append((vis_text, []))
            return False
        if jpegs:
            ack["note"] = f"the {kind} follows in the next message"
            _deliver_frames(jpegs, self._block_frames)
            self.pending_vision.append((vis_text, jpegs))
            return True
        ack["error"] = f"render returned no image bytes; no {kind} will follow"
        return False

    def _crops_of(self, fid, marks: list | None = None,
                   preamble: str = "") -> dict:
        # A list asks for one crop per row.
        fids = [str(f) for f in fid] if isinstance(fid, (list, tuple)) \
            else [str(fid)]
        reqs, skipped, ref_notes = crop_requests(
            fids, marks, rows=self._ns_rows, resolve=self._point,
            carry=self.carry, stops=self._plan_stops, obs_minor_m=self.obs_minor_m,
            obs_span_m=self.obs_span_m, frame=self._standing_frame())
        ack: dict = {"looked": len(reqs)}
        if skipped:
            ack["skipped"] = list(dict.fromkeys(skipped))[:20]
        if ref_notes:
            ack["ref_notes"] = ref_notes[:20]
        if not reqs:
            ack["error"] = ("; ".join(ref_notes) if ref_notes
                            else f"{', '.join(fids)} is not on the live "
                                 "surface")
            return ack
        try:
            crops = self.look_renderer(reqs)
        except Exception as exc:  # noqa: BLE001 - render never crashes the mission
            log.warning("%s SAY crops render failed: %s", self.rid_tag,
                        _summarise_exc(exc))
            return {"error": f"crop render failed: {_summarise_exc(exc)}"}
        jpegs = [c["jpeg"] for c in crops if c.get("jpeg")]
        vis_text = preamble + look_vision_text(crops)
        if self._ship(ack, "crop", jpegs, vis_text) and marks:
            # Badges drawn (the legend IS the drawn set) and the renderer's
            # own out-of-frame list, on a crop that shipped.
            ack["badged"] = sum(
                len(c.get("legend") or []) for c in crops if c.get("jpeg"))
            ack["not_drawn"] = [
                str(f) for c in crops if c.get("jpeg")
                for f in (c.get("marks_out_of_view") or [])][:20]
        return ack

    def _crops_set(self, handle: str, marks: list | None = None) -> dict:
        """look(rows): one crop per row of the handle, in its own order, each
        badged with the row it shows; the crop budget is stated when it
        bites."""
        rows = [r for r in (self.carry.read(handle) or []) if r.get("fid")]
        if not rows:
            mp = self._mission_py
            name = mp.ref_name(handle) or mp.call_arg("look")
            return {"error": (f"{name or handle} holds no rows to look at; "
                              "name a handle that has some"
                              + ("" if withheld("library") else ", or find() one first"))}
        fids = [row_token(r) for r in rows]
        return self._crops_of(
            fids, marks=marks,
            preamble=("One crop per row, in order, each badged with the "
                      "row it shows. "))

    def _crops(self, of, fid, marks) -> dict:
        mk = [m for m in marks or [] if isinstance(m, str) and m.strip()]
        crop = (self._crops_set(str(of), marks=mk or None) if of
                else self._crops_of(fid, marks=mk or None))
        return self._acked("crop", crop)

    def _map(self, around: str) -> dict:
        """show_map(): one map around a place, badging the set in scope."""
        h = self._those
        # The set in scope only badges the map: a refused re-derive is
        # noted, never the map's own refusal.
        try:
            mrows = [r for r in (self.carry.read(h) or [])
                     if r.get("fid")] if h else []
        except CodeError as exc:
            mrows, scope_err = [], f"{exc}; this map badges none of it"
        else:
            scope_err = None
        marks = mrows[:_MAP_MAX_MARKS]
        spec, m_skipped = map_spec(around, marks, rows=self._ns_rows,
                                   resolve=self._point,
                                   frame=self._standing_frame())
        mp: dict = {"maps": int(spec is not None)}
        if m_skipped:
            mp["skipped"] = m_skipped[:20]
        if spec is None:
            mp["error"] = f"{around} is not on the live surface"
        else:
            try:
                from maps.render.map_render import (  # noqa: PLC0415
                    render_map_from_spec,
                )
                rendered = render_map_from_spec(
                    spec, self._map_features_cached(),
                    tile_origin_xy=self.ctx.tile_origin_xy,
                    cell_size_m=float(self.ctx.cell_size_m),
                    grid_shape=tuple(self.ctx.grid_shape))
                maps = [rendered] if rendered is not None else []
            except Exception as exc:  # noqa: BLE001 - render never crashes the mission
                log.warning("%s SAY map render failed: %s", self.rid_tag,
                            _summarise_exc(exc))
                mp = {"error": f"map render failed: {_summarise_exc(exc)}"}
            else:
                self._ship(mp, "map", [m["jpeg"] for m in maps if m.get("jpeg")],
                           map_vision_text(maps))
        if scope_err:
            mp.setdefault("ref_notes", []).append(scope_err)
        if marks:
            # The set under the model's own name for it.
            name = self._mission_py.ref_name(h)
            mp["badged_from"] = name or "the set in scope"
            if len(mrows) > len(marks):
                mp.setdefault("ref_notes", []).append(
                    f"{len(mrows)} rows in {name or 'the set in scope'}; "
                    f"this map badges the first {len(marks)}")
        return self._acked("map", mp)
