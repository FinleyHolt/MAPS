"""The GT-perfect perception tier.

Emits the CORRECT footprint mask plus the offline ground-truth color for every
GT object whose footprint lies in the crop. It is a diagnostic UPPER BOUND
(non-headline, config-gated): it answers whether the CityNav SR ceiling is a
PERCEPTION cap (fixable by a better sensor) or genuine cue/reasoning ambiguity.

Robot-agnostic: takes ``list[GTObject]``; it does not know what a CityNav
block or a Matterport scan or a warehouse layout is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from maps.perception.base import DetectedObject, GTObject
from maps.perception.observation import Observation
from maps.util.mask_region import MaskRegion


@dataclass
class GTPerfectPerception:
    """Deterministic *perfect sensor* for the GT-PERCEPTION diagnostic.

    Hands the reasoner the CORRECT footprint mask + the offline GT color for
    EVERY ground-truth object whose footprint lies in the crop, with ``P=1`` and
    zero noise. It is a diagnostic UPPER BOUND (non-headline, config-gated): it
    answers whether the CityNav SR ceiling is a PERCEPTION cap (fixable by a
    better sensor) or genuine cue/reasoning ambiguity (not).

    It emits the object's TRUE footprint — the CityRefer ``contour``
    rasterized by the EXACT inverse of the orthographic geolocator's affine
    (``geolocate.project_mask_to_tile_cells_ortho``: ``world = FL + u·(FR−FL) +
    v·(BL−FL)`` over the view-square corners ``obs.ortho_view_corners_xy``), so
    the projected ground footprint round-trips to the GT cells. Detections flow
    through the unchanged detect()→geolocate→tracker→upsert path exactly like a
    real detection; the injected color rides in ``attributes`` to the reasoner's
    color surface.

    ANTI-CHEAT: ``detect`` never reads ``instruction_hint`` (no target leak); the
    color is the object's true appearance from the offline cue-independent table
    (never the cue); ``label`` is a class noun set by the seam from ``object_type``
    (never the CityRefer name), so no catalog label reaches the reasoner.
    """

    objects: list[GTObject]
    # Footprint rasterization resolution. The ortho geolocator normalizes the
    # mask by its OWN (h, w) and maps it onto the view square, so any square
    # resolution that spans the crop AND clears the projector's ~16 px floor is
    # correct. A ~4 m car in a ~100-200 m view square at 224 px is only a few px
    # (silently dropped); 768 px keeps even small cars well above the floor.
    mask_px: int = 768
    # Floor on the rasterized footprint pixel count so the projector never drops
    # a real object for being small (> PERCEPTION_MIN_VALID_DEPTH_PIXELS, ~16).
    min_blob_px: int = 24
    # Per-stop detection cap. 0 = uncapped (the perfect sensor emits EVERY
    # in-crop object — correctness over speed; the parity gate's count match
    # surfaces a regression if this is ever hit).
    max_per_step: int = 0

    def reset(self) -> None:
        # Stateless; present so the runner's per-episode ``reset()`` is a no-op.
        return

    def detect(
        self,
        obs: Observation,
        pose_xy: tuple[float, float],
        instruction_hint: str | None = None,
    ) -> list[DetectedObject]:
        del pose_xy, instruction_hint  # perfect sensor: no FoV falloff, no task hint
        corners = getattr(obs, "ortho_view_corners_xy", None)
        h = w = int(self.mask_px)
        detections: list[DetectedObject] = []

        # Pre-resolve the inverse view-square affine once (ortho path). M maps
        # (u, v) -> world offset from FL; we invert it to take world -> (u, v).
        inv = None
        fl = None
        if corners is not None:
            fl = (float(corners[0][0]), float(corners[0][1]))
            fr = (float(corners[1][0]), float(corners[1][1]))
            bl = (float(corners[3][0]), float(corners[3][1]))
            m00, m01 = fr[0] - fl[0], bl[0] - fl[0]
            m10, m11 = fr[1] - fl[1], bl[1] - fl[1]
            det = m00 * m11 - m01 * m10
            if abs(det) < 1e-9:  # degenerate view square — should never happen
                return []
            inv = (m11 / det, -m01 / det, -m10 / det, m00 / det)

        x_min, y_min, x_max, y_max = obs.world_window
        span_x = max(x_max - x_min, 1e-6)
        span_y = max(y_max - y_min, 1e-6)

        def world_to_uv(x: float, y: float) -> tuple[float, float]:
            """World (x, y) -> normalized crop coords (u along FR, v along BL)."""
            if inv is not None and fl is not None:
                dx, dy = x - fl[0], y - fl[1]
                return inv[0] * dx + inv[1] * dy, inv[2] * dx + inv[3] * dy
            # Axis-aligned fallback (non-ortho backends): u east, v south.
            return (x - x_min) / span_x, (y_max - y) / span_y

        for obj in self.objects:
            ox, oy = float(obj.world_xy[0]), float(obj.world_xy[1])
            cu, cv = world_to_uv(ox, oy)
            # In-crop gate: SQUARE membership (the whole view square), never the
            # synthetic circular FoV cut — a corner car must not be missed.
            if not (-1e-6 <= cu <= 1.0 + 1e-6 and -1e-6 <= cv <= 1.0 + 1e-6):
                continue

            mask = self._footprint_mask(obj, world_to_uv, h, w, cu, cv)
            if mask is None or not mask.any():
                continue
            rs, cs = np.where(mask)
            bbox = (int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max()))

            world_contour = self._tight_world_contour(obj)
            detections.append(
                DetectedObject(
                    world_xy=(ox, oy),  # exact GT centroid; zero position noise
                    label=obj.label or obj.object_type or "object",
                    confidence=1.0,
                    class_hint=obj.class_hint or "object",
                    source_tier="gt_perfect",
                    metadata={
                        "source_id": obj.id,
                        "object_type": str(obj.object_type),
                        **(
                            {"world_contour": world_contour}
                            if world_contour
                            else {}
                        ),
                    },
                    attributes=dict(obj.attributes),  # GT color table row
                    mask=MaskRegion.from_full(mask, bbox=bbox),
                )
            )
            if self.max_per_step and len(detections) >= self.max_per_step:
                break

        return detections

    # WORLD_CONTOUR stamp cap: fine footprint polygons ride the ingest payload
    # and the row ``boundary``; 16 vertices keeps a car/building outline
    # faithful at trivial byte cost.
    _CONTOUR_MAX_VERTS = 16

    def _tight_world_contour(self, obj) -> list[list[float]] | None:
        """The exact CityRefer contour, TIGHTNESS-GATED, anchored at the GT
        centroid, decimated to <= 16 vertices (2 dp world meters) and CLOSED
        — the chain ends on the vertex it started from — or None.

        TIGHTNESS GATE (measured 2026-07-04, rendered-contour audit over 3
        blocks): raw CityRefer CAR contours are LOOSE ANNOTATION HULLS, not
        car masks — shoelace_area / (dim_x * dim_y) has median 3.41x with
        only 21% within 1.6x (n=640), and the annotated dims themselves
        include absurd entries (a 21.8 x 22.2 m "car"). Buildings: median
        1.35x, 63% within 1.6x — usable. So a contour is stamped ONLY when
        ``area <= 1.6 * dim_x * dim_y`` with both dims in (0.5, 100) m; in
        practice buildings/lots pass and cars almost never do — that is
        CORRECT (a loose hull on a car would re-poison exactly the geometry
        this substrate exists to fix; the honest fine-car source is the SAM
        pixel-mask path, which is pixel-tight and needs no gate). Do NOT
        "fix" the gate to let cars through.

        The polygon is translated so its vertex mean sits on ``obj.world_xy``
        — the SAME anchor ``_footprint_mask`` uses — so the stamped contour
        and the rasterized mask/cells describe one footprint.
        """
        contour = getattr(obj, "contour", None)
        if not contour or len(contour) < 3:
            return None
        dims = getattr(obj, "dimension", None)
        if dims is None or len(dims) < 2:
            return None
        dx, dy = float(dims[0]), float(dims[1])
        if not (0.5 < dx < 100.0 and 0.5 < dy < 100.0):
            return None
        con = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
        # Strip the duplicated closing vertex so the stride cannot land on
        # it; the chain is re-closed on the way out.
        if con.shape[0] >= 2 and np.allclose(con[0], con[-1]):
            con = con[:-1]
        if con.shape[0] < 3:
            return None
        # Shoelace area of the RAW contour (translation-invariant).
        x, y = con[:, 0], con[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        if area > 1.6 * dx * dy:
            return None
        shift = np.asarray(obj.world_xy, dtype=np.float64) - con.mean(axis=0)
        con = con + shift
        if con.shape[0] > self._CONTOUR_MAX_VERTS:
            stride = int(math.ceil(con.shape[0] / self._CONTOUR_MAX_VERTS))
            con = con[::stride]
        ring = [[round(float(px), 2), round(float(py), 2)] for px, py in con]
        # An annotation footprint ENCLOSES, and the chain says so by ending
        # on the vertex it started from (``geom_compute.is_closed_chain``).
        return ring + [list(ring[0])]

    def _footprint_mask(self, obj, world_to_uv, h, w, cu, cv):
        """Rasterize ``obj.contour`` into an (h, w) bool mask in crop-pixel space.

        Each contour vertex maps world -> (u, v) -> pixel (col = u·(w−1),
        row = v·(h−1)) — the exact inverse of the ortho projector's forward
        affine, so the projected ground footprint reproduces the GT cells. Only
        the polygon's pixel bbox is tested (cheap). Falls back to / tops up with
        a centroid disc so the rasterized blob always clears ``min_blob_px``
        (else the ~16 px projector floor silently drops the object)."""
        mask = np.zeros((h, w), dtype=bool)
        contour = getattr(obj, "contour", None)
        if contour and len(contour) >= 3:
            try:
                from matplotlib.path import Path as MplPath
            except ImportError:
                MplPath = None  # type: ignore[assignment]
            if MplPath is not None:
                # Anchor the footprint at the GT centroid (obj.world_xy), not the raw
                # contour centroid: CityRefer car contours are sometimes loose/irregular
                # blobs whose geometric center drifts up to ~16 m from the canonical
                # `position`. Translating the polygon onto world_xy preserves its SHAPE
                # (for geometry ops) while keeping the perceived location == the GT
                # point the 20 m success radius is scored against, and keeps world_xy
                # consistent with the mask the projector centroids.
                con = np.asarray(contour, dtype=np.float64)
                # Strip a duplicated closing vertex BEFORE the centroid mean —
                # parity with _tight_world_contour's anchor (review 4ded6741:
                # a closed low-vertex contour would otherwise shift the
                # rasterized cells ~1-3 m off the stamped world_contour).
                if len(con) >= 4 and np.allclose(con[0], con[-1]):
                    con = con[:-1]
                shift = np.asarray(obj.world_xy, dtype=np.float64) - con.mean(axis=0)
                con = con + shift
                poly_px = np.array(
                    [
                        [u * (w - 1), v * (h - 1)]
                        for (u, v) in (world_to_uv(float(px), float(py)) for px, py in con)
                    ],
                    dtype=np.float64,
                )
                c0 = max(0, int(np.floor(poly_px[:, 0].min())))
                c1 = min(w - 1, int(np.ceil(poly_px[:, 0].max())))
                r0 = max(0, int(np.floor(poly_px[:, 1].min())))
                r1 = min(h - 1, int(np.ceil(poly_px[:, 1].max())))
                if c1 >= c0 and r1 >= r0:
                    cc, rr = np.meshgrid(
                        np.arange(c0, c1 + 1), np.arange(r0, r1 + 1)
                    )
                    pts = np.stack([cc.ravel() + 0.0, rr.ravel() + 0.0], axis=1)
                    inside = MplPath(poly_px).contains_points(pts).reshape(rr.shape)
                    mask[r0 : r1 + 1, c0 : c1 + 1] = inside

        if int(mask.sum()) < self.min_blob_px:
            # Top up with a centroid disc sized to clear the floor (covers
            # missing/degenerate contours and tiny footprints at high altitude).
            self._paint_centroid_disc(mask, h, w, cu, cv)
        return mask

    def _paint_centroid_disc(self, mask, h, w, cu, cv) -> None:
        row = int(round(cv * (h - 1)))
        col = int(round(cu * (w - 1)))
        if not (0 <= row < h and 0 <= col < w):
            return
        radius = max(2.0, math.sqrt(self.min_blob_px / math.pi) + 1.0)
        r_int = int(math.ceil(radius))
        r0, r1 = max(0, row - r_int), min(h, row + r_int + 1)
        c0, c1 = max(0, col - r_int), min(w, col + r_int + 1)
        rr, cc = np.ogrid[r0:r1, c0:c1]
        mask[r0:r1, c0:c1] |= (rr - row) ** 2 + (cc - col) ** 2 <= radius * radius
