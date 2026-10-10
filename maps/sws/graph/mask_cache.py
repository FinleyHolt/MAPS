"""Per-graph node ledger — fid → (centroid, cell_count).

A thin scalar ledger: fid → centroid (grid row/col) + cell_count, indexed by
insertion order, attached per-graph so the re-id / ingest bookkeeping
(``fid in cache``, ``len(cache)``, ``cache.centroids``, ``cache.cell_counts``)
has somewhere to read. It allocates no ``H*W`` arrays — nothing on the
SR/commit path reads dense masks, and rasterizing every prebake mask (~10k-30k
rows, never evicted) was the dominant host-RAM cost in the SR fleet.

``add`` / ``replace`` accept a ``mask`` argument and ignore it: every live
caller passes centroid and cell_count directly, and the mask is only a
fallback to derive them.
"""

from __future__ import annotations

import numpy as np

from maps.sws.graph.model import ObjectGraph

_INITIAL_CAPACITY = 64


class MaskCache:
    """Scalar per-node ledger for an ``ObjectGraph``: fid → centroid + cell_count.

    Two aligned capacity buffers indexed by insertion order:

    * ``centroids``   — (N, 2) float32, row/col in grid units.
    * ``cell_counts`` — (N,) int64.

    ``_n`` tracks the logical count; the public ``centroids`` / ``cell_counts``
    properties return views into ``[:_n]``. Capacity doubles in one block copy
    when ``_n`` hits ``_capacity``.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        *,
        cell_size_m: float,
        initial_capacity: int = _INITIAL_CAPACITY,
    ) -> None:
        self.H = int(grid_shape[0])
        self.W = int(grid_shape[1])
        self.grid_shape: tuple[int, int] = (self.H, self.W)
        self.cell_size_m: float = float(cell_size_m)
        self.fids: list[str] = []
        self._fid_to_idx: dict[str, int] = {}
        self._n = 0
        self._capacity = max(1, int(initial_capacity))
        self._centroids_buf: np.ndarray = np.empty((self._capacity, 2), dtype=np.float32)
        self._cell_counts_buf: np.ndarray = np.empty((self._capacity,), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.fids)

    def __contains__(self, fid: str) -> bool:
        return fid in self._fid_to_idx

    def copy(self) -> "MaskCache":
        """Snapshot the ledger so a sibling caller can append independently.

        Per-reasoner runtime graph clones each need their own detected-row
        namespace so one reasoner's ``cache.add(detected_0, ...)`` cannot
        collide with another's.
        """
        new = MaskCache.__new__(MaskCache)
        new.H = self.H
        new.W = self.W
        new.grid_shape = self.grid_shape
        new.cell_size_m = self.cell_size_m
        new.fids = list(self.fids)
        new._fid_to_idx = dict(self._fid_to_idx)
        new._n = self._n
        new._capacity = self._capacity
        new._centroids_buf = self._centroids_buf.copy()
        new._cell_counts_buf = self._cell_counts_buf.copy()
        return new

    # ---- public views (slices into the capacity buffers) ------------------

    @property
    def centroids(self) -> np.ndarray:
        return self._centroids_buf[: self._n]

    @centroids.setter
    def centroids(self, _: np.ndarray) -> None:
        raise AttributeError(
            "MaskCache.centroids is a read-only view; assign through cache.centroids[idx] = ..."
        )

    @property
    def cell_counts(self) -> np.ndarray:
        return self._cell_counts_buf[: self._n]

    @cell_counts.setter
    def cell_counts(self, _: np.ndarray) -> None:
        raise AttributeError(
            "MaskCache.cell_counts is a read-only view; assign through cache.cell_counts[idx] = ..."
        )

    # ---- capacity management ---------------------------------------------

    def _grow(self, needed: int) -> None:
        """Double (or more) the ledger buffers to fit ``needed`` rows."""
        new_cap = self._capacity
        while new_cap < needed:
            new_cap *= 2
        cent_new = np.empty((new_cap, 2), dtype=np.float32)
        cnt_new = np.empty((new_cap,), dtype=np.int64)
        if self._n > 0:
            cent_new[: self._n] = self._centroids_buf[: self._n]
            cnt_new[: self._n] = self._cell_counts_buf[: self._n]
        self._centroids_buf = cent_new
        self._cell_counts_buf = cnt_new
        self._capacity = new_cap

    # ---- inserts / updates ------------------------------------------------

    @staticmethod
    def _derive(mask, centroid, cell_count):
        """Resolve (centroid, cell_count), deriving from ``mask`` only when a
        caller omitted them (every live caller passes both)."""
        if centroid is not None and cell_count is not None:
            return (float(centroid[0]), float(centroid[1])), int(cell_count)
        if mask is None:
            raise ValueError("MaskCache.add/replace need centroid+cell_count or a mask")
        bool_mask = np.asarray(mask, dtype=bool)
        rs, cs = np.where(bool_mask)
        cc = int(cell_count) if cell_count is not None else int(rs.size)
        if centroid is not None:
            cen = (float(centroid[0]), float(centroid[1]))
        elif rs.size:
            cen = (float(rs.mean()), float(cs.mean()))
        else:
            cen = (0.0, 0.0)
        return cen, cc

    def add(
        self,
        fid: str,
        mask: np.ndarray | None = None,
        *,
        centroid: tuple[float, float] | None = None,
        cell_count: int | None = None,
        dilated: np.ndarray | None = None,
    ) -> None:
        """Record one node in the ledger. Idempotent on duplicate fid. The
        ``mask``/``dilated`` arguments are accepted for call-site
        compatibility and only used to derive centroid/cell_count when those
        are not supplied — nothing is stored from them."""
        if fid in self._fid_to_idx:
            return
        self._insert_row(fid, mask, centroid=centroid, cell_count=cell_count)

    def replace(
        self,
        fid: str,
        mask: np.ndarray | None = None,
        *,
        centroid: tuple[float, float] | None = None,
        cell_count: int | None = None,
        dilated: np.ndarray | None = None,
    ) -> None:
        """Update an existing node's centroid/cell_count in place (inserting if
        new). The row index stays stable."""
        idx = self._fid_to_idx.get(fid)
        if idx is None:
            self._insert_row(fid, mask, centroid=centroid, cell_count=cell_count)
            return
        cen, cc = self._derive(mask, centroid, cell_count)
        self._centroids_buf[idx, 0] = cen[0]
        self._centroids_buf[idx, 1] = cen[1]
        self._cell_counts_buf[idx] = np.int64(cc)

    def _insert_row(
        self,
        fid: str,
        mask: np.ndarray | None,
        *,
        centroid: tuple[float, float] | None,
        cell_count: int | None,
    ) -> None:
        cen, cc = self._derive(mask, centroid, cell_count)
        if self._n >= self._capacity:
            self._grow(self._n + 1)
        idx = self._n
        self._fid_to_idx[fid] = idx
        self.fids.append(fid)
        self._centroids_buf[idx, 0] = cen[0]
        self._centroids_buf[idx, 1] = cen[1]
        self._cell_counts_buf[idx] = np.int64(cc)
        self._n += 1


def infer_grid_shape(graph: ObjectGraph) -> tuple[int, int]:
    """Best-effort grid shape from the graph's node bboxes.

    Used only when a caller hasn't explicitly passed ``grid_shape``. Takes the
    maximum row/col across all node bboxes and adds one-cell padding.
    """
    if not graph.nodes:
        return (1, 1)
    max_r = 0
    max_c = 0
    for node in graph.nodes.values():
        _, _, r1, c1 = node.bbox
        if int(r1) > max_r:
            max_r = int(r1)
        if int(c1) > max_c:
            max_c = int(c1)
    return (max_r + 2, max_c + 2)


def build_mask_cache(
    graph: ObjectGraph,
    grid_shape: tuple[int, int],
    *,
    features: dict | None = None,
) -> MaskCache:
    """Populate a ``MaskCache`` ledger from an ``ObjectGraph``'s nodes.

    Records only ``centroid`` + ``cell_count`` per
    node (both already on the ``MapObject``), so it never allocates any
    ``H*W`` arrays. ``features`` is accepted for signature compatibility and
    no longer read (cells came from it to rasterize the dropped stack).
    """
    cache = MaskCache(grid_shape, cell_size_m=float(graph.cell_size_m))
    for fid, node in graph.nodes.items():
        cache.add(
            fid,
            centroid=(float(node.centroid[0]), float(node.centroid[1])),
            cell_count=int(node.cell_count),
        )
    return cache


def attach_mask_cache(
    graph: ObjectGraph,
    grid_shape: tuple[int, int],
    *,
    features: dict | None = None,
) -> MaskCache:
    """Build and stash a ``MaskCache`` ledger on ``graph._mask_cache``."""
    cache = build_mask_cache(graph, grid_shape, features=features)
    graph.__dict__["_mask_cache"] = cache
    return cache


def get_mask_cache(graph: ObjectGraph) -> MaskCache | None:
    """Read the stashed mask cache, or ``None`` if none was attached."""
    return graph.__dict__.get("_mask_cache")
