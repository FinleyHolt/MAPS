"""The least squares over a window of frames: each frame's translation, and one yaw and one scale for the window, from anchors and ties.

An anchor is a frame's own registration against the map (the metres to add
to the chain's pose), a tie one ground point seen by two frames. Both are
linear in the frames' corrections on a flat floor: a translation ``d_k``
moves the ground under every pixel of frame k by ``d_k``, a yaw ``psi`` moves
it by the pixel's lever ``J`` times ``psi``, and a scale ``eps`` by the
pixel's arm from the nadir ``R`` times ``eps``:

    anchor:  d_k + J_k(c) psi + R_k(c) eps                               = raw_k
    tie:     d_i - d_j + (J_i - J_j) psi + (R_i - R_j) eps                = G_j(uv_j) - G_i(uv_i)

with ``G`` the ground under the pixel through the CHAIN pose. The scale is
the frames' height over the real ground against the height the floor gives
them. Each equation is scaled by its sigma and the whole is Huber-weighted
(:func:`irls_fit`), the anchors and the ties each against their own spread,
so a false peak loses its say smoothly and the many tight ties never read the
few loose anchors as outliers.

The yaw and the scale columns are freed only where the window's ties can
carry them (:data:`MIN_ROTATE_MARKS` ties whose rms lever about the nadir is
:data:`MIN_ROTATE_ARM` of the ground half-diagonal); a fitted yaw past
:data:`MAX_YAW_DEG` or a scale past :data:`MAX_SCALE` drops its column and
the system is solved again. A connected component of the tie graph that no
anchor reaches is unreached and keeps the corrections it came with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Ties a window needs, and their rms lever about the nadir as a share of the
#: frame's ground half-diagonal, before a yaw or a scale is fitted.
MIN_ROTATE_MARKS = 6
MIN_ROTATE_ARM = 0.2
#: A fitted yaw past this is not a crab, and a scale past this share is not a
#: floor a few metres off: the column is dropped.
MAX_YAW_DEG = 8.0
MAX_SCALE = 0.1
#: The yaw and the scale each carry a weak prior of zero at these sigmas, a
#: population of their own in the weighting: inert where the ties determine
#: them, and a brake where two ties from one false lock would turn the strip.
YAW_PRIOR_DEG = 3.0
SCALE_PRIOR = 0.05
#: The whole-frame registration's sigma at a response of one, metres; a row's
#: sigma is this over its response.
ANCHOR_SIGMA_M = 1.5
#: A tie's sigma, metres: one work pixel, since the peak is an integer one.
TIE_SIGMA_M = 0.3
#: The Huber cut in a population's own scale, the least scale a cut is read
#: from (every row is already scaled by its own sigma), and the reweightings.
HUBER_K = 2.0
SCALE_FLOOR = 1.0
IRLS_ITERATIONS = 8


@dataclass(frozen=True)
class Node:
    """A frame in the solve: its nadir, and its footprint's half-diagonal on the floor."""

    nadir_xy: tuple[float, float]
    half_diagonal_m: float


@dataclass(frozen=True)
class Anchor:
    """A frame's registration: the metres to add to the chain pose, and the levers at the centre."""

    frame: object
    raw: tuple[float, float]
    sigma_m: float
    jac: tuple[float, float]
    rad: tuple[float, float]


@dataclass(frozen=True)
class TieRow:
    """One ground point seen by two frames: where each chain pose puts it, and each pixel's levers."""

    i: object
    j: object
    g_i: tuple[float, float]
    g_j: tuple[float, float]
    jac_i: tuple[float, float]
    jac_j: tuple[float, float]
    rad_i: tuple[float, float]
    rad_j: tuple[float, float]


@dataclass
class Adjustment:
    """Each reached frame's correction and sigma, the window's yaw and scale
    where fitted, and how many ties each frame carries."""

    correction: dict = field(default_factory=dict)
    sigma: dict = field(default_factory=dict)
    yaw_deg: dict = field(default_factory=dict)
    scale: dict = field(default_factory=dict)
    ties_per_frame: dict = field(default_factory=dict)


def irls_fit(A, b: np.ndarray, *, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(x, sigmas)``: iteratively reweighted least squares on a sparse ``A x = b``, Huber-weighted per observation.

    An observation is two consecutive rows, weighted by the norm of its
    residual together. ``groups`` labels each observation with a population
    that gets a Huber scale of its own. The normal matrix is solved by
    Cholesky, its sigmas the inverse's diagonal; a design that is not full
    rank raises ``ValueError``.
    """
    import scipy.linalg  # noqa: PLC0415

    n, p = A.shape[0] // 2, A.shape[1]
    groups = np.asarray(groups)
    weights = np.ones(n, dtype=np.float64)
    factor = None
    for _ in range(IRLS_ITERATIONS):
        x, factor = _normal_solve(A, b, np.repeat(weights, 2))
        norms = np.linalg.norm(np.asarray(b - A @ x).reshape(n, 2), axis=1)
        cut = np.empty(n, dtype=np.float64)
        for g in np.unique(groups):
            cut[groups == g] = HUBER_K * max(_huber_scale(norms[groups == g]), SCALE_FLOOR)
        new_weights = np.where(norms <= cut, 1.0, cut / np.maximum(norms, 1e-12))
        if np.allclose(new_weights, weights, atol=1e-9):
            weights = new_weights
            break
        weights = new_weights
    w = np.repeat(weights, 2)
    resid_w = np.asarray(b - A @ x) * w
    var = float(resid_w @ resid_w) / max(2 * n - p, 1)
    cov_diag = np.diag(scipy.linalg.cho_solve(factor, np.eye(p))) * var
    return x, np.sqrt(np.maximum(cov_diag, 0.0))


def _huber_scale(norms: np.ndarray) -> float:
    """The MAD of one population's residual norms as a Gaussian sigma; the RMS
    when more than half are identical, since a MAD a millionth of the RMS is
    the solver's rounding, not a spread."""
    rms = float(np.sqrt(np.mean(norms**2)))
    scale = 1.4826 * float(np.median(np.abs(norms - np.median(norms))))
    if scale <= max(1e-12, 1e-6 * rms):
        scale = max(rms, 1e-12)
    return scale


def _normal_solve(A, b: np.ndarray, w: np.ndarray):
    """``(x, factor)``: the weighted normal equations solved by Cholesky."""
    import scipy.linalg  # noqa: PLC0415

    aw = A.multiply(w[:, None]).tocsr()
    normal = (aw.T @ aw).toarray()
    rhs = np.asarray(aw.T @ (np.asarray(b) * w)).reshape(-1)
    try:
        factor = scipy.linalg.cho_factor(normal, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the design is not full rank: a parameter no equation observes") from exc
    return scipy.linalg.cho_solve(factor, rhs, check_finite=False), factor


def _components(frames: list, ties: list[TieRow]) -> list[set]:
    """The tie graph's connected components over ``frames``."""
    import scipy.sparse  # noqa: PLC0415
    from scipy.sparse.csgraph import connected_components  # noqa: PLC0415

    index = {f: k for k, f in enumerate(frames)}
    rows = [index[t.i] for t in ties] + [index[t.j] for t in ties]
    cols = [index[t.j] for t in ties] + [index[t.i] for t in ties]
    graph = scipy.sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(frames), len(frames)))
    _n, labels = connected_components(graph, directed=False)
    out: dict[int, set] = {}
    for f, label in zip(frames, labels):
        out.setdefault(int(label), set()).add(f)
    return list(out.values())


def _levered(nodes: dict, ties: list[TieRow], solved: set, *, radial: bool) -> set:
    """The frames whose pooled ties can carry the window's yaw (or, ``radial``,
    its scale): every tied frame, or none. The yaw's lever is the tie's ground
    about the nadir, the scale's the row's own radial arm."""
    arms: dict = {}
    for t in ties:
        if t.i not in solved or t.j not in solved:
            continue
        if radial:
            arms.setdefault(t.i, []).append(t.rad_i)
            arms.setdefault(t.j, []).append(t.rad_j)
        else:
            arms.setdefault(t.i, []).append(np.subtract(t.g_i, nodes[t.i].nadir_xy))
            arms.setdefault(t.j, []).append(np.subtract(t.g_j, nodes[t.j].nadir_xy))
    if not arms:
        return set()
    pooled = [a for f in arms for a in arms[f]]
    half = float(np.median([nodes[f].half_diagonal_m for f in arms]))
    if len(pooled) < MIN_ROTATE_MARKS:
        return set()
    arm = np.asarray(pooled, dtype=np.float64)
    lever = float(np.sqrt(np.mean(np.sum(arm * arm, axis=1))))
    return set(arms) if half > 0.0 and lever >= MIN_ROTATE_ARM * half else set()


def _system(anchors: list[Anchor], ties: list[TieRow], columns: dict, yaw_cols: dict, scale_cols: dict, width: int):
    """``(A, b, priors)``: two equations a row, each scaled by its sigma, the priors first."""
    import scipy.sparse  # noqa: PLC0415

    data, rows, cols, b = [], [], [], []
    n = 0

    def put(row: int, col: int, val: float) -> None:
        rows.append(row)
        cols.append(col)
        data.append(val)

    priors = 0
    for col in sorted(set(yaw_cols.values())):
        put(n, col, 1.0 / YAW_PRIOR_DEG)
        b.extend([0.0, 0.0])
        n += 2
        priors += 1
    for col in sorted(set(scale_cols.values())):
        put(n + 1, col, 1.0 / SCALE_PRIOR)
        b.extend([0.0, 0.0])
        n += 2
        priors += 1
    for a in anchors:
        base = columns[a.frame]
        s = 1.0 / a.sigma_m
        for c in range(2):
            put(n + c, base + c, s)
            if a.frame in yaw_cols:
                put(n + c, yaw_cols[a.frame], a.jac[c] * s)
            if a.frame in scale_cols:
                put(n + c, scale_cols[a.frame], a.rad[c] * s)
            b.append(a.raw[c] * s)
        n += 2
    for t in ties:
        bi, bj = columns[t.i], columns[t.j]
        s = 1.0 / TIE_SIGMA_M
        for c in range(2):
            put(n + c, bi + c, s)
            put(n + c, bj + c, -s)
            if t.i in yaw_cols:
                put(n + c, yaw_cols[t.i], t.jac_i[c] * s)
            if t.j in yaw_cols:
                put(n + c, yaw_cols[t.j], -t.jac_j[c] * s)
            if t.i in scale_cols:
                put(n + c, scale_cols[t.i], t.rad_i[c] * s)
            if t.j in scale_cols:
                put(n + c, scale_cols[t.j], -t.rad_j[c] * s)
            b.append((t.g_j[c] - t.g_i[c]) * s)
        n += 2
    a = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n, width)).tocsr()
    return a, np.asarray(b, dtype=np.float64), priors


def rows_from(geometry: dict, measurements: dict, ties) -> tuple[dict, list[Anchor], list[TieRow]]:
    """``(nodes, anchors, tie_rows)`` from each frame's ``FrameGeometry`` (by
    key), the registrations ``key -> (east, north, response)`` and the ties
    ``(key_i, key_j, uv_i, uv_j)``."""
    nodes = {k: Node(g.nadir_xy, g.half_diagonal_m) for k, g in geometry.items()}
    anchors = [Anchor(k, (float(e), float(n)), ANCHOR_SIGMA_M / float(response), geometry[k].lever(geometry[k].centre),
                      geometry[k].radial(geometry[k].centre))
               for k, (e, n, response) in measurements.items() if response > 0.0]
    tie_rows = [TieRow(i, j, geometry[i].ground(uv_i), geometry[j].ground(uv_j), geometry[i].lever(uv_i),
                       geometry[j].lever(uv_j), geometry[i].radial(uv_i), geometry[j].radial(uv_j))
                for i, j, uv_i, uv_j in ties]
    return nodes, anchors, tie_rows


def solve(nodes: dict, anchors: list[Anchor], ties: list[TieRow]) -> Adjustment:
    """The adjustment over ``nodes`` (``frame -> Node``) from its anchors and ties.

    A frame in no row, or in a tie component no anchor reaches, gets no
    correction here; the caller keeps what it had.
    """
    frames = list(nodes)
    anchored = {a.frame for a in anchors}
    reached: set = set()
    for component in _components(frames, ties):
        if component & anchored:
            reached |= component
    out = Adjustment()
    anchors = [a for a in anchors if a.frame in reached]
    ties = [t for t in ties if t.i in reached and t.j in reached]
    if not anchors:
        return out
    for t in ties:
        out.ties_per_frame[t.i] = out.ties_per_frame.get(t.i, 0) + 1
        out.ties_per_frame[t.j] = out.ties_per_frame.get(t.j, 0) + 1
    rotated = _levered(nodes, ties, reached, radial=False)
    scaled = _levered(nodes, ties, reached, radial=True)
    solved = sorted(reached, key=frames.index)
    for _attempt in range(3):
        columns, width = {}, 0
        for f in solved:
            columns[f] = width
            width += 2
        yaw_cols, scale_cols = {}, {}
        if rotated:
            yaw_cols = {f: width for f in rotated}
            width += 1
        if scaled:
            scale_cols = {f: width for f in scaled}
            width += 1
        a, b, priors = _system(anchors, ties, columns, yaw_cols, scale_cols, width)
        x, sigmas = irls_fit(a, b, groups=np.r_[np.full(priors, 2, dtype=np.int64),
                                                np.zeros(len(anchors), dtype=np.int64),
                                                np.ones(len(ties), dtype=np.int64)])
        too_far = {f for f, col in yaw_cols.items() if abs(x[col]) > MAX_YAW_DEG}
        too_big = {f for f, col in scale_cols.items() if abs(x[col]) > MAX_SCALE}
        if not too_far and not too_big:
            break
        rotated -= too_far
        scaled -= too_big
    for f in solved:
        c = columns[f]
        out.correction[f] = (float(x[c]), float(x[c + 1]))
        out.sigma[f] = (float(sigmas[c]), float(sigmas[c + 1]))
        if f in yaw_cols:
            out.yaw_deg[f] = float(x[yaw_cols[f]])
        if f in scale_cols:
            out.scale[f] = float(x[scale_cols[f]])
    return out
