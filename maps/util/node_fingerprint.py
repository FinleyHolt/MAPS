"""Does this graph node still hold what it held before?

One hash of a node's whole content; the analyze surface
(``analyze_scene.geom_program``) reuses a row while its node's fingerprint is
unchanged.

Content, never identity: a node rebuilt with the same values is unchanged,
and no hand-maintained dirty list has to know every mutation path. The whole
``__dict__`` is hashed rather than the fields a caller reads, so a new
``MapObject`` field cannot be invisible to the key.
"""

from __future__ import annotations

import numpy as np


def fingerprint_value(v):
    """A hashable summary of ``v`` that changes whenever its content does.

    Arrays hash their BYTES: ``cells`` is the bulk of a node (3.9 M cells over
    the Cal City graph) and identity says nothing about it after a round trip.
    Numeric sequences take the same path, so a centerline costs one hash
    rather than one tuple per vertex.
    """
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, np.ndarray):
        return ("nd", v.shape, v.dtype.str, hash(v.tobytes()))
    if isinstance(v, (list, tuple)):
        try:
            a = np.asarray(v)
            if a.dtype.kind in "ifb":
                return ("nd", a.shape, a.dtype.str, hash(a.tobytes()))
        except Exception:  # noqa: BLE001 - ragged/object sequence, walk it
            pass
        return ("seq", tuple(fingerprint_value(x) for x in v))
    if isinstance(v, dict):
        return ("map", tuple(sorted((str(k), fingerprint_value(x))
                                    for k, x in v.items())))
    return ("repr", repr(v))


def fingerprint_node(n):
    """Fingerprint a graph node's whole ``__dict__``."""
    d = getattr(n, "__dict__", None)
    if d is None:
        return ("repr", repr(n))
    return tuple(sorted((k, fingerprint_value(v)) for k, v in d.items()))
