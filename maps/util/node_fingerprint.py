"""Does this graph node still hold what it held before?

One hash of a node's whole content, used wherever the answer "nothing about
this node changed" is worth more than the work it saves:

  * the analyze surface reuses a row rather than rebuilding it
    (``analyze_scene.geom_program``).

CONTENT, never identity. Both callers sit downstream of a snapshot round trip
that rebuilds every node object with the same values, so ``is`` answers "new
object" on nodes that are byte-for-byte what they were. And never a
hand-maintained dirty list either: that has to be right about every mutation
path in the tree forever, and the first one anyone forgets is a stale answer
with nothing saying so.

The WHOLE ``__dict__`` is hashed, not the fields a caller happens to read
today. Enumerating them would be faster and would rot -- a new ``MapObject``
field would be invisible to the key, and the caller would go on serving
answers computed before it existed. That is the same defect class the ingest
client's ``IngestServerStale`` guard exists for.
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


def fingerprint_nodes(nodes) -> dict:
    """``{fid: fingerprint}`` over a ``{fid: node}`` mapping."""
    return {str(f): fingerprint_node(n) for f, n in nodes.items()}
