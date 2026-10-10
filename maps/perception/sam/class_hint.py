"""Map free-form object labels to the open-vocab class_hint bucket.

Perception adapters that consume open-vocabulary model outputs (SegFormer
class names, SAM 3.1 concept labels) emit ``DetectedObject`` with a
coarse ``class_hint`` string. Downstream consumers (runtime↔runtime SAM
mask dedup, the IoU re-id fallback) group by ``class_hint``. The valid
buckets are ``object | buildings | water | roads | parking | natural |
landuse`` — an open vocabulary in the sense that unknown free-form labels
(and every discrete object: car, person, bench, sign, …) collapse to the
generic ``object`` bucket rather than being rejected. (The reasoner-facing
node class is the FINE SAM label, set separately on the graph node; this
coarse bucket only backs dedup / re-id.)

The mapping itself lives in ``prompts/class_hint_map.json`` and ships as
package-data, so it can be edited without a code change. The resolver
normalizes the query (lowercase, strip), tries an exact hit first, then
falls back to ``difflib.get_close_matches`` with a tight cutoff so a
SegFormer "buildings" vs "building" variant still lands in the right
bucket.
"""

from __future__ import annotations

import difflib
import functools
import json
from importlib import resources

_FALLBACK = "object"
_CLOSE_MATCH_CUTOFF = 0.85


@functools.lru_cache(maxsize=1)
def _load_map() -> dict[str, str]:
    text = (
        resources.files("maps.perception")
        .joinpath("prompts/class_hint_map.json")
        .read_text()
    )
    raw = json.loads(text)
    return {k.lower(): v for k, v in raw.items() if not k.startswith("_")}


@functools.lru_cache(maxsize=2048)
def resolve_class_hint(free_form: str) -> str:
    """Return the open-vocab bucket for ``free_form``.

    Lookup order: exact lower-case match, then close-match against the
    mapping keys (cutoff 0.85). Returns ``"object"`` when nothing matches.
    """
    if not free_form:
        return _FALLBACK
    key = free_form.strip().lower()
    if not key:
        return _FALLBACK
    mapping = _load_map()
    direct = mapping.get(key)
    if direct is not None:
        return direct
    close = difflib.get_close_matches(
        key, mapping.keys(), n=1, cutoff=_CLOSE_MATCH_CUTOFF
    )
    if close:
        return mapping[close[0]]
    return _FALLBACK
