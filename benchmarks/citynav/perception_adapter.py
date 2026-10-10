"""CityRefer → perception.GTObject glue.

Lives on the benchmark side because only the benchmark knows about
CityRefer object shape. Perception stays robot-agnostic; this module is the
only place that imports both sides.
"""

from __future__ import annotations

from maps.perception.base import GTObject

from benchmarks.citynav.adapter import OBJTYPE_TO_LABEL, CityRefObject

# Mirrors benchmarks.citynav.adapter._OBJTYPE_TO_CATEGORY (keep in sync so
# detected nodes sort into the same graph buckets as prebaked CityRefer
# nodes when that helps).
_OBJTYPE_TO_CATEGORY: dict[str, str] = {
    "Building": "buildings",
    "Wall": "buildings",
    "Bridge": "buildings",
    "Water": "water",
    "TrafficRoad": "roads",
    "Rail": "roads",
    "Footpath": "roads",
    "Parking": "parking",
    "HighVegetation": "natural",
    "Ground": "landuse",
}


def citynav_to_gt_perfect_objects(
    cityref_objects: list[CityRefObject],
    color_table: dict[str, dict] | None = None,
) -> list[GTObject]:
    """Flatten CityRefObjects to GTObject seeds for the GT-PERFECT sensor.

    Two anti-cheat-critical properties:

    1. **Label is a CLASS NOUN, never the CityRefer name.** The runtime sets the
       graph node ``name = det.label`` (``search_loop._detection_to_mapobject``), and
       the ``find`` op surfaces node name/class — so passing ``obj.name`` would
       leak the catalog identity to the reasoner and break the descriptions
       wall. We map ``object_type`` -> the canonical scene noun
       (``OBJTYPE_TO_LABEL``, e.g. ``"Car" -> "car"``) instead.
    2. **Carries the true footprint contour + the offline GT color.** The
       perfect sensor rasterizes ``contour`` into the real mask, and injects the
       cue-independent GT color row (``{"color", "color_conf"}``) from
       ``color_table`` keyed by the string object id. ``descriptions``/``name``
       are read NOWHERE here — only ``object_type``/``position``/``contour`` and
       the pre-built pixel-color table feed the seed.
    """
    table = color_table or {}
    out: list[GTObject] = []
    for obj in cityref_objects:
        label = OBJTYPE_TO_LABEL.get(str(obj.object_type)) or (
            str(obj.object_type).lower() or "object"
        )
        class_hint = _OBJTYPE_TO_CATEGORY.get(str(obj.object_type), label)
        out.append(
            GTObject(
                world_xy=(float(obj.position[0]), float(obj.position[1])),
                label=label,
                class_hint=class_hint,
                object_type=str(obj.object_type),
                id=int(obj.id),
                contour=list(obj.contour) or None,
                attributes=dict(table.get(str(obj.id), {})),
                dimension=(
                    tuple(float(d) for d in obj.dimension)
                    if obj.dimension is not None
                    else None
                ),
            )
        )
    return out
