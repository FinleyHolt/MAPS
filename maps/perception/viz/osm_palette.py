"""OSM-label palette + label rule for the model-facing world-state map.

The map render (``maps.search.render.map_render``) paints by KIND first,
label second: the question the schematic answers is spatial ("which house
fronts which road"), and the ground truth of the digest palette is that
``building`` (104,219,92), ``residential`` (97,219,92) and ``tertiary``
(92,219,110) — the house, the street it fronts, and the landuse behind
both — land on one indistinguishable green (measured on the facing render
gate, 12/12 sheets). So the load-bearing kinds carry RESERVED colors
(:func:`paint_color`): roads light gray, prior buildings tan, areas their
digest hue muted to context, detections their class's digest hue at full
saturation. Hue identity within a kind still comes from the digest, so a
label keeps one hue on every surface.

``osm_label`` and the raw digest pair (:func:`color_for` /
:func:`color_word`) are the label rule and the colour rule every surface
reads.
"""

from __future__ import annotations

import colorsys
import hashlib

# The tag whose value names a feature's kind, first hit wins. ``highway=residential`` -> "residential"; a bare flag value
# (``building=yes``) labels as its key ("building").
_LABEL_TAG_KEYS = (
    "highway", "building", "natural", "landuse", "leisure", "amenity",
    "railway", "waterway", "water", "man_made", "aeroway", "barrier",
)

# 12 hue-bucket words (30 deg each, centered on 0=red) so a text legend can
# name a digest-derived hue: the model reads "residential=teal" beside the
# teal region. Coarse on purpose -- a schematic's legend, not a color science.
_HUE_WORDS = (
    "red", "orange", "yellow", "olive", "green", "spring-green",
    "cyan", "azure", "blue", "violet", "magenta", "pink",
)


def osm_label(tags: dict | None, fallback: str = "prior") -> str:
    """A graph node's display label from its raw OSM tags (the one label
    rule): the tag value when it names a kind, the key on a bare flag,
    else ``fallback`` -- lowercased."""
    for key in _LABEL_TAG_KEYS:
        val = (tags or {}).get(key)
        if isinstance(val, str) and val.strip():
            v = val.strip().lower()
            return key if v in ("yes", "true", "1") else v
    return str(fallback or "prior").strip().lower()


def color_for(label: str) -> tuple[int, int, int]:
    """A stable, distinct RGB for an OSM label (same label, same colour,
    every surface)."""
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    r, g, b = colorsys.hsv_to_rgb(hue, 0.58, 0.86)
    return round(r * 255), round(g * 255), round(b * 255)


def color_word(label: str) -> str:
    """The hue-bucket word for a label's colour, for the map's text legend
    ("residential=teal"). Derived from the SAME digest as :func:`color_for`
    so word and paint can never disagree."""
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return _HUE_WORDS[int(hue * 12.0) % 12]


# The schematic's reserved kind colors (RGB). Roads are linework, buildings
# are the figure, areas are ground: three visual ranks the digest alone
# cannot guarantee.
_ROAD_RGB = (208, 208, 208)
_BUILDING_RGB = (206, 160, 102)

# Tag keys that make a feature a ROAD / a BUILDING for painting purposes.
# Everything else (landuse, natural, leisure, water, barriers, POIs) is
# context and paints as an AREA.
_ROAD_TAG_KEY = "highway"
_BUILDING_TAG_KEY = "building"


def osm_kind(tags: dict | None) -> str:
    """A prior's paint kind from its raw OSM tags: ``road`` | ``building``
    | ``area``. The label alone cannot carry this — ``highway=residential``
    (a street) and ``landuse=residential`` (a tract) both label
    ``residential``."""
    t = tags or {}
    if _ROAD_TAG_KEY in t:
        return "road"
    if _BUILDING_TAG_KEY in t:
        return "building"
    return "area"


def paint_color(label: str, *, kind: str = "area",
                detected: bool = False) -> tuple[int, int, int]:
    """The schematic paint for one feature: reserved color per kind,
    digest hue for identity where hue is the information.

    detections -> the class's digest hue at full saturation (the newest
    knowledge, the brightest ink); roads -> light gray (linework);
    prior buildings -> tan (the figure); areas -> the label's digest hue
    muted to ground. Stable per (label, kind, provenance) by construction.
    """
    if detected:
        digest = hashlib.sha256(label.encode("utf-8")).digest()
        hue = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
        r, g, b = colorsys.hsv_to_rgb(hue, 0.80, 0.95)
        return round(r * 255), round(g * 255), round(b * 255)
    if kind == "road":
        return _ROAD_RGB
    if kind == "building":
        return _BUILDING_RGB
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    r, g, b = colorsys.hsv_to_rgb(hue, 0.30, 0.34)
    return round(r * 255), round(g * 255), round(b * 255)


def paint_word(label: str, *, kind: str = "area",
               detected: bool = False) -> str:
    """The legend word for :func:`paint_color` — named from the SAME rule
    so word and paint can never disagree."""
    if detected or kind not in ("road", "building"):
        return color_word(label)
    return "light-gray" if kind == "road" else "tan"
