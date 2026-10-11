"""The tile layout a prior footprint is projected through."""

from maps.perception.geometry.geolocate import TileFrame as TileBounds

# ponytail: alias for osm_prior's import; it goes when osm_prior imports TileFrame
__all__ = ["TileBounds"]
