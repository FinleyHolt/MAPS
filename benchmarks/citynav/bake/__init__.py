"""Preflight tile baking for the CityNav benchmark (run once, offline).

Bakes the per-block prior sidecars from the OpenStreetMap caches, one block
per raster (:mod:`benchmarks.citynav.bake.city`); none of it is on the
per-episode path (``benchmarks/citynav/data_prep/bake_tiles.sh`` wraps it).
"""
