"""Preflight asset baking for the CityNav benchmark (run once, offline).

Generates the per-block geometry, RGBD rasters, satellite views, feature graphs,
and resolver caches the runtime reads from disk. None of this is on the
per-episode flight path -- it is the one-time setup ceremony
(`benchmarks/citynav/data_prep/` wraps the common invocations).
"""
