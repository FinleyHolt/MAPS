"""Rendering the reasoner looks at — currently the nadir ``look`` crop renderer.

Pure functions over resolved crop specs + the block raster; no loop state, so
live and offline-replay crops are byte-identical by construction.
"""
