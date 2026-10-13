"""ViSA (arXiv 2603.08007) on this harness's substrate.

Perception (Set-of-Mark candidates) -> three-stage Verification (literal
attributes, spatial topology by mark id, geographic boundary against the
landmark prior) -> Executor (stop / move / ascend / descend), with the
Verification module's guidance signal fed back to Perception. Waypoints come
from the paper's contour-union -> grid -> greedy set cover -> TSP executor.
The paper released no code; this is the port from its text. Every deviation
is a row in ``baselines/README.md``.
"""
