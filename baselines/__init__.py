"""Published methods re-measured on this harness's substrate.

A baseline fills the span between "episode loaded" and "stops flown, target
committed", and nothing else: the sensor, the OSM prior, the six-action replay
with its commit-leg reserve, the scorer and the row contract are the bench's.
``baselines/seam.py`` is that contract in code;
``baselines/<policy>/policy.py`` is one method each, and
``baselines/released/`` runs published checkpoints through their
own evaluators. The package sits beside the benches it measures on and
imports them; ``benchmarks.citynav`` reaches it only by name, through
``importlib`` on a config's ``policy:``.
"""
