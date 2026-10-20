"""A human operator on this harness's substrate.

The operator flies the episode from a browser console: the instruction, the
run's own north-up frame at the current pose with the OSM landmark prior
outlined on it, and the action budget left. Each click flies a stop through
the same budget the other policies spend, and the commit is the operator's
own point, so a human row and a MAPS row differ only in what chose the
stops.
"""
