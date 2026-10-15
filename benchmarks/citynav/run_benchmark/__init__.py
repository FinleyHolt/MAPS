"""CityNav benchmark runner.

The episode engine is a package of purpose-named modules: ``cli`` (main),
``dispatch`` -> ``episode_runner`` (per-episode flow) -> ``mission_engine``
(the one-conversation fly-on-request executor) -> ``trajectory`` (discrete
replay), with ``assets`` / ``osm_prior`` as leaves and ``config``
alongside. Importers name the module they need.
"""
