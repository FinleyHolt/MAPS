"""The ``analyze`` op-program — the reasoner's one query surface over world state.

Not LLM transport (that is ``maps.llm``); this is the typed op-program the
reasoner emits plus the kernels that execute it:

- ``geom_program`` — runs the typed op-DAG (``find`` / ``compare`` /
  ``filter_items`` / ``score_by``) over the ``(cands, anchors)`` surface.
- ``geom_compute`` — the single source of frame-correct spatial math.
- ``text_match`` — the lexical ``find`` matcher.
- ``tools`` — the read-only tile-query tool surface the reasoner calls.
- ``compactor`` — deterministic clear-to-census autocompaction of the reasoner
  transcript (no LLM call).

Consumed by ``parse_compile/`` stages; the compiler calls into this library,
never the reverse.
"""
