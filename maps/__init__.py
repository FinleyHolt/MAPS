"""MAPS: Multimodal Agentic Planning and Search.

Five subpackages: ``sws`` (the structured world-state graph), ``perception``
(the SAM 3.1 detection client, the nadir ortho crop and the color
extractor), ``search`` (the single persistent reasoner, the ``analyze``
op-program, and the commit), ``llm`` (the LLM/VLM transport), and ``util``
(cross-cutting plumbing). The eval harness lives in ``benchmarks.citynav``
(the ``maps-bench`` CLI and the canonical SR readers under
``benchmarks.citynav.analysis``).
"""
