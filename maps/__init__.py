"""MAPS: Multimodal Agentic Planning and Search.

Seven subpackages: ``mission`` (the mission conversation: one persistent
reasoner, one served tool, the turn loop), ``library`` (the Python the model
writes: the block interpreter, the calls and the rows they hand back),
``query`` (the op program the calls compile onto, over the surface rows),
``render`` (the look crop and the map schematic), ``world_state`` (the
structured world-state graph), ``perception`` (the SAM 3.1 detection client,
the nadir ortho crop and the colour read) and ``llm`` (the reasoner
transport). ``xy_token`` holds the ``[x, y]`` point grammar. The benchmarks and the flight harness live in
``benchmarks``.
"""
