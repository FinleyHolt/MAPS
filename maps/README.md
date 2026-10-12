# maps

The method: one mission conversation in which a reasoning model writes Python
into one `run` tool, against a library of calls over a world-state graph that
holds the OSM prior and the segmenter's detections.

`maps` imports nothing from `benchmarks`. A benchmark or the flight harness
hands the loop a `MissionSpec` (`search/parse_compile/reasoner_loop.py`)
whose `fly` executes one move (resolve a stop, observe, segment, fold the
detections into the graph) and returns the flight report the model reads.

## Concepts and modules

| Concept | Module |
|---|---|
| The mission conversation: one persistent reasoner, one served tool, the turn loop, operator taskings | `search/parse_compile/reasoner_loop.py` |
| The request each turn sends: model, messages, tools, `max_tokens` | `search/parse_compile/reasoner_agent.py` |
| The served text: system turn, the `run` tool's description and schema | `search/parse_compile/prompts.py` |
| Printing the served contract | `search/parse_compile/contract.py` |
| The Python surface: executing a block, the library calls, rows, `.filter` and `.order` | `search/parse_compile/python_surface.py` |
| Compiling library calls onto geometry ops | `search/parse_compile/code_surface.py` |
| Which classes the segmenter watches | `search/parse_compile/perception_director.py`, `search/search_loop.py` (`declare_mission_concepts`) |
| The geometry op program behind `find`, `intersection`, `ends_of`, `distance` | `search/analyze_scene/geom_program.py`, `search/analyze_scene/geom_compute.py`, `search/analyze_scene/row_geometry.py` |
| Matching a query to rows by name | `search/analyze_scene/text_match.py` |
| Transcript compaction when the conversation outgrows the window | `search/analyze_scene/compactor.py` |
| The per-episode search state machine and its observation step | `search/search_loop.py` |
| The `look` crop and the `show_map` schematic | `search/render/look_render.py`, `search/render/map_render.py` |
| The bench ablation withholds | `search/withhold.py` |
| The world-state graph: map features and detected nodes, their upsert and serialization | `sws/graph/` |
| The orthographic observation: a pose to a cropped window of the raster | `perception/observation.py`, `perception/ortho/` |
| Sensor specs (the C-12 Oryx, the CityNav nadir camera) and pixel-to-world projection | `perception/geometry/` |
| The SAM 3.1 client and the classes it is asked for | `perception/sam/detect_client.py`, `perception/sam/concept_provider.py` |
| Detections stamped onto the tile grid | `perception/ortho/detection_stamping.py` |
| World-space tracking of detections across flights | `perception/world_tracker.py` |
| Seen colour of a detection | `perception/viz/color.py` |
| Catalog-prior footprints projected onto the ortho crop | `perception/ortho/prior_overlay.py` |
| Endpoint discovery: served model, window, output reserve | `llm/endpoint_spec.py` |
| Per-model sampling envelopes and the reasoning effort | `llm/model_cards.py` |
| The chat-completions transport and its flight log | `llm/http.py` |
| The request shape a hosted OpenAI model takes | `llm/openai_wire.py` |
| Two reasoner turns per effort against the configured endpoint, before a run | `llm/endpoint_preflight.py` |
| An episode's LLM activity as one row block (`metrics.llm`) | `llm/flight_summary.py` |
| Mask records on the wire and on disk | `util/mask_codec.py`, `util/mask_region.py` |
