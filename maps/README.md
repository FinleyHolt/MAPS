# maps

The method: one mission conversation in which a reasoning model writes Python
into one `run` tool, against a library of calls over a world-state graph that
holds the OSM prior and the segmenter's detections.

`maps` imports nothing from `benchmarks`. A benchmark or the flight harness
hands the loop a `MissionSpec` (`mission/spec.py`)
whose `fly` executes one move (resolve a stop, observe, segment, fold the
detections into the graph) and returns the flight report the model reads.

## Concepts and modules

| Concept | Module |
|---|---|
| The mission conversation: one persistent reasoner, one served tool, the turn loop, operator taskings | `mission/loop.py` |
| The verbs a library call runs on: grounding and running a program, flying, pictures, the watchlist | `mission/program_verbs.py`, `mission/fly_verbs.py`, `mission/picture_verbs.py`, `mission/watch_verbs.py`, `mission/surface_state.py`, `mission/acks.py` |
| The turns that are not a reply and the nudge each draws; the request failures that end a mission | `mission/reply_repair.py`, `mission/serve_refusals.py` |
| Result handles: the op-program runner, each handle re-derived over the current surface | `library/result_handles.py` |
| What a look and a map render and the captions their pictures carry | `mission/picture_requests.py` |
| An explore's stops over a place or a set | `mission/explore_stops.py` |
| The request each turn sends: model, messages, tools, `max_tokens` | `mission/reasoner.py` |
| The served text: system turn, the `run` tool's description and schema | `mission/served_text.py` |
| Printing the served contract | `mission/contract.py` |
| The library the model writes against: executing a block, the calls, rows, `.filter` and `.order` | `library/interpreter.py`, `library/block_scan.py`, `library/block_ceilings.py`, `library/calls.py`, `library/call_arguments.py`, `library/rows.py` |
| Compiling library calls onto ops | `library/op_compiler.py` |
| Which classes the segmenter watches | `perception/watchlist.py`, `mission/episode_world.py` (`declare_mission_concepts`) |
| The surface the op program runs over: one row per graph node, its served point, mask and colour fields | `query/surface.py` |
| The op program behind `find`, `intersection`, `ends_of`, `distance` | `query/program.py`, `query/op_io.py`, `query/op_find.py`, `query/op_roads.py`, `query/op_cut_and_sort.py`, `query/op_compare.py`, `query/spatial_math.py`, `query/color_words.py`, `query/row_geometry.py` |
| What a result serves: the table, its legend, census and hints | `query/result_tables.py` |
| Matching a query to rows by name | `query/text_match.py` |
| Transcript compaction when the conversation outgrows the window | `mission/compactor.py` |
| The per-episode world: the one graph a mission reads and each flight's detections fold into | `mission/episode_world.py` |
| The `look` crop and the `show_map` schematic | `render/look_render.py`, `render/map_render.py` |
| Marks drawn on a picture, and the map's label rule and colours | `render/mark_overlay.py`, `render/osm_palette.py` |
| The bench ablation withholds | `withhold.py` |
| The `[x, y]` point grammar every slot that takes a fid or a point reads | `xy_token.py` |
| The world-state graph: map features and detected nodes, their upsert, the context a query reads it through | `world_state/` |
| The orthographic observation: a pose to a cropped window of the raster | `perception/observation.py`, `perception/ortho/` |
| Sensor specs (the C-12 Oryx, the CityNav nadir camera) and pixel-to-world projection | `perception/geometry/` |
| The SAM 3.1 client and the classes it is asked for | `perception/sam/detect_client.py` |
| Detections stamped onto the tile grid | `perception/ortho/detection_stamping.py` |
| World-space tracking of detections across flights | `perception/world_tracker.py` |
| Seen colour of a detection | `perception/color.py` |
| Catalog-prior footprints projected onto the ortho crop | `perception/ortho/prior_overlay.py` |
| Endpoint discovery: served model, window, output reserve | `llm/endpoint_spec.py` |
| Per-model sampling envelopes and the reasoning effort | `llm/model_cards.py` |
| The chat-completions transport and its flight log | `llm/http.py` |
| The request shape a hosted OpenAI model takes | `llm/openai_wire.py` |
| Two reasoner turns per effort against the configured endpoint, before a run | `mission/endpoint_preflight.py` |
| An episode's LLM activity as one row block (`metrics.llm`) | `llm/flight_summary.py` |
| Mask records on the wire and on disk | `perception/mask_codec.py`, `perception/mask_region.py` |
