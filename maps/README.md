# maps

`maps` is the engine: one mission conversation in which a reasoning model
writes Python into one `run` tool, against a library of calls over a
world-state graph that holds the OSM prior and the segmenter's detections.
For the overview, see [How it works](../README.md#how-it-works). For one
mission followed through these packages, see
[`docs/architecture.md`](../docs/architecture.md).

## Packages

| Package | Contents | Read first |
|---|---|---|
| `mission/` | Conversation: the turn loop, the request each turn sends, the served text, the verbs a library call runs on, the per-episode world, compaction. | `mission/loop.py`, `mission/spec.py`, `mission/served_text.py` |
| `library/` | Python the model writes: the interpreter, the calls, rows with `.filter` and `.order`, result handles, the compiler from calls to ops. | `library/calls.py`, `library/interpreter.py`, `library/rows.py` |
| `query/` | Op program behind `find`, the cuts and sorts, `intersection`, `ends_of`, and `distance`; the surface it runs over; the tables a result serves. | `query/program.py`, `query/surface.py`, `query/result_tables.py` |
| `world_state/` | Graph of prior features and detected nodes, its build from a bake, the upserts a flight folds in, the context a query reads it through. | `world_state/graph.py`, `world_state/upsert.py`, `world_state/context.py` |
| `perception/` | Orthographic observation, SAM 3.1 client, watchlist, world-space tracking, seen colour, sensor geometry. | `perception/observation.py`, `perception/sam/detect_client.py`, `perception/detection.py` |
| `render/` | `look` crop, `show_map` schematic, and the marks drawn on them. | `render/look_render.py`, `render/map_render.py` |
| `llm/` | Endpoint discovery, per-model sampling cards, the chat-completions transport, the hosted OpenAI wire. | `llm/model_cards.py`, `llm/endpoint_spec.py`, `llm/http.py` |
| `settings.py` | Run settings, read from the environment once. | |
| `xy_token.py` | `[x, y]` point grammar that every slot taking a fid or a point reads. | |

## Boundaries

- `maps` imports nothing from `benchmarks`. A benchmark or the flight harness
  hands the loop a `MissionSpec` (`mission/spec.py`) whose `fly` executes one
  move (resolve a stop, observe, segment, fold the detections into the graph)
  and returns the flight report the model reads.
- `maps` reads the environment in one place, `settings.py`. An entry point
  builds a `Settings` (endpoints, sampling) once and hands it to
  `run_mission`, with the mission's own directory and routing key, so two
  missions in one process keep their own settings and directory.

To print the served contract (the system turn and the tool schema the model
receives):

```sh
python -m maps.mission.contract --sensor oryx_orx32_25mm_nadir   # --agl-m <m>, default configs/calcity.yaml's 914.4
```

## Module index

Paths are relative to `maps/`.

| Concept | Module |
|---|---|
| Mission conversation: one persistent reasoner, one served tool, the turn loop, operator taskings | `mission/loop.py` |
| Verbs a library call runs on: grounding and running a program, flying, pictures, the watchlist | `mission/program_verbs.py`, `mission/fly_verbs.py`, `mission/picture_verbs.py`, `mission/watch_verbs.py`, `mission/surface_state.py`, `mission/acks.py` |
| Turns that aren't a reply and the nudge each draws; request failures that end a mission | `mission/reply_repair.py`, `mission/serve_refusals.py` |
| Result handles: the op-program runner, each handle re-derived over the current surface | `library/result_handles.py` |
| What a look and a map render, and the captions their pictures carry | `mission/picture_requests.py` |
| Stops of an explore over a place or a set | `mission/explore_stops.py` |
| Request each turn sends: model, messages, tools, `max_tokens` | `mission/reasoner.py` |
| Served text: system turn, the `run` tool's description and schema | `mission/served_text.py` |
| Printing the served contract | `mission/contract.py` |
| Library the model writes against: executing a block, the calls, rows, `.filter` and `.order` | `library/interpreter.py`, `library/block_scan.py`, `library/block_ceilings.py`, `library/calls.py`, `library/call_arguments.py`, `library/rows.py` |
| Compiling library calls onto ops | `library/op_compiler.py` |
| Classes the segmenter watches | `perception/watchlist.py`, `mission/episode_world.py` (`declare_mission_concepts`) |
| Surface the op program runs over: one row per graph node, its served point, mask, and colour fields | `query/surface.py` |
| Op program behind `find`, `intersection`, `ends_of`, `distance` | `query/program.py`, `query/op_io.py`, `query/op_find.py`, `query/op_roads.py`, `query/op_cut_and_sort.py`, `query/op_compare.py`, `query/spatial_math.py`, `query/color_words.py`, `query/row_geometry.py` |
| What a result serves: the table, its legend, census, and hints | `query/result_tables.py` |
| Matching a query to rows by name | `query/text_match.py` |
| Transcript compaction when the conversation outgrows the window | `mission/compactor.py` |
| Per-episode world: the one graph a mission reads and each flight's detections fold into | `mission/episode_world.py` |
| `look` crop and `show_map` schematic | `render/look_render.py`, `render/map_render.py` |
| Marks drawn on a picture, and the map's label rule and colours | `render/mark_overlay.py`, `render/osm_palette.py` |
| Run settings, read from the environment once | `settings.py` |
| `[x, y]` point grammar that every slot taking a fid or a point reads | `xy_token.py` |
| World-state graph: map features and detected nodes, their upsert, the context a query reads it through | `world_state/` |
| Orthographic observation: a pose to a cropped window of the raster | `perception/observation.py`, `perception/ortho/` |
| Sensor specs (the C-12 Oryx, the CityNav nadir camera) and pixel-to-world projection | `perception/geometry/` |
| SAM 3.1 client and the classes it is asked for | `perception/sam/detect_client.py` |
| Detections stamped onto the tile grid | `perception/ortho/detection_stamping.py` |
| World-space tracking of detections across flights | `perception/world_tracker.py` |
| Seen colour of a detection | `perception/color.py` |
| Catalog-prior footprints projected onto the ortho crop | `perception/ortho/prior_overlay.py` |
| Endpoint discovery: served model, window, output reserve | `llm/endpoint_spec.py` |
| Per-model sampling envelopes and the reasoning effort | `llm/model_cards.py` |
| Chat-completions transport and its flight log | `llm/http.py` |
| Request shape a hosted OpenAI model takes | `llm/openai_wire.py` |
| Two reasoner turns per effort against the configured endpoint, before a run | `mission/endpoint_preflight.py` |
| LLM activity of an episode as one row block (`metrics.llm`) | `llm/flight_summary.py` |
| Mask records on the wire and on disk | `perception/mask_codec.py`, `perception/mask_region.py` |
