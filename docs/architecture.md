# Architecture

The top-level README's [How it works](../README.md#how-it-works) section is
the short form. This page follows one mission through the code, then says
where another system connects. [`maps/README.md`](../maps/README.md) lists
every module.

## The layers

| Layer | Package | Holds |
|---|---|---|
| Mission | `maps/mission/` | The conversation: the reasoner, the turn loop, the served text, the verbs a library call runs on, compaction. |
| Library | `maps/library/` | The Python the model writes: the interpreter, the calls, rows and result handles, the compiler from calls to ops. |
| Query | `maps/query/` | The op program behind `find`, the cuts and sorts, `intersection`, `ends_of`, `distance`, and the tables a result serves. |
| World state | `maps/world_state/` | The graph of prior features and detections, and the upserts a flight folds into it. |
| Perception | `maps/perception/` | The orthographic observation, the SAM 3.1 client, the watchlist, tracking and colour. |
| Render | `maps/render/` | The `look` crop and the `show_map` schematic. |
| LLM | `maps/llm/` | Endpoint discovery, model cards, the chat-completions transport. |

`maps` imports nothing from `benchmarks`. A bench or the flight harness
hands the mission loop a `MissionSpec` (`maps/mission/spec.py`) and the loop
calls back through it. `tests/test_layers.py` holds the import directions.

## One mission, step by step

1. **An entry point builds the settings and the bench.** `maps/settings.py`
   reads the environment once into a `Settings`; a bench entry point builds
   a `Bench` value (`benchmarks/bench.py`) that names the data roots, the
   success criterion and the episode engine.
2. **The episode is hydrated.** The engine
   (`benchmarks/citynav/engine.py`, `benchmarks/citynav/run_benchmark/episode_runner.py`) loads
   the block's bake into an `EpisodeWorld` (`maps/mission/episode_world.py`):
   one `ObjectGraph` seeded with a `MapObject` per baked OSM feature
   (`maps/world_state/prior_graph.py`).
3. **The conversation opens.** `run_mission` (`maps/mission/loop.py`) sends
   the system turn and the operator's words with one tool, `run`
   (`maps/mission/served_text.py`). The request is built by
   `maps/mission/reasoner.py` and sent by `maps/llm/http.py`.
4. **The model answers with a block of Python.** `MissionPython`
   (`maps/library/interpreter.py`) scans the block
   (`block_scan.py`, `block_ceilings.py`) and runs it in the mission's one
   namespace. The library calls (`calls.py`) are that namespace's builtins.
5. **A reading call compiles to ops.** `find`, a `.filter`, an `.order`,
   `intersection`, `ends_of` and `distance` compile
   (`maps/library/op_compiler.py`) onto the op program (`maps/query/program.py`),
   which runs over the surface (`maps/query/surface.py`): one row per graph
   node. The result is a handle (`maps/library/result_handles.py`) that is
   re-derived over the current surface on every read, so a result bound
   before a flight holds that flight's rows after it.
6. **A moving call flies.** `go_to` and `explore` call `MissionSpec.fly`
   once per stop (`maps/mission/fly_verbs.py`, `explore_stops.py`). The
   engine's `MissionSession.fly`
   (`benchmarks/citynav/run_benchmark/mission_engine.py`) resolves the stop,
   obtains an orthographic observation, sends it to the segmenter, folds the
   detections into the graph and returns the flight report.
7. **A picture call renders.** `look` crops the sensor's pixels at a row
   (`maps/render/look_render.py`); `show_map` draws the world state as a
   schematic (`maps/render/map_render.py`). Both return as images in the
   next request.
8. **The receipt goes back.** The tool result carries the block's printed
   lines, its last value as a table (`maps/query/result_tables.py`), each
   flight's report and any error with its line.
9. **The mission ends on a plain reply.** A reply with no tool call ends the
   conversation; the commit is the target of the last flight. A turn that
   is neither a reply nor a call draws a nudge (`maps/mission/reply_repair.py`).
   When the transcript outgrows the model's window, the compactor folds
   earlier turns (`maps/mission/compactor.py`).

## The structured world state

The world state is one graph (`maps/world_state/graph.py`). A node is a
`MapObject` keyed by its fid:

| Field | Prior node (`prior_<i>`) | Detected node (`detected_<i>`) |
|---|---|---|
| `provenance` | `prebaked` | `detected` |
| What it is | Its OSM `tags`, verbatim, and its `name` where OSM has one. | The segmenter's label (`class_name`). |
| Geometry | `world_contour` in world metres (a point, an open way or a closed ring) and its `cells` on the bake's 2 m grid. | The mask's footprint cells and contour, and the tracker's `world_xy`. |
| Seen attributes | Colour read from a flight's pixels over its footprint (`upsert_prior_percept`). | Colour read from its mask; `track_id` from the world tracker. |

The frame is local metres, x east and y north, with the bake's
`tile_origin_xy` and `cell_size_m` held by the `QueryContext`
(`maps/world_state/context.py`).

The prior enters once, at hydrate, from the block's bake
(`<block>_features.json`, written by `benchmarks/citynav/bake/city.py` from
an OpenStreetMap extract). Detections enter after every flight:
`maps/perception/sam/detect_client.py` returns masks for the watched
classes, `maps/perception/world_tracker.py` gives each a stable identity across
flights, and `maps/world_state/upsert.py` folds it onto the node carrying that
`track_id` or mints one.

The model never reads the graph directly. It reads the surface
(`maps/query/surface.py`), which serves each node as a row: `fid`, `xy`
(the point a flight to it targets), `name` and `tags` or `class`, the
colour fields, and the mask. Relating calls attach measured columns to the
rows they return (`distance to <place>`, `facing <place>`, `<dir> of
<place>`).

| Call | Reads | Extends |
|---|---|---|
| `find`, `.filter`, `.order`, `intersection`, `ends_of`, `distance` | Rows of the surface, prior and detected alike. | Nothing. |
| `look_for` | Nothing. | The watchlist the segmenter is asked for on every later flight (`maps/perception/watchlist.py`). |
| `go_to`, `explore` | The row or set flown to. | Detected nodes for every watched class in view; perceived colour on in-view priors. |
| `look` | The pixels flights have collected at a row. | Nothing: it returns a picture. |
| `show_map` | The graph around a place. | Nothing: it returns a picture. |

## Where another system connects

The world state is the interface between the reasoner and everything else,
and each side of it has one seam.

| You have | The seam | What you implement |
|---|---|---|
| An aircraft with a camera | `Vehicle` (`benchmarks/flight/vehicle.py`) | `post(leg)`, `pose(t)`, `index(since)`, `frame(seq)`, `now()`, `sleep(s)`. The flight harness does the rest. See [`flight.md`](flight.md#bring-your-own-aircraft). |
| A planner or autonomy stack that executes moves and returns imagery | `MissionSpec` (`maps/mission/spec.py`) | `fly(item)`: execute one move to `item["target_fid"]`, obtain an observation, and return the flight report. `MissionSession` in `mission_engine.py` is the reference: it segments the observation and folds the detections through `EpisodeWorld`. |
| Another detector | `build_episode`'s `perception_factory` (`benchmarks/bench.py`, `EpisodeEngine`) | An object with `set_concepts(classes)` and `detect(observation)` returning `DetectedObject`s (`maps/perception/detection.py`), as `SamDetectClient` does. |
| Another map source | The bake (`<block>_features.json`, `<block>_meta.json`) | Features with OSM-style tags and a `world_contour` in the block's local metres; `build_object_graph` (`maps/world_state/prior_graph.py`) seeds the graph from them. |
| Another reasoner | A model card (`maps/llm/model_cards.py`) | A card for the served model name on an OpenAI-compatible endpoint ([`endpoints.md`](endpoints.md)). |

## Two ways a flight happens

| | Bench | Flight |
|---|---|---|
| Entry | `benchmarks/citynav/bench_fullset.py`, `benchmarks/calcity/bench_fullset.py` | `benchmarks/flight/mission.py` (`Mission`), driven by the twin, the replay or the station |
| A move | The pose is set to the stop. | A `Leg` is posted to the `Vehicle`; the leg has arrived when the recording holds the area it was flown for. |
| The observation | A north-up crop of the block's raster under the pose (`maps/perception/observation.py`). | Camera frames fused into one north-up orthophoto (`benchmarks/flight/live_source.py`). |
| After it | The same path: segment, fold, report. | The same path: segment, fold, report. |
| Scoring | `benchmarks/citynav/eval.py`: CityNav replays the flights as its six discrete actions and scores the final pose within 20 m; Cal City scores the commit point inside the target's footprint. | `benchmarks/flight/measures.py` over the run's records. |

Both write the same run record (`benchmarks/run_record/`): a ledger row per
attempt, the conversation, the detections and the LLM transcript.
