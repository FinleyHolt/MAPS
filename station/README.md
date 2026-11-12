# station

A web console over a bench's finished runs and over one mission flown from
the seat. `serve.py` builds one bench value from the environment when it
starts (`--bench citynav`, the default, or `--bench calcity`), holds the
mission in the seat, publishes its state and serves the console from one
port. The console (`web/`) is a browser page and holds no state of its own.

| Bench | Review | Fly from the seat |
|---|---|---|
| CityNav | Any run root: the transcript with the pictures the model was sent, its report, the path, each stop, the target, the commit and the score. | A row of a split, through the bench's own runner: the same episode build, mission session, replay and scorer as `python -m benchmarks.citynav.run_benchmark`. |
| Cal City | The same, plus the recorded track under a scrubber and each waypoint's observation. | A row, or your own words, on the kinematic twin (`benchmarks.flight`). |

## Run it on CityNav

```sh
pip install -e '.[station]'
python -m station.serve --out output/citynav          # review bench run roots
python -m station.serve                               # fly episodes into output/station, review them after
```

Then open `http://127.0.0.1:8788`.

### Review a bench root

`--out` names one run root or a directory of them. A run root is a folder
holding `episodes.jsonl`, as `python -m benchmarks.citynav.bench_fullset
--out-root <root>` writes it ([`benchmarks/citynav`](../benchmarks/citynav/README.md)).
Each ledger row with a `workdir` is a run in the list, under its mission, with
its `SR` and `NE` on the row. A run opens as a review:

| What the review shows | Where it is read from |
|---|---|
| The conversation, each `run` block's reasoning, code and printed receipt | `<workdir>/conversation.jsonl` |
| The pictures the model was sent | `<workdir>/frames/<sha>.jpg`, named by each transcript row |
| The model's report | `<workdir>/flight/llm/transcript.jsonl`, the wire record |
| The path, drawn on the map | The row's `visited_xy` |
| Each stop, numbered | `<workdir>/flight/tour_cursor.json` |
| The target's footprint and centre, the commit point | The split row and `objects.json` when they are on disk; the row's `target_xy` and `commit_xy` always |
| The score | The row's `metrics` |

A bench run's aircraft is placed at each stop and never flown between them,
so the run has no `flight_track.jsonl`. The review draws `visited_xy` as the
path and shows no scrubber. A bench run has no `observations/` folder: the
picture book pages the pictures the model was sent, in the order it was sent
them, and the transcript shows each beside the turn that asked for it. A run
folder copied without its `frames/` reviews as text.

What a review needs:

| For | Needed |
|---|---|
| The run list, the transcript, the pictures, the report, the score | The run root alone. No CityNav data, model or GPU. |
| The run placed on the map (path, stops, target, commit) | The block's bake under `SEARCH_CITYNAV_TILES` (default `tiles/citynav_2m_osmv3`) and the shipped OSM cache (`tiles/osm_cache`). |
| The basemap under it | The block's raster pair, `<block>.tif` and `<block>.png`, under `SEARCH_CITYNAV_RGBD_DIR`; or `--basemap`. |
| The target's footprint; the mission list ordered as the split | The split JSONs and `objects.json` under `CITYNAV_ROOT`. Without them every run lists by its own words and the target draws as its centre. |

[`docs/data.md`](../docs/data.md#citynav) gives the commands that fetch the
CityNav data, the rasters and the bake.

### Fly an episode

Pick a split in the mission list's menu, click a row and send. The station
flies that row as the bench's runner flies it: the bench's episode build and
mission session, each flight a placement at the stop with the raster cropped
there, then the six-action replay and the bench's scorer. The conversation
streams to the console as it happens. Editing the row's words before sending
flies the same row (its start, its target, its score) under your words.

One thing differs from a bench run. The model's reply is a report: the
mission holds for your tasking, and a tasking is one more turn of the same
conversation (`maps.mission.loop.continue_mission`), with every name, flown
stop and watched class the model already has. A tasking sent while the model
flies lands on its next turn. `+` ends the mission. The bench then scores it
on every flight flown and the run lands in `<out>/<UTC stamp>/` as a run root
of its own: `run_manifest.json`, `episodes.jsonl` with the scored row, and
the attempt folder. A mission ended while a turn is in flight is scored on
the flights flown by then, with `ended_by: stopped`.

The row lands when the mission ends. A station killed while a mission holds
at its report leaves the attempt folder and no row; stopping the station
with Ctrl-C ends the mission first.

Flying needs what a bench run needs: the CityNav data, the block's raster
pair and bake ([`docs/data.md`](../docs/data.md#citynav)), and a reasoner and
a segmenter named by the environment
([`docs/endpoints.md`](../docs/endpoints.md)). A CityNav mission is always a
row of a split; words with no row picked start nothing.

## Run it on Cal City

```sh
python -m station.serve --bench calcity --ortho <raster dir>     # fly missions on the twin, review them after
python -m station.serve --bench calcity --out <runs> --basemap <picture>     # review only
```

With `--ortho` the station flies a mission on the twin
(`benchmarks.flight.mission.Mission` over `benchmarks.flight.twin.Twin`): a
row of the Cal City split, or your own words from the map's centre,
unscored. A flight is a leg flown for an area, and each waypoint's
observation lands in the picture book. A report holds for your tasking as
above, and the row is banked at each report. Flying needs the Cal City bake
and raster pair ([`docs/data.md`](../docs/data.md#cal-city)) and the two
endpoints. [`docs/flight.md`](../docs/flight.md) describes the twin and the
run's records. A run flown by `python -m benchmarks.flight.twin --out <out>`
reviews the same way.

## Flags

| Flag | What it sets |
|---|---|
| `--bench citynav\|calcity` | The bench whose catalog, maps and runs the station serves (default `citynav`). |
| `--out <dir>` | Where runs land and are reviewed from: one run root, or a directory of them (default `output/station`). |
| `--config <yaml>` | The run config (default `configs/<bench>.yaml`). |
| `--basemap <picture or dir>` | For a map with no raster: a north-up picture of its bake's whole extent, or a directory of them named `<map>.jpg` or `<map>.png`. |
| `--osm-cache <dir>` | The OSM caches a CityNav block is placed by (default `tiles/osm_cache`). |
| `--ortho <dir>` | Cal City: the raster directory the twin's frames and the basemap are cut from. Unset, the station reviews runs only. |
| `--register <dir>` | Cal City: the raster directory each frame is registered against. Unset registers nothing. |
| `--host`, `--port` | Where the station listens (default `127.0.0.1:8788`). |

The station listens on `127.0.0.1` and has no login. `run/file` and every
path a request names are confined to `--out`; a map name is a file stem and
nothing else.

## Many maps

CityNav has one map per block. The mission list, the basemap, the working
area, the prior and every placement follow the map of the mission on screen:
the run under review, else the row about to fly, else the mission in the
seat. The console asks for the map's `location`, `area` and prior again when
that map changes.

| Per map | Source |
|---|---|
| The georeference (local metres to WGS84 and back) | Cal City: the bake's own `georef` (`benchmarks/flight/tile_georef.py`). CityNav: the city's shipped OSM cache, `tiles/osm_cache/<city>_osm_local.geojson`, whose `citynav_osm_cache.georef` states `osm_crs` and `citynav_from_osm_affine`; `placement.py` inverts that transform at run time. The CityNav bake and its files are read and never changed. |
| The basemap | Web-mercator tiles cut from the map's raster pair under the bench's raster directory, loaded on first use (three maps held at a time); else the `--basemap` picture. The first view of a CityNav block builds the bench's own ortho cache beside the rasters, which takes a few seconds. |
| The prior | The map's `<map>_features.json` under the bake. |

A map with no georeference reviews with its transcript, pictures, report and
score, and nothing placed on the map; the map's caption says so.

## The console

| Part | What it shows |
|---|---|
| Mission list | The split's rows, searchable, with a menu of the bench's splits. A row's words go into the box; send flies it. A row with finished runs expands to them, each with its score; a run opens as a review. A run with no row in the split shown lists first, by its own words. |
| Transcript | The conversation as it happens: each `run` block's reasoning, code and printed receipt, the pictures the model was sent, and the report. On the twin, each leg posted and what the pass held. |
| Chat box | Speaks to the mission in the seat: a tasking lands on the model's next turn, and an answer wakes a mission holding at its report. `+` ends the mission and empties the seat. |
| Map | The viewed map's basemap, the aircraft and its trail, the target's footprint or centre, each stop, and the commit of a scored run. On the twin, each leg's area as a mask of the points the recording holds. |
| Picture book | Cal City: each waypoint's observation as the model's segmenter received it, with the map's features drawn over it on request. A bench run: the pictures the model was sent, in order. |
| Scrubber | Under review of a run with a recorded track: the aircraft along it. |

## The requests

The console speaks to `serve.py` through these, all relative to the page:

| Request | Answer |
|---|---|
| `GET station` | The bench: `bench`, `split` (the default), `splits`, `map` (the default map), `flies`, `why` (when it does not), `own_words`. |
| `GET missions?q=&limit=&split=` | The catalog of a split (default the bench's): `episode_id`, `instruction`, `map`, `target_lonlat`. 404 for a split the bench does not have. |
| `GET missions/runs` | Every finished run under `--out`, off the ledgers alone: its path, status, score and counts, no transcript. |
| `GET replay?path=` | One run: its state document with the transcript, `flights`, `target_lonlat`, `target_point`, `commit_lonlat` and `score`, plus `track` and `duration_s` (null on a run with no recorded track, whose `track` is its `visited_xy`). |
| `GET mission/events` | Server-sent events: the seat's whole state document each time it changes, `null` with the seat empty. |
| `POST mission/next` `{episode_id?, instruction, reasoning_effort?, split?}` | Flies a catalog row; on the twin, your own words too. 409 while a mission is live or on a station that flies nothing. |
| `POST mission/tasking` `{text}` | Speaks to the live mission. |
| `POST mission/clear` | Ends the mission and empties the seat. A bench episode is scored before this answers. |
| `GET reasoner` | The served model and the reasoning efforts its card takes. |
| `GET mission/ownship[?path=&t=]` | The seat's aircraft now, or a reviewed run's at mission second `t` (null on a run with no recorded track). |
| `GET location?map=`, `GET tiles/{map}/{z}/{x}/{y}.png`, `GET area?map=` | A map's basemap manifest, its web-mercator tiles, and the bake's extent. 404 for a map that is not placed. |
| `GET mission/prior.json?map=` | A map's bake features in local metres, grouped for drawing. |
| `GET run/file?path=` | One file of a run, by `<root>/<workdir>/<name>`. |

A run's `path` is `<root>/<workdir>`, the root's folder name under `--out`
(`.` when `--out` is itself a root) and the ledger row's `workdir`.

## The modules

| Module | What it holds |
|---|---|
| `serve.py` | The station: the catalog, the runs, the seat, the basemap tiles and the requests. |
| `bench_seat.py` | One bench episode in the seat, flown through the engine's `fly_attempt` with the conversation held open for the operator. |
| `placement.py` | A map's local metres to WGS84 and back, off its bake's georef or its city's OSM cache. |

A station for your own aircraft keeps this console: implement
`benchmarks.flight.vehicle.Vehicle` for the aircraft and hand it to
`open_live` where `Station` builds the twin.

## Build the console

`web/dist` is committed, so the station needs no Node. After changing
`web/app`:

```sh
cd station/web && npm ci && npm run check && npm test && npm run build
```
