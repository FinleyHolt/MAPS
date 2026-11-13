# station

The station is a web console for reviewing a bench's finished runs and for
flying one mission from the seat. `serve.py` builds one bench from the
environment at startup, holds the mission in the seat, and serves the console
(`web/`) from one port. The console is a browser page and holds no state.

| Bench | Review | Fly from the seat |
|---|---|---|
| CityNav (`--bench citynav`, the default) | Any run root: transcript, pictures the model was sent, report, path, stops, target, commit, and score. | A row of a split, through the bench's runner: the same episode build, mission session, replay, and scorer as `python -m benchmarks.citynav.run_benchmark`. |
| Cal City (`--bench calcity`) | The same, plus the recorded track under a scrubber and each waypoint's observation. | A row, or your own words, on the kinematic twin (`benchmarks.flight`). |

## Review CityNav runs

1. Install the station extra:

   ```sh
   pip install -e '.[station]'
   ```

2. Start the station on a run root or a directory of run roots:

   ```sh
   python -m station.serve --out output/citynav
   ```

3. Open `http://127.0.0.1:8788`.

4. On a mission, click the run count, then the run.

A run root is a folder that holds `episodes.jsonl`, as
`python -m benchmarks.citynav.bench_fullset --out-root <root>` writes it. See
[`benchmarks/citynav`](../benchmarks/citynav/README.md). Each ledger row with
a `workdir` is a run, listed under its mission with its `SR` and `NE`.

The review reads the following:

| Shown | Source |
|---|---|
| Conversation: each `run` block's reasoning, code, and printed receipt | `<workdir>/conversation.jsonl` |
| Pictures the model was sent | `<workdir>/frames/<sha>.jpg`, named by each transcript row |
| Model's report | `<workdir>/flight/llm/transcript.jsonl`, the wire record |
| Path on the map | The row's `visited_xy` |
| Numbered stops | `<workdir>/flight/tour_cursor.json` |
| Target footprint and centre, commit point | The split row and `objects.json` when they're on disk; the row's `target_xy` and `commit_xy` always |
| Score | The row's `metrics` |
| Turns and flights, in the header | The row's own counts |

A bench run places the aircraft at each stop and doesn't fly between stops,
so the run has no `flight_track.jsonl` and no `observations/` folder. The
review draws `visited_xy` as the path, shows no scrubber, and pages the
pictures the model was sent in the picture book. A run folder copied without
its `frames/` reviews as text.

A review needs the following data:

| To show | Requirement |
|---|---|
| Run list, transcript, pictures, report, score | The run root alone. No CityNav data, model, or GPU. |
| Run placed on the map (path, stops, target, commit) | The block's bake under `SEARCH_CITYNAV_TILES` (default `tiles/citynav_2m_osmv3`) and the shipped OSM cache (`tiles/osm_cache`). |
| Basemap | The block's raster pair, `<block>.tif` and `<block>.png`, under `SEARCH_CITYNAV_RGBD_DIR`; or `--basemap`. |
| Target footprint; mission list in split order | The split JSONs and `objects.json` under `CITYNAV_ROOT`. Without them, every run lists by its own words and the target draws as its centre. |

A station that can't fly (no raster, or missing data) is a reviewing station:
it disables the chat box and says how to open its runs. For the commands that
fetch the CityNav data, the rasters, and the bake, see
[`docs/data.md`](../docs/data.md#citynav).

## Fly a CityNav episode

Flying needs what a bench run needs: the CityNav data, the block's raster
pair and bake ([`docs/data.md`](../docs/data.md#citynav)), and a reasoner and
a segmenter named by the environment
([`docs/endpoints.md`](../docs/endpoints.md)).

1. Start the station. Runs land in `output/station`:

   ```sh
   python -m station.serve
   ```

2. Open `http://127.0.0.1:8788`.

3. Pick a split in the mission list's menu.

4. Click a row. A CityNav mission is always a row of a split; words with no
   row picked start nothing.

5. Optional: edit the row's words. The station flies the same row (its start,
   target, and score) under your words.

6. Send. The conversation streams to the console.

7. When the model reports, send a tasking to continue, or click `+` to end
   the mission.

The station flies the row as the bench's runner does: the bench's episode
build and mission session, each flight a placement at the stop with the
raster cropped there, then the six-action replay and the bench's scorer. It
differs from a bench run in the following ways:

- The model's reply is a report, and the mission holds for your tasking. A
  tasking is one more turn of the same conversation
  (`maps.mission.loop.continue_mission`), with every name, flown stop, and
  watched class the model already has.
- A tasking sent while the model flies lands on its next turn.
- `+` ends the mission. The bench scores it on every flight flown, and the
  run lands in `<out>/<UTC stamp>/` as its own run root: `run_manifest.json`,
  `episodes.jsonl` with the scored row, and the attempt folder.
- A mission ended while a turn is in flight is scored on the flights flown by
  then, with `ended_by: stopped`.
- The row lands when the mission ends. A station killed while a mission holds
  at its report leaves the attempt folder and no row. Ctrl-C ends the mission
  first.

## Run the station on Cal City

To review only:

```sh
SEARCH_CALCITY_TILES=<bake> python -m station.serve --bench calcity --out <runs> --basemap <picture>
```

`<bake>` is the tile bake the runs were flown on; the example runs carry
theirs as `bake/`.

To fly missions on the twin and review them:

```sh
python -m station.serve --bench calcity --ortho <raster dir>
```

With `--ortho`, the station flies a mission on the twin
(`benchmarks.flight.mission.Mission` over `benchmarks.flight.twin.Twin`):

- A mission is a row of the Cal City split, or your own words from the map's
  centre, unscored.
- A flight is a leg flown for an area. Each waypoint's observation lands in
  the picture book.
- A report holds for your tasking, as on CityNav. The row is banked at each
  report.

Flying needs the Cal City bake and raster pair
([`docs/data.md`](../docs/data.md#cal-city)) and the two endpoints. A run
flown by `python -m benchmarks.flight.twin --out <out>` reviews the same way.
For the twin and the run's records, see [`docs/flight.md`](../docs/flight.md).

## Flags

| Flag | Description |
|---|---|
| `--bench citynav\|calcity` | Bench whose catalog, maps, and runs the station serves. Default `citynav`. |
| `--out <dir>` | Where runs land and are reviewed from: one run root, or a directory of them. Default `output/station`. |
| `--config <yaml>` | Run config. Default `configs/<bench>.yaml`. |
| `--basemap <picture or dir>` | North-up picture of a map's whole bake extent, or a directory of them named `<map>.jpg` or `<map>.png`: the basemap of a map with no raster pair, and the first choice of a reviewing station. |
| `--osm-cache <dir>` | OSM caches that place a CityNav block. Default `tiles/osm_cache`. |
| `--ortho <dir>` | Cal City: raster directory that the twin's frames and the basemap are cut from. Unset, the station reviews runs only. |
| `--register <dir>` | Cal City: raster directory that each frame is registered against. Unset, nothing is registered. |
| `--host`, `--port` | Listening address. Default `127.0.0.1:8788`. |
| `--memory-gib <GiB>` | Process's memory cap, GiB: a finite number between 1 and 2^33. Default 16. A block that allocates past it raises `MemoryError`, served as the block's error. |

The station has no login. `run/file` and every path a request names are
confined to `--out`. A map name is a file stem and nothing else. The station
runs its missions under the isolation its host allows; see
[Security](../README.md#security).

## Console

| Part | Contents |
|---|---|
| Mission list | Split's rows, searchable, with a menu of the bench's splits. The list opens on the split of the newest run under `--out`. Rows that have runs list first. A run with no row in the split shown lists ahead of them, by its own words. A row's words go into the chat box; send flies it. A row with finished runs expands to them, each with its score; a run opens as a review. |
| Header | Mission's turns and flights. A review shows the run's own. |
| Transcript | Conversation as it happens: each `run` block's reasoning, code, and printed receipt, the pictures the model was sent, and the report. On the twin, each leg posted and what the pass held. |
| Chat box | Speaks to the mission in the seat: a tasking lands on the model's next turn, and an answer wakes a mission holding at its report. `+` ends the mission and empties the seat. Disabled on a reviewing station. |
| Map | Viewed map's basemap, the aircraft and its trail, the target's footprint or centre, each stop, and the commit of a scored run. A review frames the run on the basemap. On the twin, each leg's area as a mask of the points the recording holds. |
| Picture book | Cal City: each waypoint's observation as the segmenter received it, with the map's features drawn over it on request. A bench run: the pictures the model was sent, in order. |
| Scrubber | In a review of a run with a recorded track: the aircraft along it. |

## Maps

CityNav has one map per block. The mission list, basemap, working area,
prior, and every placement follow the map of the mission on screen: the run
under review, else the row about to fly, else the mission in the seat. When
that map changes, the console requests the map's `location`, `area`, and
prior again.

| Per map | Source |
|---|---|
| Georeference (local metres to WGS84 and back) | Cal City: the bake's own `georef` (`benchmarks/flight/tile_georef.py`). CityNav: the city's shipped OSM cache, `tiles/osm_cache/<city>_osm_local.geojson`, whose `citynav_osm_cache.georef` states `osm_crs` and `citynav_from_osm_affine`; `placement.py` inverts that transform at run time. The station reads the CityNav bake and its files and never changes them. |
| Basemap | Web-mercator tiles cut from the map's raster pair, loaded on first use (three maps held at a time), else from the `--basemap` picture; a reviewing station tries the picture first. The raster pair lies under `--ortho` on Cal City and under the bench's raster directory on CityNav. The first view of a CityNav block builds the bench's ortho cache beside the rasters, which takes a few seconds. A reviewing station under bubblewrap reads that cache read-only, so a block whose cache is not built loads from its rasters in memory on each first view. |
| Prior | The map's `<map>_features.json` under the bake. |

A map with no georeference reviews with its transcript, pictures, report, and
score, and nothing placed on the map. The map's caption says so.

## Requests

The console calls `serve.py` through these requests, all relative to the
page.

| Request | Response |
|---|---|
| `GET station` | Bench: `bench`, `split` (the split the mission list opens on), `splits`, `map` (the default map), `flies`, `why` (when it doesn't), `own_words`. |
| `GET missions?q=&limit=&split=` | Catalog of a split (default: the split the list opens on): `episode_id`, `instruction`, `map`, `target_lonlat`. 404 for a split the bench doesn't have. |
| `GET missions/runs` | Finished runs under `--out`, from the ledgers alone: path, status, score, and counts, with no transcript. |
| `GET replay?path=` | Run's state document with the transcript, `flights`, `target_lonlat`, `target_point`, `commit_lonlat`, and `score`, plus `track` and `duration_s` (null on a run with no recorded track, whose `track` is its `visited_xy`). |
| `GET mission/events` | Server-sent events: seat's whole state document each time it changes, `null` with the seat empty. |
| `POST mission/next` `{episode_id?, instruction, reasoning_effort?, split?}` | Flies a catalog row; on the twin, your own words too. 409 while a mission is live or on a station that flies nothing. |
| `POST mission/tasking` `{text}` | Speaks to the live mission. |
| `POST mission/clear` | Ends the mission and empties the seat. A bench episode is scored before the response. |
| `GET reasoner` | Served model and the reasoning efforts its card takes. |
| `GET mission/ownship[?path=&t=]` | Seat's aircraft, or a reviewed run's at mission second `t` (null on a run with no recorded track). |
| `GET location?map=`, `GET tiles/{map}/{z}/{x}/{y}.png`, `GET area?map=` | Map's basemap manifest, its web-mercator tiles, and the bake's extent. 404 for a map that isn't placed. |
| `GET mission/prior.json?map=` | Map's bake features in local metres, grouped for drawing. |
| `GET run/file?path=` | One file of a run, by `<root>/<workdir>/<name>`. |

A run's `path` is `<root>/<workdir>`: the root's folder name under `--out`
(`.` when `--out` is itself a root) and the ledger row's `workdir`.

## Modules

| Module | Contents |
|---|---|
| `serve.py` | Station: the catalog, the runs, the seat, the basemap tiles, and the requests. |
| `bench_seat.py` | One bench episode in the seat, flown through the engine's `fly_attempt` with the conversation held open for the operator. |
| `placement.py` | A map's local metres to WGS84 and back, from its bake's georef or its city's OSM cache. |

To run the console over your own aircraft, implement
`benchmarks.flight.vehicle.Vehicle` and hand it to `open_live` where `Station`
builds the twin.

## Build the console

`web/dist` is committed, so the station needs no Node. After you change
`web/app`, rebuild:

```sh
cd station/web && npm ci && npm run check && npm test && npm run build
```
