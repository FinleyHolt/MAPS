# station

A web console over one mission flown on the twin, and over every finished
run. `serve.py` holds the mission (`benchmarks.flight.mission.Mission` over
`benchmarks.flight.twin.Twin`), publishes its state and serves the console
from one port; the console (`web/`) is a browser page and holds no state of
its own.

## Run it

```sh
pip install -e '.[station]'
python -m station.serve --ortho <raster dir>     # fly Cal City missions on the twin, review them after
python -m station.serve                          # review the runs under --out
```

Then open `http://127.0.0.1:8788`. Flying a mission needs what the twin
needs: the Cal City bake and raster pair (the top-level README's
[Cal City](../README.md#cal-city) section) and a reasoner and a segmenter
named by the environment ([Endpoints](../README.md#endpoints)). Reviewing
needs none of them; with the bake on disk the review places the run on the
map, and with a raster it draws the basemap.

| Flag | What it sets |
|---|---|
| `--ortho <dir>` | The raster directory the twin's frames and the basemap tiles are cut from. Unset, the station reviews runs only. |
| `--basemap <image>` | A north-up picture of the bake's whole extent, for a reviewing station's map. `--ortho` replaces it. |
| `--register <dir>` | The raster directory each frame is registered against. Unset registers nothing. |
| `--out <dir>` | Where runs land and are reviewed from (default `output/station`). |
| `--config <yaml>` | The flight config (default `configs/calcity.yaml`). |
| `--host`, `--port` | Where the station listens (default `127.0.0.1:8788`). |

Each mission lands in `<out>/<UTC stamp>/` with the run records a flown run
carries ([`benchmarks/flight`](../benchmarks/flight/README.md#the-runs-records)).
A run flown by `python -m benchmarks.flight.twin --out <out>`, or a run
folder copied in from another machine, reviews the same way.

## The console

| Part | What it shows |
|---|---|
| Mission list | The Cal City taskings, searchable. A row's words go into the box; send flies it. Words with no row picked fly as your own tasking from the map's centre, unscored. A row with finished runs expands to them; a run opens as a review. |
| Transcript | The conversation as it happens: each `run` block's reasoning, code and printed receipt, the pictures the model was sent, each leg posted and what the pass held, and the reply. |
| Chat box | Speaks to the mission in the seat: a re-task lands on the model's next turn, and an answer wakes a mission holding at its report. `+` stops the mission and empties the seat. |
| Map | The raster, the aircraft and its trail, the target's footprint, each flight's stop, and each leg's area as a mask of the points the recording holds. |
| Picture book | Each waypoint's observation as the model's segmenter received it, with the map's features drawn over it on request. |
| Scrubber | Under review: the aircraft along the recorded track. |

## The requests

The console speaks to `serve.py` through these, all relative to the page:

| Request | Answer |
|---|---|
| `GET missions?q=&limit=` | The catalog: `episode_id`, `instruction`, `map`, `target_lonlat`. |
| `GET mission/events` | Server-sent events: the seat's whole state document each time it changes, `null` with the seat empty. |
| `POST mission/next` `{episode_id?, instruction, reasoning_effort?}` | Flies a catalog row, or your own words. 409 while a mission is live. |
| `POST mission/tasking` `{text}` | Speaks to the live mission. |
| `POST mission/clear` | Stops the mission and empties the seat. |
| `GET reasoner` | The served model and the reasoning efforts its card takes. |
| `GET missions/runs` | Every finished run under `--out`. |
| `GET replay?path=` | One run: its state document, `duration_s` and `track`. |
| `GET mission/ownship[?path=&t=]` | The twin's pose now, or a reviewed run's at mission second `t`. |
| `GET location`, `GET tiles/{z}/{x}/{y}.png`, `GET area` | The basemap manifest, its web-mercator tiles cut from the raster, and the bake's extent. |
| `GET mission/prior.json` | The bake's features in local metres, grouped for drawing. |
| `GET run/file?path=` | One file of a run. |

A station for your own aircraft keeps this console: implement
`benchmarks.flight.vehicle.Vehicle` for the aircraft and hand it to
`open_live` where `Station.start` builds the twin.

## Build the console

`web/dist` is committed, so the station needs no Node. After changing
`web/app`:

```sh
cd station/web && npm ci && npm run check && npm test && npm run build
```
