# benchmarks/flight

This package is the flight harness: the method half of the station that flew
MAPS on a C-12 over California City on six days from 11 to 18 September 2026. It hydrates
a mission as the bench does, flies each flight as a leg on an aircraft, and
fuses each observation from the frames the aircraft recorded. It reaches the
aircraft through one protocol, `Vehicle`. The package also holds a kinematic
twin of the aircraft and the closed-loop measures.

This README is the reference. For a short guide to the twin and an aircraft
of your own, see [`docs/flight.md`](../../docs/flight.md).

A mission receives its bench as a value (`benchmarks.bench.Bench`). The twin
and the measures build the Cal City bench from the environment at startup
(`benchmarks.calcity.paths.calcity_bench`), and the mission reaches the
episode engine through it.

## Flown chain

One flight runs through these steps, from the model's `go_to` or `explore`
stop to the observation it reads:

1. **Stop to leg.** `Legs.fly` (`legs.py`) resolves the stop to a
   point, takes the area the leg is flown for (`leg_area`: the target's
   footprint, or a disc about a point), and asks the recording whether it
   holds that area already. When it does, no leg is flown.
2. **Leg to aircraft.** `Vehicle.post(Leg)` (`vehicle.py`) hands the point,
   the rest of the pass, and the area to the aircraft. On the C-12, the crew
   flew it off the cockpit display.
3. **Frame registration.** The corrector (`LiveSource.start_corrector`,
   `live_source.py`) reads frames off `Vehicle.index` and `Vehicle.frame`. A
   simulated vehicle registers every frame up to its clock; a real one
   registers the newest each `CORRECTOR_PERIOD_S`, skipping to it when
   behind. The corrector places each frame by the vehicle's pose at its
   stamp, with the timing offset applied (`nav_track.py`), through the
   boresight. It registers the frame against the map rendered at that pose
   (`register_frame`, `frame_offset.py`, `map_render.py`, `locator.py`) and
   ties it to the frame before it in the pose graph (`pose_graph.py`,
   `frame_ties.py`, `tie_adjust.py`).
4. **Arrival by coverage.** `LiveSource.coverage` reads the ring's index (one
   pose per frame, no pixels; `footprint.py`) and reports the share of the
   area's points that some frame holds. The leg has arrived at 90%
   (`legs.COVERED_FRACTION`), or when one wait on it passes
   `ARRIVAL_TIMEOUT_S`, with what the pass held. A short pass asked again
   starts a new wait. When the index holds the area but none of its frames could be
   served, the leg is flown and only the frames recorded after it count.
5. **Fusion.** `LiveSource.observe_area` fetches, for each area point, the
   frame that saw it nearest nadir. It places each frame by its pose (as the
   corrector poses it; the frame's own tag where the vehicle has no pose at
   its stamp) and registration, composites them on one north-up grid
   (`composite`), registers the mosaic on the map as one similarity
   (`register_mosaic`), and hands it over as an orthographic observation on
   flat ground at the bake's floor (`ortho_observation`).
6. **Engine.** The observation goes through the bench's observation path:
   SAM 3.1 segments the watched classes, detections join the graph, and the
   flight report returns to the model. `run_imagery.py` keeps what the run
   captured for `look` to crop.

| Input | Source |
|---|---|
| Calibration (boresight, GNSS bias, timing offset) | `open_live`'s `boresight_path`, a `boresight.json` the integrator supplies, applied in `live_source.py` alone. The twin and the station pass none and fly the datasheet camera. |
| Antenna-to-camera geometry of the C-12 | `airframe.py` |
| Bake's local metres to WGS84 | `tile_georef.py` |
| Ground under each frame | The block's bare-earth surface where the bake carries one: `<map>_dem.tif` beside `<map>_meta.json` (`map_render.terrain`, read through `map_render.ground_msl` by the twin and the live source). Otherwise the block's floor (`tile_georef.FLOOR_MSL_M`). |

Cal City's surface is USGS 3DEP 1/3 arc-second heights (NAVD88) on 10 m
cells in EPSG:32611 over the bake's bounds. To make it from
the public 3DEP data, see
[`docs/data.md`](../../docs/data.md#flight-harness-dem).

## Vehicle protocol

```python
class Vehicle(Protocol):
    last_error: str

    def post(self, leg: Leg) -> None: ...
    def pose(self, t: float | None = None) -> Pose | None: ...
    def index(self, since: int | None = None) -> list[dict] | None: ...
    def frame(self, seq: int) -> tuple[bytes, float, int, int, dict] | None: ...
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...
```

| Member | Description |
|---|---|
| `post` | Hands the aircraft a leg and returns at once. Never blocks on the aircraft flying it. |
| `pose` | Returns the antenna's pose. |
| `index` | Returns `{"seq", "log_time", "t_host", "pose"}` rows with no pixels. |
| `frame` | Returns one frame's JPEG and its pose tag. |
| `now`, `sleep` | Clock the harness waits on. The vehicle owns it. |

One implementation is in the tree: the twin (`twin.py`).

A `Leg` carries `point` and `ahead`, which the twin flies, and `target`,
`step`, `area`, `revision`, and `note`: the hand-off that a vehicle for your
own aircraft reads to show the crew (on the C-12, the cockpit display) or to
drive an autopilot.

## Fly the twin

```sh
python -m benchmarks.flight.twin --config configs/calcity.yaml --episode 'calcity::6#6' --ortho <raster dir>
```

The twin flies each leg straight at the C-12's speed, wings level at the
mission's height, and renders frames from the raster pair on request.

| Flag | Description |
|---|---|
| `--config <yaml>` | Run config. Default `configs/calcity.yaml`. |
| `--episode <id>` | Episode to fly. Required. With `--start-xy`, the ID of a mission with no row. |
| `--ortho <raster dir>` | Raster pair the frames are rendered from. Required. |
| `--split <split>` | Split the episode is read from. Default: the Cal City bench's. |
| `--instruction <words>` | Operator's words in place of the episode's. |
| `--start-xy <x,y>` | Flies a mission with no row from this point in the bake's frame. Requires `--instruction`. |
| `--then <tasking>` | Speaks a tasking after each report, in order. |
| `--reasoning-effort <effort>` | Run's reasoning effort. |
| `--register <raster dir>` | Registers each frame. |
| `--out <dir>` | Where the sortie lands, as `<dir>/<UTC stamp>/`. Default `output/flight`. |
| `--memory-gib <GiB>` | Process's memory cap, GiB. Default 16. A block that allocates past it raises `MemoryError`, served as the block's error. |

The run lands under `<out>/<UTC stamp>/` with the run records a flown
run carries, the ownship track (`flight_track.jsonl`,
`Mission._record_track`) among them. `measures` scores it as it scores a
flown sortie.

A mission records every time (transcript `t_s`, the row's `started_at_utc`
and `elapsed_s`, the track's `t_unix`) on the vehicle's clock. The twin's
clock moves only while the harness waits on it, so a time to identification
is flying time.

## Run records

A mission writes these files. Run-level files are in its run folder;
sortie-level files are in the sortie folder.

| File | Contents |
|---|---|
| `episode.json`, and one line of the sortie's `episodes.jsonl` | Ledger row. |
| `conversation.jsonl` | Every transcript row, stamped `t_s`. |
| `detections.jsonl` | One line per detection, stamped with its step. |
| `frames/<sha>.jpg` | Every picture the model was sent. |
| `frames.jsonl` | One row per observation. |
| `observations/<sha>.jpg` | Observations. |
| `run.log` | Process log while the mission runs. |
| `flight/llm/transcript.jsonl` | Wire record. |
| Sortie's `flight_track.jsonl` | Ownship track. |

The measures read `conversation.jsonl`'s `mission_end`, `leg`, and `waypoint`
rows. No measure or decision reads the following fields:

| Fields | Purpose |
|---|---|
| `frames.jsonl`: `log_time`, `t_host`, `age_s`, `attitude`, `boresight`, `registration` (with its `frames` count). Ledger row: `ts` (the host's wall clock at the start). | Provenance of the recording, for a person auditing a flight against its camera recording. |
| `waypoint` row: the leg posted (`flight`, `n`, `label`, `lon`, `lat`, `coverageFloor`). `leg` row that ends a posted leg: its `area` and the `coverage` the recording held of it (`fraction`, one `held` flag per area point, `frames`). `frames.jsonl` row: the observation's ground (`world_window`, `shape`) and time (`t_s`). | What the station ([`station/`](../../station/README.md)) draws. |
| `conversation.jsonl` diagnostic rows: `malformed_envelope`, `empty_turn`, `code_shaped_reply`, `truncated_turn`, `no_flight_reply`, `transient_error`, `oversize_rejection`, `content_filter`, `request_refused`, and a `run` row's `reasoning`, `said`, `printed`, `errored`, and `frames`. | Turn-by-turn audit record. |

## Closed-loop measures

```sh
python -m benchmarks.flight.measures --root <sorties>/* --truth <truth>... [--json]
```

`--truth` takes directories of analysts' `<sortie>.json` entries, the tasking
set's split JSON, or both.

The command prints the totals, the totals by intel quality and by model and
reasoning effort, the median turn radius the way back is flown at, then one
line per run. The module's docstring says how each measure is read off the
records.

| Measure | Definition |
|---|---|
| SR | The commit point lies inside the target's mask: the analyst's hand-drawn one, else the tasking set's footprint. |
| NE | Distance from the commit point to the mask's centroid. |
| OSR | The aircraft's path (its GNSS track or the centres of the observations it fused) came within one camera footprint width of the target. The width is the C-12 camera's (`airframe.C12_SENSOR`) at the height of the closest approach. |
| Time to identification, SPL, tokens | See the module's docstring. |

| Run | Scoring |
|---|---|
| Serve failed before the run's first report (the bench's infra rule) | Out of every rate, and named. |
| Ends after a report without another (the crew's Stop, a harness ending, a failure) | Scored at that report: its end, time, and flights. |
