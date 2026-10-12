# benchmarks/flight

The flight harness: the method half of the station that flew MAPS on a C-12
over California City from 11 to 18 September 2026. A mission is hydrated as
the bench hydrates one, its flights are legs an aircraft flies, and its
observations are fused from the frames the aircraft recorded. The aircraft is
reached through one protocol, `Vehicle`. This package also holds a kinematic
twin of the aircraft, the as-flown replay and the closed-loop measures.

## The flown chain

One flight, from the model's `go_to` or `explore` stop to the observation it
reads:

1. **The stop becomes a leg.** `Mission._fly_live` (`mission.py`) resolves the
   stop to a point, takes the area the leg is flown for (`leg_area`: the
   target's footprint, or a disc about a point), and asks the recording
   whether it holds that area already. When it does, no leg is flown.
2. **The leg goes to the aircraft.** `Vehicle.post(Leg)` (`vehicle.py`) hands
   the point, the rest of the pass and the area to the aircraft. On the C-12
   the crew flew it off the cockpit display.
3. **Every frame is registered as it arrives.** The `Corrector` thread
   (`live_source.py`) reads the newest frame off `Vehicle.index` and
   `Vehicle.frame`, places it by its tagged pose (`nav_track.py`) through the
   boresight, registers it against the map rendered at that pose
   (`register_frame`, `frame_offset.py`, `map_render.py`, `locator.py`), and
   ties it to the frame before it in the pose graph (`pose_graph.py`,
   `frame_ties.py`, `tie_adjust.py`).
4. **Arrival is coverage.** `LiveSource.coverage` reads the ring's index (one
   pose a frame, no pixels; `footprint.py`) and reports the share of the
   area's points some frame holds. The leg has arrived at 90 %
   (`mission.COVERED_FRACTION`), or after `ARRIVAL_TIMEOUT_S` with what the
   pass held. When the index holds the area but none of its frames could be
   fetched, the leg is flown and only the frames recorded after it count.
5. **The observation is fused.** `LiveSource.observe_area` fetches, for each
   area point, the frame that saw it nearest nadir, places each by its pose
   and registration, composites them on one north-up grid (`composite`),
   registers the mosaic on the map as one similarity (`register_mosaic`),
   and hands it over as an orthographic observation on flat ground at the
   bake's floor (`ortho_observation`).
6. **The engine reads it.** The observation goes through the bench's
   observation path: SAM 3.1 segments the watched classes, detections join
   the graph, and the flight report returns to the model. `run_imagery.py`
   keeps what the run captured for `look` to crop.

The calibration (boresight, GNSS bias, timing offset) is read from a
`boresight.json` and applied in `live_source.py` alone; a station started
with none flies the datasheet camera. `airframe.py` holds the C-12's
antenna-to-camera geometry and `tile_georef.py` the bake's local metres to
WGS84.

The ground under each frame is the block's bare-earth surface where the bake
carries one, `<map>_dem.tif` beside `<map>_meta.json` (`map_render.terrain`,
read through `map_render.ground_msl` by the twin and the live source), and the block's floor (`tile_georef.FLOOR_MSL_M`) where it
does not. The flown bake carried Cal City's: USGS 3DEP 1/3 arc-second heights
(NAVD88) on 10 m cells in EPSG:32611 over the bake's bounds; the top-level
README's Cal City section says how to make it from the public 3DEP data.

## The Vehicle protocol

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

`post` never blocks on the aircraft flying the leg. `index` returns
`{"seq", "log_time", "t_host", "pose"}` rows with no pixels; `frame` returns
one frame's JPEG and its pose tag. The vehicle owns the clock the harness
waits on. Two implementations exist: the twin (`twin.py`) and a camera
recording read back (`recorded_ring.py`). A `Leg` carries `point` and
`ahead`, which the twin flies, and `target`, `step`, `area`, `revision` and
`note`: the hand-off a vehicle for your own aircraft reads to show the crew
(on the C-12, the cockpit display) or to drive an autopilot.

## Fly the twin

```sh
python -m benchmarks.flight.twin --config configs/calcity.yaml --episode 'calcity::6#6' --ortho <raster dir>
```

The twin flies each leg straight at the C-12's speed, wings level at the
mission's height, and renders frames from the raster pair on request.
`--then` speaks taskings after each report, `--boresight` applies a solve,
and `--register <raster dir>` registers each frame. The run lands under
`output/flight/<UTC stamp>/` with the run records a flown run carries, the
ownship track (`flight_track.jsonl`, `Mission._record_track`) among them, and
`measures` scores it as it scores a flown sortie. A mission records every
time (transcript `t_s`, the row's `started_at_utc` and `elapsed_s`, the
track's `t_unix`) on the vehicle's clock; the twin's moves only while the
harness waits on it, so a time to identification is flying time.

## Replay a flown run

```sh
python -m benchmarks.flight.replay <sortie>/<run dir> [--bank week]
python -m benchmarks.flight.replay <sortie>/<run dir> --forced
python -m benchmarks.flight.replay <sortie>/<run dir> --frames <sensors.mcap> --boresight <chain.json> --raster <raster dir>
```

| Tier | What answers a flight | What it proves |
|---|---|---|
| Observation bank (default) | The recorded observations holding the leg's area (`--bank run`: the run's own; `--bank week`: every run under the sorties root). Your reasoner and segmenter answer; the recorded taskings are spoken after each report. | How the method decides over the ground the aircraft actually saw. |
| Forced (`--forced`) | The recorded reasoner responses, observations and detections. Every request's text is diffed against the recording into `replay_diff.json`; differing pictures are counted (`pictures_differ`); a text divergence exits non-zero. No endpoint is contacted. | That this harness sends the flown request text turn for turn. |
| Frames (`--frames`) | The camera recording as the aircraft's ring, re-fused through the live chain at every waypoint. Needs the `recording` extra. | The observation chain on the real frames. |

The station registered frames against NAIP 2024 imagery (the `calcity.png`
of the v5 Cal City data release beside the flown bake); pass its directory as
`--raster`. A replay's ledger row names the run it replays (`replay_of`) and
is scored by `measures` on SR and NE (it records no start time and no
track), not by the bench reader. The recordings are not
distributed.

## Closed-loop measures

```sh
python -m benchmarks.flight.measures --root <sorties>/* --truth <adjudications dir> <flight-week tasking JSON> --tiles <Cal City bake whose calcity_features.json holds the flown prior> [--json]
```

Per run: SR two ways (the committed area contains the target centroid; the
commit lies within 20 m of it), NE, OSR (the aircraft's path, GNSS track,
else keyframe poses, came within one camera footprint width of the target), time to identification, SPL, and
tokens. A run the serve failed under (the bench's infra rule) is out of every
rate and named. It prints the totals, the totals by intel quality and by model
and reasoning effort, the count of `prior_<i>` commits whose outline in
`--tiles` does not hold the commit (a sign of another bake), then one line per run. `--check <master table>` compares SR and OSR
row by row against a table.
