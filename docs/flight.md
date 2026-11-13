# Flight: harness, twin, and your own aircraft

This page is the short guide to the flight harness. MAPS flew 91 closed-loop
runs on a C-12 over California City, CA, on six days from 11 to 18 September
2026, at about 3,000 to 6,500 ft above ground. A run counts when the model
wrote at least one block in it. The reasoner and SAM 3.1 ran on ground
compute over the datalink; a computer aboard held the mission.
[`benchmarks/flight`](../benchmarks/flight/README.md) is the method half of
the station that flew them. Its README is the reference: the flown chain step
by step, the `Vehicle` protocol, the run's records, and the closed-loop
measures.

| Task | Tool | Requirement |
|---|---|---|
| Fly a mission without an aircraft | Twin. See [Fly the twin](#fly-the-twin). | The Cal City bake and raster pair ([`data.md`](data.md#cal-city)), a reasoner, and the SAM serve ([`endpoints.md`](endpoints.md)). |
| Watch or re-task a twin mission in a browser, or review finished runs | Station. See [`station/README.md`](../station/README.md). | To fly: what the twin needs. To review: the run folders. |
| Score runs | [`benchmarks/flight/measures.py`](../benchmarks/flight/measures.py). See [`experiments.md`](experiments.md#closed-loop-measures). | Sortie folders and truth for their targets. |
| Fly your own aircraft | A `Vehicle`. See [Bring your own aircraft](#bring-your-own-aircraft). | Your aircraft, its camera recording, and a bake of your area. |

## Harness

| Component | Role |
|---|---|
| `Mission` (`benchmarks/flight/mission.py`) | Hydrates the engine as the bench does, holds the conversation open across operator taskings, and writes the run's record. |
| `Legs` (`benchmarks/flight/legs.py`) | Flight on the aircraft: the leg posted for an area, arrived when the recording holds that area, a short pass asked again, and the pass served as the flight's observation. |
| `Vehicle` (`benchmarks/flight/vehicle.py`) | Boundary to the aircraft: `post(leg)`, `pose(t)`, `index(since)`, `frame(seq)`, `now()`, `sleep(s)`. |
| `LiveSource` (`benchmarks/flight/live_source.py`) | Turns the recorded frames into the observation the model reads: coverage, per-frame registration, the pose-graph corrector, compositing, and the mosaic fit. |

## Fly the twin

The twin is a kinematic `Vehicle` over a raster pair. It flies a Cal City
mission end to end without an aircraft:

```sh
python -m benchmarks.flight.twin --config configs/calcity.yaml --episode 'calcity::6#6' --ortho <root>/data/rgbd_upstream
```

The run lands under `output/flight/<UTC stamp>/` with the records a flown run
carries, and `measures` scores it as it scores a flown sortie. The twin, the
live source, and the frame registration stand each frame on the bake's
`calcity_dem.tif` where it is present. See
[`data.md`](data.md#flight-harness-dem).

## Watch a mission

[`station/`](../station/README.md) is a web console over a bench's runs and,
with `--bench calcity --ortho <raster dir>`, over the twin. Pick a Cal City
tasking or type your own, watch the model reason turn by turn on the map, and
re-task it while it flies. Every finished run under the output root opens as
a review with a scrubber. For the command and flags, see the station's
README.

## Bring your own aircraft

The harness reaches an aircraft through one protocol, `Vehicle`. On the C-12,
the crew flew each leg off a cockpit display; an autopilot can take the same
hand-off. These parts of the harness carry the C-12 and Cal City as they
stand, and the steps name each one.

| Part | C-12 or Cal City value | Read by |
|---|---|---|
| Antenna-to-camera lever arm | `airframe.C12_LEVER_M` | `LiveSource.camera_pose` (`live_source.py`) and the twin |
| Camera of the OSR footprint width | `airframe.C12_SENSOR` | `measures.py` |
| Bench | `calcity_bench(os.environ)` | The twin, `measures.py`, and the station's twin |
| Floor heights | The `calcity` entries of `GROUND_LEVEL` and `tile_georef.FLOOR_MSL_M` | `open_live` |

1. Implement `Vehicle` (`benchmarks/flight/vehicle.py`). For its members,
   see [Vehicle protocol](../benchmarks/flight/README.md#vehicle-protocol).
   `benchmarks/flight/twin.py` is the implementation in the tree.

2. Describe the camera. A config's `sensor` names a rig in
   `maps/perception/geometry/sensor_spec.py`. `open_live` takes the
   calibration (boresight, GNSS bias, timing offset) as `boresight_path`, a
   `boresight.json`; the twin and the station pass none and fly the
   datasheet camera. The lever arm from the GNSS antenna to the camera is
   `airframe.C12_LEVER_M`; an airframe of your own needs its own value
   there.

3. Bake your area. The harness reads a tile bake with a georef
   (`<map>_meta.json`, `<map>_features.json`) and the block's floor heights
   (`GROUND_LEVEL` in `maps/perception/ortho/ground_level.py`,
   `tile_georef.FLOOR_MSL_M`). The
   [Cal City build steps](../benchmarks/calcity/README.md#build-the-bench)
   are the worked example; an area of your own needs its own bundle, raster,
   and floor entries.

4. Build a bench over the bake. `Mission` takes a `Bench`
   (`benchmarks/bench.py`); `calcity_bench` (`benchmarks/calcity/paths.py`)
   is the worked example, and the twin, the measures, and the station build
   that one. A mission over a row of the bench's split takes the row's
   episode ID; a mission with no row takes `start_xy` and `instruction`
   (the twin's `--start-xy` and `--instruction`).

5. Open the mission. Hand the vehicle to `open_live`
   (`benchmarks/flight/live_source.py`) and the result to `Mission`, as
   `twin.py`'s `main` does. The station builds its twin in `Station`
   (`station/serve.py`); to watch your aircraft there, hand your vehicle to
   `open_live` in its place. The station's sandbox gives a vehicle the
   network and its bound paths only: serial devices, `/run` sockets, and
   files in home stay outside it, so a vehicle that needs them runs from
   your own entry point. See [`station/README.md`](../station/README.md).

6. Score the runs. `measures.py` takes the OSR footprint width from
   `airframe.C12_SENSOR`; a camera of your own needs its rig there.

A planner or autonomy stack that executes moves itself, without the leg and
recording chain, connects one layer down, at `MissionSpec`. See
[`architecture.md`](architecture.md#where-another-system-connects).
