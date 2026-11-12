# Flight: harness, twin, replay, and your own aircraft

This page is the short guide to the flight harness. MAPS flew 87 closed-loop
runs on a C-12 over California City, CA, from 11 to 18 September 2026, at
about 3,000 to 6,500 ft above ground. The reasoner ran off board over the
datalink; SAM 3.1 ran on board.
[`benchmarks/flight`](../benchmarks/flight/README.md) is the method half of
the station that flew them. Its README is the reference: the flown chain step
by step, the `Vehicle` protocol, the run's records, and the closed-loop
measures.

| Task | Tool | Requirement |
|---|---|---|
| Fly a mission without an aircraft | Twin. See [Fly the twin](#fly-the-twin). | The Cal City bake and raster pair ([`data.md`](data.md#cal-city)), a reasoner, and the SAM serve ([`endpoints.md`](endpoints.md)). |
| Watch or re-task a twin mission in a browser, or review finished runs | Station. See [`station/README.md`](../station/README.md). | To fly: what the twin needs. To review: the run folders. |
| Fly a recorded run again | Replay. See [Replay a flown run](#replay-a-flown-run). | A flown run's folder. The flight recordings aren't distributed. |
| Score runs | [`benchmarks/flight/measures.py`](../benchmarks/flight/measures.py). See [`experiments.md`](experiments.md#closed-loop-measures-e6). | Sortie folders and truth for their targets. |
| Fly your own aircraft | A `Vehicle`. See [Bring your own aircraft](#bring-your-own-aircraft). | Your aircraft, its camera recording, and a bake of your area. |

## Harness

| Component | Role |
|---|---|
| `Mission` (`benchmarks/flight/mission.py`) | Hydrates the engine as the bench does and holds the conversation open across operator taskings. A flight is a leg the aircraft flies, arrived when the recording holds the area it was flown for. |
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

## Replay a flown run

`python -m benchmarks.flight.replay` flies a recorded run again on this
repository's harness, at three tiers:

| Tier | Command |
|---|---|
| Observation bank | `replay <run>` |
| Forced | `replay <run> --forced` |
| Frames | `replay <run> --frames <sensors.mcap> --boresight <chain.json> --raster <raster dir>` |

For what answers a flight at each tier and what each proves, see
[Replay a flown run](../benchmarks/flight/README.md#replay-a-flown-run) in
the reference.

## Bring your own aircraft

The harness reaches an aircraft through one protocol and assumes nothing else
about it. On the C-12, the crew flew each leg off a cockpit display; an
autopilot can take the same hand-off.

1. Implement `Vehicle` (`benchmarks/flight/vehicle.py`). For its members,
   see [Vehicle protocol](../benchmarks/flight/README.md#vehicle-protocol).
   `benchmarks/flight/twin.py` and `benchmarks/flight/recorded_ring.py` are
   the two implementations in the tree.

2. Describe the camera. A config's `sensor` names a rig in
   `maps/perception/geometry/sensor_spec.py`. The calibration (boresight,
   GNSS bias, timing offset) is a `boresight.json` read by `live_source.py`.

3. Bake your area. The harness reads a tile bake with a georef
   (`<map>_meta.json`, `<map>_features.json`) and the block's floor height
   (`tile_georef.FLOOR_MSL_M`). The
   [Cal City build steps](../benchmarks/calcity/README.md#build-the-bench)
   are the worked example; an area of your own needs its own bundle, raster,
   and floor entry.

4. Open the mission. Hand the vehicle to `open_live`
   (`benchmarks/flight/live_source.py`) and the result to `Mission`, as
   `twin.py`'s `main` does. The station's console works over any such
   mission. See [`station/README.md`](../station/README.md).

A planner or autonomy stack that executes moves itself, without the leg and
recording chain, connects one layer down, at `MissionSpec.fly`. See
[`architecture.md`](architecture.md#where-another-system-connects).
