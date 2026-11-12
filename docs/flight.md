# Flight: the harness, the twin, replay and your own aircraft

MAPS flew 87 closed-loop runs on a C-12 over California City, CA, from 11 to
18 September 2026, at about 3,000 to 6,500 ft above ground. The reasoner ran
off board over the datalink; SAM 3.1 ran on board.
[`benchmarks/flight`](../benchmarks/flight/README.md) is the method half of
the station that flew them, and its README is the reference for everything
on this page: the flown chain step by step, the `Vehicle` protocol, the run's
records and the closed-loop measures.

| You want to | Use | Needs |
|---|---|---|
| Fly a mission without an aircraft | The twin, [below](#fly-the-twin) | The Cal City bake and raster pair ([`data.md`](data.md#cal-city)), a reasoner and the SAM serve ([`endpoints.md`](endpoints.md)). |
| Watch or re-task a twin mission in a browser, or review finished runs | The station ([`station/README.md`](../station/README.md)) | To fly: what the twin needs. To review: the run folders. |
| Fly a recorded run again | The replay, [below](#replay-a-flown-run) | A flown run's folder. The flight recordings are not distributed. |
| Score runs | [`benchmarks/flight/measures.py`](../benchmarks/flight/measures.py), in [`experiments.md`](experiments.md#the-closed-loop-measures-e6) | Sortie folders and truth for their targets. |
| Fly your own aircraft | A `Vehicle`, [below](#bring-your-own-aircraft) | Your aircraft, its camera recording and a bake of your area. |

## The harness

* `Mission` (`benchmarks/flight/mission.py`) hydrates the engine as the bench
  does and holds the conversation open across operator taskings. A flight is
  a leg the aircraft flies, arrived when the recording holds the area it was
  flown for.
* `Vehicle` (`benchmarks/flight/vehicle.py`) is the boundary to the aircraft:
  `post(leg)`, `pose(t)`, `index(since)`, `frame(seq)`, `now()`, `sleep(s)`.
* `LiveSource` (`benchmarks/flight/live_source.py`) turns the recorded frames
  into the observation the model reads: coverage, per-frame registration, the
  pose-graph corrector, compositing and the mosaic fit.

## Fly the twin

The twin is a kinematic `Vehicle` over a raster pair, so you can fly a Cal
City mission end to end without an aircraft:

```sh
python -m benchmarks.flight.twin --config configs/calcity.yaml --episode 'calcity::6#6' --ortho <root>/data/rgbd_upstream
```

The run lands under `output/flight/<UTC stamp>/` with the records a flown
run carries, and `measures` scores it as it scores a flown sortie. The twin,
the live source and the frame registration stand each frame on the bake's
`calcity_dem.tif` where it is present ([`data.md`](data.md#the-dem-the-flight-harness-reads)).

## Watch a mission

[`station/`](../station/README.md) is a web console over a bench's runs and,
with `--bench calcity --ortho <raster dir>`, over the twin: pick a
Cal City tasking or type your own, watch the model reason turn by turn, see
the aircraft, each leg's area and the observation it brought back on the
map, and re-task the model while it flies. Every finished run under its
output root opens as a review with a scrubber. The station's README gives
its command and flags.

## Replay a flown run

`python -m benchmarks.flight.replay` flies a recorded run again on this
repository's harness, at three tiers:

| Command | Answers each flight with | Shows |
|---|---|---|
| `replay <run>` | The recorded observations that hold the leg's area (`--bank week`: every run under the sorties root), with your reasoner and segmenter. | How the method decides on the ground the aircraft actually saw. |
| `replay <run> --forced` | The recorded reasoner responses, observations and detections. Every request's text is diffed against the recording; differing pictures are counted (`pictures_differ`). | Whether this harness sends the flown request text turn for turn. |
| `replay <run> --frames <sensors.mcap> --boresight <chain.json> --raster <raster dir>` | Observations re-fused from the camera recording through the live chain. | The flight-side observation chain on the real frames. |

The station registered frames against NAIP 2024 imagery: the `calcity.png`
of the v5 Cal City data release beside the flown bake. Pass that raster's
directory as `--raster` to replay as flown. The `--frames` tier needs the
`recording` extra.

## Bring your own aircraft

The harness reaches an aircraft through one protocol and assumes nothing
else about it. On the C-12 the crew flew each leg off a cockpit display; an
autopilot can take the same hand-off.

1. **Implement `Vehicle`** (`benchmarks/flight/vehicle.py`). `post(leg)`
   hands the aircraft a `Leg` (the point, the rest of the pass, the area the
   leg is flown for) and returns at once. `pose(t)` answers the antenna's
   pose. `index(since)` lists recorded frames with their pose tags and no
   pixels; `frame(seq)` returns one frame's JPEG. `now()` and `sleep(s)` are
   the clock the harness waits on. `benchmarks/flight/twin.py` and
   `benchmarks/flight/recorded_ring.py` are the two implementations in the
   tree.
2. **Describe the camera.** A config's `sensor` names a rig in
   `maps/perception/geometry/sensor_spec.py`; the calibration (boresight,
   GNSS bias, timing offset) is a `boresight.json` read by
   `live_source.py`.
3. **Bake your area.** The harness reads a tile bake with a georef
   (`<map>_meta.json`, `<map>_features.json`) and the block's floor height
   (`tile_georef.FLOOR_MSL_M`). The Cal City build steps in
   [`data.md`](data.md#cal-city) are the worked example; an area of your own
   needs its own bundle, raster and floor entry.
4. **Open the mission.** Hand the vehicle to `open_live`
   (`benchmarks/flight/live_source.py`) and the result to `Mission`, as
   `twin.py`'s `main` does. The station's console works over any such
   mission ([`station/README.md`](../station/README.md)).

A planner or autonomy stack that executes moves itself, without the leg and
recording chain, connects one layer down, at `MissionSpec.fly`
([`architecture.md`](architecture.md#where-another-system-connects)).

The engine differs from the engine as flown in the places
[`differences-from-flight.md`](differences-from-flight.md) lists.
