# Flight: harness and twin

This page is the short guide to the flight harness.
[`benchmarks/flight`](../benchmarks/flight/README.md) is the method half of
the station that flew the campaign below. Its README is the reference: the flown chain step
by step, the `Vehicle` protocol, the run's records, and the closed-loop
measures.

## Flight campaign

MAPS flew a crewed C-12 over California City, CA, on six days from 11 to 18
September 2026, at about 3,000 to 6,500 ft above ground. The crew flew each
leg the model requested off a cockpit display. The counts are the paper's.

| Count | Searches | Rule |
|---|---|---|
| Flown | 75 | A California City search flown in the air. |
| Excluded | 18 | 11 lost to the infrastructure, 4 ended by the crew, 1 without ground truth, 2 never answered. None of the 11 excluded searches that answered was correct. |
| Complete | 57 | Not excluded: ground truth labelled and an answer committed. |
| Identified | 34 of 57 | The committed point lies inside the described object's hand-drawn mask. |

| Component | Location |
|---|---|
| Reasoner (Inkling Small) and SAM 3.1 | NPS GB300 on the ground, over the datalink, for all 57 complete searches. |
| Mission (the engine and the flight harness) | Jetson Thor aboard the aircraft. |

The flight recordings and their adjudications are not in this repository or
its release.

| Task | Tool | Requirement |
|---|---|---|
| Fly a mission without an aircraft | Twin. See [Fly the twin](#fly-the-twin). | The Cal City bake and raster pair ([`data.md`](data.md#cal-city)), a reasoner, and the SAM serve ([`endpoints.md`](endpoints.md)). |
| Watch or re-task a twin mission in a browser, or review finished runs | Station. See [`station/README.md`](../station/README.md). | To fly: what the twin needs. To review: the run folders. |
| Score runs | [`benchmarks/flight/measures.py`](../benchmarks/flight/measures.py). See [`experiments.md`](experiments.md#closed-loop-measures). | Sortie folders and truth for their targets. |

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
