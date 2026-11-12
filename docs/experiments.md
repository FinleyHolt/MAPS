# Experiments

Every reported number is derived from run records by a reader. A bench run
writes a run root (`run_manifest.json` beside `episodes.jsonl`); the readers
in [`benchmarks/citynav/analysis`](../benchmarks/citynav/analysis/README.md)
read run roots, and
[`benchmarks/flight/measures.py`](../benchmarks/flight/measures.py) reads
flown sorties. That README maps every claim to its reader and command.

Before you run an experiment, prepare the data ([`data.md`](data.md)) and
start the two services ([`endpoints.md`](endpoints.md)).

## The experiments

| Experiment | Split and episodes | Arms | Config and flags | Reader |
|---|---|---|---|---|
| E2 | CityNav `test_unseen`, all 5,311 | Qwen 3.8 Flash Next at `low`, `medium`, `xhigh`; Inkling Small at 0.99; Gemma 4 26B-A4B; Gemma 4 E2B; Gemma 4 E4B | `configs/citynav.yaml`, `--split test_unseen` | `arm_table`, `run_report` |
| E3 | Cal City, the 1,100 reviewed episodes | Inkling Small at 0.0, 0.1 ... 0.9 and 0.99 (11 rungs); Gemma 4 26B-A4B | `configs/calcity.yaml`, `--episode-allowlist` from `benchmarks.calcity.data_prep.reviewed` | `arm_table` |
| E4 | CityNav `test_unseen`, 1,000 per arm (`--limit 1000`, seed 1234) | Per model: its ablation arms (below), each paired with the model's E2 root on the shared episodes | `configs/withhold/<arm>.yaml` | `arm_table`, `ab_report` |
| E6 | The closed-loop flight: 87 runs on a C-12 over California City | The models and efforts flown | The flown sorties' own records | `benchmarks.flight.measures` |

E2 and E4 need the CityNav data. E3 needs the Cal City raster and OSM
bundle, which this repository does not ship ([`data.md`](data.md#cal-city)).
E6 reads the flight recordings and the analysts' adjudications, which are
not distributed.

## Reproduce the bench experiments

Each command writes one run root. Name a model by serving it at
`SEARCH_REASONER_URL`; the run manifest records the served model the
endpoint reports. A rerun of the same command resumes the root.

E2, each arm over the whole split:

```sh
SEARCH_SAMPLING_REASONING_EFFORT=xhigh \
  python -m benchmarks.citynav.bench_fullset --split test_unseen --out-root output/e2/qwen38-flash-next-xhigh
```

The other arms change the effort (`low`, `medium`, or `0.99` with Inkling
Small served), or serve Gemma 4 26B-A4B, E2B or E4B with no effort set.

E3, the reviewed Cal City episodes per effort rung:

```sh
mkdir -p output && python -m benchmarks.calcity.data_prep.reviewed > output/reviewed_episodes.txt
for e in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99; do
  SEARCH_SAMPLING_REASONING_EFFORT=$e \
    python -m benchmarks.calcity.bench_fullset --episode-allowlist output/reviewed_episodes.txt \
    --out-root output/e3/inkling-small-e$e
done
python -m benchmarks.calcity.bench_fullset \
  --episode-allowlist output/reviewed_episodes.txt --out-root output/e3/gemma4-26b-a4b
```

E4, one root per ablation arm, per model. `configs/withhold/`
holds one config per withhold (`maps/withhold.py`) and one per
family: `seeing` (look_for, look, images, map) and `text` (doc, system_turn).
Under `end_commit` the mission still ends on a plain reply and scores the last
flight; only what the model is told about it is withheld.

```sh
# inkling-small:     arms="mission_map system_turn look_for end_commit"
# qwen38-flash-next: arms="prior look_for look images system_turn mission_map"
# gemma4-26b-a4b:    arms="prior library seeing text loop images"
for arm in $arms; do
  SEARCH_SAMPLING_REASONING_EFFORT=<model's E2 effort> CITYNAV_BENCH_CONFIG=configs/withhold/$arm.yaml \
    python -m benchmarks.citynav.bench_fullset --split test_unseen --limit 1000 --sample-seed 1234 \
    --out-root output/e4/<model>/$arm
done
```

The control is the model's E2 root (Inkling Small at 0.99, Qwen 3.8 Flash Next at `xhigh`, Gemma 4
26B-A4B with none), whose 5,311 episodes hold each arm's 1,000; `ab_report` pairs them on the
shared episodes and refuses a pair whose efforts differ.

## Read the results

```sh
python -m benchmarks.citynav.analysis.arm_table output/e2/*            # SR, OSR, NE per arm
python -m benchmarks.citynav.analysis.run_report output/e2/inkling-small-0.99   # one root in full
python -m benchmarks.citynav.analysis.ab_report output/e2/<model's E2 root> output/e4/<model>/<arm>
python -m benchmarks.citynav.analysis.spend <out-root>                 # a hosted endpoint's tokens and USD
```

| Reader | Reads | Prints |
|---|---|---|
| `run_report` | Each root in turn | SR, NE, SPL, OSR, SR by target shape, by referent bucket and by intel quality, commit provenance, the bake check. |
| `arm_table` | Many roots, one per arm | One row per arm: effort, n, SR, OSR, NE. |
| `ab_report` | Two roots | Paired SR with a bootstrap interval on the difference, McNemar, the flip counts. |
| `spend` | Every attempt's transcript | Tokens and USD at a hosted endpoint, USD per scored episode. |
| `benchmarks.flight.measures` | Flown or twin sorties | SR, OSR, NE, time to identification, SPL per run and in total. |

`run_report`, `arm_table` and `ab_report` refuse a root that scores fewer
episodes than its manifest selected; resume the run into the same root, or
pass `--allow-partial`.

## The closed-loop measures (E6)

The closed-loop measures are read from the flown sorties' own records by
[`benchmarks/flight/measures.py`](../benchmarks/flight/measures.py), over the
analysts' adjudications: of 87 runs, 18 leave every rate with their evidence
as the reason (13 lost to the infrastructure, 4 stopped by the crew, 1 whose
target has no mask), and 69 are scored.

| Measure | Value | Definition |
|---|---|---|
| SR | 40/69 | The commit point lies inside the target's hand-drawn mask (one run with none: its tasking footprint). |
| OSR | 60/69 | The aircraft's path (its GNSS track and the centres of the observations the run fused) came within one camera footprint width of the target centroid. |

| Intel quality (the run's first tasking) | Scored runs | SR | OSR |
|---|---|---|---|
| Good | 27 | 22/27 | 26/27 |
| Medium | 32 | 13/32 | 25/32 |
| Poor | 10 | 5/10 | 9/10 |

```sh
python -m benchmarks.flight.measures --root <sorties>/* --truth <adjudications dir> <flight-week tasking JSON>
```

The tasking JSON is the flight week's own tasking set (it scores a run with no
hand-drawn mask, matched by the run's description); the released Cal City split
is a different set of episodes.

A run the analysts' adjudication marks excluded is out of every rate, and so
is a run the serve failed under (the bench's infra rule,
`benchmarks/run_record/rows.py`'s `is_infra_abort`), which the output names.
The flight recordings and the adjudications are not distributed, so these
two tables are the record of the flight week and the command above runs only
on your own sorties: a twin run ([`flight.md`](flight.md)) is scored the same
way.

The engine in this repository differs from the engine as flown in the places
[`differences-from-flight.md`](differences-from-flight.md) lists.
