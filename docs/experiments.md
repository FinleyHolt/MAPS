# Experiments

This page lists each experiment, the command that flies each arm, and the
reader that derives each number. Every reported number is derived from run
records by a reader:

- A bench run writes a run root (`run_manifest.json` beside
  `episodes.jsonl`). The readers in
  [`benchmarks/citynav/analysis`](../benchmarks/citynav/analysis/README.md)
  read run roots. That README maps every claim to its reader and command.
- [`benchmarks/flight/measures.py`](../benchmarks/flight/measures.py) reads
  flown sorties.

Before you run an experiment, prepare the data ([`data.md`](data.md)) and
start the two services ([`endpoints.md`](endpoints.md)).

## Experiment list

| Experiment | Split and episodes | Arms | Config and flags | Reader |
|---|---|---|---|---|
| E2 | CityNav `test_unseen`, all 5,311 | Qwen 3.8 Flash Next at `low`, `medium`, `xhigh`; Inkling Small at 0.99; Gemma 4 26B-A4B; Gemma 4 E2B; Gemma 4 E4B | `configs/citynav.yaml`, `--split test_unseen` | `arm_table`, `run_report` |
| E3 | Cal City, the 1,100 reviewed episodes | Inkling Small at 0.0, 0.1 ... 0.9 and 0.99 (11 rungs); Gemma 4 26B-A4B | `configs/calcity.yaml`, `--episode-allowlist` from `benchmarks.calcity.data_prep.reviewed` | `arm_table` |
| E6 | The closed-loop flight: 87 runs on a C-12 over California City | The models and efforts flown | The flown sorties' own records | `benchmarks.flight.measures` |

| Experiment | Requirement |
|---|---|
| E2 | The CityNav data. |
| E3 | The Cal City raster and OSM bundle, which this repository doesn't ship. See [`data.md`](data.md#cal-city). |
| E6 | The flight recordings and the analysts' adjudications, which aren't distributed. |

## Reproduce the bench experiments

Each command writes one run root. To name a model, serve it at
`SEARCH_REASONER_URL`; the run manifest records the served model the endpoint
reports. Rerun a command to resume its root.

### E2

Fly each arm over the whole split:

```sh
SEARCH_SAMPLING_REASONING_EFFORT=xhigh \
  python -m benchmarks.citynav.bench_fullset --split test_unseen --out-root output/e2/qwen38-flash-next-xhigh
```

For the other arms, change the effort (`low`, `medium`, or `0.99` with
Inkling Small served), or serve Gemma 4 26B-A4B, E2B, or E4B with no effort
set.

### E3

Fly the reviewed Cal City episodes per effort rung:

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

## Read the results

```sh
python -m benchmarks.citynav.analysis.arm_table output/e2/*            # SR, OSR, NE per arm
python -m benchmarks.citynav.analysis.run_report output/e2/inkling-small-0.99   # one root in full
python -m benchmarks.citynav.analysis.ab_report <root a> <root b>      # paired SR of two roots
python -m benchmarks.citynav.analysis.spend <out-root>                 # a hosted endpoint's tokens and USD
```

For each reader's input and output, and for reading a root that scores fewer
episodes than its manifest selected, see
[`benchmarks/citynav/analysis`](../benchmarks/citynav/analysis/README.md#readers).
`benchmarks.flight.measures` reads flown or twin sorties.

## Closed-loop measures (E6)

[`benchmarks/flight/measures.py`](../benchmarks/flight/measures.py) reads the
closed-loop measures from the flown sorties' own records, over the analysts'
adjudications. Of 87 runs, 69 are scored. The other 18 leave every rate, with
their evidence as the reason: 13 lost to the infrastructure, 4 stopped by the
crew, and 1 whose target has no mask.

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

- The tasking JSON is the flight week's own tasking set. It scores a run with
  no hand-drawn mask, matched by the run's description. The released Cal City
  split is a different set of episodes.
- A run that the analysts' adjudication marks excluded is out of every rate.
  So is a run the serve failed under (the bench's infra rule,
  `benchmarks/run_record/rows.py`'s `is_infra_abort`), which the output
  names.
- The flight recordings and the adjudications aren't distributed, so the two
  tables are the record of the flight week, and the command runs only on your
  own sorties. A twin run ([`flight.md`](flight.md)) is scored the same way.
