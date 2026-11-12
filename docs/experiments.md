# Experiments

This page lists the experiments that the main branch reproduces directly: the
CityNav and Cal City runs of the full method, with the command that flies
each run and the reader that derives each number.

The ablation arms, the comparison against the baselines, and the as-flown
replay reproduce from the tag
[`paper-v1`](https://github.com/FinleyHolt/MAPS/tree/paper-v1)
(`git checkout paper-v1`), which freezes the tree that produced the paper.

Before you run an experiment, prepare the data ([`data.md`](data.md)) and
start the two services ([`endpoints.md`](endpoints.md)).

## Experiment list

| Experiment | Split and episodes | Runs | Config and flags | Reader |
|---|---|---|---|---|
| E2 | CityNav `test_unseen`, all 5,311 | Qwen 3.8 Flash Next at `low`, `medium`, `xhigh`; Inkling Small at 0.99; Gemma 4 26B-A4B; Gemma 4 E2B; Gemma 4 E4B | `configs/citynav.yaml`, `--split test_unseen` | `run_table`, `run_report` |
| E3 | Cal City, the 1,100 reviewed episodes | Inkling Small at 0.0, 0.1 ... 0.9 and 0.99 (11 rungs); Gemma 4 26B-A4B | `configs/calcity.yaml`, `--episode-allowlist` from `benchmarks.calcity.data_prep.reviewed` | `run_table` |

| Experiment | Requirement |
|---|---|
| E2 | The CityNav data. |
| E3 | The Cal City raster and OSM bundle, which this repository doesn't ship. See [`data.md`](data.md#cal-city). |

## Reproduce the bench experiments

Each command writes one run root (`run_manifest.json` beside
`episodes.jsonl`). To name a model, serve it at `SEARCH_REASONER_URL`; the
run manifest records the served model the endpoint reports. Rerun a command
to resume its root.

### E2

Fly each run over the whole split:

```sh
SEARCH_SAMPLING_REASONING_EFFORT=xhigh \
  python -m benchmarks.citynav.bench_fullset --split test_unseen --out-root output/e2/qwen38-flash-next-xhigh
```

For the other runs, change the effort (`low`, `medium`, or `0.99` with
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
python -m benchmarks.citynav.analysis.run_table output/e2/*            # SR, OSR, NE per run
python -m benchmarks.citynav.analysis.run_report output/e2/inkling-small-0.99   # one root in full
python -m benchmarks.citynav.analysis.ab_report <root a> <root b>      # paired SR of two roots
python -m benchmarks.citynav.analysis.spend <out-root>                 # a hosted endpoint's tokens and USD
```

For each reader's input and output, and for reading a root that scores fewer
episodes than its manifest selected, see
[`benchmarks/citynav/analysis`](../benchmarks/citynav/analysis/README.md#readers).

## Closed-loop measures

[`benchmarks/flight/measures.py`](../benchmarks/flight/measures.py) reads SR,
OSR, NE, time to identification, and SPL from sortie folders: a twin's
([`flight.md`](flight.md)) or your own aircraft's.

```sh
python -m benchmarks.flight.measures --root <sorties>/* --truth <truth>...
```

`--truth` takes directories of analysts' adjudications, the tasking set's
split JSON, or both. For the closed-loop flight's own records (E6), see
[Flight recordings](data.md#flight-recordings). For each measure's
definition, see
[Closed-loop measures](../benchmarks/flight/README.md#closed-loop-measures).
