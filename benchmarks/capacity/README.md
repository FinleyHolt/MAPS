# Capacity

How many correct finds per hour one operator lands as offered report load
rises, flying the reports alone or with agents flying them (assisted). This
package holds the headless pieces. The participant front end is not in this
repository and writes the same session shape.

| Piece | What it is |
|---|---|
| `streams.py` | Seeded Poisson report streams over the 1,100 reviewed Cal City episodes (`benchmarks.calcity.data_prep.reviewed`), drawn without replacement |
| `bench_fullset --arrivals` | The assisted arm: the shared batch driver releases each report at its arrival time to `--workers N` concurrent missions, one process each, frames from the bench's ortho crops |
| `reader.py` | Correct finds per hour against offered load, the saturation load, and alone against assisted paired on the same stream |

## A session

```bash
python -m benchmarks.capacity.streams --load 30 --hours 1 --seed 1 --out stream.json
python -m benchmarks.calcity.bench_fullset --workers 4 --arrivals stream.json --out-root output/capacity/l30_s1
python -m benchmarks.capacity.reader --assisted output/capacity/l*_s* [--alone <roots>] [--json]
```

A session flies into a fresh `--out-root`: its clock cannot resume.
`--arrivals` is refused with `--episode-allowlist`, `--limit` or
`--sample-seed`, which select episodes the stream already chose.

## Session shape (both arms)

- `<root>/arrivals.json`: the stream as `streams.py` wrote it (`load_per_h`, `hours`, `seed`, `reports`).
- `<root>/episodes.jsonl`: one row per report attempt, read through `benchmarks.citynav.analysis.rows.load_attempts`, carrying `episode_id`, `workdir`, `status`, `arrival_s`, `claimed_s` (session seconds), `elapsed_s` (the attempt's wall seconds) and `metrics.SR`.

A report is found at `claimed_s + elapsed_s`. It counts toward finds per hour
when its SR is 1 and it is found inside the session's `hours`. A session
reads as it stands: a report whose every attempt was an infrastructure
casualty is offered and never found, and `infra` counts it. Sessions pair
across arms by `(load_per_h, seed, hours)`, one session per arm per stream;
the reader refuses a second session on a stream in one arm and a pair whose
reports differ.
