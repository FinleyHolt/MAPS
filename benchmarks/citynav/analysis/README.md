# benchmarks/citynav/analysis

These readers derive every reported number. Each reads run roots, the
directories a bench run writes (`run_manifest.json` beside `episodes.jsonl`).
No reader defines a metric of its own: `rows.load_attempts` decides which
attempt is an episode's result (the first fully complete attempt by start
time), and `arm_table.arm_row` averages.

## Readers

| Reader | Input | Output |
|---|---|---|
| `benchmarks/run_record/rows.py` | One root's ledger | Nothing. It's the library every reader picks attempts with. |
| `run_report.py <root>...` | Each root in turn | SR, NE, SPL, OSR and the OSR x SR joint, SR by target shape, SR by referent bucket and by intel quality (Cal City; a recorded row's own class; a pick without one is bucketed by its recorded description and graded from the run's split), commit provenance, the bake check. |
| `arm_table.py <root>...` | Many roots, one per arm | One row per arm: effort, n, SR, OSR, shape and radius SR, NE. Refuses roots that mix success criteria, discs of two radii included. `--json` prints the rows. |
| `ab_report.py <a> <b>` | Two roots | Paired SR with a bootstrap interval on the difference, McNemar, and the flip counts. |
| `spend.py <root>` | Every attempt's transcript, flying or re-queued included | A hosted endpoint's tokens and USD, USD per scored episode, the projection for the selection's remainder, and every `served_model` that answered. Cache reads, cache writes, and the rest are priced apart, from each transcript's `cached_tokens` and `cache_write_tokens` (`maps.llm.flight_summary`), null where the serve reports none. |

`run_report`, `arm_table`, and `ab_report` refuse a root that scores fewer
episodes than its manifest selected (`selection.episodes_n`) and name the
shortfall: episodes that were only ever an infrastructure casualty, and
episodes that never landed a row. To finish the root, resume the run into it.
To read it as it stands, pass `--allow-partial`, which prints the shortfall
to stderr.

## Claims and their readers

The run roots are named as in the
[reproduction commands](../../../docs/experiments.md#reproduce-the-bench-experiments).
Run every command from the repository root.

| Claim | Quantity | Reader | Command |
|---|---|---|---|
| E2 SR and OSR per arm, CityNav `test_unseen`, 5,311 episodes | `SR_pct`, `OSR_pct`, `n` per arm | `arm_table` | `python -m benchmarks.citynav.analysis.arm_table output/e2/*` |
| E3 SR per effort rung and Gemma, Cal City reviewed 1,100 | `SR_pct` (shape criterion), `n` per arm | `arm_table` | `python -m benchmarks.citynav.analysis.arm_table output/e3/*` |
| E6 closed loop: SR (the commit point inside the target's mask), OSR | `sr`, `osr` of the summary | `benchmarks.flight.measures` | `python -m benchmarks.flight.measures --root <sorties>/* --truth <adjudications dir> <flight-week tasking JSON>` |
| E6 by intel quality: runs, SR, OSR | `by_intel_quality` | `benchmarks.flight.measures` | the E6 command; `--json` for the record |
| E6 per flight: runs, efforts, SR, OSR | the summary and `by_effort` (per model and effort) over that flight's sorties | `benchmarks.flight.measures` | the E6 command with `--root` naming that flight's sorties |
| E6 one run: NE, time to identification, flights and turns, SPL, AGL (the fix nearest the target's, else the median keyframe height, else the median commanded waypoint height), model and effort | the run's entry in `runs` (`ne_ft`, `time_to_id_s`, `flights`, `turns`, `spl`, `agl_ft`, `agl_source`, `model`, `reasoning_effort`) | `benchmarks.flight.measures` | the E6 command with `--json` over every sortie (SPL reads the turn radius the crews flew across all of them) |
