# benchmarks/citynav/analysis

The readers every reported number is derived with. Each reads run roots, the
directories a bench run writes (`run_manifest.json` beside `episodes.jsonl`),
and none defines a metric of its own: `rows.load_attempts` decides which
attempt is an episode's result (the first fully complete attempt by start
time) and `arm_table.arm_row` averages.

`run_report`, `arm_table`, `ab_report` and the figures refuse a root that
scores fewer episodes than its manifest selected (`selection.episodes_n`) and
names the shortfall: episodes only ever an infrastructure casualty, and
episodes that never landed a row. Resuming the run into the same root finishes
it; `--allow-partial` reads it as it stands and prints the shortfall to
stderr.

| Reader | Reads | Prints |
|---|---|---|
| `rows.py` | One root's ledger | Nothing: the library every reader picks attempts with. |
| `run_report.py <root>...` | Each root in turn | SR, NE, SPL, OSR and the OSR x SR joint, SR by target shape, SR by referent bucket and by intel quality (Cal City; a recorded row's own class; a pick without one is bucketed by its recorded description and graded from the run's split), commit provenance, the bake check. |
| `arm_table.py <root>...` | Many roots, one per arm | One row per arm: effort, n, SR, OSR, shape and radius SR, NE; refuses roots that mix success criteria, discs of two radii included. `--json` for the rows. |
| `ab_report.py <a> <b>` | Two roots | Paired SR with a bootstrap interval on the difference, McNemar, and the flip counts. |
| `spend.py <root>` | Every attempt's transcript, flying or re-queued included | A hosted endpoint's tokens and USD (cache reads, cache writes and the rest priced apart, off each transcript's `cached_tokens` and `cache_write_tokens` (`maps.llm.flight_summary`), null where the serve reports none), USD per scored episode, the projection for the selection's remainder, and every `served_model` that answered. |

The figures in `figures/` read the same rows: `capability_effort_arms` adds
the cost of the same picked attempts (`metrics.llm`) and the arm's identity
off its run manifest.

## Every claim and its reader

The run roots are named as in the
[README's reproduction commands](../../../README.md#reproduce-the-experiments).
Every command runs from the repository root.

| Claim | Quantity | Reader | Command |
|---|---|---|---|
| E1 SR per model, CityNav `train_seen`, 1,000 episodes | `SR_pct`, `SR_n` per arm | `arm_table` | `python -m benchmarks.citynav.analysis.arm_table output/e1/*` |
| E1 OSR per model | `OSR_pct` per arm | `arm_table` | `python -m benchmarks.citynav.analysis.arm_table output/e1/*` |
| E1 outcomes per model: reached (OSR), inside (SR 20 m) | `overall OSR` and the `OSRxSR` line | `run_report` | `python -m benchmarks.citynav.analysis.run_report output/e1/<model>` |
| E2 SR and OSR per arm, CityNav `test_unseen`, 5,311 episodes | `SR_pct`, `OSR_pct`, `n` per arm | `arm_table` | `python -m benchmarks.citynav.analysis.arm_table output/e2/*` |
| E3 SR per effort rung and Gemma, Cal City reviewed 1,100 | `SR_pct` (shape criterion), `n` per arm | `arm_table` | `python -m benchmarks.citynav.analysis.arm_table output/e3/*` |
| E3 effort-ladder figure: SR and median NE against effort | `SR_pct`, `NE_med` per arm with a numeric effort | `figures.plot_effort_ladder` over `arm_table` | `python -m figures.plot_effort_ladder output/e3/inkling-small-e*` |
| E4 SR with parts withheld, per model (each arm's `withhold:` list, names in `maps/search/withhold.py`) | `SR_pct` per arm, labelled `withhold=` and the arm's parts, comma-joined | `arm_table` | `python -m benchmarks.citynav.analysis.arm_table output/e1/<model> output/e4/<model>/*` |
| E4 change in SR against the full harness | paired delta, its interval, flips | `ab_report` | `python -m benchmarks.citynav.analysis.ab_report output/e1/<model> output/e4/<model>/<arm>` |
| E5 Cal City SR against prompt plus completion tokens per episode | `sr`, `total_tok_mean` per arm | `figures.capability_effort_arms`, `figures.plot_capability_effort` | `python -m figures.plot_capability_effort --split calcity --cost total output/e3/*` |
| E5 CityNav `test_unseen` SR against completion tokens per episode | `sr`, `completion_tok_mean` per arm | `figures.capability_effort_arms`, `figures.plot_capability_effort` | `python -m figures.plot_capability_effort --split test_unseen output/e2/*` |
| E5 the numbers behind either figure | every arm's record | `figures.capability_effort_arms` | `python -m figures.capability_effort_arms --out output/arms.json output/e2/* output/e3/*` |
| E7 Inkling CityNav ladder: SR against effort for MAPS, ViSA and GeoNav | `SR_pct`, `NE_med` per arm with a numeric effort | `arm_table`, `figures.plot_effort_ladder` | `python -m figures.plot_effort_ladder output/e7/inkling-small-maps-e*` |
| E8 baselines at each Qwen effort, paired with E2 | paired delta, its interval, flips | `ab_report` | `python -m benchmarks.citynav.analysis.ab_report output/e2/qwen38-flash-next-<effort> output/e8/qwen38-flash-next-<policy>-<effort>` |
| E9 the released baselines' protocol (100 actions, 50 m start) | `SR_pct`, `OSR_pct` per arm; paired SR against a released run | `arm_table`, `baselines.released.outcome_rows pair` | `python -m baselines.released.outcome_rows pair output/released_baselines/<run> output/e9/<model>` |
| E6 closed loop: SR (the commit point inside the target's mask), OSR | `sr`, `osr` of the summary | `benchmarks.flight.measures` | `python -m benchmarks.flight.measures --root <sorties>/* --truth <adjudications dir> <flight-week tasking JSON>` |
| E6 by intel quality: runs, SR, OSR | `by_intel_quality` | `benchmarks.flight.measures` | the E6 command; `--json` for the record |
| E6 per flight: runs, efforts, SR, OSR | the summary and `by_effort` (per model and effort) over that flight's sorties | `benchmarks.flight.measures` | the E6 command with `--root` naming that flight's sorties |
| E6 one run: NE, time to identification, flights and turns, SPL, AGL (the fix nearest the target's, else the median keyframe height, else the median commanded waypoint height), model and effort | the run's entry in `runs` (`ne_ft`, `time_to_id_s`, `flights`, `turns`, `spl`, `agl_ft`, `agl_source`, `model`, `reasoning_effort`) | `benchmarks.flight.measures` | the E6 command with `--json` over every sortie (SPL reads the turn radius the crews flew across all of them) |
