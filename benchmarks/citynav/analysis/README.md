# benchmarks/citynav/analysis

The readers every reported number is derived with. Each reads run roots, the
directories a bench run writes (`run_manifest.json` beside `episodes.jsonl`),
and none defines a metric of its own: `rows.load_attempts` decides which
attempt is an episode's result (the first fully complete attempt by start
time) and `eval.aggregate` averages.

Every reader refuses a root that scores fewer episodes than its manifest
selected (`selection.episodes_n`) and names the shortfall: episodes only ever
an infrastructure casualty, and episodes that never landed a row. Resuming the
run into the same root finishes it; `--allow-partial` reads it as it stands
and prints the shortfall to stderr. A sortie root selects no episodes up front
and is read as it stands.

| Reader | Reads | Prints |
|---|---|---|
| `rows.py` | One root's ledger | The canonical attempt pick, shared by every reader. |
| `run_report.py <root>...` | Each root in turn | SR, NE, SPL, OSR and the OSR x SR joint, SR by referent bucket, commit provenance, the bake check. |
| `arm_table.py <root>...` | Many roots, one per arm | One row per arm: effort, n, SR, OSR, shape and radius SR, NE; refuses roots that mix success criteria. `--json` for the rows. |
| `ab_report.py <a> <b>` | Two roots | Paired SR with a bootstrap interval on the difference, McNemar, and the flip counts. |

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
| E3 SR per effort rung and Gemma, Cal City reviewed 603 | `SR_pct` (shape criterion), `n` per arm | `arm_table` | `python -m benchmarks.citynav.analysis.arm_table output/e3/*` |
| E3 effort-ladder figure: SR and median NE against effort | `SR_pct`, `NE_med` per arm with a numeric effort | `figures.plot_effort_ladder` over `arm_table` | `python -m figures.plot_effort_ladder output/e3/inkling-small-e*` |
| E4 SR with one part withheld, per model (OSM prior, `look_for`, `look`, images, MISSION MAP, system turn) | `SR_pct` per arm, labelled `withhold=<part>` | `arm_table` | `python -m benchmarks.citynav.analysis.arm_table output/e1/<model> output/e4/<model>/*` |
| E4 change in SR against the full harness | paired delta, its interval, flips | `ab_report` | `python -m benchmarks.citynav.analysis.ab_report output/e1/<model> output/e4/<model>/<part>` |
| E5 Cal City SR against prompt plus completion tokens per episode | `sr`, `total_tok_mean` per arm | `figures.capability_effort_arms`, `figures.plot_capability_effort` | `python -m figures.plot_capability_effort --split calcity --cost total output/e3/*` |
| E5 CityNav `test_unseen` SR against completion tokens per episode | `sr`, `completion_tok_mean` per arm | `figures.capability_effort_arms`, `figures.plot_capability_effort` | `python -m figures.plot_capability_effort --split test_unseen output/e2/*` |
| E5 the numbers behind either figure | every arm's record | `figures.capability_effort_arms` | `python -m figures.capability_effort_arms --out output/arms.json output/e2/* output/e3/*` |
| E6 closed loop: SR (the commit point inside the target's mask), OSR | `sr`, `osr` of the summary | `benchmarks.flight.measures` | `python -m benchmarks.flight.measures --root <sorties>/* --truth <adjudications dir> <flight-week tasking JSON>` |
| E6 by intel quality: runs, SR, OSR | `by_intel_quality` | `benchmarks.flight.measures` | the E6 command; `--json` for the record |
| E6 per flight: runs, efforts, SR, OSR | the summary and `by_effort` (per model and effort) over that flight's sorties | `benchmarks.flight.measures` | the E6 command with `--root` naming that flight's sorties |
| E6 one run: NE, time to identification, flights and turns, SPL, AGL (the fix nearest the target's, else the median keyframe height, else the median commanded waypoint height), model and effort | the run's entry in `runs` (`ne_ft`, `time_to_id_s`, `flights`, `turns`, `spl`, `agl_ft`, `agl_source`, `model`, `reasoning_effort`) | `benchmarks.flight.measures` | the E6 command with `--json` over every sortie (SPL reads the turn radius the crews flew across all of them) |
