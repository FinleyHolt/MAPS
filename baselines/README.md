# Baselines under the harness

The two adapted baselines of the paper's controlled comparison, measured on
the substrate MAPS is measured on: GeoNav (arXiv 2504.09587; code
`github.com/Xhtshr/geonav-official`, MIT) and the ViSA-inspired policy (arXiv
2603.08007; no code release). Each fills the span between "episode built" and
"stops flown, target committed", and nothing else. `seam.py` is that
contract; `geonav/policy.py` and `visa/policy.py` are one method each, and
`geonav_released/` is GeoNav under its released token caps with thinking
off. The released checkpoints (CityNav MGP, HETT, FlightGPT) run through
their own evaluators instead: [`released/`](released/README.md).
`human/` puts a person at the same seam: [A human operator](#a-human-operator).

## What is held fixed

| part | value | where |
| --- | --- | --- |
| episode loop, selection, resume, rows | the bench's; a config's `policy:` routes the built episode to `seam.fly` in place of the mission conversation | `benchmarks/citynav/run_benchmark/episode_runner.py` (`run_episode`) |
| served model | discovered from `/v1/models` at startup, the same serve as the MAPS arm | `maps/llm/endpoint_spec.py` |
| sampling envelope | the served model's card (`maps.llm.model_cards.sampling_kwargs`) at each paper's temperature (`seam.TEMPERATURE`); an OpenAI-wire card sends no temperature | `seam.BaselineHarness.llm_json` |
| thinking dial | the request-side `reasoning_effort` per method (`seam.REASONING_EFFORT`); `None` takes the arm's `SEARCH_SAMPLING_REASONING_EFFORT`, as the mission does | `seam.BaselineHarness.llm_json` |
| sensor | the run's `sensor:`, north-up, rendered by the episode's `Hydrated.crop`, the call the mission makes; the view's ground footprint comes from the sensor's field of view | `seam.BaselineHarness.observe`, `seam.BaselineHarness.view_half_m` |
| start pose | the runner's: the dataset start (or the config's `spawn`), at the dataset's start z or the config's `start_altitude_agl_m` | `episode_runner.start_altitude_m` |
| prior | the OSM rows of the block; a landmark is a row with a `name` tag | `seam.BaselineHarness.landmarks` |
| detector | the run's own SAM 3.1 client | `episode_runner.make_perception`, `seam.BaselineHarness.detect` |
| budget | the config's `max_steps`, priced per teleport in the six-action space from the replay's start pose and heading | `seam.Budget` |
| replay and score | the stops and the commit, as flights, through `synthesize_compliant_trajectory`, scored by `eval.score_arrival` under the bench's criterion | `episode_runner.run_episode` |
| row | the MAPS row; `policy`, `commit_path` and `committed_fid` name the method, its commit tier and its commit, and the policy's own numbers ride as `metrics.<policy>_*`; `metrics.llm` counts every call, read off the same `flight/llm/transcript.jsonl` | `seam.fly` |

Each `configs/baselines/<policy>.yaml` is `base: ../citynav.yaml` plus
`policy:`, and each `configs/baselines/calcity_<policy>.yaml` is
`base: ../calcity.yaml` plus `policy:`;
`baselines/tests/test_baseline_configs.py` pins that each resolves to its
parent plus `policy` and nothing else. The bench loader refuses a `policy`
that names no `baselines/<policy>/policy.py`.

Every attempt directory carries `<policy>/`: each prompt, reply, render and
mark set the policy produced, `calls.json` (one entry per model call) and
`decision.json` (the commit and its tier), so a row can be read back call by
call. `python -m baselines.visa.figures <attempt dir>` draws the ViSA paper's
Fig. 1/2 content for one episode.

## Rerun

Serve the model at `SEARCH_REASONER_URL` as for a MAPS run, and fly each
baseline with the selection flags and reasoning effort of the MAPS root it is
read against. `CITYNAV_BENCH_CONFIG` names the config. Against the E2 roots
(CityNav `test_unseen`, the 1,000 episodes of seed 1234 that each E2 root holds):

```sh
for policy in geonav geonav_released visa; do
  SEARCH_SAMPLING_REASONING_EFFORT=<the MAPS root's effort> CITYNAV_BENCH_CONFIG=configs/baselines/$policy.yaml \
    python -m benchmarks.citynav.bench_fullset --split test_unseen --limit 1000 --sample-seed 1234 \
    --out-root output/baselines/<model>/$policy
done
```

Against the E3 roots (the 1,100 reviewed Cal City episodes):

```sh
for policy in geonav geonav_released visa; do
  SEARCH_SAMPLING_REASONING_EFFORT=<the MAPS root's effort> CITYNAV_BENCH_CONFIG=configs/baselines/calcity_$policy.yaml \
    python -m benchmarks.calcity.bench_fullset --episode-allowlist output/reviewed_episodes.txt \
    --out-root output/baselines/<model>/calcity_$policy
done
```

Read a root, pair it against the MAPS root on the shared episodes, and table
each benchmark's arms (the CityNav arms are scored on the ring, the Cal City
arms on the shape):

```sh
python -m benchmarks.citynav.analysis.run_report output/baselines/<model>/visa
python -m benchmarks.citynav.analysis.ab_report output/e2/<model's E2 root> output/baselines/<model>/visa --label maps_vs_visa
python -m benchmarks.citynav.analysis.arm_table output/baselines/<model>/{geonav,geonav_released,visa}
python -m benchmarks.citynav.analysis.arm_table output/e3/<model's E3 root> output/baselines/<model>/calcity_*
```

`ab_report` reads a baseline root as an arm of the MAPS root: the two
configs differ only in `policy`, which its config check sets aside as it sets
aside `withhold`. It still refuses a pair across splits, tile bakes, code
commits, served models or reasoning efforts. A `geonav_released` root's rows
and manifest stamp `none`, its dial off (`seam.dial_off`), so its pair against
a MAPS root runs only with `--allow-provenance-mismatch`.

## Cal City

The Cal City arms fly `configs/calcity.yaml` unchanged: the seeded uniform
spawn, 914.4 m above ground, the Oryx 2048x1536 sensor and the shape
criterion. Two things follow from the sensor, both read off it rather than
assumed:

* the ground footprint is the Oryx's 16.09 x 12.10 degree field of view, so
  a view is about 258 x 194 m at cruise, not the 2 x altitude square both
  papers assume. GeoNav's view-area map and ViSA's coverage grid size their
  squares from `view_half_short_m`, the footprint's short side;
* `max_steps: 100000` leaves each method's own loop bound in charge: GeoNav's
  20 timesteps and ViSA's 20-stop plan cap.

Both ports plan over `BaselineHarness.search_window`, a square centred on
the start whose half-side reaches the farthest matched landmark, held to
512–1,280 m. Every CityNav block is narrower than 512 m, so on CityNav the
window is the block. On the 13 km Cal City block it bounds what a policy
allocates: GeoNav's map and ViSA's coverage grid sized from the whole block
need more than 16 GB per episode. A Cal City row that starts more than about
1.3 km from its target plans over a window that may not hold the target.

## A human operator

`human/policy.py` puts a person behind the seam. Each episode serves a
console page at `http://$HUMAN_CONSOLE_HOST:$HUMAN_CONSOLE_PORT/` (default
`127.0.0.1:8077`) and blocks until the operator commits. The page shows the
instruction, the run's north-up frame at the current pose with every named
OSM landmark in view outlined, the action budget left, and a search over the
block's named landmarks with their offsets. A click on the frame, or a
selected landmark's fly-over, is one stop priced by `seam.Budget`; a stop the
budget cannot afford is refused. The commit is the clicked point, or the
current position. No detector runs. Every action and its time land in
`<attempt>/human/session.json`, with one frame per stop.

The arm calls no model and no segmenter. Pin the endpoint the paired MAPS
root served, so the manifest and `ab_report`'s config check match, and the
driver probes nothing:

```sh
SEARCH_ENDPOINT_OFFLINE=1 SEARCH_ENDPOINT_SPEC_OVERRIDE='<the MAPS root's model>:<max_model_len>' \
  CITYNAV_BENCH_CONFIG=configs/baselines/human.yaml \
  python -m benchmarks.citynav.bench_fullset --split test_unseen --episode-allowlist <list> \
  --workers 1 --episode-timeout-s 1800 --out-root output/baselines/human/<participant label>
```

On a compute node, forward the port to the operator's machine
(`ssh -L 8077:127.0.0.1:8077 <node>` through the submit host). Out-roots
carry a participant label, never a name.

## The human study

Three arms on the same episodes of one MAPS root: MAPS alone (the root),
a person alone (`human/policy.py`, above), and the team, where MAPS flies
alone and a person audits the finished episode (`human/audit_console.py`).
The auditor sees the instruction, the imagery around MAPS's commit with
every object MAPS detected outlined and its pick marked, MAPS's reasoning
for the pick, its full trace and frames, and a search over the named prior.
The auditor confirms MAPS's target or selects another object or point. The
console reads the instruction, the commit and the attempt's own records,
never the row's target or metrics, and shows no score at any point.

1. Deal the episodes. A participant flies one pool of blocks alone and audits
   the other, never one block in both arms, and meets each target object at
   most once per arm. Pools and arm order alternate across participants.

   ```sh
   python -m baselines.human.session_plan --maps-root output/e2/<root> \
     --participants p01,p02,p03,p04 --per-arm 20 --out output/human_study/<study>
   ```

2. Fly each participant's `solo.txt` with the human operator (above), and
   audit their `team.txt`, in the order `plan.json` names:

   ```sh
   python -m baselines.human.audit_console --maps-root output/e2/<root> \
     --episode-allowlist output/human_study/<study>/p01/team.txt --out output/human_study/<study>/p01
   ```

   The console serves `http://127.0.0.1:8078/`, appends each decision to
   `<out>/decisions.jsonl`, and resumes after the last one when restarted.
   It reads the block's imagery (`SEARCH_CITYNAV_RGBD_DIR`) and its tile
   bake (`SEARCH_CITYNAV_TILES`) as the bench does.

3. Score the audits. A confirmed episode is the MAPS row unchanged; a
   replaced one flies one more leg, from where MAPS ended to the auditor's
   point, priced against the budget MAPS left (the console refuses a point
   the budget cannot reach), and is scored there.

   ```sh
   python -m baselines.human.team_rows --maps-root output/e2/<root> \
     --decisions output/human_study/<study>/p01/decisions.jsonl --out-root output/baselines/audit/p01
   python -m benchmarks.citynav.analysis.ab_report output/e2/<root> output/baselines/audit/p01 --label maps_vs_team
   ```

   Each team row carries `metrics.maps_SR` beside `SR`, and
   `audit_decision`, `audit_review_s` and `audit_trace_turns_opened`.
   A Cal City replacement is refused: its shape criterion needs the target
   footprint, which `team_rows` does not load.

## How a baseline row can differ from a MAPS row

* A model call that fails past the transport's own retries ends the episode.
  The bench records a transient fault as an infra row outside SR and a
  refusal as a scored miss; the MAPS reasoner loop retries a transient
  round instead. Read `run_report`'s infra count beside SR.
* A baseline row carries no `metrics.mission` block: no mission conversation
  ran. `committed_provenance` reads `point`, the commit being a world point.
* The reshaping call and every `geonav_released` call run with the dial off
  (`seam.dial_off`): `reasoning_effort: none` where the model card lists that
  rung (Inkling Small, GPT-5.6 Sol), else `chat_template_kwargs:
  {enable_thinking: false}` where the template reads that switch (Qwen 3.8
  Flash Next). A baseline arm on a model with neither (the Gemma 4 cards) is
  refused when the driver starts, before any episode (`seam.NoDialOff`).

## The serve fact every baseline call answers to

| fact | measured | what the seam does |
| --- | --- | --- |
| an output grammar applies from the first token, so a `json_schema` call never opens the reasoning span | a recorded GeoNav action call: 49 completion tokens and no reasoning under `json_schema` strict; 8,192 tokens of reasoning and no answer without it | every call is free of grammar first (the model reasons, then answers in the JSON shape the prompt asks for); an answer that is not that JSON is reshaped by a second, constrained call with the dial off; a first call whose cap the reasoning exhausted falls back to the constrained call and is counted under `metrics.<policy>_reasoning_truncated`. `metrics.llm.reasoning_tokens` is the proof the dial applied. |

## Substitutions, per method

Each row is one place the port differs from the paper or the released code,
with the reason. Anything not listed is verbatim.

### GeoNav

The port follows the released code, `Xhtshr/geonav-official` (commit of
2026-07-06, MIT), not the paper's prose where the two differ; each such place
is marked `# released:` at the line. Substitutions are marked `# substitution:`.

| upstream | this port | why |
| --- | --- | --- |
| `agent.py:1645-1648`: `subtask_status` never resets, so the Navigate system prompt serves the whole episode | replicated | the released behaviour is the baseline |
| `city_scene_graph.py:174-201`: the merge score compares node kind, not class (α 0.95, ρ 4 m, threshold 0.8), so two object nodes merge only within ~1 m | replicated; the cross-kind semantic term is 0.0 where upstream loads MiniLM | `sentence-transformers` is not a dependency; the term moves the cross-kind merge radius by 7 cm |
| `llm_controller.py:475-477`: the complex-query path raises on a one-vs-two return mismatch and falls through to the operation-chain path | the chain path is called directly and the fall-through recorded | the same control flow without replaying dead calls |
| `city_scene_graph.py:1024-1042` and `agent.py:1743-1747`: two relation validators coerce in opposite directions (`adjacent_to`/`near_corner` → `contains`, and back) | both replicated | |
| `agent.py:989-1024`: `max_tokens=800` on the planner, action and chain calls; the extractor uncapped; DashScope `thinking: disabled` | two arms. `geonav` (headline): the serve's reasoning rung, the served output reserve (`EndpointSpec.max_output_tokens`, the MAPS mission's own cap) on every call. `geonav_released`: the dial off (`seam.dial_off`), 800 on those three calls and the landmark-extraction call, 2,048 on the extractor | the controlled table holds the model and its reasoning rung fixed across methods; thinking tokens count against `max_tokens` here, so the released 800 truncates a reasoning call to an empty object (measured on a recorded run). DashScope's default output cap bounded the released extractor at 2,048 |
| `landmark_map.py:55-65`: per-episode landmark strings from `processed_descriptions.json`, matched by nearest Levenshtein name with no floor | one extraction call on the instruction (the paper's GPT-3.5 step) and `seam.match_landmarks` on distinctive tokens with a 0.6 floor | the file is not on disk; whole-name matching sent four of sixteen places to the wrong landmark on the first audit |
| `agent.py:1488`: Navigate ends when every landmark's centre is within 40 m; `get_enhanced_geo_relation` reports distance to the centre | distance to the landmark's area (`geonav.policy.landmark_distance`), 0 inside it | an OSM street unioned and buffered has a centroid hundreds of metres from every point of it, so Navigate never ended on a street landmark |
| GroundingDINO + MobileSAM, two passes per step (target, surroundings), mask pixels painted onto the map | one SAM 3.1 pass over the target and surrounding captions, split by label; the frame box's corners → world polygon painted; the 50 m / 3000 m² filters kept | the substrate's detector; the map scale is ~1.2 px/m |
| CityRefer landmark ids `f"{object_type}_{name}"` | the OSM row's bake category and name | the tenet: no CityRefer name reaches a prompt |
| upstream `cropclient` 1024² crop at a fixed 50 m; extractor boxes read as pixels of that crop | the run's sensor frame (1008x1008 on CityNav, 2048x1536 on Cal City) at the runner's start altitude; the extractor is told the frame's pixel size ahead of the verbatim prompt, and a box whose four values all lie in [0, 1] is read as fractions of the frame | the protocol held fixed across methods; the served model answered in unit fractions on the smoke, which read as pixels would put every node in the frame's corner |
| the image part before the text part | the harness's user message puts text first | one message builder for both baselines |
| `selected_pos` used verbatim; teleports never clamped | both clamped to the block bounds | an ortho crop outside the block is empty |
| a Locate result becomes the pose | the Locate position is the commit and a priced teleport; unaffordable, it stays the commit and the replay's reserve flies it | the harness's budget and replay |
| `ExplorationAnalyzer` hue entropy via scikit-image | `matplotlib.colors.rgb_to_hsv` | not a dependency |
| a raw-frame PNG per step | map renders as JPEG and every prompt/reply per call under `<workdir>/geonav/` | |
| the cognitive map spans the block (480 px over 410 m) | the map spans the search window at the same resolution, and teleports clamp to it | the window is the block on CityNav; the 13 km Cal City block sized the map past 16 GB |

Commit tiers: `geonav:graph` (a scene-graph node answered the query),
`geonav:vlm_pos` (the Locate prompt's `selected_pos`), `geonav:locate_pose`
(a Locate answered no position; the pose is the commit), `geonav:budget` (the
budget ended the walk), `geonav:timeout` (the walk ended, at twenty
iterations or the plan's last subgoal, without a Locate).

### ViSA

| paper | this port | why |
| --- | --- | --- |
| VPG: Qwen3-VL-PLUS proposes open-vocabulary boxes | SAM 3.1 (`harness.detect`) proposes masks for the analysis call's concept phrases; the VLM verifies by mark id on the SoM frame | a live probe on the serve put the model's own boxes nowhere near a 45 px target in a 2016 px frame (the paper tree's frame); the substrate's detector is the one the headline arm uses |
| stop: pixel centroid unprojected through intrinsics and altitude | the SAM mask's own ortho unprojection (`DetectedObject.world_xy`) | the exact form of the same operation on this sensor |
| guidance signal `g` as prose, "passed back to the Perception Phase as the textual prompt for the subsequent detection round"; the worked example rejects a candidate, sends `g` and "triggers a move primitive", and the next waypoint's VPG "adjusts its attention based on the feedback" | the VM's decision is one of the paper's four primitives with `g` beside it; `g` (prose plus `guidance_concepts`, the noun phrases a detector can act on) is the prompt of the next perception round wherever the executor goes, a waypoint after `move` or the new height after `ascend`/`descend` | a detector, not a VLM, acts on the signal here; re-detecting at the same pose would contradict the paper's example, which moves before the next round |
| landmark prior from CityRefer (name, centroid, contour) | OSM rows with a `name` tag, matched by `seam.match_landmarks` on the names' distinctive tokens (ratio ≥ 0.6) | the tenet: no CityRefer name reaches a prompt; matching on whole names sent "Helmore building" to Mumford Building |
| the VM's stated inputs are the SoM frame and the instruction; landmarks reach it as marks (the depot in Fig. 2 is a detection box) | the matched landmarks also reach the VM as text (`VERIFY_PROMPT`'s `K_prior` block: each landmark's centroid offset from the view centre and whether its outline is in view) | the geographic-boundary stage needs the landmark's position when no mask marks it; the paper's Executor holds K_prior and the paper does not say how the VM sees it |
| landmark contours: CityRefer's paved surfaces and footprints | a street's OSM centreline widened to its carriageway (`width` tag, else `lanes` at 3.25 m, else a half-width by highway class, 5 m residential); a point feature a 10 m disc; a building its footprint | a 6 m ribbon never contained a parked car, so "on Willmore Road" could not hold where CityRefer's polygon held it |
| prompts: none published | `visa/prompts.py`, written from the paper's three-stage description; the paper's example guidance sentence is quoted as the shape of `guidance` | |
| step budget `T` unstated; altitude-change cap unstated | the config's `max_steps` through `seam.Budget`; ≤ 1 altitude change per waypoint | the harness budget; the cap bounds a loop the paper leaves open |
| waypoint count unstated | the greedy cover stops at `MAX_STOPS = 20` waypoints (`visa/waypoints.py`) | bounds the plan on a block-sized landmark union; the paper states no cap |
| the merged region is the matched landmarks' union, the block when none | the union inside the search window, the window when none | a city-long road or the whole Cal City block made a coverage grid past 16 GB |
| stage 1 "defers decisions by marking targets as pending when visual evidence remains insufficient" | `pending_marks` beside `candidate_marks`; a pending mark is a fallback candidate below every attribute-matched one | the paper says pending exists, not what it feeds |
| the instruction's landmarks and target phrase reach the Executor and the VPG ("the Executor extracts the geographic contour of 'The Tram Depot' from the prior database"), by an unstated mechanism | one instruction-analysis call (`ANALYSIS_PROMPT`) yields the target phrase, its class and attributes, the landmark names and the surrounding phrases | the paper names the outcome, not the step |
| contour merging "identif[ies] intersection areas as priority search regions" | not implemented; the order is the TSP from the start over the greedy cover, which the same paragraph also prescribes | how priority and the TSP combine is unstated |
| waypoint generation is introduced "for instructions involving complex spatial constraints" | every episode plans coverage over its matched landmarks | what a simple instruction does instead is unstated |
| no-stop fallback unstated | the strongest pending candidate (attribute-matched, most stages passed, then confidence) → `visa:pending`; none → the last stop → `visa:exhausted`; the budget ending the walk → `visa:budget` | every episode commits something the ring can score, and the tier says how |
| `max_tokens` unstated | the served output reserve (`EndpointSpec.max_output_tokens`, reasoning included) on the analysis and verification calls | the reasoning span is never cut short of the answer |
| the VPG proposes boxes for the instruction's own phrases ("red car", the depot) | SAM 3.1 is prompted with the attributed target phrase, the surrounding phrases and the landmark nouns, the bare class (or its plural) only for a target with no phrase; up to 80 masks are marked (`MAX_MARKS`), the described object's class first, surroundings next, landmark-class masks last, each tier by confidence | a bare class returned every car in view, 80 marks on a car instruction; the probe on twenty first frames measured the verifier hitting 5 of 18 with fifteen class marks against 2 of 18 with eighty |
