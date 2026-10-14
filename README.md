# MAPS: Multimodal Agentic Planning and Search

MAPS is a method for aerial language-guided target search. An operator
describes a target in words ("the car parked north of the church"), and a
multimodal language model flies an aircraft to find it, reasoning over an
OpenStreetMap prior, the imagery its own flights collect, and an open-vocabulary
segmenter.

This repository holds the method, the two benchmarks it is measured on
(CityNav and Cal City), the flight harness that flew it in closed loop on a
C-12 over California City, and the readers that derive every reported number
from run records.

## The method

A search is one mission conversation with one reasoning model. The model
receives a system turn and the operator's words, and acts through exactly one
tool, `run`, whose argument is a block of ordinary Python. The block executes
in a namespace that persists for the whole mission, against a library of
calls over the world state:

| Call | What it does |
|---|---|
| `find` | Reads rows of the world state (the OSM prior plus every detection) by name or class, ordered or cut by place, side and colour. |
| `look_for` | Names up to four classes the segmenter detects on every later flight. |
| `go_to` | Flies to one thing on the map. |
| `explore` | Flies a covering pass over a thing or a set, segmenting at every stop, and returns the rows it found. |
| `look` | Returns a close crop of the sensor's pixels at a row, as a picture. |
| `show_map` | Returns the world-state map around a place, drawn as a schematic. |
| `stand` | Adopts a ground viewpoint, without flying, that later rows and pictures face. |
| `more_of` | Returns the next block of a long result. |
| `intersection` | Returns where two roads cross. |
| `ends_of` | Returns where a road ends, one row per terminus. |
| `distance` | Returns the metres between two footprints. |

A result is a set of rows with `.filter` (a cut) and `.order` (one sort
key). A bare handle on its own line shows its rows; `print()` and the block's
last value come back in the tool result, the receipt. Only `go_to` and
`explore` move the aircraft.

Perception is SAM 3.1 run over an orthographic observation: each flight's
view is a north-up orthophoto of the ground under the aircraft (a crop of a
raster on the benchmarks, fused camera frames in flight), segmented for the
classes the model armed with `look_for`. Detections join the OSM prior in one
world-state graph.

The mission ends when the model replies without a tool call. The commit is
the target of the mission's last flight.

The code lives in [`maps/`](maps/README.md). Print the served contract (the
system turn and the tool schema the model receives) with:

```sh
python -m maps.search.parse_compile.contract --sensor oryx_orx32_25mm_nadir   # --agl-m <m>, default configs/calcity.yaml's 914.4
```

## The closed-loop flight

MAPS flew 87 closed-loop runs on a C-12 over California City, CA, from 11 to
18 September 2026, at about 3,000 to 6,500 ft above ground. The reasoner ran off
board over the datalink; SAM 3.1 ran on board. The station's method half is
[`benchmarks/flight`](benchmarks/flight/README.md) in this repository.

The flown engine is this repository's engine. The station served the same
text (system turn, tool schema and every string the model reads) on every
run from 14 September on. The engine here differs from the flown one in
these places, the served-text changes among them:

| Difference | Effect on a flight |
|---|---|
| A block whose Python parses but fails to compile (a repeated keyword argument) comes back as a tool error the model reads. | The flown engine ended such an episode as an infrastructure failure. |
| A receipt that prints a row's type names the `maps` package. | The flown receipt named the flown package. |
| The aircraft is reached through the `Vehicle` protocol (post a leg, read poses and recorded frames). | The station spoke to the recorder and cockpit display over HTTP with the same semantics. |
| The bench ablation withholds (`maps/search/withhold.py`) exist; the flight mission refuses a run that names one. | None; a flight serves every part. |
| A mission that raises before its first report is out of the rates only on a serve fault (the bench's infra rule: a segmenter outage, a transient shared-server fault, or a reasoner that does not answer discovery); any other failure is a scored miss carrying its error. A mission that ends after a report without a new one (a crew Stop, the turn cap, an infeasible compaction, or a failure) is scored at its last report (that reply's end, time and flights stand; a failure's error is recorded), never infra. A config file the flight refuses is refused before the sortie folder is written. | The flown station marked every failed mission as infrastructure, out of every rate; a follow-up tasking that ended without a new report replaced the report with that ending. |
| A Cal City bench episode succeeds when its commit point lies inside the target's mask as drawn, by the even-odd test the flight measures use, so a self-crossing mask keeps every lobe. | The flown sweep scored the six-action replay's final pose against the mask repaired with `buffer(0)`, which drops lobes of a self-crossing mask (4 reviewed masks), widened to a 5 m floor. |
| The bench driver's wait for a down reasoner ends with `--max-wall-s` or an abort. | The driver waited for the reasoner indefinitely. |
| The Cal City bench reads a user-supplied map raster and derives the reviewed episode set; its `look` caption states the raster's own ground sample distance. | Bench only; flight captions state the live imager's. |
| The table legend under a response's first table explains `distance to`, `facing`, `<dir> of` and bearings. | The flown legend also explained a line chart's `along <name>` and lateral offset, columns no op attaches. |
| A refusal that lists present numeric keys names only the columns the model is served; an off-palette `color_*` key gets the palette refusal; every op resolves `items` by one rule; a refusal carries no op-signature tail. | Flown refusals listed unserved keys, a retired `color_score` name, per-op item rules, and `find`'s ignored `color`/`provenance` arguments, and a grounding refusal ended with the whole `find` signature. |
| A `view` watchlist change and a placed boost's concepts reach the segmenter in flight as on the bench. | The flown station had no watchlist channel; those requests never reached SAM. |
| A cancelled reasoner call is shut down on the wire, and its answer is never used. | The flown station left a cancelled call's connection open, and a late answer to it could still be used. |
| A python set or list of rows in a place slot is the union of those rows, and an unnamed union member keeps its own id; a refusal quotes a user value cut to 120 characters. | The flown executor failed those programs ("program failed: 'fid'", "cannot pickle") and quoted values whole. |
| A `look` caption's no-data warning tests the subject's own row of the north-up validity raster. | The flown caption tested the row mirrored about the crop's centre, so a subject near a data edge could be warned about wrongly or not at all. |
| A repeated signed op error carries the repeat count and live census like any other error; a `run` call whose arguments do not parse is answered with the parse error; a quoted `[x, y]` boost target addresses the flown stop at that point; `rows.filter(pred, <cardinal>=…)` refuses instead of ignoring the cardinal; the `move` refusal names what `move` takes; a gapless `filter(near=)` points at `rows.order(nearest=)`. | The flown loop cut the repeat facts off signed errors, answered an unparseable call as an empty block, treated a quoted point as a fid, ignored a cardinal beside a predicate, offered staged handles to `move`, and sent a gapless near-filter to a refused `find(near=)`. |
| The segmenter receives every frame losslessly. | From 14 September the flown station sent each frame to SAM as a JPEG at quality 92 through a ground tile proxy; the lossy wire moved about one detection in eight. |
| A `/segment` request the segmenter refuses (a 4xx other than 408 or 429) raises at once. | The flown client dropped that tile and the flight carried on without its detections. |
| The mosaic's similarity sweep keeps the coarse yaw when no finer yaw reads better, and marks imaged only the ground a frame's quad lands on. | The flown mosaic could adopt a worse fine yaw, and marked a frame's warp border as imaged, so a `look` caption under-reported unobserved ground. |
| A prior with no footprint cells is skipped from a stop's prior overlay. | The flown station dropped the stop's whole prior overlay. |
| A truncated result head names `more_of(<rN>)` as the call that pages it; the paging, union-size, filter-predicate and unknown-keyword refusals name the methods and arguments the surface serves. | The flown texts named calls and arguments the surface does not serve. |
| The sensor block states the footprint at the height the mission commands (a flight `Mission` takes `agl_m`; the bench uses its start height). | The flown text quoted the 6,000 ft AGL band (~517 x 388 m) on every run, whatever the aircraft flew. |
| The segmenter client has one retry layer: a connect error, 408, 429 or 5xx moves to the next replica after 3, 6 and 12 s (four attempts); a read timeout is never resent to the replica that timed out; one 900 s deadline bounds a tile. The serve owns its score floor (0.3) and resolution (1008); the request carries neither. | The flown client nested a urllib3 retry ladder inside a three-attempt loop, so a tile could be sent twelve times and resent after an 11-minute read timeout; the request carried the floor and resolution, which the serve read only on its first call. |
| A leg whose ground the index holds but whose frames cannot be fetched is flown again; only frames recorded after the leg is posted arrive it and are fused. Arrival is asked of the recording at every poll, with or without an ownship pose. | The flown harness counted the old frames, so such a leg arrived at once and served no observation; a leg flown with no readable pose waited out the arrival timeout. |
| A recorded camera frame that does not decode is refused by name and counts as an unplaced frame; a waypoint's observation is the pass its flight fetched, and a flight whose ground no recorded frame served says so ("no observation: the aircraft's recording served no frame of this flight's ground"). | A frame that did not decode killed the pose-correction thread or raised out of the flight; an empty pass was fetched and fused a second time, and the empty flight read "no live observation: the camera or the receiver has nothing to pose". |
| `look_for` refuses a class that is not a string (`look_for("house", "pool")`); every filter refuses a predicate beside a cut keyword in one text. | The flown surface armed the segmenter on the printed list (`look_for(["car"])` watched `"['car']"`), and two of the three filter receivers named a shorter keyword list. |
| The terrain reads the DEM cell a point lies in, counted from the raster's north-west corner. | The flown station rounded from the south-west corner, reading a neighbouring cell for most points (typically centimetres, up to about 11 m). |
| A mission records the aircraft's track (`flight_track.jsonl`, one fix a second, each read through the boresight's timing offset like the start's UTC) and stamps every row on the vehicle's clock, so a MAPS sortie is scored like a flown one. | The flown station's track came from the ground recorder beside it; the harness itself wrote none. |
| The pose corrector's refusal of a frame does not replace the reason a leg row quotes. | The flown station's corrector thread could overwrite it. |
| Hints, the `more_of`, index and tied-view refusals, a map's badged set, a looked slice and an empty look name a result as the model named it (its variable, else its own text); a step whose rows a later op in the same call cut, sorted or measured serves only its notes that do not describe those rows; compare's containment note, a fact of each row it names, still rides; a folded sibling keeps its own status line under its name and drops only the lines the answer already states; cap notes name `look()` and `show`; the crop cap is stated once; `go_to` refuses a python set as it refuses a list. | The flown texts named internal result refs (`r0`, `r11`) the model's code could not use, quoted a keyword-only find as `find('')`, repeated step hints, served a cut or re-sorted find's match count and Best/Head list on an answer holding other rows, named nonexistent `view`/`crops`/`analyze` calls, stated the crop cap twice, and answered `go_to(set)` with a raw error. |
| The map legend names the 90-120 degree hue bucket yellow-green, the colour it paints. | The flown legend called it olive. |
| Every read of a result re-derives it over the current surface, so `look`, `explore` and the map's badges see a flight's rows in the first block after it. | The flown loop re-derived a result on such a read only after something else in the block had read one; otherwise that first read served the rows from before the flight. |
| A row and a python list of rows refuse a cut or a sort with the result set's own texts (a non-callable predicate or key, a reversed direction, two keys, an order with no key, whose refusal names nearest=, color=, direction= and a key function); a predicate beside a numeric within= gets the metres refusal; print in a block takes end=, sep= and flush=, and its 60-line cap counts printed lines; a slice of one row (`r[0:1]`) is a list of rows that chains `.filter` and `.order`. | A row and a list raised raw TypeErrors or silently accepted these; print refused flush= and ignored end=, and the flown cap counted print calls, so one print of joined lines passed whole up to 8,000 characters; a one-row slice was a plain list, so `.filter` or `.order` on it raised AttributeError. |
| Endpoint discovery that gets no answer, a 5xx, a 402, 408 or 429, or a truncated body raises `EndpointUnreachable`, a serve fault under the infra rule; any other refusal is a config error. | Every discovery failure, an unanswered endpoint included, was one config error. |
| A plain reply on a re-task turn after a report answers with the mission's last flight; the zero-flight nudge and its abstain count the flights of every turn. | Each re-task turn counted its flights from zero, so a reply that did not fly again drew "you have flown nowhere" twice and then abstained on a mission that had flown. |
| A bare `filter()` reads one refusal on a set, a row and a list of rows. | A list of rows served its own refusal text for a bare `filter()`. |
| A fused ring frame is posed as the corrector poses every frame, by the vehicle's pose at its stamp plus the boresight's timing offset (its tag only where the vehicle has no pose then), so its registration and the pose graph share one time base. | Ring frames were posed by their tag at the stamp without the timing offset the corrector applied; under a nonzero offset a correct ring fit could be refused. The flown boresight's offset was 0. |
| Bake bookkeeping tags (`source`, `osm_id`, `worldkit_layer`, `worldkit_height_m`, `source_tag`) never reach a prior row: not its `tags`, its search text, the census, the Best/Head labels, the grid or find's tag scoring. | Flown rows carried them: `row.tags` and `print(row)` served them, a find could match on them (`find("landuse")` matched every Cal City row stamped `worldkit_layer=landuse`), and the census and match labels skipped only `osm_id`. |
| A bench attempt that loses its reasoner mid-episode is infra `reasoner_down_midrun`. | The flown driver named it `vllm_down_midrun`. |
| On a ring run a crash or an unrecorded timeout scores `spl_pos` 0, so `run_report`'s mean SPL covers every scored episode. | The mean covered recorded rows only. |
| The closed-loop measures name the path the closest approach came off (`track_source`: GNSS track or keyframe poses) and label the median-height fallback `keyframes (median)`; the mission stamps the reasoning effort stripped, as its requests carry it. | `track_source` read "GNSS track" whenever the window held a fix, `keyframes` named both heights, and the raw environment effort was stamped. |
| The map caption's "N buildings carry fid chips, M more in frame are unchipped" counts prior building chips only; a stacked prior chip that cannot be placed counts as unchipped. | Dropped detection chips were added to the unchipped building count ("0 buildings carry fid chips, 531 more ... unchipped" in a window with none), and a stacked prior chip that was not drawn still counted as chipped. |
| A detection's class label gives its colour only through a whole palette word. | Labels matched palette words as substrings: "water tank" read tan, "covered parking" red. |
| `find(name="X")` with no query is refused ("the name IS the query"); refusals carry no `rule N:`/`rule F:` labels and name the real verbs (`look_for` + `fly`, or `fly` under the look_for withhold), and the repeated-program note names what changes the world (`fly`, a placed `look_for(..., at=)`); an unresolved boost serves `error` and `unknown_where`. | A bare `find(name=)` ignored the name and served a count of every row; refusals were labelled with rules no served text defined and named a `perceive` verb that does not exist (the repeated-program note said `no flight, no perceive`); the boost refusal also served `"pending": []`. |
| A mission's own reasoning effort (the twin's or a replay's `--reasoning-effort`, a forced replay's recorded one) is checked against the served model's card before any request. | Only the environment's effort was checked; a mission's own was sent as given. |
| The bench resolves `sensor: {}` to the default sensor's override and serves its block, as the flight mission does. | The bench read `sensor: {}` as no sensor. |
| A crop caption names a point subject (a flight view of an intersection point, or any go_to under the prior withhold) by its point label. | The caption named it `[?]`. |
| The map caption describes the prior chips the renderer draws (bare-digit `NNNN`; on a map around a fid or with marks, only the centre and marks), says a white outline or ring marks the centre, and lists the legend without digits. | The caption promised `pNNNN` chips on every big-enough building, numbered badges and a `1 = ...` legend the map never drew; the pixels are the same. |
| The bake places an OSM vertex in the cell it falls in (floored); a vertex up to one cell west or south of the block lies in no cell, so the CityNav bake holds 69 fewer edge priors over 23 of 34 blocks, with `prior_<i>` renumbered there. | The bake truncated toward zero, baking such a vertex as a phantom prior in column or row 0; the Cal City bake is unaffected. |
| One mission state holds across operator turns: bound names, `rN` handles, flown stops, staged viewpoints, explored fids and the watchlist persist; each turn has its own flight budget, loop-breaker and repeat streak. | Each operator turn after a report started from a fresh mission state (no flown conversation ran after a park). |
| A move inside a comprehension or generator expression is refused as inside a loop; a list where one place goes is refused naming the slot; a stance that cannot bind serves its reason as a sentence; both filter refusals list the same cuts, `direction=` included. | A comprehension flew every leg; a two-item list served "could not convert string to float"; the stance refusal served a Python repr; the two filter lists disagreed and left out `direction=`. |
| Under the prior withhold no served text names a survey, the prior map or OSM tags; under the look withhold the coordinate refusal says points are for relating. | Those arms served the default texts (withholds never flew). |
| An image-store upload carries no reasoner key, a cancel aborts it within a quarter second, and a store fault raises its own error that the mission retries as a turn. | The upload ran to completion or its 180 s timeout before a cancel was seen, and a store fault surfaced as an LLM status or transport error. |
| A move is refused inside a `def` or `lambda` and when its name is read as a value (`list(map(go_to, r1))`, `g = go_to`); a function defined in an earlier block runs under the line and time ceilings; `filter(dwithin=)` without `near=` is refused; `r[a:b][c:d]` holds what the same slice of a list holds; a repeated program's handles re-derive after the world moves; the stray "Eight words here too." is gone. | A move wrapped in a def or lambda, or passed as a value, flew every row; an earlier block's function ran unbounded; `dwithin=` was dropped beside a compass or facing cut; a window of a window could grow past its parent; a repeated program's handles kept their old rows; the sentence was served. |
| Under the images withhold a pass's catch keeps its delivered, shown and not-drawn facts beside the withheld note. | The whole catch was replaced by the note (withholds never flew). |
| A detection record carries the label, score, mask and the SAM object id and score; it has no coarse class bucket. | Flown records also carried a class bucket and unread pixel and model metadata. |

## Results

### Closed loop

The closed-loop measures are read from the flown sorties' own records by
[`benchmarks/flight/measures.py`](benchmarks/flight/measures.py), over the
analysts' adjudications as of commit `54135a3e7` of the development
repository: of 87 runs, 18 leave every rate with their evidence as the reason
(13 lost to the infrastructure, 4 stopped by the crew, 1 whose target has no
mask), and 69 are scored.

| Measure | Value | Definition |
|---|---|---|
| SR | 40/69 | The commit point lies inside the target's hand-drawn mask (one run with none: its tasking footprint). |
| OSR | 59/69 | The aircraft's path (the GNSS track; beside a track holding under 95% of the run, also the centres of the observations the run fused) came within one camera footprint width of the target centroid. |

| Intel quality (the run's first tasking) | Scored runs | SR | OSR |
|---|---|---|---|
| Good | 27 | 22/27 | 26/27 |
| Medium | 32 | 13/32 | 25/32 |
| Poor | 10 | 5/10 | 8/10 |

```sh
python -m benchmarks.flight.measures --root <sorties>/* --truth <adjudications dir> <flight-week tasking JSON>
```

The tasking JSON is the flight week's own tasking set (it scores a run with no
hand-drawn mask, matched by the run's description); the released Cal City split
is a different set of episodes.

A run the analysts' adjudication marks excluded is out of every rate, and so
is a run the serve failed under (the bench's infra rule, `rows.is_infra_abort`),
which the output names. The flight recordings and the adjudications are not
distributed.

### Bench experiments

Each experiment is a set of run roots; the readers in
[`benchmarks/citynav/analysis`](benchmarks/citynav/analysis/README.md) derive
every number from them. [Reproduce the experiments](#reproduce-the-experiments)
lists the exact commands.

| Experiment | Split and episodes | Arms | Config and flags | Reader |
|---|---|---|---|---|
| E1 | CityNav `train_seen`, 1,000 per arm (`--limit 1000`, seed 1234) | Inkling Small at effort 0.99; Qwen 3.8 Flash Next at `xhigh`; Gemma 4 26B-A4B | `configs/citynav.yaml`, `SEARCH_SAMPLING_REASONING_EFFORT` | `arm_table`, `run_report` |
| E2 | CityNav `test_unseen`, all 5,311 | Qwen 3.8 Flash Next at `low`, `medium`, `xhigh`; Inkling Small at 0.99; Gemma 4 E2B; Gemma 4 E4B | `configs/citynav.yaml`, `--split test_unseen` | `arm_table`, `run_report` |
| E3 | Cal City, the 603 reviewed episodes | Inkling Small at 0.0, 0.1 ... 0.9 and 0.99 (11 rungs); Gemma 4 26B-A4B | `configs/calcity.yaml`, `--episode-allowlist` from `benchmarks.calcity.data_prep.reviewed` | `arm_table`, `figures.plot_effort_ladder` |
| E4 | CityNav `train_seen`, 1,000 per arm | Per model: the control and one arm per withhold | `configs/withhold/<part>.yaml` | `arm_table`, `ab_report` |
| E5 | E2 and E3 roots | Every arm | none beyond E2 and E3 | `figures.capability_effort_arms`, `figures.plot_capability_effort` |

## Install

MAPS needs Python 3.12 or later. With [uv](https://docs.astral.sh/uv/):

```sh
uv venv .venv --python 3.12
uv pip install --python .venv -e '.[dev]'
```

Or with pip:

```sh
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

The commands below run in the activated environment (`source .venv/bin/activate`).

Extras:

| Extra | For |
|---|---|
| `recording` | Reading a camera recording for the as-flown frame replay (`mcap`). |
| `dev` | The test suite and linter. |
| `baselines` | The GeoNav and ViSA baseline arms and the ViSA figures (`matplotlib`). |

The SAM 3.1 serve runs in its own environment, because Meta's `sam3` package
requires `numpy<2`:

```sh
micromamba env create -f envs/sam3_1.yml
micromamba activate sam3_1
pip install <a checkout of Meta's SAM 3 source>
```

The checkpoint `facebook/sam3.1` is gated on Hugging Face: accept its terms,
export `HF_TOKEN`, and the serve fetches it on first load, or point
`SAM3_CHECKPOINT` at a local file.

Run the tests:

```sh
.venv/bin/pytest -q
```

## Endpoints

MAPS calls two services: an OpenAI-compatible reasoner and the SAM 3.1 serve.

Start the SAM serve from the repository root, in the `sam3_1` environment:

```sh
python -m serving.sam_server --port 8772
```

The reasoner is any OpenAI-compatible chat-completions endpoint whose
`/v1/models` lists the served model and its `max_model_len` (vLLM reports
both). The served model name selects a sampling envelope from
[`maps/llm/model_cards.py`](maps/llm/model_cards.py) (tokenizer, `top_p`,
`top_k`, the reasoning efforts it takes), and must be one of that table's
keys exactly: `inkling-small`, `qwen38-flash-next-nvfp4`,
`gemma4-26b-a4b-nvfp4` (E1 to E3), `gemma4-e4b`, `gemma4-e2b` (E2),
`gemma4-26b-a4b-gguf` (the llama.cpp serve flown aboard), and
`openai.gpt-5.6-sol` ([GPT-5.6 Sol on Amazon Bedrock](#gpt-56-sol-on-amazon-bedrock));
any other name is refused at startup with the list. Every request runs in
thinking mode at temperature 1.0, and `max_tokens` is the output reserve of
the discovered window (the OpenAI-wire card below takes both in OpenAI's own
terms).

| Variable | Read by | Meaning |
|---|---|---|
| `SEARCH_REASONER_URL` | client | The reasoner's base URL, one endpoint. Default `http://127.0.0.1:8001/v1`. |
| `SEARCH_REASONER_API_KEY` | client | Bearer token for the reasoner. |
| `SEARCH_REASONER_TIMEOUT_S` | client | Per-call transport timeout, seconds. |
| `SEARCH_ENDPOINT_SPEC_OVERRIDE` | client | `<model_id>:<max_model_len>` for an endpoint whose `/v1/models` reports no window. |
| `SEARCH_REASONER_MODELS_URL` | client | Where the served models are listed, for an endpoint that lists them somewhere other than `SEARCH_REASONER_URL/models`. |
| `SEARCH_SAMPLING_REASONING_EFFORT` | client | The arm's reasoning effort, sent verbatim as the request's `reasoning_effort` string (`0.99`, `xhigh`); discovery refuses a value the served model's card does not take. Unset sends none. |
| `SEARCH_COMPACT_TOKENIZER_NAME` | client | The HF tokenizer repo, replacing the model card's. |
| `SAM_SERVER_URL` | client | The SAM serve's URL, or a comma list of replicas. Default `http://127.0.0.1:8772` when unset; a set but blank value, or `SAM_VIDEO_SERVER_URL` set without it, is refused. |
| `SAM_PORT`, `SAM_MODEL_ID`, `SAM_DEVICE`, `SAM_DTYPE` | serve | Port (8772), checkpoint id (`facebook/sam3.1`), device, dtype (`bfloat16`). |
| `SAM3_CHECKPOINT` | serve | A local checkpoint file. |
| `SAM_MAX_INFLIGHT`, `SAM_INFLIGHT_WAIT_S`, `SAM_PREWARM`, `SAM_HEALTH_MIN_FREE_GB` | serve | Admission, prewarm and health limits (`serving/sam_server.py`). |
| `MAPS_IMAGE_STORE_URL` | client | An image store one fast hop from the reasoner: each inline image is PUT once to `<store>/looks/<sha>` and rides as `sha256:<sha>`; a 409 naming missing shas re-sends them. |

### GPT-5.6 Sol on Amazon Bedrock

Amazon Bedrock's `bedrock-mantle` endpoint serves GPT-5.6 Sol as
`openai.gpt-5.6-sol` under OpenAI's own chat-completions contract, not a chat
template. Its card (`wire: "openai"` in `maps/llm/model_cards.py`) shapes the
request to that contract (`maps/llm/openai_wire.py`): no sampling envelope or
template kwargs, the output cap as `max_completion_tokens`, and each message
with only the fields its role takes. OpenAI refuses a function description
over 1,024 characters, so the `run` tool's documentation rides the system
turn verbatim under its own heading and the tool points at it. Bedrock
refuses a request carrying more than 100 images, so past that the reasoner
evicts the oldest crops, down to 80. The
thinking span never comes back, so the reasoning count is the provider's
`usage.completion_tokens_details` meter. A Bedrock API key is the bearer
token.

```sh
export SEARCH_REASONER_URL=https://bedrock-mantle.us-east-1.api.aws/openai/v1
export SEARCH_REASONER_MODELS_URL=https://bedrock-mantle.us-east-1.api.aws/v1/models
export SEARCH_REASONER_API_KEY=<Bedrock API key>
export SEARCH_ENDPOINT_SPEC_OVERRIDE=openai.gpt-5.6-sol:262144
export SEARCH_REASONER_TIMEOUT_S=600
python -m maps.llm.endpoint_preflight --effort none --effort medium --effort high
```

The override pins the window Inkling Small is served with, so the compactor
holds GPT-5.6 Sol to the same context budget as the other arms; the model's
own window is 1,050,000 tokens. The card takes the efforts `none`, `low`,
`medium`, `high`, `xhigh` and `max`. The preflight flies two turns per effort
through the reasoner's request path, the second streaming and carrying the
tool call and its result, and prints either the endpoint's refusal or each
reply and the token meter; it starts no run. Fly an arm as any other,
for example the reviewed Cal City episodes at one effort:

```sh
SEARCH_SAMPLING_REASONING_EFFORT=medium \
  python -m benchmarks.calcity.bench_fullset --episode-allowlist output/reviewed_episodes.txt \
  --out-root output/gpt56-sol-medium
```

## Data

### CityNav

CityNav's episodes, the CityRefer catalog and the SensatUrban point clouds
come from their upstream sources; this repository redistributes none of them.

```sh
export CITYNAV_ROOT=~/datasets/citynav-arxiv
benchmarks/citynav/data_prep/fetch_data.sh             # episodes and catalog; prints the SensatUrban steps
benchmarks/citynav/data_prep/rasterize_citynav_native.sh   # nadir rasters from the block point clouds
PY=.venv/bin/python benchmarks/citynav/data_prep/bake_tiles.sh   # the OSM prior, from the shipped caches
```

The rasterizer runs upstream's PDAL pipeline in an environment with `pdal`,
`rasterio` and `pillow` (`RASTERIZE_PYTHON` names its interpreter). The bake
writes `tiles/citynav_2m_osmv3` (the blocks are the rasters it finds); every run
manifest records its `bake_sha256`, which for these steps is
`12d699d686c1d68270f0b98e0e2c7e54397b1528fa61e304ce3e16b2f960b76e`. See
[`benchmarks/citynav`](benchmarks/citynav/README.md) for the variables that
relocate each part.

### Cal City

The Cal City corpus (`benchmarks/calcity/data/calcity.json`, 1,023 episodes)
and the positions of its reviewed targets
(`benchmarks/calcity/data_prep/authoring/ground_truth.json`) ship with the
repository. You supply two inputs:

* The C-12 v6 map raster of the block, `<map>_rgb.tif`.
* A world bundle of OpenStreetMap layers over the block (`osm/<layer>.geojson`
  in WGS84 for each of the eight `bundle.OSM_LAYERS`), named by `CALCITY_BUNDLE`.

Then build the catalog, the raster pair and the tile bake:

```sh
export CALCITY_BUNDLE=<bundle dir>
python -m benchmarks.calcity.data_prep.build_objects_json
python -m benchmarks.calcity.data_prep.build_rgbd_ortho --raster <map>_rgb.tif
python -m benchmarks.calcity.data_prep.bake_tiles
```

The raster pair lands in `<root>/data/rgbd_upstream`, `<root>` being
`SEARCH_CALCITY_DATA` (default `~/datasets/calcity`), where the bench reads it.

The flight harness also reads the block's bare-earth surface, as the flown
bake carried it: `calcity_dem.tif` beside `calcity_meta.json` under
`SEARCH_CALCITY_TILES`, heights in metres NAVD88 on 10 m cells in EPSG:32611
over the bake's bounds. It is public USGS 3DEP data (the 1/3 arc-second DEM,
from The National Map), resampled onto the bake's frame, for example:

```sh
gdalwarp -t_srs EPSG:32611 -tr 10 10 -r bilinear -te <origin_e> <origin_n> <origin_e + x_max> <origin_n + y_max> \
  USGS_13_*.tif "$SEARCH_CALCITY_TILES/calcity_dem.tif"
```

`origin_e` and `origin_n` are the bake's `georef` in `calcity_meta.json`,
`x_max` and `y_max` its `bounds`. Without the file the flight places every
frame over flat ground at the block's floor (`tile_georef.FLOOR_MSL_M`).

See [`benchmarks/calcity`](benchmarks/calcity/README.md).

## Reproduce the experiments

Each command writes one run root. Name a model by serving it at
`SEARCH_REASONER_URL`; the run manifest records the served model the
endpoint reports. A rerun of the same command resumes the root.

E1, one model at a time:

```sh
SEARCH_SAMPLING_REASONING_EFFORT=0.99 \
  python -m benchmarks.citynav.bench_fullset --split train_seen --limit 1000 --sample-seed 1234 \
  --out-root output/e1/inkling-small
SEARCH_SAMPLING_REASONING_EFFORT=xhigh \
  python -m benchmarks.citynav.bench_fullset --split train_seen --limit 1000 --sample-seed 1234 \
  --out-root output/e1/qwen38-flash-next
python -m benchmarks.citynav.bench_fullset --split train_seen --limit 1000 --sample-seed 1234 \
  --out-root output/e1/gemma4-26b-a4b
```

E2, each arm over the whole split:

```sh
SEARCH_SAMPLING_REASONING_EFFORT=xhigh \
  python -m benchmarks.citynav.bench_fullset --split test_unseen --out-root output/e2/qwen38-flash-next-xhigh
```

The other arms change the effort (`low`, `medium`, or `0.99` with Inkling
Small served), or serve Gemma 4 E2B or E4B with no effort set.

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

E4, the control and one root per withheld part, per model:

```sh
for part in prior look_for look images mission_map system_turn; do
  SEARCH_SAMPLING_REASONING_EFFORT=<model's E1 effort> CITYNAV_BENCH_CONFIG=configs/withhold/$part.yaml \
    python -m benchmarks.citynav.bench_fullset --split train_seen --limit 1000 --sample-seed 1234 \
    --out-root output/e4/<model>/$part
done
```

The control is the model's E1 root, so each arm runs at the model's E1 effort
(blank for a model run with none); `ab_report` refuses a pair whose efforts differ.

Read the results:

```sh
python -m benchmarks.citynav.analysis.arm_table output/e2/*            # SR, OSR, NE per arm
python -m benchmarks.citynav.analysis.run_report output/e1/inkling-small   # one root in full
python -m benchmarks.citynav.analysis.ab_report output/e1/<model> output/e4/<model>/look
python -m figures.plot_effort_ladder output/e3/inkling-small-e*
python -m figures.plot_capability_effort --split test_unseen output/e2/*
python -m figures.plot_capability_effort --split calcity --cost total output/e3/*
```

Figures land in `output/figures/` as light and dark SVGs. Every claimed
number and figure maps to its reader in
[`benchmarks/citynav/analysis/README.md`](benchmarks/citynav/analysis/README.md).

## Rerun the baselines

[`baselines/`](baselines/README.md) holds the published methods the paper
compares against. GeoNav, GeoNav as released, and the ViSA-inspired policy
fly the bench in place of the mission conversation: a config's `policy:`
hands each episode to the method, and the sensor, prior, budget, replay,
scorer and row are the bench's. Each config in `configs/baselines/` is a
control config plus `policy:`, so a baseline root reads as an arm of the
MAPS root it shares a selection with. The methods draw with matplotlib, which
the `baselines` extra installs (`pip install -e '.[baselines]'`; `dev` carries
it too):

```sh
for policy in geonav geonav_released visa; do
  SEARCH_SAMPLING_REASONING_EFFORT=<model's E1 effort> CITYNAV_BENCH_CONFIG=configs/baselines/$policy.yaml \
    python -m benchmarks.citynav.bench_fullset --split train_seen --limit 1000 --sample-seed 1234 \
    --out-root output/baselines/<model>/$policy
  SEARCH_SAMPLING_REASONING_EFFORT=<the E3 root's effort> CITYNAV_BENCH_CONFIG=configs/baselines/calcity_$policy.yaml \
    python -m benchmarks.calcity.bench_fullset --episode-allowlist output/reviewed_episodes.txt \
    --out-root output/baselines/<model>/calcity_$policy
done
python -m benchmarks.citynav.analysis.run_report output/baselines/<model>/visa
python -m benchmarks.citynav.analysis.ab_report output/e1/<model> output/baselines/<model>/visa
```

The released checkpoints (CityNav MGP, HETT, FlightGPT) run under their
authors' own evaluators, from pinned clones, through
`baselines/released/released.sh`; their rows read with `run_report` and pair
against a MAPS root with `python -m baselines.released.outcome_rows pair`.
[`baselines/released/README.md`](baselines/released/README.md) has the
stages and every deviation from upstream.

## Fly the method

[`benchmarks/flight`](benchmarks/flight/README.md) holds the flight harness:

* `Mission` (`benchmarks/flight/mission.py`) hydrates the engine as the bench
  does and holds the conversation open across operator taskings. A flight is
  a leg the aircraft flies, arrived when the recording holds the area it was
  flown for.
* `Vehicle` (`benchmarks/flight/vehicle.py`) is the boundary to the aircraft:
  `post(leg)`, `pose(t)`, `index(since)`, `frame(seq)`, `now()`, `sleep(s)`.
* `LiveSource` (`benchmarks/flight/live_source.py`) turns the recorded frames
  into the observation the model reads: coverage, per-frame registration, the
  pose-graph corrector, compositing and the mosaic fit.

The twin is a kinematic `Vehicle` over a raster pair, so you can fly a Cal
City mission end to end without an aircraft:

```sh
python -m benchmarks.flight.twin --config configs/calcity.yaml --episode 'calcity::6#6' --ortho <root>/data/rgbd_upstream
```

The twin, the live source and the frame registration stand each frame on
the bake's `calcity_dem.tif` where it is present (see [Cal City](#cal-city)).

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
directory as `--raster` to replay as flown.

## Repository layout

| Path | Contents |
|---|---|
| [`maps/`](maps/README.md) | The method: mission loop, Python surface, world state, perception client, LLM transport. |
| [`benchmarks/citynav/`](benchmarks/citynav/README.md) | The CityNav bench: episode runner, scoring, prior bake, data preparation. |
| [`benchmarks/citynav/analysis/`](benchmarks/citynav/analysis/README.md) | The run readers every number is derived with. |
| [`benchmarks/calcity/`](benchmarks/calcity/README.md) | The Cal City bench: corpus, reviewed set, raster and bake build. |
| [`benchmarks/flight/`](benchmarks/flight/README.md) | The flight harness, the twin, as-flown replay and the closed-loop measures. |
| [`benchmarks/capacity/`](benchmarks/capacity/README.md) | The human-capacity study: seeded report streams and the finds-per-hour reader. |
| [`baselines/`](baselines/README.md) | The published methods re-measured on the bench (GeoNav, ViSA) and the released checkpoints' runner. |
| `figures/` | The effort-ladder and capability-effort figures. |
| `serving/` | The SAM 3.1 serve. |
| `configs/` | Bench configs: the CityNav control, Cal City, the withhold arms and the baseline arms. |
| `envs/` | The SAM 3.1 serve environment. |

## Citation

```bibtex
@software{holt2026maps,
  title   = {MAPS},
  author  = {Holt, Finley R.},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/FinleyHolt/MAPS}
}
```

MAPS evaluates on CityNav (Lee et al.,
[arXiv:2406.14240](https://arxiv.org/abs/2406.14240)), built on SensatUrban and
CityRefer; cite them as well. [`CITATION.cff`](CITATION.cff) carries the same
record.

## License

MIT; see [`LICENSE`](LICENSE). Third-party code and data are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
