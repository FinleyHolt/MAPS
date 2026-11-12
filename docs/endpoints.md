# Install and endpoints

MAPS calls two services: an OpenAI-compatible reasoner and the SAM 3.1 serve.
This page covers the install, both services and every environment variable
the engine reads.

## Install

MAPS needs Python 3.12 or later. With [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/FinleyHolt/MAPS && cd MAPS
uv venv .venv --python 3.12
uv pip install --python .venv -e '.[dev]'
```

Or with pip:

```sh
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

Commands in these documents run from the repository root in the activated
environment (`source .venv/bin/activate`).

| Extra | For |
|---|---|
| `recording` | Reading a camera recording for the as-flown frame replay (`mcap`). |
| `station` | The station's server (`fastapi`, `uvicorn`). |
| `dev` | The test suite and linter, plus both extras above. |

Every run records the commit it ran from. A tree without git, such as a
`git archive` export, reads that commit from a `.commit` file at its root:

```sh
git archive HEAD | tar -x -C <dest> && git rev-parse HEAD > <dest>/.commit
```

Run the tests:

```sh
.venv/bin/pytest -q
```

## The SAM 3.1 serve

The serve runs in its own environment, because Meta's `sam3` package
requires `numpy<2`:

```sh
micromamba env create -f envs/sam3_1.yml
micromamba activate sam3_1
pip install <a checkout of Meta's SAM 3 source>
```

The checkpoint `facebook/sam3.1` is gated on Hugging Face: accept its terms,
export `HF_TOKEN`, and the serve fetches it on first load, or point
`SAM3_CHECKPOINT` at a local file.

Start the serve from the repository root, in the `sam3_1` environment:

```sh
python -m serving.sam_server --port 8772
```

It runs on one GPU (`SAM_DEVICE`). The client finds it at `SAM_SERVER_URL` (default
`http://127.0.0.1:8772`), which also takes a comma list of replicas.

## The reasoner

The reasoner is any OpenAI-compatible chat-completions endpoint whose
`/v1/models` lists the served model and its `max_model_len` (vLLM reports
both). The served model name selects a sampling envelope from
[`maps/llm/model_cards.py`](../maps/llm/model_cards.py) (tokenizer, `top_p`,
`top_k`, the reasoning efforts it takes), and must be one of that table's
keys exactly:

| Served model name | Model |
|---|---|
| `inkling-small`, `qwen38-flash-next-nvfp4`, `gemma4-26b-a4b-nvfp4` | Inkling Small, Qwen 3.8 Flash Next and Gemma 4 26B-A4B, the models of E2 to E4. |
| `gemma4-e4b`, `gemma4-e2b` | Gemma 4 E4B and E2B (E2). |
| `gemma4-26b-a4b-gguf` | Gemma 4 26B-A4B on the llama.cpp serve flown aboard. |
| `kimi-k3.1-pro` | Kimi K3.1 Pro. |
| `deepseek-v4.1-flash-1017` | DeepSeek V4.1 Flash 1017. |
| `gpt-5.6-sol` | GPT-5.6 Sol on an Azure deployment ([below](#gpt-56-sol-on-azure)). |

Any other name is refused at startup with the list. To fly another model,
add its card to that table. Every request runs in thinking mode at
temperature 1.0, and `max_tokens` is the output reserve of the discovered
window (the OpenAI-wire card below takes both in OpenAI's own terms).

Check an endpoint before a run:

```sh
python -m maps.mission.endpoint_preflight --effort <effort>
```

The preflight flies two turns per effort through the reasoner's request
path, the second streaming and carrying the tool call and its result, and
prints either the endpoint's refusal or each reply and the token meter; it
starts no run.

## Environment variables

Each entry point reads the client variables once, when it starts
(`maps/settings.py`), and a run carries what it read: a value changed in a
shell reaches the next process and leaves a running one as it is.

| Variable | Read by | Meaning |
|---|---|---|
| `SEARCH_REASONER_URL` | client | The reasoner's base URL, one endpoint. Default `http://127.0.0.1:8001/v1`. |
| `SEARCH_REASONER_API_KEY` | client | Bearer token for the reasoner. |
| `SEARCH_REASONER_TIMEOUT_S` | client | Per-call transport timeout, seconds (default 2400: one 40,000-token turn under a saturated batch). |
| `SEARCH_ENDPOINT_SPEC_OVERRIDE` | client | `<model_id>:<max_model_len>` for an endpoint whose `/v1/models` reports no window. |
| `SEARCH_ENDPOINT_OFFLINE` | client | `1`: nothing is probed and the override is the endpoint's spec. |
| `SEARCH_REASONER_MODELS_URL` | client | Where the served models are listed, for an endpoint that lists them somewhere other than `SEARCH_REASONER_URL/models`. |
| `SEARCH_SAMPLING_REASONING_EFFORT` | client | The arm's reasoning effort, sent verbatim as the request's `reasoning_effort` string (`0.99`, `xhigh`); discovery refuses a value the served model's card does not take. Unset sends none. |
| `SEARCH_COMPACT_TOKENIZER_NAME` | client | The HF tokenizer repo, or a local checkpoint directory on an offline machine, replacing the model card's. |
| `SEARCH_WITHHOLD` | client | A comma list of withheld parts (`maps/withhold.py`); a bench config's `withhold` replaces it. |
| `SAM_SERVER_URL` | client | The SAM serve's URL, or a comma list of replicas. Default `http://127.0.0.1:8772` when unset; a set but blank value, or `SAM_VIDEO_SERVER_URL` set without it, is refused. |
| `SAM_PORT`, `SAM_DEVICE` | serve | Port (8772), device. |
| `SAM3_CHECKPOINT` | serve | A local checkpoint file. |
| `SAM_MAX_INFLIGHT`, `SAM_INFLIGHT_WAIT_S`, `SAM_PREWARM`, `SAM_HEALTH_MIN_FREE_GB` | serve | Admission, prewarm and health limits (`serving/sam_server.py`). |
| `MAPS_IMAGE_STORE_URL` | client | An image store one fast hop from the reasoner: each inline image is PUT once to `<store>/looks/<sha>` and rides as `sha256:<sha>`; a 409 naming missing shas re-sends them. |

The variables that locate each bench's data are in [`data.md`](data.md).

## GPT-5.6 Sol on Azure

An Azure (Microsoft Foundry) deployment serves GPT-5.6 Sol on Azure's v1 API
under OpenAI's own chat-completions contract, with no chat template. The
request names the deployment, so the deployment carries the card's alias,
`gpt-5.6-sol`. The card (`wire: "openai"` in `maps/llm/model_cards.py`)
shapes the request to that contract (`maps/llm/openai_wire.py`): no sampling
envelope or template kwargs, the output cap as `max_completion_tokens`, and
each message with only the fields its role takes. OpenAI refuses a function
description over 1,024 characters, so the `run` tool's documentation rides
the system turn verbatim under its own heading and the tool points at it.
Azure refuses a GPT-5.x request carrying more than 50 images, so past that
the reasoner evicts the oldest crops, down to 30. Its chat completions refuse
function tools beside any reasoning effort but `none`, so the card
(`api: "responses"`) flies the Responses API: the engine builds and records
the chat-shaped request, and `openai_wire.to_responses` /
`from_responses` translate it at the wire, statelessly (`store: false`). The
thinking span never comes back in the clear; each assistant turn carries the
encrypted reasoning items its response returned, and the next request sends
them back before that turn's calls, as a self-hosted chat template renders an
earlier turn's `reasoning_content`. The compactor prices those items by the
reasoning tokens that wrote them. The reasoning count is the provider's
meter. The deployment's key is the bearer token.

Azure's `/openai/v1/models` lists the resource's base models and leaves out
its deployments, so for this card discovery takes the listing answering the
key as the live check, and the first request is refused for a deployment name
the resource does not carry. A Global Standard deployment's tokens-per-minute
limit admits each request at its prompt plus `max_completion_tokens` (the
output reserve, 40,000 tokens at the pinned window), whatever it bills; size
the limit to the bench's concurrency on that count.

```sh
export SEARCH_REASONER_URL=https://<resource>.openai.azure.com/openai/v1
export SEARCH_REASONER_API_KEY=<deployment key>
export SEARCH_ENDPOINT_SPEC_OVERRIDE=gpt-5.6-sol:262144
python -m maps.mission.endpoint_preflight --effort none --effort medium --effort xhigh
```

The override pins the window Inkling Small is served with, so the compactor
holds GPT-5.6 Sol to the same context budget as the other arms; the model's
own window is 1,050,000 tokens. The card takes the efforts `none`, `low`,
`medium`, `high`, `xhigh` and `max`. Each request carries the episode's
routing key as `prompt_cache_key`, and every turn's bytes open with the last
turn's until an image eviction or a compaction fire rewrites an earlier
message, so Azure's prompt cache serves the prefix; the preflight fails an
effort whose second turn reads no cache hit. A run's spend, flying attempts
and re-queued ones included, reads off its transcripts with
`python -m benchmarks.citynav.analysis.spend <out-root>`. Fly an arm as any
other, for example the reviewed Cal City episodes at one effort:

```sh
SEARCH_SAMPLING_REASONING_EFFORT=medium \
  python -m benchmarks.calcity.bench_fullset --episode-allowlist output/reviewed_episodes.txt \
  --out-root output/gpt56-sol-medium
```
