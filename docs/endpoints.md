# Install and endpoints

MAPS calls two services: an OpenAI-compatible reasoner and the SAM 3.1 serve.
This page covers the install, both services, and the environment variables
that configure them.

## Install

MAPS requires Python 3.12 or later. Run the commands in these documents from
the repository root, in the activated environment
(`source .venv/bin/activate`).

1. Clone the repository:

   ```sh
   git clone https://github.com/FinleyHolt/MAPS && cd MAPS
   ```

2. Install with [uv](https://docs.astral.sh/uv/):

   ```sh
   uv venv .venv --python 3.12
   uv pip install --python .venv -e '.[dev]'
   ```

   Or with pip:

   ```sh
   python -m venv .venv && .venv/bin/pip install -e '.[dev]'
   ```

3. Run the tests:

   ```sh
   .venv/bin/pytest -q
   ```

| Extra | Contents |
|---|---|
| `station` | Station's server (`fastapi`, `uvicorn`). |
| `dev` | Test suite and linter, plus the extra above. |

Every run records the commit it ran from. A tree without git, such as a
`git archive` export, reads that commit from a `.commit` file at its root:

```sh
git archive HEAD | tar -x -C <dest> && git rev-parse HEAD > <dest>/.commit
```

## SAM 3.1 serve

The serve runs in its own environment, because Meta's `sam3` package requires
`numpy<2`.

1. Create the environment:

   ```sh
   micromamba env create -f envs/sam3_1.yml
   micromamba activate sam3_1
   pip install git+https://github.com/facebookresearch/sam3@2814fa619404a722d03e9a012e083e4f293a4e53
   ```

2. Get the checkpoint. `facebook/sam3.1` is gated on Hugging Face: accept its
   terms and export `HF_TOKEN`, and the serve fetches it on first load. Or
   point `SAM3_CHECKPOINT` at a local file.

3. Start the serve from the repository root, in the `sam3_1` environment:

   ```sh
   python -m serving.sam_server --port 8772
   ```

The serve runs on one GPU (`SAM_DEVICE`). The client finds it at
`SAM_SERVER_URL` (default `http://127.0.0.1:8772`), which also takes a comma
list of replicas. The serve listens on `0.0.0.0`, every interface, by default
and has no authentication; `--host` sets the address it binds.

## Reasoner

The reasoner is any OpenAI-compatible chat-completions endpoint whose
`/v1/models` lists the served model and its `max_model_len` (vLLM reports
both). The served model name selects a sampling envelope from
[`maps/llm/model_cards.py`](../maps/llm/model_cards.py) (tokenizer, `top_p`,
`top_k`, the reasoning efforts it takes), and must be one of that table's
keys exactly:

| Served model name | Model |
|---|---|
| `inkling-small`, `qwen38-flash-next-nvfp4`, `gemma4-26b-a4b-nvfp4` | Inkling Small, Qwen 3.8 Flash Next, and Gemma 4 26B-A4B, the models of E2 and E3. |
| `gemma4-e4b`, `gemma4-e2b` | Gemma 4 E4B and E2B (E2). |
| `gemma4-26b-a4b-gguf` | Gemma 4 26B-A4B on a llama.cpp serve. |
| `kimi-k3.1-pro` | Kimi K3.1 Pro. |
| `deepseek-v4.1-flash-1017` | DeepSeek V4.1 Flash 1017. |
| `gpt-5.6-sol` | GPT-5.6 Sol on an Azure deployment. See [GPT-5.6 Sol on Azure](#gpt-56-sol-on-azure). |

Startup refuses any other name and prints the list. To fly another model, add
its card to that table. A chat-template card's request carries temperature
1.0 and `enable_thinking: true`, except `gemma4-26b-a4b-nvfp4` at effort
`none`, which sends `enable_thinking: false`. `max_tokens` is the output
reserve of the discovered window. The OpenAI-wire card takes the effort and
the output cap in OpenAI's own terms.

The compactor counts tokens with the card's tokenizer, which it downloads from
the Hugging Face Hub at first use. A gated tokenizer repository needs
`HF_TOKEN` with access to it; an offline machine needs
`SEARCH_COMPACT_TOKENIZER_NAME` naming a local copy. For each card's
repository and its license, see
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md#used-and-not-redistributed).

To check an endpoint before a run:

```sh
python -m maps.mission.endpoint_preflight --effort <effort>
```

The preflight flies two turns per effort through the reasoner's request path,
the second streaming and carrying the tool call and its result. It prints
either the endpoint's refusal or each reply and the token meter. It starts no
run.

## Environment variables

Each entry point reads the client variables once, at startup
(`maps/settings.py`), and a run carries what it read. A value changed in a
shell reaches the next process and leaves a running one as it is. For the
variables that locate each bench's data, see
[`benchmarks/citynav`](../benchmarks/citynav/README.md#prepare-the-data) and
[`benchmarks/calcity`](../benchmarks/calcity/README.md#build-the-bench).

| Variable | Read by | Description |
|---|---|---|
| `SEARCH_REASONER_URL` | Client | Reasoner's base URL, one endpoint. Default `http://127.0.0.1:8001/v1`. |
| `SEARCH_REASONER_API_KEY` | Client | Bearer token for the reasoner. |
| `SEARCH_REASONER_TIMEOUT_S` | Client | Per-call transport timeout, seconds. Default 2400: one 40,000-token turn under a saturated batch. |
| `SEARCH_ENDPOINT_SPEC_OVERRIDE` | Client | `<model_id>:<max_model_len>` for an endpoint whose `/v1/models` reports no window. |
| `SEARCH_ENDPOINT_OFFLINE` | Client | `1`: nothing is probed and the override is the endpoint's spec. |
| `SEARCH_REASONER_MODELS_URL` | Client | Where the served models are listed, for an endpoint that lists them somewhere other than `SEARCH_REASONER_URL/models`. |
| `SEARCH_SAMPLING_REASONING_EFFORT` | Client | Run's reasoning effort, sent verbatim as the request's `reasoning_effort` string (`0.99`, `xhigh`). Discovery refuses a value the served model's card doesn't take. Unset sends none. |
| `SEARCH_COMPACT_TOKENIZER_NAME` | Client | HF tokenizer repo, or a local checkpoint directory on an offline machine, replacing the model card's. |
| `SAM_SERVER_URL` | Client | SAM serve's URL, or a comma list of replicas. Default `http://127.0.0.1:8772` when unset. A set but blank value is refused. |
| `SAM_PORT`, `SAM_DEVICE` | Serve | Port (8772), device. |
| `SAM3_CHECKPOINT` | Serve | Local checkpoint file. |
| `SAM_MAX_INFLIGHT`, `SAM_INFLIGHT_WAIT_S`, `SAM_PREWARM`, `SAM_HEALTH_MIN_FREE_GB` | Serve | Admission, prewarm, and health limits (`serving/sam_server.py`). |
| `PYTORCH_CUDA_ALLOC_CONF` | Serve | PyTorch's allocator settings. The serve sets `expandable_segments:True` when unset. |
| `MAPS_IMAGE_STORE_URL` | Client | Image store one fast hop from the reasoner: each inline image is PUT once to `<store>/looks/<sha>` and rides as `sha256:<sha>`; a 409 naming missing shas re-sends them. For the protocol, see the docstring of `maps/llm/http.py`. |
| `SEARCH_SEED_BASE` | Runner | Base of the per-turn request seed. Unset, each mission takes one from the wall clock. The run manifest records it when set; `run_benchmark --seed-base` overrides it. |
| `SEARCH_LLM_CACHE_KEY` | Client | Prompt-cache key an OpenAI-wire request carries in place of the episode's routing key. The bench driver sets `<out-root name>/w<worker id>` in each attempt child's environment. |
| `CITYNAV_BENCH_CONFIG` | Driver | Run config, relative to the repository root, that the bench driver passes each attempt child as `--config`. Default `configs/citynav.yaml`; the Cal City driver's default is `configs/calcity.yaml`. |

The bench driver sets these in each attempt child's environment:

| Variable | Description |
|---|---|
| `SEARCH_BENCHMARK_NAME` | Bench the child flies, from the bench's own environment: Cal City's carries `calcity`, and the child flies CityNav when it is unset. |
| `CITYNAV_EPISODE_DEADLINE_EPOCH` | Epoch at which the mission stops flying: the episode's wall cap less a 10% reserve for the model's reply. |
| `CITYNAV_BENCH_PARENT_PID` | Driver's process ID, or `1` under bubblewrap (the sandbox's init, killed with the driver); the child exits when that parent is gone. |
| `HF_HUB_OFFLINE` | `1` under bubblewrap: the child reads the hub cache read-only, holding the tokenizer the driver's discovery cached. |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENCV_FOR_THREADS_NUM`, `RAYON_NUM_THREADS`, `MALLOC_ARENA_MAX` | One thread per native pool and two malloc arenas, unless the shell set its own. |
| `XDG_RUNTIME_DIR`, `DBUS_SESSION_BUS_ADDRESS` | The launcher's own values, when the attempt runs in a scope: `systemd-run` finds the user manager through them. |

Each attempt child sees the launcher's environment cut to an allowlist
(`benchmarks/sandbox.py`, `child_env`): the `SEARCH_`, `CITYNAV_`, `CALCITY_`,
`SAM_`, `MAPS_`, `HF_`, `HUGGINGFACE_`, and `LC_` variables, Hugging Face's token excepted, and the
locale, path, proxy, and thread-count variables. The station, the twin CLI,
and the runner restart themselves under the same allowlist.

## GPT-5.6 Sol on Azure

An Azure (Microsoft Foundry) deployment serves GPT-5.6 Sol on Azure's v1 API
under OpenAI's own chat-completions contract, with no chat template.

1. Name the deployment `gpt-5.6-sol`. The request names the deployment, so
   the deployment carries the card's alias.

2. Point MAPS at the deployment and check it. The deployment's key is the
   bearer token:

   ```sh
   export SEARCH_REASONER_URL=https://<resource>.openai.azure.com/openai/v1
   export SEARCH_REASONER_API_KEY=<deployment key>
   export SEARCH_ENDPOINT_SPEC_OVERRIDE=gpt-5.6-sol:262144
   python -m maps.mission.endpoint_preflight --effort none --effort medium --effort xhigh
   ```

3. Fly a run as any other. For example, the reviewed Cal City episodes at
   one effort:

   ```sh
   SEARCH_SAMPLING_REASONING_EFFORT=medium \
     python -m benchmarks.calcity.bench_fullset --episode-allowlist output/reviewed_episodes.txt \
     --out-root output/gpt56-sol-medium
   ```

4. Read the run's spend, flying attempts and re-queued ones included, off its
   transcripts:

   ```sh
   python -m benchmarks.citynav.analysis.spend <out-root>
   ```

The card takes the efforts `none`, `low`, `medium`, `high`, `xhigh`, and
`max`.

| Topic | Behaviour |
|---|---|
| Request shape | The card (`wire: "openai"` in `maps/llm/model_cards.py`) shapes the request to OpenAI's contract (`maps/llm/openai_wire.py`): no sampling envelope or template kwargs, the output cap as `max_completion_tokens`, and each message with only the fields its role takes. |
| Tool documentation | OpenAI refuses a function description over 1,024 characters, so the `run` tool's documentation rides the system turn verbatim under its own heading and the tool points at it. |
| Images | Azure refuses a GPT-5.x request carrying more than 50 images, so past that the reasoner evicts the oldest crops, down to 30. |
| Responses API | Azure's chat completions refuse function tools beside any reasoning effort but `none`, so the card (`api: "responses"`) flies the Responses API. The engine builds and records the chat-shaped request, and `openai_wire.to_responses` and `from_responses` translate it at the wire, statelessly (`store: false`). |
| Reasoning | The thinking span doesn't come back in the clear. Each assistant turn carries the encrypted reasoning items its response returned, and the next request sends them back before that turn's calls, as a self-hosted chat template renders an earlier turn's `reasoning_content`. The compactor prices those items by the reasoning tokens that wrote them. The reasoning count is the provider's meter. |
| Discovery | Azure's `/openai/v1/models` lists the resource's base models and leaves out its deployments. For this card, discovery takes the listing answering the key as the live check, and the first request is refused for a deployment name the resource doesn't carry. |
| Rate limit | A Global Standard deployment's tokens-per-minute limit admits each request at its prompt plus `max_completion_tokens` (the output reserve, 40,000 tokens at the pinned window), whatever it bills. Size the limit to the bench's concurrency on that count. |
| Window | The override pins the window Inkling Small is served with, so the compactor holds GPT-5.6 Sol to the same context budget as the other runs. The model's own window is 1,050,000 tokens. |
| Prompt cache | Each request carries the episode's routing key as `prompt_cache_key`, and every turn's bytes open with the last turn's until an image eviction or a compaction fire rewrites an earlier message, so Azure's prompt cache serves the prefix. The preflight fails an effort whose second turn reads no cache hit. |
