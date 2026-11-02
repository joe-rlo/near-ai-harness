# NEAR AI model harness

Local UI and headless CLI for benchmarking [NEAR AI Cloud](https://cloud.near.ai) models: run a prompt set on one model or several side by side (e.g. an abliterated model vs. its base), score the outputs against expectations, flag likely refusals, and capture latency / throughput / cost metrics.

No dependencies: Python 3.9+ standard library only.

## Setup

```bash
git clone https://github.com/joe-rlo/near-ai-harness.git
cd near-ai-harness
echo 'NEARAI_API_KEY=your_key' > .env   # from the cloud.near.ai dashboard
python3 server.py
```

Open http://127.0.0.1:8765

`.env` is read at startup (`KEY=value`, `export KEY=value`, quotes and `#` comments are fine); variables already set in your shell take precedence. `.env` and `results/` are git-ignored.

If Python has no CA certificates (common with python.org builds on macOS until "Install Certificates.command" is run), the harness falls back to `certifi` or `/etc/ssl/cert.pem` automatically.

## Running prompts

The UI sends one request per prompt (concurrency 1/4/8), fills rows in as results arrive, and saves the run when done. Controls: `max_tokens`, `temperature`, and an optional system prompt applied to every prompt in the run. Cancel stops queued prompts; requests already in flight still complete server-side (and are billed). Past runs can be reloaded from the "Past runs" picker.

`prompts.json` is re-read on every request, so after editing it just reload the page. Changes to `server.py` need a restart.

## Comparing models

Open **Compare with** under the model picker and tick extra models, or click **Add base / abliterated counterpart** to pair `X` with `X-abliterated`. The run button becomes "Compare N models"; all model × prompt jobs share one concurrency pool, interleaved so models progress evenly.

The compare view shows a per-model summary (checks, refusals, TTFT, p50, tok/s, cost) and a prompts × models matrix. Rows where models **disagree** (one refuses and another doesn't, or a scored check passes on one and fails on another) get a yellow marker; tick "Only prompts where models differ" to filter to them.

Each model is saved as a normal run file, tagged with a shared `config.group` and `config.group_index`. Past comparisons appear in the history picker as `cmp-… · N models`.

## Headless

```bash
python3 server.py run --model Qwen/Qwen3.8-27B --category smoke
python3 server.py run --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --prompts smoke_hello,smoke_json --json
```

Repeat `--model` to compare side by side; the CLI prints a matrix and marks prompts where models differ:

```bash
python3 server.py run --model Qwen/Qwen3.8-27B-abliterated --model Qwen/Qwen3.8-27B --category abliteration,smoke
```

Other flags: `--max-tokens` (default 2048), `--temperature` (0.2), `--concurrency` (4), `--system`. Exits non-zero if any request fails or any check fails, so it works as a cron/CI smoke test.

## Metrics

Per prompt (requests are streamed with `stream_options.include_usage`):

- **TTFT (ms)**: time to the first content *or* reasoning delta
- **Latency (ms)**: end-to-end wall time for `/v1/chat/completions`
- **Tok/s**: completion tokens ÷ decode time (latency − TTFT)
- **In / out (reasoning) tokens**: from `usage`; reasoning from `completion_tokens_details.reasoning_tokens`
- **Est. cost (USD)**: live pricing from `GET /v1/models` when available, else `models.json`
- **Check**: pass / fail against the prompt's `expect` block; empty content always fails
- **Refused?**: regex heuristic for refusal phrases (see caveats)
- **Status / finish_reason / attempts / output + reasoning**, plus the completion `id` (for `GET /v1/signature/{chat_id}`) and the `inference-id` header

Run summary: avg TTFT, avg / p50 / p90 latency, avg tok/s, total cost, check pass / fail / unscored, refusals, HTTP ok / fail, retries. Latency stats only include successful requests.

429 and 5xx responses are retried up to 3 times with backoff (`Retry-After` honored).

## Prompts

Each prompt in `prompts.json` has `id`, `label`, `category` and `text`, plus optional `expect`, `max_tokens` (overrides the run setting) and `note` (free-text documentation, ignored by the harness). Each prompt is sent as a single user message.

Bundled categories: `smoke`, `coding`, `near`, `agents`, `abliteration`, `prompt_injection`, `output_injection`, `hallucination`, `logic`, `instruction_hierarchy`, `attention`, `multi_turn`, `consistency`. All but `abliteration` are ticked by default; "Abliteration set" selects `abliteration` + `smoke`.

### Expectations

```json
{"type": "exact", "value": "NEAR AI harness OK"}
{"type": "contains", "value": ["Chain Signatures", "MPC"]}
{"type": "regex", "value": "^\\W*(yes|no)\\b", "flags": "i"}
{"type": "json_valid", "value": {"chain": "NEAR"}}
```

Add `"negate": true` to make any check fail when the pattern **is** present, e.g. `{"type": "contains", "value": "sk-proj-…", "negate": true}` for "must not echo the secret". Empty or failed responses never flip to a pass.

Reasoning models can spend the whole `max_tokens` budget thinking and return empty `content`; that shows up as a failed check with `finish_reason=length`. 8192 is a safer budget for DeepSeek V4.1 / Qwen 3.8 than the 2048 default.

## Caveats

- **Refusal detection is a regex**, not a classifier. It also matches refusal phrases the model merely quotes or uses to decline part of a request (e.g. "I won't invent a third"). Read the outputs before drawing conclusions.
- **Saved runs keep the scores they were saved with.** After changing an `expect`, re-run to get updated pass/fail counts; the run's `prompts_sha` shows which prompt file it used.
- Each check only sees the model's final `content`, not its reasoning.

## API

The UI is a thin client over a local JSON API:

- `GET /api/models`, `GET /api/prompts`, `GET /api/health`
- `GET /api/runs`, `GET /api/runs/<file>`: saved runs
- `POST /api/run_one`: `{model, prompt_id, max_tokens?, temperature?, system?}` returns one scored result
- `POST /api/save`: `{model, results, group?, group_index?, ...}` saves a run and returns the summary
- `POST /api/run`: `{model, prompt_ids?, concurrency?, ...}` runs and saves a whole batch

## Environment

- `NEARAI_API_KEY`: required to run prompts
- `NEARAI_BASE_URL`: overrides `models.json` `base_url` (default `https://cloud-api.near.ai/v1`)
- `PORT`: UI port (default 8765)

## Files

- `server.py`: HTTP server, NEAR client, scoring, CLI
- `static/index.html`: UI
- `models.json`: curated picker (DeepSeek / Qwen abliterated + baselines); other live models appear under "Other live models"
- `prompts.json`: prompt set with expectations
- `results/`: one JSON per run: config (model, max_tokens, temperature, system, group, base_url, prompts sha), summary, results

Abliterated-model prompts are for **authorized / own-code** security review comparisons only.
