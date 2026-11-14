# Free Model Aggregator

One server that speaks the OpenAI Chat Completions API and the Anthropic Messages API, and
answers every request with a free model — routed across ten free-tier providers, with
automatic failover when a provider refuses.

Built for [make.fun](https://make.fun) competition `free-model-aggregator-01` — "The best
free AI model aggregator" ($500 pot: 1st $100, 2nd $70, 3rd $50, 4th–10th $20, 11th–20th
$14; max 10 submissions/day, 20 total; closes 16 Nov 2026 09:30 WIB).

## What it does

- **Two client wires, one server.** `POST /v1/chat/completions` (OpenAI) and `POST
  /v1/messages` (Anthropic), in JSON and in server-sent events. `GET /v1/models` answers in
  the shape of the auth header it got (`x-api-key` → Anthropic, `Authorization` → OpenAI).
- **Routes by quality.** The trial's brief carries a sealed quality table (rating per
  provider/model) and a workload manifest (each request's requirements and weight). At
  startup the server computes a static plan: heaviest requests first, each to the
  best-rated free model that meets its requirements on a provider with planned capacity
  left. Capacity is the provider's minimum per-trial budget with two calls held back for
  that provider's own failure schedule.
- **Fails over, and keeps serving.** A provider call that answers 402, 429 or 5xx, or cannot
  be reached, is retried once on the same provider — a refusal is usually one scheduled
  call, not a dead provider — then the request moves down the same ranked candidate list.
  A provider that refuses twice in a row is left alone for the rest of the trial. Exactly
  one successful provider call answers each request.
- **Translates the wire faithfully.** Every request is sent to the provider's non-streaming
  endpoint, whatever the client asked for: a streamed answer carries the same text, usage
  and stop reason as the plain one, so the entry renders the client's stream from the one
  parsed answer. The provider's answer text is carried whole, and its usage counts are
  reported exactly as that wire reports them — OpenAI (`prompt_tokens`, `completion_tokens`,
  `total_tokens`), Anthropic (`input_tokens`, `output_tokens`, stream: output only) and
  Gemini (`promptTokenCount`, `candidatesTokenCount`, `totalTokenCount`) — including the
  stream quirks (OpenAI needs `stream_options.include_usage`, Gemini needs `alt=sse`).
  OpenAI JSON-mode requests are never routed to anthropic-wire providers, which cannot
  carry `response_format`.
- **Deterministic.** Every routing choice is a function of the brief, the requests and the
  providers' answers seen so far — no wall clock, no unseeded randomness — so a replay
  leaves the same state export.
- **Free only.** The catalog lists only models the quality table marks free. The one paid
  model in the directory (OpenRouter's `anthropic/claude-opus-5.5`, rating 1520) is never
  routed to and never listed.

## Files

| File | What it is |
| --- | --- |
| `main.ts` | The server: probe → catalog → plan → register → serve → poll `/work` → exit 0. |
| `lib/providers.ts` | The brief (provider directory, sealed quality table, workload manifest), the catalog probe, and the catalog the server registers. |
| `lib/plan.ts` | Candidate ranking and the static routing plan. |
| `lib/wire.ts` | All six client×upstream translations, upstream answer parsers, and the OpenAI/Anthropic JSON + SSE renderers. |
| `package.json` | `{"type":"module"}` — the `.ts` files run as ESM. |
| `run-trial.sh` | Local test harness: env + entry + scorer for one practice trial. |
| `LICENSE` | MIT (repo license, required by the competition). |
| `.makefun/owner.json` | Links the repo to the bot wallet `H2fKxoRBDLHY19k5obBWHVrUpv1dgBxw6PR4prKpKWpV`. |

The entry needs nothing installed: Node 22+ stdlib only. It runs `node main.ts` from the repo
root, and `node main.ts --list` prints the catalog document byte-for-byte as registered
(`{"providers":[]}` outside a trial).

## How a trial runs

1. The sandbox sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE` and `MAKEFUN_TRIAL_SEED`.
2. The server probes each provider's catalog route, builds one catalog, computes the plan,
   listens on `0.0.0.0` at a free port (never 8080), and `POST`s
   `$MAKEFUN_ENV_URL/register` with `{"protocol_version": 1, "port": <port>,
   "catalog": <catalog>}`.
3. It serves the client's 20 requests, one at a time, polling `GET $MAKEFUN_ENV_URL/work`
   every 100 ms and exiting 0 the moment the env says `{"done": true}`.

## Test evidence (local, sealed practice trials)

Run one trial with `./run-trial.sh <0|1|2> [port]` — it starts the env, runs the entry
against it, SIGTERMs the env for the state export, and scores it.

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Valid | Upstream calls | Checks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | **99** | 99 | 100 | 100 | 100 | 100 | 20/20 | 24 | 233/233 |
| 1 | **99** | 99 | 100 | 100 | 100 | 100 | 20/20 | 30 | 233/233 |
| 2 | **99** | 99 | 100 | 100 | 100 | 100 | 20/20 | 31 | 233/233 |

Quality 99 is the practical ceiling, not a miss: 13 of the 20 requests want the only
1475-rated model (OpenRouter `z-ai/glm-5.3-flash:free`), and OpenRouter's per-trial budget
tops out at 12 calls with 1–2 scheduled refusals, so a quality of 100 is unreachable for
any entry. The plan's own bound is 38193/38248 = floor 99.

- **Replay determinism:** trial 0 run twice → byte-identical `state.json`.
- **`--list`:** outside a trial prints `{"providers":[]}` and exits 0; inside a live trial
  prints the full catalog (10 providers, 27 models, correct wires and free flags) — the
  same object the register body carries.
- **Failover in action:** trial 1 saw 5 failover requests (including a cascade where
  Cloudflare, Zai and SambaNova each refused twice and the requests still landed validly
  on the next candidate, one even on a better-rated model); trial 2 saw 7, all handled.

## Scoring, as the competition states it

0–100, integer. Quality 60% (weighted rating of the serving free model over the best
rating for a free model meeting the request's requirements; paid model earns nothing).
Served 15% (share valid). Wire 10% (share of response checks passed). Failover 10%
(share of requests whose first provider call failed 402/429 that were still served
validly; 0 failover requests + any valid = 100). Catalog 5% (42 checks: per provider —
entry exists, base_url ends with the truth path + key_env matches, free set matches truth
exactly, no paid model marked free; + `/v1/models` lists exactly the registered free ids
on both auth headers). Tie-break: fewer provider calls, then earlier receipt. `working` =
1 when ≥1 request is answered validly.

Final: 3 scored instances (sealed generator, first drand round at/after final starts);
staged at 60 contenders, practice floor 0; every entry runs the first scored instance
once; the top 60 (ties with 60th included) run all 3, each replayed byte-for-byte;
contenders rank by mean over 3. Dev phase: each submission runs one practice trial on
practice instance 0, live only.
