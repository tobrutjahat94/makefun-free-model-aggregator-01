// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry serves the OpenAI Chat Completions API and the Anthropic Messages API over the
// ten simulated free-tier providers the trial gives it, in JSON and in server-sent events, and
// registers the catalog it read from them. Each request is answered with the best free model
// that can serve it, and the server keeps answering when the provider it picked is out.
//
// HOW A TRIAL RUNS
//   The sandbox sets MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE and MAKEFUN_TRIAL_SEED. The brief
//   (lib/providers.ts) carries the provider directory, the sealed quality table and the
//   workload manifest. This server probes each provider's catalog route, builds one catalog,
//   computes the routing plan (lib/plan.ts), then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// HOW A REQUEST IS SERVED
//   The plan (lib/plan.ts) names the first-choice (provider, model) for each request: the
//   best-rated free model meeting its requirements on a provider with planned capacity left.
//   The attempt order is the planned candidate first, then every free model meeting the
//   requirements in rating order. A provider call that answers 402, 429 or 5xx, or that cannot
//   be reached, is retried once on the same provider — a refusal is usually one scheduled call
//   on that provider's failure schedule, not a dead provider — and a provider that refuses
//   twice in a row is left alone for the rest of the trial. Exactly one successful provider
//   call answers a request: the first 2xx ends the attempts, and the answer is rendered in the
//   wire the client used, with the provider's text whole and its usage counts as that wire
//   reports them (lib/wire.ts).
//
//   The upstream call is always the provider's non-streaming endpoint, whatever the client
//   asked for: a streamed answer carries the same text, usage and stop reason as the plain
//   one, and the entry renders the client's stream from the one parsed answer.
//
// Determinism: every routing choice is a function of the brief, the requests and the
// providers' answers seen so far — no wall clock and no unseeded randomness — so a replay
// leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  isRecord,
  joinBase,
  authHeaders,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  openAIError,
  openAIModelList,
  parseUpstreamAnswer,
  renderAnthropic,
  renderOpenAI,
  upstreamRequest,
  type ParsedAnswer,
  type ServedModel,
} from './lib/wire.ts';
import { attemptOrder, candidatesFor, computePlan, type Candidate } from './lib/plan.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let plan = new Map<string, Candidate>();
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** What this server has learned about one provider, from its answers so far. */
interface ProviderState {
  calls: number;
  consecutiveFailures: number;
  dead: boolean;
}
const providerState = new Map<string, ProviderState>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider, build the one catalog, and compute the routing plan. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) {
    providerState.set(probe.provider.id, { calls: 0, consecutiveFailures: 0, dead: false });
    if (!probe.ok && probe.error !== null) log(probe.error);
  }
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  plan = computePlan(brief.workload?.manifest ?? [], brief.ratings, probes);
  log(`catalog: ${catalog.providers.length} providers, ${catalog.providers.reduce((n, p) => n + p.models.length, 0)} models, ${plan.size} requests planned`);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): { requires: Requirement; requestId: string | undefined } {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { requires: entry.requires, requestId: entry.id };
  return { requires: requirementsFromBody(body), requestId: undefined };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One call to one provider for one client request: the parsed answer, or null when it refused. */
async function callCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<ParsedAnswer | null> {
  const provider = candidate.provider.provider;
  const request = upstreamRequest(clientApi, body, provider.wire, candidate.model.id);
  const url = joinBase(ENV_URL, provider.base) + request.path;
  const headers: Record<string, string> = { ...request.headers, ...authHeaders(provider.auth, provider.key) };
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(request.body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch {
    return null;
  }
  if (!res.ok) return null;
  let text: string;
  try {
    text = await res.text();
  } catch {
    return null;
  }
  return parseUpstreamAnswer(provider.wire, text);
}

/** One client request, served by the first provider call that answers it. */
async function serveRequest(clientApi: 'openai' | 'anthropic', body: Json): Promise<{ model: string; answer: ParsedAnswer } | null> {
  const { requires, requestId } = requirementsFor(body);
  const ranked = candidatesFor(requires, clientApi, brief.ratings, probes);
  const planned = requestId !== undefined ? plan.get(requestId) : undefined;
  for (const candidate of attemptOrder(planned, ranked)) {
    const state = providerState.get(candidate.provider.provider.id);
    if (state === undefined || state.dead) continue;
    // A refusal is usually one scheduled call, not a dead provider: retry it once before
    // moving on. Two refusals in a row leave the provider alone for the rest of the trial.
    for (let attempt = 0; attempt < 2; attempt++) {
      state.calls += 1;
      const answer = await callCandidate(candidate, clientApi, body);
      if (answer !== null) {
        state.consecutiveFailures = 0;
        return { model: candidate.model.id, answer };
      }
      state.consecutiveFailures += 1;
      if (state.consecutiveFailures >= 2) {
        state.dead = true;
        log(`${candidate.provider.provider.id} refused twice in a row and is left alone`);
        break;
      }
    }
  }
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered in the OpenAI wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serveRequest('openai', body);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  const stream = body.stream === true;
  const text = renderOpenAI(served.answer, served.model, stream);
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  } else {
    res.writeHead(200, { 'content-type': 'application/json' });
  }
  res.end(text);
}

/** POST /v1/messages: an Anthropic-shaped client, answered in the Anthropic wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serveRequest('anthropic', body);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  const stream = body.stream === true;
  const text = renderAnthropic(served.answer, served.model, stream);
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  } else {
    res.writeHead(200, { 'content-type': 'application/json' });
  }
  res.end(text);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
