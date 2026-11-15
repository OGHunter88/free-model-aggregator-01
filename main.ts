// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// Routing: every request's free candidates are ranked by the quality table's rating_milli. The first
// pick spreads inside a 15-milli band of the best rating by fewest calls so far, so no one provider's
// per-trial budget is spent on requests another provider answers nearly as well; failover then walks
// the ranking strictly by rating. A 402 marks a provider dead for the trial; a 429 or 5xx moves on to
// the next candidate. Upstream wires: OpenAI, Anthropic and Gemini, in both directions, with the
// client's system prompt, user text and images, tools, stop sequences, max_tokens and JSON mode carried
// through untouched, and the provider's usage reported back verbatim.
//
// Determinism: every choice comes from the brief, the requests and the providers' answers. No clock,
// no unseeded randomness, so a replay leaves the same state export, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
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
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToAnthropicResponse,
  geminiToOpenAIResponse,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most upstream attempts for one client request: enough for every provider, bounded so a request ends. */
const MAX_ATTEMPTS = 8;
/** The rating band, in milli-points, the first pick spreads inside: a model within this of the best is nearly as good, and using it saves the best provider's budget. */
const SPREAD_MILLI = 15;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Provider calls made so far this trial, by provider id: the spread and the failover read it. */
const callsByProvider = new Map<string, number>();
/** Providers whose budget is spent (a 402): never called again this trial. */
const deadProviders = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
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

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

const candidateKey = (c: Candidate): string => `${c.probe.provider.id}/${c.model.id}`;

/**
 * Every free candidate that meets one request's requirements, best rating first. A JSON-mode request
 * cannot ride an anthropic wire: that wire's echo never carries json_mode, so the client would see
 * the request's JSON mode lost. Ties break by provider id, then model id: the ranking is deterministic.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (jsonMode && probe.provider.wire === 'anthropic') continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/** The first pick: the best rating, spread inside the band by fewest calls so far. */
function pickFirst(cands: readonly Candidate[], tried: ReadonlySet<string>): Candidate | null {
  const live = cands.filter((c) => !deadProviders.has(c.probe.provider.id) && !tried.has(candidateKey(c)));
  if (live.length === 0) return null;
  const best = live[0].rating;
  const band = live.filter((c) => c.rating >= best - SPREAD_MILLI);
  band.sort(
    (a, b) =>
      (callsByProvider.get(a.probe.provider.id) ?? 0) - (callsByProvider.get(b.probe.provider.id) ?? 0) ||
      b.rating - a.rating ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return band[0];
}

/** A failover pick: the best rating left, skipping dead providers and candidates already tried. */
function pickNext(cands: readonly Candidate[], tried: ReadonlySet<string>): Candidate | null {
  for (const c of cands) {
    if (deadProviders.has(c.probe.provider.id)) continue;
    if (tried.has(candidateKey(c))) continue;
    return c;
  }
  return null;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
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

/** Whether the client's request asks for JSON mode (an OpenAI client says so with response_format). */
function jsonModeOf(body: Json): boolean {
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
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

interface UpstreamCall {
  url: string;
  body: Json;
}

/** The upstream request for one candidate: the provider's own route, wire and auth. */
function upstreamCall(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const model = candidate.model.id;
  if (provider.wire === 'openai') {
    const upstream = clientApi === 'anthropic' ? anthropicToOpenAI(body, model) : { ...body, model };
    // A streaming OpenAI answer must carry its final usage chunk: the client checks the usage it
    // reports against the provider's, and the provider only sends that chunk when the request asks
    // for it. Ask for it on every streamed call, whatever the client sent.
    if (body.stream === true) {
      const options = isRecord(upstream.stream_options) ? { ...upstream.stream_options } : {};
      options.include_usage = true;
      upstream.stream_options = options;
    }
    return { url: `${base}/v1/chat/completions`, body: upstream };
  }
  if (provider.wire === 'anthropic') {
    const upstream = clientApi === 'openai' ? openAIToAnthropicRequest(body, model) : { ...body, model };
    return { url: `${base}/v1/messages`, body: upstream };
  }
  // gemini: the model goes in the URL, never in the body.
  const upstream = clientApi === 'anthropic' ? anthropicToGeminiRequest(body) : openAIToGeminiRequest(body);
  const stream = body.stream === true;
  return { url: `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`, body: upstream };
}

/** POST one upstream request; null when the provider could not be reached at all. */
async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<Response | null> {
  const provider = candidate.probe.provider;
  const { url, body: upstreamBody } = upstreamCall(candidate, clientApi, body);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (body.stream === true) headers.accept = 'text/event-stream';
  try {
    return await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch {
    return null;
  }
}

/** Record one provider call against its budget, and mark the provider dead on a 402. */
function noteCall(candidate: Candidate, status: number | null): void {
  const id = candidate.probe.provider.id;
  callsByProvider.set(id, (callsByProvider.get(id) ?? 0) + 1);
  if (status === 402) deadProviders.add(id);
}

/** Translate one successful upstream JSON answer into the client's wire. */
function translateJson(clientApi: 'openai' | 'anthropic', candidate: Candidate, answer: Json): Json {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (clientApi === 'openai') {
    if (wire === 'anthropic') return anthropicToOpenAIResponse(answer, model);
    if (wire === 'gemini') return geminiToOpenAIResponse(answer, model);
    return answer;
  }
  if (wire === 'openai') return openAIToAnthropic(answer, model);
  if (wire === 'gemini') return geminiToAnthropicResponse(answer, model);
  return answer;
}

/** Stream one successful upstream event stream to the client, in the client's wire. */
async function pipeTranslatedStream(clientApi: 'openai' | 'anthropic', candidate: Candidate, body: Json, stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if (clientApi === 'openai' && wire === 'openai') {
    const reader = stream.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } finally {
      reader.releaseLock();
    }
  } else if (clientApi === 'openai' && wire === 'anthropic') {
    // The usage chunk is always emitted: the client checks the streamed usage against the provider's.
    for await (const event of openAIEventsFromAnthropic(stream, model, true)) res.write(event);
  } else if (clientApi === 'openai' && wire === 'gemini') {
    for await (const event of openAIEventsFromGemini(stream, model, true)) res.write(event);
  } else if (clientApi === 'anthropic' && wire === 'openai') {
    for await (const event of anthropicEventsFromOpenAI(stream, model)) res.write(event);
  } else if (clientApi === 'anthropic' && wire === 'gemini') {
    for await (const event of anthropicEventsFromGemini(stream, model)) res.write(event);
  } else {
    // anthropic client, anthropic provider: the provider's events are already the client's.
    const reader = stream.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } finally {
      reader.releaseLock();
    }
  }
  res.end();
}

/**
 * Serve one client request: rank the candidates, call the first pick, and on a 402, 429 or 5xx (or an
 * unreachable provider, or an answer that is not JSON) fail over to the next candidate, until one
 * answers or the attempts run out. Nothing is written to the client until an upstream call succeeds,
 * so a failed first pick never costs the request its one answering provider.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const requires = requirementsFor(body);
  const cands = candidatesFor(requires, jsonModeOf(body));
  if (cands.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  const tried = new Set<string>();
  let candidate = pickFirst(cands, tried);
  let lastStatus = 0;
  for (let attempt = 0; attempt < MAX_ATTEMPTS && candidate !== null; attempt += 1) {
    tried.add(candidateKey(candidate));
    const upstream = await callUpstream(candidate, clientApi, body);
    if (upstream === null) {
      noteCall(candidate, null);
      candidate = pickNext(cands, tried);
      continue;
    }
    noteCall(candidate, upstream.status);
    if (!upstream.ok) {
      lastStatus = upstream.status;
      candidate = pickNext(cands, tried);
      continue;
    }
    if (body.stream === true && upstream.body !== null) {
      await pipeTranslatedStream(clientApi, candidate, body, upstream.body, res);
      return;
    }
    let answer: unknown;
    try {
      answer = await upstream.json();
    } catch {
      candidate = pickNext(cands, tried);
      continue;
    }
    if (!isRecord(answer)) {
      candidate = pickNext(cands, tried);
      continue;
    }
    sendJson(res, 200, translateJson(clientApi, candidate, answer));
    return;
  }
  return sendError(res, clientApi, lastStatus === 402 ? 402 : 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('openai', body, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('anthropic', body, res);
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
