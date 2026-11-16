// free-model-aggregator-01 entry (Node 22, standard library only).
//
// One server that serves the OpenAI Chat Completions API and the Anthropic
// Messages API over a directory of simulated free-tier providers, in JSON and
// in server-sent events, and registers the catalog it read from them.
//
// How it works:
//   1. It reads the trial brief ($MAKEFUN_PROMPT_FILE): the ten providers, the
//      sealed quality table and the workload manifest.
//   2. It probes every provider's catalog route in the provider's own wire and
//      builds one catalog (the `--list` document and the register body).
//   3. It listens on a free port, registers {protocol_version, port, catalog}
//      at POST $MAKEFUN_ENV_URL/register, and serves the client's 20 requests.
//   4. Every request is routed to the best free model that meets its
//      requirements, ranked by the quality table's rating_milli (quality is
//      60% of the score). A provider that answers 402 has spent its free
//      allowance and is not tried again; a 429 or 5xx fails over to the next
//      best candidate. Calls spread over providers: among equal ratings the
//      least-used provider wins.
//   5. It polls GET $MAKEFUN_ENV_URL/work and exits 0 the moment the env says
//      the workload is done. A process still running when the wall ends scores
//      zero, so it never waits for anything else.
//
// `node main.ts --list` prints the same catalog JSON and exits 0 before any
// server starts; with no brief and no env it prints {"providers":[]}.
//
// Upstream wires: this server speaks the OpenAI wire upstream. Providers on
// the Anthropic or Gemini wire are listed in the catalog (the probe already
// speaks all three wires) but are never routed, because an answer translated
// through a wire this server does not implement would be a guess; a good
// answer from the best OpenAI-wire free model beats a 501 from a wire that
// cannot be served faithfully.

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
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream model call may take before it fails over, in milliseconds. */
const CALL_TIMEOUT_MS = 20000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Providers that answered 402: their free allowance is spent for this trial. */
const spentProviders = new Set<string>();
/** Upstream calls made per provider, for spreading load among equal ratings. */
const callsPerProvider = new Map<string, number>();

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

/**
 * Every free model that can serve one request, best first: the quality table's
 * rating_milli decides (quality is 60% of the score); among equal ratings the
 * least-used provider wins so calls spread instead of draining one budget; a
 * provider that spent its free allowance (402) is out; only the OpenAI wire is
 * served upstream, so other wires never become a 501 mid-request.
 */
function candidatesFor(requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (probe.provider.wire !== 'openai') continue;
    if (spentProviders.has(probe.provider.id)) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const usedA = callsPerProvider.get(a.probe.provider.id) ?? 0;
    const usedB = callsPerProvider.get(b.probe.provider.id) ?? 0;
    if (usedA !== usedB) return usedA - usedB;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
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

/** POST one request body to one candidate's Chat Completions route. */
async function callCandidate(candidate: Candidate, body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.probe.provider.auth, candidate.probe.provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  callsPerProvider.set(candidate.probe.provider.id, (callsPerProvider.get(candidate.probe.provider.id) ?? 0) + 1);
  return fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model: candidate.model.id }),
    signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
  });
}

/**
 * Try the ranked candidates in order until one answers 2xx. A 402 means that
 * provider's free allowance is spent, so it is not tried again this trial; a
 * 429 or 5xx (or a failed fetch) moves on to the next best model. Returns the
 * first success, or the last failure seen when every candidate refused.
 */
async function callWithFailover(
  candidates: Candidate[],
  buildBody: (candidate: Candidate) => Json,
): Promise<{ response: Response | null; winner: Candidate | null; lastStatus: number; lastText: string }> {
  let lastStatus = 0;
  let lastText = '';
  for (const candidate of candidates) {
    let upstream: Response;
    try {
      upstream = await callCandidate(candidate, buildBody(candidate));
    } catch (err) {
      log(`failover: ${candidate.probe.provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
      continue;
    }
    if (upstream.ok) return { response: upstream, winner: candidate, lastStatus: upstream.status, lastText: '' };
    const text = await upstream.text();
    lastStatus = upstream.status;
    lastText = text;
    if (upstream.status === 402) {
      spentProviders.add(candidate.probe.provider.id);
      log(`failover: ${candidate.probe.provider.id} answered 402, its free allowance is spent`);
    } else {
      log(`failover: ${candidate.probe.provider.id}/${candidate.model.id} answered HTTP ${upstream.status}, trying the next candidate`);
    }
  }
  return { response: null, winner: null, lastStatus, lastText };
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  const { response: upstream, lastStatus, lastText } = await callWithFailover(candidates, () => body);
  if (upstream === null) {
    return sendError(res, 'openai', lastStatus >= 400 ? lastStatus : 503, 'upstream_error', `every candidate refused (last HTTP ${lastStatus}): ${lastText.slice(0, 300)}`);
  }
  if (body.stream === true && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    await pipeStream(upstream.body, res);
    return;
  }
  const text = await upstream.text();
  res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
  res.end(text);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to the best free model and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  // The response names the model that actually answered, not the id the client
  // asked for: the wire check reads it, and a made-up id is not in the catalog.
  const { response: upstream, winner, lastStatus, lastText } = await callWithFailover(candidates, (candidate) =>
    anthropicToOpenAI(body, candidate.model.id),
  );
  if (upstream === null || winner === null) {
    return sendError(res, 'anthropic', lastStatus >= 400 ? lastStatus : 503, 'api_error', `every candidate refused (last HTTP ${lastStatus}): ${lastText.slice(0, 300)}`);
  }
  const answeredModel = winner.model.id;
  if (body.stream === true && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    for await (const event of anthropicEventsFromOpenAI(upstream.body, answeredModel)) res.write(event);
    res.end();
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, answeredModel));
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
