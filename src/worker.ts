/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/wasm';
import { Tokenizer } from '@huggingface/tokenizers';
// ONNX Runtime's own loader and binary, emitted as separate files. Its thread
// workers must start from the loader, not from this bundled script: if they
// load this file they run init() again and replace the runtime's message
// handler, and session creation never returns.
import ortLoaderUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import type { Bands, FromWorker, ToWorker } from './protocol';

const REPO = 'https://huggingface.co/VoightKampffLabs/voi-slop-detect-380M/resolve/main';
const MODEL_URL = `${REPO}/onnx/model_quantized.onnx`;
/** Size of model_quantized.onnx, used when the response carries no Content-Length. */
const MODEL_BYTES = 440_745_440;
const MAX_TOKENS = 512;
/** Shorter texts are outside what the model was trained on and are not scored. */
const MIN_TOKENS = 30;
const CACHE_NAME = 'voi-slop-detect-380M-q8';

const post = (message: FromWorker) => self.postMessage(message);

const t0 = performance.now();
/** Console trace of every loading and scoring step, with seconds since start. */
function log(...args: unknown[]): void {
  console.log(`[slop ${((performance.now() - t0) / 1000).toFixed(2)}s]`, ...args);
}

let tokenizer: Tokenizer;
let session: ort.InferenceSession;
/** How many special tokens the tokenizer adds to every text. */
let specialTokens = 0;

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${REPO}/${path}`);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

/** Reads a response body into one buffer, reporting progress as it goes. */
async function readWithProgress(response: Response, cached: boolean): Promise<Uint8Array> {
  const total = Number(response.headers.get('Content-Length')) || MODEL_BYTES;
  let buffer = new Uint8Array(total);
  let loaded = 0;
  let lastReport = 0;

  const reader = response.body!.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > buffer.length) {
      // The size estimate was too small: grow.
      const grown = new Uint8Array(Math.max(buffer.length * 2, loaded + value.length));
      grown.set(buffer.subarray(0, loaded));
      buffer = grown;
    }
    buffer.set(value, loaded);
    loaded += value.length;

    const now = performance.now();
    if (now - lastReport > 50) {
      lastReport = now;
      post({ type: 'progress', loaded, total: Math.max(total, loaded), cached });
    }
  }
  post({ type: 'progress', loaded, total: loaded, cached });
  return buffer.subarray(0, loaded);
}

/** Downloads the model, or reads it back from the Cache API on later visits. */
async function loadModelBytes(): Promise<Uint8Array> {
  // The Cache API can be missing or throw (private windows, blocked storage).
  const cache = await caches.open(CACHE_NAME).catch(() => null);

  log('cache storage', cache ? 'available' : 'unavailable');

  const hit = await cache?.match(MODEL_URL).catch(() => undefined);
  if (hit?.body) {
    log('model: reading from cache');
    const bytes = await readWithProgress(hit, true);
    log(`model: ${bytes.length} bytes from cache`);
    return bytes;
  }

  log('model: downloading', MODEL_URL);
  const response = await fetch(MODEL_URL);
  if (!response.ok || !response.body) throw new Error(`model download: HTTP ${response.status}`);
  log(`model: HTTP ${response.status}, Content-Length ${response.headers.get('Content-Length')}`);
  const bytes = await readWithProgress(response, false);
  log(`model: ${bytes.length} bytes downloaded`);

  // Not awaited: storing 441 MB should not delay the first score, and a quota
  // error only means the next visit downloads again.
  cache
    ?.put(MODEL_URL, new Response(bytes.slice(), { headers: { 'Content-Length': String(bytes.length) } }))
    .then(
      () => log('model: stored in cache'),
      (error) => log('model: could not be cached:', describe(error)),
    );
  return bytes;
}

/** The model was trained with fenced code blocks emptied to a bare ``` pair. */
function emptyCodeBlocks(text: string): string {
  return text.replace(/```[^\n]*\n[\s\S]*?```/g, '```\n```');
}

function halfToNumber(half: number): number {
  const sign = half & 0x8000 ? -1 : 1;
  const exponent = (half >> 10) & 0x1f;
  const fraction = half & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** First value of a float tensor. The q8 graph may emit float16 logits. */
function firstValue(tensor: ort.Tensor): number {
  const data = tensor.data as ArrayLike<number | bigint>;
  // Older runtimes hand float16 back as raw bits in a Uint16Array.
  if (tensor.type === 'float16' && data instanceof Uint16Array) return halfToNumber(data[0]);
  return Number(data[0]);
}

async function score(id: number, text: string): Promise<void> {
  const all = tokenizer.encode(emptyCodeBlocks(text)).ids;
  const tokens = all.length - specialTokens;
  const limits = { minTokens: MIN_TOKENS, maxTokens: MAX_TOKENS - specialTokens };

  if (tokens < MIN_TOKENS) {
    log(`score #${id}: ${tokens} tokens, under ${MIN_TOKENS}, model not run`);
    post({ type: 'result', id, result: { tokens, ...limits, score: null, inferenceMs: null } });
    return;
  }

  // The only special token is the leading <|endoftext|>, so plain truncation
  // matches what the Python tokenizer does with max_length=512.
  const ids = all.slice(0, MAX_TOKENS);
  const n = ids.length;
  const feeds = {
    input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, n]),
    attention_mask: new ort.Tensor('int64', new BigInt64Array(n).fill(1n), [1, n]),
  };

  const started = performance.now();
  const output = await session.run(feeds);
  const inferenceMs = performance.now() - started;

  const logit = firstValue(output.logits);
  const value = 1 / (1 + Math.exp(-logit));
  log(
    `score #${id}: ${tokens} tokens${all.length > MAX_TOKENS ? ` (first ${limits.maxTokens} scored)` : ''},`,
    `logit ${logit.toFixed(3)} (${output.logits.type}), score ${value.toFixed(3)},`,
    `inference ${Math.round(inferenceMs)} ms`,
  );

  post({ type: 'result', id, result: { tokens, ...limits, score: value, inferenceMs } });
}

async function init(): Promise<void> {
  const threads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 1, 8) : 1;
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.wasmPaths = { mjs: ortLoaderUrl, wasm: ortWasmUrl };
  // ONNX Runtime's own messages (graph loading, optimisation) go to the console too.
  ort.env.logLevel = 'info';
  log(
    `start: crossOriginIsolated=${self.crossOriginIsolated},`,
    `hardwareConcurrency=${navigator.hardwareConcurrency}, threads=${threads}`,
  );
  log('runtime files:', ortLoaderUrl, ortWasmUrl);

  const [bytes, tokenizerJson, tokenizerConfig, bands] = await Promise.all([
    loadModelBytes(),
    fetchJson<object>('tokenizer.json'),
    fetchJson<object>('tokenizer_config.json'),
    fetchJson<Bands>('bands.json'),
  ]);
  log('tokenizer files and bands fetched', bands);

  post({ type: 'compiling' });
  tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
  specialTokens = tokenizer.encode('').ids.length;
  log(`tokenizer built (${specialTokens} special token per text); creating the inference session`);
  const sessionStart = performance.now();
  session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 1,
  });
  log(
    `session created in ${Math.round(performance.now() - sessionStart)} ms;`,
    `inputs ${session.inputNames.join(', ')}; outputs ${session.outputNames.join(', ')}`,
  );

  post({ type: 'ready', bands, threads });
}

const describe = (error: unknown) => (error instanceof Error ? error.message : String(error));

const ready = init();
ready.catch((error) => {
  console.error('[slop] loading failed', error);
  post({ type: 'error', message: describe(error) });
});

self.onmessage = (event: MessageEvent<ToWorker>) => {
  const { id, text } = event.data;
  ready
    .then(() => score(id, text))
    .catch((error) => post({ type: 'error', message: describe(error) }));
};
