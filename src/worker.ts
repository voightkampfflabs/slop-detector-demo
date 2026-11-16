/// <reference lib="webworker" />
import type * as Ort from 'onnxruntime-web';
import { Tokenizer } from '@huggingface/tokenizers';
// ONNX Runtime's own loaders and binaries, emitted as separate files. Its thread
// workers must start from the loader, not from this bundled script: if they
// load this file they run init() again and replace the runtime's message
// handler, and session creation never returns. The WebGPU build of the runtime
// has its own pair (asyncify); the plain pair is half the size and CPU only.
import wasmLoaderUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import wasmBinaryUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import gpuLoaderUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';
import gpuBinaryUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import type { Backend, Bands, FromWorker, ToWorker } from './protocol';
import { modelInput } from './text';

const REPO = 'https://huggingface.co/VoightKampffLabs/voi-slop-detect-380M/resolve/main';
const MODEL_URL = `${REPO}/onnx/model_quantized.onnx`;
/** Size of model_quantized.onnx, used when the response carries no Content-Length. */
const MODEL_BYTES = 450_063_595;
const MAX_TOKENS = 512;
/** Shorter texts are outside what the model was trained on and are not scored. */
const MIN_TOKENS = 30;
/** Named after the export: the opset 18 graph of the old cache is wrong on WebGPU. */
const CACHE_NAME = 'voi-slop-detect-380M-q8-opset23';
const OLD_CACHES = ['voi-slop-detect-380M-q8'];

const post = (message: FromWorker) => self.postMessage(message);

const t0 = performance.now();
const seconds = () => `${((performance.now() - t0) / 1000).toFixed(2)}s`;

function printable(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value instanceof Error) return value.stack || value.message;
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}

/** Sends one trace line to the page: a phone has no console to read. */
function trace(args: unknown[]): void {
  post({ type: 'log', line: `${seconds()} ${args.map(printable).join(' ')}` });
}

/** Trace of every loading and scoring step, with seconds since start. */
function log(...args: unknown[]): void {
  console.log(`[slop ${seconds()}]`, ...args);
  trace(args);
}

// ONNX Runtime reports its warnings and errors on the console: trace those too.
for (const level of ['warn', 'error'] as const) {
  const original = console[level].bind(console);
  console[level] = (...args: unknown[]) => {
    original(...args);
    trace([`console.${level}:`, ...args]);
  };
}

self.addEventListener('error', (event) =>
  trace(['uncaught error:', event.message, `at ${event.filename}:${event.lineno}`]),
);
self.addEventListener('unhandledrejection', (event) =>
  trace(['unhandled rejection:', event.reason]),
);

let tokenizer: Tokenizer;
/** The runtime build of the backend in use; the two builds are loaded on demand. */
let ort: typeof Ort;
let session: Ort.InferenceSession;
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
  for (const name of OLD_CACHES) void caches.delete(name).catch(() => false);

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
  // The cache takes a second stream of the same download, so no second copy of the
  // model sits in memory. A quota error only means the next visit downloads again.
  const stored = cache?.put(MODEL_URL, response.clone()).then(
    () => log('model: stored in cache'),
    (error) => log('model: could not be cached:', describe(error)),
  );
  const bytes = await readWithProgress(response, false);
  log(`model: ${bytes.length} bytes downloaded`);

  // Awaited: whatever the cache write still buffers must be gone before the session
  // is created, which is where memory peaks (mobile Safari kills the tab there).
  await stored;
  return bytes;
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
function firstValue(tensor: Ort.Tensor): number {
  const data = tensor.data as ArrayLike<number | bigint>;
  // Older runtimes hand float16 back as raw bits in a Uint16Array.
  if (tensor.type === 'float16' && data instanceof Uint16Array) return halfToNumber(data[0]);
  return Number(data[0]);
}

/** One forward pass over token ids; the logit of the single output. */
async function infer(ids: number[]): Promise<number> {
  const n = ids.length;
  const output = await session.run({
    input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, n]),
    attention_mask: new ort.Tensor('int64', new BigInt64Array(n).fill(1n), [1, n]),
  });
  return firstValue(output.logits);
}

/** True when the browser offers a hardware WebGPU adapter. */
async function hasGpu(): Promise<boolean> {
  type Adapter = { info?: { isFallbackAdapter?: boolean }; isFallbackAdapter?: boolean };
  const gpu = (navigator as { gpu?: { requestAdapter(): Promise<Adapter | null> } }).gpu;
  const adapter = await gpu?.requestAdapter().catch(() => null);
  // A fallback adapter is a software rasteriser: far slower than WebAssembly.
  return !!adapter && !(adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter);
}

/** Loads the runtime build of one backend and creates the session on it. */
async function createSession(backend: Backend, bytes: Uint8Array, threads: number): Promise<void> {
  if (backend === 'webgpu') {
    ort = await import('onnxruntime-web/webgpu');
    ort.env.wasm.wasmPaths = { mjs: gpuLoaderUrl, wasm: gpuBinaryUrl };
  } else {
    ort = await import('onnxruntime-web/wasm');
    ort.env.wasm.wasmPaths = { mjs: wasmLoaderUrl, wasm: wasmBinaryUrl };
  }
  // The GPU does the work on WebGPU: one thread, so no shared memory or thread workers.
  ort.env.wasm.numThreads = backend === 'webgpu' ? 1 : threads;
  // ONNX Runtime's own messages (graph loading, optimisation) go to the console too.
  ort.env.logLevel = 'info';
  session = await ort.InferenceSession.create(bytes, {
    executionProviders: [backend],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 1,
  });
  if (backend === 'webgpu') {
    // Compiles the shaders before the first real text, and proves the GPU path runs:
    // a driver that fails here sends us to WebAssembly instead of failing on the user.
    try {
      const logit = await infer(new Array<number>(MIN_TOKENS).fill(0));
      if (!Number.isFinite(logit)) throw new Error(`warm-up returned ${logit}`);
    } catch (error) {
      // Frees the model's copy in this runtime's heap and its GPU buffers. The
      // WebAssembly fallback is a second runtime and loads the model again.
      await session.release().catch(() => undefined);
      throw error;
    }
  }
}

/**
 * WebGPU when the browser has a real adapter, WebAssembly otherwise. One backend
 * at a time: given a list, the runtime falls back silently and does not say which
 * provider it picked.
 */
async function createSessionOnBestBackend(bytes: Uint8Array, threads: number): Promise<Backend> {
  if (await hasGpu()) {
    try {
      await createSession('webgpu', bytes, threads);
      return 'webgpu';
    } catch (error) {
      log('webgpu: failed, falling back to wasm:', describe(error));
    }
  } else {
    log('webgpu: no hardware adapter');
  }
  await createSession('wasm', bytes, threads);
  return 'wasm';
}

async function score(id: number, text: string): Promise<void> {
  const all = tokenizer.encode(modelInput(text)).ids;
  const tokens = all.length - specialTokens;
  const limits = { minTokens: MIN_TOKENS, maxTokens: MAX_TOKENS - specialTokens };

  if (tokens < MIN_TOKENS) {
    log(`score #${id}: ${tokens} tokens, under ${MIN_TOKENS}, model not run`);
    post({ type: 'result', id, result: { tokens, ...limits, score: null, inferenceMs: null } });
    return;
  }

  // The only special token is the leading <|endoftext|>, so plain truncation
  // matches what the Python tokenizer does with max_length=512.
  const started = performance.now();
  const logit = await infer(all.slice(0, MAX_TOKENS));
  const inferenceMs = performance.now() - started;

  const value = 1 / (1 + Math.exp(-logit));
  log(
    `score #${id}: ${tokens} tokens${all.length > MAX_TOKENS ? ` (first ${limits.maxTokens} scored)` : ''},`,
    `logit ${logit.toFixed(3)}, score ${value.toFixed(3)},`,
    `inference ${Math.round(inferenceMs)} ms`,
  );

  post({ type: 'result', id, result: { tokens, ...limits, score: value, inferenceMs } });
}

async function init(): Promise<void> {
  const threads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 1, 8) : 1;
  log(
    `start: crossOriginIsolated=${self.crossOriginIsolated},`,
    `hardwareConcurrency=${navigator.hardwareConcurrency}, threads=${threads},`,
    `navigator.gpu=${'gpu' in navigator}`,
  );

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
  const backend = await createSessionOnBestBackend(bytes, threads);
  log(
    `session created on ${backend} in ${Math.round(performance.now() - sessionStart)} ms;`,
    `inputs ${session.inputNames.join(', ')}; outputs ${session.outputNames.join(', ')}`,
  );

  post({ type: 'ready', bands, backend, threads });
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
