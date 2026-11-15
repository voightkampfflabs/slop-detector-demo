// Runs one benchmark configuration and posts the result. One worker per page, like the demo:
// the model never runs on the main thread.

self.onmessage = async ({ data: cfg }) => {
  try {
    self.postMessage(await run(cfg));
  } catch (error) {
    self.postMessage({ error: String(error?.stack ?? error) });
  }
};

function halfToNumber(half) {
  const sign = half & 0x8000 ? -1 : 1;
  const exponent = (half >> 10) & 0x1f;
  const fraction = half & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** First value of a float tensor; float16 can arrive as raw bits in a Uint16Array. */
function firstValue(tensor) {
  if (tensor.type === 'float16' && tensor.data instanceof Uint16Array) return halfToNumber(tensor.data[0]);
  return Number(tensor.data[0]);
}

async function adapterInfo() {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return null;
  const { vendor, architecture, device, description } = adapter.info;
  return {
    vendor,
    architecture,
    device,
    description,
    fallback: adapter.info.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null,
    shaderF16: adapter.features.has('shader-f16'),
  };
}

async function run(cfg) {
  // Unbundled runtime files straight from node_modules: each build picks its own .wasm and
  // starts its thread workers from its own loader.
  const ort = await import(`/ort/ort.${cfg.build}.min.mjs`);
  ort.env.wasm.wasmPaths = '/ort/';
  ort.env.wasm.numThreads = cfg.threads;
  ort.env.logLevel = cfg.verbose ? 'verbose' : 'warning';

  const env = {
    crossOriginIsolated: self.crossOriginIsolated,
    hardwareConcurrency: navigator.hardwareConcurrency,
    userAgent: navigator.userAgent,
    ort: ort.env.versions?.web ?? null,
    adapter: cfg.ep === 'wasm' ? null : await adapterInfo(),
  };

  const loadStart = performance.now();
  const session = await ort.InferenceSession.create(cfg.modelUrl, {
    executionProviders: [cfg.ep],
    graphOptimizationLevel: 'all',
    logSeverityLevel: cfg.verbose ? 0 : 2,
    ...cfg.sessionOptions,
  });
  const loadMs = performance.now() - loadStart;

  const lens = [];
  for (const ids of cfg.inputs) {
    const n = ids.length;
    const once = async () => {
      // Tensors are built inside the timed region on purpose: the page pays for them too.
      const started = performance.now();
      const output = await session.run({
        input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, n]),
        attention_mask: new ort.Tensor('int64', new BigInt64Array(n).fill(1n), [1, n]),
      });
      const logit = firstValue(output.logits);
      return { ms: performance.now() - started, logit };
    };

    const first = await once(); // includes shader compilation and first-touch allocation
    const ms = [];
    let logit = first.logit;
    for (let i = 0; i < cfg.runs; i++) {
      const result = await once();
      ms.push(result.ms);
      logit = result.logit;
    }
    lens.push({ len: n, firstMs: first.ms, ms, logit });
  }

  await session.release();
  return { env, loadMs, lens };
}
