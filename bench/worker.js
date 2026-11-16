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

/** The values of a float tensor; float16 can arrive as raw bits in a Uint16Array. */
function values(tensor) {
  if (tensor.type === 'float16' && tensor.data instanceof Uint16Array) return Array.from(tensor.data, halfToNumber);
  return Array.from(tensor.data, Number);
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

  /** One forward pass over a batch: rows shorter than the longest are padded and masked. */
  const once = async (rows) => {
    // Tensors are built inside the timed region on purpose: the page pays for them too.
    const started = performance.now();
    const n = Math.max(...rows.map((row) => row.length));
    const ids = new BigInt64Array(rows.length * n).fill(BigInt(cfg.pad));
    const mask = new BigInt64Array(rows.length * n);
    rows.forEach((row, r) => {
      ids.set(BigInt64Array.from(row, BigInt), r * n);
      mask.fill(1n, r * n, r * n + row.length);
    });
    const output = await session.run({
      input_ids: new ort.Tensor('int64', ids, [rows.length, n]),
      attention_mask: new ort.Tensor('int64', mask, [rows.length, n]),
    });
    const logits = values(output.logits);
    return { ms: performance.now() - started, logits };
  };

  const cases = [];
  for (const rows of cfg.inputs) {
    // Each distinct row alone, to compare with its logit inside the batch. Not timed.
    const alone = new Map();
    if (rows.length > 1) {
      for (const row of rows) if (!alone.has(row.length)) alone.set(row.length, (await once([row])).logits[0]);
    }

    const first = await once(rows); // includes shader compilation and first-touch allocation
    const ms = [];
    let logits = first.logits;
    for (let i = 0; i < cfg.runs; i++) {
      const result = await once(rows);
      ms.push(result.ms);
      logits = result.logits;
    }
    cases.push({
      len: rows[0].length,
      batch: rows.length,
      firstMs: first.ms,
      ms,
      logit: logits[0],
      // How far each row's logit in the batch is from the same row alone: the largest difference.
      batchDelta: rows.length > 1 ? Math.max(...rows.map((row, r) => Math.abs(logits[r] - alone.get(row.length)))) : null,
    });
  }

  await session.release();
  return { env, loadMs, cases };
}
