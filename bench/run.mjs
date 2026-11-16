#!/usr/bin/env node
// In-browser inference benchmark. Every (model, backend, threads) combination runs in a
// fresh Chrome process against the .onnx files in bench/models.
//
//   npm run bench -- --download model_quantized            # fetch a published graph, run everything
//   npm run bench -- --models model_quantized --eps wasm --threads 4,8 --lens 512
//
// Any <name>.onnx copied into bench/models is a variant; logits are compared with --ref.
import { createReadStream, createWriteStream, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { Tokenizer } from '@huggingface/tokenizers';
import { chromium } from 'playwright-core';

const BENCH = path.dirname(fileURLToPath(import.meta.url));
const MODELS = path.join(BENCH, 'models');
const RESULTS = path.join(BENCH, 'results');
const ORT_DIST = path.join(BENCH, '..', 'node_modules', 'onnxruntime-web', 'dist');
const REPO = 'https://huggingface.co/VoightKampffLabs/voi-slop-detect-380M/resolve/main';
const PUBLISHED = ['model_quantized', 'model_fp16', 'model'];

const { values: args } = parseArgs({
  options: {
    models: { type: 'string' }, // names of bench/models/<name>.onnx; default: all of them
    eps: { type: 'string', default: 'wasm,webgpu' },
    // The defaults are the quick check, about 15 s per model: the demo's thread count, a full
    // window, 3 runs. Widen with e.g. --threads 1,4,8 --lens 128,512 --runs 10.
    threads: { type: 'string', default: String(Math.min(os.availableParallelism(), 8)) }, // wasm rows; webgpu uses the last value
    lens: { type: 'string', default: '512' },
    // Texts per forward pass. Above 1, the first row is the text at full length and the others
    // are the text at half length, padded: what a page of mixed comments costs.
    batch: { type: 'string', default: '1' },
    runs: { type: 'string', default: '3' },
    ref: { type: 'string', default: 'model' }, // logits are compared with this model's
    build: { type: 'string' }, // onnxruntime-web build (wasm, webgpu, jspi, all); default per backend
    'session-options': { type: 'string', default: '{}' }, // JSON merged into the session options
    text: { type: 'string', default: path.join(BENCH, 'text.txt') },
    download: { type: 'string' }, // comma-separated published names, or "all"
    channel: { type: 'string', default: 'chrome' }, // Playwright channel; "chromium" for the bundled one
    'chrome-args': { type: 'string', default: '' },
    headed: { type: 'boolean', default: false },
    verbose: { type: 'boolean', default: false },
    timeout: { type: 'string', default: '900' }, // seconds per configuration
  },
});

const list = (value) => value.split(',').map((item) => item.trim()).filter(Boolean);
const numbers = (value) => list(value).map(Number);

async function download(url, file) {
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`${url}: HTTP ${response.status}`);
  const megabytes = (Number(response.headers.get('content-length')) / 1e6).toFixed(0);
  console.error(`downloading ${path.basename(file)} (${megabytes} MB)`);
  await pipeline(Readable.fromWeb(response.body), createWriteStream(`${file}.part`));
  renameSync(`${file}.part`, file);
}

async function ensureFiles() {
  mkdirSync(MODELS, { recursive: true });
  mkdirSync(RESULTS, { recursive: true });
  for (const name of ['tokenizer.json', 'tokenizer_config.json']) {
    if (!existsSync(path.join(MODELS, name))) await download(`${REPO}/${name}`, path.join(MODELS, name));
  }
  const wanted = args.download === 'all' ? PUBLISHED : list(args.download ?? '');
  for (const name of wanted) {
    if (!PUBLISHED.includes(name)) throw new Error(`--download ${name}: published graphs are ${PUBLISHED.join(', ')}`);
    const file = path.join(MODELS, `${name}.onnx`);
    if (!existsSync(file)) await download(`${REPO}/onnx/${name}.onnx`, file);
  }
}

/** Token ids of the benchmark text: one batch per requested length and batch size, and the padding id. */
function inputs(lens, batches) {
  const read = (name) => JSON.parse(readFileSync(path.join(MODELS, name), 'utf8'));
  const tokenizer = new Tokenizer(read('tokenizer.json'), read('tokenizer_config.json'));
  const ids = tokenizer.encode(readFileSync(args.text, 'utf8')).ids;
  const longest = Math.max(...lens);
  if (ids.length < longest) throw new Error(`${args.text} has ${ids.length} tokens, --lens asks for ${longest}`);
  const batchesOf = (len) =>
    batches.map((batch) => [ids.slice(0, len), ...Array(batch - 1).fill(ids.slice(0, len >> 1))]);
  // The tokenizer pads with <|endoftext|>, the token it also puts in front of every text.
  return { pad: ids[0], batches: lens.flatMap(batchesOf) };
}

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm' };

/** Static files with the cross-origin isolation headers that multi-threaded WASM needs. */
function serve() {
  const roots = [['/ort/', ORT_DIST], ['/models/', MODELS], ['/', BENCH]];
  const server = createServer((request, response) => {
    const url = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    const [prefix, root] = roots.find(([p]) => url.startsWith(p));
    const file = path.join(root, url.slice(prefix.length) || 'index.html');
    if (!file.startsWith(root) || !existsSync(file) || !statSync(file).isFile()) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, {
      'Content-Type': TYPES[path.extname(file)] ?? 'application/octet-stream',
      'Content-Length': statSync(file).size,
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cache-Control': 'no-store',
    });
    createReadStream(file).pipe(response);
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server)));
}

function launch() {
  const linux = process.platform === 'linux';
  return chromium.launch({
    channel: args.channel,
    headless: !args.headed,
    args: [
      // A real GPU in headless Chrome on Linux: without these WebGPU is absent or SwiftShader.
      '--enable-unsafe-webgpu',
      '--ignore-gpu-blocklist',
      ...(linux ? ['--enable-features=Vulkan', '--use-angle=vulkan'] : []),
      ...(linux && !args.headed ? ['--disable-vulkan-surface'] : []),
      ...list(args['chrome-args']),
    ],
  });
}

async function runOne(origin, cfg, log) {
  const browser = await launch();
  let timer;
  try {
    const page = await browser.newPage();
    page.on('console', (message) => log.write(`[${cfg.label}] ${message.type()}: ${message.text()}\n`));
    page.on('pageerror', (error) => log.write(`[${cfg.label}] pageerror: ${error.message}\n`));
    await page.goto(origin);
    const timeout = new Promise((resolve) => {
      timer = setTimeout(() => resolve({ error: `timed out after ${args.timeout} s` }), Number(args.timeout) * 1000);
    });
    const result = await Promise.race([page.evaluate((c) => window.bench(c), cfg), timeout]);
    return { ...result, browser: browser.version() };
  } catch (error) {
    return { error: String(error) };
  } finally {
    clearTimeout(timer); // a pending timer would keep the process alive after the report
    await browser.close().catch(() => {});
  }
}

const quantile = (sorted, q) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];

function report(rows, ref) {
  const reference = new Map(); // len -> logit of the first successful row of the reference model
  for (const row of rows) {
    if (row.model === ref && row.logit != null && !reference.has(row.len)) reference.set(row.len, row.logit);
  }
  const head = [
    'model', 'backend', 'threads', 'tokens', 'batch', 'load s', 'first ms', 'median ms', 'ms per text', 'p90 ms', 'min ms',
    'logit', `Δ vs ${ref}`, 'max Δ in batch',
  ];
  const lines = [`| ${head.join(' | ')} |`, `|${head.map(() => '---').join('|')}|`];
  for (const row of rows) {
    if (row.error) {
      lines.push(`| ${row.model} | ${row.ep} | ${row.threads} | | | | | | | | | | | ${row.error.split('\n')[0].slice(0, 120)} |`);
      continue;
    }
    const delta = reference.has(row.len) ? (row.logit - reference.get(row.len)).toFixed(3) : '';
    lines.push(
      `| ${row.model} | ${row.ep} | ${row.threads} | ${row.len} | ${row.batch} | ${(row.loadMs / 1000).toFixed(1)} | ` +
        `${row.firstMs.toFixed(0)} | ${row.median.toFixed(0)} | ${(row.median / row.batch).toFixed(0)} | ${row.p90.toFixed(0)} | ` +
        `${row.min.toFixed(0)} | ${row.logit.toFixed(3)} | ${delta} | ${row.batchDelta?.toFixed(3) ?? ''} |`,
    );
  }
  return lines.join('\n');
}

await ensureFiles();
const available = readdirSync(MODELS).filter((f) => f.endsWith('.onnx')).map((f) => f.slice(0, -5));
const models = args.models ? list(args.models) : available;
const missing = models.filter((name) => !available.includes(name));
if (!models.length || missing.length) {
  console.error(
    `no bench/models/<name>.onnx for: ${missing.join(', ') || '(nothing to run)'}\n` +
      `fetch the published graphs with --download ${PUBLISHED.join(',')} or copy your own export there`,
  );
  process.exit(1);
}

const threads = numbers(args.threads);
const cfgInputs = inputs(numbers(args.lens), numbers(args.batch));
const sessionOptions = JSON.parse(args['session-options']);
const server = await serve();
const origin = `http://127.0.0.1:${server.address().port}`;
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const log = createWriteStream(path.join(RESULTS, `${stamp}.log`));

const rows = [];
let environment = {};
for (const model of models) {
  for (const ep of list(args.eps)) {
    for (const count of ep === 'wasm' ? threads : threads.slice(-1)) {
      const label = `${model} ${ep} t${count}`;
      console.error(`running ${label}`);
      const result = await runOne(origin, {
        label,
        modelUrl: `/models/${model}.onnx`,
        ep,
        build: args.build ?? (ep === 'wasm' ? 'wasm' : 'webgpu'),
        threads: count,
        runs: Number(args.runs),
        inputs: cfgInputs.batches,
        pad: cfgInputs.pad,
        sessionOptions,
        verbose: args.verbose,
      }, log);
      if (result.error) {
        console.error(`  failed: ${result.error.split('\n')[0]}`);
        rows.push({ model, ep, threads: count, error: result.error });
        continue;
      }
      // wasm rows carry no adapter; keep the one a webgpu row reported
      environment = { ...environment, ...result.env, browser: result.browser, adapter: result.env.adapter ?? environment.adapter ?? null };
      for (const { len, batch, firstMs, ms, logit, batchDelta } of result.cases) {
        const sorted = [...ms].sort((a, b) => a - b);
        rows.push({
          model, ep, threads: count, len, batch, loadMs: result.loadMs, firstMs, ms, logit, batchDelta,
          median: quantile(sorted, 0.5), p90: quantile(sorted, 0.9), min: sorted[0],
        });
        console.error(`  ${len} tokens x ${batch}: ${quantile(sorted, 0.5).toFixed(0)} ms, logit ${logit.toFixed(3)}`);
      }
    }
  }
}
server.close();
log.end();

const header = { date: new Date().toISOString(), cpu: os.cpus()[0]?.model, platform: `${os.platform()} ${os.release()}`, args, ...environment };
writeFileSync(path.join(RESULTS, `${stamp}.json`), JSON.stringify({ header, rows }, null, 1));

const adapter = header.adapter;
console.log(`Chrome ${header.browser}, onnxruntime-web ${header.ort}, ${header.cpu}, crossOriginIsolated=${header.crossOriginIsolated}`);
console.log(
  adapter
    ? `WebGPU adapter: ${adapter.vendor} ${adapter.architecture} ${adapter.description} (fallback=${adapter.fallback}, shader-f16=${adapter.shaderF16})`
    : 'WebGPU adapter: none reported',
);
console.log(`${args.runs} warm runs per row, after one "first" run; raw data in bench/results/${stamp}.json\n`);
console.log(report(rows, args.ref));
