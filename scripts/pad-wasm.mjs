// Writes a copy of ONNX Runtime's WebGPU (Asyncify) binary that Safari can run.
//
// WebKit's optimising WebAssembly compiler (OMG) can take gigabytes while it compiles
// the mid-sized functions of an Asyncify build, and iOS then kills the tab
// (https://bugs.webkit.org/show_bug.cgi?id=304810). WebKit leaves a function above a
// size limit to its baseline compiler, so every function body of 40 to 160 KB is padded
// to 160 KB with nops that a branch jumps over. No added instruction runs, and the nops
// compress to nothing. Bodies above 160 KB are over the limit already.
//
// Run by `npm run dev` and `npm run build`; the output is not committed.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/** Bodies from this size up were the ones that blew up in the WebKit bug's test case. */
const MIN_BODY = 40_000;
/** The body size that test case padded to, above WebKit's maximumOMGCandidateCost. */
const PADDED_BODY = 160_000;

const CODE_SECTION = 10;
const BLOCK = 0x02;
const EMPTY_TYPE = 0x40;
const BR = 0x0c;
const NOP = 0x01;
const END = 0x0b;

/** Reads an unsigned LEB128 number: its value and the offset after it. */
function readLeb(bytes, at) {
  let value = 0;
  let shift = 0;
  for (;;) {
    const byte = bytes[at++];
    value += (byte & 0x7f) * 2 ** shift;
    shift += 7;
    if (!(byte & 0x80)) return [value, at];
  }
}

function leb(value) {
  const out = [];
  do {
    const byte = value % 128;
    value = Math.floor(value / 128);
    out.push(value ? byte | 0x80 : byte);
  } while (value);
  return Uint8Array.from(out);
}

/** Offset of the first instruction of a function body, after its local declarations. */
function instructionsStart(body) {
  let [groups, at] = readLeb(body, 0);
  while (groups--) {
    [, at] = readLeb(body, at); // how many locals of this type
    const type = body[at++];
    // (ref null? <heap type>) carries a heap type; every other value type is one byte.
    if (type === 0x63 || type === 0x64) [, at] = readLeb(body, at);
  }
  return at;
}

/** The body with `block; br 0; nop…; end` in front of its instructions, PADDED_BODY long. */
function pad(body) {
  const start = instructionsStart(body);
  const nops = PADDED_BODY - body.length - 5; // block, its type, br, its label, end
  const out = new Uint8Array(PADDED_BODY);
  out.set(body.subarray(0, start));
  out.set([BLOCK, EMPTY_TYPE, BR, 0], start);
  out.fill(NOP, start + 4, start + 4 + nops);
  out[start + 4 + nops] = END;
  out.set(body.subarray(start), start + 5 + nops);
  return out;
}

/** The code section's contents with the mid-sized bodies padded, and how many were. */
function padCode(code) {
  const [count, first] = readLeb(code, 0);
  const parts = [code.subarray(0, first)];
  let padded = 0;
  let at = first;
  for (let i = 0; i < count; i++) {
    const [size, start] = readLeb(code, at);
    let body = code.subarray(start, start + size);
    if (size >= MIN_BODY && size < PADDED_BODY) {
      body = pad(body);
      padded++;
    }
    parts.push(leb(body.length), body);
    at = start + size;
  }
  return [concat(parts), padded];
}

function concat(parts) {
  const out = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.length;
  }
  return out;
}

/** The module with its code section padded. Every other section is copied as it is. */
function padModule(wasm) {
  const parts = [wasm.subarray(0, 8)]; // magic and version
  let padded = 0;
  let at = 8;
  while (at < wasm.length) {
    const id = wasm[at];
    const [size, start] = readLeb(wasm, at + 1);
    const end = start + size;
    if (id === CODE_SECTION) {
      const [code, count] = padCode(wasm.subarray(start, end));
      parts.push(Uint8Array.of(id), leb(code.length), code);
      padded += count;
    } else {
      parts.push(wasm.subarray(at, end));
    }
    at = end;
  }
  return [concat(parts), padded];
}

const require = createRequire(import.meta.url);
const source = require.resolve('onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm');
const target = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  '../src/generated/ort-wasm-simd-threaded.asyncify.padded.wasm',
);

const original = readFileSync(source);
const [result, padded] = padModule(original);
if (!padded) throw new Error(`${source}: no function body of ${MIN_BODY} to ${PADDED_BODY} bytes`);
// Where this Node can validate the original, the padded module has to validate too.
if (WebAssembly.validate(original) && !WebAssembly.validate(result)) {
  throw new Error('the padded module is not valid WebAssembly');
}

mkdirSync(path.dirname(target), { recursive: true });
writeFileSync(target, result);
const megabytes = (bytes) => (bytes / 1e6).toFixed(1);
console.log(
  `pad-wasm: ${padded} function bodies padded to ${PADDED_BODY} bytes,`,
  `${megabytes(original.length)} MB -> ${megabytes(result.length)} MB`,
);
