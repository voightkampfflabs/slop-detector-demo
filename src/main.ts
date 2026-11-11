import '@fontsource-variable/source-serif-4/wght.css';
import '@fontsource-variable/source-serif-4/wght-italic.css';
import '@fontsource-variable/source-sans-3/wght.css';
import './style.css';
import type { Bands, FromWorker, ScoreResult, ToWorker } from './protocol';

/** Pause after the last keystroke before the text is sent to the model. */
const DEBOUNCE_MS = 250;
const RELOAD_FLAG = 'coi-reloaded';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const textarea = $<HTMLTextAreaElement>('text');
const status = $('status');
const load = $('load');
const loadFill = $('load-fill');
const gauge = $('gauge');
const scoreEl = $('score');
const verdictEl = $('verdict');
const ruler = $('ruler');
const fine = $('fine');
const statTokens = $('stat-tokens');
const statLatency = $('stat-latency');

const FINE_IDLE = fine.textContent ?? '';

/**
 * GitHub Pages cannot send the headers that make a page cross-origin isolated,
 * so a service worker adds them. It only takes effect on the next navigation:
 * register it, then reload once.
 */
async function ensureIsolation(): Promise<void> {
  if (crossOriginIsolated || !('serviceWorker' in navigator)) return;
  try {
    await navigator.serviceWorker.register('./coi-sw.js');
    await navigator.serviceWorker.ready;
    if (sessionStorage.getItem(RELOAD_FLAG)) return; // Already tried; run on one thread.
    sessionStorage.setItem(RELOAD_FLAG, '1');
    location.reload();
    await new Promise(() => {}); // Hold everything until the reload happens.
  } catch {
    // No service worker (private window, blocked storage): one thread it is.
  }
}

// --- Ruler -----------------------------------------------------------------

function place(className: string, at: number, attrs: Record<string, string> = {}): HTMLElement {
  const el = document.createElement('div');
  el.className = className;
  el.style.left = `${at * 100}%`;
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  ruler.append(el);
  return el;
}

function drawTicks(): void {
  for (let i = 0; i <= 20; i++) {
    place('tick', i / 20, i % 2 === 0 ? { 'data-major': '' } : {});
  }
  place('tick-label', 0, { 'data-edge': 'start' }).textContent = '0';
  place('tick-label', 0.5).textContent = '0.5';
  place('tick-label', 1, { 'data-edge': 'end' }).textContent = '1';
}

interface Band {
  name: string;
  from: number;
  to: number;
  el: HTMLElement;
}
let bandList: Band[] = [];
let bands: Bands | null = null;

function drawBands(b: Bands): void {
  bands = b;
  place('cut', b.human_max);
  place('cut', b.generated_min);
  place('cut', b.flag_min, { 'data-kind': 'flag' });

  const spans: Array<[string, number, number]> = [
    ['human', 0, b.human_max],
    ['AI-edited', b.human_max, b.generated_min],
    ['AI-generated', b.generated_min, 1],
  ];
  bandList = spans.map(([name, from, to]) => {
    const el = place('band', from);
    el.style.width = `${(to - from) * 100}%`;
    el.textContent = name;
    return { name, from, to, el };
  });
  // The first band starts at the ruler's edge and needs no inset.
  bandList[0].el.style.paddingLeft = '0';
}

// --- Rendering -------------------------------------------------------------

const megabytes = (bytes: number) => Math.round(bytes / 1e6);

function showProgress(loaded: number, total: number, cached: boolean): void {
  const fraction = total ? loaded / total : 0;
  loadFill.style.transform = `scaleX(${fraction})`;
  load.setAttribute('aria-valuenow', String(Math.round(fraction * 100)));
  status.textContent = cached
    ? 'reading the model from cache'
    : `${megabytes(loaded)} / ${megabytes(total)} MB`;
}

function showError(message: string): void {
  status.dataset.error = '';
  status.textContent = `failed: ${message}`;
  delete gauge.dataset.busy;
}

const DASH = '–';

/** Empties the score side of the gauge; the stats are set by the caller. */
function clearScore(state: 'idle' | 'short'): void {
  gauge.dataset.state = state;
  gauge.style.setProperty('--score', '0');
  scoreEl.textContent = DASH;
  verdictEl.textContent = '';
  for (const band of bandList) delete band.el.dataset.active;
}

function renderIdle(): void {
  clearScore('idle');
  delete gauge.dataset.busy;
  fine.textContent = FINE_IDLE;
  statTokens.textContent = DASH;
  statLatency.textContent = DASH;
}

/** A number followed by a small unit, e.g. "840 ms". */
function setStat(el: HTMLElement, value: string, unit = ''): void {
  el.textContent = value;
  if (!unit) return;
  const small = document.createElement('small');
  small.textContent = ` ${unit}`;
  el.append(small);
}

function renderResult(result: ScoreResult): void {
  const { score, tokens, minTokens, maxTokens } = result;
  setStat(statTokens, tokens.toLocaleString('en-US'));

  if (score === null || result.inferenceMs === null) {
    clearScore('short');
    statLatency.textContent = DASH;
    fine.textContent = `Not scored: the model needs at least ${minTokens} tokens, and this text has ${tokens}.`;
    return;
  }

  setStat(statLatency, Math.round(result.inferenceMs).toLocaleString('en-US'), 'ms');
  gauge.dataset.state = 'scored';
  gauge.style.setProperty('--score', score.toFixed(4));
  scoreEl.textContent = score.toFixed(2);

  const active = bandList.find((band) => score < band.to) ?? bandList[bandList.length - 1];
  for (const band of bandList) {
    if (band === active) band.el.dataset.active = '';
    else delete band.el.dataset.active;
  }
  verdictEl.textContent = active?.name ?? '';

  const notes: string[] = [];
  if (bands) {
    notes.push(
      score >= bands.flag_min
        ? `At or above ${bands.flag_min.toFixed(2)}: flagged as not human.`
        : `Below ${bands.flag_min.toFixed(2)}: not flagged.`,
    );
  }
  if (tokens > maxTokens) notes.push(`Only the first ${maxTokens} tokens were scored.`);
  fine.textContent = notes.join(' ');
}

// --- Scoring loop ----------------------------------------------------------

function start(): void {
  const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  const send = (message: ToWorker) => worker.postMessage(message);

  let ready = false;
  let timer: number | undefined;
  let lastId = 0;
  /** Text of the request the worker is running now, if any. */
  let inFlight: string | null = null;
  /** Newest text typed while the worker was busy. Older ones are dropped. */
  let waiting: string | null = null;
  let lastScored = '';

  function submit(text: string): void {
    if (inFlight !== null) {
      waiting = text;
      return;
    }
    if (text === lastScored) return;
    inFlight = text;
    gauge.dataset.busy = '';
    send({ type: 'score', id: ++lastId, text });
  }

  function onInput(): void {
    window.clearTimeout(timer);
    const text = textarea.value;
    if (!text.trim()) {
      // Bumping the id makes any result still in flight stale.
      lastId++;
      waiting = null;
      lastScored = '';
      renderIdle();
      return;
    }
    timer = window.setTimeout(() => submit(text), DEBOUNCE_MS);
  }

  worker.onmessage = (event: MessageEvent<FromWorker>) => {
    const message = event.data;
    switch (message.type) {
      case 'progress':
        showProgress(message.loaded, message.total, message.cached);
        break;

      case 'compiling':
        status.textContent = 'preparing the model';
        break;

      case 'ready':
        ready = true;
        drawBands(message.bands);
        load.dataset.done = '';
        status.textContent =
          message.threads > 1 ? `ready · ${message.threads} threads` : 'ready · 1 thread';
        textarea.disabled = false;
        textarea.placeholder = 'Type or paste some English text.';
        textarea.focus();
        break;

      case 'result': {
        const text = inFlight ?? '';
        inFlight = null;
        if (message.id === lastId) {
          lastScored = text;
          renderResult(message.result);
        }
        if (waiting !== null) {
          const next = waiting;
          waiting = null;
          submit(next);
        }
        if (inFlight === null) delete gauge.dataset.busy;
        break;
      }

      case 'error':
        inFlight = null;
        waiting = null;
        showError(message.message);
        if (!ready) textarea.placeholder = 'The model could not be loaded.';
        break;
    }
  };

  worker.onerror = (event) => showError(event.message || 'the worker crashed');
  textarea.addEventListener('input', onInput);
}

drawTicks();
ensureIsolation().then(start);
