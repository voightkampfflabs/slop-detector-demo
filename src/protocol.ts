/** Messages between the page (main.ts) and the inference worker (worker.ts). */

export interface Bands {
  /** Where the training target's human band ends. Not shown: human texts often score above it. */
  human_max: number;
  /** At or above this, flag the text as not human (1% false positives). The page's human edge. */
  flag_min: number;
  /** At or above this the text counts as AI-generated. */
  generated_min: number;
}

export interface ScoreResult {
  /** Tokens in the whole text, special tokens not counted. */
  tokens: number;
  /** The model's minimum and maximum input, in the same unit. */
  minTokens: number;
  maxTokens: number;
  /** Null when the text is under minTokens and the model was not run. */
  score: number | null;
  /** Time spent in the model alone, in milliseconds. Null when it was not run. */
  inferenceMs: number | null;
}

export type ToWorker = { type: 'score'; id: number; text: string };

export type FromWorker =
  | { type: 'progress'; loaded: number; total: number; cached: boolean }
  | { type: 'compiling' }
  | { type: 'ready'; bands: Bands; threads: number }
  | { type: 'result'; id: number; result: ScoreResult }
  | { type: 'error'; message: string };
