/**
 * The text clean-up the model was trained behind, ported from slopdetect's
 * `markdown.py`. Every training text had its typography turned to ASCII and its
 * fenced code blocks emptied, so the same has to happen before tokenizing here.
 */

const URL_CHARS = /[^\s<>\[\]`]/.source;
/** Balanced parentheses are allowed, as in wiki URLs. */
const HREF = /(?:[^()\s]|\([^()\s]*\))*/.source;

/**
 * Code, math and link spans: `SPAN_RE` of markdown.py. A fence runs to its
 * closing line or, when unclosed, to the end of the text.
 */
const SPAN = new RegExp(
  [
    /(?<code>^(?<fence>`{3,}|~{3,}).*?(?:^\k<fence>[ \t]*$|$(?![\s\S]))|``[^\n]*?``|`[^`\n]*`)/.source,
    /(?<math>\$\$.*?\$\$|\$ ?(?!\s)[^$\n]*?(?<!\s) ?\$(?!\d)|\\[A-Za-z]+)/.source,
    /<(?<auto>https?:\/\/[^\s<>]+)>/.source,
    `\\[(?<text>[^\\]\\n]*)\\]\\((?<href>https?://${HREF})\\)`,
    `(?<bare>https?://${URL_CHARS}+)`,
  ].join('|'),
  'gms',
);

const TYPOGRAPHY: Array<[RegExp, string]> = [
  [/[‘’‚]/g, "'"],
  [/[“”„]/g, '"'],
  [/…/g, '...'],
  [/[ \t]*—[ \t]*| – /g, ' - '],
];

function typography(text: string): string {
  for (const [pattern, replacement] of TYPOGRAPHY) text = text.replace(pattern, replacement);
  return text;
}

/** Curly quotes, the ellipsis and the em dash as ASCII, outside code, math and links. */
export function asciiTypography(text: string): string {
  let out = '';
  let pos = 0;
  for (const match of text.matchAll(SPAN)) {
    out += typography(text.slice(pos, match.index)) + match[0];
    pos = match.index + match[0].length;
  }
  return out + typography(text.slice(pos));
}

/** Each fenced code block emptied to a bare ``` pair; inline code stays. */
export function emptyCodeBlocks(text: string): string {
  return text.replace(SPAN, (...args) => {
    const groups = args[args.length - 1] as Record<string, string | undefined>;
    return groups.fence ? '```\n```' : (args[0] as string);
  });
}

/** What the tokenizer gets: the same two steps as the Python scorer. */
export function modelInput(text: string): string {
  return emptyCodeBlocks(asciiTypography(text));
}
