/** MAPS's final reply, the explanation of its answer (CapacityConsole.tsx), read as the light
 * markdown MAPS writes: paragraphs, bullet lines, **bold**, *emphasis* and `code`. Text only:
 * the console renders each span as text, so nothing in a reply becomes markup. */

export interface Span { text: string; bold?: boolean; em?: boolean; code?: boolean }
export interface Block { kind: "p" | "li"; spans: Span[] }

const BULLET = /^\s*[-*•]\s+/;

/** The reply's paragraphs and bullet lines; a line under a bullet, indented, continues it. */
export function replyBlocks(text: string): Block[] {
  const raw: { kind: "p" | "li"; text: string }[] = [];
  let open: { kind: "p" | "li"; text: string } | null = null;
  for (const line of text.split("\n")) {
    if (!line.trim()) open = null;
    else if (BULLET.test(line)) raw.push((open = { kind: "li", text: line.replace(BULLET, "") }));
    else if (open && (open.kind === "p" || /^\s/.test(line))) open.text += "\n" + line.trim();
    else raw.push((open = { kind: "p", text: line.trim() }));
  }
  return raw.map((b) => ({ kind: b.kind, spans: spans(b.text) }));
}

/** The text in runs of one style; a marker with no partner reads as itself. */
function spans(text: string): Span[] {
  const out: Span[] = [];
  let bold = false;
  let em = false;
  let buf = "";
  const push = (s: string, code = false) => {
    const last = out.at(-1);
    if (!code && last && !last.code && !!last.bold === bold && !!last.em === em) { last.text += s; return; }
    out.push({ text: s, ...(bold ? { bold } : {}), ...(em ? { em } : {}), ...(code ? { code } : {}) });
  };
  const flush = () => { if (buf) push(buf); buf = ""; };
  for (let i = 0; i < text.length;) {
    if (text[i] === "`") {
      const end = text.indexOf("`", i + 1);
      if (end > i + 1) { flush(); push(text.slice(i + 1, end), true); i = end + 1; continue; }
    } else if (text.startsWith("**", i) && (bold || text.indexOf("**", i + 2) > i + 2)) {
      flush(); bold = !bold; i += 2; continue;
    } else if (text[i] === "*" && (em ? /\S/.test(text[i - 1] ?? "") : /\S/.test(text[i + 1] ?? "") && /\S\*/.test(text.slice(i + 1)))) {
      flush(); em = !em; i += 1; continue;
    }
    buf += text[i];
    i += 1;
  }
  flush();
  return out;
}

/** The step whose code ran ``call`` (a cited tool call, ``look(r1)``), quotes and spacing aside;
 * null for a plain word or a call no step ran. */
export function stepOfCall(call: string, said: string[]): number | null {
  if (!/\w\s*\(/.test(call)) return null;
  const norm = (s: string) => s.replace(/\s+/g, "").replace(/'/g, '"');
  const i = said.findIndex((s) => norm(s).includes(norm(call)));
  return i < 0 ? null : i;
}
