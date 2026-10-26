/** The OSM prior's places a stretch of MAPS's reasoning names (CapacityConsole.tsx): the text
 * in pieces, each named place its own piece, so the console can link it to the map. */

/** Plain text, or a place: the name as the prior spells it, and the text as MAPS wrote it. */
export type Piece = string | { name: string; text: string };

/** Names shorter than this read as words more often than as places. */
const MIN_NAME = 4;

/** A splitter over ``names``: whole words in any case, the longest name first, so "Lime Avenue"
 * is one place and not "Lime" and a word. */
export function placeSplitter(names: string[]): (text: string) => Piece[] {
  const spelled = new Map<string, string>();
  for (const n of names) if (n.length >= MIN_NAME && !spelled.has(n.toLowerCase())) spelled.set(n.toLowerCase(), n);
  if (!spelled.size) return (text) => [text];
  const alternatives = [...spelled.values()].sort((a, b) => b.length - a.length)
    .map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
  const re = new RegExp(`(?<![\\p{L}\\p{N}])(?:${alternatives})(?![\\p{L}\\p{N}])`, "giu");
  return (text) => {
    const out: Piece[] = [];
    let at = 0;
    for (const m of text.matchAll(re)) {
      if (m.index! > at) out.push(text.slice(at, m.index));
      out.push({ name: spelled.get(m[0].toLowerCase())!, text: m[0] });
      at = m.index! + m[0].length;
    }
    if (at < text.length) out.push(text.slice(at));
    return out;
  };
}
