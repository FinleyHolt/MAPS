/** What the capacity console shows of a session's state, and what it sends back. */

export interface SessionView { current: null | { episode_id: string } }

/** An act names the report it answers, so a stale tab cannot answer the next one. */
export const actBody = (st: SessionView | null, body: Record<string, unknown>) =>
  st?.current ? { ...body, episode_id: st.current.episode_id } : body;

export type KeyCommand = "confirm" | "pick" | "clear" | "street" | "maps";

/** The console's keys; a held key acts once. */
export function keyCommand(e: { key: string; repeat: boolean }, mode: "verify" | "search", canPick: boolean): KeyCommand | null {
  if (e.repeat) return null;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (k === "Enter") return mode === "verify" ? "confirm" : null;
  if (k === "s") return canPick ? "pick" : null;
  return ({ Escape: "clear", o: "street", m: "maps" } as Record<string, KeyCommand>)[k] ?? null;
}
