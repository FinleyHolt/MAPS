/** What the capacity console shows of a session's state, and what it sends back. */

export interface SessionView {
  arm: "solo" | "team"; practice?: boolean; remaining_s: number;
  answered_n: number; skipped_n: number; current: null | { episode_id: string };
}

export const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/** An act names the report it answers, so a stale tab cannot answer the next one. */
export const actBody = (st: Pick<SessionView, "current"> | null, body: Record<string, unknown>) =>
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

/** Which session this is, and its time left; practice runs no clock. */
export function header(st: Pick<SessionView, "arm" | "practice" | "remaining_s">): { session: string; clock: string | null } {
  if (st.practice) return { session: "Practice", clock: null };
  return { session: st.arm === "team" ? "Session: with MAPS" : "Session: alone", clock: `${clock(st.remaining_s)} left` };
}

export function overLines(st: Pick<SessionView, "practice" | "answered_n" | "skipped_n">): string[] {
  if (st.practice) return ["Practice finished. Press Continue to go back to the study page."];
  return [`The session is over: ${st.answered_n} answered, ${st.skipped_n} skipped.`,
          "Every answer is saved. The report open when time ran out is not counted."];
}

/** The session's state, or null when the study did not answer with one. */
export async function pollState<T>(get: () => Promise<{ ok: boolean; json: () => Promise<unknown> }>): Promise<T | null> {
  try {
    const r = await get();
    return r.ok ? ((await r.json()) as T) : null;
  } catch {
    return null;
  }
}
