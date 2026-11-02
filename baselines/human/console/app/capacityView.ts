/** What the capacity console shows of a session's state, and what it sends back. */

export interface SessionView {
  arm: "solo" | "team"; practice?: boolean; remaining_s: number;
  answered_n: number; skipped_n: number; current: null | { episode_id: string };
}

export const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/** An act names the report it answers, so a stale tab cannot answer the next one; with no
 * report open there is nothing to act on (null). */
export const actBody = (st: Pick<SessionView, "current"> | null, body: Record<string, unknown>) =>
  st?.current ? { ...body, episode_id: st.current.episode_id } : null;

/** The line under Confirm and Pick another: the study's refusal of the person's point (null or
 * absent while MAPS could reach it), or what to do next. */
export function verifyHint(sel: { refusal?: string | null } | null): string {
  if (!sel) return "Press Confirm if MAPS's answer is right. If not, click the right place, then press Pick another.";
  return sel.refusal ?? "Your point is on the map. Press Pick another to answer with it.";
}

export type KeyCommand = "confirm" | "pick" | "clear" | "street" | "maps" | "back";

/** The console's keys; a held key acts once. */
export function keyCommand(e: { key: string; repeat: boolean }, mode: "verify" | "search", canPick: boolean): KeyCommand | null {
  if (e.repeat) return null;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (k === "Enter") return mode === "verify" ? "confirm" : null;
  if (k === "s") return canPick ? "pick" : null;
  return ({ Escape: "clear", o: "street", m: "maps", b: "back" } as Record<string, KeyCommand>)[k] ?? null;
}

/** The map takes clicks from a second after it is ready (``readyAt``, 0 while it is not):
 * the second click of a double-clicked start button lands on the map. */
export const mapTakesClicks = (readyAt: number, now: number) => readyAt > 0 && now - readyAt >= 1000;

/** Which session this is, and its time left ``age_s`` after the study sent ``st``: the clock runs
 * on between polls and while contact is lost. Practice runs no clock. */
export function header(st: Pick<SessionView, "arm" | "practice" | "remaining_s">, age_s = 0): { session: string; clock: string | null } {
  if (st.practice) return { session: "Practice", clock: null };
  return { session: st.arm === "team" ? "Session: with MAPS" : "Session: alone", clock: `${clock(Math.max(0, st.remaining_s - age_s))} left` };
}

/** A mapped object's number: its OSM label's, else its id. */
export const objectNo = (c: { id: number; label: string }) => /(\d+)$/.exec(c.label)?.[1] ?? String(c.id);

/** A mapped object as the console names it: its kind and number where the study gives a kind
 * ("house 568"), else "object" and its number. */
export const objectName = (c: { id: number; label: string; kind?: string | null }) => `${c.kind || "object"} ${objectNo(c)}`;

/** MAPS's answer as the console names it: what kind of thing it is, without the label's number. */
export const aKind = (kind: string) => `${/^[aeiou]/i.test(kind) ? "an" : "a"} ${kind}`;

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

/** How the console names an earlier answer's source, on a report opened again. */
export const EARLIER: Record<string, string> = {
  confirm: "you confirmed MAPS's answer", replace: "you picked another place", own_search: "you placed a point", skip: "you skipped it",
};
