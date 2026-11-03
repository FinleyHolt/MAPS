/** What the person does between opening a report and answering it, for the study's record
 * (``capacity.Capacity.log``): the panels they open, their searches, the street map, the ruler,
 * the view, and the page going out of sight. Queued here and posted every few seconds, and at
 * once when the page is hidden; the server stamps each with the open report and its own clock.
 * Every act the server takes (a point, a commit, a confirm, a refusal) it logs itself. */

export interface StudyEvent { kind: string; t: number; [k: string]: unknown }

const queue: StudyEvent[] = [];

/** One event, stamped with the browser's clock, seconds. */
export function logEvent(kind: string, detail: Record<string, unknown> = {}): void {
  queue.push({ kind, t: Date.now() / 1000, ...detail });
}

/** Post what is queued; a post that fails puts its events back for the next one. */
export function flushEvents(): void {
  if (!queue.length) return;
  const events = queue.splice(0, queue.length);
  fetch("capacity/events", { method: "POST", headers: { "Content-Type": "application/json" },
                             body: JSON.stringify({ events }), keepalive: true })
    .then((r) => { if (!r.ok && r.status !== 409) queue.unshift(...events); })
    .catch(() => queue.unshift(...events));
}
