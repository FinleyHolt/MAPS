/** What the person does between opening a report and answering it, for the study's record
 * (``capacity.Capacity.log``): the panels they open, their searches, the street map, the ruler,
 * the view, and the page going out of sight. Queued here and posted every few seconds, and at
 * once when the page is hidden. Each carries the report open when it happened; the server adds
 * that report's place in the list and the session's clock when the event lands.
 * Every act the server takes (a point, a commit, a confirm, a refusal) it logs itself. */

export interface StudyEvent { kind: string; t: number; [k: string]: unknown }

const queue: StudyEvent[] = [];
let report: string | null = null;

/** The report open now: every event logged until the next call carries it. */
export function setReport(episodeId: string | null): void {
  report = episodeId;
}

/** One event, stamped with the browser's clock, seconds. */
export function logEvent(kind: string, detail: Record<string, unknown> = {}): void {
  queue.push({ kind, t: Date.now() / 1000, ...(report ? { episode_id: report } : {}), ...detail });
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
