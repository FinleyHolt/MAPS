/** The operator console of the capacity study (baselines/human/study_server.py over
 * baselines/human/capacity.py).
 *
 * One console for both conditions. The person always works the earliest open report of
 * their list. With MAPS, a report whose result had landed when it opened is verified
 * (confirm, or pick another); one that had not is searched by hand, and a result landing
 * meanwhile does not interrupt the search. Alone, every report is searched by hand. A
 * skip moves on and counts as a miss. The session runs a fixed time (capacity.SESSION_S).
 *
 * Its screenshot is a panel of the paper's capacity figure, so it wears the paper's look
 * (studyMap.tsx's "paper" theme and the .capacity styles): one sans face, the figure
 * palette, no orange. ``?capture=1`` hides the cursor and every
 * hover state and exposes ``window.__capture()``: the episode, the map's bounds and the
 * on-screen boxes the figure places its callouts on.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { OsmSearch, OsmToggle, pointFeature, ringFeature, useStudyMap, type LonLat, type Ring } from "@/app/studyMap";
import { Trace, type Turn } from "@/app/Trace";

interface Candidate { id: number; label: string; dist_m: number; lonlat: LonLat; ring: Ring }
interface Review {
  instruction: string; ended_by?: string; maps_pick?: number | null; maps_commit?: string; maps_reason?: string;
  turns?: Turn[]; candidates?: Candidate[]; maps_lonlat?: LonLat; maps_prior_ring?: Ring; stops?: LonLat[];
  selected?: { candidate: number | null; lonlat: LonLat; affordable: boolean } | null;
}
interface Row { n: number; episode_id: string; status: string; current: boolean }
interface State {
  arm: "solo" | "team"; elapsed_s: number; remaining_s: number; over: boolean; total: number;
  answered_n: number; skipped_n: number; ready_n: number; list: Row[];
  current: null | { n: number; episode_id: string; instruction: string; mode: "verify" | "search"; point: LonLat | null; review: Review | null };
}

const STATUS: Record<string, string> = {
  not_started: "not started", searching: "MAPS searching", ready: "result ready",
  answered: "answered", skipped: "skipped", open: "open", waiting: "waiting",
};
/** Labelled objects listed before "show more". */
const NEAREST = 5;
const CAPTURE = new URLSearchParams(window.location.search).has("capture");

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail ?? `${path} failed (${r.status})`);
  return (await r.json()) as T;
}

const objectNo = (c: Candidate | undefined) => (c ? (/(\d+)$/.exec(c.label)?.[1] ?? String(c.id)) : "");

export function CapacityConsole() {
  const [st, setSt] = useState<State | null>(null);
  const [fault, setFault] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [moreObjects, setMoreObjects] = useState(false);
  const [moreReason, setMoreReason] = useState(false);
  const [showIds, setShowIds] = useState(false);
  const stRef = useRef<State | null>(null);
  stRef.current = st;

  const act = useCallback(async (body: Record<string, unknown>) => {
    setBusy(true);
    try {
      const out = await postJson<{ ok: boolean; error?: string; state: State }>("capacity/act", body);
      setSt(out.state);
      setFault(out.ok ? null : out.error ?? "refused");
    } catch (e) {
      setFault(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  }, []);

  const m = useStudyMap((at) => {
    const s = stRef.current;
    if (s && !s.over && s.current) void act({ action: "select", lon: at[0], lat: at[1] });
  }, "paper");

  // The list's statuses move on the replay's clock, so the state is asked for every few seconds.
  useEffect(() => {
    const poll = () => fetch("capacity/state", { cache: "no-store" }).then((r) => r.json()).then(setSt).catch(() => undefined);
    void poll();
    const id = window.setInterval(poll, 3000);
    return () => window.clearInterval(id);
  }, []);

  const cur = st?.current ?? null;
  const review = cur?.review ?? null;
  const verifying = cur?.mode === "verify" && review !== null;

  // Each new report opens framed on the briefed AO.
  const shown = useRef<string | null>(null);
  useEffect(() => {
    if (!m.ready || !cur || shown.current === cur.episode_id) return;
    shown.current = cur.episode_id;
    m.clearFound();
    setMoreObjects(false);
    setMoreReason(false);
    m.frameAo(review?.maps_lonlat, 18.5);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cur?.episode_id, m.ready]);

  // MAPS's marks and the person's pick, in the figure's ink.
  useEffect(() => {
    if (!m.ready) return;
    if (!verifying || !review) {
      ["cands", "maps", "stops"].forEach((id) => m.set(id, []));
      m.set("sel", cur?.point ? [pointFeature(cur.point, {})] : []);
      return;
    }
    m.set("cands", (review.candidates ?? []).filter((c) => c.ring && c.id !== review.maps_pick).map((c) => ringFeature(c.ring!, { id: c.id })));
    const pick = (review.candidates ?? []).find((c) => c.id === review.maps_pick);
    m.set("maps", [
      ...(pick?.ring ? [ringFeature(pick.ring, {})] : review.maps_prior_ring ? [ringFeature(review.maps_prior_ring, { area: true })] : []),
      ...(review.maps_lonlat ? [pointFeature(review.maps_lonlat, {})] : []),
    ]);
    const stops = review.stops ?? [];
    m.set("stops", [
      ...(stops.length > 1 ? [{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: stops } } as GeoJSON.Feature] : []),
      ...stops.map((p) => pointFeature(p, {})),
    ]);
    const selRing = review.selected?.candidate != null ? (review.candidates ?? []).find((c) => c.id === review.selected!.candidate)?.ring : null;
    m.set("sel", review.selected ? [...(selRing ? [ringFeature(selRing, {})] : []), pointFeature(review.selected.lonlat, {})] : []);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [st, m.ready, verifying]);

  // Candidate ids are off the imagery unless asked for.
  const idMarkers = useRef<import("maplibre-gl").Marker[]>([]);
  useEffect(() => {
    idMarkers.current.forEach((mk) => mk.remove());
    idMarkers.current = [];
    const map = m.mapRef.current;
    const maplibre = m.maplibreRef.current;
    if (!map || !maplibre || !showIds || !verifying || !review) return;
    for (const c of (review.candidates ?? []).slice(0, 60)) {
      const el = document.createElement("span");
      el.className = "judge-label paper-name";
      el.textContent = objectNo(c);
      idMarkers.current.push(new maplibre.Marker({ element: el, anchor: "left", offset: [6, 0] }).setLngLat(c.lonlat).addTo(map));
    }
  }, [showIds, verifying, review, m.ready, m.mapRef, m.maplibreRef]);

  const sel = review?.selected ?? null;
  const canPick = verifying ? !!sel && sel.affordable : !!cur?.point;
  const pickVerb = verifying ? "replace" : "commit";

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const s = stRef.current;
      if (!s || s.over || !s.current || busy || (e.target as HTMLElement)?.tagName === "INPUT") return;
      if (e.key === "Enter" && s.current.mode === "verify") { e.preventDefault(); void act({ action: "confirm" }); }
      else if ((e.key === "s" || e.key === "S") && canPick) void act({ action: pickVerb });
      else if (e.key === "Escape") { void act({ action: "clear" }); m.setRuler([]); }
      else if (e.key === "o" || e.key === "O") m.setOsmOn((on) => !on);
      else if ((e.key === "m" || e.key === "M") && review?.maps_lonlat) m.flyTo(review.maps_lonlat, 18.5);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [act, busy, canPick, pickVerb, review, m]);

  // The figure's callouts: where each part of the console sits on screen.
  useEffect(() => {
    if (!CAPTURE) return;
    document.documentElement.classList.add("is-capture");
    const box = (sel: string) => {
      const r = document.querySelector(sel)?.getBoundingClientRect();
      return r ? { x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height) } : null;
    };
    (window as unknown as { __capture: () => unknown }).__capture = () => {
      const s = stRef.current;
      const map = m.mapRef.current;
      const canvas = document.querySelector(".judge-canvas")?.getBoundingClientRect();
      const at = s?.current?.review?.maps_lonlat;
      const p = map && at && canvas ? map.project(at) : null;
      const b = map?.getBounds();
      return {
        episode_id: s?.current?.episode_id ?? null, report_n: s?.current?.n ?? null, arm: s?.arm ?? null,
        view_bounds_lonlat: b ? [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()] : null,
        viewport: { w: window.innerWidth, h: window.innerHeight, dpr: window.devicePixelRatio },
        boxes: {
          report: box(".cap-report"), confirm: box(".cap-confirm"), pick_another: box(".cap-pick"),
          status_list: box(".cap-list"), map: box(".judge-canvas"),
          commit_marker: p && canvas ? { x: Math.round(canvas.left + p.x - 12), y: Math.round(canvas.top + p.y - 12), w: 24, h: 24 } : null,
        },
      };
    };
  }, [m.mapRef]);

  const objects = review?.candidates ?? [];
  const pick = objects.find((c) => c.id === review?.maps_pick);

  return (
    <div className="station capacity">
      <header className="cap-bar">
        <span className="cap-title">MAPS · Operator console</span>
        <span className="cap-mission">{cur ? `report ${cur.n}` : st?.over ? "session over" : "loading"}</span>
        <span className="cap-arm">{st?.arm === "team" ? "with MAPS" : st ? "alone" : ""}</span>
        <span className="station-spacer" />
        {st ? <span className="cap-count">{st.answered_n} answered · {st.skipped_n} skipped</span> : null}
        {st ? <span className="cap-clock" title="time left in this session">{clock(st.remaining_s)}</span> : null}
      </header>

      {st?.over ? (
        <div className="judge-done">
          <p className="cap-report-text">The session is over: {st.answered_n} answered, {st.skipped_n} skipped.</p>
          <p className="cap-hint">Every answer is saved. The report open when time ran out is not counted. You can close this page.</p>
        </div>
      ) : (
        <div className="station-body cap-body">
          <aside className="sidebar cap-side">
            <section className="cap-block cap-report">
              <p className="cap-kicker">Report {cur?.n ?? ""}</p>
              <p className="cap-report-text">{cur?.instruction ?? ""}</p>
            </section>

            {/* Everything between the report and the controls scrolls; the controls never move. */}
            <div className="cap-scroll">
            {verifying ? (
              <section className="cap-block">
                <p className="cap-kicker">Why MAPS chose it</p>
                <p className={moreReason ? "cap-reason" : "cap-reason is-clamped"}>{review?.maps_reason ?? ""}</p>
                <p className="cap-hint">
                  <button type="button" className="cap-link" onClick={() => setMoreReason((v) => !v)}>{moreReason ? "less" : "more"}</button>
                </p>
                {moreReason && review ? (
                  <details className="judge-full"><summary>The full run, {review.turns?.length ?? 0} turns</summary><Trace instruction={review.instruction} turns={review.turns ?? []} /></details>
                ) : null}
              </section>
            ) : null}

            {verifying ? (
              <section className="cap-block">
                <p className="cap-kicker">Labelled objects nearest MAPS's commit
                  <button type="button" className="cap-link cap-right" onClick={() => setShowIds((v) => !v)}>{showIds ? "hide ids" : "show ids"}</button></p>
                {(moreObjects ? objects : objects.slice(0, NEAREST)).map((c) => (
                  <button type="button" key={c.id} className="cap-row" data-on={sel?.candidate === c.id} data-maps={c.id === review?.maps_pick}
                          onClick={() => { void act({ action: "candidate", id: c.id }); m.flyTo(c.lonlat); }}>
                    <span>object {objectNo(c)}{c.id === review?.maps_pick ? " · MAPS" : ""}</span><span className="cap-dim">{c.dist_m} m</span>
                  </button>
                ))}
                {objects.length > NEAREST ? (
                  <p className="cap-hint"><button type="button" className="cap-link" onClick={() => setMoreObjects((v) => !v)}>
                    {moreObjects ? "show fewer" : `show ${objects.length - NEAREST} more`}</button></p>
                ) : null}
              </section>
            ) : null}

            <section className="cap-block cap-list">
              <p className="cap-kicker">Your reports{st?.arm === "team" ? `, ${st.ready_n} results ready` : ""}</p>
              <div className="cap-rows">
                {(st?.list ?? []).map((r) => (
                  <div key={r.episode_id} className="cap-listrow" data-current={r.current}>
                    <span className="cap-n">{r.n}</span>
                    <span className={`cap-chip is-${r.current ? "open" : r.status}`}>
                      {r.status === "searching" ? <span className="cap-spin" /> : null}{r.current ? "open" : STATUS[r.status] ?? r.status}
                    </span>
                  </div>
                ))}
              </div>
            </section>
            </div>

            <section className="cap-block cap-dock">
              {verifying ? (
                <>
                  <p className="cap-status"><span className="cap-chip is-ready">result ready</span>
                    MAPS committed to {pick ? `object ${objectNo(pick)}` : review?.maps_commit ?? "a point"}</p>
                  <div className="cap-actions">
                    <button type="button" className="cap-btn cap-confirm" disabled={busy} onClick={() => void act({ action: "confirm" })}>
                      Confirm <kbd>Enter</kbd>
                    </button>
                    <button type="button" className="cap-btn cap-outline cap-pick" disabled={busy || !canPick} onClick={() => void act({ action: "replace" })}>
                      Pick another <kbd>S</kbd>
                    </button>
                  </div>
                  <p className="cap-hint">{sel ? (sel.affordable ? "Your pick is on the map. Pick another commits it." : "MAPS's remaining budget cannot fly there.")
                    : "Confirm MAPS's commit, or click the map where the target is and pick another."}</p>
                </>
              ) : (
                <>
                  <p className="cap-status">
                    {st?.arm === "team" ? <span className="cap-chip is-searching"><span className="cap-spin" />MAPS searching</span> : null}
                    Search by hand
                  </p>
                  <div className="cap-actions">
                    <button type="button" className="cap-btn cap-confirm" disabled={busy || !canPick} onClick={() => void act({ action: "commit" })}>
                      Commit <kbd>S</kbd>
                    </button>
                  </div>
                  <p className="cap-hint">{cur?.point ? "Your point is on the map. Commit it, or click again to move it." : "Click the map where the target is, then commit."}</p>
                </>
              )}
              <p className="cap-hint">
                <button type="button" className="cap-link" disabled={busy} onClick={() => void act({ action: "skip" })}>skip this report</button>
                {" "}(counts as a miss)
                {verifying ? <> · <button type="button" className="cap-link" onClick={() => review?.maps_lonlat && m.flyTo(review.maps_lonlat, 18.5)}>fly to MAPS's commit <kbd>M</kbd></button></> : null}
              </p>
              {fault ? <p className="judge-fault">{fault}</p> : null}
            </section>

          </aside>

          <div className="instrument judge-map">
            <div className="judge-canvas-wrap">
              <div ref={m.containerRef} className="judge-canvas" />
              <OsmToggle on={m.osmOn} onChange={m.setOsmOn} hover={CAPTURE ? null : m.hover} rulerM={m.rulerM} onClearRuler={() => m.setRuler([])}
                         lit={m.lit} pieces={m.lit ? m.piecesOf(m.lit) : 0} litPiece={m.litPiece} onStep={m.stepPiece} />
              <OsmSearch key={cur?.episode_id ?? ""} names={m.names} lit={m.lit} onPick={(n) => m.showName(n.name)} onCross={m.markSpots}
                         crossingsOf={m.crossingsOf} approachOf={m.approachOf} kindsNear={m.kindsNear} piecesOf={m.piecesOf} nearOf={m.nearOf} />
            </div>
            <p className="cap-legend">
              <span className="sw cap-sw-maps" /> MAPS's commit <span className="sw cap-sw-cand" /> labelled objects
              <span className="sw cap-sw-pick" /> your pick · north up · O toggles OSM · shift-click measures
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
