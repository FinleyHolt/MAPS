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
 * palette, no orange. The street map's places MAPS's reasoning names are links: each is drawn
 * on the map in its own colour, pointing at the name lights it, clicking frames it.
 * ``?capture=1`` hides the cursor, every hover state and those places, and exposes
 * ``window.__capture()``: the episode, the map's bounds and the on-screen boxes the figure
 * places its callouts on.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";

import { placeSplitter } from "@/app/placeNames";
import { OsmSearch, OsmToggle, pointFeature, ringFeature, SAID, useStudyMap, type LonLat, type Ring } from "@/app/studyMap";
import { Trace, type Turn } from "@/app/Trace";
import { actBody, header, keyCommand, mapTakesClicks, objectName, objectNo, overLines, pollState, verifyHint } from "@/app/capacityView";

interface Candidate { id: number; label: string; kind?: string | null; dist_m: number; lonlat: LonLat; ring: Ring }
interface Review {
  instruction: string; ended_by?: string; maps_pick?: number | null; maps_reason?: string;
  turns?: Turn[]; candidates?: Candidate[]; maps_lonlat?: LonLat; maps_prior_ring?: Ring; stops?: LonLat[];
  selected?: { candidate: number | null; lonlat: LonLat; affordable: boolean; refusal?: string | null } | null;
}
interface Row { n: number; episode_id: string; status: string; current: boolean }
interface State {
  arm: "solo" | "team"; practice?: boolean; elapsed_s: number; remaining_s: number; over: boolean; total: number;
  answered_n: number; skipped_n: number; list: Row[];
  current: null | { n: number; episode_id: string; instruction: string; mode: "verify" | "search"; point: LonLat | null; review: Review | null };
}

const STATUS: Record<string, string> = {
  not_started: "not started", searching: "MAPS searching", ready: "result ready",
  answered: "answered", skipped: "skipped", open: "open", waiting: "waiting",
};
/** Mapped objects listed before "show more". */
const NEAREST = 5;
/** Where MAPS's answer opens, and where M goes: a few houses and the street they front. */
const MAPS_ZOOM = 18.5;
const CAPTURE = new URLSearchParams(window.location.search).has("capture");

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail ?? `${path} failed (${r.status})`);
  return (await r.json()) as T;
}


export function CapacityConsole() {
  const [st, setState] = useState<State | null>(null);
  // When the study sent ``st``, and a second's tick: the clock runs on between polls.
  const [stAt, setStAt] = useState(0);
  const [now, setNow] = useState(0);
  const setSt = useCallback((s: State) => { setState(s); setStAt(performance.now()); }, []);
  const [fault, setFault] = useState<string | null>(null);
  const [lost, setLost] = useState(false);
  const [busy, setBusy] = useState(false);
  const [moreObjects, setMoreObjects] = useState(false);
  const [moreReason, setMoreReason] = useState(false);
  const [stepsOpen, setStepsOpen] = useState(false);
  const [showIds, setShowIds] = useState(false);
  const [askSkip, setAskSkip] = useState(false);
  const [hot, setHot] = useState<string | null>(null);
  const readyAt = useRef(0);
  const stRef = useRef<State | null>(null);
  stRef.current = st;

  const act = useCallback(async (body: Record<string, unknown>) => {
    const sent = actBody(stRef.current, body);
    if (!sent) return;
    setBusy(true);
    try {
      const out = await postJson<{ ok: boolean; error?: string; state: State }>("capacity/act", sent);
      setSt(out.state);
      setFault(out.ok ? null : out.error ?? "refused");
    } catch (e) {
      setFault(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  }, [setSt]);

  const m = useStudyMap((at) => {
    const s = stRef.current;
    if (s && !s.over && s.current && mapTakesClicks(readyAt.current, performance.now())) void act({ action: "select", lon: at[0], lat: at[1] });
  });
  useEffect(() => { if (m.ready) readyAt.current = performance.now(); }, [m.ready]);

  // The list's statuses move on the replay's clock, so the state is asked for every few seconds;
  // a poll the study does not answer says so until one it does.
  useEffect(() => {
    const poll = () => pollState<State>(() => fetch("capacity/state", { cache: "no-store" })).then((s) => {
      setLost(s === null);
      if (s) setSt(s);
    });
    void poll();
    const id = window.setInterval(poll, 3000);
    const tick = window.setInterval(() => setNow(performance.now()), 1000);
    return () => { window.clearInterval(id); window.clearInterval(tick); };
  }, [setSt]);

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
    setStepsOpen(false);
    setAskSkip(false);
    setHot(null);
    if (verifying && review?.maps_lonlat) m.flyTo(review.maps_lonlat, MAPS_ZOOM);
    else m.frameAo(review?.maps_lonlat, MAPS_ZOOM);
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

  // The places MAPS's reasoning names, in the order it names them, one colour each: those of
  // "Why MAPS chose it", and those of the whole search's steps while they are open.
  const split = useMemo(() => placeSplitter(CAPTURE ? [] : m.names.map((n) => n.name)), [m.names]);
  const said = verifying && review ? [...new Set([review.maps_reason ?? "", ...(stepsOpen ? (review.turns ?? []).map((t) => t.reasoning ?? "") : [])]
    .flatMap(split).flatMap((p) => (typeof p === "string" ? [] : [p.name])))] : [];
  const colour = (name: string) => SAID[said.indexOf(name) % SAID.length];
  useEffect(() => {
    if (m.ready) m.setSaid(said.map((name) => ({ name, colour: colour(name) })), hot);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [said.join("|"), hot, m.ready]);
  const place = (name: string, text: string, key: number | string) => (
    <button type="button" key={key} className="cap-place" style={{ "--place": colour(name) } as CSSProperties}
            onMouseEnter={() => setHot(name)} onMouseLeave={() => setHot(null)} onFocus={() => setHot(name)} onBlur={() => setHot(null)}
            onClick={() => m.frameName(name)}>{text}</button>
  );
  const linked = (text: string): ReactNode => split(text).map((p, i) => (typeof p === "string" ? p : place(p.name, p.text, i)));

  // Object numbers are off the imagery unless asked for.
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
      const k = keyCommand(e, s.current.mode, canPick);
      if (k === "confirm") { e.preventDefault(); void act({ action: "confirm" }); }
      else if (k === "pick") void act({ action: pickVerb });
      else if (k === "clear") { void act({ action: "clear" }); m.setRuler([]); }
      else if (k === "street") m.setOsmOn((on) => !on);
      else if (k === "maps" && review?.maps_lonlat) m.flyTo(review.maps_lonlat, MAPS_ZOOM);
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

  const head = st ? header(st, stAt && now > stAt ? (now - stAt) / 1000 : 0) : null;
  const objects = review?.candidates ?? [];
  const pick = objects.find((c) => c.id === review?.maps_pick);

  return (
    <div className="station capacity">
      <div>
      <header className="cap-bar">
        <span className="cap-title">Search study</span>
        <span className="cap-arm">{head?.session ?? ""}</span>
        <span className="cap-mission">{cur ? `report ${cur.n}` : st?.over ? "finished" : "loading"}</span>
        <span className="station-spacer" />
        {st?.practice && !st.over ? <a className="cap-continue" href="/visit">Done practicing</a> : null}
        {st ? <span className="cap-count">{st.answered_n} answered · {st.skipped_n} skipped</span> : null}
        {head?.clock ? <span className="cap-clock">{head.clock}</span> : null}
      </header>
      {lost ? (
        <p className="cap-lost" role="alert">
          Lost contact with the study. If the text window that runs the study was closed, start the study again as in
          READ ME FIRST step 1. Your session's clock keeps running.{" "}
          <a href="/visit">Back to the study page</a>
        </p>
      ) : null}
      </div>

      {st?.over ? (
        <div className="judge-done">
          {overLines(st).map((line, i) => <p key={line} className={i ? "cap-hint" : "cap-report-text"}>{line}</p>)}
          <p><a className="cap-continue" href="/visit">Continue</a></p>
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
                <p className={moreReason ? "cap-reason" : "cap-reason is-clamped"}>{linked(review?.maps_reason ?? "")}</p>
                <p className="cap-hint">
                  <button type="button" className="cap-link" onClick={() => { setHot(null); setStepsOpen(false); setMoreReason((v) => !v); }}>{moreReason ? "less" : "more"}</button>
                </p>
                {said.length ? (
                  <p className="cap-key">{said.map((name) => (
                    <span key={name}><span className="cap-key-sw" style={{ "--place": colour(name) } as CSSProperties} />{place(name, name, name)}</span>
                  ))}</p>
                ) : null}
                {moreReason && review ? (
                  <details className="judge-full" onToggle={(e) => { setHot(null); setStepsOpen(e.currentTarget.open); }}><summary>MAPS's whole search, {review.turns?.length ?? 0} steps</summary><Trace instruction={review.instruction} turns={review.turns ?? []} link={linked} /></details>
                ) : null}
              </section>
            ) : null}

            {verifying ? (
              <section className="cap-block">
                <p className="cap-kicker">Mapped objects near MAPS's answer
                  <button type="button" className="cap-link cap-right" onClick={() => setShowIds((v) => !v)}>{showIds ? "hide numbers" : "number them on the map"}</button></p>
                {(moreObjects ? objects : objects.slice(0, NEAREST)).map((c) => (
                  <button type="button" key={c.id} className="cap-row" data-on={sel?.candidate === c.id} data-maps={c.id === review?.maps_pick}
                          onClick={() => { void act({ action: "candidate", id: c.id }); m.flyTo(c.lonlat); }}>
                    <span>{objectName(c)}{c.id === review?.maps_pick ? " · MAPS" : ""}</span><span className="cap-dim">{c.dist_m} m</span>
                  </button>
                ))}
                {objects.length > NEAREST ? (
                  <p className="cap-hint"><button type="button" className="cap-link" onClick={() => setMoreObjects((v) => !v)}>
                    {moreObjects ? "show fewer" : `show ${objects.length - NEAREST} more`}</button></p>
                ) : null}
              </section>
            ) : null}

            <section className="cap-block cap-list">
              <p className="cap-kicker">Your reports</p>
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
                    MAPS's answer: {pick ? objectName(pick, true) : "the point marked on the map"}</p>
                  <div className="cap-actions">
                    <button type="button" className="cap-btn cap-confirm" disabled={busy} onClick={() => void act({ action: "confirm" })}>
                      Confirm <kbd>Enter</kbd>
                    </button>
                    <button type="button" className="cap-btn cap-outline cap-pick" disabled={busy || !canPick} onClick={() => void act({ action: "replace" })}>
                      Pick another <kbd>S</kbd>
                    </button>
                  </div>
                  <p className="cap-hint">{verifyHint(sel)}</p>
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
                  <p className="cap-hint">{cur?.point ? "Your point is on the map. Press Commit, or click again to move it." : "Click the map on the place, then press Commit."}</p>
                </>
              )}
              <p className="cap-hint">
                {askSkip ? (
                  <>Skip this report?{st?.practice ? "" : " It counts as a miss."}{" "}
                    <button type="button" className="cap-link" disabled={busy || !cur} onClick={() => { setAskSkip(false); void act({ action: "skip" }); }}>Yes, skip</button>
                    {" / "}<button type="button" className="cap-link" onClick={() => setAskSkip(false)}>No</button></>
                ) : (
                  <><button type="button" className="cap-link" disabled={busy || !cur} onClick={() => setAskSkip(true)}>skip this report</button>{st?.practice ? null : " (counts as a miss)"}</>
                )}
                {verifying ? <> · <button type="button" className="cap-link" onClick={() => review?.maps_lonlat && m.flyTo(review.maps_lonlat, MAPS_ZOOM)}>go to MAPS's answer <kbd>M</kbd></button></> : null}
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
              {st?.arm === "team" ? <><span className="sw cap-sw-maps" /> MAPS's answer <span className="sw cap-sw-cand" /> mapped objects</> : null}
              <span className="sw cap-sw-pick" /> your point · north up · O shows the street map · shift-click measures
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
