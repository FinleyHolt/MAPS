/** The station console: pick a mission, watch it, talk to it.
 *
 * One conversation. The mission list has one job: fill the chat box with a
 * starting point. Everything past that is the same chat turn whether it is
 * the first message of a mission or a re-task mid-flight: starting IS
 * sending the first message.
 *
 * The station serves one bench (`GET station`): its splits fill the split
 * menu, and the map every pane draws is the map of the mission being looked
 * at (`viewedMap`), since a bench has one map per block. On the twin, words
 * with no row picked fly as the operator's own mission; a bench episode is
 * always a row of its split.
 *
 * A row of the list expands in place to that mission's finished runs, and
 * clicking one opens the REVIEW: the transcript, map, observations and
 * ownship then read that run instead of the seat. Review is a fact about
 * THIS browser, not a mode on the station, so Return to live is a state
 * change here rather than a verb there.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { flownPasses } from "@/app/coverage";
import { MapPane } from "@/app/MapPane";
import { Transcript } from "@/app/Transcript";
import { Viewer } from "@/app/Viewer";
import type { MissionState, Ownship, ReplayState, StationAbout } from "@/app/mission";
import {
  STATUS_LABEL,
  armOutage,
  formatCommit,
  formatScore,
  isLive,
  missionClock,
  postJson,
  runLabel,
  trackLegs,
  viewedMap,
} from "@/app/mission";

interface CatalogEntry {
  episode_id: string;
  instruction: string;
  map: string;
  /** The target's footprint as a `[lon, lat]` ring; null where the row draws none. */
  target_lonlat: [number, number][] | null;
}

// The two-fifths default, and how far the divider may travel from it.
const SIDEBAR_DEFAULT = 0.4;
const SIDEBAR_MIN = 0.22;
const SIDEBAR_MAX = 0.62;

// The same, vertically, inside the sidebar: how much of it the mission
// list takes before the run view starts.
const MISSIONS_DEFAULT = 0.3;
const MISSIONS_MIN = 0.1;
const MISSIONS_MAX = 0.62;

// How often the aircraft's position is asked for, and the wall second
// review playback advances by. A poll of its own rather than a field of the
// mission document: the ownship changes every tick and would re-render the
// whole transcript under an operator reading it.
const OWNSHIP_POLL_MS = 1000;


const clamp = (v: number, lo: number, hi: number) =>
  Math.min(hi, Math.max(lo, v));

export function MissionConsole() {
  // The bench this station serves and what its seat can do; null until it answers.
  const [about, setAbout] = useState<StationAbout | null>(null);
  // The split the catalog shows; "" is the bench's default.
  const [split, setSplit] = useState("");
  useEffect(() => {
    fetch("station")
      .then((r) => (r.ok ? r.json() : null))
      .then((body) => setAbout(body as StationAbout | null))
      .catch(() => {
        // A station that does not say what it is still lists and reviews.
      });
  }, []);
  const [catalog, setCatalog] = useState<CatalogEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [query, setQuery] = useState("");
  const [mission, setMission] = useState<MissionState | null>(null);
  const [draft, setDraft] = useState("");
  const [fault, setFault] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [sidebarFrac, setSidebarFrac] = useState(SIDEBAR_DEFAULT);
  const [missionsFrac, setMissionsFrac] = useState(MISSIONS_DEFAULT);
  const [dragging, setDragging] = useState<"col" | "row" | null>(null);
  const [runs, setRuns] = useState<MissionState[]>([]);
  // The finished run under review, off GET replay, or null for the live seat.
  const [replay, setReplay] = useState<ReplayState | null>(null);
  const [expandedMission, setExpandedMission] = useState<string | null>(null);
  const [ownship, setOwnship] = useState<Ownship | null>(null);
  const [effort, setEffort] = useState("");
  // The served model and the efforts its card takes, off GET reasoner; "" sends none and the model's default stands.
  const [reasoner, setReasoner] = useState<{ model: string | null; efforts: string[] }>({ model: null, efforts: [] });
  useEffect(() => {
    fetch("reasoner")
      .then((r) => r.json())
      .then((body) => setReasoner({ model: body.model ?? null, efforts: body.efforts ?? [] }))
      .catch(() => {
        // No reasoner: the menu offers the default alone, and a send says why it cannot fly.
      });
  }, []);
  // A row, clicked: its words go into the box and its target onto the map.
  const [picked, setPicked] = useState<CatalogEntry | null>(null);
  const [playing, setPlaying] = useState(true);
  // Seconds into the reviewed run, mirrored into state so the scrubber and
  // the clock re-render. simRef is the value the poll reads.
  const [simT, setSimT] = useState(0);
  const simRef = useRef(0);
  // Asks for the aircraft now, so a scrub lands on the dropped instant.
  const askOwnshipRef = useRef<() => void>(() => {});
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const sidebarRef = useRef<HTMLElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const playingRef = useRef(playing);
  playingRef.current = playing;

  // Each catalog row's finished runs, grouped once per read.
  const runsByEpisode = useMemo(() => {
    const byEpisode: Record<string, MissionState[]> = {};
    for (const past of runs) {
      if (past.status !== "ended") continue;
      (byEpisode[past.episode_id] ??= []).push(past);
    }
    return byEpisode;
  }, [runs]);

  const openReplay = useCallback(async (past: MissionState) => {
    if (!past.path) return;
    setFault(null);
    try {
      const response = await fetch(`replay?path=${encodeURIComponent(past.path)}`);
      if (!response.ok) {
        const detail = await response.json().then((b) => b?.detail).catch(() => null);
        throw new Error(detail ?? `replay: ${response.status}`);
      }
      setReplay((await response.json()) as ReplayState);
    } catch (err) {
      setFault(String((err as Error).message ?? err));
    }
  }, []);

  // Back to the seat. The review is which run this browser puts in its
  // queries, so dropping it IS returning.
  const returnToLive = useCallback(() => {
    setReplay(null);
    setFault(null);
  }, []);

  // The whole track the reviewed run flew, broken where the recording is.
  const legs = useMemo(() => trackLegs(replay?.track ?? []), [replay]);

  // Everything reads the mission the console is looking at: the replay when
  // one is open, the seat otherwise.
  const activeMission = replay ?? mission;

  // The legs the active mission has finished with, each with the station's
  // final word on its area. The waiting leg draws live, and only while
  // nothing is under review.
  const waitingLeg = replay ? null : mission?.leg ?? null;
  const passes = useMemo(
    () => flownPasses(activeMission?.transcript ?? [], waitingLeg),
    [activeMission, waitingLeg],
  );

  const extent = replay?.duration_s ?? null;

  const fetchRuns = useCallback(() => {
    fetch("missions/runs")
      .then((r) => r.json())
      .then((body) => setRuns(body.runs ?? []))
      .catch(() => {
        // The run list is a convenience read: a failed fetch must not block the seat.
      });
  }, []);

  // Read at load, and again whenever the seat's mission changes state: a
  // mission that just ended is a run the list should show.
  const seatStatus = mission?.status ?? null;
  useEffect(() => {
    fetchRuns();
  }, [fetchRuns, seatStatus]);

  // The catalog, filtered server-side: the search is the picker.
  useEffect(() => {
    const timer = setTimeout(() => {
      const url = "missions?limit=1000"
        + (query ? `&q=${encodeURIComponent(query)}` : "")
        + (split ? `&split=${encodeURIComponent(split)}` : "");
      fetch(url)
        .then((r) => r.json())
        .then((body) => {
          setCatalog(body.missions ?? []);
          setTotal(body.total ?? 0);
        })
        .catch((err) => setFault(`catalog unavailable: ${err.message}`));
    }, 200);
    return () => clearTimeout(timer);
  }, [query, split]);

  // The mission as it happens. The stream carries the whole state document
  // per message (null for an empty seat), so a console that joins late shows
  // where the mission IS.
  useEffect(() => {
    const source = new EventSource("mission/events");
    source.onmessage = (event) => {
      try {
        setMission(JSON.parse(event.data));
      } catch {
        // A malformed message is one message; the next carries the same state.
      }
    };
    return () => source.close();
  }, []);

  const pick = useCallback((entry: CatalogEntry) => {
    setFault(null);
    setPicked(entry);
    setDraft(entry.instruction);
    inputRef.current?.focus();
  }, []);

  // The aircraft, asked for once a second. Live it is wherever the twin is.
  // Under review the console owns the clock: each tick advances the run's
  // time by the wall second just spent and asks where the aircraft was then.
  const replayPath = replay?.path ?? null;
  const replayDuration = replay?.duration_s ?? null;
  useEffect(() => {
    simRef.current = 0;
    setSimT(0);
    setPlaying(replayPath !== null);
    let cancelled = false;
    let asked = 0;
    const ask = () => {
      const mine = ++asked;
      const url = replayPath === null
        ? "mission/ownship"
        : `mission/ownship?path=${encodeURIComponent(replayPath)}&t=${simRef.current}`;
      fetch(url)
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null)
        .then((body) => {
          // An answer overtaken by a newer ask is dropped.
          if (!cancelled && mine === asked) setOwnship((body as Ownship | null) ?? null);
        });
    };
    askOwnshipRef.current = ask;
    ask();
    const id = window.setInterval(() => {
      if (replayPath !== null && playingRef.current) {
        const next = simRef.current + OWNSHIP_POLL_MS / 1000;
        // Playback stops on the end rather than running past it.
        if (replayDuration !== null && next >= replayDuration) {
          simRef.current = replayDuration;
          setPlaying(false);
        } else {
          simRef.current = next;
        }
        setSimT(simRef.current);
      }
      ask();
    }, OWNSHIP_POLL_MS);
    return () => {
      cancelled = true;
      askOwnshipRef.current = () => {};
      window.clearInterval(id);
    };
  }, [replayPath, replayDuration]);

  // The divider drags, both of them. Listeners sit on window because a fast
  // drag outruns the handle's own hit area. One effect for both axes: the
  // column handle measures against the body, the row handle against the
  // sidebar.
  useEffect(() => {
    if (!dragging) return;
    const col = dragging === "col";
    const axis = {
      col: { box: bodyRef, set: setSidebarFrac, min: SIDEBAR_MIN, max: SIDEBAR_MAX },
      row: { box: sidebarRef, set: setMissionsFrac, min: MISSIONS_MIN, max: MISSIONS_MAX },
    }[dragging];
    const onMove = (event: PointerEvent) => {
      const box = axis.box.current as HTMLElement | null;
      if (!box) return;
      const rect = box.getBoundingClientRect();
      const frac = col
        ? (event.clientX - rect.left) / rect.width
        : (event.clientY - rect.top) / rect.height;
      axis.set(clamp(frac, axis.min, axis.max));
    };
    const onUp = () => setDragging(null);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    const prevCursor = document.body.style.cursor;
    const prevSelect = document.body.style.userSelect;
    document.body.style.cursor = col ? "col-resize" : "row-resize";
    document.body.style.userSelect = "none";
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      document.body.style.cursor = prevCursor;
      document.body.style.userSelect = prevSelect;
    };
  }, [dragging]);

  const live = isLive(mission);
  const flying = mission?.status === "flying";
  // ONE verb. A picked row's words fly that row; otherwise the box speaks
  // to the live mission (a re-task in flight, an answer to a parked
  // report); with neither, the words fly as the operator's own mission.
  const startsPicked = picked !== null && !(live && picked.episode_id === mission?.episode_id);
  // On a bench a mission is a row of the split: words alone start nothing.
  const ownWords = about?.own_words ?? true;
  const speaks = live && !startsPicked;
  const canSend = draft.trim() !== "" && (speaks || startsPicked || ownWords) && (speaks || (about?.flies ?? true));
  const benchSeat = mission?.kind === "bench";

  // The next mission starts from nothing: the seat emptied on the station
  // (a mission flying is stopped), and this browser's own half let go with it.
  const newMission = useCallback(async () => {
    setBusy(true);
    setFault(null);
    try {
      await postJson("mission/clear", {});
      setMission(null);
      setDraft("");
      setPicked(null);
      setReplay(null);
      fetchRuns();
    } catch (err) {
      setFault(String((err as Error).message ?? err));
    } finally {
      setBusy(false);
    }
  }, [fetchRuns]);

  const send = useCallback(async () => {
    const text = draft.trim();
    if (!text) return;
    setBusy(true);
    setFault(null);
    try {
      if (live && !startsPicked) {
        setMission(await postJson<MissionState>("mission/tasking", { text }));
      } else {
        // A catalog row when one is picked; else the operator's own tasking,
        // flown from the map centre, unscored.
        setMission(await postJson<MissionState>("mission/next", {
          ...(startsPicked && picked ? { episode_id: picked.episode_id } : {}),
          instruction: text,
          ...(effort ? { reasoning_effort: effort } : {}),
          ...(split ? { split } : {}),
        }));
        setReplay(null);
      }
      setDraft("");
    } catch (err) {
      setFault(String((err as Error).message ?? err));
    } finally {
      setBusy(false);
    }
  }, [effort, draft, live, startsPicked, picked, split]);

  // The box grows with its content; a max-height in CSS keeps a long
  // tasking from swallowing the conversation above it.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [draft]);

  const outage = armOutage(mission);
  const reviewOnly = about !== null && !about.flies;
  const emptySeatHint = reviewOnly
    ? `${runs.length} recorded run${runs.length === 1 ? "" : "s"} to review. A mission marked completed expands to its runs: click the run count on its row, then the run.`
    : ownWords
      ? "Click a mission: its words go in the box, its target on the map, and send flies it. Or type your own tasking: send flies it from the map centre, unscored."
      : "Click a mission: its words go in the box, its target on the map, and send flies it as the bench flies it. Edit the words to fly the same row under your own.";

  const hint = fault
    ? fault
    : live
      ? flying
        ? "The agent is flying. Your message lands on its next turn."
        : benchSeat
          ? "The agent reported and is holding for your tasking. + ends the mission: the bench scores it and its run lists under its row."
          : "The agent reported and is holding for your tasking."
      : mission?.status === "ended" || mission?.status === "failed"
        ? `Mission ${mission.status}${mission.ended_by ? ` (${mission.ended_by})` : ""}. A row, clicked and sent, flies again.`
        : emptySeatHint;

  return (
    <div className="station" data-review={replay !== null}>
      {replay ? (
        // Which run is on screen is the one thing an operator must not get
        // wrong, so this is a band across the whole console: the map, the
        // observations and the conversation are all a past run's while it
        // stands, and Return to live is the only way out.
        <div className="review-bar" role="status">
          <span className="review-word">Reviewing a past run</span>
          <span className="review-run">
            {[runLabel(replay), replay.episode_id, replay.path].filter(Boolean).join(" · ")}
          </span>
          <span className="review-when">
            {replay.started_at ? new Date(replay.started_at * 1000).toLocaleString() : ""}
          </span>
          <span className="station-spacer" />
          <button type="button" className="review-exit" onClick={returnToLive}>
            Return to live
          </button>
        </div>
      ) : null}
      <header className="station-bar">
        <span className="station-name">MAPS · station{about ? ` · ${about.bench}` : ""}</span>
        <span className="station-mission">
          {mission
            ? [runLabel(mission), mission.episode_id].filter(Boolean).join(" · ")
            : "no mission"}
        </span>
        <span className="station-spacer" />
        <Readout label="turns" value={activeMission?.turns_used ?? 0} />
        <Readout label="flights" value={activeMission?.flights_flown ?? activeMission?.flights?.length ?? 0} />
        <span className="status-chip" data-status={replay ? "review" : mission?.status ?? "idle"}>
          {replay ? "Review" : STATUS_LABEL[mission?.status ?? "idle"] ?? mission?.status}
        </span>
        {outage ? (
          <span
            className="arm-down"
            data-testid="arm-down"
            title="the reasoner is not answering; the mission holds its turn and resumes on its own the moment it answers"
          >
            reasoner not answering · attempt {outage.attempt}{outage.since ? ` · ${missionClock(Math.max(0, Math.round(Date.now() / 1000 - outage.since)))}` : ""} · holding the turn
          </span>
        ) : null}
      </header>

      <div
        className="station-body"
        ref={bodyRef}
        style={{ gridTemplateColumns: `${sidebarFrac * 100}% 6px 1fr` }}
      >
        <aside
          className="sidebar"
          ref={sidebarRef}
          style={{
            gridTemplateRows: `${missionsFrac * 100}% 6px minmax(0, 1fr) auto auto`,
          }}
        >
          <div className="missions">
            <div className="missions-head">
              <span className="missions-title">Missions</span>
              {about && about.splits.length > 1 ? (
                <select
                  className="missions-split"
                  aria-label="Split"
                  title={`the ${about.bench} split the list shows and a picked row flies from`}
                  value={split || about.split}
                  onChange={(event) => {
                    setSplit(event.target.value);
                    setPicked(null);
                    setExpandedMission(null);
                  }}
                >
                  {about.splits.map((name) => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
              ) : null}
              <span className="missions-count">
                {catalog.length} of {total}
              </span>
            </div>
            <input
              className="missions-search"
              type="search"
              placeholder="Search missions"
              aria-label="Search missions"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
            <div className="missions-list">
              {catalog.length === 0 ? (
                <p className="rail-empty">
                  {query ? `No mission matches "${query}".` : "No missions in this catalog."}
                </p>
              ) : (
                catalog.map((entry) => {
                  const past = runsByEpisode[entry.episode_id] ?? [];
                  const expanded = expandedMission === entry.episode_id;
                  return (
                    <div key={entry.episode_id} className="mission-group">
                      <div className="mission-row-line">
                        <div
                          className="mission-row"
                          role="button"
                          tabIndex={0}
                          aria-current={mission?.episode_id === entry.episode_id}
                          aria-selected={picked?.episode_id === entry.episode_id}
                          title="put this mission's words in the box and its target on the map; send flies it"
                          onClick={() => pick(entry)}
                          onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                              event.preventDefault();
                              pick(entry);
                            }
                          }}
                        >
                          <span className="mission-text">{entry.instruction}</span>
                          <span className="mission-id">
                            {entry.episode_id}
                            {past.length ? (
                              <span className="mission-flown" title="this mission has a finished run">
                                completed
                              </span>
                            ) : null}
                          </span>
                        </div>
                        {past.length > 0 ? (
                          <button
                            type="button"
                            className="mission-runs-toggle"
                            aria-expanded={expanded}
                            aria-label={`${past.length} past run${past.length === 1 ? "" : "s"} of this mission`}
                            onClick={() =>
                              setExpandedMission(expanded ? null : entry.episode_id)
                            }
                          >
                            {past.length} run{past.length === 1 ? "" : "s"}
                            <span className="mission-runs-caret">
                              {expanded ? "▾" : "▸"}
                            </span>
                          </button>
                        ) : null}
                      </div>
                      {expanded
                        ? past.map((run) => (
                            <button
                              key={run.path ?? `${run.episode_id}-${run.started_at}`}
                              type="button"
                              className="past-row past-row-nested"
                              data-status={run.status}
                              aria-current={replay !== null && run.path === replay.path}
                              title="review this run"
                              onClick={() => void openReplay(run)}
                            >
                              <span className="past-row-text">
                                {runLabel(run) ?? "RUN"}
                                {run.started_at
                                  ? ` · ${new Date(run.started_at * 1000).toLocaleString()}`
                                  : ""}
                              </span>
                              <span className="past-row-meta">
                                {run.ended_by ?? STATUS_LABEL[run.status] ?? run.status}
                                {" · "}
                                {run.turns_used}t{run.flights_flown == null ? "" : ` / ${run.flights_flown}f`}
                                {formatCommit(run.git_commit, run.git_dirty) ? (
                                  <span className="past-row-commit">
                                    {" · "}
                                    {formatCommit(run.git_commit, run.git_dirty)}
                                  </span>
                                ) : null}
                              </span>
                              {formatScore(run.score) ? (
                                <span className="past-row-score">{formatScore(run.score)}</span>
                              ) : null}
                            </button>
                          ))
                        : null}
                    </div>
                  );
                })
              )}
            </div>
          </div>

          <div
            className="divider"
            data-axis="row"
            role="separator"
            aria-orientation="horizontal"
            aria-label="Resize the mission list"
            aria-valuenow={Math.round(missionsFrac * 100)}
            aria-valuemin={Math.round(MISSIONS_MIN * 100)}
            aria-valuemax={Math.round(MISSIONS_MAX * 100)}
            tabIndex={0}
            data-dragging={dragging === "row"}
            onPointerDown={(event) => {
              event.preventDefault();
              setDragging("row");
            }}
            onKeyDown={(event) => {
              const step = event.shiftKey ? 0.05 : 0.02;
              if (event.key === "ArrowUp") {
                event.preventDefault();
                setMissionsFrac((f) => Math.max(MISSIONS_MIN, f - step));
              } else if (event.key === "ArrowDown") {
                event.preventDefault();
                setMissionsFrac((f) => Math.min(MISSIONS_MAX, f + step));
              } else if (event.key === "Home") {
                event.preventDefault();
                setMissionsFrac(MISSIONS_DEFAULT);
              }
            }}
          >
            <span className="divider-grip" />
          </div>

          <div className="run-view">
            <div className="run-scroll">
              {activeMission?.error ? <p className="fault">{activeMission.error}</p> : null}
              {/* The score, plainly: a reviewed run's, or the seat's once the bench has scored it. */}
              {activeMission && formatScore(activeMission.score) ? (
                <p className="past-detail-score">
                  {formatScore(activeMission.score)}
                  {formatCommit(activeMission.git_commit, activeMission.git_dirty) ? (
                    <span className="past-detail-commit">
                      {formatCommit(activeMission.git_commit, activeMission.git_dirty)}
                    </span>
                  ) : null}
                </p>
              ) : null}
              {activeMission ? (
                <Transcript mission={activeMission} />
              ) : (
                <p className="run-empty">{emptySeatHint}</p>
              )}
            </div>
          </div>

          <div className="chatbox">
            <textarea
              ref={inputRef}
              className="chatbox-input"
              aria-label="Message the agent"
              placeholder={reviewOnly ? "This station reviews recorded runs; it flies none" : startsPicked ? "The mission's words: edit them if you like, then send to fly it" : flying ? "Message the agent" : live ? "Answer the report, or click a mission" : ownWords ? "Click a mission, or type your own tasking" : "Click a mission"}
              disabled={busy || reviewOnly}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void send();
                }
              }}
              rows={1}
            />
            <select
              className="chatbox-effort"
              aria-label="Reasoning effort"
              title={`the reasoning effort the next mission flies at${reasoner.model ? `, as ${reasoner.model} takes it` : ""}; greyed while a mission is live, whose own pick stands`}
              value={effort}
              disabled={busy || live}
              onChange={(event) => setEffort(event.target.value)}
            >
              <option value="">default</option>
              {reasoner.efforts.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
            <button
              type="button"
              className="chatbox-send"
              data-new
              aria-label="New mission"
              title={flying ? "new mission: stop the one flying, empty the seat and the transcript" : "new mission: empty the seat and the transcript, then click a row or type your own tasking"}
              onClick={newMission}
              disabled={busy}
            >
              +
            </button>
            <button
              type="button"
              className="chatbox-send"
              aria-label="Send"
              onClick={send}
              disabled={!canSend || busy}
            >
              ↑
            </button>
          </div>
          <p className="chatbox-hint" data-fault={fault !== null}>
            {hint}
          </p>
        </aside>

        <div
          className="divider"
          data-axis="col"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize the sidebar"
          aria-valuenow={Math.round(sidebarFrac * 100)}
          aria-valuemin={Math.round(SIDEBAR_MIN * 100)}
          aria-valuemax={Math.round(SIDEBAR_MAX * 100)}
          tabIndex={0}
          data-dragging={dragging === "col"}
          onPointerDown={(event) => {
            event.preventDefault();
            setDragging("col");
          }}
          onKeyDown={(event) => {
            const step = event.shiftKey ? 0.05 : 0.02;
            if (event.key === "ArrowLeft") {
              event.preventDefault();
              setSidebarFrac((f) => Math.max(SIDEBAR_MIN, f - step));
            } else if (event.key === "ArrowRight") {
              event.preventDefault();
              setSidebarFrac((f) => Math.min(SIDEBAR_MAX, f + step));
            } else if (event.key === "Home") {
              event.preventDefault();
              setSidebarFrac(SIDEBAR_DEFAULT);
            }
          }}
        >
          <span className="divider-grip" />
        </div>

        <div className="instrument">
          <MapPane
            map={viewedMap({ replay, starting: startsPicked ? picked : null, mission, picked, about })}
            missionLabel={activeMission?.episode_id ?? null}
            ownship={ownship}
            legs={legs}
            flights={activeMission?.flights}
            target={replay
              ? replay.target_lonlat
              : startsPicked ? picked?.target_lonlat : (mission?.target_lonlat ?? picked?.target_lonlat ?? null)}
            targetPoint={replay ? replay.target_point : startsPicked ? null : mission?.target_point}
            commit={activeMission?.commit_lonlat}
            leg={waitingLeg}
            passes={passes}
          />

          <div className="band">
            <Viewer mission={activeMission} />
          </div>

          {extent != null ? (
            <div className="transport">
              <button
                type="button"
                className="transport-play"
                aria-label={playing ? "Pause" : "Play"}
                onClick={() => {
                  // Play at the end starts it over.
                  if (!playing && simRef.current >= extent) {
                    simRef.current = 0;
                    setSimT(0);
                  }
                  setPlaying((v) => !v);
                }}
              >
                {playing ? "❚❚" : "▶"}
              </button>
              <input
                className="transport-scrub"
                type="range"
                aria-label="Seek within the reviewed run"
                min={0}
                max={extent}
                step={0.1}
                value={Math.min(simT, extent)}
                onChange={(event) => {
                  const at = Number(event.target.value);
                  simRef.current = at;
                  setSimT(at);
                  // The aircraft lands on the dropped instant, not on the next tick.
                  askOwnshipRef.current();
                }}
              />
              <span className="transport-clock">
                {missionClock(simT)} / {missionClock(extent)}
              </span>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function Readout({ label, value }: { label: string; value: number }) {
  return (
    <span className="readout">
      <span className="readout-label">{label}</span>
      <span className="readout-value">{value}</span>
    </span>
  );
}
