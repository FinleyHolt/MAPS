/** The mission state document, as the station publishes it.
 *
 * The station is the source of truth for every field here; nothing in the
 * browser derives mission state of its own.
 */

import type { LegCoverage, LonLat, WaitingLeg } from "@/app/coverage";

export type MissionStatus = "ready" | "flying" | "awaiting_tasking" | "ended" | "failed";

/** One thing that happened, in the order it happened.
 *
 * The model is served ONE tool named `run` carrying a block of Python, so
 * `run` is the ordinary entry and `said` is that code.
 */
export interface TranscriptEntry {
  kind: string;
  stage?: string;
  turn_idx?: number;
  // The model's own reasoning-channel text, ahead of the code it decided
  // on. "run" entries only.
  reasoning?: string;
  said?: string;
  // print()'d text from running the code block. "run" entries only.
  printed?: string;
  /** Every image this turn put in front of the model, by the sha256 of its
   * bytes: `frames/<sha>.jpg` under the run holds each one as delivered. */
  frames?: string[];
  errored?: boolean;
  error?: string | null;
  ended_by?: string;
  turns?: number;
  /** The scored mission's metrics block, on a "scored" entry (`SR` is the
   * verdict, `NE` metres to the target's centroid). */
  metrics?: Record<string, number | string | boolean | null>;
  streak?: number;
  attempt?: number;
  /** Wall-clock seconds at the first fault of this outage. "transient_error" entries only. */
  since?: number | null;
  n_crops?: number;
  watchlist?: boolean;
  boost?: boolean;
  // "waypoint" entries: one waypoint as the station posted it. `n` is its
  // number within the flight.
  flight?: number;
  n?: number;
  label?: string;
  lat?: number;
  lon?: number;
  observationRadiusMeters?: number;
  altitudeAglMeters?: number;
  /** The ground the leg asks for, as the station samples it, and the share
   * it must hold to arrive; `coverage` is the station's final word once the
   * leg has ended (app/coverage.ts). A "leg" row answered off a pass already
   * held carries `area` and `coverage` too. */
  area?: LonLat[];
  coverageFloor?: number;
  coverage?: LegCoverage | null;
}

export interface Flight {
  target?: string;
  resolved_xy?: [number, number];
  step_idx?: number;
  stop_id?: string;
  // Present only when the mission's frame is georeferenced to the served
  // basemap. Without them the map draws no marker rather than guess.
  lat?: number;
  lon?: number;
}

/** Where the aircraft is: `GET mission/ownship`.
 *
 * Live it is the twin's pose; under review it is the recorded run's pose at
 * the instant the console names. `lon`/`lat` are absent on a map with no
 * georef, and the map draws nothing rather than guessing.
 */
export interface Ownship {
  x: number;
  y: number;
  lon?: number;
  lat?: number;
  /** Degrees, 0 = north, clockwise: what the aircraft marker is rotated to. */
  track_deg: number;
  agl_m: number;
  speed_mps: number;
  map: string;
  t_s?: number;
  duration_s?: number;
  /** The ground the camera covers at this pose: the frame's four corners on
   * flat ground as [lon, lat]. Null where the frame has no ground under it. */
  frame_corners?: [number, number][] | null;
  /** Where the aircraft has been, [lon, lat] oldest first. Live only. */
  trail?: [number, number][];
  live?: boolean;
}

/** A run's score. Every field is optional: a criterion that does not
 * populate a field leaves the key out. */
export interface MissionScore {
  SR?: number | null;
  OSR?: number | null;
  NE?: number | null;
  spl_pos?: number | null;
  success_criterion?: string;
}

/** One waypoint's observation: `observations/<sha>.jpg` under the run is the
 * fused image before any annotation; the picture book draws the map's
 * features on it itself (`observation_frame.ts`) off `world_window` and
 * `shape`. */
export interface ObservationDownlink {
  sha: string;
  target: string;
  xy: [number, number];
  looks: number;
  ground_m2: number;
  placed: number;
  t_s: number;
  /** The ground the image holds, map-local metres `[x0, y0, x1, y1]`, and
   * its size `[h, w]`. */
  world_window?: [number, number, number, number];
  shape?: [number, number];
  /** A file name under `observations/`, on the ONE observation the run
   * committed over: these pixels with the committed thing lassoed on them. */
  commit?: string;
}

/** The URL of a file under a run's folder, or null before the folder exists. */
export function missionFile(mission: MissionState, name: string): string | null {
  return mission.path ? `run/file?path=${encodeURIComponent(mission.path + "/" + name)}` : null;
}

/** The run's commit as a picture: the observation it was drawn on, and the
 * URL of the drawing. Null when nothing was committed or no picture exists. */
export function commitLasso(mission: MissionState): { url: string; target: string } | null {
  const shot = (mission.observations ?? []).filter((o) => o.commit).pop();
  const url = shot?.commit ? missionFile(mission, `observations/${shot.commit}`) : null;
  return url && shot ? { url, target: shot.target } : null;
}

/** `GET station`: the bench this station serves and what its seat can do. */
export interface StationAbout {
  bench: string;
  /** The split the catalog shows when none is asked for, and every split on offer. */
  split: string;
  splits: string[];
  /** The map a request that names none is answered for; null on a station with no catalog. */
  map: string | null;
  flies: boolean;
  /** Why the seat flies nothing, when it does not. */
  why: string | null;
  /** Whether words with no row picked fly as a mission of their own (the twin's), or a mission is always a row. */
  own_words: boolean;
}

/** The map the console is looking at: the reviewed run's, else the row about
 * to fly, else the seat's, else the row last clicked, else the station's
 * default. The basemap, the working area and every placement follow it. */
export function viewedMap(view: {
  replay?: { map_name?: string | null } | null;
  /** The clicked row, when sending would start it. */
  starting?: { map: string } | null;
  mission?: { map_name?: string | null } | null;
  picked?: { map: string } | null;
  about?: { map: string | null } | null;
}): string | null {
  if (view.replay) return view.replay.map_name ?? null;
  return view.starting?.map ?? view.mission?.map_name ?? view.picked?.map ?? view.about?.map ?? null;
}

/** What the picture book says while it holds no observation. A mission on
 * the twin lands one per waypoint; a bench run records none, and its
 * pictures are the ones the model was sent, in the transcript. */
export function emptyBook(mission: Pick<MissionState, "kind" | "status"> | null): string {
  if (mission && (mission.kind !== "twin" || mission.status === "ended" || mission.status === "failed")) {
    return "THIS RUN RECORDED NO OBSERVATIONS · THE PICTURES THE MODEL WAS SENT ARE IN THE TRANSCRIPT";
  }
  return "NO OBSERVATIONS YET · EACH WAYPOINT FLOWN OVER LANDS HERE AS ONE IMAGE";
}

export interface MissionState {
  version: number;
  /** The run folder, relative to the station's output root; null before it exists. */
  path: string | null;
  episode_id: string;
  split: string;
  /** "twin" for a mission flown on the simulated aircraft, "bench" for a
   * bench episode flown from the seat, "replay" for a recorded run. */
  kind: string;
  status: MissionStatus;
  instruction: string;
  ended_by: string | null;
  error: string | null;
  reply_text: string | null;
  turns_used: number;
  /** Null in the run list on a bench run, whose row does not count them: the review counts its stops. */
  flights_flown: number | null;
  /** Unix seconds. */
  started_at: number | null;
  // Which code flew this, read once when the mission starts. `git_dirty`
  // means uncommitted work was in the tree, so the sha names the nearest
  // commit rather than exactly what ran.
  git_commit?: string | null;
  git_dirty?: boolean | null;
  map_name?: string | null;
  /** The run's target footprint as a `[lon, lat]` ring; null where it draws none. */
  target_lonlat?: [number, number][] | null;
  /** The target's centre, `[lon, lat]`: what a ring-scored run is measured to. */
  target_point?: [number, number] | null;
  /** Where a scored bench run committed, `[lon, lat]`; null on a run that never flew or records none. */
  commit_lonlat?: [number, number] | null;
  transcript: TranscriptEntry[];
  flights: Flight[];
  /** Each waypoint's observation, fused as one north-up image, in order. */
  observations?: ObservationDownlink[];
  /** The leg the station is waiting on, with its area and the share the
   * passes hold of it so far; null when no leg waits. */
  leg?: WaitingLeg | null;
  score?: MissionScore | null;
  run_no?: number | null;
}

/** "RUN 3", or null on a run with no number. */
export function runLabel(mission: MissionState | null | undefined): string | null {
  if (!mission || mission.run_no == null) return null;
  return `RUN ${mission.run_no}`;
}

/** One sample of the path a run's aircraft flew. `lon`/`lat` are absent
 * where the map carries no georef, and such a sample is not drawn. */
export interface TrackPoint {
  t_s: number;
  x: number;
  y: number;
  track_deg: number;
  agl_m: number;
  lon?: number;
  lat?: number;
}

/** `GET replay`: a finished run, plus its extent and the path it flew. */
export interface ReplayState extends MissionState {
  /** The scrubber's length, seconds. Null on a run with nothing to scrub: a
   * bench run's aircraft was placed at each stop, never flown between them. */
  duration_s: number | null;
  /** The flown path, oldest first. A gap in the record is a real gap in this
   * array: `trackLegs` breaks the line there. A run with no recorded track
   * carries its `visited_xy` here, one sample a pose in order. */
  track: TrackPoint[];
  path: string;
}

/** The route as drawable legs, `[lon, lat]` oldest first, broken wherever
 * the recording is.
 *
 * The station samples the track at a fixed step, so a step wider than that
 * is a dropout: a line drawn across it would claim a track that was never
 * recorded. A sample the map cannot place breaks the leg for the same reason.
 *
 * ponytail: the step is the smallest gap in the track, which a track of two
 * samples cannot tell from a dropout. A nominal step on the response would
 * settle that case.
 */
export function trackLegs(track: TrackPoint[]): [number, number][][] {
  const placed = (p: TrackPoint) => typeof p.lon === "number" && typeof p.lat === "number";
  const deltas = track.slice(1).map((p, i) => p.t_s - track[i].t_s).filter((d) => d > 0);
  const step = deltas.length ? Math.min(...deltas) : 0;
  const legs: [number, number][][] = [];
  let leg: [number, number][] = [];
  track.forEach((point, i) => {
    const previous = track[i - 1];
    const broken = !placed(point)
      || (previous !== undefined && (!placed(previous) || point.t_s - previous.t_s > step * 1.5));
    if (broken && leg.length) {
      legs.push(leg);
      leg = [];
    }
    if (placed(point)) leg.push([point.lon as number, point.lat as number]);
  });
  if (leg.length) legs.push(leg);
  return legs.filter((l) => l.length > 1);
}

/** What the status chip reads. Sentence case, said from the operator's side. */
export const STATUS_LABEL: Record<string, string> = {
  idle: "No mission",
  ready: "Ready",
  flying: "Flying",
  awaiting_tasking: "Holding for tasking",
  ended: "Ended",
  failed: "Failed",
};

/** Rows the STATION writes, not the reasoner. The aircraft flies a leg with
 * the model never called, posting waypoints and legs the whole way, so none
 * of them is evidence that the reasoner answered. */
const STATIONS_OWN: ReadonlySet<string> = new Set(["waypoint", "leg", "view", "retask", "summary"]);

/** The reasoner is not answering under a flying mission: the loop is holding
 * its turn and the last thing the REASONER wrote is a fault. Null once the
 * reasoner answers again. The station's own rows are skipped, so an outage
 * that began inside a leg wait stays annunciated through the leg. */
export function armOutage(
  mission: Pick<MissionState, "status" | "transcript"> | null | undefined,
): { attempt: number; since: number | null } | null {
  if (!mission || mission.status !== "flying") return null;
  const rows = mission.transcript ?? [];
  for (let i = rows.length - 1; i >= 0; i--) {
    if (STATIONS_OWN.has(rows[i].kind)) continue;
    if (rows[i].kind !== "transient_error") return null;
    return { attempt: rows[i].attempt ?? 0, since: rows[i].since ?? null };
  }
  return null;
}

/** A mission the operator can still speak to. */
export function isLive(mission: MissionState | null): boolean {
  return mission?.status === "flying" || mission?.status === "awaiting_tasking";
}

/** "SR 1 · OSR — · NE 1.6m · SPL —": every field named. An em dash reads as
 * "not applicable under this criterion", not as a missing measurement. */
export function formatScore(score: MissionScore | null | undefined): string | null {
  if (!score) return null;
  // `== null` takes the absent field as well as the null one.
  const num = (v: number | null | undefined, unit = "") =>
    v == null ? "—" : `${Number.isInteger(v) ? v : v.toFixed(2)}${unit}`;
  return `SR ${num(score.SR)} · OSR ${num(score.OSR)} · NE ${num(score.NE, "m")} · SPL ${num(score.spl_pos)}`;
}

/** "1:23": where in a mission an instant is, off its start. */
export function missionClock(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/** "a1b2c3d4": the run's build, short. A dirty tree gets a trailing "+".
 * Null when the build is unknown. */
export function formatCommit(
  commit: string | null | undefined,
  dirty?: boolean | null,
): string | null {
  if (!commit) return null;
  return `${commit.slice(0, 8)}${dirty ? "+" : ""}`;
}

export async function postJson<T = unknown>(
  path: string,
  body: unknown,
): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    // The station returns its refusals as `detail`, written to be read.
    const detail = await response
      .json()
      .then((b) => b?.detail)
      .catch(() => null);
    throw new Error(detail ?? `${path} failed (${response.status})`);
  }
  return (await response.json()) as T;
}
