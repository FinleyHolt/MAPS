/** The viewer: the mission's observations as a picture book.
 *
 * Each waypoint flown over lands here as one fused north-up image
 * (`observations/<sha>.jpg` under the run), newest first, paged back and
 * forth. The map's features draw on or off over it to inspect the
 * registration, and the observation the run committed over shows the
 * committed thing lassoed. The transcript stays the model's own words and
 * pictures; these are the operator's to look at.
 */

import { useEffect, useState } from "react";

import { emptyBook, missionClock, missionFile } from "@/app/mission";
import type { MissionState } from "@/app/mission";
import { framePrior, priorFor, type PriorFeature } from "@/app/observation_frame";

export function Viewer({ mission }: {
  /** The mission whose observations the book pages: the seat's, or the run under review. */
  mission: MissionState | null;
}) {
  const shots = mission?.observations ?? [];
  // The page held, or null for the newest: a new arrival turns to it.
  const [held, setHeld] = useState<number | null>(null);
  useEffect(() => { setHeld(null); }, [shots.length, mission?.path]);
  const i = held === null ? shots.length - 1 : Math.min(held, shots.length - 1);
  const shot = i >= 0 ? shots[i] : null;
  const [osm, setOsm] = useState(false);
  const [lasso, setLasso] = useState(true);
  // The prior, once per page load per map, on the first OSM press.
  const map = mission?.map_name ?? null;
  const [prior, setPrior] = useState<{ map: string; features: PriorFeature[] | null } | null>(null);
  useEffect(() => {
    if (!osm || !map) return;
    let live = true;
    priorFor(map).then((features) => { if (live) setPrior({ map, features }); });
    return () => { live = false; };
  }, [osm, map]);
  const features = prior?.map === map ? prior.features : null;
  const drawn = osm && shot?.world_window && shot.shape && features
    ? framePrior(features, shot.world_window, shot.shape) : null;
  const committed = lasso && shot?.commit ? shot.commit : null;
  const url = shot && mission
    ? missionFile(mission, `observations/${committed ?? `${shot.sha}.jpg`}`) : null;
  const caption = shot
    ? `WAYPOINT ${i + 1} OF ${shots.length} · ${shot.target.toUpperCase()} AT (${shot.xy[0].toFixed(0)}, `
      + `${shot.xy[1].toFixed(0)}) · ${shot.looks} LOOK${shot.looks === 1 ? "" : "S"} FUSED · ${shot.ground_m2.toFixed(0)} M² · `
      + `${shot.placed}/${shot.looks} PLACED · T+${missionClock(shot.t_s)}${committed ? " · COMMITTED" : ""}${drawn ? " · OSM" : ""}`
    : "";
  return (
    <section className="viewer" aria-label="The observations">
      <div className="viewer-body">
        {url ? <img className="camera-image" src={url} alt={`waypoint ${i + 1} of ${shots.length}`} /> : null}
        {drawn && shot?.shape ? (
          <svg className="camera-overlay" viewBox={`0 0 ${shot.shape[1]} ${shot.shape[0]}`}
               preserveAspectRatio="xMidYMid meet" aria-hidden="true">
            {drawn.map((f, k) => f.points.length === 1
              ? <circle key={k} cx={f.points[0][0]} cy={f.points[0][1]} r={3} fill={f.colour} />
              : f.closed
                ? <polygon key={k} points={f.points.map((p) => p.join(",")).join(" ")} fill="none"
                           stroke={f.colour} strokeWidth={2} vectorEffect="non-scaling-stroke" />
                : <polyline key={k} points={f.points.map((p) => p.join(",")).join(" ")} fill="none"
                            stroke={f.colour} strokeWidth={2} vectorEffect="non-scaling-stroke" />)}
          </svg>
        ) : null}
        <p className="viewer-title">Observations</p>
        <div className="camera-controls">
          <button type="button" className="camera-toggle" disabled={i <= 0}
                  onClick={() => setHeld(Math.max(0, i - 1))} title="the observation before">
            ‹
          </button>
          <button type="button" className="camera-toggle" disabled={i < 0 || i >= shots.length - 1}
                  onClick={() => setHeld(Math.min(shots.length - 1, i + 1))} title="the observation after">
            ›
          </button>
          <button type="button" className="camera-toggle" aria-pressed={osm} onClick={() => setOsm((v) => !v)}
                  title="The map's features drawn on the observation, to inspect its registration">
            OSM {osm ? "ON" : "OFF"}
          </button>
          {shot?.commit ? (
            <button type="button" className="camera-toggle" aria-pressed={lasso} onClick={() => setLasso((v) => !v)}
                    title="What the run committed to, lassoed on this observation">
              COMMIT {lasso ? "ON" : "OFF"}
            </button>
          ) : null}
        </div>
        {!shot ? (
          <p className="camera-annunciator">{emptyBook(mission)}</p>
        ) : null}
      </div>
      <p className="viewer-caption">{caption}</p>
    </section>
  );
}
