/** The mission conversation, in the order it happened.
 *
 * The left rule says who acted: ink for the operator, orange for the
 * model's own decisions, red for a fault.
 *
 * The model is served ONE tool named `run` carrying a block of Python, so
 * an ordinary turn renders as code. The loop truncates that code at 400
 * characters on its way into the transcript, which the entry says out loud
 * rather than showing a block that quietly stops mid-statement.
 */

import { useEffect, useState } from "react";

import { coverageLabel, coverageMet } from "@/app/coverage";
import { commitLasso, missionFile } from "@/app/mission";
import type { MissionState, TranscriptEntry } from "@/app/mission";

// What the loop truncates `said` to. Stated here so the
// console can say so instead of showing a silently clipped block.
const SAID_LIMIT = 400;

// Repairs: the loop nudging the model back onto the channel. Worth
// showing, because they cost a turn, but they are not the model deciding
// anything.
const REPAIRS: Record<string, string> = {
  malformed_envelope: "Reply carried envelope residue. Repaired.",
  empty_turn: "Empty turn. Repaired.",
  code_shaped_reply: "Reply was code on the wrong channel. Repaired.",
  no_flight_reply: "Replied without flying. Repaired.",
  no_call: "No tool call.",
};

export function Transcript({ mission }: { mission: MissionState }) {
  // The panel scrolls when a person scrolls it and at no other time: a
  // row landing under an operator who is reading moves nothing.

  // A leg's pass fuses and the model's turn on it takes a minute or two.
  // Under the last leg or waypoint row of a flying mission, say the model
  // is looking, with the clock running from when this page first saw that row.
  const last = mission.transcript.length - 1;
  const looking =
    mission.status === "flying" && ["leg", "waypoint"].includes(mission.transcript[last]?.kind ?? "");
  const [saw, setSaw] = useState<{ row: number; at: number } | null>(null);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!looking) return;
    setSaw((s) => (s?.row === last ? s : { row: last, at: Date.now() }));
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [looking, last]);

  return (
    <div className="transcript-inner">
      {mission.instruction ? (
        <div className="entry" data-who="operator">
          <p className="entry-label">You</p>
          <p className="entry-prose">{mission.instruction}</p>
        </div>
      ) : null}

      {mission.transcript.map((entry, i) => (
        <Entry key={`${entry.kind}-${entry.turn_idx ?? i}-${i}`} entry={entry} mission={mission} />
      ))}

      {looking && saw?.row === last ? (
        <p className="entry-note">
          the model is looking at the pass · {Math.max(0, Math.round((now - saw.at) / 1000))} s
        </p>
      ) : null}

      {mission.reply_text ? (
        <Report text={mission.reply_text} mission={mission} />
      ) : null}
    </div>
  );
}

/** The images this turn put in front of the model, exactly as delivered
 *  (`frames/<sha256>.jpg` under the run, written by the loop as it sent
 *  them, annotations and all): the audit of what the model was looking at
 *  when it decided. Each opens full size in a new tab. */
function Frames({ entry, mission }: { entry: TranscriptEntry; mission: MissionState }) {
  const shas = entry.frames ?? [];
  if (!shas.length) return null;
  return (
    <div className="entry-frames">
      {shas.map((sha) => {
        const url = missionFile(mission, `frames/${sha}.jpg`);
        if (!url) return null;
        return (
          <a key={sha} href={url} target="_blank" rel="noreferrer" title={`what the model saw · ${sha.slice(0, 12)}`}>
            <img src={url} alt={`what the model saw, ${sha.slice(0, 8)}`} loading="lazy" />
          </a>
        );
      })}
    </div>
  );
}

function Entry({ entry, mission }: { entry: TranscriptEntry; mission: MissionState }) {
  const turn =
    entry.turn_idx === undefined ? null : (
      <span className="entry-turn">turn {entry.turn_idx + 1}</span>
    );

  if (entry.kind === "retask") {
    return (
      <div className="entry" data-who="operator">
        <p className="entry-label">You {turn}</p>
        <p className="entry-prose">{entry.said}</p>
      </div>
    );
  }

  if (entry.kind === "run" || entry.kind === "say") {
    const said = entry.said ?? "";
    return (
      <div className="entry" data-who="model" data-errored={entry.errored}>
        <p className="entry-label">
          {entry.kind === "run" ? "Model" : "Model, spoken"} {turn}
        </p>
        {entry.reasoning ? (
          <p className="entry-reasoning">{entry.reasoning}</p>
        ) : null}
        {entry.kind === "run" ? (
          <pre className="entry-code">{said}</pre>
        ) : (
          <p className="entry-prose">{said}</p>
        )}
        {said.length >= SAID_LIMIT ? (
          <p className="entry-truncated">
            recorded to {SAID_LIMIT} characters
          </p>
        ) : null}
        <Frames entry={entry} mission={mission} />
        {entry.printed ? <pre className="entry-output">{entry.printed}</pre> : null}
        {entry.errored ? (
          <p className="entry-note">The model read an error and re-decided.</p>
        ) : null}
      </div>
    );
  }

  // One waypoint, said as the station posted it.
  // Under it, the share of its area the passes hold: live off the waiting
  // leg while the station waits on this one, with how close the aircraft
  // came and how many passes fell short, else the row's own final word.
  if (entry.kind === "waypoint") {
    const waiting = mission.leg && mission.leg.n === entry.n && mission.leg.flight === entry.flight
      ? mission.leg : null;
    const floor = waiting?.coverageFloor ?? entry.coverageFloor;
    const coverage = waiting ? waiting.coverage : entry.coverage;
    const share = floor != null && (waiting || coverage !== undefined)
      ? [
          coverageLabel(coverage, floor),
          waiting && Number.isFinite(waiting.closest_m) ? `closest ${waiting.closest_m.toFixed(0)} m` : null,
          waiting && waiting.short_passes > 0 ? `short passes: ${waiting.short_passes}` : null,
        ].filter(Boolean).join(" · ")
      : null;
    const said = [
      entry.label,
      entry.lat != null && entry.lon != null
        ? `${entry.lat.toFixed(5)}, ${entry.lon.toFixed(5)}`
        : null,
      entry.observationRadiusMeters != null
        ? `ring ${entry.observationRadiusMeters.toFixed(0)} m`
        : null,
      entry.altitudeAglMeters != null
        ? `${entry.altitudeAglMeters.toFixed(0)} m AGL`
        : null,
    ].filter(Boolean);
    return (
      <div className="entry" data-who="model">
        <p className="entry-label">
          Waypoint {entry.n} {turn}
        </p>
        <p className="entry-prose">{said.join(" · ")}</p>
        {share ? (
          <p className="entry-coverage" data-state={coverageMet(coverage, floor as number) ? "held" : "short"}>
            {share}
          </p>
        ) : null}
      </div>
    );
  }

  // The leg the station flew for a go_to, and which frame answered it, in
  // the station's own words.
  if (entry.kind === "leg") {
    return (
      <div className="entry" data-who="model">
        <p className="entry-label">Leg {turn}</p>
        <p className="entry-prose">{entry.said}</p>
      </div>
    );
  }

  if (entry.kind === "view") {
    const channels = [
      entry.watchlist ? "watchlist" : null,
      entry.boost ? "boost" : null,
      entry.n_crops ? `${entry.n_crops} crops` : null,
    ].filter(Boolean);
    return (
      <div className="entry" data-who="model">
        <p className="entry-label">Model directed perception {turn}</p>
        <p className="entry-prose">{channels.join(" · ") || "no channel"}</p>
      </div>
    );
  }

  if (entry.kind === "transient_error") {
    return (
      <div className="entry" data-who="fault">
        <p className="entry-label">Reasoner unreachable, attempt {entry.attempt}</p>
        <pre className="entry-code">{entry.error}</pre>
      </div>
    );
  }

  if (entry.kind === "mission_end") {
    return (
      <div className="entry" data-who="model">
        <p className="entry-label">Turn ended</p>
        <p className="entry-prose">
          {entry.ended_by === "reply"
            ? `The model reported after ${entry.turns} turns.`
            : `Ended by ${entry.ended_by} after ${entry.turns} turns.`}
        </p>
      </div>
    );
  }

  // The mission's own verdict, written into the transcript once the run
  // ends. SR is the verdict; the sentence carries the distances.
  if (entry.kind === "scored") {
    const hit = Number(entry.metrics?.SR ?? 0) > 0;
    return (
      <div className="entry" data-who="operator">
        <p className="entry-label">Scored: {hit ? "SUCCESS" : "miss"}</p>
        <p className="entry-prose">{entry.said}</p>
      </div>
    );
  }

  const repair = REPAIRS[entry.kind];
  if (repair) {
    return (
      <div className="entry" data-who="fault">
        <p className="entry-label">Channel {turn}</p>
        <p className="entry-prose">{repair}</p>
      </div>
    );
  }

  // An unrecognised kind is shown rather than dropped. The loop's event set
  // grows, and a console that silently skipped what it did not know would
  // read as a mission with gaps in it.
  return (
    <div className="entry">
      <p className="entry-label">
        {entry.kind} {turn}
      </p>
      {entry.said ? <pre className="entry-code">{entry.said}</pre> : null}
    </div>
  );
}

/** The model's report, and the picture of what it committed to: the run's
 *  own observation with the committed thing lassoed on it. The flight line
 *  under it names the target. Nothing shows when no picture exists. */
function Report({ text, mission }: { text: string; mission: MissionState }) {
  const flights = mission.flights;
  const last = flights[flights.length - 1];
  const commit = commitLasso(mission);
  return (
    <div className="report">
      <p className="entry-label">Model report</p>
      <p className="report-text">{text}</p>
      {commit ? (
        <div className="entry-frames">
          <a href={commit.url} target="_blank" rel="noreferrer"
             title={`what was committed to · ${commit.target}`}>
            <img src={commit.url} alt={`what was committed to, ${commit.target}`} loading="lazy" />
          </a>
        </div>
      ) : null}
      <p className="flight-line">
        {flights.length === 0
          ? "No flights. Nothing was committed."
          : `${flights.length} flight${flights.length === 1 ? "" : "s"} · last target ${
              last?.target ?? "unknown"
            }${
              last?.resolved_xy
                ? ` at ${last.resolved_xy[0].toFixed(0)}, ${last.resolved_xy[1].toFixed(0)}`
                : ""
            }`}
      </p>
    </div>
  );
}
