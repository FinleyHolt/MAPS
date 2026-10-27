/** MAPS's search on a verified report, step by step: its reasoning (through ``link``, which
 * links the places it names), the block it ran, what came back, and the frames it looked at
 * (`judge/frames/<sha>.jpg`, as the run saved them). With ``onPick``, a step's code is a button:
 * the console shows on the map what that step used, ``picked`` the step it shows. */

import type { ReactNode } from "react";

import type { Refs } from "@/app/fed";

export interface Turn { reasoning?: string; said?: string; printed?: string; frames?: string[]; refs?: Refs }

export function Trace({ instruction, turns, link, picked, onPick }: {
  instruction: string; turns: Turn[]; link: (text: string) => ReactNode; picked?: number | null; onPick?: (i: number) => void;
}) {
  return (
    <div className="transcript-inner">
      <div className="entry" data-who="operator">
        <p className="entry-label">Report</p>
        <p className="entry-prose">{instruction}</p>
      </div>
      {turns.map((t, i) => (
        <div className="entry" data-who="model" data-on={picked === i} key={i}>
          <p className="entry-label">MAPS <span className="entry-turn">step {i + 1}{picked === i ? " · shown on the map" : ""}</span></p>
          {t.reasoning ? <p className="entry-reasoning">{link(t.reasoning)}</p> : null}
          {t.said ? (onPick ? (
            <button type="button" className="entry-code entry-call" aria-pressed={picked === i} title="show on the map what this step used"
                    onClick={() => onPick(i)}>{t.said}</button>
          ) : <pre className="entry-code">{t.said}</pre>) : null}
          {t.frames?.length ? (
            <div className="entry-frames">
              {t.frames.map((sha) => (
                <a key={sha} href={`judge/frames/${sha}.jpg`} target="_blank" rel="noreferrer">
                  <img src={`judge/frames/${sha}.jpg`} alt={`what MAPS saw, ${sha.slice(0, 8)}`} loading="lazy" />
                </a>
              ))}
            </div>
          ) : null}
          {t.printed ? (onPick ? (
            <button type="button" className="entry-output entry-call" aria-pressed={picked === i} title="show on the map what this step used"
                    onClick={() => onPick(i)}>{t.printed}</button>
          ) : <pre className="entry-output">{t.printed}</pre>) : null}
        </div>
      ))}
    </div>
  );
}
