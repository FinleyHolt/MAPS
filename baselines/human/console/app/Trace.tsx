/** MAPS's search on a verified report, step by step: its reasoning, the block it ran, what
 * came back, and the frames it looked at (`judge/frames/<sha>.jpg`, as the run saved them). */

export interface Turn { reasoning?: string; said?: string; printed?: string; frames?: string[] }

export function Trace({ instruction, turns }: { instruction: string; turns: Turn[] }) {
  return (
    <div className="transcript-inner">
      <div className="entry" data-who="operator">
        <p className="entry-label">Report</p>
        <p className="entry-prose">{instruction}</p>
      </div>
      {turns.map((t, i) => (
        <div className="entry" data-who="model" key={i}>
          <p className="entry-label">MAPS <span className="entry-turn">step {i + 1}</span></p>
          {t.reasoning ? <p className="entry-reasoning">{t.reasoning}</p> : null}
          {t.said ? <pre className="entry-code">{t.said}</pre> : null}
          {t.frames?.length ? (
            <div className="entry-frames">
              {t.frames.map((sha) => (
                <a key={sha} href={`judge/frames/${sha}.jpg`} target="_blank" rel="noreferrer">
                  <img src={`judge/frames/${sha}.jpg`} alt={`what MAPS saw, ${sha.slice(0, 8)}`} loading="lazy" />
                </a>
              ))}
            </div>
          ) : null}
          {t.printed ? <pre className="entry-output">{t.printed}</pre> : null}
        </div>
      ))}
    </div>
  );
}
