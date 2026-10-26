# The capacity study: run it yourself

How many reports a person resolves correctly in a fixed session, alone and with MAPS. The
whole visit runs on one computer, with no GPU and no network once the files are
downloaded, in under two hours:

| Step | Time |
|---|---|
| Setup: clone, install, download | about 15 min |
| Read the [briefing](briefing.md) | 5 min |
| Practice: four reports that never count | 10 min |
| The first condition | 40 min |
| A break | 5 min |
| The second condition | 40 min |
| Send your results | 2 min |

MAPS does not run on your machine. It searched every report of the study once, 96 searches
at a time, and the download holds that run: each answer it committed, its reasoning and the
pictures it looked at, and how long each search took. With MAPS, the console releases each
of its answers when that run says it landed, so you meet them in the order and at the rate
they arrived.

## What you need

- Linux or macOS, 8 GB of memory, 6 GB of free disk.
- Python 3.12 or newer, and `git`.
- A desktop browser (Chrome or Firefox) in a window at least 1600 by 1000 pixels, and a mouse.
- Your participant label (`p01` to `p40`) from the study lead. The label is the only thing
  that identifies you in any file; never use your name.
- Access to the study's files on Hugging Face: sign in at
  [huggingface.co](https://huggingface.co), open
  [FinleyHolt/maps-capacity-study](https://huggingface.co/datasets/FinleyHolt/maps-capacity-study)
  and request access. The study lead approves it.

## Setup, once

```sh
git clone https://github.com/FinleyHolt/MAPS.git && cd MAPS
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[study]'

hf auth login                       # a read token from huggingface.co/settings/tokens
hf download FinleyHolt/maps-capacity-study maps-study.tar --repo-type dataset --local-dir ~
tar xf ~/maps-study.tar -C ~ && rm ~/maps-study.tar
```

`~/maps-study` now holds the imagery, MAPS's run and the plan. Every command below runs
from the `MAPS` folder with the environment active (`. .venv/bin/activate`). To keep the
files elsewhere, untar them there and add `--kit <folder>` to each command.

## The visit

Read [briefing.md](briefing.md) first. Then, with your own label in place of `p07`:

```sh
python -m baselines.human.study_server p07 practice
```

Open <http://127.0.0.1:8788/> in the browser. Work the four practice reports and try every
tool the briefing lists. Stop the console with Ctrl-C in the terminal when done.

```sh
python -m baselines.human.study_server p07 first
```

The plan decides whether you start alone or with MAPS; the console's top bar says which.
**The 40 minutes start when this command starts**, so be seated and ready. Reload the
browser page. The console ends the session itself and refuses further answers. Then Ctrl-C.

Take the break, then:

```sh
python -m baselines.human.study_server p07 second
```

During a session, use only the console: no other maps, websites or phones. If the console
or the browser stops, run the same command again. The session resumes on its own clock,
with the lost time counted against it. Never delete a session's folder to start it again.

## Send your results

```sh
tar czf ~/maps-study-p07.tgz -C ~/maps-study results/p07
```

Email `~/maps-study-p07.tgz` to the study lead. It holds your answers and when you gave
them, and nothing else about you.

## For the study lead

Each session writes `results/<p>/<condition>/answers.jsonl` (one line per answer: its
source, `own_search`, `confirm`, `replace` or `skip`, the point, and when the report was
opened and answered, UTC) and `session.json` (the session's start, length and list).

Score and check a participant's two conditions against the download's MAPS run:

```sh
python -m baselines.human.study_qc p07            # --misses lists every report they got wrong
```

It scores each condition with `team_rows` into `~/maps-study/scored/p07-solo` and `p07-team`,
prints correct finds per session hour, answers by source and how the person met MAPS's answers
(wrong and confirmed, wrong and fixed, right and replaced with a miss), and exits non-zero when a
ledger fails a check: a report answered twice, an answer after the session's end, a lost clock.

The scored root reads like any bench root (`benchmarks.citynav.analysis.run_report`). With
MAPS it also holds every MAPS answer released within the session that the person never
reached, scored as MAPS scored it. A skip scores as a miss.

The download is built from the bench's own files by `baselines.human.study_package`: the
ortho as map tiles, MAPS's run as the verify screens read it, the OSM prior and labelled
objects, and the plan (`baselines.human.capacity plan` for `p01` to `p40`, halves of equal
MAPS SR, counterbalanced in groups of four, and `capacity practice`). The console's web
source is `baselines/human/console` (`npm ci && npm run build`); the download carries the
built copy.

The repository ships the corpus with its targets (`benchmarks/calcity/data`), so a
participant who went looking could find answers. Run sessions where that would be noticed,
or accept it as the price of a study anyone can rerun.
