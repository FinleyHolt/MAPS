# The capacity study

How many reports a person resolves correctly in a fixed session, alone and with MAPS. A
participant runs the whole visit on their own computer, on their own time, from one zip file:
no account, no terminal, no GPU.

| Step | Time |
|---|---|
| Download and unzip; the first start sets itself up | about 15 min |
| Read the [briefing](briefing.md) | 5 min |
| Practice: four reports that never count | 10 min |
| The first condition | 40 min |
| A break | 5 min |
| The second condition | 40 min |
| Send the results file | 2 min |

MAPS does not run on the participant's machine. It searched every report of the study once,
96 searches at a time, and the download holds that run: each answer it committed, its
reasoning and the pictures it looked at, and how long each search took. With MAPS, the console
releases each of its answers when that run says it landed, so the participant meets them in
the order and at the rate they arrived.

## For participants

Everything is in `READ ME FIRST.txt` inside the download ([its source](bundle/READ%20ME%20FIRST.txt)).
In short: double-click the launcher for your system, type your label in the page that opens,
and follow the page. The first start fetches [uv](https://docs.astral.sh/uv/), and with it
Python and the console's packages, into the unzipped folder and uv's cache; that needs the
internet once. A Mac or Windows computer with 8 GB of memory, 6 GB of free disk, Chrome or
Firefox and a mouse is enough.

## For the study lead

### Build and share the download

```sh
python -m baselines.human.study_package bundle --kit ~/datasets/calcity-campaign/study-kit --out MAPS-study.zip
```

The zip (about 2.5 GB) holds the kit below, the console's Python code (never the bench's
corpus and its targets), the briefing, and a launcher per system from [`bundle/`](bundle/):
`Start study (Mac).command`, `Start study (Windows).bat` and `start-study.sh`.

Upload **the zip file itself** to Google Drive and share it by link. Never share an unzipped
folder: Drive zips a folder on download and drops the Mac launcher's permission to run. Send
each participant the link and their label (`p01` to `p40`). The label is the only thing that
identifies them in any file.

To rehearse the whole visit quickly, start a launcher with `MAPS_STUDY_SESSION_S=300` in its
environment: each session then lasts five minutes. Rehearse in a separate unzip, since a
session's clock and answers stay in its folder.

The launchers are unsigned, so macOS and Windows warn on the first start; the READ ME walks
through both warnings.

### Collect and score

At the end the visit page offers the participant `maps-study-<p>.tgz`, which they email. It
holds `results/<p>/<condition>/answers.jsonl` (one line per answer: its source, `own_search`,
`confirm`, `replace` or `skip`, the point, and when the report was opened and answered, UTC)
and `session.json` (the session's start, length and list). Unpack it into a kit and check it:

```sh
tar xzf maps-study-p07.tgz -C ~/maps-study
python -m baselines.human.study_qc p07            # --misses lists every report they got wrong
```

`study_qc` scores each condition with `team_rows` into `~/maps-study/scored/p07-solo` and
`p07-team`, prints correct finds per session hour, answers by source and how the person met
MAPS's answers (wrong and confirmed, wrong and fixed, right and replaced with a miss), and
exits non-zero when a ledger fails a check: a report answered twice, an answer after the
session's end, a lost clock.

The scored root reads like any bench root (`benchmarks.citynav.analysis.run_report`). With
MAPS it also holds every MAPS answer released within the session that the person never
reached, scored as MAPS scored it. A skip scores as a miss.

### Run it from a clone

From the repository with `pip install -e '.[study]'`, the same visit is
`python -m baselines.human.study_server [p07] --kit <kit>`; naming a condition after the label
(`practice`, `first`, `second`, `solo`, `team`) opens that session at once.

### The kit

The kit is built from the bench's own files by `baselines.human.study_package` (`tiles`, then
`records`): the ortho as map tiles, MAPS's run as the verify screens read it, the OSM prior and
labelled objects, and the plan (`baselines.human.capacity plan` for `p01` to `p40`, halves of
equal MAPS SR, counterbalanced in groups of four, and `capacity practice`). The console's web
source is `baselines/human/console` (`npm ci && npm run build`); the kit carries the built
copy in `console/`.

Sessions run unsupervised, so nothing enforces the rule against other maps during a session;
the briefing states it and the answer times show a session that stalled.
