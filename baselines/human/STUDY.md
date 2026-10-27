# The capacity study

How many reports a person resolves correctly in a fixed session, alone and with MAPS. Between
subjects: each participant works one condition, dealt half and half by the plan, since
everything a person does in the study is practice that would carry into a second condition. A
participant runs the whole visit on their own computer, on their own time, from one zip file:
no account, no GPU.

| Step | Time |
|---|---|
| Download and unzip; the first start sets itself up | about 15 min |
| Read the [briefing](briefing.md) | 5 min |
| Practice: four reports that never count | 10 min |
| The session, in the participant's one condition | 40 min |
| Send the results file | 2 min |

MAPS does not run on the participant's machine. It searched every report of the study once,
96 searches at a time, and the download holds that run: each answer it committed, its
reply explaining it, its reasoning and the pictures it looked at, and how long each search took. With MAPS, the console
releases each of its answers when that run says it landed, so the participant meets them in
the order and at the rate they arrived.

The briefing awaits the owner's approval.

## For participants

Everything is in `READ ME FIRST.txt` inside the download ([its source](bundle/READ%20ME%20FIRST.txt)).
In short: unzip, start the launcher for your system, type your label in the page that
opens, and follow the page. The first start fetches [uv](https://docs.astral.sh/uv/) and the console's
packages into the unzipped folder (uv keeps the Python it fetches in the user's own folder);
that needs the internet once.
A Mac with macOS 12 or later (macOS 11 or later on Apple silicon), a computer with 64-bit
Windows 10 or 11, or a 64-bit Linux computer (started from a terminal), with 8 GB of memory,
6 GB of free disk, Chrome or Firefox and a mouse, is enough.

## For the study lead

The kit is one folder, here `$KIT`; every command below names it with `--kit`.

```sh
KIT=~/datasets/capacity-study/kit
```

### Build and share the download

The kit carries the console's built web code. After any change under `console/`, rebuild it
into the kit:

```sh
cd baselines/human/console && npm ci && npm run build && rm -rf "$KIT/console" && cp -r dist "$KIT/console"
```

Then build the zip outside the repository, with the address participants write to:

```sh
python -m baselines.human.study_package bundle --kit "$KIT" --out ~/MAPS-study.zip \
    --contact "frholt@stanford.edu and rlutz@unc.edu"
```

It refuses a kit without `console/index.html`, `plan/plan.json`, `calcity_ortho.mbtiles` or
`maps_root/episodes.jsonl`, and it follows symlinked folders in the kit.

The zip (about 2.5 GB) holds the kit, the console's Python code (never the bench's corpus and
its targets), the briefing, `kit/contact.txt`, and the launchers from [`bundle/`](bundle/):
`Start study (Mac).command`, `Start study (Windows).bat` and `Start study (Linux).sh`. The Mac and
Linux ones run `app/run.sh`; the Windows one does the same its own way. The READ ME and the visit
page's last step give the `--contact` text as written, so it reads as the end of "Email that file
to ...".

Upload **the zip file itself** to Google Drive and share it by link. Never share an unzipped
folder: Drive zips a folder on download and drops the Mac launcher's permission to run. Send
each participant the link and their label (`p01` to `p14`). The label is the only thing that
identifies them in any file. An invitation:

> Thank you for taking part in the search study. Your participant label is **p07**; it is the
> only thing that identifies you, so never type your name in the study.
>
> Download the study here: <link>. It is one zip file of about 2.5 GB. Google Drive says it is
> too large to preview and cannot scan it for viruses; both are normal, so press "Download
> anyway".
>
> Unzip it before anything else. On a Mac, double-click the zip. On Windows, right-click the
> zip, choose Properties, tick "Unblock" if it is there and press OK; then right-click it
> again and choose "Extract All". On Linux, choose "Extract Here". Open the new MAPS-study folder and follow
> "READ ME FIRST.txt". The Mac or Windows will warn about the launcher the first time; the
> READ ME says which buttons to press.
>
> You need a Mac with macOS 12 or later (11 on Apple silicon), a computer with 64-bit Windows
> 10 or 11, or a 64-bit Linux computer, 8 GB of memory, 6 GB of free disk space, an internet
> connection for the first start, a mouse, Chrome or Firefox and about an hour and a quarter. Questions about the setup:
> <email>.

The launchers are unsigned, so macOS and Windows warn on the first start; the READ ME walks
through both warnings.

### A wrong label, or one copy for a second person

The label stays changeable on the visit page until session 1 starts. After that the unzipped
folder belongs to that label. To correct it, or to hand the same copy to someone else, have
the participant save their results file first, then move `MAPS-study/kit/results/<run id>` out
of the folder and start the study again: the page asks for a label as on the first start.

### Rehearse

To rehearse the whole visit quickly, start the launcher from a terminal with
`MAPS_STUDY_SESSION_S` set, for example `MAPS_STUDY_SESSION_S=600 sh MAPS-study/app/run.sh`:
each session then lasts ten minutes. The variable sets a session's length when the session
first starts; a resumed session keeps the length its `session.json` recorded. Give a session
long enough to meet MAPS's answers on the verify screen, since the console releases them on
MAPS's own timing. Rehearse in a separate unzip, since a session's clock and answers stay in
its folder.

### Collect and score

The first label a participant types mints the visit's run id: the label, a dash and eight
hex characters (`p07-3f9a2c1e`). The label picks the plan's lists; the run id names the
results folder and file, so two visits under one label never collide. The visit page shows it
beside the label.

At the end the visit page offers `maps-study-<run id>.json`, one JSON document:
`"format": "maps-capacity-results"`, `"version": 1`, the run id, the label, when it was saved
(UTC), and under `"sessions"` each of `practice`, `solo` and `team` that started, with its
`"session"` (the `session.json` object: the run id and label, the session's start, length and
list) and its `"answers"` (the `answers.jsonl` rows: the run id and label, the source,
`own_search`, `confirm`, `replace` or `skip`, the point, and when the report was opened and
answered, UTC), plus `"unreadable_lines"` when the ledger held lines that do not parse. No name,
computer account or path is in it.

1. The participant emails `maps-study-<run id>.json` to frholt@stanford.edu and rlutz@unc.edu,
   so each file reaches both inboxes. The lead collects from their own inbox; Ryan's copies are
   the backup: a file missing from one inbox is taken from the other.
2. Save every attachment, unrenamed, into one inbox folder, here `~/datasets/capacity-study/inbox`.
   Saving both copies of a file there is safe: the second is noted as already collected.
3. Collect the inbox into the kit and the paper's figure:

   ```sh
   python -m baselines.human.study_collect --kit "$KIT" --inbox ~/datasets/capacity-study/inbox \
       --study-json ~/Github/MAPS-SciRob/figures/capacity/study.json
   ```

   For each `maps-study-*.json` not collected yet it checks the file is a results file of
   one run of a label on the plan, with only the sessions above, and writes each session's
   `session.json` and `answers.jsonl` into `$KIT/results/<run id>/<session>/` as the visit
   wrote them, without replacing a file. A `maps-study-*.zip`, the results file of an earlier
   version of the visit, must hold `results/<run id>/` and nothing else, and unpacks the same
   way. It then runs `study_qc`'s checks and scoring into `$KIT/scored/<run id>-solo` and
   `<run id>-team`, records the file (name, SHA-256, run id, label, time, checks) in
   `$KIT/collected.jsonl`, and enters every run in the study.json as
   `participants["<run id>"]` with its `"qc"` (`"pass"`, or `"fail"` with what failed), leaving
   its other entries as they are. The figure draws the passing runs. It ends with a table of
   every collected run: label, conditions, correct out of answered per condition, and QC.
   `$KIT/actions.csv` lists every answer of every collected run, one row each (run id, label,
   session, list position, report, source, point, opened, answered, seconds, whether MAPS's
   answer was ready), rewritten from `$KIT/results` on every pass. It is for eyeballing; the
   JSON results files and `$KIT/results` are the record.
   A second copy of a file (the same SHA-256, or the same sessions saved again) is noted as
   already collected and recorded once. Running it again over the same inbox changes nothing, so run it whenever mail arrives.
4. Refresh the figure from the SciRob repository:
   `~/Github/MAPS/.venv/bin/python figures/capacity/render_capacity.py --refresh`, then the same
   command without `--refresh` to draw it.
   The refresh crops the example report's imagery from the archived ortho
   (`~/datasets/calcity-campaign/data/rgbd_upstream`), which lives on hamming; run it where
   that folder is.
5. Back up the inbox, `$KIT/results`, `$KIT/scored` and `$KIT/collected.jsonl` together. They
   name participants by label and run id only.

`study_collect` exits non-zero and says why when:

- a file is REFUSED for its contents (a JSON file that is not a results file, of another
  version, with a session other than `practice`, `solo` or `team`, or a session without its
  clock and answers; a zip entry outside `results/<run id>/`, an unknown file, a link, two
  runs, a damaged zip; a label off the plan): nothing of it is written. Ask the participant to
  send the file the visit page saved, unopened and unrenamed.
- a file is REFUSED because its run id is already in the kit with other files: the kit keeps
  the first. Compare the two by hand and keep the complete one; to replace the kit's copy,
  move `$KIT/results/<run id>` out and collect again.
- a run fails QC: it is recorded, scored and entered in study.json flagged `"qc": "fail"` with
  what failed, and the figure leaves it out. Read it with
  `python -m baselines.human.study_qc <run id> --kit "$KIT"` (`--misses` lists every report
  they got wrong). To let a flagged run stand, set its `"qc"` to `"pass"` by hand.

`study_qc` prints correct finds per session hour, answers by source and how the person met
MAPS's answers (wrong and confirmed, wrong and fixed, right and replaced with a miss), and
exits non-zero when a ledger fails a check: a report answered twice, an answer after the
session's end, an answer to a report not on the list, an unreadable line, a clock moved back
(an answer before the session's start or before the previous answer), a lost clock, a session
not 40 minutes long (a rehearsal?), or a missing session. It takes a run id, or a bare label
with one visit in the kit; a label with several lists their run ids. It refuses a label that is
not on the kit's plan (`P07` is `p07`). A results folder named by a bare label (`results/p07`)
reads as the run id `p07`.

The scored root reads like any bench root (`benchmarks.citynav.analysis.run_report`). With
MAPS it also holds every MAPS answer released within the session that the person never
reached, scored as MAPS scored it. Alone, a skip scores as a miss; with MAPS, a skip keeps
MAPS's answer when it landed within the session, and is a miss otherwise.

### Run it from a clone

From the repository with `pip install -e '.[study]'`, the same visit is
`python -m baselines.human.study_server --kit "$KIT"`.

### The kit

The kit is built from the bench's own files by `baselines.human.study_package` (`tiles`, then
`records`): the ortho as map tiles, MAPS's run as the verify screens read it (`replies` adds
MAPS's final replies to a kit from a copy of the run's transcripts alone), the OSM prior and
labelled objects, and the plan (`baselines.human.capacity plan` for `p01` to `p14`: seven alone
and seven with MAPS, dealt by the seed, each working every report in their own shuffled order;
and `capacity practice`). The console's web
source is `baselines/human/console`; the kit carries the built copy in `console/`.

Sessions run unsupervised, so nothing enforces the rule against other maps during a session;
the briefing states it and the answer times show a session that stalled.
