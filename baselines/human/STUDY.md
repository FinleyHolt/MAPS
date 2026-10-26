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

The briefing awaits the owner's approval.

## For participants

Everything is in `READ ME FIRST.txt` inside the download ([its source](bundle/READ%20ME%20FIRST.txt)).
In short: unzip, double-click the launcher for your system, type your label in the page that
opens, and follow the page. The first start fetches [uv](https://docs.astral.sh/uv/) and the console's
packages into the unzipped folder (uv keeps the Python it fetches in the user's own folder);
that needs the internet once.
A Mac with macOS 12 or later (macOS 11 or later on Apple silicon) or a computer with 64-bit
Windows 10 or 11, with 8 GB of memory, 6 GB of free disk, Chrome or Firefox and a mouse, is
enough.

## For the study lead

The kit is one folder, here `$KIT`; every command below names it with `--kit`.

```sh
KIT=~/datasets/calcity-campaign/study-kit
```

### Build and share the download

The kit carries the console's built web code. After any change under `console/`, rebuild it
into the kit:

```sh
cd baselines/human/console && npm ci && npm run build && rm -rf "$KIT/console" && cp -r dist "$KIT/console"
```

Then build the zip outside the repository, with the address participants write to:

```sh
python -m baselines.human.study_package bundle --kit "$KIT" --out ~/MAPS-study.zip --contact <email>
```

It refuses a kit without `console/index.html`, `plan/plan.json`, `calcity_ortho.mbtiles` or
`maps_root/episodes.jsonl`, and it follows symlinked folders in the kit.

The zip (about 2.5 GB) holds the kit, the console's Python code (never the bench's corpus and
its targets), the briefing, `kit/contact.txt`, and the launchers from [`bundle/`](bundle/):
`Start study (Mac).command` and `Start study (Windows).bat`. Inside it, `app/run.sh` starts
the study on Linux too. The READ ME and the visit page's last step give the `--contact`
address.

Upload **the zip file itself** to Google Drive and share it by link. Never share an unzipped
folder: Drive zips a folder on download and drops the Mac launcher's permission to run. Send
each participant the link and their label (`p01` to `p40`). The label is the only thing that
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
> again and choose "Extract All". Open the new MAPS-study folder and follow
> "READ ME FIRST.txt". The Mac or Windows will warn about the launcher the first time; the
> READ ME says which buttons to press.
>
> You need a Mac with macOS 12 or later (11 on Apple silicon) or a computer with 64-bit
> Windows 10 or 11, 8 GB of memory, 6 GB of free disk space, an internet connection for the
> first start, a mouse, Chrome or Firefox and about two hours. Questions about the setup:
> <email>.

The launchers are unsigned, so macOS and Windows warn on the first start; the READ ME walks
through both warnings.

### A wrong label, or one copy for a second person

The label stays changeable on the visit page until session 1 starts. After that the unzipped
folder belongs to that label. To correct it, or to hand the same copy to someone else, have
the participant save their results file first, then move `MAPS-study/kit/results/<label>` out
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

At the end the visit page offers the participant `maps-study-<p>.zip`, which they email. It
holds `results/<p>/<condition>/answers.jsonl` (one line per answer: its source, `own_search`,
`confirm`, `replace` or `skip`, the point, and when the report was opened and answered, UTC)
and `session.json` (the session's start, length and list). Unpack it into the kit, never over
a file already there, and check it:

```sh
unzip -n maps-study-p07.zip -d "$KIT"
python -m baselines.human.study_qc p07 --kit "$KIT"      # --misses lists every report they got wrong
```

`study_qc` scores each condition with `team_rows` into `$KIT/scored/p07-solo` and `p07-team`,
prints correct finds per session hour, answers by source and how the person met MAPS's
answers (wrong and confirmed, wrong and fixed, right and replaced with a miss), and exits
non-zero when a ledger fails a check: a report answered twice, an answer after the session's
end, an answer to a report not on the list, an unreadable line, a clock moved back (an answer
before the session's start or before the previous answer), a lost clock, a session not 40
minutes long (a rehearsal?). It also exits non-zero when the label lacks either session. It
refuses a label that is not on the kit's plan (`P07` is `p07`).

The scored root reads like any bench root (`benchmarks.citynav.analysis.run_report`). With
MAPS it also holds every MAPS answer released within the session that the person never
reached, scored as MAPS scored it. A skip scores as a miss.

### Run it from a clone

From the repository with `pip install -e '.[study]'`, the same visit is
`python -m baselines.human.study_server --kit "$KIT"`.

### The kit

The kit is built from the bench's own files by `baselines.human.study_package` (`tiles`, then
`records`): the ortho as map tiles, MAPS's run as the verify screens read it, the OSM prior and
labelled objects, and the plan (`baselines.human.capacity plan` for `p01` to `p40`, halves of
equal MAPS SR, counterbalanced in groups of four, and `capacity practice`). The console's web
source is `baselines/human/console`; the kit carries the built copy in `console/`.

Sessions run unsupervised, so nothing enforces the rule against other maps during a session;
the briefing states it and the answer times show a session that stalled.
