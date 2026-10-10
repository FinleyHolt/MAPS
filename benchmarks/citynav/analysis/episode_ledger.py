#!/usr/bin/env python3
"""Per-episode outcomes indexed by commit: rows are episodes, columns are runs.

WHY this exists: a paired A/B needs BOTH arms, so re-measuring a baseline that
has already been run costs a second fleet. The 2026-07-26 refactor gate spent
an 8-GPU arm for ~11 h regenerating ``8fc0fd8``'s numbers purely because no
per-episode record of them survived. A cell here -- "episode X under commit Y
scored Z" -- is an immutable historical fact, so storing it is not the cached
mutable state the tenets forbid: there is no later state of affairs it can go
stale against.

Per-EPISODE, not summary SR, is the load-bearing choice. The reasoner-arming
regression read as 0.35 -> 0.19 in aggregate, which is indistinguishable from
noise plus a bad week. Per episode it read as nine flips to FAIL and zero to
SUCCESS, which is a mechanism.

What the ledger may NOT do is let a column be compared to anything. Each column
carries its provenance and comparison goes through
``ab_report.comparability_issues``, whose ``fatal`` class (split, tile bake) is
unwaivable -- the retired ``citynav_2m`` bake scored the same episodes 0.351
where ``citynav_2m_osmv2`` scores 0.169.

Stored columns decay: identical code and tiles still ran against whatever model
weights and vLLM build were served that day. So a stored column is sound for
triage, trend, and finding chronically-hard episodes, while a headline claim
wants a concurrent arm. Revalidating a stored column needs no new verb: re-run
a sample of it, ``add`` that run, and ``pair`` the two columns of the same
commit. Agreement means the stored column still reproduces and can be trusted
as the A side; disagreement means the environment moved under it and the full
arm has to be re-run.

``pair`` compares TWO columns, which is one draw against one draw. When a
family has been flown several times, ``group`` pools the replicates per episode
instead: each episode contributes its MEAN outcome in each group, so the
per-run draw noise averages down rather than being compared.

The load-bearing number is a LABEL permutation, and it is reported twice
because the honest p depends on something no data can tell you: whether the
grouping was fixed before the numbers were seen. Shuffling run labels within
each episode gives the null "these columns are exchangeable"; scoring the
observed gap against the SAME split's null gap answers the pre-specified case,
and against the LARGEST gap any split reaches in that draw answers the chosen-
after case. On the eight sentence arms those read 0.030 and 0.415. The bracket
IS the result.

The episode bootstrap is printed but is not a test of the grouping: it
resamples episodes with the columns held fixed, so it contains none of the
run-to-run draw variance a grammar claim rests on. Neither is the descriptive
count of same-sized splits -- ranking columns by SR and cutting the top k tops
that list by construction, which is exactly how this verb was first misread.

Usage:
  python -m benchmarks.citynav.analysis.episode_ledger add <run_root> [...]
  python -m benchmarks.citynav.analysis.episode_ledger table [--ledger P]
  python -m benchmarks.citynav.analysis.episode_ledger pair <col_a> <col_b>
  python -m benchmarks.citynav.analysis.episode_ledger group <a,a,..> <b,b,..>
  python -m benchmarks.citynav.analysis.episode_ledger null
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
from pathlib import Path

from benchmarks.citynav.analysis.ab_report import (
    _bootstrap_ci,
    _run_provenance,
    ab_stats,
    comparability_issues,
    render,
)
from benchmarks.citynav.analysis.run_report import load_run

# In the TREE, not output/. The whole point is never paying for a baseline
# twice, and output/ is gitignored and machine-local -- the 8fc0fd8 column below
# cost an 8-GPU arm for ~11 h and would not survive a clean checkout there.
# Append-only JSONL, so a merge conflict resolves by keeping both sides.
DEFAULT_LEDGER = Path(__file__).resolve().parent / "ledger.jsonl"

# The provenance a column is USELESS without. Kept explicit so a column written
# by an older runner is stored with nulls rather than silently looking valid.
PROV_KEYS = (
    "git_commit",
    "config_sha256",
    "served_model",
    "served_build",
    "look_tool",
    "split",
    "seed",
    "search_env_flags",
    # Content identity of the tile bake. The path rides in search_env_flags;
    # this says WHICH bake was at that path, which the Cal City name cannot.
    "bake_sha256",
)


# The CLI's verbs, named once. The docstring above and ``main``'s choices are
# checked against THIS rather than against a hand-copied list: the copy went
# stale the moment ``null`` was added, and a test asserting a doc matches the
# code is worthless if it is really asserting the doc matches a third copy.
# (It had gone stale a second way: a shadowing redefinition next to ``main``
# was the binding Python actually read, making this one dead — the exact
# drift the paragraph above warns about, one copy over.)
VERBS = ("add", "table", "pair", "group", "null")


def column_id(run_root: Path, prov: dict) -> str:
    """``<short-sha>:<run-dir-name>`` -- sha alone is not unique (a commit can be
    run many times), and the dir name alone loses the code identity."""
    sha = str(prov.get("git_commit") or "nosha")[:7]
    return f"{sha}:{run_root.name}"


def _precise_cell(m: dict) -> int | None:
    """This episode's SHAPE verdict, with ``None`` kept as ``None``.

    ``None`` means NOT MEASURED — the row is ring-scored, the episode carries
    no shape yet, or the run predates shape scoring. Collapsing it to 0 would
    count "not scored on a shape" as a failure, and since coverage is exactly
    what this cell is supposed to move with, that would hide the only signal
    worth pairing on.

    ``SR`` gated on ``precise_target``, not a second key: the verdict under a
    shape mask IS the row's ``SR``. The cell name is stored bytes in an
    append-only file, so it keeps the name 50,788 stored cells were written
    under.
    """
    if m.get("precise_target") is None:
        return None
    value = m.get("SR")
    return None if value is None else int(bool(value))


def legacy_criterion(prov: dict) -> str:
    """The criterion of a column stored before the rows carried the stamp.

    Derived ONCE, from the split the column already records — a Cal City
    column is a shape column, everything else is a ring column. This is not
    the coverage heuristic it replaces: the split is a FATAL comparability
    key (``ab_report.comparability_issues``), so it is recorded ground truth
    rather than a property of how many rows happened to resolve a shape.
    """
    from benchmarks.calcity.paths import SPLIT_FILES as _CALCITY_SPLITS

    return "shape" if prov.get("split") in _CALCITY_SPLITS else "ring"


def read_run(
    run_root: Path,
) -> tuple[str, dict, dict[str, int], dict[str, int | None], str]:
    """``(column_id, provenance, {episode: SR}, {episode: shape verdict},
    criterion)``.

    Every outcome comes from ONE canonical read, so the precise column always
    covers exactly the episodes the SR column does.
    """
    from benchmarks.citynav.analysis.run_report import run_criterion

    scored, _ = load_run(run_root)
    prov = _run_provenance(run_root)
    outcomes = {eid: int(bool(m.get("SR"))) for eid, m in scored.items()}
    precise = {eid: _precise_cell(m) for eid, m in scored.items()}
    stamped = run_criterion(list(scored.values()))
    return (
        column_id(run_root, prov),
        {k: prov.get(k) for k in PROV_KEYS},
        outcomes,
        precise,
        stamped or legacy_criterion(prov),
    )


def append(ledger: Path, run_root: Path) -> tuple[str, int]:
    """Append one column: a ``kind=column`` header carrying the provenance, then
    one thin ``kind=cell`` row per episode.

    Provenance belongs to the COLUMN, so it is written once. Repeating it on
    every cell cost ~425 bytes a row against ~80, which is 22 MB of duplicated
    metadata at 5,311 episodes across ten commits.

    Append-only: a cell is never edited. Re-adding a column_id appends a fresh
    header and cells that the reader takes last-wins, so a re-scored run
    corrects itself without truncating the file.
    """
    col, prov, outcomes, precise, criterion = read_run(run_root)
    header = {
        "kind": "column", "column": col,
        # WHICH MASK THIS COLUMN'S CELLS ARE VERDICTS UNDER. A column-level
        # fact, so it is written once, beside the provenance. Without it the
        # reader is back to inferring the criterion from how many cells
        # happen to carry a shape verdict -- which is how a Cal City column
        # came to be paired on the 20 m disc under a report that printed the
        # shape number.
        "criterion": criterion,
        "run_root": str(run_root), "provenance": prov,
    }
    # The run's receipt, pinned by sha: the column stays auditable after
    # the run tree is archived (the sha names WHICH manifest scored it,
    # even when the file itself has moved to a *_archive sibling).
    manifest = run_root / "run_manifest.json"
    if manifest.is_file():
        from benchmarks import sha256_file

        header["manifest_sha256"] = sha256_file(manifest)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as fh:
        fh.write(json.dumps(header, sort_keys=True) + "\n")
        for eid, sr in sorted(outcomes.items()):
            fh.write(json.dumps(
                {
                    "kind": "cell", "column": col, "episode_id": eid, "sr": sr,
                    # Written even when null, so a cell says "measured, no
                    # shape" rather than leaving a reader to guess whether the
                    # key is absent because the run predates the metric.
                    "sr_precise": precise.get(eid),
                },
                sort_keys=True) + "\n")
    return col, len(outcomes)


def load(
    ledger: Path,
) -> tuple[
    dict[str, dict[str, int]],
    dict[str, dict],
    dict[str, dict[str, int | None]],
    dict[str, str],
]:
    """``({column: {episode: sr}}, {column: provenance},
    {column: {episode: sr_precise}}, {column: criterion})``, last record wins.

    The precise map is returned SEPARATELY rather than folded into the cells,
    so every existing reader of the SR column keeps reading plain ints and a
    ``None`` in the precise map cannot be mistaken for a zero on the way past.

    A column stored before the header carried ``criterion`` gets it derived
    from its split (:func:`legacy_criterion`). STORED CELLS ARE NEVER
    REWRITTEN — the file is append-only and its two cell names are bytes on
    disk, not identifiers with callers. The header says which of them is the
    verdict.
    """
    cols: dict[str, dict[str, int]] = {}
    provs: dict[str, dict] = {}
    precise: dict[str, dict[str, int | None]] = {}
    crits: dict[str, str] = {}
    if not ledger.exists():
        return cols, provs, precise, crits
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        col = r.get("column")
        if not isinstance(col, str):
            continue
        if r.get("kind") == "column":
            provs[col] = r.get("provenance") or {}
            crits[col] = r.get("criterion") or legacy_criterion(provs[col])
            cols.setdefault(col, {})
            precise.setdefault(col, {})
            continue
        eid = r.get("episode_id")
        if isinstance(eid, str):
            cols.setdefault(col, {})[eid] = int(r.get("sr") or 0)
            raw = r.get("sr_precise")
            precise.setdefault(col, {})[eid] = None if raw is None else int(raw)
    return cols, provs, precise, crits


def verdict_cells(
    col: str,
    cols: dict[str, dict[str, int]],
    precise: dict[str, dict[str, int | None]],
    crits: dict[str, str],
) -> dict[str, int | None]:
    """This column's cells under ITS OWN criterion.

    A ring column's verdict is the ``sr`` cell; a shape column's is the
    ``sr_precise`` cell, which is where a shape verdict has always been
    stored. Every verb reads through here, so ``pair`` cannot be
    criterion-aware while ``group``, ``null`` and ``table`` quietly read the
    ring cell of a Cal City column — which is what they did.
    """
    if crits.get(col) == "shape":
        return dict(precise.get(col, {}))
    return dict(cols.get(col, {}))


def refuse_unscorable(col: str, cells: dict[str, int | None]) -> dict[str, int]:
    """A shape column with no shape verdict answers nothing; say so.

    Three stored Cal City columns predate shape scoring and carry only ring
    cells. Reading them anyway is the trap this whole change closes: the
    number that came out was the 20 m disc while every report beside it said
    the shape.
    """
    scored = {e: v for e, v in cells.items() if v is not None}
    if not scored:
        raise SystemExit(
            f"column {col!r} carries no verdict under its own criterion -- it "
            "was stored before this bench was scored on the target's shape. "
            "Re-run it into a NEW column; stored cells are never rewritten."
        )
    return scored


def _same_stem(col_a: str, col_b: str) -> bool:
    """Do two column names differ only in a trailing seed/rep token?

    Provenance cannot see every treatment: ``sxs_mtp0_750`` and
    ``sxs_mtp3_750`` share a commit, a config and a model, and are a
    speculative-decoding A/B. Nothing recorded distinguishes them. The name
    does, so a same-stem pair is flagged as the STRONGEST null evidence and
    the rest are shown with that caveat rather than silently trusted.
    """
    def stem(name: str) -> str:
        tail = name.split(":")[-1]
        return re.sub(r"[_-]?(k|r|rep|seed)?\d+$", "", tail)
    return stem(col_a) == stem(col_b) and col_a != col_b


def null_reference(ledger: Path) -> list[dict]:
    """What IDENTICAL code produced, from every same-commit column pair stored.

    A p-value read alone invites a story. These pairings differ only in seed,
    so every delta they show is noise by construction — the scale any real
    delta has to be read against.

    Measured on this ledger at the time of writing: two of six same-commit
    pairings would read "significant" at p<0.05, one of them at p=0.007 with a
    +9.5 pt delta. A -4.7 pt result at p=0.219 was being treated as
    suspicious in the same session.
    """
    cols, provs, precise, crits = load(ledger)
    by_commit: dict[str, list[str]] = {}
    for name, prov in provs.items():
        sha = prov.get("git_commit")
        if sha and name in cols:
            by_commit.setdefault(sha, []).append(name)
    out: list[dict] = []
    for sha, names in sorted(by_commit.items()):
        for i, a in enumerate(sorted(names)):
            for b in sorted(names)[i + 1:]:
                fatal, arm = comparability_issues(provs[a], provs[b])
                if fatal or arm:
                    # Same commit is NOT enough. Two columns can share a sha
                    # and still be different treatments -- the oracle arms
                    # (base vs gt) do exactly that. A null pairing has to
                    # differ in NOTHING the provenance records, which is what
                    # an empty ``arm`` list means.
                    continue
                # Each side under its own criterion, and a column with no
                # verdict under it is skipped rather than read on the disc:
                # this block is printed UNASKED after every pair and group,
                # as the scale a real delta is read against.
                ca = {e: v for e, v in
                      verdict_cells(a, cols, precise, crits).items() if v is not None}
                cb = {e: v for e, v in
                      verdict_cells(b, cols, precise, crits).items() if v is not None}
                shared = sorted(set(ca) & set(cb))
                if len(shared) < 50:
                    continue
                st = ab_stats([(ca[e], cb[e]) for e in shared])
                out.append({
                    "a": a, "b": b, "n": len(shared),
                    "replicate": _same_stem(a, b),
                    "delta": float(st["delta"]),
                    "p": float(st["mcnemar"]["p_value"]),
                    "wins": st["wins_fail_to_success"],
                    "losses": st["losses_success_to_fail"],
                })
    return out


def render_null(rows: list[dict]) -> str:
    if not rows:
        return (
            "  (no same-commit column pairs stored -- this ledger cannot show you\n"
            "   the null yet, so treat any p-value below with suspicion)"
        )
    worst = max(rows, key=lambda r: abs(r["delta"] or 0.0))
    best_p = min((r for r in rows), key=lambda r: r["p"] if r["p"] is not None else 1.0)
    reps = [r for r in rows if r["replicate"]]
    out = ["  Same commit, same recorded provenance -- the scale to read a result",
           "  against. (*) differs only in a seed/rep token, the strongest evidence;",
           "  provenance does not record every treatment, so the rest may be real",
           "  A/Bs and are shown for context, not as noise."]
    for r in sorted(rows, key=lambda r: -abs(r["delta"])):
        mark = "*" if r["replicate"] else " "
        # Discordance, not the delta, is what sizes the instrument: under the
        # McNemar null SD(delta) = sqrt(disc)/n, so a pair's own +/-1.96 SD band
        # says whether its delta is inside the noise WITHOUT a second opinion.
        disc = (r["wins"] or 0) + (r["losses"] or 0)
        band = 1.96 * math.sqrt(disc) / r["n"] if r["n"] else float("nan")
        out.append(f"  {mark} n={r['n']:4d}  delta={r['delta']:+.3f}  p={r['p']:.3f}  "
                   f"disc={disc:4d} ({disc / r['n']:.0%})  null band=+/-{band:.3f}   "
                   f"{r['a'].split(':')[-1]} vs {r['b'].split(':')[-1]}")
    if reps:
        w = max(reps, key=lambda r: abs(r["delta"]))
        bp = min(reps, key=lambda r: r["p"])
        out.append(f"  seed-only pairs: largest delta {w['delta']:+.3f}, "
                   f"smallest p {bp['p']:.3f}, over {len(reps)} of {len(rows)}")
    else:
        out.append(f"  no seed-only pairs; {len(rows)} same-provenance pairing(s) shown")
    out.append(f"  overall spread: largest {worst['delta']:+.3f}, "
               f"smallest p {best_p['p']:.3f}")
    return "\n".join(out)


def pair(ledger: Path, col_a: str, col_b: str, episodes: set[str] | None = None) -> dict:
    """Paired stats over the shared episodes of two stored columns.

    Refuses on a ``fatal`` provenance mismatch. ``arm`` differences are NOT
    refused here -- a stored-column comparison is nearly always across commits,
    which is the whole point; the caller sees them in the returned dict.

    ``episodes`` narrows the read to a listed subset of the shared cells (the
    rows whose target the program's map covers, a component's rows), applied
    AFTER the identity checks below, which are about the columns and not the
    question.
    """
    cols, provs, precise, crits = load(ledger)
    for c in (col_a, col_b):
        if c not in cols:
            raise SystemExit(f"no such column: {c!r} (have: {', '.join(sorted(cols))})")
    if crits.get(col_a) != crits.get(col_b):
        raise SystemExit(
            f"INVALID pairing -- {col_a} is scored on {crits.get(col_a)!r} and "
            f"{col_b} on {crits.get(col_b)!r}. The two columns' cells are "
            "verdicts under different success masks, so their difference is "
            "not a delta. This is not overridable."
        )
    fatal, arm = comparability_issues(provs.get(col_a, {}), provs.get(col_b, {}))
    if fatal:
        raise SystemExit(
            "INVALID pairing -- the columns score against different ground truth:\n  - "
            + "\n  - ".join(fatal)
            + "\nThis is not overridable: the numbers answer different questions."
        )
    shared = sorted(set(cols[col_a]) & set(cols[col_b]))
    if not shared:
        raise SystemExit(f"no shared episodes between {col_a} and {col_b}")
    # A corpus edit (rebuild, review deletion) shifts row_index and with it
    # episode_id, so two honest full-split columns can intersect on a handful
    # of accidental survivors — same split, same bake, nothing fatal above,
    # and a McNemar over the remnant reads as a verdict on the split. The
    # 1029-row calcity column vs its 1023-row successor shared 29 ids.
    smaller = min(len(cols[col_a]), len(cols[col_b]))
    if len(shared) < 0.9 * smaller:
        raise SystemExit(
            f"INVALID pairing -- only {len(shared)} shared episodes between "
            f"{col_a} ({len(cols[col_a])} cells) and {col_b} ({len(cols[col_b])} cells).\n"
            "The columns mostly disagree on episode identity, which means the corpus "
            "was rebuilt or edited between them; a delta over the accidental overlap "
            "answers a different question. Re-fly one arm on the current corpus."
        )
    # UNDER EACH COLUMN'S OWN CRITERION, which the header records and the
    # refusal above has already checked is the same on both sides. Cal City
    # scores the target's own shape and its 20 m disc is a fallback that
    # never fires, so a paired read on the disc answers a different question
    # from the run report above it -- and an A/B is exactly where that
    # discrepancy gets believed. What used to sit here inferred the criterion
    # from how many cells carried a shape verdict, which read the three
    # pre-shape Cal City columns as disc columns and paired them anyway.
    subset = None
    if episodes is not None:
        listed = len(shared)
        shared = [e for e in shared if e in episodes]
        if not shared:
            raise SystemExit(
                f"none of the {len(episodes)} listed episodes is shared by "
                f"{col_a} and {col_b}; the list is in another id form or another split")
        subset = {"listed": len(episodes), "matched": len(shared), "of": listed}
    ca = verdict_cells(col_a, cols, precise, crits)
    cb = verdict_cells(col_b, cols, precise, crits)
    sa = refuse_unscorable(col_a, {e: ca.get(e) for e in shared})
    sb = refuse_unscorable(col_b, {e: cb.get(e) for e in shared})
    scored_n = sorted(set(sa) & set(sb))
    pairs = [(sa[e], sb[e]) for e in scored_n]
    st = ab_stats(pairs)
    # The renderer prints ``metric`` beside every rate, so naming it here is
    # what stops the number being quoted as the other one.
    st["metric"] = "SR"
    st["criterion"] = ("the target's own shape" if crits.get(col_a) == "shape"
                       else "the 20 m disc")
    st["shared"] = len(shared)
    st["unscored"] = len(shared) - len(scored_n)
    st["label"] = f"{col_a} -> {col_b}"
    st["arm_differences"] = arm
    st["subset"] = subset
    return st


def group(ledger: Path, a_cols: list[str], b_cols: list[str]) -> dict:
    """Pool replicate columns per episode and ask whether the split is special.

    Each episode contributes its MEAN outcome within each group, so N flights
    of one episode average rather than compete; the gap is then the mean of the
    per-episode differences, with a percentile bootstrap over episodes.

    The permutation is the part that decides anything. It re-splits the SAME
    union of columns every way of the same sizes and counts how many reach the
    observed |gap|. Measured on the eight sentence arms, the v7+v8 vs v11*
    split was the largest of all 56 three-vs-five splits (p = 0.036) -- and the
    second largest, +0.032 against +0.036, mixed a v11 run into the "early"
    side. That closeness is the finding, and only the permutation shows it.
    """
    raw_cols, provs, precise, crits = load(ledger)
    missing = [c for c in a_cols + b_cols if c not in raw_cols]
    if missing:
        raise SystemExit(f"no such column(s): {', '.join(missing)} "
                         f"(have: {', '.join(sorted(raw_cols))})")
    if not a_cols or not b_cols:
        raise SystemExit("group needs at least one column on each side")
    every = a_cols + b_cols
    if len({crits.get(c) for c in every}) > 1:
        raise SystemExit(
            "INVALID grouping -- these columns are scored on different "
            "success masks: "
            + ", ".join(f"{c}={crits.get(c)!r}" for c in every)
            + "\nPooling verdicts under two masks averages two questions.")
    for i, x in enumerate(every):
        for y in every[i + 1:]:
            fatal, _ = comparability_issues(provs.get(x, {}), provs.get(y, {}))
            if fatal:
                raise SystemExit(
                    f"INVALID grouping -- {x} and {y} score against different "
                    "ground truth:\n  - " + "\n  - ".join(fatal))
    # Each column under its own criterion. ``group`` read the ring cell
    # unconditionally, so pooling a Cal City column meant pooling its disc
    # number under a report that printed its shape number.
    cols = {c: refuse_unscorable(c, verdict_cells(c, raw_cols, precise, crits))
            for c in every}
    shared = sorted(set.intersection(*[set(cols[c]) for c in every]))
    if len(shared) < 20:
        raise SystemExit(f"only {len(shared)} episodes shared across "
                         f"{len(every)} columns -- pooling needs a common selection")

    def gap(a: list[str], b: list[str]) -> float:
        return sum(sum(cols[c][e] for c in a) / len(a)
                   - sum(cols[c][e] for c in b) / len(b) for e in shared) / len(shared)

    obs = gap(a_cols, b_cols)
    diffs = [sum(cols[c][e] for c in a_cols) / len(a_cols)
             - sum(cols[c][e] for c in b_cols) / len(b_cols) for e in shared]
    lo, hi = _bootstrap_ci(diffs)
    combos = list(itertools.combinations(every, len(a_cols)))
    splits = [gap(list(cb), [c for c in every if c not in cb]) for cb in combos]
    ge = sum(1 for g in splits if abs(g) >= abs(obs) - 1e-12)

    # The column permutation above CANNOT clear a split that was chosen by
    # looking at the numbers. Ranking the columns by SR and cutting the top k
    # lands on the largest gap by construction, so it reads p <= 2/len(splits)
    # with no evidence in it at all. This null shuffles the run labels WITHIN
    # each episode -- which is what "these columns are exchangeable" means --
    # and scores the grouping two ways: against the gap of the SAME split
    # (right when the split was fixed in advance) and against the LARGEST gap
    # any split reaches in that draw (right when it was chosen after the
    # fact). Measured on the eight sentence arms: 0.023 fixed, 0.415 selected.
    idx_a = list(range(len(a_cols)))
    rng = random.Random(12345)
    fixed = selected = 0
    draws = 2000
    for _ in range(draws):
        perm = []
        for e in shared:
            vals = [cols[c][e] for c in every]
            rng.shuffle(vals)
            perm.append(vals)
        def pgap(ai: list[int]) -> float:
            bi = [i for i in range(len(every)) if i not in set(ai)]
            return sum(sum(v[i] for i in ai) / len(ai)
                       - sum(v[i] for i in bi) / len(bi) for v in perm) / len(perm)
        if abs(pgap(idx_a)) >= abs(obs) - 1e-12:
            fixed += 1
        best = max(abs(pgap([every.index(c) for c in cb])) for cb in combos)
        if best >= abs(obs) - 1e-12:
            selected += 1
    return {
        "n_episodes": len(shared), "a": a_cols, "b": b_cols,
        "sr_a": sum(sum(cols[c][e] for c in a_cols) / len(a_cols)
                    for e in shared) / len(shared),
        "sr_b": sum(sum(cols[c][e] for c in b_cols) / len(b_cols)
                    for e in shared) / len(shared),
        "gap": obs, "gap_ci95": [lo, hi],
        "n_splits": len(splits), "splits_at_least_as_large": ge,
        "permutation_p": ge / len(splits),
        "split_range": [min(splits), max(splits)],
        "label_p_fixed_split": fixed / draws,
        "label_p_split_selected": selected / draws,
        "label_draws": draws,
    }


def render_group(st: dict) -> str:
    a, b = ", ".join(c.split(":")[-1] for c in st["a"]), \
        ", ".join(c.split(":")[-1] for c in st["b"])
    return "\n".join([
        f"pooled over {st['n_episodes']} shared episodes",
        f"  A ({len(st['a'])} col): {a}",
        f"     mean SR = {st['sr_a']:.4f}",
        f"  B ({len(st['b'])} col): {b}",
        f"     mean SR = {st['sr_b']:.4f}",
        f"  gap A-B = {st['gap']:+.4f}",
        f"    episode-bootstrap 95%CI[{st['gap_ci95'][0]:+.4f},"
        f"{st['gap_ci95'][1]:+.4f}]  -- resamples EPISODES with these columns"
        " FIXED,",
        "      so it carries no run-to-run draw variance and is NOT a test of"
        " the grouping",
        f"  label permutation ({st['label_draws']} draws, run labels shuffled"
        " within each episode):",
        f"    p = {st['label_p_fixed_split']:.3f}   if this split was fixed"
        " BEFORE the numbers were seen",
        f"    p = {st['label_p_split_selected']:.3f}   if it was chosen after"
        " -- the max-gap-over-all-splits null",
        f"  descriptive: {st['splits_at_least_as_large']}/{st['n_splits']}"
        f" same-sized splits of these columns reach |gap|; they range"
        f" {st['split_range'][0]:+.4f} .. {st['split_range'][1]:+.4f}",
        "    (a split picked by ranking columns on SR tops that list by"
        " construction -- not evidence)",
    ])


def render_table(
    cols: dict[str, dict[str, int]],
    crits: dict[str, str] | None = None,
    limit: int = 40,
) -> str:
    """Episodes down, columns across. '.' = fail, 'X' = success, ' ' = not run.

    Every SR is labelled with the mask it is a verdict under, because the
    columns in one table are not all scored on the same one.
    """
    crits = crits or {}
    names = sorted(cols)
    episodes = sorted({e for c in cols.values() for e in c})
    width = max((len(e) for e in episodes[:limit]), default=10)
    head = "  ".join(f"{i}" for i in range(len(names)))
    out = [f"{'episode'.ljust(width)}  {head}"]
    for i, n in enumerate(names):
        got = cols[n]
        label = crits.get(n) or "?"
        out.append(
            f"  [{i}] {n}  n={len(got)}  "
            f"SR[{label}]={sum(got.values())}/{len(got)}"
        )
    out.append("")
    for e in episodes[:limit]:
        cells = "  ".join("X" if cols[n].get(e) else ("." if e in cols[n] else " ")
                          for n in names)
        out.append(f"{e.ljust(width)}  {cells}")
    if len(episodes) > limit:
        out.append(f"... {len(episodes) - limit} more episodes")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=VERBS)
    ap.add_argument("args", nargs="*")
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--episodes", type=Path, default=None,
                    help="pair only the shared episodes listed in this file, one id per "
                         "line")
    a = ap.parse_args(argv[1:])

    if a.cmd == "add":
        if not a.args:
            raise SystemExit("add needs at least one run root")
        for root in a.args:
            col, n = append(a.ledger, Path(root))
            print(f"added {col}  ({n} episodes)  -> {a.ledger}")
    elif a.cmd == "null":
        print(render_null(null_reference(a.ledger)))
    elif a.cmd == "table":
        raw, _, precise, crits = load(a.ledger)
        if not raw:
            raise SystemExit(f"empty ledger: {a.ledger}")
        # Each column's own verdict. Reading the ring cell of a shape column
        # here printed one number in the table and a different one in the
        # report the table was meant to summarise.
        print(render_table(
            {c: {e: v for e, v in verdict_cells(c, raw, precise, crits).items()
                 if v is not None}
             for c in raw},
            crits))
    elif a.cmd == "group":
        if len(a.args) != 2:
            raise SystemExit(
                "group needs two comma-separated column lists: "
                "group <a1,a2,...> <b1,b2,...>")
        st = group(a.ledger, [c for c in a.args[0].split(",") if c],
                   [c for c in a.args[1].split(",") if c])
        print(render_group(st))
        # Same reason ``pair`` prints it: the moment someone is about to
        # believe a pooled gap is the moment the null is worth seeing.
        print("\nnull reference (same commit, different seed):")
        print(render_null(null_reference(a.ledger)))
    else:
        if len(a.args) != 2:
            raise SystemExit("pair needs exactly two column ids")
        subset = None
        if a.episodes is not None:
            subset = {ln.strip() for ln in a.episodes.read_text().splitlines() if ln.strip()}
        st = pair(a.ledger, a.args[0], a.args[1], episodes=subset)
        print(render(st))
        if st.get("subset"):
            print(f"  subset: {st['subset']['matched']} of {st['subset']['of']} shared episodes "
                  f"are among the {st['subset']['listed']} listed in {a.episodes}")
        print(f"  criterion: {st['criterion']}"
              + (f"  ({st['unscored']} of {st['shared']} shared episodes "
                 "carry no shape and are out of this read)"
                 if st.get("unscored") else ""))
        for d in st.get("arm_differences") or []:
            print(f"  note: {d}")
        # Unasked-for on purpose. A p-value read alone invites a story, and
        # the null is only useful at the moment someone is about to believe
        # one — not behind a verb they have to remember to run.
        print("\nnull reference (same commit, different seed):")
        print(render_null(null_reference(a.ledger)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
