"""The capacity study's download, built once from the bench's own files (``STUDY.md``).

A participant's machine needs no reasoner, no segmenter and no 46 GB ortho: only the
pictures the console shows and the records MAPS's results are read from. This writes:

  ``calcity_ortho.mbtiles``  the bench's ortho (``<map>_rgb.tif``) as web-mercator WebP
                             tiles, zoom 12 to 20, no-data transparent
  ``maps_root/``             the MAPS root the plan replays: its ledger and timing, and per
                             attempt the records a verify screen reads (the turns, the
                             detections, the stops, the frames, and MAPS's final reply as
                             ``reply.md``)
  ``tiles/``                 the bench's OSM prior bake (``<map>_meta.json``, ``<map>_features.json``)
  ``data/cityrefer/objects.json``  the labelled objects
  ``plan/``                  the participants' lists, dealt by ``capacity plan``, and the
                             practice reports

    python -m baselines.human.study_package tiles --raster <map>_rgb.tif --out <dir>/calcity_ortho.mbtiles
    python -m baselines.human.study_package records --maps-root <root> --study <plan dir> --out <dir>
    python -m baselines.human.study_package replies --transcripts <mirror of the root's transcripts> --kit <dir>
    python -m baselines.human.study_package bundle --kit <dir> --out <outside the repo>/MAPS-study.zip --contact <email>

``replies`` writes ``reply.md`` into a kit's attempts from a folder that holds only the root's
``<attempt>/flight/llm/transcript.jsonl`` files, laid out as the root lays them out.

``bundle`` writes the one file a participant downloads: the download above, the console's code
(the Python only, never the bench's corpus and its targets), the briefing, the study lead's
contact address, and a launcher for macOS and one for Windows (``bundle/``) that fetch uv, and
with it Python and the console's packages, on the first start.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from baselines.human.study_server import VIDEO

#: Web mercator's half-width, metres.
MERC_O = 20037508.342789244
#: Zoom 12 frames the whole block; 20 is 0.12 m a pixel at Cal City, finer than the
#: bench's 0.15 m, so the closest zoom shows every pixel the bench has.
ZOOMS = range(12, 21)
TILE_PX = 256
#: The attempt files a verify screen reads (``audit_console.blind_review``), and its frames.
ATTEMPT_FILES = ("conversation.jsonl", "detections.jsonl", "flight/tour_cursor.json")
#: An attempt's LLM wire log, whose last message is MAPS's final reply; the kit keeps the reply alone.
TRANSCRIPT = "flight/llm/transcript.jsonl"

#: The open raster per overview level, one set per worker process.
_SRC: dict = {}


def _tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    s = 2 * MERC_O / 2**z
    return -MERC_O + x * s, MERC_O - (y + 1) * s, -MERC_O + (x + 1) * s, MERC_O - y * s


def _render(job: tuple[str, int, int, int]) -> tuple[int, int, int, bytes | None]:
    """One tile, warped from the raster's coarsest overview still as fine as the tile (a
    warp off the full raster reads all of it for a zoomed-out tile). Black is no data in
    the bench's raster, so it is cut clear."""
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject

    raster, z, x, y = job
    if not _SRC:
        _SRC[None] = rasterio.open(raster)
    base = _SRC[None]
    # A tile's ground metres a pixel: mercator's, shrunk by the cosine of its latitude.
    lat = np.arctan(np.sinh(np.pi * (1 - 2 * (y + 0.5) / 2**z)))
    tile_m = 2 * MERC_O / 2**z / TILE_PX * np.cos(lat)
    level = None
    for i, f in enumerate(base.overviews(1)):
        if base.res[0] * f <= tile_m:
            level = i
    if level not in _SRC:
        _SRC[level] = rasterio.open(raster, overview_level=level)
    a = np.zeros((3, TILE_PX, TILE_PX), np.uint8)
    reproject(rasterio.band(_SRC[level], (1, 2, 3)), a, dst_transform=from_bounds(*_tile_bounds(z, x, y), TILE_PX, TILE_PX),
              dst_crs="EPSG:3857", resampling=Resampling.bilinear)
    rgb = a.transpose(1, 2, 0)
    alpha = np.where(rgb.max(axis=2) == 0, 0, 255).astype(np.uint8)
    if not alpha.any():
        return z, x, y, None
    buf = io.BytesIO()
    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(buf, format="WEBP", quality=80)
    return z, x, y, buf.getvalue()


def build_tiles(raster: Path, out: Path, workers: int = 8) -> int:
    """Every tile over the raster at every zoom in ``ZOOMS``, into one MBTiles file."""
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(raster) as src:
        l, b, r, t = transform_bounds(src.crs, "EPSG:3857", *src.bounds)
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    jobs = []
    for z in ZOOMS:
        step = 2 * MERC_O / 2**z
        for x in range(int((l + MERC_O) // step), int((r + MERC_O) // step) + 1):
            for y in range(int((MERC_O - t) // step), int((MERC_O - b) // step) + 1):
                jobs.append((str(raster), z, x, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.executescript("CREATE TABLE metadata (name TEXT, value TEXT);"
                     "CREATE TABLE tiles (zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB);")
    meta = {"name": raster.stem, "format": "webp", "type": "baselayer", "minzoom": min(ZOOMS), "maxzoom": max(ZOOMS),
            "bounds": f"{w},{s},{e},{n}", "center": f"{(w + e) / 2},{(s + n) / 2},{min(ZOOMS) + 1}"}
    db.executemany("INSERT INTO metadata VALUES (?, ?)", [(k, str(v)) for k, v in meta.items()])
    n_tiles = 0
    with ProcessPoolExecutor(workers) as pool:
        for i, (z, x, y, data) in enumerate(pool.map(_render, jobs, chunksize=64)):
            if data is not None:
                # MBTiles rows count from the south (TMS).
                db.execute("INSERT INTO tiles VALUES (?, ?, ?, ?)", (z, x, 2**z - 1 - y, data))
                n_tiles += 1
            if i % 5000 == 0:
                db.commit()
                print(f"tiles: {i}/{len(jobs)} ({n_tiles} kept)", flush=True)
    db.execute("CREATE UNIQUE INDEX tile_index ON tiles (zoom_level, tile_column, tile_row)")
    db.commit()
    db.close()
    tmp.replace(out)
    return n_tiles


def reply_text(transcript: Path) -> str:
    """MAPS's final reply, the explanation of its answer: the content of the transcript's last
    response; empty when the transcript is absent or holds no response."""
    from benchmarks.citynav.analysis.rows import iter_json_rows

    reply = ""
    for row in iter_json_rows(transcript) if transcript.is_file() else ():
        choices = (row.get("response") or {}).get("choices") if row.get("kind") == "response" else None
        if choices:
            reply = str((choices[0].get("message") or {}).get("content") or "")
    return reply.strip()


def write_reply(transcript: Path, attempt: Path) -> bool:
    """``<attempt>/reply.md`` from the attempt's transcript; whether there was a reply to write."""
    reply = reply_text(transcript)
    if reply:
        attempt.mkdir(parents=True, exist_ok=True)
        (attempt / "reply.md").write_text(reply, encoding="utf-8")
    return bool(reply)


def copy_records(maps_root: Path, out: Path) -> int:
    """The MAPS root's ledger, timing and manifest, and per attempt what a verify screen reads."""
    from baselines.human.session_plan import scored_rows

    out.mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "batch_timing.json", "run_manifest.json"):
        if (maps_root / name).is_file():
            shutil.copy2(maps_root / name, out / name)
    rows = scored_rows(maps_root)
    for row in rows.values():
        src, dst = maps_root / row["workdir"], out / row["workdir"]
        for rel in ATTEMPT_FILES:
            if (src / rel).is_file():
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / rel, dst / rel)
        if (src / "frames").is_dir():
            shutil.copytree(src / "frames", dst / "frames", dirs_exist_ok=True)
        write_reply(src / TRANSCRIPT, dst)
    return len(rows)


def copy_replies(transcripts: Path, kit: Path) -> list[str]:
    """``reply.md`` into every attempt of the kit's MAPS root, from ``transcripts``, a folder of
    the root's transcripts alone; the episodes left without one."""
    from baselines.human.session_plan import scored_rows

    root = kit / "maps_root"
    return [eid for eid, row in scored_rows(root).items()
            if not write_reply(transcripts / row["workdir"] / TRANSCRIPT, root / row["workdir"])]


REPO = Path(__file__).resolve().parents[2]
BUNDLE = Path(__file__).with_name("bundle")
#: The launchers at the bundle's top, by their name in the bundle; ``app/`` holds the rest.
TOP = {"Start study (Mac).command": "mac.command", "Start study (Windows).bat": "windows.bat",
       "Start study (Linux).sh": "linux.sh"}
#: Already compressed: stored as is, so the zip builds in minutes.
STORED = (".mbtiles", ".jpg", ".png", ".webp", ".mp4")
#: What a kit cannot start without: the console, the plan, the map and MAPS's run.
REQUIRED = ("console/index.html", "plan/plan.json", "calcity_ortho.mbtiles", "maps_root/episodes.jsonl")
#: The capacity study's own record (IRB-90304), which grading and the download rest on alone:
#: ``reports.json`` (the reports dealt), ``targets.json`` (what each is graded on), ``plan/`` (the
#: lists dealt) and ``lock.json`` (the checksum of each, the kit's MAPS ledger and the zip).
STUDY = Path(__file__).with_name("study")
STUDY_REPORTS = STUDY / "reports.json"
STUDY_TARGETS = STUDY / "targets.json"
STUDY_LOCK = STUDY / "lock.json"
#: The one MAPS metric a participant's copy keeps: the steps MAPS took price a replacement
#: against the budget it left. The target and every score stay with the lead.
SHIPPED_METRICS = ("steps_taken",)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dealt(kit: Path) -> set[str]:
    """Every report id the kit's plan deals, practice included."""
    return {ln.strip() for f in (kit / "plan").glob("*/*.txt") for ln in f.read_text().splitlines() if ln.strip()}


def off_manifest(kit: Path, manifest: Path = STUDY_REPORTS) -> list[str]:
    """Report ids the kit's plan deals that the study manifest does not list, sorted."""
    doc = json.loads(manifest.read_text())
    return sorted(dealt(kit) - set(doc["session"]) - set(doc["practice"]))


def shipped_row(row: dict) -> dict:
    """A MAPS row as the participant's download carries it: no target point, no score."""
    out = {k: v for k, v in row.items() if k != "target_xy"}
    out["metrics"] = {k: row["metrics"][k] for k in SHIPPED_METRICS if k in row.get("metrics", {})}
    return out


def _plain(v):
    """A footprint as JSON lists of floats, however the bench holds it."""
    return [_plain(x) for x in v] if isinstance(v, (list, tuple, np.ndarray)) else float(v)


def export_targets(kit: Path, study: Path = STUDY) -> int:
    """``targets.json``: each session report's text as the kit's MAPS run shows it, and its
    footprint and target point off the corpus as committed now. Refuses a dirty corpus or a
    report whose text is not the corpus's."""
    from baselines.human.study_server import use_kit

    use_kit(kit)
    from baselines.human import team_rows
    from baselines.human.session_plan import scored_rows
    from benchmarks.citynav.adapter import load_split, parse_episode
    from benchmarks.citynav.paths import split_path

    corpus_file = split_path("calcity")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(corpus_file)], cwd=REPO).returncode:
        raise SystemExit(f"{corpus_file.name} has uncommitted changes: commit them before freezing the study's targets")
    commit = subprocess.run(["git", "log", "-1", "--format=%h", "--", str(corpus_file)], cwd=REPO, check=True,
                            capture_output=True, text=True).stdout.strip()
    corpus, rows = load_split(corpus_file), scored_rows(kit / "maps_root")
    shape_of = team_rows.TargetShapes("calcity")
    targets = {}
    for e in json.loads((study / "reports.json").read_text())["session"]:
        i = int(e.rsplit("#", 1)[1])
        row, ep = rows[e], parse_episode(corpus[i], row_index=i)
        if row["description"] not in corpus[i]["descriptions"]:
            raise SystemExit(f"{e}: the text MAPS's run shows is not the corpus's")
        targets[e] = {"description": row["description"], "target_xy": _plain(ep.target_xy), "footprint": _plain(shape_of(e))}
    doc = {"_README": "What each capacity-study report is graded on (IRB-90304): the text participants read, the "
                      "target point and the footprint a commit must land in. Exported by study_package freeze off "
                      f"benchmarks/calcity/data/calcity.json at {commit}; grading reads this file, never the corpus.",
           "corpus_commit": commit, "targets": targets}
    (study / "targets.json").write_text(json.dumps(doc, indent=0) + "\n")
    return len(targets)


def _study_files(kit: Path, study: Path) -> dict[str, Path]:
    files = {"reports.json": study / "reports.json", "targets.json": study / "targets.json",
             "kit/maps_root/episodes.jsonl": kit / "maps_root" / "episodes.jsonl"}
    files |= {f"plan/{p.relative_to(study / 'plan').as_posix()}": p for p in sorted((study / "plan").rglob("*")) if p.is_file()}
    return files


def freeze(kit: Path, study: Path = STUDY) -> dict:
    """The kit's plan copied into the study, ``targets.json`` exported and ``lock.json`` written."""
    shutil.rmtree(study / "plan", ignore_errors=True)
    shutil.copytree(kit / "plan", study / "plan")
    export_targets(kit, study)
    return write_lock(kit, study)


def write_lock(kit: Path, study: Path = STUDY) -> dict:
    """``lock.json``: the sha256 of the study's files and the kit's MAPS ledger, as they stand."""
    lock = {"_README": "The sha256 of every file the capacity study rests on, written by study_package freeze; "
                       "check and bundle refuse a kit or a study folder that no longer matches. zip: the last "
                       "download bundle built.",
            "files": {k: _sha(p) for k, p in _study_files(kit, study).items()}}
    (study / "lock.json").write_text(json.dumps(lock, indent=1) + "\n")
    return lock


def unlocked(kit: Path, study: Path = STUDY, zip_file: Path | None = None) -> list[str]:
    """Where the study folder, the kit or a built zip strays from ``lock.json``."""
    if not (study / "lock.json").is_file():
        return ["the study has no lock.json: run study_package freeze"]
    lock = json.loads((study / "lock.json").read_text())
    have = {k: _sha(p) for k, p in _study_files(kit, study).items() if p.is_file()}
    out = [f"{k}: {'missing' if k not in have else 'changed since the freeze'}" for k, v in lock["files"].items() if have.get(k) != v]
    out += [f"{k}: not in the lock" for k in have if k not in lock["files"]]
    kit_plan = {p.relative_to(kit / "plan").as_posix(): _sha(p) for p in (kit / "plan").rglob("*") if p.is_file()}
    study_plan = {k.removeprefix("plan/"): v for k, v in have.items() if k.startswith("plan/")}
    out += [f"kit/plan/{k}: not the study's plan" for k in sorted(set(kit_plan) | set(study_plan)) if kit_plan.get(k) != study_plan.get(k)]
    if zip_file is not None and _sha(zip_file) != (lock.get("zip") or {}).get("sha256"):
        out.append(f"{zip_file.name}: not the zip the lock records")
    return out


def ungradable(kit: Path, study: Path = STUDY) -> tuple[list[str], list[str]]:
    """Every session report the kit's plan deals, graded the way a participant's answer will be,
    on the study's ``targets.json``: the text it shows is the frozen text, a point inside its
    footprint scores right and one 565 m off scores wrong. The failures as ``"<id>: <why>"``, and
    the reports where MAPS's root, scored on the masks of its day, disagrees with the frozen
    footprint (``team_rows.masked_now`` regrades those)."""
    ids = sorted(dealt(kit) & set(json.loads((study / "reports.json").read_text())["session"]))
    if not ids:
        return [], []
    from baselines.human.study_server import use_kit

    use_kit(kit)
    from shapely.geometry import MultiPolygon, Polygon

    from baselines.human import team_rows
    from baselines.human.session_plan import scored_rows

    rows, cfg = scored_rows(kit / "maps_root"), team_rows.root_config(kit / "maps_root")
    targets = team_rows.StudyTargets(study / "targets.json")
    bad, moved = [], []
    for e in ids:
        row, frozen = rows.get(e), targets.targets.get(e)
        if row is None or frozen is None:
            bad.append(f"{e}: not in the kit's MAPS root" if row is None else f"{e}: not in targets.json")
            continue
        if row["description"] != frozen["description"]:
            bad.append(f"{e}: the text it shows is not the frozen text")
        if row["metrics"].get("success_criterion") != "shape":
            bad.append(f"{e}: scored on a ring, not its footprint")
            continue
        row = {**row, "target_xy": frozen["target_xy"]}
        shape = frozen["footprint"]
        geom = None
        if shape and len(shape) >= 3:
            geom = (Polygon(shape) if isinstance(shape[0][0], (int, float)) else MultiPolygon([Polygon(q) for q in shape])).buffer(0)
        if geom is None or geom.is_empty:
            bad.append(f"{e}: no footprint to grade on")
            continue
        p = geom.representative_point()
        hit = team_rows.team_row(row, {"decision": "commit", "xy": [p.x, p.y]}, cfg, targets)["metrics"]["SR"]
        miss = team_rows.team_row(row, {"decision": "commit", "xy": [p.x + 400, p.y + 400]}, cfg, targets)["metrics"]["SR"]
        if (hit, miss) != (1, 0):
            bad.append(f"{e}: a point inside scores {hit}, one 565 m off scores {miss}")
        now = team_rows.masked_now(row, cfg, targets)["metrics"]["SR"]
        if now != row["metrics"]["SR"]:
            moved.append(f"{e}: MAPS {row['metrics']['SR']} as flown, {now} on the frozen footprint")
    return bad, moved


def build_bundle(kit: Path, out: Path, contact: str, video: Path | None = None, study: Path = STUDY) -> int:
    """The participant's one download, as a zip whose launchers keep their execute bit. The
    READ ME and ``kit/contact.txt`` (which the visit page reads) carry the lead's ``contact``.
    Built only from a kit that matches the study's lock and grades whole; MAPS's ledger ships
    without its targets and scores (``shipped_row``), and the lock records the zip."""
    name = "MAPS-study"
    code = subprocess.run(["git", "ls-files", "maps", "benchmarks", "baselines"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout.split()
    code = [f for f in code if f.endswith(".py") and "/tests/" not in f] + ["baselines/human/briefing.md", "baselines/human/information.md"]
    texts = {f"{name}/READ ME FIRST.txt": (BUNDLE / "READ ME FIRST.txt").read_text().replace("{CONTACT}", contact),
             f"{name}/kit/contact.txt": contact + "\n"}
    files = [(f"{name}/{arc}", BUNDLE / src) for arc, src in TOP.items()]
    if video is not None:
        # Already compressed: stored as is.
        files.append((f"{name}/{VIDEO}", video))
    files += [(f"{name}/app/run.sh", BUNDLE / "run.sh"), (f"{name}/app/requirements.txt", BUNDLE / "requirements.txt")]
    files += [(f"{name}/app/{f}", REPO / f) for f in code]
    # A kit of symlinked folders ships what the links point at.
    kept = sorted(Path(d, f).relative_to(kit) for d, _, fs in os.walk(kit, followlinks=True) for f in fs)
    # The roster (first names by label) is the lead's record: participants type the code their email gives.
    kept = [rel for rel in kept if rel.parts[0] not in ("results", "scored") and rel != Path("roster.json")]
    missing = [rel for rel in REQUIRED if Path(rel) not in kept]
    if missing:
        raise SystemExit(f"{kit} is not a whole kit: it lacks {', '.join(missing)}")
    stray = off_manifest(kit, study / "reports.json")
    if stray:
        raise SystemExit(f"{kit}'s plan deals {len(stray)} reports outside reports.json, "
                         f"first {stray[0]}: regenerate the plan, or change the manifest with a STUDY.md note")
    drift = unlocked(kit, study)
    if drift:
        raise SystemExit(f"{len(drift)} files stray from the study's lock.json, first {drift[0]}")
    bad, _moved = ungradable(kit, study)
    if bad:
        raise SystemExit(f"{len(bad)} of the study's reports cannot be graded, first {bad[0]}")
    ledger = Path("maps_root/episodes.jsonl")
    from benchmarks.citynav.analysis.rows import iter_json_rows

    texts[f"{name}/kit/{ledger.as_posix()}"] = "".join(json.dumps(shipped_row(r)) + "\n" for r in iter_json_rows(kit / ledger))
    files += [(f"{name}/kit/{rel.as_posix()}", kit / rel) for rel in kept if rel != ledger]
    files = [(arc, src) for arc, src in files if arc not in texts]
    # Windows' own line endings, so Notepad and cmd read them as lines.
    crlf = lambda b: b.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, text in texts.items():
            # The ledger is read as JSON lines, not by people: its own line endings.
            z.writestr(arc, text.encode() if arc.endswith(".jsonl") else crlf(text.encode()))
        for arc, src in files:
            if arc.endswith(".bat"):
                info = zipfile.ZipInfo.from_file(src, arc)
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, crlf(src.read_bytes()))
            else:
                # The file's own mode rides along: the launchers keep their execute bit.
                z.write(src, arc, zipfile.ZIP_STORED if arc.endswith(STORED) else zipfile.ZIP_DEFLATED)
    tmp.replace(out)
    lock = json.loads((study / "lock.json").read_text())
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    lock["zip"] = {"sha256": _sha(out), "bytes": out.stat().st_size, "commit": commit}
    (study / "lock.json").write_text(json.dumps(lock, indent=1) + "\n")
    return len(texts) + len(files)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tiles", help="the ortho as one MBTiles file")
    t.add_argument("--raster", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--workers", type=int, default=8)
    r = sub.add_parser("records", help="the MAPS root, the prior, the labels and the plan, as the download lays them out")
    r.add_argument("--maps-root", type=Path, required=True)
    r.add_argument("--study", type=Path, required=True, help="the plan's folder (capacity plan and practice wrote it)")
    r.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("replies", help="MAPS's final replies into a kit, from a mirror of the root's transcripts")
    p.add_argument("--transcripts", type=Path, required=True)
    p.add_argument("--kit", type=Path, required=True)
    c = sub.add_parser("check", help="the kit against the study's lock and reports, and every report graded once")
    c.add_argument("--kit", type=Path, required=True)
    c.add_argument("--zip", type=Path, help="a downloaded bundle, checked against the zip the lock records")
    f = sub.add_parser("freeze", help="the kit's plan into the study, its targets exported and its lock written")
    f.add_argument("--kit", type=Path, required=True)
    b = sub.add_parser("bundle", help="the participant's one download: the kit, the code and the launchers")
    b.add_argument("--kit", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True, help="the zip, outside the repository")
    b.add_argument("--contact", required=True, help="the address or addresses for questions and the results, as one phrase (\"a@x.edu and b@y.edu\")")
    b.add_argument("--video", type=Path, help=f"the tutorial video, shipped at the top as \"{VIDEO}\"")
    args = ap.parse_args(argv)
    if args.cmd == "freeze":
        lock = freeze(args.kit)
        print(f"{len(lock['files'])} files locked -> {STUDY_LOCK}")
        return 0
    if args.cmd == "check":
        stray, drift, (bad, moved) = off_manifest(args.kit), unlocked(args.kit, zip_file=args.zip), ungradable(args.kit)
        for line in [*(f"off the manifest: {e}" for e in stray), *drift, *bad, *moved]:
            print(line)
        print(f"{len(stray)} dealt off the manifest, {len(drift)} off the lock, {len(bad)} ungradable, "
              f"{len(moved)} MAPS verdicts regraded on the frozen footprints")
        return 1 if stray or drift or bad else 0
    if args.cmd == "bundle":
        print(f"{build_bundle(args.kit, args.out, args.contact, args.video)} files -> {args.out}")
        return 0
    if args.cmd == "replies":
        missing = copy_replies(args.transcripts, args.kit)
        print(f"replies -> {args.kit / 'maps_root'}; {len(missing)} attempts without one" + "".join(f"\n  {e}" for e in missing))
        return 0
    if args.cmd == "tiles":
        print(f"{build_tiles(args.raster, args.out, args.workers)} tiles -> {args.out}")
        return 0
    from benchmarks.citynav.paths import OBJECTS_JSON, TILES_DIR

    print(f"{copy_records(args.maps_root, args.out / 'maps_root')} attempts -> {args.out / 'maps_root'}")
    (args.out / "tiles").mkdir(parents=True, exist_ok=True)
    for f in Path(TILES_DIR).glob("*_meta.json"):
        shutil.copy2(f, args.out / "tiles" / f.name)
        feats = f.with_name(f.name.replace("_meta.json", "_features.json"))
        if feats.is_file():
            shutil.copy2(feats, args.out / "tiles" / feats.name)
    (args.out / "data/cityrefer").mkdir(parents=True, exist_ok=True)
    shutil.copy2(OBJECTS_JSON, args.out / "data/cityrefer/objects.json")
    # The lists and the practice root, never a session a pilot already wrote.
    shutil.copytree(args.study, args.out / "plan", dirs_exist_ok=True,
                    ignore=lambda d, names: [n for n in names if n.startswith(".")
                                             or (Path(d).parent == args.study and Path(d).name != "practice"
                                                 and n in ("solo", "team", "practice"))])
    print(f"prior, labels and plan -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
