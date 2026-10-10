"""Workdir-resume artifact hydration (checkpoint dashboard trees
mirrored into the runtime workdir via hardlinks).
"""

from __future__ import annotations

from pathlib import Path
import os
import logging

log = logging.getLogger("run_benchmark")


def _link_or_copy(src: Path, dst: Path) -> None:
    """copytree that prefers hardlinks, falling back to byte copy.

    ``output/`` is one mount so source and destination almost always share
    a device. Hardlinking is instant regardless of tree size and the
    dashboard reads files (not inodes), so behavior is identical. The
    fallback on ``OSError`` (cross-device, read-only FS, hardlink-disabled
    filesystem) keeps the call correct off the happy path.
    """
    import shutil

    try:
        shutil.copytree(src, dst, copy_function=os.link)
    except OSError:
        shutil.copytree(src, dst)


def hydrate_resume_artifacts(
    *,
    src_workdir: str | Path | None,
    dst_workdir: str | None,
    turn: int,
) -> None:
    """Mirror a checkpoint's dashboard artifacts into the runtime workdir.

    The dashboard reads the belief/reasoner raster from
    ``<workdir>/flight/turns/turn_<NNN>/``, the compile tree from
    ``<workdir>/compile/turn_<NNNN>/``, and per-reasoner transcripts
    from ``<workdir>/h<i>/``. When ``SearchLoop.restore_from_workdir``
    seeds the engine with a prior run's state, the runner does not
    re-emit those artifacts — so the SPA panels stay empty even though
    the planner is flying the loaded belief. Hardlinking the source
    tree into the dst workdir hydrates the SPA panels for the duration
    of the resume run without copying bytes.
    """
    if not src_workdir or not dst_workdir:
        return
    src_p = Path(src_workdir).resolve()
    dst_p = Path(dst_workdir).resolve()
    if not src_p.is_dir() or not dst_p.is_dir():
        return
    if src_p == dst_p:
        return
    flight_turn_src = src_p / "flight" / "turns" / f"turn_{turn:03d}"
    compile_turn_src = src_p / "compile" / f"turn_{turn:04d}"
    llm_src = src_p / "flight" / "llm"
    flight_turn_dst = dst_p / "flight" / "turns" / f"turn_{turn:03d}"
    compile_turn_dst = dst_p / "compile" / f"turn_{turn:04d}"
    llm_dst = dst_p / "flight" / "llm"
    copied: list[str] = []
    if flight_turn_src.is_dir() and not flight_turn_dst.exists():
        flight_turn_dst.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(flight_turn_src, flight_turn_dst)
        copied.append(str(flight_turn_dst.relative_to(dst_p)))
    if compile_turn_src.is_dir() and not compile_turn_dst.exists():
        compile_turn_dst.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(compile_turn_src, compile_turn_dst)
        copied.append(str(compile_turn_dst.relative_to(dst_p)))
    # Stage 1.5 SAM 3.1 concept-union artifacts (segment_summary.json,
    # by_concept_masks.npz, ...) live under compile/union/. Both the SPA's
    # preflight SAM panel and the in-flight sensor-view legend chips read
    # from this directory; without it the panels render blank on resume.
    union_src = src_p / "compile" / "union"
    union_dst = dst_p / "compile" / "union"
    if union_src.is_dir() and not union_dst.exists():
        union_dst.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(union_src, union_dst)
        copied.append(str(union_dst.relative_to(dst_p)))
    # The LLM transport directory holds flight/llm/transcript.jsonl, the
    # reasoner's record stream the flight summary reads.
    if llm_src.is_dir() and not llm_dst.exists():
        llm_dst.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(llm_src, llm_dst)
        copied.append(str(llm_dst.relative_to(dst_p)))
    for h_dir in sorted(src_p.glob("h[0-9]*")):
        if not h_dir.is_dir():
            continue
        h_dst = dst_p / h_dir.name
        if h_dst.exists():
            continue
        _link_or_copy(h_dir, h_dst)
        copied.append(str(h_dst.relative_to(dst_p)))
    if copied:
        log.info(
            "Hydrated dashboard artifacts from %s -> %s: %s",
            src_p,
            dst_p,
            ", ".join(copied),
        )
