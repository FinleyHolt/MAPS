"""ViSA's per-episode loop over the harness.

One instruction analysis, one coverage plan over the matched landmarks'
contours, then at each time step: perceive (SAM 3.1 candidates marked on the
frame), verify (the three stages by mark id), execute one task primitive
(stop / move / ascend / descend). The verification module's guidance signal
is the textual prompt of the NEXT perception round, wherever the executor
takes the UAV: the paper's example rejects a candidate, sends the signal back
and moves, and the next waypoint's perception "adjusts its attention based
on the feedback".
"""

from __future__ import annotations

import math
import re

from baselines.seam import BaselineHarness, BaselineResult, Landmark, match_landmarks
from baselines.visa import prompts
from baselines.visa.som import legend, mark_frame, mark_world_xy
from baselines.visa.waypoints import plan_coverage

DELTA_Z_M = 10.0
# None: the seam's free-call cap, the headline arm's output reserve.
ANALYSIS_MAX_TOKENS = None
VERIFY_MAX_TOKENS = None
# ponytail: a frame cap on candidates. SAM 3.1 over a 270 m view proposes
# several hundred masks (564 measured at 155 m over a Birmingham block), and a
# Set-of-Mark frame with that many numbered discs is unreadable by any model.
# The paper's VLM-proposed candidates are bounded by what a VLM will list, so
# the top-K by detector confidence stands in for that bound; raise it if the
# target is measured to fall outside it.
MAX_MARKS = 80


def _compass(dx: float, dy: float) -> str:
    names = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    ang = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
    return names[int((ang + 22.5) // 45) % 8]


def landmark_text(landmarks: list[Landmark], centre: tuple[float, float], half: float) -> str:
    if not landmarks:
        return "(no named landmark of the instruction matched the map)"
    from shapely.geometry import Polygon, box

    view = box(centre[0] - half, centre[1] - half, centre[0] + half, centre[1] + half)
    lines = []
    for lm in landmarks:
        dx, dy = lm.centroid[0] - centre[0], lm.centroid[1] - centre[1]
        dist = math.hypot(dx, dy)
        try:
            overlaps = Polygon(lm.contour).buffer(0).intersects(view)
        except (ValueError, TypeError):
            overlaps = False
        lines.append(
            f"- {lm.name}: centroid {dist:.0f} m to the {_compass(dx, dy)}"
            + ("; its outline is inside this view" if overlaps else "; outside this view")
        )
    return "\n".join(lines)


def detection_concepts(analysis: dict) -> list[str]:
    """The phrases the detector is prompted with: the paper's VPG proposed the
    instruction's own phrases ("red car"), so the attributed target phrase,
    the surrounding phrases and the landmark nouns are prompted and the bare
    class never is. A bare class returned every car in view, eighty marks on
    a car instruction, and the probe measured the verifier hitting twice as
    often on fifteen."""
    target = analysis.get("target") or {}
    phrase = str(target.get("phrase") or "").strip()
    cls = str(target.get("class") or "").strip().lower()
    out = [phrase or cls]
    out += [str(x) for x in (analysis.get("surroundings") or [])]
    out += [str(x) for x in (analysis.get("landmarks") or [])]
    out += [str(x) for x in (analysis.get("concepts") or []) if str(x).strip().lower() != cls]
    return _dedup(out)


def _terms(phrases: list[str]) -> set[str]:
    """The content words of some phrases, for matching a detector label."""
    out: set[str] = set()
    for p in phrases:
        for t in re.findall(r"[a-z]+", str(p).lower()):
            if len(t) >= 3 and t not in ("the", "and", "with", "near", "side", "left", "right", "front", "behind"):
                out.add(t.rstrip("s"))
    return out


def mark_tier(label: str, target_terms: set[str], surround_terms: set[str]) -> int:
    """2 for a mask of the described object's class, 1 for a surrounding
    class the instruction names, 0 for anything else (landmark-class masks)."""
    words = {t.rstrip("s") for t in re.findall(r"[a-z]+", label.lower())}
    if words & target_terms:
        return 2
    if words & surround_terms:
        return 1
    return 0


def _dedup(seq):
    out = []
    for s in seq:
        s = str(s).strip()
        if s and s.lower() not in {o.lower() for o in out}:
            out.append(s)
    return out


def run_episode(h: BaselineHarness) -> BaselineResult:
    cfg = h.cfg
    instruction = h.episode.description
    analysis = h.llm_json(
        [
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": prompts.ANALYSIS_PROMPT.format(instruction=instruction)},
        ],
        prompts.ANALYSIS_SCHEMA,
        "visa_instruction_analysis",
        max_tokens=ANALYSIS_MAX_TOKENS,
    )
    target = analysis.get("target") or {}
    target_class = str(target.get("class") or "object")
    concepts = detection_concepts(analysis)
    matched = match_landmarks(list(analysis.get("landmarks") or []), h.landmarks())
    h.record("analysis", {"instruction": instruction, "analysis": analysis,
                          "matched_landmarks": [lm.name for lm in matched], "concepts": concepts})

    alt = float(h.start_alt_m)
    plan = plan_coverage(
        matched, h.search_window(matched, clip_to_block=False), h.episode.start_xy, alt, h.ground_z_m,
        fov_m=2.0 * h.view_half_short_m(alt),
    )
    h.record("plan", {
        "waypoints": plan.waypoints, "fov_m": plan.fov_m, "grid_step_m": plan.grid_step_m,
        "grid_points": plan.grid_points, "contours": plan.contours,
        "cells_total": plan.cells_total, "cells_covered": plan.cells_covered,
        "cell_m": plan.cell_m, "stats": plan.stats, "start_xy": list(h.episode.start_xy),
    })

    stops: list[tuple[tuple[float, float], float]] = []
    best: dict | None = None  # the strongest pending candidate: (score, conf, mark, step)
    step = 0
    guided_moves = 0
    altitude_changes = 0
    visited = 0
    alt_lo = h.ground_z_m + 20.0
    alt_hi = float(cfg.altitude_cap_m)

    def consider(marks: list[dict], vm: dict) -> None:
        nonlocal best
        stages = [
            set(vm.get("attribute_match", {}).get("candidate_marks") or []),
            set(vm.get("spatial_topology", {}).get("consistent_marks") or []),
            set(vm.get("geographic_boundary", {}).get("valid_marks") or []),
        ]
        # A mark the first stage deferred ("pending") is a candidate the
        # fallback may still commit, below every attribute-matched one.
        pending = set(vm.get("attribute_match", {}).get("pending_marks") or [])
        for m in marks:
            if m["id"] not in stages[0] and m["id"] not in pending:
                continue
            score = sum(m["id"] in s for s in stages)
            key = (score, m["confidence"])
            if best is None or key > (best["score"], best["confidence"]):
                best = {"score": score, "confidence": m["confidence"], "mark": m, "step": step}

    def verify(obs, marks: list[dict], marked, guidance: str | None) -> dict:
        nonlocal step
        step += 1
        half = h.view_half_m(obs.altitude_m)
        text = prompts.VERIFY_PROMPT.format(
            instruction=instruction,
            target_phrase=target.get("phrase") or target_class,
            target_class=target_class,
            target_attributes=", ".join(target.get("attributes") or []) or "none stated",
            view_side_m=2 * half,
            altitude_m=obs.altitude_m,
            legend=legend(marks),
            landmarks=landmark_text(matched, obs.pose_xy, half),
            guidance_block=prompts.guidance_block(guidance),
        )
        h.save_image(f"step_{step:02d}_marked", h.jpeg(marked))
        vm = h.llm_json(
            [{"role": "system", "content": prompts.SYSTEM}, h.user_message(text, [marked])],
            prompts.VERIFY_SCHEMA,
            "visa_verify",
            max_tokens=VERIFY_MAX_TOKENS,
        )
        h.record(f"step_{step:02d}_vm", {"pose_xy": list(obs.pose_xy), "altitude_m": obs.altitude_m,
                                       "guidance_in": guidance, "prompt": text, "response": vm})
        consider(marks, vm)
        return vm

    target_terms = _terms([target_class, target.get("phrase") or ""])
    surround_terms = _terms(list(analysis.get("surroundings") or []))

    def perceive(obs, extra: list[str], tag: str):
        dets = h.detect(obs, _dedup(concepts + extra))
        n_all = len(dets)
        # The cap cuts the way the paper's detector would list: the described
        # object's class first, its surroundings next, landmark-class masks
        # last, each tier by confidence.
        dets = sorted(dets, key=lambda d: (-mark_tier(str(d.label), target_terms, surround_terms), -float(d.confidence)))[:MAX_MARKS]
        marked, marks = mark_frame(obs.rgb, dets)
        h.record(f"step_{step + 1:02d}_vpg", {"concepts": _dedup(concepts + extra),
                                              "n_detections": n_all, "n_marked": len(dets),
                                              "marks": marks, "pass": tag})
        return marked, marks

    commit_xy = None
    commit_label = ""
    commit_tier = ""
    lock_step = None
    # The guidance signal of the last verification, the textual prompt of
    # the next perception round wherever the executor takes the UAV.
    guidance: str | None = None
    guidance_concepts: list[str] = []

    def signal(vm: dict) -> None:
        nonlocal guidance, guidance_concepts
        guidance = str(vm.get("guidance") or "").strip() or None
        guidance_concepts = [str(c) for c in (vm.get("guidance_concepts") or []) if str(c).strip()]

    for wp in plan.waypoints:
        if not h.budget.affords(wp, alt):
            commit_tier = "visa:budget"
            break
        h.budget.move(wp, alt)
        stops.append(((float(wp[0]), float(wp[1])), alt))
        visited += 1
        obs = h.observe(wp, alt)
        marked, marks = perceive(obs, guidance_concepts, "guided" if guidance else "initial")
        vm = verify(obs, marks, marked, guidance)
        changed_alt = False
        while True:
            verdict = vm.get("verdict")
            target_mark = int(vm.get("target_mark") or 0)
            by_id = {m["id"]: m for m in marks}
            if verdict == "stop" and target_mark in by_id:
                commit_xy = mark_world_xy(by_id[target_mark])
                commit_label = f"mark_{target_mark}"
                commit_tier = "visa:stop"
                lock_step = step
                break
            signal(vm)
            if verdict in ("ascend", "descend") and not changed_alt:
                new_alt = min(alt_hi, max(alt_lo, alt + (DELTA_Z_M if verdict == "ascend" else -DELTA_Z_M)))
                if abs(new_alt - alt) > 0.5 and h.budget.affords(wp, new_alt):
                    h.budget.move(wp, new_alt)
                    alt = new_alt
                    changed_alt = True
                    altitude_changes += 1
                    stops.append(((float(wp[0]), float(wp[1])), alt))
                    obs = h.observe(wp, alt)
                    marked, marks = perceive(obs, guidance_concepts, verdict)
                    vm = verify(obs, marks, marked, guidance)
                    continue
            break  # move: the next waypoint, the signal riding with it
        if commit_xy is not None:
            break
        if guidance:
            guided_moves += 1
    if commit_xy is None:
        # No stop verdict: the strongest pending candidate, else the last
        # stop. ``visa:budget`` stays when the budget ended the walk.
        if best is not None:
            commit_xy = mark_world_xy(best["mark"])
            commit_label = f"mark_{best['mark']['id']}@step{best['step']}"
            commit_tier = commit_tier or "visa:pending"
            lock_step = best["step"]
        else:
            commit_label = "last_stop"
            commit_tier = commit_tier or "visa:exhausted"
    h.record("result", {"commit_xy": commit_xy, "commit_label": commit_label, "commit_tier": commit_tier,
                        "lock_step": lock_step, "stops": stops, "actions_priced": h.budget.used})
    return BaselineResult(
        stops=stops,
        commit_xy=commit_xy,
        commit_label=commit_label,
        commit_tier=commit_tier,
        lock_step=lock_step,
        extra_metrics={
            "waypoints_planned": len(plan.waypoints),
            "waypoints_visited": visited,
            "guided_moves": guided_moves,
            "altitude_changes": altitude_changes,
            "landmarks_matched": len(matched),
        },
    )
