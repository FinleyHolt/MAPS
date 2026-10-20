"""The team arm's root: the MAPS root's rows, each judged by its auditor.

An auditor confirms MAPS's commit or replaces it with another point. A
confirmed row is the MAPS row unchanged, so the team and MAPS score the same
episode the same way. A replaced row flies one more leg, from where MAPS
ended to the auditor's point, priced in the six-action space against the
budget MAPS left, and is scored there by ``eval.score_arrival``. The root
this writes reads like any bench root: ``run_report`` tables it, and
``ab_report`` pairs it with the MAPS root (policy ``audit``) or with a solo
human root.

    python -m baselines.human.team_rows --maps-root output/e2/<root> \\
        --decisions output/human_study/<study>/p01/decisions.jsonl --out-root output/baselines/audit/p01
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from baselines.human.session_plan import scored_rows
from benchmarks.citynav.action_snap import Pose, snap_to_actions
from benchmarks.citynav.analysis.rows import iter_json_rows
from benchmarks.citynav.eval import score_arrival


def select_bench(maps_root: Path) -> None:
    """Point the shared engine's paths at the bench the root flew, before
    anything imports them: a Cal City root reads Cal City's split, catalog,
    bake and imagery."""
    if json.loads((maps_root / "run_manifest.json").read_text()).get("split") == "calcity":
        import benchmarks.calcity

        benchmarks.calcity.use()


class TargetShapes:
    """Each episode's success footprint, as the runner resolves it
    (``episode_runner.success_shape``), off the root's split and catalog."""

    def __init__(self, split: str) -> None:
        from benchmarks.citynav.adapter import load_objects, load_split
        from benchmarks.citynav.paths import OBJECTS_JSON, split_path

        self.rows = load_split(split_path(split))
        self.objects = load_objects(OBJECTS_JSON)

    def __call__(self, episode_id: str):
        from benchmarks.citynav.adapter import find_row, parse_episode
        from benchmarks.citynav.run_benchmark.episode_runner import success_shape

        found = find_row(self.rows, episode_id)
        if found is None:
            raise ValueError(f"{episode_id}: not in the split the root flew")
        index, row = found
        return success_shape(parse_episode(row, row_index=index), self.objects)


def root_config(maps_root: Path) -> dict:
    """The resolved config the MAPS root flew, off its manifest."""
    return yaml.safe_load(json.loads((maps_root / "run_manifest.json").read_text())["config_yaml"])


def correction_leg(row: dict, xy: tuple[float, float]) -> tuple[int, tuple[float, float]]:
    """The actions from where MAPS ended to ``xy``, and the point they reach.

    ponytail: the row keeps positions only, so the leg flies at the final
    altitude with the heading of the last flown segment; a row that recorded
    its final pose would price the leg's opening turns exactly."""
    path = [tuple(map(float, p)) for p in row.get("visited_xy") or [row["start_xy"]]]
    end = path[-1]
    prev = next((p for p in reversed(path) if p != end), end)
    heading = math.degrees(math.atan2(end[0] - prev[0], end[1] - prev[1])) % 360.0
    actions, reached = snap_to_actions(Pose(end[0], end[1], 0.0, heading), Pose(float(xy[0]), float(xy[1]), 0.0, heading))
    return len(actions), (reached.x, reached.y)


def affordable(row: dict, xy: tuple[float, float], max_steps: int) -> bool:
    """Whether the budget MAPS left flies the leg to ``xy``."""
    return int(row["metrics"]["steps_taken"]) + correction_leg(row, xy)[0] <= max_steps


def team_row(row: dict, decision: dict, cfg: dict, shape_of=None) -> dict:
    """The MAPS row as its auditor left it. ``shape_of`` (episode id ->
    footprint) scores a replacement on a shape bench."""
    out = json.loads(json.dumps(row))
    m = out["metrics"]
    audit = {"audit_decision": decision["decision"], "audit_review_s": decision.get("review_s"),
             "audit_trace_turns_opened": len(decision.get("opened") or []),
             "audit_candidate": decision.get("candidate")}
    if decision["decision"] == "replace":
        criterion = m.get("success_criterion")
        if criterion == "shape" and shape_of is None:
            raise ValueError(f"{row['episode_id']}: a shape row is scored on its target's footprint: pass shape_of")
        xy = (float(decision["xy"][0]), float(decision["xy"][1]))
        n, reached = correction_leg(row, xy)
        steps = int(m["steps_taken"]) + n
        if steps > int(cfg["max_steps"]):
            raise ValueError(f"{row['episode_id']}: the replacement leg needs {n} actions past the budget")
        visited = [tuple(p) for p in out["visited_xy"]] + [reached]
        m.update(score_arrival(criterion=criterion, final_pos_xy=reached, target_xy=tuple(out["target_xy"]),
                               pos_trajectory=visited, success_dist_m=float(cfg["success_radius_m"]),
                               gt_polygon=shape_of(row["episode_id"]) if criterion == "shape" else None,
                               commit_xy=xy))
        m["steps_taken"] = steps
        out["visited_xy"] = [list(p) for p in visited]
        out["commit_xy"] = list(xy)
    m.update(audit, policy="audit", commit_path=f"audit:{decision['decision']}")
    m["maps_SR"] = row["metrics"]["SR"]
    return out


def write_root(maps_root: Path, decisions: list[dict], out_root: Path) -> list[dict]:
    """The team root: one row per decided episode, and the MAPS manifest with policy ``audit``."""
    rows = scored_rows(maps_root)
    cfg = root_config(maps_root)
    manifest = json.loads((maps_root / "run_manifest.json").read_text())
    manifest["config_yaml"] = yaml.safe_dump({**cfg, "policy": "audit"}, sort_keys=False)
    manifest["out_root"] = str(out_root.resolve())
    manifest["audit"] = {"maps_root": str(maps_root.resolve()), "decisions": len(decisions)}
    # The team root holds the decided episodes, not the MAPS root's whole selection.
    manifest["selection"] = {**(manifest.get("selection") or {}), "episodes_n": len(decisions), "allowlist": "audit decisions"}
    shapes = TargetShapes(manifest["split"]) if any(
        rows[d["episode_id"]]["metrics"].get("success_criterion") == "shape" and d["decision"] == "replace"
        for d in decisions) else None
    out = [team_row(rows[d["episode_id"]], d, cfg, shapes) for d in decisions]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=1))
    (out_root / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--maps-root", type=Path, required=True)
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args(argv)
    select_bench(args.maps_root)
    # The last decision per episode: a resumed session may have re-judged one.
    decided = {d["episode_id"]: d for d in iter_json_rows(args.decisions)}
    rows = write_root(args.maps_root, list(decided.values()), args.out_root)
    print(f"{len(rows)} team rows -> {args.out_root / 'episodes.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
