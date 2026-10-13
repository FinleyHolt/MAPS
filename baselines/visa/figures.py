"""Re-create the content of ViSA's Fig. 1 / Fig. 2 from a flown episode.

    python -m baselines.visa.figures <workdir> [--out DIR]
    python -m baselines.visa.figures --root <run root> [--out DIR]

Panel (a), one per verification step: the marked frame with its Set-of-Mark
legend and the three-stage reasoning trace beside it, the guidance signal
when one fired. Panel (b): the coverage plan over the block -- matched
contours, grid, chosen FOV squares in visit order -- and the flown replay
path, the start, the target and the commit from ``episode.json``. Reads only
what ``policy.py`` and the runner recorded.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def step_panels(workdir: Path, out: Path) -> list[Path]:
    vd = workdir / "visa"
    written = []
    for vm_path in sorted(vd.glob("step_*_vm.json")):
        k = vm_path.name.split("_")[1]
        vm = _load(vm_path)
        vpg = _load(vd / f"step_{k}_vpg.json") or {}
        img = vd / f"step_{k}_marked.jpg"
        if vm is None or not img.exists():
            continue
        r = vm["response"]
        fig, (ax_im, ax_tx) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1]})
        ax_im.imshow(Image.open(img))
        ax_im.set_axis_off()
        ax_im.set_title(f"step {k}: pose ({vm['pose_xy'][0]:.0f}, {vm['pose_xy'][1]:.0f}) at {vm['altitude_m']:.0f} m")
        marks = vpg.get("marks", [])
        # The legend names the marks the verification cites (and the target),
        # not every candidate: eighty lines would push the trace off the panel.
        cited = set(r["attribute_match"]["candidate_marks"]) | set(r["spatial_topology"]["consistent_marks"]) \
            | set(r["geographic_boundary"]["valid_marks"]) | {r.get("target_mark")}
        shown = [m for m in marks if m["id"] in cited][:24]
        legend = "\n".join(f"[{m['id']}] {m['label']}{', ' + m['color'] if m.get('color') else ''}" for m in shown)
        legend += f"\n({len(marks)} marks in frame; {len(shown)} cited shown)"
        trace = [
            "Set-of-Mark legend:", legend, "",
            "1. Literal attribute matching -> " + str(r["attribute_match"]["candidate_marks"]),
            textwrap.fill(r["attribute_match"]["reasoning"], 70), "",
            "2. Spatial topology -> " + str(r["spatial_topology"]["consistent_marks"]),
            textwrap.fill(r["spatial_topology"]["reasoning"], 70), "",
            "3. Geographic boundary -> " + str(r["geographic_boundary"]["valid_marks"]),
            textwrap.fill(r["geographic_boundary"]["reasoning"], 70), "",
            f"verdict: {r['verdict']}  target_mark: {r['target_mark']}",
        ]
        if vm.get("guidance_in"):
            trace.insert(0, "guidance in: " + textwrap.fill(vm["guidance_in"], 70))
        if r.get("guidance"):
            trace.append("guidance out: " + textwrap.fill(r["guidance"], 70))
        ax_tx.set_axis_off()
        ax_tx.text(0.0, 1.0, "\n".join(trace), va="top", ha="left", family="monospace", fontsize=8)
        p = out / f"{workdir.name}_step_{k}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        written.append(p)
    return written


def plan_panel(workdir: Path, out: Path) -> Path | None:
    plan = _load(workdir / "visa" / "plan.json")
    if plan is None:
        return None
    ep = _load(workdir / "episode.json") or {}
    fig, ax = plt.subplots(figsize=(10, 10))
    for c in plan["contours"]:
        xs, ys = zip(*c)
        ax.fill(xs, ys, alpha=0.25, color="tab:gray")
        ax.plot(list(xs) + [xs[0]], list(ys) + [ys[0]], color="tab:gray", lw=1)
    gx, gy = zip(*plan["grid_points"]) if plan["grid_points"] else ([], [])
    ax.scatter(gx, gy, s=6, color="lightgray", label=f"grid (step {plan['grid_step_m']:.0f} m)")
    half = plan["fov_m"] / 2
    for i, (x, y) in enumerate(plan["waypoints"]):
        ax.add_patch(Rectangle((x - half, y - half), 2 * half, 2 * half, fill=False, ec="tab:blue", lw=1))
        ax.text(x, y, str(i + 1), ha="center", va="center", color="tab:blue", fontsize=10, weight="bold")
    if plan["waypoints"]:
        wx, wy = zip(*plan["waypoints"])
        ax.plot(wx, wy, "--", color="tab:blue", lw=1, label="planned order")
    vis = ep.get("visited_xy") or []
    if vis:
        vx, vy = zip(*vis)
        ax.plot(vx, vy, color="tab:orange", lw=1.5, label="flown replay")
    if ep.get("start_xy"):
        ax.plot(*ep["start_xy"], marker="^", color="black", ms=10, ls="none", label="start")
    if ep.get("target_xy"):
        ax.plot(*ep["target_xy"], marker="*", color="tab:green", ms=14, ls="none", label="target")
    if ep.get("commit_xy"):
        ax.plot(*ep["commit_xy"], marker="x", color="tab:red", ms=10, ls="none", label="commit")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{workdir.name}: coverage plan, FOV {plan['fov_m']:.0f} m, "
                 f"{plan['cells_covered']}/{plan['cells_total']} cells covered")
    p = out / f"{workdir.name}_plan.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def render(workdir: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = step_panels(workdir, out)
    p = plan_panel(workdir, out)
    return paths + ([p] if p else [])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir", nargs="?", help="one attempt directory")
    ap.add_argument("--root", help="a run root: every attempt directory with a visa/ folder")
    ap.add_argument("--out", help="output directory (default <root or workdir>/figures)")
    a = ap.parse_args(argv)
    if not a.workdir and not a.root:
        ap.error("give a workdir or --root")
    roots = [Path(a.workdir)] if a.workdir else sorted(p.parent for p in Path(a.root).glob("*/visa"))
    out = Path(a.out) if a.out else (Path(a.root) if a.root else Path(a.workdir)) / "figures"
    n = 0
    for wd in roots:
        for p in render(wd, out):
            print(p)
            n += 1
    print(f"{n} panel(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
