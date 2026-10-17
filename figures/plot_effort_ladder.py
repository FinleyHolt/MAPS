"""Draw a reasoning-effort ladder as one dual-axis SVG, light and dark.

One arm per run root, one model, policy and withhold set over a swept effort
(roots that differ in policy or withhold are refused): success rate on the left
axis, median navigation error on the right, as two static SVGs.

It defines no metric. ``arm_table.table`` is the canonical reader and hands
back exactly the row a printed table shows, so the figure and the table
agree. The title names the split and the subtitle the served model, arm
count, episodes per arm and commit, all read off the run manifests.

Usage:
    python -m figures.plot_effort_ladder <root> [<root> ...] [--out-dir output/figures]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

from benchmarks.citynav.analysis import arm_table
from benchmarks.citynav.analysis import rows as _rows
from figures import FONT, MONO, bench_label, esc, themes
from figures.capability_effort_arms import effort_value

#: Geometry, in viewBox units. One coordinate system for both themes.
W, H = 920, 512
L, R, T, B = 92, 826, 58, 372

#: Left axis is success, right axis is distance. Each axis takes its series'
#: colour -- with two scales on one frame, a shared neutral tick makes the
#: reader guess which line it belongs to.
THEMES = themes(light={"grid": "#E1E5EA", "sr": "#B8560D", "ne": "#26708F"},
                dark={"grid": "#272E38", "sr": "#F0A44C", "ne": "#63B6DA"})


def axis(values: list[float], step: int) -> tuple[int, int]:
    """The data's range widened to whole steps, a step at least."""
    lo = int(math.floor(min(values) / step) * step)
    hi = int(math.ceil(max(values) / step) * step)
    return lo, max(hi, lo + step)


def _duplicates(rows: list[dict]) -> set[int]:
    """Indices of a second arm at an effort already on the line.

    A second arm at one effort is a second draw, not a second x position.
    Joining it bends the line back on itself and reads as a real
    non-monotonicity, so it is drawn open and left off the line.
    """
    seen: set[float] = set()
    out: set[int] = set()
    for i, r in enumerate(rows):
        if r["effort"] in seen:
            out.add(i)
        seen.add(r["effort"])
    return out


def render(rows: list[dict], theme: str, title: str, subtitle: str) -> str:
    c = THEMES[theme]
    title, subtitle = esc(title), esc(subtitle)
    dup = {id(rows[i]) for i in _duplicates(rows)}
    is_dup = lambda r: id(r) in dup  # noqa: E731
    sr_lo, sr_hi = axis([r["SR_pct"] for r in rows], 10)
    ne_lo, ne_hi = axis([0.0] + [r["NE_med"] for r in rows], 10)
    e_lo, e_hi = 0.0, max(1.0, max(r["effort"] for r in rows))
    sx = lambda e: L + (R - L) * (e - e_lo) / (e_hi - e_lo)  # noqa: E731
    sy_sr = lambda v: B - (B - T) * (v - sr_lo) / (sr_hi - sr_lo)  # noqa: E731
    sy_ne = lambda v: B - (B - T) * (v - ne_lo) / (ne_hi - ne_lo)  # noqa: E731

    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" font-family="{FONT}" '
        f'role="img" aria-label="{title}: success rate and navigation error '
        f'against reasoning effort">',
        f'<rect width="{W}" height="{H}" fill="{c["bg"]}"/>',
    ]

    # ---- title -------------------------------------------------------
    p.append(
        f'<text x="{L}" y="26" fill="{c["ink"]}" font-size="17">'
        f'{title}: success and error against reasoning effort</text>'
    )
    p.append(
        f'<text x="{L}" y="44" fill="{c["muted"]}" font-size="12.5">'
        f'{subtitle}</text>'
    )

    # ---- grid and left axis ------------------------------------------
    for v in range(sr_lo, sr_hi + 1, 10):
        y = sy_sr(v)
        p.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" '
                 f'stroke="{c["grid"]}" stroke-width="1"/>')
        p.append(f'<text x="{L - 12}" y="{y + 4:.1f}" text-anchor="end" '
                 f'fill="{c["sr"]}" font-size="12" font-family="{MONO}">{v}</text>')
    # ---- right axis ---------------------------------------------------
    for v in range(ne_lo, ne_hi + 1, 10):
        y = sy_ne(v)
        p.append(f'<line x1="{R}" y1="{y:.1f}" x2="{R + 6}" y2="{y:.1f}" '
                 f'stroke="{c["spine"]}" stroke-width="1"/>')
        p.append(f'<text x="{R + 13}" y="{y + 4:.1f}" fill="{c["ne"]}" '
                 f'font-size="12" font-family="{MONO}">{v}</text>')

    # ---- x axis -------------------------------------------------------
    for i in range(11):
        e = e_lo + (e_hi - e_lo) * i / 10
        x = sx(e)
        p.append(f'<line x1="{x:.1f}" y1="{B}" x2="{x:.1f}" y2="{B + 6}" '
                 f'stroke="{c["spine"]}" stroke-width="1"/>')
        p.append(f'<text x="{x:.1f}" y="{B + 22}" text-anchor="middle" '
                 f'fill="{c["muted"]}" font-size="12" font-family="{MONO}">'
                 f'{e:.1f}</text>')
    p.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" '
             f'stroke="{c["spine"]}" stroke-width="1.4"/>')
    p.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{B}" '
             f'stroke="{c["spine"]}" stroke-width="1.4"/>')
    p.append(f'<line x1="{R}" y1="{T}" x2="{R}" y2="{B}" '
             f'stroke="{c["spine"]}" stroke-width="1.4"/>')

    p.append(f'<text x="{(L + R) / 2:.0f}" y="{B + 48}" text-anchor="middle" '
             f'fill="{c["body"]}" font-size="13.5">reasoning effort</text>')
    p.append(f'<text x="{L - 54}" y="{(T + B) / 2:.0f}" text-anchor="middle" '
             f'fill="{c["sr"]}" font-size="13.5" '
             f'transform="rotate(-90 {L - 54} {(T + B) / 2:.0f})">'
             f'success rate, %</text>')
    p.append(f'<text x="{R + 62}" y="{(T + B) / 2:.0f}" text-anchor="middle" '
             f'fill="{c["ne"]}" font-size="13.5" '
             f'transform="rotate(90 {R + 62} {(T + B) / 2:.0f})">'
             f'median error, m</text>')

    # ---- series -------------------------------------------------------
    for key, ycvt, colour, shape in (
        ("SR_pct", sy_sr, c["sr"], "circle"),
        ("NE_med", sy_ne, c["ne"], "square"),
    ):
        line = [r for r in rows if not is_dup(r)]
        pts = " ".join(f'{sx(r["effort"]):.1f},{ycvt(r[key]):.1f}' for r in line)
        p.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                 f'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>')
        for r in rows:
            x, y = sx(r["effort"]), ycvt(r[key])
            fill = c["bg"] if is_dup(r) else colour
            sw = 2.2 if is_dup(r) else 0
            if shape == "circle":
                p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}" '
                         f'stroke="{colour}" stroke-width="{sw}"/>')
            else:
                p.append(f'<rect x="{x - 4.4:.1f}" y="{y - 4.4:.1f}" width="8.8" '
                         f'height="8.8" fill="{fill}" stroke="{colour}" '
                         f'stroke-width="{sw}"/>')

    # ---- endpoint labels ---------------------------------------------
    # Only the right-hand endpoints are labelled. At x=0 the two series cross,
    # so a value there lands on the other line whichever side it is placed.
    last = max(rows, key=lambda r: r["effort"])
    p.append(f'<text x="{sx(last["effort"]) - 10:.0f}" y="{sy_sr(last["SR_pct"]) - 12:.0f}" '
             f'text-anchor="end" fill="{c["sr"]}" font-size="12.5" '
             f'font-family="{MONO}">{last["SR_pct"]:.1f}%</text>')
    p.append(f'<text x="{sx(last["effort"]) - 14:.0f}" y="{sy_ne(last["NE_med"]) - 12:.0f}" '
             f'text-anchor="end" fill="{c["ne"]}" font-size="12.5" '
             f'font-family="{MONO}">{last["NE_med"]:.1f} m</text>')
    # ---- legend, below the frame so no series can collide with it -----
    ly = B + 82
    p.append(f'<line x1="{L}" y1="{ly}" x2="{L + 26}" y2="{ly}" '
             f'stroke="{c["sr"]}" stroke-width="2.4"/>')
    p.append(f'<circle cx="{L + 13}" cy="{ly}" r="5" fill="{c["sr"]}"/>')
    p.append(f'<text x="{L + 34}" y="{ly + 4}" fill="{c["body"]}" font-size="12.5">'
             f'success rate (left axis)</text>')
    p.append(f'<line x1="{L + 190}" y1="{ly}" x2="{L + 216}" y2="{ly}" '
             f'stroke="{c["ne"]}" stroke-width="2.4"/>')
    p.append(f'<rect x="{L + 198.6}" y="{ly - 4.4}" width="8.8" height="8.8" '
             f'fill="{c["ne"]}"/>')
    p.append(f'<text x="{L + 224}" y="{ly + 4}" fill="{c["body"]}" font-size="12.5">'
             f'median error (right axis)</text>')
    if dup:
        p.append(f'<circle cx="{L + 400}" cy="{ly}" r="5" fill="{c["bg"]}" '
                 f'stroke="{c["sr"]}" stroke-width="2.2"/>')
        p.append(f'<text x="{L + 414}" y="{ly + 4}" fill="{c["body"]}" font-size="12.5">'
                 f'a second arm at an effort already on the line</text>')

    p.append("</svg>")
    return "".join(p)


def heading(rows: list[dict]) -> tuple[str, str]:
    """(title, subtitle) off the arms' own run manifests."""
    mans = [_rows.read_manifest(Path(r["root"])) for r in rows]
    models = sorted({str(m.get("served_model") or "unrecorded model").rstrip("/").rsplit("/", 1)[-1]
                     for m in mans})
    splits = sorted({str(m.get("split") or "") for m in mans} - {""})
    commits = sorted({str(m.get("git_commit") or "")[:7] for m in mans} - {""})
    policies = sorted({str((yaml.safe_load(m.get("config_yaml") or "{}") or {}).get("policy") or "")
                       for m in mans} - {""})
    sizes = sorted({r["n"] for r in rows})
    title = " / ".join(bench_label(s) for s in splits) or "Effort ladder"
    n = f"{sizes[0]:,}" if len(sizes) == 1 else f"{sizes[0]:,} to {sizes[-1]:,}"
    sub = (f"{', '.join(models)}{', policy ' + ', '.join(policies) if policies else ''}, "
           f"{len(rows)} arms, {n} episodes each, commit {', '.join(commits) or 'unrecorded'}")
    return title, sub


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path, help="run roots, one per arm")
    ap.add_argument("--out-dir", type=Path, default=Path("output/figures"))
    ap.add_argument("--stem", default=None, help="file stem (default effort_ladder_<split>)")
    _rows.add_allow_partial(ap)
    args = ap.parse_args(argv)

    rows = []
    for r in arm_table.table(args.roots, allow_partial=args.allow_partial):
        effort = effort_value(r["effort"])
        if isinstance(effort, float):
            rows.append({**r, "effort": effort})
        else:
            print(f"plot_effort_ladder: {r['root']} left out: effort {r['effort']!r} is not numeric", file=sys.stderr)
    # Vacuity floor: this figure is two lines over a swept knob. A single
    # point, or a row with no metric, is not a ladder and must not render as
    # a confident-looking empty frame.
    if len(rows) < 3:
        raise SystemExit(f"plot_effort_ladder: {len(rows)} arms carry a numeric effort; need >= 3")
    for key in ("SR_pct", "NE_med"):
        missing = [r["label"] for r in rows if r.get(key) is None]
        if missing:
            raise SystemExit(f"plot_effort_ladder: {key} absent for {missing}")
    # One ladder is one arm swept over effort: two policies or withhold sets
    # would draw as one line.
    cfgs = [yaml.safe_load(_rows.read_manifest(Path(r["root"])).get("config_yaml") or "{}") or {}
            for r in rows]
    arms = {(str(c.get("policy") or ""), tuple(sorted(c.get("withhold") or ()))) for c in cfgs}
    if len(arms) > 1:
        raise SystemExit(f"plot_effort_ladder: the roots differ in policy or withhold {sorted(arms)}; "
                         "plot one arm per ladder")

    title, sub = heading(rows)
    splits = sorted({str(_rows.read_manifest(Path(r["root"])).get("split") or "") for r in rows} - {""})
    stem = args.stem or "_".join(["effort_ladder", *splits])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        path = args.out_dir / f"{stem}_{theme}.svg"
        path.write_text(render(rows, theme, title, sub))
        print(f"wrote {path}  ({len(rows)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
