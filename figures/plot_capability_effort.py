"""Draw success against inference cost for one split, light and dark.

Every arm is a point: what it spent per episode, what it scored. A model with
several efforts is a trajectory through its rungs, joined cheapest to
dearest; a model with one arm is one open marker. Every mark carries its
model's name, as the run manifest records it.

Cost is the x axis, since effort dials differ across models: ``--cost
completion`` is mean completion tokens per episode, ``--cost total`` mean
prompt plus completion tokens. One split per figure; the numbers come from
``capability_effort_arms.build``.

Usage:
    python -m figures.plot_capability_effort --split test_unseen <root>...
    python -m figures.plot_capability_effort --split calcity --cost total <root>...
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from benchmarks.citynav.analysis import rows as _rows
from figures import FONT, MONO, bench_label, esc, themes
from figures.capability_effort_arms import build

W, H = 940, 540
L, R = 74, 620
TOP, BOT = 108, 424

SR_LO, SR_HI, SR_STEP = 0, 100, 20

THEMES = themes(light={"grid": "#E8ECF0", "arm": "#2a78d6"},
                dark={"grid": "#242B34", "arm": "#3987e5"})

COST_KEY = {"completion": "completion_tok_mean", "total": "total_tok_mean"}
COST_LABEL = {"completion": "completion tokens per episode (mean)",
              "total": "prompt + completion tokens per episode (mean)"}

#: The order named effort rungs are read in; an unlisted rung sorts last.
RUNG_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh")

#: Minimum vertical separation between two labels, in viewBox units.
LABEL_PITCH = 22.0


def token_ticks(values: list[float]) -> list[float]:
    """1-3-10 ticks spanning the data, one decade at least."""
    lo = math.floor(math.log10(max(min(values), 1.0)))
    hi = max(math.ceil(math.log10(max(max(values), 1.0))), lo + 1)
    ticks = [m * 10 ** e for e in range(lo, hi) for m in (1, 3)]
    return ticks + [10.0 ** hi]


def tick_label(t: float) -> str:
    return f"{t / 1000:g}k" if t >= 1000 else f"{t:g}"


def sy(pct: float) -> float:
    return BOT - (BOT - TOP) * (pct - SR_LO) / (SR_HI - SR_LO)


def make_sx(ticks: list[float]):
    lo, hi = math.log10(ticks[0]), math.log10(ticks[-1])

    def sx(tokens: float) -> float:
        v = min(max(math.log10(max(tokens, 1.0)), lo), hi)
        return L + (R - L) * (v - lo) / (hi - lo)
    return sx


def _rung_key(effort) -> tuple:
    if isinstance(effort, (int, float)):
        return (0, float(effort))
    name = str(effort)
    return (1, RUNG_ORDER.index(name) if name in RUNG_ORDER else len(RUNG_ORDER))


def dial_range(arms: list[dict]) -> str:
    """The span a trajectory swept, read off the rung order, not the cost order."""
    efforts = sorted((a["effort"] for a in arms if a["effort"] is not None), key=_rung_key)
    if not efforts:
        return ""
    fmt = lambda v: f"{v:g}" if isinstance(v, float) else str(v)  # noqa: E731
    lo, hi = fmt(efforts[0]), fmt(efforts[-1])
    return f"effort {lo}" if lo == hi else f"effort {lo} → {hi}"


def spread(labels: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """Push labels apart so none sits on another."""
    out: list[tuple[float, str]] = []
    for y, text in sorted(labels):
        if out and y - out[-1][0] < LABEL_PITCH:
            y = out[-1][0] + LABEL_PITCH
        out.append((y, text))
    return out


def render(data: dict, theme: str, split: str, cost: str = "completion") -> str:
    c = THEMES[theme]
    key = COST_KEY[cost]
    arms = [r for r in data["records"]
            if r["split"] == split and r["n"] and r[key] and r["sr"] is not None]
    ticks = token_ticks([r[key] for r in arms] or [100.0, 1000.0])
    sx = make_sx(ticks)
    crit = sorted({str(r["criterion"]) for r in arms if r["criterion"]})
    sizes = sorted({r["n"] for r in arms})
    priced = sorted({r["priced_n"] for r in arms})
    sub = (f"{'/'.join(crit) or 'unrecorded'} criterion · "
           f"{', '.join(f'{n:,}' for n in sizes) or 0} episodes per arm · "
           f"{', '.join(f'{n:,}' for n in priced) or 0} priced per arm · "
           f"{len(arms)} arms")
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" role="img" aria-label="Success rate against '
        f'{esc(COST_LABEL[cost])} on {esc(bench_label(split))}.">',
        f'<rect width="{W}" height="{H}" fill="{c["bg"]}"/>',
        f'<text x="{L}" y="34" font-family="{FONT}" font-size="17" '
        f'font-weight="600" fill="{c["ink"]}">Success against inference cost</text>',
        f'<text x="{L}" y="{TOP - 30}" font-family="{FONT}" font-size="13" '
        f'font-weight="600" fill="{c["ink"]}">{esc(bench_label(split))}</text>',
        f'<text x="{L}" y="{TOP - 14}" font-family="{MONO}" font-size="9.5" '
        f'fill="{c["muted"]}">{esc(sub)}</text>',
    ]
    for pct in range(SR_LO, SR_HI + 1, SR_STEP):
        y = sy(pct)
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" '
                   f'stroke="{c["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{L - 10}" y="{y + 3.5}" text-anchor="end" '
                   f'font-family="{MONO}" font-size="10" fill="{c["muted"]}">{pct}</text>')
    for tok in ticks:
        x = sx(tok)
        out.append(f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{BOT}" '
                   f'stroke="{c["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{BOT + 17}" text-anchor="middle" '
                   f'font-family="{MONO}" font-size="10" '
                   f'fill="{c["muted"]}">{tick_label(tok)}</text>')
    out.append(f'<line x1="{L}" y1="{TOP}" x2="{L}" y2="{BOT}" stroke="{c["spine"]}"/>')
    out.append(f'<line x1="{L}" y1="{BOT}" x2="{R}" y2="{BOT}" stroke="{c["spine"]}"/>')

    by_model: dict[str, list] = {}
    for r in arms:
        by_model.setdefault(r["model"], []).append(r)
    margin: list[tuple[float, str, float, float, str]] = []
    beside: list[tuple[float, str, float, float, str]] = []
    for model, group in sorted(by_model.items()):
        group.sort(key=lambda r: r[key])
        pts = [(sx(r[key]), sy(r["sr"])) for r in group]
        if len(pts) > 1:
            d = " ".join(("M" if j == 0 else "L") + f"{x:.1f},{y:.1f}"
                         for j, (x, y) in enumerate(pts))
            out.append(f'<path d="{d}" fill="none" stroke="{c["arm"]}" '
                       f'stroke-width="1.8" stroke-linejoin="round" opacity="0.85"/>')
            for x, y in pts[:-1]:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" '
                           f'fill="{c["arm"]}" opacity="0.75"/>')
            x, y = pts[-1]
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{c["arm"]}" '
                       f'stroke="{c["bg"]}" stroke-width="1.6"/>')
            margin.append((y, model, x, y, dial_range(group)))
        else:
            x, y = pts[0]
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.6" fill="{c["bg"]}" '
                       f'stroke="{c["arm"]}" stroke-width="1.8"/>')
            beside.append((y, model, x, y, dial_range(group)))

    def place(group: list, label_x) -> None:
        anchors = {text: (ax, ay, dial) for _y, text, ax, ay, dial in group}
        for ly, text in spread([(y, text) for y, text, _ax, _ay, _d in group]):
            ax, ay, dial = anchors[text]
            lx = label_x(ax)
            out.append(f'<text x="{lx:.1f}" y="{ly + 3.5:.1f}" font-family="{FONT}" '
                       f'font-size="9.5" fill="{c["ink"]}">{esc(text)}</text>')
            if dial:
                out.append(f'<text x="{lx:.1f}" y="{ly + 13:.1f}" font-family="{MONO}" '
                           f'font-size="8" fill="{c["muted"]}">{esc(dial)}</text>')
            if abs(ly - ay) > 2.0:
                out.append(f'<line x1="{lx - 3:.1f}" y1="{ly:.1f}" x2="{ax + 6:.1f}" '
                           f'y2="{ay:.1f}" stroke="{c["spine"]}" stroke-width="0.8"/>')

    place(margin, lambda _ax: R + 9)
    place(beside, lambda ax: ax + 9)
    if not arms:
        out.append(f'<text x="{(L + R) / 2:.1f}" y="{(TOP + BOT) / 2:.1f}" '
                   f'text-anchor="middle" font-family="{FONT}" font-size="11" '
                   f'fill="{c["muted"]}">no scored arm on this split</text>')

    out.append(f'<text x="{L - 52}" y="{(TOP + BOT) / 2:.1f}" '
               f'transform="rotate(-90 {L - 52} {(TOP + BOT) / 2:.1f})" '
               f'text-anchor="middle" font-family="{FONT}" font-size="11" '
               f'fill="{c["body"]}">success rate, %</text>')
    out.append(f'<text x="{(L + R) / 2:.1f}" y="{BOT + 40}" text-anchor="middle" '
               f'font-family="{FONT}" font-size="11" fill="{c["body"]}">'
               f'{esc(COST_LABEL[cost])}</text>')
    ly = BOT + 74
    out.append(f'<path d="M{L},{ly} L{L + 30},{ly}" stroke="{c["arm"]}" stroke-width="1.8"/>')
    out.append(f'<circle cx="{L + 30}" cy="{ly}" r="5" fill="{c["arm"]}"/>')
    out.append(f'<text x="{L + 42}" y="{ly + 4}" font-family="{FONT}" font-size="10" '
               f'fill="{c["body"]}">one model across its efforts, cheapest to dearest</text>')
    out.append(f'<circle cx="{L + 378}" cy="{ly}" r="4.6" fill="{c["bg"]}" '
               f'stroke="{c["arm"]}" stroke-width="1.8"/>')
    out.append(f'<text x="{L + 390}" y="{ly + 4}" font-family="{FONT}" font-size="10" '
               f'fill="{c["body"]}">a model with one arm</text>')
    out.append(f'<text x="{L}" y="{H - 14}" font-family="{MONO}" font-size="9" '
               f'fill="{c["muted"]}">SR via benchmarks.citynav.analysis.arm_table · '
               f'cost via the picked attempts\' metrics.llm</text>')
    out.append("</svg>")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path, help="run roots, one per arm")
    ap.add_argument("--split", required=True, help="the split to draw (test_unseen, calcity, ...)")
    ap.add_argument("--cost", choices=sorted(COST_KEY), default="completion")
    ap.add_argument("--out-dir", type=Path, default=Path("output/figures"))
    ap.add_argument("--stem", default=None)
    _rows.add_allow_partial(ap)
    args = ap.parse_args(argv)

    data = build([r for r in args.roots if r.is_dir()], allow_partial=args.allow_partial)
    splits = sorted({str(r["split"]) for r in data["records"]})
    if args.split not in splits:
        print(f"plot_capability_effort: no root carries split {args.split!r}; the roots carry {splits}",
              file=sys.stderr)
        return 2
    stem = args.stem or f"capability_effort_{args.split}_{args.cost}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        path = args.out_dir / f"{stem}_{theme}.svg"
        path.write_text(render(data, theme, args.split, args.cost))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
