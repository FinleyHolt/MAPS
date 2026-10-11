"""Figures drawn from run roots through the benchmark readers, and their shared SVG vocabulary.

Run from the repo root: ``python -m figures.plot_effort_ladder <root>...``.
A figure owns scales, ticks and paths; every number comes from
``benchmarks.citynav.analysis``. Every colour is written into the file and the
type is a plain system stack, so either theme displays anywhere.
"""

FONT = "'DejaVu Sans','Helvetica Neue',Helvetica,Arial,sans-serif"
MONO = "'DejaVu Sans Mono','SFMono-Regular',Menlo,Consolas,monospace"

#: The neutrals both themes share; each figure adds its grid and series colours.
NEUTRALS = {
    "light": {"bg": "#FAFBFC", "ink": "#161A22", "body": "#404955", "muted": "#69727F",
              "spine": "#B9C1CB"},
    "dark": {"bg": "#161A21", "ink": "#E7EBF1", "body": "#BAC2CE", "muted": "#8B95A3",
             "spine": "#3A434F"},
}


def themes(**extra: dict) -> dict:
    """``NEUTRALS`` with each theme's figure-specific colours added."""
    return {t: {**base, **extra[t]} for t, base in NEUTRALS.items()}


def esc(t) -> str:
    return (str(t).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def bench_label(split: str) -> str:
    return "Cal City" if split == "calcity" else f"CityNav {split}"
