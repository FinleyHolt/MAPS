"""Every identity-prefixed env read in the run path is a DECLARED knob.

The manifest can only record what the vocabulary names: an env var read by
run-path code but absent from ``benchmarks.citynav.run_manifest``'s classes
is an invisible knob — exactly the undocumented-approach hazard the manifest
exists to close. This derived scan holds the two in lockstep: add a
``SEARCH_*`` / ``CITYNAV_BENCH_*`` / ``BENCH_*`` read anywhere under
``maps/`` / ``benchmarks/`` and this test fails until the
name is classified — ``AUDITED_ENV`` when it changes what runs or what is
measured (the default when unsure: recording is free, an invisible knob is
not), ``MECHANICS_ENV`` when it moves only wall clock / placement /
coordination, ``DERIVED_ENV`` when the harness itself writes it mid-run,
``SECRET_ENV`` never recorded.

Scope honesty: the scan sees LITERAL reads (``os.environ.get("X")``,
``os.environ["X"]``, ``os.getenv("X")``, and a module's own one-line env
helper ``_env_value``/``_env_float``/``_env_int``/``_env_bool``); a computed
key evades it, and the vacuity floor keeps the scan itself from rotting
silently. Shell scripts are composition (they export), not knob reads —
out of scope.

The helper names are load-bearing, not decoration: a helper takes the key as
a PARAMETER, so the literal never sits beside ``os.environ`` and the read
becomes invisible. ``maps/llm/sampling.py``'s
``_env_float("SEARCH_SAMPLING_TEMPERATURE")`` sat outside this scan — and so
outside the manifest, and so outside ``ab_report.comparability_issues`` —
for as long as it existed, which made a temperature A/B pairable against a
default-temperature run as a NULL.
"""

from __future__ import annotations

import re
from pathlib import Path

from benchmarks.citynav import run_manifest

REPO = Path(__file__).resolve().parents[3]
SCAN_PACKAGES = ("maps", "benchmarks")

_READ = re.compile(
    r'(?:os\.environ\.get|os\.environ\[|os\.getenv|environ\.get'
    r'|_env_value|_env_float|_env_int|_env_bool)'
    r'\(?\s*\(?\s*[fr]?"(SEARCH_[A-Z0-9_]+|CITYNAV_BENCH_[A-Z0-9_]+|BENCH_[A-Z0-9_]+)"'
)


def _scan() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for pkg in SCAN_PACKAGES:
        for p in (REPO / pkg).rglob("*.py"):
            if "tests" in p.parts or p.name == "conftest.py":
                continue
            for m in _READ.finditer(p.read_text(errors="ignore")):
                found.setdefault(m.group(1), []).append(
                    str(p.relative_to(REPO))
                )
    return found


def test_every_identity_env_read_is_declared() -> None:
    found = _scan()
    # Vacuity floor: the scan saw the run path (over 55 names),
    # and two load-bearing reads are pinned by name so a regex or layout
    # change cannot hollow the scan out silently.
    assert len(found) >= 55, f"scan found only {len(found)} names — broken scan?"
    for pin in (
        "CITYNAV_BENCH_CONFIG",
        "SEARCH_SKIP_VLM_PERCEPTION",
        # Read only through sampling.py's ``_env_float`` helper: the pin that
        # proves the scan still follows an indirection, which is the shape
        # that hid this knob in the first place.
        "SEARCH_SAMPLING_TEMPERATURE",
    ):
        assert pin in found, f"scan no longer sees {pin} — broken scan?"

    declared = (
        set(run_manifest.AUDITED_ENV)
        | set(run_manifest.ARM_ENV_FLAGS)
        | set(run_manifest.SECRET_ENV)
        | set(run_manifest.MECHANICS_ENV)
        | set(run_manifest.DERIVED_ENV)
    )
    prefixes = (
        run_manifest.AUDITED_ENV_PREFIXES
        + run_manifest.ARM_ENV_PREFIXES
    )
    undeclared = {
        name: sorted(set(paths))
        for name, paths in found.items()
        if name not in declared and not name.startswith(prefixes)
    }
    assert not undeclared, (
        "undeclared run-path env knob(s) — classify each in "
        "benchmarks/citynav/run_manifest.py (AUDITED_ENV if it changes what "
        "runs or what is measured — the default when unsure; MECHANICS_ENV "
        "if wall-clock/placement only; DERIVED_ENV if the harness writes it "
        "mid-run):\n  " + "\n  ".join(
            f"{k}  read at {v[0]}" for k, v in sorted(undeclared.items())
        )
    )
