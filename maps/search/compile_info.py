"""Per-turn compile-info shape the engine records.

Splits the engine's turn output into two views:

* ``info`` — the dashboard-facing summary the recorder writes into
  ``step.compile``: parsed-instruction string, timing. Stable shape,
  JSON-safe values.
* ``raw`` — the full ``MapsCompiler`` dict (alive_rids, broken_rids,
  reasoners, sam_concepts_union, ...). Consumed by the per-turn receipt
  writer and the per-turn log summary.

The ``extract_*`` helpers below pull the human-readable bits out of
``raw`` so callers don't have to navigate the result schema themselves.
They are pure functions of the compiler result; no engine state needed.
"""

from __future__ import annotations

from dataclasses import dataclass


def extract_program_summary(result: dict) -> str:
    """Compact summary built from the compile result (source-of-truth)."""
    concepts = result.get("sam_concepts_union") or []
    alive = result.get("alive_rids") or []
    broken = result.get("broken_rids") or []
    return (
        f"concepts={', '.join(str(c) for c in concepts[:6]) or '?'}"
        f" | alive={len(alive)} broken={len(broken)}"
    )


@dataclass(frozen=True)
class CompileTurnResult:
    """Engine output for one turn.

    ``info`` is the dashboard-facing dict (stable keys, JSON-safe).
    ``raw`` is the full compiler result; the harness passes it into the
    per-turn receipt writer and the summary logger.
    """

    info: dict
    raw: dict


def build_compile_info(
    *,
    result: dict,
    instruction: str,
    wall_s: float,
) -> dict:
    """Assemble the ``info`` dict from a compiler ``result``.

    Centralised so the engine and any out-of-loop tooling agree on the
    dashboard-facing shape.
    """
    return {
        "wall_s": round(float(wall_s), 3),
        "parsed_instruction": extract_program_summary(result),
        "instruction": instruction,
        "alive_rids": list(result.get("alive_rids") or []),
        "broken_rids": list(result.get("broken_rids") or []),
        "timing_ms": result.get("timing_ms") or {},
    }
