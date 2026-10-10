"""Typed compile failure for infrastructure faults.

When pre-flight tour synthesis fails because of a GPU/device fault (a
CUDA OOM in the voxel pose raycaster, a device error) rather than a
model or grounding outcome, the compiler raises this so the runner
aborts loudly with an attributed cause instead of degrading to a
silent empty tour that reads like "no reasoner produced a plan".
"""

from __future__ import annotations


class CompileInfraError(RuntimeError):
    """Pre-flight compile hit an infrastructure fault, not a model miss.

    ``cause_tag`` is a short machine-readable reason (e.g.
    ``pose_resolution_oom``). ``device`` / ``mem_snapshot`` carry the
    GPU attribution so the failure is self-describing in the log.
    """

    def __init__(
        self,
        message: str,
        *,
        cause_tag: str,
        device: str | None = None,
        mem_snapshot: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.cause_tag = cause_tag
        self.device = device
        self.mem_snapshot = mem_snapshot or {}
