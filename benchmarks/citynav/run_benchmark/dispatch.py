"""One-episode dispatch: compiler construction, run provenance, and
the shared batch-loop/server episode wrapper.
"""

from __future__ import annotations

from benchmarks.citynav.adapter import parse_episode
from benchmarks.citynav.live_recorder import LiveRecorder
from benchmarks.citynav.run_benchmark.config import RunConfig
from benchmarks.citynav.run_benchmark.error_taxonomy import (
    _exc_is_cuda_oom,
    _exc_is_sam_infra,
    _exc_is_transient_infra,
)
from benchmarks.citynav.episodes import STATUS_INFRA, STATUS_OK, write_episode_json
from pathlib import Path
from maps.llm import endpoint_spec as endpoint_spec_module
from maps.perception.geometry.sensor_spec import resolve_sensor
from maps.search import MapsCompiler
import os
import time
from benchmarks.citynav.run_benchmark.episode_runner import run_episode
import logging

log = logging.getLogger("run_benchmark")


def _build_compiler(cfg: RunConfig) -> MapsCompiler:
    """Construct the MapsCompiler (sensor-patched system prompts).

    Extracted verbatim from the inline ctor so the persistent
    episode-server path builds an identical compiler. The active sensor
    description is patched into the parse-compile reasoner + scout system
    prompts so ``sam_concepts`` stays constrained to surfaces the rig
    can observe. The served model / endpoint facts come from the installed
    ``EndpointSpec`` (the compiler reads ``spec()`` itself) — there is no
    client-side model knob.
    """
    _sensor = resolve_sensor(cfg.sensor) if cfg.sensor else None
    _sensor_description = _sensor.description if _sensor is not None else ""
    return MapsCompiler(sensor_description=_sensor_description)


# The arm-flag inventory and stamp live with the rest of the run-identity
# env vocabulary in benchmarks.citynav.run_manifest (one home for "which
# env vars are run identity"); this module stamps rows with it.
from benchmarks.citynav.run_manifest import arm_env_stamp

_RUN_PROVENANCE: dict | None = None
_RUN_CONTEXT: dict = {}


def set_run_context(*, config_path: str | None, limit: int | None) -> None:
    """Run facts only the CLI knows (its parsed argv): the config the runner
    actually loaded and the row cap. Recorded once per process so row
    provenance reads the resolved run, never ambient env (an env-carried
    config path is null whenever the launching shell forgets the export)."""
    _RUN_CONTEXT["config_path"] = config_path
    _RUN_CONTEXT["limit"] = limit


def _cached_run_provenance(search_flags: dict) -> dict:
    """Process-cached provenance (git commit + resolved package paths + config
    sha256 + DISCOVERED served model + flag set) embedded into every episode
    row so "which code ran" is auditable from disk alone. The git subprocess
    runs ONCE per worker process, not once per episode."""
    global _RUN_PROVENANCE
    if _RUN_PROVENANCE is None:
        from benchmarks import repo_provenance

        _spec = (
            endpoint_spec_module.spec()
            if endpoint_spec_module.spec_installed()
            else None
        )
        model = _spec.model_id if _spec else None
        build = _spec.tokenizer_name if _spec else None
        _seed = os.environ.get("CITYNAV_BENCH_SAMPLE_SEED")
        _RUN_PROVENANCE = repo_provenance(
            config_path=_RUN_CONTEXT.get("config_path"),
            split=os.environ.get("CITYNAV_BENCH_SPLIT") or None,
            seed=int(_seed) if (_seed or "").lstrip("-").isdigit() else None,
            limit=_RUN_CONTEXT.get("limit"),
            search_flags=search_flags,
            served_model=model,
            served_build=build,
        )
    return _RUN_PROVENANCE


def _dispatch_one(
    *,
    row: dict,
    orig_idx: int,
    idx: int,
    total: int,
    compiler: MapsCompiler | None,
    objects_by_map: dict,
    cfg: RunConfig,
    recorder: LiveRecorder | None,
    episode_dir: Path,
    per_episode: list[dict],
) -> tuple[dict, bool]:
    """Run one episode and write its row to ``<episode_dir>/episode.json``,
    shared by the batch loop and the persistent episode server.

    Returns ``(row, failed_oom)``. The row's ``status`` is ``ok`` for a
    scored attempt and ``infra`` when the algorithm never got a fair run:
    the episode raised, the mission ended ``transient_failures``, or the
    runner produced no record. The batch loop appends the row to the run's
    ledger; the server uses ``failed_oom`` to tear itself down so its next
    episode gets a fresh CUDA context.
    """
    failed_oom = False
    t0 = time.monotonic()
    try:
        _eid = parse_episode(row, row_index=orig_idx).episode_id
    except Exception:
        _eid = f"row{orig_idx}"

    # Stamp the experiment-arm identity onto every row. The SEARCH_* arm
    # flags live only in the launching shell; without this, rows from
    # different A/B arms are indistinguishable post hoc (a resume under a
    # shell that forgot a flag silently flies the other arm with no on-disk
    # trace).
    _arm_stamp = arm_env_stamp()
    _prov = _cached_run_provenance(_arm_stamp)

    def _stamp(rec: dict) -> dict:
        return {
            **rec,
            "search_env_flags": rec.get("search_env_flags") or _arm_stamp,
            "provenance": rec.get("provenance") or _prov,
            "withhold": list(cfg.withhold),
            "elapsed_s": round(time.monotonic() - t0, 1),
            "ts": round(time.time(), 1),
        }

    def _provisional_sink(rec_provisional: dict) -> None:
        # The pre-flight floor, on disk before takeoff so a SIGKILL between
        # here and the commit still leaves a scored row.
        write_episode_json(episode_dir, _stamp(rec_provisional))

    try:
        rec = run_episode(
            row,
            compiler,
            objects_by_map,
            cfg,
            recorder=recorder,
            row_index=orig_idx,
            provisional_sink=_provisional_sink,
        )
    except Exception as e:
        # exc_info lands the traceback in run.log through the root
        # handler; a bare print_exc reaches only stderr.
        log.error("  fail row %d: %s", idx, e, exc_info=True)
        # The greppable tokens the fleet driver's log classifier reads.
        # Each names an infra death before commit -- a CUDA/CuPy contention
        # OOM, a SAM outage, a transient shared-server fault -- and each
        # makes the warm runner respawn so the re-attempt starts clean.
        if _exc_is_cuda_oom(e):
            token = "CITYNAV_EPISODE_OOM_ABORT"
            log.error("%s episode_id=%s", token, _eid)
        elif _exc_is_sam_infra(e) or _exc_is_transient_infra(e):
            token = "CITYNAV_EPISODE_INFRA_ABORT"
            log.error("%s episode_id=%s type=%s", token, _eid, type(e).__qualname__)
        else:
            token = None
        failed_oom = token is not None
        err = f"{type(e).__name__}: {e}"
        out = _stamp({
            "episode_id": _eid,
            "status": STATUS_INFRA,
            "error": f"{token}: {err}" if token else err,
        })
        write_episode_json(episode_dir, out)
        return out, failed_oom
    if rec is None:
        # The runner exited clean with no record (no objects for the map,
        # the reasoner unreachable pre-flight, a --stop-after exit): not a
        # scored attempt, so it is excluded and re-queued.
        out = _stamp({
            "episode_id": _eid,
            "status": STATUS_INFRA,
            "error": "no_prediction_unclassified",
        })
        write_episode_json(episode_dir, out)
        return out, failed_oom
    per_episode.append(rec["metrics"])
    m = rec["metrics"]
    # A mission the serve failed out from under is the cluster's weather,
    # not the model's skill: excluded from SR and re-queued.
    serve_failed = (m.get("mission") or {}).get("ended_by") == "transient_failures"
    out = _stamp({
        **rec,
        "status": STATUS_INFRA if serve_failed else STATUS_OK,
        "error": "mission ended_by=transient_failures" if serve_failed else None,
    })
    write_episode_json(episode_dir, out)
    log.info(
        # ``%s`` on the verdict, not ``%d``: this line sits OUTSIDE the
        # per-episode try above, so a null SR would take down the batch loop
        # and the episode server rather than print oddly.
        "[%d/%d] %s NE=%.1f SR[%s]=%s steps=%d commit=%s(%s) (%.1fs)",
        idx + 1,
        total,
        rec["episode_id"],
        m["NE"],
        m.get("success_criterion") or "?",
        m["SR"],
        m["steps_taken"],
        # The perceive-vs-prior signal: ``detected`` = committed a runtime
        # SAM detection (e.g. a car, no longer in the prior), ``prior`` =
        # committed a prebaked landmark.
        m.get("committed_provenance") or "none",
        m.get("committed_fid") or "-",
        m["compile_s"],
    )
    return out, failed_oom
