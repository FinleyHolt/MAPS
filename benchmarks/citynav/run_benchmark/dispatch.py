"""One-episode dispatch: run provenance, the infra exception classifiers
and the episode wrapper."""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

from benchmarks import repo_provenance
from benchmarks.citynav.adapter import CityNavEpisode
from benchmarks.citynav.analysis import rows as _rows
from benchmarks.citynav.episodes import STATUS_INFRA, STATUS_OK, write_episode_json
from benchmarks.citynav.live_recorder import LiveRecorder
from benchmarks.citynav.run_benchmark.config import RunConfig
from benchmarks.citynav.run_benchmark.episode_runner import run_episode
from benchmarks.citynav.run_manifest import arm_stamp
from maps.llm import endpoint_spec as endpoint_spec_module
from maps.llm.http import TRANSIENT_CLIENT_STATUS, ImageStoreError
from maps.perception.sam.detect_client import SamServerOutage

log = logging.getLogger("run_benchmark")

_RUN_PROVENANCE: dict | None = None


def _cached_run_provenance(search_flags: dict, *, config_path: str | None, split: str) -> dict:
    """The provenance every episode row carries (git commit, config sha256,
    the discovered served model, the flag set), computed once per process."""
    global _RUN_PROVENANCE
    if _RUN_PROVENANCE is None:
        spec = endpoint_spec_module.spec()
        _RUN_PROVENANCE = repo_provenance(
            config_path=config_path,
            split=split,
            search_flags=search_flags,
            served_model=spec.model_id,
            served_build=spec.tokenizer_name,
        )
    return _RUN_PROVENANCE


# Exception classifiers: an infrastructure casualty (excluded from the SR
# denominator and re-queued) or a genuine algorithm failure (a scored miss),
# walking the full ``__cause__`` / ``__context__`` chain.


def _chain(exc: BaseException):
    """``exc`` and every exception below it, each once."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def _exc_is_sam_infra(exc: BaseException) -> bool:
    """True if any link in the chain is a ``SamServerOutage`` (an empty
    frame never raises)."""
    return any(isinstance(cur, SamServerOutage) for cur in _chain(exc))


# HTTP-client modules that ONLY talk to the shared inference services (the
# reasoner and SAM). An exception whose type lives in one of these is a transport fault to
# a shared server. Constrained to these modules on purpose: a bare builtin
# ``TimeoutError`` / ``ConnectionError`` / ``OSError`` can be raised by
# algorithm or file code, and misclassifying one as infra would drop a real
# CityNav failure from SR.
_HTTP_CLIENT_MODULES = ("httpx", "httpcore", "requests", "urllib3")


def _exc_is_transient_infra(exc: BaseException) -> bool:
    """True iff a link in the chain is a transport fault to a shared
    inference server: its type is from an HTTP-client module, and it carries
    no HTTP status below 500 outside ``TRANSIENT_CLIENT_STATUS``."""
    for cur in _chain(exc):
        if (type(cur).__module__ or "").split(".", 1)[0] not in _HTTP_CLIENT_MODULES:
            continue
        status = getattr(getattr(cur, "response", None), "status_code", None)
        return not (
            isinstance(status, int)
            and status < 500
            and status not in TRANSIENT_CLIENT_STATUS
        )
    return False


def is_infra_fault(exc: BaseException) -> bool:
    """The infra rule for a raised attempt: a SAM outage, a transient
    shared-server fault, an image-store refusal or a reasoner that did not
    answer discovery. Any other exception is the method's own, a scored miss."""
    return (_exc_is_sam_infra(exc) or _exc_is_transient_infra(exc)
            or any(isinstance(cur, (ImageStoreError, endpoint_spec_module.EndpointUnreachable))
                   for cur in _chain(exc)))


def _dispatch_one(
    *,
    episode: CityNavEpisode,
    idx: int,
    total: int,
    objects_by_map: dict,
    cfg: RunConfig,
    split: str,
    recorder: LiveRecorder | None,
    episode_dir: Path,
) -> dict:
    """Run one episode, write its row to ``<episode_dir>/episode.json`` and
    return it. ``status`` is ``ok`` for a scored attempt (a method exception
    included) and ``infra`` when an infrastructure fault raised, the mission
    ended ``transient_failures``, or the runner produced no record.
    """
    t0 = time.monotonic()

    # Every row's provenance carries the arm identity, as the manifest stamps it.
    _prov = _cached_run_provenance(arm_stamp(cfg.policy), config_path=cfg.source_path, split=split)

    def _stamp(rec: dict) -> dict:
        return {**rec, "provenance": _prov, "elapsed_s": round(time.monotonic() - t0, 1)}

    try:
        rec = run_episode(episode, objects_by_map, cfg, recorder=recorder)
    except Exception as e:
        # exc_info lands the traceback in run.log through the root
        # handler; a bare print_exc reaches only stderr.
        log.error("  fail row %d: %s", idx, e, exc_info=True)
        # An infra death is excluded and re-queued; anything else the method
        # raised is its own failure: a scored miss (no metrics reads SR 0)
        # carrying its traceback.
        infra = is_infra_fault(e)
        out = _stamp({
            "episode_id": episode.episode_id,
            "status": STATUS_INFRA if infra else STATUS_OK,
            "error": f"{type(e).__name__}: {e}",
            **({} if infra else {"traceback": traceback.format_exc()}),
        })
        write_episode_json(episode_dir, out)
        return out
    if rec is None:
        # The runner skipped the episode (its map has no CityRefer objects):
        # not a scored attempt, so it is excluded and re-queued.
        out = _stamp({
            "episode_id": episode.episode_id,
            "status": STATUS_INFRA,
            "error": "no_prediction_unclassified",
        })
        write_episode_json(episode_dir, out)
        return out
    m = rec["metrics"]
    # A mission the serve failed out from under is the cluster's weather,
    # not the model's skill: excluded from SR and re-queued.
    serve_failed = _rows.is_infra_abort(rec)
    out = _stamp({
        **rec,
        "status": STATUS_INFRA if serve_failed else STATUS_OK,
        "error": "mission ended_by=transient_failures" if serve_failed else None,
    })
    write_episode_json(episode_dir, out)
    log.info(
        # ``%s`` on the verdict: a null SR must not raise outside the try.
        "[%d/%d] %s NE=%.1f SR[%s]=%s steps=%d commit=%s (%.1fs)",
        idx + 1,
        total,
        rec["episode_id"],
        m["NE"],
        m.get("success_criterion") or "?",
        m["SR"],
        m["steps_taken"],
        # ``detected`` (a runtime detection), ``prior`` (a baked landmark) or
        # ``point`` (a world point).
        m.get("committed_provenance") or "none",
        out["elapsed_s"],
    )
    return out
