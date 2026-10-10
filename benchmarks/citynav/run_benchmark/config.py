"""``RunConfig`` — the master knob object for one CityNav run — with its YAML
loader, env exports, required-field validation, and the ground-truth-isolation
guard. A configuration leaf: depends only on stdlib + ``maps.perception`` +
``flight_model``; nothing here reaches back into the episode engine.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from maps.perception.sam.detect_client import SamDetectClient

from benchmarks.citynav.flight_model import FlightModelConfig

log = logging.getLogger("run_benchmark")


class _Required:
    """Sentinel for a sim-setup knob that the named YAML must supply.

    Fields defaulted to ``_REQUIRED`` carry no usable value out of the
    dataclass; ``load_config`` raises and names every still-sentinel key
    plus the config-file path. The YAML is the single source of truth for
    sim setup, so a missing key fails loudly instead of falling back to a
    silent dataclass default.
    """

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return "<REQUIRED-from-YAML>"


_REQUIRED = _Required()


@dataclass
class RunConfig:
    # "sam" (production real-SAM) | "gt_perfect" (diagnostic perfect
    # sensor). YAML-authoritative (see _REQUIRED).
    perception: str = _REQUIRED  # type: ignore[assignment]
    # Step budget per episode. Under the strict CityNav action space
    # (5m forward / 30deg turn / 2m altitude) an episode needs hundreds
    # of discrete steps to walk to several tour stops and descend; 200
    # matches the GeoNav (arXiv 2504.09587) / HETT / CityNavAgent
    # community standard. YAML wins; this dataclass default is the
    # fallback only when no config supplies the key.
    max_steps: int = 200
    fov_m: float = _REQUIRED  # type: ignore[assignment]  # YAML-authoritative
    altitude_m: float = 50.0
    # GT-PERFECT diagnostic perception (perfect sensor). Knobs:
    # ``color_table_path`` (JSON {map:{object_id:{color,color_conf}}}, the offline
    # cue-independent GT color table), ``mask_px``, ``min_blob_px``.
    gt_perfect: dict = field(default_factory=dict)
    sam: dict = field(default_factory=dict)
    # Sensor parameterization. String = registered sensor or suite (see
    # ``perception.sensor_spec.list_sensors``); dict =
    # inline override. YAML-authoritative (see _REQUIRED): the named config picks
    # the sensor. CityNav runs use ``nadir_1008_90``, a single 1008x1008
    # ``SensorSpec`` (one nadir frame per step).
    sensor: str | dict = _REQUIRED  # type: ignore[assignment]
    # Legacy knobs; prefer ``sensor``. When ``render.pitch_deg`` is set it
    # overrides the sensor spec's mount pitch (useful for comparing
    # nadir vs forward-looking runs without editing the registry).
    render: dict = field(default_factory=dict)
    cell_size_m: float = 2.0
    success_radius_m: float = 20.0
    # How close to an annotated target SHAPE counts as arriving, for
    # a shape-scored SR (eval.score_precise). 0.0 is strict containment: the agent
    # must finish inside the footprint. Raise it to allow a margin around the
    # edge — a car-sized target is only a few metres across, so the right
    # value depends on how big the annotated shapes actually are.
    precise_tolerance_m: float = 0.0
    # The tightest target the shape criterion will ask for. A shape narrower than this
    # is widened until a circle of this radius fits inside it, so no episode
    # is scored on a precision finer than the agent can resolve. Measured on
    # the Cal City corpus: on episodes the agent already gets right it lands
    # 2.9 m from the target at the median and within 5 m 69% of the time,
    # while a road junction is 4 m across. 5 m is the knee — it reaches every
    # junction and still leaves 83% of buildings scored as drawn, where 6 m
    # would take half of them. Set 0.0 to score every shape exactly as drawn.
    precise_floor_radius_m: float = 5.0
    # Where an episode starts: "episode" flies the dataset's recorded start
    # pose (CityNav's 50-350 m regime); "uniform" replaces it with a seeded
    # uniform point over the tile bounds (run_benchmark.spawn), deterministic
    # per (spawn_seed, episode_id) — one seed reproduces a run exactly, two
    # seeds give independent spawn sets. The flown start replaces the
    # episode's everywhere downstream (records, metrics, floors).
    spawn: str = "episode"
    spawn_seed: int = 0
    # Upstream start-altitude band [lo, hi] (paper SS3.2: random 100-150 m).
    # Sampled per episode deterministically (seeded by episode_id) on the ortho
    # backend.
    start_altitude_band_m: tuple[float, float] = (100.0, 150.0)
    # AGL override (meters above ground) for the ortho backend. When
    # set (non-None), the per-episode start altitude becomes GROUND_LEVEL[map]
    # + this value INSTEAD of sampling start_altitude_band_m — a UNIFORM
    # above-ground cruise across maps of differing elevation (Birmingham ~10 m,
    # Cambridge ~38 m AMSL), so the altitude-driven view square 2*(z - ground)
    # is identical everywhere. None preserves the strict AMSL-band behavior
    # byte-for-byte.
    start_altitude_agl_m: float | None = None
    # Hard physical flight ceiling (paper caps at 200 m). The ortho
    # altitude executor clamps GO_UP / per-target altitude into
    # [GROUND_LEVEL[map] + eps, altitude_cap_m].
    altitude_cap_m: float = 200.0
    # Model-input resolution for the ortho crop. The native-resolution
    # crop (real ground detail, warped directly from the raster) is resampled to
    # this square size — UP or DOWN, since the native pixel span varies with
    # altitude — and the reasoner, SAM, and dashboard all consume it. ``None`` =
    # no resample (raw native span). SAM no longer reads a separate crop; this is
    # the single model-input frame.
    rgb_upscale_px: int | None = None
    # When set, the step loop samples flight at camera_hz between tour
    # stops instead of teleporting one waypoint per tick. See
    # benchmarks/citynav/flight_model.py.
    flight_model: FlightModelConfig | None = None
    # Universal resume from a prior run's workdir. When set,
    # ``SearchLoop.restore_from_workdir`` rebuilds belief, target_spec,
    # last_intel_text, SAM video concepts, and compile_seq from the
    # source's on-disk artifacts before the step loop starts. Pre-flight
    # compile is then suppressed (``should_compile`` returns False for an
    # unchanged instruction) and the runtime resumes as if the prior run
    # had reached ``resume_turn`` already. ``hydrate_resume_artifacts``
    # mirrors the dashboard tree for the SPA in the same call site.
    resume_from_workdir: str | None = None
    # Optional explicit turn index. ``None`` (default) picks the latest
    # ``compile/turn_<NNNN>/`` whose finalized reasoner carries
    # reconstructable sam_concepts.
    resume_turn: int | None = None
    # Per-turn run-checkpoint resume target for the reasoning engine. When
    # set (one of ``segment``, ``ingest``, ``stage3``, ``paint``,
    # ``grade``), each compile call reads prior phase outputs from
    # ``<workdir>/compile/turn_<step>/`` and skips every earlier phase.
    # Wired from ``--resume-from`` in ``main()``; default ``None`` runs
    # the full pipeline (``SearchLoop.restore_from_workdir``).
    # Commit lever (ablation): prefer a runtime SAM detection over a prebake
    # prior at commit. "off" | "prefer" | "only". Forwarded into
    # ``SEARCH_DETECTED_TARGET_MODE`` so SearchLoopConfig.from_env picks it up.
    detected_target_mode: str = "off"
    # Descent planner: meters of altitude to drop per planning tick
    # while ``confirmed_target_xy`` is set.
    descent_step_m: float = 5.0
    # Target altitude for landing. Episode terminates when
    # ``state.altitude_m <= landing_altitude_m + altitude_tolerance_m``
    # AND the drone is within ``landing_xy_tolerance_m`` of the
    # confirmed target XY.
    landing_altitude_m: float = 5.0
    altitude_tolerance_m: float = 1.0
    landing_xy_tolerance_m: float = 5.0
    # CityNav SR is 2D (NE over x,y only), so descending the committed
    # centroid to landing_altitude never changes pred_xy's xy -- it only
    # burns step budget (~22 steps at 50 m, ~60 at 125 m). Default False:
    # traverse to the committed centroid AT cruise altitude, leaving the
    # whole budget for navigation. Set True for a benchmark variant that
    # needs the drone to actually descend.
    descend_to_land: bool = False
    # Autocompact (LLM-summary context compaction). Per-stage budgets are
    # input-token caps; calls under budget pass through identity. ``enable``
    # toggles the whole subsystem. Values get exported as ``SEARCH_COMPACT_*``
    # env vars at startup so every LLM call site sees them; CLI / shell
    # overrides win because env vars are not overwritten when already set.
    compact: dict = field(default_factory=dict)
    # The resolved YAML path this config was loaded from. Written by
    # ``load_config`` (never by YAML): row provenance hashes THIS path, so
    # config_sha256 pins the file that actually resolved, not the argv
    # spelling (which may only exist relative to the repo root).
    source_path: str | None = None


# The compaction budget is the server-discovered ``EndpointSpec.input_budget``
# (maps.llm.endpoint_spec), never a YAML/env window pin — only the live
# knobs (enable / tokenizer) map to a read env var.
_COMPACT_YAML_TO_ENV: dict[str, str] = {
    "enable": "SEARCH_COMPACT_ENABLE",
    "tokenizer_name": "SEARCH_COMPACT_TOKENIZER_NAME",
}


def _export_compact_env(compact: dict) -> None:
    """Translate the ``compact:`` yaml block into ``SEARCH_COMPACT_*`` env vars.

    Existing env values win (so a shell-level ``SEARCH_COMPACT_ENABLE``
    override is honored). The compactor reads these env vars at call time.
    """
    if not compact:
        return
    for key, env_name in _COMPACT_YAML_TO_ENV.items():
        if key not in compact:
            continue
        if os.environ.get(env_name) is not None:
            # Shell / CLI override wins.
            continue
        value = compact[key]
        if isinstance(value, bool):
            os.environ[env_name] = "1" if value else "0"
        else:
            os.environ[env_name] = str(value)


# Sim-setup knobs the named YAML must supply. Each defaults to ``_REQUIRED``
# on ``RunConfig``; the YAML is the single source of truth for sim setup.
_REQUIRED_SETUP_FIELDS: tuple[str, ...] = (
    "perception",
    "fov_m",
    "sensor",
)


def _validate_required(cfg: RunConfig, src: str) -> None:
    """Fail loudly if the named config omits any required sim-setup knob.

    Names every missing key and the config path so the operator knows
    exactly what to add and where. No silent dataclass fallback.
    """
    missing: list[str] = [
        name for name in _REQUIRED_SETUP_FIELDS if getattr(cfg, name) is _REQUIRED
    ]
    if missing:
        raise RuntimeError(
            f"{src}: missing required sim-setup keys: {sorted(missing)}. "
            f"Sim setup is YAML-authoritative; add the listed key(s) to "
            f"{src} (no dataclass fallback)."
        )


def load_config(path: str | None) -> RunConfig:
    if not path:
        raise RuntimeError(
            "a --config YAML is required; sim setup is YAML-authoritative. "
            "Pass --config configs/citynav.yaml."
        )
    candidates = [Path(path)]
    if not candidates[0].is_absolute():
        # Resolve against cwd first, then against the repo root, so both an
        # absolute path and a repo-root-relative
        # ``configs/<name>.yaml`` work.
        candidates.append(Path(__file__).resolve().parents[2] / path)
    for p in candidates:
        if p.exists():
            import yaml  # lazy — only required for config runs

            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            fm_block = data.pop("flight_model", None)
            if fm_block is not None:
                data["flight_model"] = FlightModelConfig(**fm_block)
            cfg = RunConfig(**data)
            cfg.source_path = str(p)
            _validate_required(cfg, str(p))
            _assert_realstack_gt_isolation(cfg, p)
            return cfg
    raise FileNotFoundError(f"Config not found: tried {[str(p) for p in candidates]}")


# The real-stack pivot GT-isolation runtime gate. TRIGGER CHOICE (revised
# 2026-07-04, blocker V1-R3/V4-G2/V5-F1): the check fires on the resolved
# config *CONTENT* — the perception tier — NOT on the config FILENAME. The
# earlier filename trigger (``citynav.yaml``) was scoped too
# narrowly: any DERIVED config (an A/B arm, a resume clone, a ``_lookoff``
# variant) inherits the real-SAM ``sam`` substrate but a filename
# gate skips it, so a carried-over ``color_table_path`` or a stray
# ``SEARCH_SKIP_VLM_PERCEPTION=1`` would fly GT-leaky with zero signal. Any
# config that resolves to the real-SAM tier now carries the guard. The
# gt_perfect DIAGNOSTIC tier (``perception: gt_perfect``) never trips it (it
# is not the real-SAM tier). An explicit, documented diagnostic opt-out
# (``SEARCH_ALLOW_GT_DIAGNOSTIC=1``) is the ONLY escape — for a deliberate
# GT-leaky probe under a sam config, a reviewable act.
_REALSTACK_PERCEPTION_TIER = "sam"
# Documented diagnostic opt-out for the C1 gate. Set to "1" ONLY for a
# deliberate GT-diagnostic probe under the real-SAM tier; NEVER on a fleet
# run. Off by default; its presence is logged loudly when it suppresses the
# guard.
_GT_DIAGNOSTIC_OPT_OUT_ENV = "SEARCH_ALLOW_GT_DIAGNOSTIC"


def _assert_realstack_gt_isolation(
    cfg: "RunConfig",
    config_path: Path | None = None,
    *,
    perception: object | None = None,
) -> None:
    """C1 GT-isolation runtime gate (SPEC_real_stack_pivot_2026-07-04.md §1 C1).

    Fires by CONTENT — whenever the resolved perception tier is the real-SAM
    ``sam`` tier — so every DERIVED config inherits the guard, not
    just the canonically-named one. Under that tier it asserts ALL of:

      (a) no GT color table is wired (``color_table_path`` unset on the
          ``sam`` AND ``gt_perfect`` config blocks) — color comes
          from pixels only;
      (b) the perception tier IS ``sam`` (defense-in-depth);
      (c) ``SEARCH_SKIP_VLM_PERCEPTION`` is NOT set — the GT-era toggle that
          once swapped in synthetic GT perception; it is refused outright so
          a stale environment can never masquerade as a real-SAM run;
      (d) if a resolved ``perception`` OBJECT is passed (post-construction),
          it IS the real SAM adapter, never a synthetic/GT-perfect adapter.

    Fails LOUD (``RuntimeError`` citing C1) at config load — before any
    compile/flight spend — and again post-construction with the object.
    ``SEARCH_ALLOW_GT_DIAGNOSTIC=1`` is the ONLY (documented, logged) escape.
    The gt_perfect diagnostic tier never trips it. The static half of the
    gate is ``distill/tests/test_c1_gt_isolation.py``.
    """
    # Tier-(c) oracle GT-injection derives a
    # viewpoint from the withheld target_xy — deliberately GT-leaky, so it
    # demands the same explicit ack as every other C1 diagnostic. Checked
    # before the tier early-return: tier-independent by design.
    if (
        os.environ.get("SEARCH_ORACLE_INJECT_GT", "").strip() == "1"
        and os.environ.get(_GT_DIAGNOSTIC_OPT_OUT_ENV) != "1"
    ):
        raise RuntimeError(
            "SEARCH_ORACLE_INJECT_GT=1 without SEARCH_ALLOW_GT_DIAGNOSTIC=1 "
            "— acknowledge the GT-injection diagnostic explicitly "
            "(tier (c))."
        )
    if cfg.perception != _REALSTACK_PERCEPTION_TIER:
        return  # not the real-SAM tier → guard does not apply (e.g. gt_perfect)
    if os.environ.get(_GT_DIAGNOSTIC_OPT_OUT_ENV) == "1":
        log.warning(
            "C1 GT-isolation gate SUPPRESSED by %s=1 — GT-diagnostic probe "
            "under the sam tier. This MUST NOT be set on a fleet run.",
            _GT_DIAGNOSTIC_OPT_OUT_ENV,
        )
        return
    where = f"{config_path}: " if config_path is not None else ""
    problems: list[str] = []
    # (b) perception tier (defense-in-depth; trivially true at the trigger, but
    # asserted so a future refactor of the trigger can't silently drop it).
    if cfg.perception != _REALSTACK_PERCEPTION_TIER:
        problems.append(
            f"perception tier is {cfg.perception!r}, must be "
            f"{_REALSTACK_PERCEPTION_TIER!r}"
        )
    # (a) no GT color table wired on either the real-SAM or gt_perfect block.
    for block_name in ("gt_perfect", "sam"):
        block = getattr(cfg, block_name, None) or {}
        if isinstance(block, dict) and block.get("color_table_path"):
            problems.append(
                f"{block_name}.color_table_path is set "
                f"({block['color_table_path']!r}), must be unset"
            )
    # (c) SEARCH_SKIP_VLM_PERCEPTION is the deleted GT-era toggle; refuse a
    # stale environment outright rather than let it imply a GT run.
    if os.environ.get("SEARCH_SKIP_VLM_PERCEPTION") == "1":
        problems.append(
            "SEARCH_SKIP_VLM_PERCEPTION=1 is set — this GT-era toggle no "
            "longer exists; unset the env (use the gt_perfect DIAGNOSTIC "
            "config for a GT run)"
        )
    # (d) if the resolved perception object is reachable, it must be the real
    # SAM adapter — not a synthetic / GT-perfect substitute.
    if perception is not None and not isinstance(
        perception, SamDetectClient
    ):
        problems.append(
            "resolved perception object is "
            f"{type(perception).__name__}, must be SamDetectClient "
            "(real SAM) under the sam tier — a synthetic/GT-perfect "
            "object here is a GT leak"
        )
    if problems:
        raise RuntimeError(
            f"{where}C1 GT-ISOLATION VIOLATION under the real-stack "
            f"sam tier — {'; '.join(problems)}. The pivot contract "
            "(distill/SPEC_real_stack_pivot_2026-07-04.md, contract C1) "
            "requires real SAM perception with pixels-only color and zero "
            "CityRefer-derived attributes on the agent path. Set "
            f"{_GT_DIAGNOSTIC_OPT_OUT_ENV}=1 ONLY for a deliberate diagnostic "
            "probe (never on a fleet run)."
        )
    # GT isolation is confirmed IN FORCE for this run — publish it so the
    # downstream bake-integrity guards (e.g. _load_block_assets' missing-
    # _features.json fail-loud, LEAK L1 re-verifier caveat) activate without
    # each re-deriving the tier. Idempotent; cleared by nothing (process-lived).
    os.environ["SEARCH_REALSTACK_GT_ISOLATION"] = "1"
