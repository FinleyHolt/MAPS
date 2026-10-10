"""Candidate expansion helpers shared by the compiler and the runtime rounds.

The round-0 prior surface is only the initial candidate pool. Runtime
evidence can surface nearby or overlapping objects that were not in
that pool, so the per-stop rounds and the deterministic end-of-tour commit
need one deterministic way to admit those
objects without teaching the LLM episode-specific shortcuts.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


def _clean_fid(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _node_label(node: Any) -> tuple[str, str, str]:
    cls = str(getattr(node, "class_name", "") or "").strip()
    name = str(getattr(node, "name", "") or "").strip()
    provenance = str(getattr(node, "provenance", "") or "").strip()
    return cls, name, provenance


def _normalize_category_label(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[_/-]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    tokens: list[str] = []
    for token in text.split():
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def labels_match(label: object, target: object) -> bool:
    label_norm = _normalize_category_label(label)
    target_norm = _normalize_category_label(target)
    if not label_norm or not target_norm:
        return False
    if label_norm == target_norm:
        return True
    label_tokens = set(label_norm.split())
    target_tokens = set(target_norm.split())
    if not label_tokens or not target_tokens:
        return False
    return target_tokens <= label_tokens


def _iter_node_match_labels(node: Any) -> list[str]:
    labels: list[str] = []
    for attr in (
        "class_name",
        "category",
        "class_label",
        "label",
        "object_class",
        "name",
    ):
        value = getattr(node, attr, None)
        if isinstance(value, str) and value.strip():
            labels.append(value.strip())
    metadata = getattr(node, "metadata", None)
    if isinstance(metadata, dict):
        for key in (
            "class_name",
            "category",
            "class_label",
            "label",
            "object_class",
            "name",
        ):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                labels.append(value.strip())
    semantic = getattr(node, "semantic_attributes", None)
    if isinstance(semantic, dict):
        for axis in ("sam_label", "category", "class", "object_class"):
            values = semantic.get(axis)
            if isinstance(values, dict):
                labels.extend(str(k).strip() for k, v in values.items() if v)
            elif isinstance(values, str) and values.strip():
                labels.append(values.strip())
    return labels


def _evidence_attributes(ev: Any) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    attrs = getattr(ev, "attributes", None)
    if not isinstance(attrs, dict):
        return out
    for axis, values in attrs.items():
        if not isinstance(values, dict):
            continue
        axis_text = str(axis)
        for raw_value, raw_count in values.items():
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if count <= 0:
                continue
            out.setdefault(axis_text, {})[str(raw_value)] = count
    return out


def _evidence_match_labels(ev: Any) -> list[str]:
    labels: list[str] = []
    attrs = _evidence_attributes(ev)
    for axis in ("sam_label", "category", "class", "object_class"):
        labels.extend(attrs.get(axis, {}).keys())
    return labels


def _matches_target_class(
    node: Any,
    target_classes: list[str] | tuple[str, ...] | set[str],
    *,
    ev: Any = None,
) -> bool:
    targets = [
        _normalize_category_label(t)
        for t in target_classes
        if _normalize_category_label(t)
    ]
    if not targets:
        return False
    labels = _iter_node_match_labels(node)
    if ev is not None:
        labels.extend(_evidence_match_labels(ev))
    for target in targets:
        if any(labels_match(label, target) for label in labels):
            return True
    return False


def _safe_prebake_evidence(graph: Any) -> dict[str, Any]:
    try:
        from maps.sws.graph.runtime import get_prebake_evidence
    except Exception:  # noqa: BLE001
        return {}
    try:
        bucket = get_prebake_evidence(graph)
    except Exception:  # noqa: BLE001
        return {}
    return bucket if isinstance(bucket, dict) else {}


def _world_xy_for_record(
    fid: str,
    node: Any,
    world_xy_resolver: Callable[[str], tuple[float, float] | None] | None,
) -> tuple[float, float] | None:
    if world_xy_resolver is not None:
        try:
            xy = world_xy_resolver(fid)
        except Exception:  # noqa: BLE001
            xy = None
        if xy is not None:
            try:
                return float(xy[0]), float(xy[1])
            except (TypeError, ValueError, IndexError):
                pass
    for attr in ("world_xy", "xy", "position_xy"):
        xy = getattr(node, attr, None)
        if xy is None:
            continue
        try:
            return float(xy[0]), float(xy[1])
        except (TypeError, ValueError, IndexError):
            continue
    metadata = getattr(node, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("world_xy", "xy", "position_xy"):
            xy = metadata.get(key)
            if xy is None:
                continue
            try:
                return float(xy[0]), float(xy[1])
            except (TypeError, ValueError, IndexError):
                continue
    return None


def _distance_m(
    fid: str,
    node: Any,
    *,
    origin_xy: tuple[float, float] | None,
    world_xy_resolver: Callable[[str], tuple[float, float] | None] | None,
) -> float | None:
    if origin_xy is None:
        return None
    xy = _world_xy_for_record(fid, node, world_xy_resolver)
    if xy is None:
        return None
    return math.hypot(
        float(xy[0]) - float(origin_xy[0]),
        float(xy[1]) - float(origin_xy[1]),
    )


def _node_confidence(node: Any) -> float | None:
    for attr in ("confidence", "score", "sam_score"):
        value = getattr(node, attr, None)
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    metadata = getattr(node, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("confidence", "score", "sam_score"):
            try:
                return float(metadata.get(key))
            except (TypeError, ValueError):
                pass
    return None


def _add_source(record: dict[str, Any], source: dict[str, Any]) -> None:
    sources = record.setdefault("sources", [])
    if not isinstance(sources, list):
        record["sources"] = sources = []
    key = (
        source.get("seed_fid"),
        source.get("relation"),
        source.get("distance_m"),
        source.get("direction"),
    )
    for existing in sources:
        if not isinstance(existing, dict):
            continue
        if (
            existing.get("seed_fid"),
            existing.get("relation"),
            existing.get("distance_m"),
            existing.get("direction"),
        ) == key:
            return
    sources.append(source)


def collect_observed_target_candidates(
    graph: Any,
    target_classes: list[str] | tuple[str, ...] | set[str],
    *,
    max_total: int = 0,  # 0 = UNCAPPED (no hardcoded candidate cap: a top-N +
    #                      the by-fid order below would drop a real GT by fid
    #                      coin-flip in a dense scene). All live callers pass 0;
    #                      the default is 0 so an omitting caller can't silently
    #                      reintroduce the cap (feedback_no_hardcoded_candidate_cap).
    radius_m: float | None = None,
    origin_xy: tuple[float, float] | None = None,
    world_xy_resolver: Callable[[str], tuple[float, float] | None] | None = None,
) -> list[dict[str, Any]]:
    """Return target-class objects surfaced by runtime observations.

    The initial Stage-2 candidate set can miss the object that SAM later
    observes from a stop. This helper promotes observed target-class
    prebake features, plus detected nodes matched to prebake features, into
    the same JSON-safe candidate-record shape used by proximity expansion.
    It is category-driven and graph-driven; there are no episode-specific
    fids or labels here.
    """
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict) or not nodes:
        return []
    if not any(_normalize_category_label(t) for t in target_classes):
        return []

    evidence = _safe_prebake_evidence(graph)
    records: dict[str, dict[str, Any]] = {}

    def _ensure_record(
        fid: str,
        *,
        seed_fid: str,
        ev: Any = None,
        match_node: Any = None,
        fallback_observation_count: int = 0,
        fallback_confidence: float | None = None,
    ) -> dict[str, Any] | None:
        fid = _clean_fid(fid)
        if not fid or fid not in nodes:
            return None
        node = nodes.get(fid)
        if node is None:
            return None
        if not _matches_target_class(match_node or node, target_classes, ev=ev):
            return None
        dist = _distance_m(
            fid,
            node,
            origin_xy=origin_xy,
            world_xy_resolver=world_xy_resolver,
        )
        if origin_xy is not None and radius_m is not None and dist is None:
            return None
        if radius_m is not None and dist is not None and dist > float(radius_m):
            return None
        cls, name, provenance = _node_label(node)
        observation_count = int(getattr(ev, "observation_count", 0) or 0)
        if observation_count <= 0:
            observation_count = int(fallback_observation_count)
        try:
            confidence = float(getattr(ev, "confidence", None))
        except (TypeError, ValueError):
            try:
                confidence_sum = float(getattr(ev, "confidence_sum", 0.0))
                confidence = confidence_sum / observation_count
            except (TypeError, ValueError, ZeroDivisionError):
                confidence = (
                    fallback_confidence if fallback_confidence is not None else 0.0
                )
        rec = records.get(fid)
        if rec is None:
            rec = {
                "fid": fid,
                "class_name": cls,
                "name": name,
                "provenance": provenance,
                "observation_count": max(0, observation_count),
                "confidence": max(0.0, float(confidence or 0.0)),
                "attributes": _evidence_attributes(ev) if ev is not None else {},
                "sources": [],
            }
            records[fid] = rec
        else:
            rec["observation_count"] = max(
                int(rec.get("observation_count") or 0),
                max(0, observation_count),
            )
            rec["confidence"] = max(
                float(rec.get("confidence") or 0.0),
                max(0.0, float(confidence or 0.0)),
            )
            if ev is not None and not rec.get("attributes"):
                rec["attributes"] = _evidence_attributes(ev)
        _add_source(
            rec,
            {
                "seed_fid": seed_fid,
                "relation": "runtime_observed_target",
                "distance_m": dist,
                "direction": None,
                "observation_count": rec.get("observation_count"),
                "confidence": rec.get("confidence"),
            },
        )
        return rec

    for fid, ev in evidence.items():
        try:
            count = int(getattr(ev, "observation_count", 0) or 0)
        except (TypeError, ValueError):
            count = 0
        if count <= 0:
            continue
        _ensure_record(str(fid), seed_fid="prebake_evidence", ev=ev)

    for fid, node in nodes.items():
        matched_prebake = _clean_fid(getattr(node, "matched_prebake_id", ""))
        if matched_prebake and matched_prebake in nodes:
            ev = evidence.get(matched_prebake)
            if _matches_target_class(node, target_classes, ev=ev):
                _ensure_record(
                    matched_prebake,
                    seed_fid=str(fid),
                    ev=ev,
                    match_node=node,
                    fallback_observation_count=1,
                    fallback_confidence=_node_confidence(node),
                )
            continue
        provenance = str(getattr(node, "provenance", "") or "").strip()
        if provenance == "prebaked":
            # A prebake node that runtime perception ENRICHED (observation_count
            # bumped by ``runtime.upsert_prior_percept`` when the drone looked at
            # its projected footprint and read a perceived color) is now an
            # observed, attribute-bearing candidate — surface it like a detected
            # node so the commit walk sees it with its perceived color (read
            # off ``node.semantic_attributes`` by the candidate listing). Bare /
            # unobserved priors (observation_count == 0) stay OUT, preserving the
            # lean pool.
            try:
                node_obs = int(getattr(node, "observation_count", 0) or 0)
            except (TypeError, ValueError):
                node_obs = 0
            if node_obs > 0 and _matches_target_class(node, target_classes):
                _ensure_record(
                    str(fid),
                    seed_fid=str(fid),
                    fallback_observation_count=node_obs,
                    fallback_confidence=_node_confidence(node),
                )
            continue
        if provenance != "detected":
            continue
        if not _matches_target_class(node, target_classes):
            continue
        _ensure_record(
            str(fid),
            seed_fid=str(fid),
            fallback_observation_count=1,
            fallback_confidence=_node_confidence(node),
        )

    # NO premature ranking: every same-class object is an EQUAL candidate. The
    # commit-time relational reasoning (the math op over the whole set) does the
    # selection — collection must NOT pre-bias toward nearest / most-observed /
    # most-confident (that salience order is wrong for a dense lot, where the
    # ordinal/relational target is usually NOT the nearest car). Order is a
    # stable, neutral by-fid sort purely for determinism; magnitudes
    # (observation_count, confidence, per-source distance_m) stay ON each record
    # for the relational op to use, they just don't ORDER the set.
    out = sorted(records.values(), key=lambda rec: str(rec.get("fid") or ""))
    if max_total > 0:
        out = out[: int(max_total)]
    return out

__all__ = [
    "collect_observed_target_candidates",
    "labels_match",
]
