"""Centralised sampling kwargs for reasoner thinking calls.

Every LLM call in the compiler runs in thinking mode: the model reasons
before it commits a structured answer, and the serve's reasoning parser
routes that span into ``message.reasoning_content``. Reasoner turns are
native tool calls (``tools`` + ``tool_choice="auto"``, per-tool
``strict: true``); the thinking span stays unconstrained either way.

The envelope is the served model's card-recommended thinking-mode set
(``maps.llm.model_cards``, keyed off the discovered
``EndpointSpec.model_id``). Qwen3.6's card note: ``min_p=0.0`` official
— measured 0 empty turns across 770 calls on the pinned serve,
SERVE_STRICT.md's 50-episode segment; the historical 0.05 floor
addressed an empty-turn mode an earlier stack exhibited.
"""

from __future__ import annotations

import os
from typing import Any

from maps.llm import endpoint_spec
from maps.llm.model_cards import envelope_kwargs


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else None


def thinking_sampling_kwargs(
    *, temperature: float, model_id: str | None = None, reasoning_effort: str | None = None
) -> dict[str, Any]:
    """Build the sampling sub-kwargs for a thinking-mode call.

    Every field is a flat top-level request key. Callers merge this into
    their request kwargs alongside ``model``, ``messages``, ``tools``,
    ``tool_choice``, ``response_format``, ``transport_kind``, etc.

    ``top_k`` and ``chat_template_kwargs`` ride at the top level rather
    than nested under ``extra_body``. vLLM reads them either way; a
    hosted OpenAI-compatible endpoint reads only the flat form and
    silently drops an ``extra_body`` object, which would take them with
    it. Flat is the shape both transports honour.

    ``model_id`` selects the model card. ``None`` resolves the installed
    ``EndpointSpec.model_id`` (the discovered served alias); a process
    with no spec installed takes the fleet-default card
    (``model_cards.card_for``).

    ``temperature``:
      - reasoner turns use ``1.0`` (model card creative default).
      - a lower value (e.g. ``0.6``, the model card precise default)
        suits a more deterministic, less exploratory commit.

    ``SEARCH_SAMPLING_TEMPERATURE`` / ``SEARCH_SAMPLING_PRESENCE_PENALTY``
    override the corresponding field for envelope-sweep measurement runs
    (a sweep sets them per cell; unset — the deployed default — leaves the
    envelope exactly as the card declares, including carrying NO
    presence_penalty key when the card declares no penalty term).
    Thinking mode is never overridable.

    ``SEARCH_SAMPLING_REASONING_EFFORT`` is the arm's reasoning effort,
    passed through VERBATIM as a top-level string. Unset sends no key and
    the served chat template's own default applies; this layer holds no
    default of its own, because a value invented here would become the
    arm's identity without an arm having asked for it. What a card accepts
    is the card's business — a named rung (``high``) or a raw float
    spelled as a string (``0.99``) — and the wire type is a STRING either
    way: a serve that takes the field refuses a JSON float for it
    (``invalid type: floating point 0.99, expected a string``), so the
    value is never parsed into a number here.

    A serve whose template has no such field, or that refuses the rung it
    was handed, answers 400 on EVERY request of the arm. What that costs
    is worth stating plainly, because it is deterministic in the request
    and the loop above cannot tell: ``llm.http`` logs the response body
    and a request summary naming the field and its value, then raises;
    ``reasoner_loop`` counts an ``HTTPStatusError`` as a transient round
    (only an oversize rejection is read as deterministic there), re-sends
    the same bytes to its cap, and ends the episode
    ``transient_failures`` — which drops it from SR as infra. So a
    misspelled effort costs the arm every episode, and the ONE place that
    says why is the request summary in the log.

    ``reasoning_effort`` given here is the mission's own word and wins over
    the environment: the field station's console names one per mission.
    None reads the arm's.
    """
    if model_id is None:
        model_id = (
            endpoint_spec.spec().model_id if endpoint_spec.spec_installed() else ""
        )
    temp_override = _env_float("SEARCH_SAMPLING_TEMPERATURE")
    pp_override = _env_float("SEARCH_SAMPLING_PRESENCE_PENALTY")
    kwargs: dict[str, Any] = {
        "temperature": temperature if temp_override is None else temp_override,
        **envelope_kwargs(model_id),
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if pp_override is not None:
        kwargs["presence_penalty"] = pp_override
    effort = (reasoning_effort if reasoning_effort is not None
              else os.environ.get("SEARCH_SAMPLING_REASONING_EFFORT", "")).strip()
    if effort:
        # ponytail: the request key is spelled `reasoning_effort` for every
        # card, and a card whose dial is spelled otherwise (muse-glimmer
        # reads `reasoning_strength`; inkling's continuous ladder is a
        # SERVE env var, `CARD_EFFORT_ENV`) takes this field and ignores
        # it, while the row stamps an effort the template never saw. The
        # upgrade path is the card: `model_cards` carries the key name and
        # the accepted values the way it already carries the envelope, and
        # this refuses a rung the card does not declare instead of handing
        # the serve a field it discards.
        kwargs["reasoning_effort"] = effort
    return kwargs
