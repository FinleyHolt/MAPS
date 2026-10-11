"""Centralised sampling kwargs for reasoner thinking calls.

Every LLM call in the compiler runs in thinking mode: the model reasons
before it commits a structured answer, and the serve's reasoning parser
routes that span into ``message.reasoning_content``. Reasoner turns are
native tool calls (``tools`` + ``tool_choice="auto"``, per-tool
``strict: true``); the thinking span stays unconstrained either way.

The envelope is the served model's card-recommended thinking-mode set
(``maps.llm.model_cards``, keyed off the discovered
``EndpointSpec.model_id``).
"""

from __future__ import annotations

import os
from typing import Any

from maps.llm import endpoint_spec
from maps.llm.model_cards import envelope_kwargs


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

    ``SEARCH_SAMPLING_REASONING_EFFORT`` is the arm's reasoning effort,
    passed through VERBATIM as a top-level string. Unset sends no key and
    the served chat template's own default applies; this layer holds no
    default of its own, because a value invented here would become the
    arm's identity without an arm having asked for it. Discovery refuses
    a value the served card does not take (``model_cards.check_effort``):
    a named rung (``high``) or a raw float spelled as a string (``0.99``).
    The wire type is a STRING either way: a serve that takes the field
    refuses a JSON float for it (``invalid type: floating point 0.99,
    expected a string``), so the value is never parsed into a number here.

    ``reasoning_effort`` given here is the mission's own word and wins over
    the environment: the field station's console names one per mission.
    None reads the arm's.
    """
    if model_id is None:
        model_id = (
            endpoint_spec.spec().model_id if endpoint_spec.spec_installed() else ""
        )
    kwargs: dict[str, Any] = {
        "temperature": temperature,
        **envelope_kwargs(model_id),
        "chat_template_kwargs": {"enable_thinking": True},
    }
    effort = (reasoning_effort if reasoning_effort is not None
              else os.environ.get("SEARCH_SAMPLING_REASONING_EFFORT", "")).strip()
    if effort:
        # ponytail: the request key is spelled `reasoning_effort` for every
        # card; a card that names no effort values (the gemma rows) takes any
        # well-formed value at discovery, and a template without the field
        # ignores it while the row stamps an effort the template never saw.
        # The upgrade path is declaring those cards' values once their serves
        # are checked.
        kwargs["reasoning_effort"] = effort
    return kwargs
