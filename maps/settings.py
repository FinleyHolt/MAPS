"""The engine's settings: what the environment says, read once.

:func:`from_environ` is the one function in ``maps`` that reads an
environment, and :func:`from_process` the one line that names
``os.environ``. An entry point calls one of them once (a bench entry point
on the mapping its bench carries) and passes the :class:`Settings` down;
nothing below it looks the environment up again. A mission's own directory and
routing key are not settings: they are per-mission values, handed to
``maps.mission.loop.run_mission``.

A field holds its variable as it was set. A default lives beside the code
that applies it, and a value is checked where it is first used, so a
variable no run reads is never refused.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

#: The image store a slow link sends each image to once (``maps.llm.http``).
IMAGE_STORE_ENV = "MAPS_IMAGE_STORE_URL"


@dataclass(frozen=True)
class Endpoint:
    """How the reasoner, the segmenter and the image store are reached."""

    #: ``SEARCH_REASONER_URL``; None is unset (``endpoint_spec.DEFAULT_REASONER_URL``).
    reasoner_url: str | None = None
    #: ``SEARCH_REASONER_API_KEY``, the reasoner's bearer; None when unset or blank.
    reasoner_api_key: str | None = field(default=None, repr=False)
    #: ``SEARCH_REASONER_TIMEOUT_S``, stripped; empty is the default (``endpoint_spec.reasoner_timeout_s``).
    reasoner_timeout_s: str = ""
    #: ``SEARCH_REASONER_MODELS_URL``, stripped; empty derives it from the base URL.
    reasoner_models_url: str = ""
    #: ``SEARCH_ENDPOINT_SPEC_OVERRIDE``, stripped: ``<model_id>:<max_model_len>``.
    spec_override: str = ""
    #: ``SEARCH_ENDPOINT_OFFLINE=1``: nothing is probed and the override is the spec.
    offline: bool = False
    #: ``SEARCH_COMPACT_TOKENIZER_NAME``, stripped; empty reads the model card's.
    tokenizer_name: str = ""
    #: ``MAPS_IMAGE_STORE_URL``, stripped, no trailing slash; empty sends images inline.
    image_store_url: str = ""
    #: ``SAM_SERVER_URL``; None is unset (``DEFAULT_SAM_SERVER_URL``).
    sam_server_url: str | None = None
    #: ``SEARCH_LLM_CACHE_KEY``, a bench worker's prompt-cache key; None when unset or blank.
    cache_key: str | None = None


DEFAULT_SAM_SERVER_URL = "http://127.0.0.1:8772"


def sam_endpoints(endpoint: Endpoint = Endpoint()) -> list[str]:
    """The segmenter's replica base URLs from the run's ``SAM_SERVER_URL``
    (``endpoint.sam_server_url``; a comma list spreads calls across
    replicas), the one resolver the client and endpoint discovery read. Unset
    is the default replica; set but blank raises ``ValueError``."""
    raw = DEFAULT_SAM_SERVER_URL if endpoint.sam_server_url is None else endpoint.sam_server_url
    eps = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    if not eps:
        raise ValueError("SAM_SERVER_URL is empty")
    return eps


@dataclass(frozen=True)
class Sampling:
    """What every request's sampling reads."""

    #: ``SEARCH_SAMPLING_REASONING_EFFORT`` as set; ``model_cards.effort`` strips it, and empty sends none.
    reasoning_effort: str = ""


@dataclass(frozen=True)
class Settings:
    """One run's settings. The default is an empty environment's."""

    endpoint: Endpoint = Endpoint()
    sampling: Sampling = Sampling()


def from_process() -> Settings:
    """The settings this process's environment states: what an entry point
    that holds no environment mapping of its own calls, once."""
    return from_environ(os.environ)


def from_environ(environ: Mapping[str, str]) -> Settings:
    """The settings ``environ`` states."""
    get = environ.get
    return Settings(
        endpoint=Endpoint(
            reasoner_url=get("SEARCH_REASONER_URL"),
            reasoner_api_key=get("SEARCH_REASONER_API_KEY") or None,
            reasoner_timeout_s=get("SEARCH_REASONER_TIMEOUT_S", "").strip(),
            reasoner_models_url=get("SEARCH_REASONER_MODELS_URL", "").strip(),
            spec_override=get("SEARCH_ENDPOINT_SPEC_OVERRIDE", "").strip(),
            offline=get("SEARCH_ENDPOINT_OFFLINE", "").strip() == "1",
            tokenizer_name=get("SEARCH_COMPACT_TOKENIZER_NAME", "").strip(),
            image_store_url=get(IMAGE_STORE_ENV, "").strip().rstrip("/"),
            sam_server_url=get("SAM_SERVER_URL"),
            cache_key=get("SEARCH_LLM_CACHE_KEY") or None,
        ),
        sampling=Sampling(reasoning_effort=get("SEARCH_SAMPLING_REASONING_EFFORT", "")),
    )
