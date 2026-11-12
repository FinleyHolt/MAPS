"""The engine's settings are read from an environment mapping in one place, once."""

from __future__ import annotations

import re
from pathlib import Path

from maps.settings import Endpoint, Sampling, Settings, from_environ, from_process

MAPS = Path(__file__).resolve().parents[1]


def test_an_empty_environment_states_the_defaults():
    assert from_environ({}) == Settings() == Settings(endpoint=Endpoint(), sampling=Sampling())
    e = Settings().endpoint
    assert (
        e.reasoner_url,
        e.reasoner_api_key,
        e.sam_server_url,
        e.sam_video_server_url,
        e.cache_key,
    ) == (None,) * 5
    assert (
        e.reasoner_timeout_s,
        e.reasoner_models_url,
        e.spec_override,
        e.tokenizer_name,
        e.image_store_url,
    ) == ("",) * 5
    assert e.offline is False and Settings().sampling.reasoning_effort == ""


def test_each_variable_reaches_its_field_as_the_engine_reads_it():
    s = from_environ(
        {
            "SEARCH_REASONER_URL": "http://r:8001/v1",
            "SEARCH_REASONER_API_KEY": "sk-1",
            "SEARCH_REASONER_TIMEOUT_S": " 900 ",
            "SEARCH_REASONER_MODELS_URL": " http://r/models ",
            "SEARCH_ENDPOINT_SPEC_OVERRIDE": " m:262144 ",
            "SEARCH_ENDPOINT_OFFLINE": " 1 ",
            "SEARCH_COMPACT_TOKENIZER_NAME": " org/tok ",
            "MAPS_IMAGE_STORE_URL": " http://store/ ",
            "SAM_SERVER_URL": "http://a:1, http://b:2",
            "SAM_VIDEO_SERVER_URL": "http://v:3",
            "SEARCH_LLM_CACHE_KEY": "run/w3",
            "SEARCH_SAMPLING_REASONING_EFFORT": " 0.9 ",
        }
    )
    assert s.endpoint == Endpoint(
        reasoner_url="http://r:8001/v1",
        reasoner_api_key="sk-1",
        reasoner_timeout_s="900",
        reasoner_models_url="http://r/models",
        spec_override="m:262144",
        offline=True,
        tokenizer_name="org/tok",
        image_store_url="http://store",
        sam_server_url="http://a:1, http://b:2",
        sam_video_server_url="http://v:3",
        cache_key="run/w3",
    )
    # The effort is kept as set (the request strips it).
    assert s.sampling == Sampling(reasoning_effort=" 0.9 ")


def test_a_blank_key_is_no_key_and_only_1_is_offline():
    blank = from_environ(
        {
            "SEARCH_REASONER_API_KEY": "",
            "SEARCH_LLM_CACHE_KEY": "",
            "SEARCH_ENDPOINT_OFFLINE": "true",
            "SEARCH_REASONER_URL": "",
            "SAM_SERVER_URL": "",
        }
    )
    assert blank.endpoint.reasoner_api_key is None and blank.endpoint.cache_key is None
    assert blank.endpoint.offline is False
    # Set but blank is not unset: discovery and the segmenter client refuse it, where unset is the default.
    assert blank.endpoint.reasoner_url == "" and blank.endpoint.sam_server_url == ""


def test_the_key_is_not_in_a_settings_repr():
    assert "sk-SECRET" not in repr(from_environ({"SEARCH_REASONER_API_KEY": "sk-SECRET"}))


def test_from_process_reads_this_process_s_environment(monkeypatch):
    monkeypatch.setenv("SEARCH_REASONER_URL", "http://here:1/v1")
    here = from_process()
    assert here.endpoint.reasoner_url == "http://here:1/v1"


def test_only_the_settings_module_reads_an_environment():
    """Every other module of the engine is handed its settings: none names
    ``os.environ`` or ``os.getenv``. The scan reads the engine's own modules
    (its tests apart), and finds the one reader it must."""
    readers = re.compile(r"os\.environ|os\.getenv|environ\.get\(")
    modules = [p for p in MAPS.rglob("*.py") if "tests" not in p.parts]
    assert len(modules) > 60
    found = sorted(
        str(p.relative_to(MAPS)) for p in modules if readers.search(p.read_text(encoding="utf-8"))
    )
    assert found == ["settings.py"]
