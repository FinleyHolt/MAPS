"""A blank SAM_SERVER_URL is refused; only an unset one means the default replica.

The client and endpoint discovery read the segmenter's URL through one
resolver (``detect_client.sam_endpoints``), so discovery probes exactly the
replicas the run posts to, and a set-but-blank value is refused by both
instead of the client quietly posting to the default.
"""

from __future__ import annotations

import pytest

from maps.llm import endpoint_spec
from maps.perception.sam import detect_client as dc


@pytest.mark.parametrize("blank", ["", " ", " , "])
def test_a_blank_sam_url_is_refused_by_the_client_and_by_discovery(monkeypatch, blank):
    monkeypatch.delenv("SAM_VIDEO_SERVER_URL", raising=False)
    monkeypatch.setenv("SAM_SERVER_URL", blank)
    # A reasoner nothing answers: discovery must refuse before probing it.
    monkeypatch.setenv("SEARCH_REASONER_URL", "http://127.0.0.1:9/v1")
    with pytest.raises(ValueError, match="SAM_SERVER_URL is empty"):
        dc.sam_endpoints()
    with pytest.raises(endpoint_spec.EndpointDiscoveryError, match="SAM_SERVER_URL is empty"):
        endpoint_spec.discover_from_env()


def test_an_unset_sam_url_is_the_default_replica(monkeypatch):
    monkeypatch.delenv("SAM_VIDEO_SERVER_URL", raising=False)
    monkeypatch.delenv("SAM_SERVER_URL", raising=False)
    assert dc.sam_endpoints() == [dc.DEFAULT_SAM_VIDEO_SERVER_URL]
