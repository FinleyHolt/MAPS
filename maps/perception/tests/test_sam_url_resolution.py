"""A blank SAM_SERVER_URL is refused; only an unset one means the default replica.

The client and endpoint discovery read the segmenter's URL through one
resolver (``settings.sam_endpoints``), so discovery probes exactly the
replicas the run posts to, and a set-but-blank value is refused by both
instead of the client quietly posting to the default.
"""

from __future__ import annotations

import pytest

from maps.llm import endpoint_spec
from maps.settings import DEFAULT_SAM_SERVER_URL, from_environ, sam_endpoints


@pytest.mark.parametrize("blank", ["", " ", " , "])
def test_a_blank_sam_url_is_refused_by_the_client_and_by_discovery(blank):
    # A reasoner nothing answers: discovery must refuse before probing it.
    settings = from_environ(
        {"SAM_SERVER_URL": blank, "SEARCH_REASONER_URL": "http://127.0.0.1:9/v1"}
    )
    with pytest.raises(ValueError, match="SAM_SERVER_URL is empty"):
        sam_endpoints(settings.endpoint)
    with pytest.raises(endpoint_spec.EndpointDiscoveryError, match="SAM_SERVER_URL is empty"):
        endpoint_spec.discover_from_settings(settings)


def test_an_unset_sam_url_is_the_default_replica():
    assert sam_endpoints(from_environ({}).endpoint) == [DEFAULT_SAM_SERVER_URL]
    assert sam_endpoints(from_environ({"SAM_SERVER_URL": "http://a:1/, http://b:2"}).endpoint) == [
        "http://a:1",
        "http://b:2",
    ]
