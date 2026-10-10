"""One image, one name: the sha in ``conversation.jsonl``'s ``frames``, the
file under ``frames/``, and the wire log's ``image_ref`` all hash the same
delivered bytes.

Three records name an image and a reader joins them by that name. If any
one hashed something else (the base64 text, a re-encode) the join would
silently return nothing, which reads exactly like "the model saw no image".
"""

from __future__ import annotations

import hashlib

from maps.llm.http import _scrub_image_parts
from maps.llm.vision_content import build_image_part
from maps.search.parse_compile.reasoner_loop import _deliver_frames

JPEG = b"\xff\xd8\xff\xe0 not really a jpeg but exactly these bytes"


def test_the_wire_log_and_the_delivery_helper_agree_on_the_sha(tmp_path, monkeypatch):
    sha = hashlib.sha256(JPEG).hexdigest()
    part = build_image_part(JPEG)
    scrubbed = _scrub_image_parts([{"role": "user", "content": [part]}])
    assert scrubbed[0]["content"][0]["sha256"] == sha

    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path))
    sink: list[str] = []
    _deliver_frames([JPEG], sink)
    assert sink == [sha]
    on_disk = tmp_path / "frames" / f"{sha}.jpg"
    assert on_disk.read_bytes() == JPEG
    assert hashlib.sha256(on_disk.read_bytes()).hexdigest() == sha


def test_the_same_image_twice_is_one_file_and_two_names(tmp_path, monkeypatch):
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path))
    sink: list[str] = []
    _deliver_frames([JPEG, JPEG], sink)
    assert len(sink) == 2 and len(set(sink)) == 1
    assert [p.name for p in (tmp_path / "frames").iterdir()] == [f"{sink[0]}.jpg"]


def test_no_workdir_records_the_name_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("SEARCH_COMPILE_WORKDIR", raising=False)
    sink: list[str] = []
    _deliver_frames([JPEG, b""], sink)
    assert sink == [hashlib.sha256(JPEG).hexdigest()]
    assert not list(tmp_path.iterdir())
