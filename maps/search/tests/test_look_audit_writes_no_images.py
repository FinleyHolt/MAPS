"""The look audit persists exactly one file per call: ``manifest.json``.

A run stores no crop pixels — the manifest crop records are the re-render
specs, and the sha256 in each is what upholds the audit claim. This pins the
no-image invariant at the writer (``_persist_look_audit``), the ``jpg``
virtual-filename contract the URLs key on, and the atomic manifest landing.
"""

from __future__ import annotations

import json

from maps.search.parse_compile.reasoner_loop import _persist_look_audit

CROPS = [
    {
        "jpeg": b"\xff\xd8\xff\xe0 delivered bytes",
        "center_fid": "ground_2",
        "map_name": "cambridge_block_7",
        "world_xy": [12.0, 22.0],
        "pose_xy": [12.0, 22.0],
        "context_m": 15.0,
        "sha256": "ab" * 32,
        "renderer_version": 1,
    },
    {
        "jpeg": b"\xff\xd8\xff\xe0 other bytes",
        "center_fid": "ground_9",
        "map_name": "cambridge_block_7",
        "world_xy": [40.0, 8.0],
        "pose_xy": [40.0, 8.0],
        "context_m": 15.0,
        "sha256": "cd" * 32,
        "renderer_version": 1,
    },
]


def test_the_audit_writes_manifest_json_and_nothing_else(tmp_path):
    _persist_look_audit(
        audit_dir=str(tmp_path), stop_id="stop_003", step_idx=3,
        rid_tag="r0", cid="tc_1", annotate=True,
        tool_ack={"n": 2}, vision_text="Overhead crops…", crops=CROPS)

    call_dir = tmp_path / "stop_003" / "tc_1"
    written = sorted(p.name for p in call_dir.iterdir())
    assert written == ["manifest.json"], written

    manifest = json.loads((call_dir / "manifest.json").read_text())
    # Vacuity floor: an empty manifest would pass a no-images scan trivially.
    assert manifest["n_crops"] == 2
    assert len(manifest["crops"]) == 2

    for i, rec in enumerate(manifest["crops"]):
        assert "jpeg" not in rec, "pixel bytes leaked into the manifest"
        # The virtual filename: delivery order + center fid, the URL key.
        assert rec["jpg"] == f"{i:02d}_{CROPS[i]['center_fid']}.jpg"
        assert rec["sha256"] == CROPS[i]["sha256"]
        assert rec["map_name"] == "cambridge_block_7"

    # No image bytes anywhere under the audit tree.
    stray = [p for p in tmp_path.rglob("*")
             if p.is_file() and p.suffix in (".jpg", ".png", ".webp")]
    assert stray == [], stray


def test_the_manifest_lands_atomically(tmp_path):
    """No ``.tmp`` residue — a mid-run reader either sees the full manifest
    or no manifest, never a partial parse."""
    _persist_look_audit(
        audit_dir=str(tmp_path), stop_id="s", step_idx=0, rid_tag="r0",
        cid="c", annotate=False, tool_ack={}, vision_text="", crops=CROPS[:1])
    names = {p.name for p in (tmp_path / "s" / "c").iterdir()}
    assert names == {"manifest.json"}
