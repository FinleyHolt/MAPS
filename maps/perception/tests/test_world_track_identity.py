"""Every admitted detection gets a track identity, and no frame-local
index ever keys a re-id.

The measured hazard (140#138, four runs): the tracker's birth floor sat
ABOVE the detector's admission threshold, so a conf-0.41 detection got no
``wt_`` id, fell back to the per-frame post-NMS index downstream, and
``_find_track``'s exact-id match re-homed its graph node onto every later
stop's unrelated detection carrying the same index -- a pass destroyed
its own rows as it flew.
"""
from __future__ import annotations

from maps.perception.base import DetectedObject
from maps.perception.world_tracker import WorldTracker, WorldTrackerConfig
from maps.search.search_loop import SearchLoop


def _det(xy, conf, idx):
    return DetectedObject(
        world_xy=(float(xy[0]), float(xy[1])), label="parking",
        confidence=conf, source_tier="sam",
        metadata={"sam3_1_object_id": idx,
                  "track_id": f"sam3_1_img_{idx}"})


def test_a_low_conf_detection_still_earns_a_track_identity():
    """conf 0.41 with score_threshold 0.3 admitted it; the tracker must
    identify it. Two frames, two far-apart objects sharing the same
    frame-local index -> two DISTINCT wt_ ids."""
    tr = WorldTracker(WorldTrackerConfig())
    a = _det((1805.0, 908.0), 0.41, 0)
    tr.update([a])
    b = _det((2100.0, 908.0), 0.41, 0)     # same frame index, 295 m away
    tr.update([b])
    ta, tb = a.metadata.get("track_id"), b.metadata.get("track_id")
    assert ta and ta.startswith("wt_"), a.metadata
    assert tb and tb.startswith("wt_"), b.metadata
    assert ta != tb, (ta, tb)


def test_the_frame_local_index_never_keys_a_detection_id():
    """With no wt_ id, the ingest key is label@xy -- a per-frame identity
    that collides only on genuine co-location -- never the churning
    ``sam3_1_<index>`` and never the churning per-frame track_id."""
    d = _det((1805.2, 908.2), 0.41, 3)
    key = SearchLoop._detection_id(d)
    assert key == "parking@1805,908", key
    # the tracker's id, once present, wins
    d.metadata["track_id"] = "wt_7"
    assert SearchLoop._detection_id(d) == "wt_7"
