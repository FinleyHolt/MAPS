"""Every colour word perception can serve counts as read by the find note."""
from __future__ import annotations

from maps.search.analyze_scene.text_match import unmatched_query_tokens


def test_a_tan_or_gold_read_is_not_an_unmatched_query_word():
    for col in ("yellow", "gold", "tan", "brown"):
        rows = [{"fid": "detected_1", "class": "car", "search_text": "car",
                 "color": col}]
        assert unmatched_query_tokens(f"{col} car", rows) == [], col
