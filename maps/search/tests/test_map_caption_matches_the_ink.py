"""The map caption claims only the ink the map renderer draws."""
from __future__ import annotations

import re

import pytest

cv2 = pytest.importorskip("cv2")

from maps.search.parse_compile.views import map_vision_text  # noqa: E402
from maps.search.render import map_render  # noqa: E402

_TILE = {"tile_origin_xy": (0.0, 0.0), "cell_size_m": 2.0, "grid_shape": (100, 100)}
_FEATS = [
    {"fid": "prior_1234", "label": "house", "kind": "building", "provenance": "prebaked",
     "contour": [[40.0, 40.0], [120.0, 40.0], [120.0, 120.0], [40.0, 120.0]],
     "cells": [[r, c] for r in range(20, 60) for c in range(20, 60)]},
    {"fid": "detected_7", "label": "car", "provenance": "detected", "cells": [[80, 80], [80, 81]]},
]
_SPEC = {"center_xy": [100.0, 100.0], "context_m": 180.0, "center_fid": "prior_1234",
         "center_label": "house prior_1234",
         "marks": [{"fid": "detected_7", "xy": [161.0, 161.0], "label": "car detected_7"}]}


def test_the_caption_names_the_chips_and_badges_the_map_draws(monkeypatch):
    drawn: list[str] = []
    put = cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda img, txt, *a, **k: (drawn.append(txt), put(img, txt, *a, **k))[1])
    rec = map_render.render_map_from_spec(_SPEC, _FEATS, **_TILE)
    caption = map_vision_text([rec])
    assert "1234" in drawn and "d7" in drawn
    # a prior chip is the fid's bare digits, and no ordinal badge is drawn
    assert "NNNN chip" in caption and "pNNNN" not in caption
    assert not any(re.fullmatch(r"[12]", t) for t in drawn)
    assert "Numbered badges" not in caption and "1 = " not in caption
    assert "house prior_1234 (center) ; car detected_7 (context)" in caption


def test_a_map_around_a_fid_chips_only_its_centre_and_marks():
    other = {"fid": "prior_55", "label": "house", "kind": "building", "provenance": "prebaked",
             "contour": [[140.0, 40.0], [180.0, 40.0], [180.0, 80.0], [140.0, 80.0]]}
    rec = map_render.render_map_from_spec({**_SPEC, "marks": []}, _FEATS + [other], **_TILE)
    assert rec["priors_chipped"] == 1 and rec["priors_unchipped"] == 0
    assert "chips only its centre and marks" in map_vision_text([rec])
