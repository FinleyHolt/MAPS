"""Bench-only ablation withholds over the flown engine.

Each name takes one served part away; with none named the model is served
exactly what flew. The bench config's ``withhold:`` list reaches the engine
as ``SEARCH_WITHHOLD`` (comma separated), read at call time, and the flight
mission refuses a run that names any.
"""

from __future__ import annotations

import os

ENV = "SEARCH_WITHHOLD"

# name -> what the model loses.
WITHHOLDS: dict[str, str] = {
    "prior": "the OSM prior's rows: the graph starts with no prior features",
    "look_for": "the segmenter watchlist verb and every sentence teaching it;"
                " no class is ever detected",
    "look": "the close-look verb and every sentence teaching it",
    "images": "every picture: a look, a pass's catch, a map and a flight's"
              " view deliver text only",
    "mission_map": "the system turn's MISSION MAP section",
    "system_turn": "the system turn",
    "end_commit": "every sentence on how the mission ends and what the"
                  " answer is -- the system turn's two bullets, the run"
                  " description's, the nudges', the loop-move refusal's -- and"
                  " the receipts' answer_if_you_reply_now; the cue-word"
                  " standard stays as one bullet",
    "loop": "the turn loop's own behaviours but compaction: the repair"
            " nudges (a text turn always ends the mission), the move-in-a-loop"
            " refusal and its sentence, the identical-program note, the"
            " point-to-row snap, and the reasoning replayed in the prefix",
    "doc": "the run tool's description: it is the bound calls' one-line"
           " signatures instead",
    "library": "every call but go_to and the seeing verbs -- find, explore,"
               " intersection, ends_of, stand, more_of, distance, the rows'"
               " .filter/.order/.distance and the match tiers -- and every"
               " sentence and remedy naming one; go_to is the one move",
    "map": "show_map, the world-state schematic, and every sentence naming"
           " it",
}


def active() -> frozenset[str]:
    names = frozenset(n.strip() for n in os.environ.get(ENV, "").split(",") if n.strip())
    unknown = sorted(names - WITHHOLDS.keys())
    if unknown:
        raise ValueError(f"{ENV} names unknown withholds {unknown}; known: {sorted(WITHHOLDS)}")
    return names


def withheld(name: str) -> bool:
    if name not in WITHHOLDS:
        raise KeyError(f"{name!r} is not a withhold (maps/withhold.py)")
    return name in active()
