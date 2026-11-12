from __future__ import annotations

import re

from maps.mission import served_text
from maps.mission.loop import _Mission
from maps.mission.reply_repair import (
    CODE_REPLY_NUDGE,
    EMPTY_TURN_NUDGE,
    MALFORMED_ENVELOPE_NUDGE,
    NO_FLIGHT_REPLY_NUDGE,
)
from maps.mission.spec import MissionReply, MissionSpec
from maps.mission.tests.fakes import FakeReasoner, bare_ctx, unarmed_boost

# Word-boundary matches, case-insensitive except CARRY (lowercase "carry" is ordinary
# English).
_BANNED_CI = ("qstop", "stop packet", "score_by", "filter_items")
_BANNED_WORDS_CI = (
    "round",
    "rounds",
    "select",
    "terminal",
    "pending",
    "tour",
    "tours",
    # the say channel, named by its own nouns. NOT the bare
    # verb "say": the legend uses it as ordinary English
    # ("how you say beside"), and banning it would forbid
    # the clearest word the doc has.
    "sentence",
    "sentences",
    "lexicon",
)
_BANNED_CS = ("CARRY",)

# Allowlisted collisions, stripped before scanning: the geometry sense of
# "round" (aspect_ratio ~1) and the ordinary English of "says" in a
# result-shaped sentence.
_ALLOW = ("square/round",)


def _violations(text: str) -> list[str]:
    t = text
    for a in _ALLOW:
        t = t.replace(a, " ")
    hits: list[str] = []
    for w in _BANNED_WORDS_CI:
        for m in re.finditer(rf"\b{w}\b", t, flags=re.IGNORECASE):
            hits.append(f"{w}: ...{t[max(0, m.start() - 30) : m.end() + 20]!r}")
    for s in _BANNED_CI:
        for m in re.finditer(re.escape(s), t, flags=re.IGNORECASE):
            hits.append(f"{s}: ...{t[max(0, m.start() - 30) : m.end() + 20]!r}")
    for s in _BANNED_CS:
        for m in re.finditer(rf"\b{s}\b", t):
            hits.append(f"{s}: ...{t[max(0, m.start() - 30) : m.end() + 20]!r}")
    return hits


def _schema_descriptions(node) -> list[str]:
    out: list[str] = []
    if isinstance(node, dict):
        d = node.get("description")
        if isinstance(d, str):
            out.append(d)
        for v in node.values():
            out.extend(_schema_descriptions(v))
    elif isinstance(node, list):
        for v in node:
            out.extend(_schema_descriptions(v))
    return out


def _static_corpus() -> list[str]:
    corpus: list[str] = [served_text.served_system_prompt()]
    for t in served_text.SERVED_TOOLS:
        corpus.append(t["function"]["description"])
        corpus.extend(_schema_descriptions(t["function"]["parameters"]))
    # The shared transport-repair nudges are served too.
    corpus.append(MALFORMED_ENVELOPE_NUDGE)
    corpus.append(CODE_REPLY_NUDGE)
    corpus.append(NO_FLIGHT_REPLY_NUDGE)
    corpus.append(EMPTY_TURN_NUDGE)
    return corpus


def test_static_mission_surface_is_commensurate():
    blob = "\n".join(_static_corpus())
    assert len(blob) > 8_000, "vacuity floor: the surface scan went empty"
    bad = _violations(blob)
    assert not bad, "retired vocabulary on the served surface:\n" + "\n".join(bad[:20])


def test_every_quoted_call_names_a_library_call():
    """The inverse of the worked-example guard in ``served_text``."""
    from maps.library.block_scan import _LIBRARY_CALLS, _MOVE_CALLS, _SAFE_BUILTINS

    known = set(_LIBRARY_CALLS) | set(_MOVE_CALLS) | set(_SAFE_BUILTINS)
    # row/handle methods and the display verb, reached as attributes or by
    # the bare-expression renderer rather than as free names
    known |= {
        "filter",
        "order",
        "distance",
        "show",
        "get",
        "keys",
        "print",
        "range",
        "len",
        "sorted",
        "min",
        "max",
        "sum",
        "abs",
        "round",
    }
    blob = "\n".join(_static_corpus())
    called = {m.group(1) for m in re.finditer(r"\b([a-z_][a-z0-9_]{2,})\(", blob)}
    unknown = sorted(c for c in called if c not in known)
    assert not unknown, "served bytes name a call the surface does not have: " + str(unknown)
    # VACUITY FLOOR: a regex that stops matching would leave this passing on
    # an empty scan.
    assert len(called & known) >= 6, sorted(called & known)


def test_dynamic_mission_output_is_commensurate():
    # Drive the serving paths a block reaches — parse refusals, the
    # executor's own errors, the repeat/census escalation, renders, the
    # flight backstop, and an undeclared tool name. Scan EVERY byte the
    # loop appends.
    def _run(code):
        return ("run", {"code": code})

    script = [
        [_run("grab the car")],  # NameError: no such call
        [_run('a = find("ghost")\na')],  # nothing matches
        [_run('a = find("shop")\na')],  # runs fresh
        [_run('a = find("shop")\na')],  # dedup note
        [_run('a = find("shop")\na')],  # repeat note
        [_run('a = find("shop")\nb = a.filter(color="gray")\nb')],  # refusal
        [_run('a = find("shop")\nb = a.filter(color="gray")\nb')],  # + census
        [_run('a = find("shop")\nlook(a)')],  # render: no image bytes
        [_run("show_map()")],  # map render path
        [
            _run('a = find("shop")\ngo_to(a[0])'),
            _run('go_to("prior_0")'),
            _run('go_to("prior_0")'),
        ],  # budget backstop (cap 1)
        [("teleport", {})],  # unknown tool
        [
            (
                "analyze",
                {
                    "program": {
                        "ops": [
                            {
                                "ref": "r0",
                                "op": "score_by",
                                "args": {
                                    "items": "r9",
                                    "emphasis": [{"key": "color_white", "w": 1}],
                                },
                            }
                        ]
                    }
                },
            )
        ],
        "the goal is the flown object",  # the ending
    ]

    def fly(item):
        return {
            "stop_id": "stop_000",
            "target": item["target_fid"],
            "pose_xy": [0.0, 0.0],
            "new_detections": 0,
            "cue": "cue",
        }

    fr = FakeReasoner(script)
    reply = _Mission(
        boost_queue=unarmed_boost,
        reasoner=fr,
        ctx=bare_ctx(),
        rid_tag="m0",
        transcript=[],
        compute_namespace={
            "cands": [],
            "anchors": [
                {"fid": "prior_0", "xy": [1.0, 2.0], "name": "shop", "search_text": "shop"}
            ],
            "cell_m": 1.0,
        },
        diag={},
        look_renderer=lambda reqs: [],
        mission_spec=MissionSpec(
            fly=fly, declare_concepts=lambda items: None, flight_budget=1, turn_cap=48
        ),
    ).converse("instruction")
    assert isinstance(reply, MissionReply)
    served = [
        m["content"]
        for m in fr.messages
        if m.get("role") in ("tool", "user") and isinstance(m.get("content"), str)
    ]
    # Vacuity floors: the scenarios really produced the surface — each path
    # is pinned by its own marker.
    assert len(served) >= 12, served
    blob = "\n".join(served)
    assert "colour never CUTS" in blob  # the refusal
    assert "This error class has now occurred 2x." in blob  # the census
    assert "not flown." in blob  # the numberless backstop fired
    assert "budget" not in blob  # no budget fact is ever served
    assert "unknown tool" in blob
    assert "unknown tool 'analyze'" in blob  # gated out of this era
    bad = _violations(blob)
    assert not bad, "retired vocabulary in mission output:\n" + "\n".join(bad[:20])


def test_the_transcript_says_what_json_says():
    # The kind vocabulary the mission loop writes is the run surface's.
    tr: list[dict] = []
    fr = FakeReasoner([[("run", {"code": 'look_for("car")'})], "over"])
    _Mission(
        boost_queue=unarmed_boost,
        reasoner=fr,
        ctx=bare_ctx(),
        rid_tag="m0",
        transcript=tr,
        compute_namespace={"cands": [], "anchors": [], "cell_m": 1.0},
        diag={},
        mission_spec=MissionSpec(
            fly=lambda i: {"error": "unused"}, declare_concepts=lambda i: None
        ),
    ).converse("i")
    kinds = {t.get("kind") for t in tr}
    assert "run" in kinds, kinds
    assert not (kinds & {"program", "say", "act", "view"}), kinds
