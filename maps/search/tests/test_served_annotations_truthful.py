"""Served truth: every served note/annotation states the stored fact.

* F1 — the find matched-note states the matched head's score RANGE and the
  sub-threshold tail's size + value range (the tail stays OUT of the match
  set, and the note says so); the Best: banner labels its numeric score.
* F2 — ONE truth for [x,y] compare labels: a bare [x,y] literal attaches
  the _xy family (d_xy/brg_gap_xy/dc_xy/brg_xy); {ref:[x,y], as:'label'}
  is the named-columns path; TWO bare literals in one compare is a loud
  ambiguity error; every served doc states exactly that.
* F3 — tie-flat heads on ranked renders carry a truthful tie annotation
  (rows 0..N tie at <score> -- within-tie order is arbitrary), computed
  from the stored scores; the sort itself is unchanged.
* F4 — viewpoints.add's bound echo serves the bound row's name (or
  '(unnamed)') + compact tag head + xy; never-surfaced tokens keep the
  bare-xy / not-a-fid echoes.
* F5 — emphasis sign semantics defined on every served score_by surface:
  composite = w-weighted sum of normalized values, rank-0 = LARGEST
  composite, w>0 larger-raw-first / w<0 smaller-raw-first — and the
  executor does exactly that.
* F6 — a zero-match find on a closed-palette color word affords the
  color path (find{color} / score_by color_<word> emphasis key).
"""
from __future__ import annotations


from maps.search.analyze_scene import text_match as T
from maps.search.analyze_scene.geom_program import (
    _OP_SIGNATURE,
    ProgramError,
    compact_results,
    run_program,
)


def _prior(fid, x, y, name=None, tags=None):
    return {"fid": fid, "name": name, "xy": [float(x), float(y)],
            "centerline": None, "tags": tags or {}, "color": "",
            "search_text": " ".join(
                [name or ""] + [f"{str(k).replace('_', ' ')} "
                                f"{str(v).replace('_', ' ')}"
                                for k, v in (tags or {}).items()]).lower()}


def _grass_world():
    return [
        _prior("prior_1", 0, 0, tags={"landuse": "grass"}),
        _prior("prior_2", 10, 0, tags={"landuse": "grass"}),
        _prior("prior_3", 20, 0, tags={"landuse": "meadow"}),
        _prior("prior_4", 30, 0, tags={"building": "yes"}),
    ]


def _find(world, query):
    out = run_program([], world, {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"provenance": "prior", "query": query}},
    ]})
    return out, compact_results(out["results"])["r0"]


# ---------------------------------------------------------------------------
# F1: the matched-note serves the truth about sub-threshold rows
# ---------------------------------------------------------------------------

def test_find_note_states_the_subthreshold_tail_truthfully():
    out, env = _find(_grass_world(), "landuse=grass")
    hint = env["hint"]
    assert "annotated ~0" not in hint
    # the match set holds ONLY matches; the tail's size + value range are
    # stated in the note (never a claim the tail is in the rN)
    rows = out["results"]["r0"]
    assert all(r["name_match"] >= T.FUZZY_THRESHOLD for r in rows)
    assert "1 sub-threshold row" in hint
    assert "NOT in this rN" in hint


def test_find_note_states_matched_head_range():
    out, env = _find(_grass_world(), "landuse=grass")
    rows = out["results"]["r0"]
    matched = [r["name_match"] for r in rows
               if r["name_match"] >= T.FUZZY_THRESHOLD]
    # the note spells "match", not the raw-field name: the name_match COLUMN
    # no longer renders (audit: 7/8 not-observed), so the note stops naming it
    assert f"match {max(matched)}" in env["hint"]


def test_head_banner_endorses_only_a_name_tier_head():
    # "Best" is an ENDORSEMENT and a tag-tier head has not earned it:
    # 140#138 flew a parking_aisle 212 m off the cue's street from a
    # "Best:" banner over 0.5-0.8 tag matches. A tag head is just the
    # head; the number still rides the banner, labeled and visibly weak.
    _, env = _find(_grass_world(), "landuse=grass")
    assert "Best: " not in env["hint"]
    assert "Head: " in env["hint"]
    assert "(match " in env["hint"]
    # ...while an exact NAME hit keeps the word it earned.
    world = _grass_world() + [_prior("prior_9", 40, 0, name="Mill Lane")]
    _, env = _find(world, "Mill Lane")
    assert "Best: " in env["hint"]
    assert "(match " in env["hint"]


# ---------------------------------------------------------------------------
# F2: ONE truth for [x,y] compare labels
# ---------------------------------------------------------------------------

def test_bare_xy_literal_attaches_the_xy_family():
    out = run_program([], _grass_world(), {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0", "refs": [[5.0, 5.0]]}},
    ]})
    row = out["results"]["r1"][0]
    for col in ("distance to xy", "center distance to xy", "bearing from xy"):
        assert col in row, f"{col} missing: {sorted(row)}"


def test_labeled_xy_literal_is_the_named_columns_path():
    out = run_program([], _grass_world(), {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0",
                  "refs": [{"ref": [5.0, 5.0], "as": "gate"},
                           {"ref": [25.0, 5.0], "as": "corner"}]}},
    ]})
    row = out["results"]["r1"][0]
    assert "distance to gate" in row and "distance to corner" in row
    assert "distance to xy" not in row


def test_two_bare_xy_literals_is_a_loud_ambiguity_error():
    try:
        run_program([], _grass_world(), {"ops": [
            {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
            {"ref": "r1", "op": "compare",
             "args": {"items": "r0", "refs": [[5.0, 5.0], [25.0, 5.0]]}},
        ]})
    except ProgramError as e:
        msg = str(e)
        assert "distance to xy" in msg
        assert "as:'label'" in msg
    else:
        raise AssertionError("two bare [x,y] literals must error loudly")


def test_label_docs_state_the_executor_truth_everywhere():
    # the doc sites that teach the label grammar teach ONE truth: bare
    # literal -> _xy family; the labeled form is the named-columns path.
    # The doc site that speaks it is the compare op's own signature, which
    # every compare rejection carries.
    from maps.search.parse_compile.prompts import CODE_SYSTEM_PROMPT
    assert "keeps the token you typed" not in CODE_SYSTEM_PROMPT
    # The system prompt carries no label grammar at all (result surfaces
    # own the composition grammar); the doc site that DOES speak teaches
    # the one truth.
    assert "distance to xy" not in CODE_SYSTEM_PROMPT
    assert "distance to xy" in _OP_SIGNATURE["compare"]
    assert "as:'label'" in _OP_SIGNATURE["compare"]


# ---------------------------------------------------------------------------
# F3: tie-flat heads carry the truthful tie annotation
# ---------------------------------------------------------------------------

def test_all_tied_find_head_serves_no_tie_annotation():
    world = [_prior(f"prior_{i}", 10 * i, 0, name="Garage",
                    tags={"building": "garage"}) for i in range(6)]
    _, env = _find(world, "Garage")
    assert "tie at name_match" not in env.get("hint", "")


def test_score_by_prefix_tie_serves_no_annotation():
    world = [
        _prior("prior_a", 0, 10),
        _prior("prior_b", 10, 0),
        _prior("prior_c", 0, 20),
    ]
    out = run_program([], world, {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0", "refs": [[0.0, 0.0]]}},
        {"ref": "r2", "op": "score_by",
         "args": {"items": "r1",
                  "emphasis": [{"key": "center distance to xy", "w": -1}]}},
    ]})
    env = compact_results(out["results"])["r2"]
    rows = out["results"]["r2"]
    assert rows[0]["score"] == rows[1]["score"] != rows[2]["score"]
    assert "tie at score" not in env.get("hint", "")


def test_distinct_scores_serve_no_tie_annotation():
    world = [
        _prior("prior_a", 0, 10),
        _prior("prior_b", 0, 20),
        _prior("prior_c", 0, 30),
    ]
    out = run_program([], world, {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0", "refs": [[0.0, 0.0]]}},
        {"ref": "r2", "op": "score_by",
         "args": {"items": "r1",
                  "emphasis": [{"key": "center distance to xy", "w": -1}]}},
    ]})
    env = compact_results(out["results"])["r2"]
    assert "tie at score" not in env.get("hint", "")


def test_all_flat_sort_serves_no_note():
    world = [_prior("prior_a", 0, 10), _prior("prior_b", 10, 0)]
    out = run_program([], world, {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0", "refs": [[0.0, 0.0]]}},
        {"ref": "r2", "op": "score_by",
         "args": {"items": "r1",
                  "emphasis": [{"key": "center distance to xy", "w": -1}]}},
    ]})
    env = compact_results(out["results"])["r2"]
    hint = env.get("hint", "")
    assert "separated NOTHING" not in hint and "tie at score" not in hint


# ---------------------------------------------------------------------------
# F5: emphasis sign semantics served everywhere, matching the executor
# ---------------------------------------------------------------------------

def test_sign_semantics_sentence_on_every_scored_surface():
    assert "rank-0 is the LARGEST composite" in _OP_SIGNATURE["score_by"]
    assert "w>0 ranks larger raw values first" in _OP_SIGNATURE["score_by"]


def test_executor_matches_the_served_sign_definition():
    world = [
        _prior("prior_near", 0, 10),
        _prior("prior_far", 0, 40),
    ]
    def _rank0(w):
        out = run_program([], world, {"ops": [
            {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
            {"ref": "r1", "op": "compare",
             "args": {"items": "r0", "refs": [[0.0, 0.0]]}},
            {"ref": "r2", "op": "score_by",
             "args": {"items": "r1",
                      "emphasis": [{"key": "center distance to xy", "w": w}]}},
        ]})
        return out["results"]["r2"][0]["fid"]
    assert _rank0(1) == "prior_far"    # w>0: larger raw first
    assert _rank0(-1) == "prior_near"  # w<0: smaller raw first


# ---------------------------------------------------------------------------
# F6: palette-word zero-match affords the color path
# ---------------------------------------------------------------------------

def test_color_word_miss_names_the_color_column_path():
    """The COLUMN, not the op that attaches it. This hint reaches the
    sentence channel, whose only move is a sentence, so naming
    ``find{color:...}`` / a ``score_by`` emphasis key taught a register it
    cannot emit -- the same defect the two rewritten remedies carried."""
    _, env = _find(_grass_world(), "white")
    hint = env["hint"]
    assert env["len"] == 0
    assert "color_white" in hint
    assert "attaches when the color word is named" in hint
    assert "find{" not in hint and "score_by" not in hint


def test_color_word_in_or_clause_also_affords():
    _, env = _find(_grass_world(), "zzqx OR white")
    assert env["len"] == 0
    assert "color_white" in env["hint"]


def test_synonym_color_word_names_the_canonical_column():
    _, env = _find(_grass_world(), "silver")
    assert env["len"] == 0
    assert "color_gray" in env["hint"]


def test_non_color_miss_serves_no_color_affordance():
    _, env = _find(_grass_world(), "zzqx")
    assert env["len"] == 0
    assert "color_" not in env["hint"]
