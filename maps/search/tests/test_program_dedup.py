"""Anti-spiral program identity: identical SELF-CONTAINED programs dedup,
with ref LABELS normalized out of the key.

Within one reasoner round the (cands, anchors) surface is frozen, so an
identical self-contained program is a pure re-render — the dispatch returns
the one-line ``_DEDUP_NOTE`` reference instead of re-executing (the observed
failure re-issued one program 18x in a stop and blew the compaction budget;
an INCREMENTING rN defeated byte-identity entirely: 356/357 identical calls,
0 firings). Emission-order permutations stay distinct (order freedom is a
feature). Programs that read carry state (an ``rN`` they do not define, or
``last``/``prev``) can legitimately return new rows as the round's results
accumulate, so they are never deduped.
"""
from __future__ import annotations

from maps.search.parse_compile.reasoner_loop import (
    _DEDUP_NOTE,
    _dedup_ref_norm,
    _program_dedup_key,
)

_FIND = {"ref": "r0", "op": "find",
         "args": {"provenance": "runtime"}}
_FILTER = {"ref": "r1", "op": "filter_items",
           "args": {"items": "r0", "key": "row_index", "op": "le", "value": 3}}


def test_identical_self_contained_programs_share_a_key():
    k1 = _program_dedup_key({"ops": [dict(_FIND), dict(_FILTER)]})
    k2 = _program_dedup_key({"ops": [dict(_FIND), dict(_FILTER)]})
    assert k1 is not None
    assert k1 == k2


def test_different_ops_get_different_keys():
    k1 = _program_dedup_key({"ops": [dict(_FIND)]})
    other = {**_FIND, "args": {"provenance": "prior", "class": "car"}}
    k2 = _program_dedup_key({"ops": [other]})
    assert k1 is not None and k2 is not None
    assert k1 != k2


def test_program_reading_undefined_ref_is_never_deduped():
    # r5 is read but never defined -> resolves from the round's carry, which
    # grows across calls -> exempt.
    ops = [{"ref": "r0", "op": "compare",
            "args": {"items": "r5", "refs": ["building_1"]}}]
    assert _program_dedup_key({"ops": ops}) is None


def test_program_reading_carry_tokens_is_never_deduped():
    for tok in ("last", "prev", "previous", "$last"):
        ops = [{"ref": "r0", "op": "rank_by",
                "args": {"items": tok, "key": "distance to xy"}}]
        assert _program_dedup_key({"ops": ops}) is None, tok


def test_self_defined_refs_do_not_exempt():
    # The program defines every ref it reads (r0 -> r1 -> r2): dedupable.
    ops = [
        dict(_FIND),
        {"ref": "r1", "op": "compare", "args": {"items": "r0",
                                                "refs": ["building_1"]}},
        {"ref": "r2", "op": "score_by", "args": {"items": "r1"}},
    ]
    assert _program_dedup_key({"ops": ops}) is not None


def test_indexed_handle_of_own_ref_still_dedupable():
    # r1[0] indexes a ref the program defines -- still self-contained.
    ops = [
        dict(_FIND),
        {"ref": "r1", "op": "rank_by", "args": {"items": "r0", "key": "distance to xy"}},
        {"ref": "r2", "op": "compare",
         "args": {"items": "cands", "view": {"reference": "r1[0]",
                                             "side": "left"}}},
    ]
    assert _program_dedup_key({"ops": ops}) is not None


def test_off_grammar_carry_ref_is_exempt():
    # An earlier call stored a result under an off-grammar handle ("foo",
    # "r1000") — a program reading it must re-resolve against the live
    # carry, never dedup. Without the carry it IS self-contained.
    for handle in ("foo", "r1000"):
        ops = [{"ref": "r0", "op": "rank_by",
                "args": {"items": handle, "key": "distance to xy"}}]
        assert _program_dedup_key({"ops": ops}, carry_refs=(handle,)) is None
        assert _program_dedup_key({"ops": ops}) is not None


def test_locally_shadowed_carry_ref_stays_dedupable():
    # The program DEFINES r0 itself; a carry entry of the same name is
    # shadowed (the executor never reads the seed), so dedup is safe.
    ops = [dict(_FIND), dict(_FILTER)]
    assert _program_dedup_key({"ops": ops}, carry_refs=("r0", "r1")) is not None


def test_malformed_programs_are_exempt():
    assert _program_dedup_key(None) is None
    assert _program_dedup_key({}) is None
    assert _program_dedup_key({"ops": []}) is None
    assert _program_dedup_key({"ops": "not-a-list"}) is None


def test_note_is_a_terse_reference():
    lower = _DEDUP_NOTE.format(turn=1, handles=' (r0)').lower()
    # It names the WORLD, not the submission: the guard fires on an
    # unmoved surface, so "no program answers differently" is the fact
    # the model can act on and "you repeated yourself" is not.
    assert "no new information since turn 1" in lower
    assert "not re-rendered" in lower
    assert "identical to the program" not in lower


# --------------------------------------------------------------------------- #
# ref-label normalization: an incrementing rN no longer defeats the guard      #
# --------------------------------------------------------------------------- #


def test_incrementing_ref_labels_share_a_key():
    # The measured pathology: the same program re-issued with r0 -> r1 ->
    # r2... (356/357 identical calls, 0 firings). Labels normalize out.
    a = {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "filter_items",
         "args": {"items": "r0", "key": "class", "op": "eq", "value": "car"}},
    ]}
    b = {"ops": [
        {"ref": "r7", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r8", "op": "filter_items",
         "args": {"items": "r7", "key": "class", "op": "eq", "value": "car"}},
    ]}
    ka, kb = _program_dedup_key(a), _program_dedup_key(b)
    assert ka is not None and ka == kb


def test_bracket_suffixed_ref_uses_normalize_too():
    a = {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "cands", "refs": ["$r0[0]"]}},
    ]}
    b = {"ops": [
        {"ref": "r3", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r4", "op": "compare",
         "args": {"items": "cands", "refs": ["$r3[0]"]}},
    ]}
    ka, kb = _program_dedup_key(a), _program_dedup_key(b)
    assert ka is not None and ka == kb


def test_different_wiring_never_collides():
    # Same shape, different READ structure: normalization must keep them
    # distinct (op 2 reads the find in A, the filter in B).
    a = {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "filter_items",
         "args": {"items": "r0", "key": "class", "op": "eq", "value": "car"}},
        {"ref": "r2", "op": "score_by",
         "args": {"items": "r0", "emphasis": [{"key": "name_match", "w": 1}]}},
    ]}
    b = {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "filter_items",
         "args": {"items": "r0", "key": "class", "op": "eq", "value": "car"}},
        {"ref": "r2", "op": "score_by",
         "args": {"items": "r1", "emphasis": [{"key": "name_match", "w": 1}]}},
    ]}
    ka, kb = _program_dedup_key(a), _program_dedup_key(b)
    assert ka is not None and kb is not None and ka != kb


def test_emission_order_permutation_stays_distinct():
    # Order freedom is a FEATURE: permuted emission is not a re-issue.
    a = {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "prior"}},
    ]}
    b = {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "find", "args": {"provenance": "runtime"}},
    ]}
    ka, kb = _program_dedup_key(a), _program_dedup_key(b)
    assert ka is not None and kb is not None and ka != kb


# --- the dedup owes the re-issue its OWN labels -----------------------------
# Identity is judged with ref labels normalized out, so a deduped program's
# results sit under the FIRST run's labels while the caller resolves slots by
# its own. ``_alias_deduped_refs`` closes that on a key hit, joining on
# ``_dedup_ref_norm`` -- the very correspondence the key is computed from, so
# an alias can never mean something the identity did not.

_CROSS = {"op": "crossings", "args": {"refs": ["prior_1", "prior_2"]}}


def test_a_reissue_can_map_its_own_labels_onto_the_stored_ones():
    """THE defect. `go to the intersection of A and B.` reported "0 distinct
    crossings" for two roads the SAME episode had just resolved with `find
    the intersection of A and B.` -- 49 says over 47 episodes on
    calcity_v3base, the prior success being the PRECONDITION, because the
    re-issue resolved its slot from a label the deduped run never wrote."""
    first = [{"ref": "r0", **_CROSS}]
    reissue = [{"ref": "r4", **_CROSS}]
    assert _program_dedup_key({"ops": first}) \
        == _program_dedup_key({"ops": reissue})
    a, b = _dedup_ref_norm(first), _dedup_ref_norm(reissue)
    assert set(a) == set(b), "a shared key must give a TOTAL correspondence"
    assert {b[n]: a[n] for n in b} == {"r4": "r0"}   # and not the identity


def test_the_correspondence_is_total_on_a_multi_op_key_hit():
    first = [{"ref": "r0", **_CROSS},
             {"ref": "r1", "op": "filter_items",
              "args": {"items": "r0", "key": "row_index",
                       "op": "le", "value": 3}}]
    reissue = [{"ref": "r7", **_CROSS},
               {"ref": "r8", "op": "filter_items",
                "args": {"items": "r7", "key": "row_index",
                         "op": "le", "value": 3}}]
    assert _program_dedup_key({"ops": first}) \
        == _program_dedup_key({"ops": reissue})
    a, b = _dedup_ref_norm(first), _dedup_ref_norm(reissue)
    assert set(a) == set(b)
    assert {b[n]: a[n] for n in b} == {"r7": "r0", "r8": "r1"}


def test_programs_whose_normalisation_differs_cannot_share_a_key():
    """A second copy of this map is how an alias starts meaning something the
    identity did not, so the key must be unable to disagree with it."""
    a = [{"ref": "r0", **_CROSS}]
    b = [{"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
         {"ref": "r1", **_CROSS}]
    assert _dedup_ref_norm(a) != _dedup_ref_norm(b)
    assert _program_dedup_key({"ops": a}) != _program_dedup_key({"ops": b})


def test_off_grammar_refs_stay_out_of_the_correspondence():
    """Only rN-grammar refs are renamed, matching the key's own rule, so an
    off-grammar name is never aliased -- it was never normalized."""
    ops = [{"ref": "anchors", "op": "find", "args": {"provenance": "prior"}},
           {"ref": "r0", **_CROSS}]
    assert _dedup_ref_norm(ops) == {"r1": "r0"}
