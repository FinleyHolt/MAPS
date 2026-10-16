"""A re-served rejection carries its repeat count and the live census."""
from __future__ import annotations

import json

from maps.search.tests.fakes import drive, run, tool_results

# Colour never cuts, so this block is refused every time it is served.
_COLOR_FILTER_BLOCK = ('a = find("car")\n'
                       'b = a.filter(color="gray")\n'
                       'b')
_OK_BLOCK = 'a = find("car")\na'
# An op-SHAPE error: its own echoed line is the fix material, so it gets the
# short count line and never the census.
_SHAPE_BLOCK = 'a = find("car")\nb = a.order(\n'

_CANDS = [
    {"fid": "detected_0", "xy": [0.0, 0.0], "class": "car",
     "color": "gray", "color_conf": 0.8, "search_text": "car"},
    {"fid": "detected_1", "xy": [10.0, 0.0], "class": "car",
     "color": "blue", "color_conf": 0.7, "search_text": "car"},
]


def drive_blocks(blocks: list[str]) -> list[dict]:
    """The served ``run`` results of one mission issuing ``blocks`` in order (the
    replies outlast the no-flight abstain streak)."""
    _, fr, _, _ = drive([[run(b)] for b in blocks] + ["done."] * 3, cands=_CANDS)
    return tool_results(fr)


def test_each_repeat_counts_up_with_the_census_and_no_fid_inventory():
    served = drive_blocks([_COLOR_FILTER_BLOCK] * 5)
    texts = [p["error"] for p in served]
    assert "colour never CUTS" in texts[0]
    assert "occurred" not in texts[0]
    assert len(set(texts)) == len(texts), texts
    for k, err in enumerate(texts[1:], start=2):
        assert "colour never CUTS" in err
        assert f"occurred {k}x." in err, err
        assert "Live surface:" in err and "Columns computed on rows so far" in err
        assert "detected_0" not in err, err
        assert len(err) < 700, len(err)


def test_a_successful_block_does_not_reset_the_class_count():
    served = drive_blocks(
        [_COLOR_FILTER_BLOCK] * 2 + [_OK_BLOCK] + [_COLOR_FILTER_BLOCK])
    assert "error" not in served[2], served[2]
    err4 = served[3]["error"]
    assert "colour never CUTS" in err4
    assert "occurred 3x." in err4


def test_shape_errors_keep_the_short_count_line():
    served = drive_blocks([_SHAPE_BLOCK] * 3)
    assert "syntax error" in served[0]["error"].lower(), served[0]
    for k, payload in enumerate(served[1:], start=2):
        err = payload["error"]
        assert f"occurred {k}x." in err, err
        # a shape error's own echoed line is the fix material; the census is
        # pure token noise beside it
        assert "Live surface:" not in err, err


def test_a_signed_op_error_repeats_with_the_census_too():
    # find's refusal of ';' repeats with its count and the live census.
    served = drive_blocks(['a = find("cuisine=pizza;kebab")\na'] * 2)
    assert "signature" not in served[0]["error"], served[0]
    err = served[1]["error"]
    assert "not a query operator" in err and "signature" not in err, err
    assert "occurred 2x." in err and "Live surface:" in err, err


def test_a_grounding_refusal_serves_no_op_signature():
    # go_to grounds its name through a plumbing find; that find's refusal is
    # served whole, so it must carry only the refusal.
    served = drive_blocks(['go_to("cuisine=pizza;kebab")'])
    err = served[0]["error"]
    assert "not a query operator" in err and "signature" not in err, err


def test_a_prior_withhold_census_counts_no_prior_rows(monkeypatch):
    blocks = ['a = find("cuisine=pizza;kebab")\na'] * 3 + [_COLOR_FILTER_BLOCK] * 3
    flown = json.dumps(drive_blocks(blocks))
    # the census on a rejection's 2nd and 3rd serve, and on a repeated dedup
    n = flown.count("prior map row fid(s)")
    assert n >= 3, flown
    assert all(m in flown for m in ("Live surface:", "rejected again", "still nothing new"))
    monkeypatch.setenv("SEARCH_WITHHOLD", "prior")
    served = json.dumps(drive_blocks(blocks))
    assert served.count("detection fid(s)") == n, served
    assert "prior map row" not in served, served


def test_an_escalation_after_an_unpunctuated_error_starts_a_new_sentence():
    served = drive_blocks(['print(r9)\n'] * 2)
    assert served[0]["error"].endswith("is not defined"), served[0]
    assert "is not defined. This error class has now occurred 2x." in served[1]["error"], served[1]


def test_a_line_prefixed_error_counts_with_its_unprefixed_class():
    # A nested empty window carries its line; the plain one does not.
    served = drive_blocks(['r1 = find("car")\nr1', "r1[30:40]\n", "r1[0:5][3:1]\n"])
    assert "occurred" not in served[1]["error"], served[1]
    assert "This error class has now occurred 2x." in served[2]["error"], served[2]


def test_a_column_list_of_another_length_is_the_same_class():
    from maps.search.parse_compile.reasoner_loop import _Mission

    def sig(cols):
        return _Mission._err_class_sig(
            None, f"line 5: AttributeError: row prior_1 carries no 'a'; "
                  f"its columns are {cols}")
    assert sig(["c", "d", "e", "f", "g", "h", "i"]) == sig(list("cdefghijklm"))


def test_a_look_whose_program_succeeded_restarts_the_consecutive_streak():
    # look(slice) runs its window program before the crop: that success
    # ends the streak even when the crop is then refused.
    served = drive_blocks(['a = find("car")\na\n']
                          + [f"look(a[{w}])\n" for w in ("0:1", "1:2", "0:2")])
    assert all("crop" in p["error"] for p in served[1:]), served
    assert "rejected again" not in served[3]["error"], served[3]
