"""Doc-arm signatures, positional counts, five-digit fids, per-thread find memo."""
import inspect
import threading


from maps.mission.tests.fakes import withhold_for_test
from maps.mission import served_text
from maps.library.interpreter import MissionPython


def test_the_doc_withhold_serves_keyword_only_calls_as_keyword_only(monkeypatch):
    """Under ``doc`` the run description is the calls' signatures; a call that
    refuses positional arguments must not be served as taking them."""
    withhold_for_test(monkeypatch, "doc")
    doc = served_text.served_doc()
    for call in ("stand", "look"):
        sig = inspect.signature(getattr(MissionPython, f"_call_{call}"))
        if any(p.kind is p.KEYWORD_ONLY for p in sig.parameters.values()):
            line = next(ln for ln in doc.splitlines() if ln.strip().startswith(f"{call}("))
            assert "*" in line, line


def test_a_positional_count_refusal_counts_the_models_arguments(monkeypatch):
    """stand("prior_7", "north") passes two arguments, not three."""
    from maps.mission.tests.fakes import drive, run, tool_results
    _r, fr, _f, _d = drive([[run('stand("prior_7", "north")\n')], "done"])
    err = tool_results(fr)[0]["error"]
    assert "3 were given" not in err, err


def test_two_missions_in_one_process_score_finds_on_their_own_surface():
    """The find memo is per surface; two threads on two surfaces must not
    share its idf or scored rows."""
    from maps.query import program as gp
    from maps.query import text_match as T
    a = [{"fid": f"prior_{i}", "xy": [i, 0], "search_text": "house" if i % 2 else "school",
          "name": None, "tags": {}} for i in range(4000)]
    b = [{"fid": f"prior_{i}", "xy": [i, 0], "search_text": "school", "name": None,
          "tags": {}} for i in range(4000)]
    orig, gate, entered = T.token_idf, threading.Event(), threading.Event()

    def slow_idf(rows):     # mission A is inside its scoring when B's find starts
        out = orig(rows)
        if rows and rows[0]["fid"] == "prior_0" and rows[1]["search_text"] == "house":
            entered.set()
            gate.wait(5)
        return out

    T.token_idf = slow_idf
    try:
        res = {}
        ta = threading.Thread(target=lambda: res.setdefault("a", gp.run_program(
            [], list(a), {"ops": [{"ref": "r1", "op": "find", "args": {"query": "house"}}]})))
        ta.start()
        entered.wait(5)
        bl = list(b)
        res["b"] = gp.run_program([], bl, {"ops": [{"ref": "r1", "op": "find",
                                                         "args": {"query": "school"}}]})
        gate.set()
        ta.join()
        # B's surface holds no house: its own find must not serve A's rows
        again = gp.run_program([], bl, {"ops": [{"ref": "r1", "op": "find",
                                                      "args": {"query": "house"}}]})
        assert not any(r.get("search_text") == "house" for r in again["results"]["r1"])
    finally:
        T.token_idf = orig


def test_a_five_digit_detection_fid_is_a_row_like_a_four_digit_one():
    """A Cal City mission's detections pass detected_9999 (upsert.py mints
    detected_<count>): the bare name is the row, and a quoted one is a fid,
    not a name to ground."""
    import json
    from maps.mission.tests.fakes import drive, run, tool_results
    cands = [{"fid": "detected_12345", "xy": [10.0, 10.0], "class": "pool",
              "search_text": "pool"},
             {"fid": "prior_7", "xy": [12.0, 10.0], "name": "Mill Lane",
              "search_text": "mill lane"}]
    _r, fr, flights, _d = drive([[run("go_to(detected_12345)\n")], "done"], cands=cands)
    assert flights == [{"target_fid": "detected_12345"}], tool_results(fr)
    _r, fr, flights, _d = drive([[run('go_to("detected_12345")\n')], "done"], cands=cands)
    assert "did not narrow" not in json.dumps(tool_results(fr)), tool_results(fr)
    _r, fr, _f, _d = drive([[run('distance("detected_12345", "prior_7")\n')], "done"],
                           cands=cands)
    assert "error" not in tool_results(fr)[0], tool_results(fr)


def test_the_doc_withhold_serves_no_parameter_that_only_refuses(monkeypatch):
    """find's extra queries, a cut's unknown keywords and show_map's missing
    centre are refused, so the signatures the doc arm reads do not offer them."""
    withhold_for_test(monkeypatch, "doc")
    doc = served_text.served_doc()
    assert "*more" not in doc and "**kw" not in doc, doc
    assert "around=None" not in doc, doc


def test_a_positional_count_with_keywords_counts_the_models_arguments():
    from maps.mission.tests.fakes import drive, run, tool_results
    _r, fr, _f, _d = drive([[run('stand("prior_7", "north", facing="east")\n')], "done"])
    err = tool_results(fr)[0]["error"]
    assert "3 positional" not in err and "takes no positional" in err, err


def test_a_move_read_before_the_block_binds_its_name_stays_a_move():
    from maps.mission.tests.fakes import drive, run
    _r, fr, flights, _d = drive([[run('x = go_to\ngo_to = 3\nfor r in [1, 2]:\n    x("prior_7")\n')], "done"])
    assert flights == [], flights
