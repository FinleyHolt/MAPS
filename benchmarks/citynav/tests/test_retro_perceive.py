"""The bench's retrospective boost: a flown stop re-rendered at its recorded
pose, re-segmented with the boost vocabulary, and folded as a flight is; the
tool result is the delta."""

from types import SimpleNamespace

from benchmarks.citynav.run_benchmark import episode_runner as er


def _h(segment):
    nodes = {"prior_1": SimpleNamespace(provenance="prebaked")}
    loop = SimpleNamespace(state=SimpleNamespace(runtime_graph=SimpleNamespace(nodes=nodes)))
    return SimpleNamespace(
        sensor=None,
        crop=lambda xy, alt: er.crop_observation(xy, altitude_m=alt),
        episode=SimpleNamespace(map_name="blk"),
        perception=SimpleNamespace(segment=segment),
        world=loop,
    ), nodes


def test_a_flown_stop_is_resegmented_at_its_pose_and_the_delta_served(monkeypatch):
    rendered, folded = [], []
    monkeypatch.setattr(
        er,
        "crop_observation",
        lambda xy, **kw: (
            rendered.append((xy, kw["altitude_m"]))
            or SimpleNamespace(rgb="px", ortho_view_corners_xy=[(0, 0)] * 4)
        ),
    )

    def fold(h, obs, dets):
        folded.append(obs)
        h.world.state.runtime_graph.nodes["detected_9"] = SimpleNamespace(provenance="detected")

    monkeypatch.setattr(er, "perceive", fold)
    h, _ = _h(lambda obs, concepts: [{"label": c} for c in concepts])
    out = er.retro_perceive(
        h, {"pose_xy": (10.0, 20.0), "alt_m": 120.0, "stop_id": "stop_000"}, ["car"]
    )
    assert rendered == [((10.0, 20.0), 120.0)] and len(folded) == 1
    assert out["n_dets"] == 1 and out["new_fids"] == ["detected_9"]
    assert out["note"] == (
        "re-segmented the recorded view at (10, 20) (stop_000) for ['car']: "
        "1 new detection(s) ['detected_9']"
    )


def test_an_unposed_stop_or_a_failing_segmenter_is_a_tool_fact(monkeypatch):
    monkeypatch.setattr(
        er,
        "crop_observation",
        lambda xy, **kw: SimpleNamespace(rgb="px", ortho_view_corners_xy=[(0, 0)] * 4),
    )

    def down(obs, concepts):
        raise ConnectionError("sam down")

    h, _ = _h(down)
    assert er.retro_perceive(h, {"pose_xy": None}, ["car"]) == {
        "error": "the flown observation's pose is unknown"
    }
    assert er.retro_perceive(h, {"pose_xy": (1.0, 2.0), "alt_m": 120.0}, ["car"]) == {
        "error": "segment failed: ConnectionError: sam down"
    }


def test_a_segmenter_outage_on_a_retro_boost_leaves_the_block(monkeypatch):
    """look_for(at=) a flown stop re-segments its recorded view; an outage
    there ends the mission as one inside a flight does (the attempt is
    infra), never a served retro error."""
    import functools

    import pytest

    from maps.mission.tests.fakes import drive, run
    from maps.perception.sam.detect_client import SamServerOutage
    from maps.perception.watchlist import SegmenterTasking

    monkeypatch.setattr(
        er,
        "crop_observation",
        lambda xy, **kw: SimpleNamespace(rgb="px", ortho_view_corners_xy=[(0, 0)] * 4),
    )

    def outage(obs, concepts):
        raise SamServerOutage("no tile of the frame got a /segment answer")

    h, _ = _h(outage)
    tasking = SegmenterTasking(retro_perceiver=functools.partial(er.retro_perceive, h))
    tasking.record_flown({"pose_xy": (1.0, 2.0), "alt_m": 100.0, "stop_id": "stop_000"})
    with pytest.raises(SamServerOutage):
        drive(
            [[run('look_for("car", at="stop_000")')], "done"],
            boost_queue=tasking.boost_queue,
        )
