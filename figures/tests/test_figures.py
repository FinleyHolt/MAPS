"""The figures read every arm off its own run root and draw every model they are given.

A model dropped from a render, or two labels printed one over the other,
both leave a figure that looks complete. And a model the figure code has
never heard of must still be read: the model is the run manifest's, not a
table's.
"""

from __future__ import annotations

import json
from pathlib import Path

from figures import capability_effort_arms, plot_capability_effort, plot_effort_ladder
from figures.plot_capability_effort import LABEL_PITCH, render, spread


def _arm(tmp_path: Path, name: str, model: str, effort: str | None, srs: list[int],
         split: str = "test_unseen", criterion: str = "ring", tokens: int = 1000) -> Path:
    root = tmp_path / name
    root.mkdir()
    rows = []
    for i, sr in enumerate(srs):
        eid = f"m::{i}#0"
        rows.append({
            "episode_id": eid, "worker_id": "w0", "status": "ok", "elapsed_s": 10.0,
            "workdir": f"w0/maps_20261001_1000{i:02d}_citynav_m__{i}_0",
            "metrics": {"episode_id": eid, "SR": sr, "OSR": 1, "NE": 5.0 + 10 * (1 - sr),
                        "success_criterion": criterion,
                        "llm": {"requests": 8, "completion_tokens": tokens, "total_tokens": 4 * tokens,
                                "reasoning_tokens": tokens // 2, "turns": 8}},
        })
    (root / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    flags = {"SEARCH_SAMPLING_REASONING_EFFORT": effort} if effort is not None else {}
    (root / "run_manifest.json").write_text(json.dumps({
        "served_model": model, "split": split, "git_commit": "abcdef0123",
        "search_env_flags": flags, "withhold": [],
    }))
    return root


def test_the_model_is_the_manifest_s_whatever_it_is(tmp_path):
    root = _arm(tmp_path, "q", "/ckpt/qwen38-flash-next-nvfp4", "xhigh", [1, 1, 0, 1])
    rec = capability_effort_arms.read_root(root)
    assert rec["model"] == "qwen38-flash-next-nvfp4"
    assert rec["effort"] == "xhigh" and rec["split"] == "test_unseen"
    assert (rec["n"], rec["sr_n"], rec["sr"], rec["osr"]) == (4, 3, 75.0, 100.0)
    assert rec["completion_tok_mean"] == 1000.0 and rec["total_tok_mean"] == 4000.0
    assert rec["sr_lo"] < 75.0 < rec["sr_hi"]
    bare = tmp_path / "bare"
    bare.mkdir()
    assert capability_effort_arms.build([root, bare])["unmatched_roots"] == [str(bare)]


def test_an_episode_with_no_transcript_prices_nothing(tmp_path):
    from maps.llm.flight_summary import blank

    root = _arm(tmp_path, "q", "qwen36-35b-a3b", "0.9", [1, 0])
    with (root / "episodes.jsonl").open("a") as fh:
        fh.write(json.dumps({
            "episode_id": "m::9#0", "worker_id": "w0", "status": "ok", "elapsed_s": 10.0,
            "workdir": "w0/maps_20261001_100099_citynav_m__9_0",
            "metrics": {"episode_id": "m::9#0", "SR": 0, "success_criterion": "ring", "llm": blank()},
        }) + "\n")
    rec = capability_effort_arms.read_root(root)
    assert rec["n"] == 3 and rec["priced_n"] == 2 and rec["completion_tok_mean"] == 1000.0
    svg = plot_capability_effort.render(capability_effort_arms.build([root]), "light", "test_unseen")
    assert "3 episodes per arm · 2 priced per arm" in svg


def test_spread_separates_and_keeps_order():
    got = spread([(100.0, "a"), (100.4, "b"), (100.8, "c"), (400.0, "d")])
    ys = [y for y, _ in got]
    assert [t for _, t in got] == ["a", "b", "c", "d"]
    assert all(ys[i + 1] - ys[i] >= LABEL_PITCH - 1e-9 for i in range(len(ys) - 1))
    assert ys[0] == 100.0 and ys[-1] == 400.0


def test_every_model_is_named_even_at_one_success_rate(tmp_path):
    models = ["gemma4-e2b", "gemma4-e4b", "inkling-small", "qwen38-flash-next-nvfp4"]
    roots = [_arm(tmp_path, m, m, None, [1, 0]) for m in models]
    roots += [_arm(tmp_path, "ink-lo", "inkling-small", "0.3", [0, 0], tokens=300)]
    svg = render(capability_effort_arms.build(roots), "light", "test_unseen", "total")
    assert all(f">{m}<" in svg for m in models)
    assert "effort 0.3 → 0.3" not in svg and "effort 0.3" in svg
    assert "no scored arm" in render(capability_effort_arms.build(roots), "dark", "calcity")


def test_the_named_rungs_read_in_rung_order():
    arms = [{"effort": "xhigh"}, {"effort": "low"}, {"effort": "medium"}]
    assert plot_capability_effort.dial_range(arms) == "effort low → xhigh"


def test_the_effort_ladder_titles_itself_from_the_manifests(tmp_path):
    roots = [_arm(tmp_path, f"e{e}", "inkling-small", e, [1, int(e == "0.99"), 0],
                  split="calcity", criterion="shape") for e in ("0.0", "0.6", "0.99")]
    roots.append(_arm(tmp_path, "e0.6b", "inkling-small", "0.6", [1, 1, 0],
                      split="calcity", criterion="shape"))
    out = tmp_path / "fig"
    assert plot_effort_ladder.main([*map(str, roots), "--out-dir", str(out)]) == 0
    svg = (out / "effort_ladder_light.svg").read_text()
    assert "Cal City: success and error" in svg
    assert "inkling-small, 4 arms, 3 episodes each, commit abcdef0" in svg
    assert "a second arm at an effort already on the line" in svg


def test_a_split_no_root_carries_exits_nonzero_naming_the_splits_there_are(tmp_path, capsys):
    root = _arm(tmp_path, "q", "qwen36-35b-a3b", "0.9", [1, 0], split="test_unseen")
    assert plot_capability_effort.main([str(root), "--split", "calcity", "--out-dir", str(tmp_path / "f")]) == 2
    assert "test_unseen" in capsys.readouterr().err
    assert not (tmp_path / "f").exists()


def test_a_root_that_is_not_there_is_named_not_dropped(tmp_path, capsys):
    root = _arm(tmp_path, "q", "qwen36-35b-a3b", "0.9", [1, 0])
    gone = str(tmp_path / "typo")
    assert capability_effort_arms.main([str(root), gone, "--out", str(tmp_path / "a.json")]) == 0
    assert json.loads((tmp_path / "a.json").read_text())["unmatched_roots"] == [gone]
    assert gone in capsys.readouterr().err
    assert plot_capability_effort.main([str(root), gone, "--split", "test_unseen",
                                        "--out-dir", str(tmp_path / "f")]) == 0
    assert gone in capsys.readouterr().err


def test_an_arm_left_off_a_figure_is_named(tmp_path, capsys):
    """An arm with no numeric effort (the ladder) or no priced episode (the
    capability plot) is named on stderr, not dropped silently."""
    roots = [_arm(tmp_path, f"e{e}", "inkling-small", e, [1, 0, 0]) for e in ("0.0", "0.6", "0.99")]
    rung = _arm(tmp_path, "xh", "inkling-small", "xhigh", [1, 0])
    assert plot_effort_ladder.main([*map(str, roots), str(rung), "--out-dir", str(tmp_path / "f")]) == 0
    assert f"{rung} left out" in capsys.readouterr().err
    unpriced = _arm(tmp_path, "free", "inkling-small", "0.3", [1, 0], tokens=0)
    assert plot_capability_effort.main([str(roots[0]), str(unpriced), "--split", "test_unseen",
                                        "--out-dir", str(tmp_path / "g")]) == 0
    assert f"{unpriced} left out" in capsys.readouterr().err


def test_a_partial_root_is_read_and_reported_once(tmp_path, capsys):
    root = _arm(tmp_path, "q", "qwen36-35b-a3b", "0.9", [1, 0])
    man = json.loads((root / "run_manifest.json").read_text())
    man["selection"] = {"episodes_n": 3}
    (root / "run_manifest.json").write_text(json.dumps(man))
    rec = capability_effort_arms.read_root(root, allow_partial=True)
    assert rec["n"] == 2
    assert capsys.readouterr().err.count("PARTIAL") == 1
