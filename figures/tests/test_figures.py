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
                        "llm": {"completion_tokens": tokens, "total_tokens": 4 * tokens,
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
