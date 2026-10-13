"""Run a released baseline's own evaluator, inside that method's environment.

This file runs under the upstream interpreter (Python 3.10/3.11 with the
authors' requirements), never under this repo's, so it imports nothing from
the repo and only the standard library at module level. It adapts the
upstream tree at run time and leaves the clone untouched:

  every method   ``--blocks``: the Cal City export's ``blocks.json`` merged
                 into the upstream ``MAP_BOUNDS`` / ``GROUND_LEVEL`` dicts
                 before any upstream module reads them
  mgp            ``--splits``: splits outside the list load as empty, so
                 ``main_goal_predictor.py`` runs verbatim on a tree that
                 holds only the requested split files
  hett           ``bert-base-uncase`` paths under ``/cver/...`` remapped to
                 ``$HETT_BERT``; the ``direction`` package the dataset
                 module imports aliased to ``multiagent``; the val/test env
                 list replaced by ``--splits``; per-episode trajectories
                 captured at ``eval_metrics`` (HETT writes only averages)
  flightgpt      ``eval.py``'s endpoint read from ``$FLIGHTGPT_API_BASE``;
                 ``run_nav_gym`` driven over ``--split-files`` (the shipped
                 ``main`` reads only val_unseen easy/medium/hard); with
                 ``--zero-gsam-cache`` every GSAM cache lookup reads empty

Each capture is ``<capture-dir>/<method>_<split>.json``::

    {"method", "split", "episodes": [{"key": [map, obj, desc],
     "trajectory": [[x, y], ...]}], "errors": [[map, obj, desc], ...],
     "upstream_metrics": {...}}

Usage (from the run root the upstream ``defaultpaths`` resolve against)::

    python launch.py --method mgp --clone SRC [--blocks F] [--splits a,b] \\
        -- --mode eval --altitude 50 ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

#: HETT's hard-coded tokenizer/model directory (``main.py``, ``agent.py``,
#: ``models/vln_model.py``).
HETT_BERT_PATH = "/cver/xcding/code/tokenizer_files/bert-base-uncase"

#: FlightGPT ``eval.py`` ``main``: total steps the agent takes, actions per step.
FLIGHTGPT_STEP_NUM = 2
FLIGHTGPT_ACTION_NUM = 75


def register_blocks(mapdata: types.ModuleType, blocks: dict) -> None:
    """Merge an export's ``blocks.json`` into an upstream ``mapdata`` module.

    The dicts are updated in place, so every module that already did
    ``from ...mapdata import MAP_BOUNDS`` sees the blocks too.
    """
    mapdata.GROUND_LEVEL.update(blocks["GROUND_LEVEL"])
    mapdata.MAP_BOUNDS.update(
        {name: mapdata.Bounds(*b) for name, b in blocks["MAP_BOUNDS"].items()}
    )


def remap_bert(name, target: str):
    """``target`` for HETT's hard-coded BERT path, anything else unchanged."""
    return target if str(name).rstrip("/") == HETT_BERT_PATH else name


def patch_from_pretrained(classes, target: str) -> None:
    """Route ``cls.from_pretrained(HETT_BERT_PATH)`` to ``target``."""
    for cls in classes:
        original = cls.from_pretrained.__func__

        def patched(klass, name, *args, _original=original, **kwargs):
            return _original(klass, remap_bert(name, target), *args, **kwargs)

        cls.from_pretrained = classmethod(patched)


def alias_package(alias: str, package: str, submodules: list[str]) -> None:
    """Make ``import alias.sub`` resolve to the already-importable ``package.sub``."""
    import importlib

    sys.modules[alias] = importlib.import_module(package)
    for sub in submodules:
        sys.modules[f"{alias}.{sub}"] = importlib.import_module(f"{package}.{sub}")


def filter_splits(loader, splits: set[str]):
    """``loader(split, ...)`` for a listed split, ``[]`` for any other."""
    def filtered(split, *args, **kwargs):
        return loader(split, *args, **kwargs) if split in splits else []
    return filtered


class ZeroCache(dict):
    """A GSAM map cache in which every ``<map>-<caption>`` reads as empty."""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __missing__(self, key):
        import numpy as np

        return np.zeros(self.shape, dtype=np.float32)


def capture_record(method: str, split: str, trajectories: dict, errors=(), metrics=None) -> dict:
    """The common capture format, from ``{episode_id_tuple: [pose, ...]}``.

    A pose is anything indexable as ``(x, y, ...)``: upstream ``Pose4D``
    named tuples or plain lists.
    """
    return {
        "method": method,
        "split": split,
        "episodes": [
            {"key": list(key), "trajectory": [[float(p[0]), float(p[1])] for p in poses]}
            for key, poses in trajectories.items()
        ],
        "errors": [list(key) for key in errors],
        "upstream_metrics": metrics or {},
    }


def _write(capture_dir: Path, record: dict) -> None:
    capture_dir.mkdir(parents=True, exist_ok=True)
    path = capture_dir / f"{record['method']}_{record['split']}.json"
    path.write_text(json.dumps(record, default=float))
    print(f"[launch] wrote {path}", flush=True)


def _blocks(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text()) if path else None


def run_mgp(args, rest: list[str]) -> None:
    import runpy

    sys.path.insert(0, str(args.clone))
    if args.blocks:
        import gsamllavanav.mapdata as mapdata

        register_blocks(mapdata, _blocks(args.blocks))
    if args.splits:
        import gsamllavanav.dataset.mturk_trajectory as mt

        mt.load_mturk_trajectories = filter_splits(mt.load_mturk_trajectories, set(args.splits))
    script = str(args.clone / "main_goal_predictor.py")
    sys.argv = [script, *rest]
    runpy.run_path(script, run_name="__main__")


def run_hett(args, rest: list[str]) -> None:
    pkg_dir = args.clone / "multiagent"
    sys.path[:0] = [str(pkg_dir), str(args.clone)]
    import transformers

    patch_from_pretrained(
        (transformers.AutoTokenizer, transformers.AutoModel, transformers.BertTokenizerFast),
        os.environ.get("HETT_BERT", "bert-base-uncased"),
    )
    alias_package("direction", "multiagent", ["cityreferobject", "space"])
    if args.blocks:
        import multiagent.mapdata as mapdata

        register_blocks(mapdata, _blocks(args.blocks))

    import env as hett_env
    import main as hett_main

    original_eval = hett_env.CityNavBatch.eval_metrics

    def eval_metrics(self, preds):
        summary, per_episode = original_eval(self, preds)
        trajectories = {tuple(p["instr_id"]): p["trajectory"] for p in preds.values()}
        _write(args.capture_dir, capture_record(
            "hett", self.split, trajectories,
            metrics={k: float(v) for k, v in summary.items()},
        ))
        return summary, per_episode

    hett_env.CityNavBatch.eval_metrics = eval_metrics
    splits = args.splits or ["val_seen", "val_unseen", "test_unseen"]

    def build_val_dataset(hargs, rank=0):
        return {
            split: hett_main.CityNavBatch(
                split, hargs, batch_size=hargs.batch_size,
                seed=hargs.seed + rank, rank=rank, world_size=1,
            )
            for split in splits
        }

    hett_main.build_val_dataset = build_val_dataset
    sys.argv = [str(pkg_dir / "main.py"), *rest]
    hett_main.main()


def run_flightgpt(args, rest: list[str]) -> None:
    del rest
    sys.path.insert(0, str(args.clone))
    if args.blocks:
        import gsamllavanav.mapdata as mapdata

        register_blocks(mapdata, _blocks(args.blocks))
    if args.zero_gsam_cache:
        from gsamllavanav.maps.gsam_map import GSamMap

        GSamMap._map_cache = ZeroCache((1, 240, 240))

    import eval as fg  # FlightGPT's eval.py; importing it loads the image cache
    from navgym.models.CityNavData import CityNavData

    fg.API_CONFIG["api_base"] = os.environ.get("FLIGHTGPT_API_BASE", fg.API_CONFIG["api_base"])
    fg.API_CONFIG["model"] = os.environ.get("FLIGHTGPT_MODEL", fg.API_CONFIG["model"])
    for split_file in args.split_files:
        split = Path(split_file).stem.removeprefix("citynav_")
        data = CityNavData(split_file)
        traj, errors, _ = fg.run_nav_gym(data, split, FLIGHTGPT_STEP_NUM, FLIGHTGPT_ACTION_NUM)
        scored = [ep for ep in data.episodes if ep.id in traj]
        metrics = fg.eval_planning_metrics(scored, traj).to_dict() if scored else {}
        _write(args.capture_dir, capture_record(
            "flightgpt", split, traj,
            errors=[data.episodes[i].id for i in errors],
            metrics={k: float(v) for k, v in metrics.items()},
        ))


def main(argv: list[str]) -> int:
    ours, rest = (argv[: argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--method", choices=("mgp", "hett", "flightgpt"), required=True)
    ap.add_argument("--clone", type=Path, required=True, help="the pinned upstream clone")
    ap.add_argument("--blocks", help="a Cal City export's blocks.json")
    ap.add_argument("--splits", type=lambda s: s.split(","), help="comma list of split names")
    ap.add_argument("--split-files", nargs="+", default=[], help="flightgpt: split JSONs to run")
    ap.add_argument("--capture-dir", type=Path, default=Path("captures"))
    ap.add_argument("--zero-gsam-cache", action="store_true")
    args = ap.parse_args(ours)
    args.clone = args.clone.resolve()
    {"mgp": run_mgp, "hett": run_hett, "flightgpt": run_flightgpt}[args.method](args, rest)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
