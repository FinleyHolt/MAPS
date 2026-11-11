"""The episode engine CityNav and Cal City fly, as the value a bench carries
(``benchmarks.bench.EpisodeEngine``).

Reading a split row or a bake needs none of the mission stack, so those
methods bind at import. The methods that build or fly an episode import
their module when first called: the episode runner brings the mission loop
and its tokenizer (``transformers``), which a row reader, a report or a
``--help`` never pays for. Every method reads its module's name at call
time, so the engine follows a module global a test swaps.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks.bench import Bench
from benchmarks.citynav import adapter
from benchmarks.citynav.run_benchmark import assets


def _episode_runner():
    from benchmarks.citynav.run_benchmark import episode_runner  # noqa: PLC0415  (the mission stack)

    return episode_runner


class Engine:
    def load_split(self, split_file: Path) -> list[dict]:
        return adapter.load_split(split_file)

    def find_row(self, rows: list[dict], episode_id: str) -> tuple[int, dict] | None:
        return adapter.find_row(rows, episode_id)

    def parse_episode(self, row: dict, row_index: int | None = None) -> adapter.CityNavEpisode:
        return adapter.parse_episode(row, row_index=row_index)

    def authored_episode(self, episode_id: str, start_xy: tuple[float, float]) -> adapter.CityNavEpisode:
        return adapter.CityNavEpisode(
            map_name=episode_id.split("::")[0], episode_id=episode_id, description="",
            start_xy=(float(start_xy[0]), float(start_xy[1])), start_z=None, target_xy=None,
            target_object_ids=[], target_mask=None,
        )

    def load_assets(self, bench: Bench, map_name: str) -> assets.BlockAssets:
        return assets.load_episode_assets(bench, map_name)

    def start_altitude_m(self, cfg: Any, map_name: str, start_z: float | None) -> tuple[float, float]:
        return _episode_runner().start_altitude_m(cfg, map_name, start_z)

    def make_perception(self) -> Any:
        return _episode_runner().make_perception()

    def build_episode(self, cfg: Any, episode: Any, block_assets: Any, bench: Bench, *,
                      perception_factory: Callable[[], Any] | None = None) -> Any:
        return _episode_runner().build_episode(cfg, episode, block_assets, bench,
                                               perception_factory=perception_factory)

    def open_session(self, h: Any, *, recorder: Any = None, observe: Any = None) -> Any:
        from benchmarks.citynav.run_benchmark import mission_engine  # noqa: PLC0415  (the mission stack)

        return mission_engine.MissionSession(h, recorder=recorder, observe=observe)

    def is_infra_fault(self, exc: BaseException) -> bool:
        from benchmarks.citynav.run_benchmark import dispatch  # noqa: PLC0415  (the mission stack)

        return dispatch.is_infra_fault(exc)


ENGINE = Engine()
