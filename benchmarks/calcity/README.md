# benchmarks/calcity

The Cal City benchmark: target search over California City, CA, the area the
C-12 flew. It runs the same engine and flight contract as
[CityNav](../citynav/README.md) over a 2D world: the block's map raster on a
flat height plane, an OpenStreetMap prior, and an authored corpus. An episode
succeeds when the final pose of its six-action replay lands inside the
target's own annotated footprint (the shape criterion).

## What ships and what you supply

| Item | Path | Source |
|---|---|---|
| The corpus, 1,023 episodes | `data/calcity.json` | Ships. |
| The positions of the reviewed targets | `data_prep/authoring/ground_truth.json` | Ships. |
| The C-12 v6 map raster of the block | `<map>_rgb.tif` | You supply it. |
| A world bundle of OpenStreetMap layers | `osm/<layer>.geojson` in WGS84 for each of the eight `bundle.OSM_LAYERS` | You supply it, covering `aoi.BUNDLE_BBOX_WGS84`. |
| The block's bare-earth surface, read by the flight harness | `calcity_dem.tif` under `SEARCH_CALCITY_TILES` | You supply it from public USGS 3DEP data (the top-level README's Cal City section). |

## Build

```sh
export CALCITY_BUNDLE=<bundle dir>
python -m benchmarks.calcity.data_prep.build_objects_json
python -m benchmarks.calcity.data_prep.build_rgbd_ortho --raster <map>_rgb.tif
python -m benchmarks.calcity.data_prep.bake_tiles
```

| Step | Writes |
|---|---|
| `build_objects_json` | The CityRefer-schema catalog, `<root>/data/cityrefer/objects.json`. |
| `build_rgbd_ortho --raster` | The raster pair under `<root>/data/rgbd_upstream`: the raster as given, `<map>_rgb.tif`, and a constant height plane on its grid, `<map>.tif`. The raster must be the block's frame at its own cell. |
| `bake_tiles` | The OSM prior bake and `calcity_meta.json` under `SEARCH_CALCITY_TILES`. |

`<root>` is the build commands' `--root`, default the bench's own build tree
(`SEARCH_CALCITY_DATA`, then `CALCITY_ROOT`); `bake_tiles --tiles-dir`
defaults to `SEARCH_CALCITY_TILES`.

| Variable | Default | Names |
|---|---|---|
| `SEARCH_CALCITY_DATA`, then `CALCITY_ROOT` | `~/datasets/calcity` | The build tree. |
| `SEARCH_CALCITY_RGBD_DIR` | `<root>/data/rgbd_upstream` | The raster pair the observation crops. A second raster of the block goes in a second directory. |
| `SEARCH_CALCITY_TILES` | `benchmarks/calcity/tiles` | The OSM prior bake. |
| `SEARCH_CALCITY_SPLITS` | `benchmarks/calcity/data` | The corpus. |
| `CALCITY_BUNDLE` | `<root>/bundle` | The world bundle. |

## The reviewed episodes

The reviewed set is derived, not listed: a corpus row is reviewed when its
first target position lies within 0.2 m of a position `ground_truth.json`
lists on its own map. It holds 596 episodes.

```sh
mkdir -p output && python -m benchmarks.calcity.data_prep.reviewed > output/reviewed_episodes.txt
```

## Run

```sh
python -m benchmarks.calcity.bench_fullset --episode-allowlist output/reviewed_episodes.txt --out-root output/calcity
```

`bench_fullset` takes the CityNav driver's flags and reads
`configs/calcity.yaml`: the C-12's Oryx camera at 3,000 ft above ground from
a seeded uniform start. One episode:

```sh
python -m benchmarks.calcity.run_benchmark --config configs/calcity.yaml --episode-id 'calcity::6#6' --out output/calcity_one
```

Read a root with [`benchmarks/citynav/analysis`](../citynav/analysis/README.md).

## Modules

| Module | Role |
|---|---|
| `__init__.py` | `use()`: selects Cal City for the CityNav engine (`SEARCH_BENCHMARK_NAME` and the `SEARCH_CITYNAV_*` roots). |
| `paths.py` | The file layout. |
| `bench_fullset.py`, `run_benchmark/` | The CityNav driver and runner with Cal City selected. |
| `data_prep/aoi.py` | The area, its local metric frame and the flat ground plane. |
| `data_prep/bundle.py` | Reads the world bundle. |
| `data_prep/build_objects_json.py`, `data_prep/build_rgbd_ortho.py`, `data_prep/bake_tiles.py` | The build steps. |
| `data_prep/reviewed.py` | The reviewed episode set. |
