# Data

This page covers the data each bench reads and how to prepare it. This
repository ships the OpenStreetMap caches the CityNav bake reads and the Cal
City corpus. It ships no imagery.

| Bench | Shipped | You supply |
|---|---|---|
| CityNav | OSM cache per city (`tiles/osm_cache/`), the vendored rasterizer. | SensatUrban point clouds, from their authors' form. A script fetches the annotations. |
| Cal City | Corpus (`benchmarks/calcity/data/calcity.json`) and the reviewed targets' positions. | The block's map raster and an OSM bundle over it. For the flight harness, a DEM from public USGS data. |

## CityNav

CityNav's episodes, the CityRefer catalog, and the SensatUrban point clouds
come from their upstream sources. This repository redistributes none of them.

1. Request the point clouds through the SensatUrban authors'
   [form](https://github.com/QingyongHu/SensatUrban) (name, email,
   institution; no account). Its confirmation page links `data_release.zip`.
   The terms are CC BY-NC-SA, non-commercial research use, and no
   redistribution, so this repository ships no point cloud, raster, or
   picture made from one.

2. Fetch the annotations, rasterize the blocks, and bake the prior:

   ```sh
   export CITYNAV_ROOT=~/datasets/citynav-arxiv
   benchmarks/citynav/data_prep/setup.sh --sensaturban <data_release.zip>
   ```

   `--sensaturban` takes the zip or its unpacked folder. With no argument,
   the script fetches the annotations and says what it is waiting for. It
   ends by naming the splits that are ready to fly and printing the bake's
   `bake_sha256`.

The script skips each step whose output is on disk. Its three scripts, in
`benchmarks/citynav/data_prep/`, also run alone.

| Step | Script | Output | Notes |
|---|---|---|---|
| Annotations | `fetch_data.sh` | `$CITYNAV_ROOT/data/citynav/*.json`, `$CITYNAV_ROOT/data/cityrefer/objects.json` | Clones `water-cookie/citynav-arxiv` and runs its downloader. |
| Point clouds | `setup.sh --sensaturban` | `$CITYNAV_ROOT/raw_ply/{train,test}/<block>.ply` | Links the 34 block PLYs out of the SensatUrban release. |
| Rasters | `rasterize_citynav_native.sh [--blocks <block>...]` | `$CITYNAV_ROOT/data/rgbd_upstream/<block>.{png,tif}` | Runs upstream's `rasterize.py` PDAL pipeline (vendored as `benchmarks/citynav/data_prep/_vendor_citynav_rasterize.py`) in an environment with `pdal`, `rasterio`, and `pillow`. `RASTERIZE_PYTHON` names its interpreter; without one, a conda-family tool on `PATH` builds the `rasterize` environment. |
| Bake | `bake_tiles.sh` | `tiles/citynav_2m_osmv3/` | Bakes every block with a raster under `SEARCH_CITYNAV_RGBD_DIR` from the city's shipped OSM cache, `tiles/osm_cache/<city>_osm_local.geojson` (already in CityNav local XY), on 2 m cells over the raster's extent. |

Every run manifest records the bake's `bake_sha256`. For these steps it is
`e110df262470ff65275973628260928d73b2f13243087d63976a2308be74990a`, and
`run_report` checks a root's stamp against the bake on disk.

For the variables that relocate each part and for the bench itself, see
[`benchmarks/citynav`](../benchmarks/citynav/README.md#prepare-the-data).

## Cal City

The corpus (`benchmarks/calcity/data/calcity.json`) and the positions of its
reviewed targets (`benchmarks/calcity/data_prep/authoring/ground_truth.json`)
ship with the repository. You supply two inputs:

- The C-12 v6 map raster of the block, `<map>_rgb.tif`.
- A world bundle of OpenStreetMap layers over the block
  (`osm/<layer>.geojson` in WGS84 for each of the eight `bundle.OSM_LAYERS`),
  named by `CALCITY_BUNDLE`.

To build the catalog, the raster pair, and the tile bake, and for the
variables that relocate each part, see
[`benchmarks/calcity`](../benchmarks/calcity/README.md#build-the-bench).

### Flight harness DEM

The flight harness also reads the block's bare-earth surface, as the flown
bake carried it.

| Property | Value |
|---|---|
| File | `calcity_dem.tif` beside `calcity_meta.json` under `SEARCH_CALCITY_TILES` |
| Heights | Metres NAVD88 on 10 m cells in EPSG:32611 over the bake's bounds |
| Source | Public USGS 3DEP data (the 1/3 arc-second DEM, from The National Map), resampled onto the bake's frame |

For example:

```sh
gdalwarp -t_srs EPSG:32611 -tr 10 10 -r bilinear -te <origin_e> <origin_n> <origin_e + x_max> <origin_n + y_max> \
  USGS_13_*.tif "$SEARCH_CALCITY_TILES/calcity_dem.tif"
```

`origin_e` and `origin_n` are the bake's `georef` in `calcity_meta.json`;
`x_max` and `y_max` are its `bounds`. The twin, the live source, and the
frame registration stand each frame on the DEM where it is present. Without
the file, the flight places every frame over flat ground at the block's floor
(`tile_georef.FLOOR_MSL_M`).

## Example runs

`MAPS-example-runs.zip`, attached to the Releases page of this repository,
holds recorded Cal City missions (each with its conversation, the pictures
the model saw, and its score), the bake they were flown on, and a basemap
picture. It needs none of the data above: the station opens it as it is. For
the steps, see the
[Quickstart](../README.md#quickstart-review-recorded-searches).

## Flight recordings

The camera recordings, sortie folders, and analysts' adjudications of the
closed-loop flight aren't distributed. The replay tiers
([`flight.md`](flight.md)) and the closed-loop measures run on recordings of
your own.
