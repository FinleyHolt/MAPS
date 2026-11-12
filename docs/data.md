# Data

The two benches read their data from directories you prepare. This
repository ships the OpenStreetMap caches the CityNav bake reads and the Cal
City corpus; it ships no imagery.

| Bench | Ships here | You supply |
|---|---|---|
| CityNav | The OSM cache per city (`tiles/osm_cache/`), the vendored rasterizer. | The SensatUrban point clouds, from their authors' form. The annotations are fetched by script. |
| Cal City | The corpus (`benchmarks/calcity/data/calcity.json`) and the reviewed targets' positions. | The block's map raster and an OSM bundle over it. For the flight harness, a DEM from public USGS data. |

## CityNav

CityNav's episodes, the CityRefer catalog and the SensatUrban point clouds
come from their upstream sources; this repository redistributes none of them.
One step is yours: SensatUrban's authors release the point clouds through a
[form](https://github.com/QingyongHu/SensatUrban) (name, email, institution;
no account), whose confirmation page links `data_release.zip`. Its terms are
CC BY-NC-SA, non-commercial research use, and no redistribution, which is
why this repository ships no point cloud, raster or picture made from one.
Then one command fetches the annotations, rasterizes the blocks and bakes
the prior:

```sh
export CITYNAV_ROOT=~/datasets/citynav-arxiv
benchmarks/citynav/data_prep/setup.sh --sensaturban <data_release.zip>
```

`--sensaturban` takes the zip or its unpacked folder. Run with no argument,
the script fetches the annotations and says what it is waiting for. It ends
by naming the splits that are ready to fly and printing the bake's
`bake_sha256`. Each step is skipped where its output is on disk, and the
three scripts it runs stand alone (`fetch_data.sh`,
`rasterize_citynav_native.sh`, `bake_tiles.sh`).

| Step | Writes | Notes |
|---|---|---|
| Annotations | `$CITYNAV_ROOT/data/citynav/*.json`, `$CITYNAV_ROOT/data/cityrefer/objects.json` | Clones `water-cookie/citynav-arxiv` and runs its downloader. |
| Point clouds | `$CITYNAV_ROOT/raw_ply/{train,test}/<block>.ply` | Links the 34 block PLYs out of the SensatUrban release. |
| Rasters | `$CITYNAV_ROOT/data/rgbd_upstream/<block>.{png,tif}` | Runs upstream's PDAL pipeline (vendored as `benchmarks/citynav/data_prep/_vendor_citynav_rasterize.py`) in an environment with `pdal`, `rasterio` and `pillow`. `RASTERIZE_PYTHON` names its interpreter; without one, a conda-family tool on `PATH` builds the `rasterize` environment. |
| Bake | `tiles/citynav_2m_osmv3/` | Bakes every block with a raster from the city's shipped OSM cache, `tiles/osm_cache/<city>_osm_local.geojson`, on 2 m cells over the raster's extent. |

Every run manifest records the bake's `bake_sha256`; for these steps it is
`e110df262470ff65275973628260928d73b2f13243087d63976a2308be74990a`, and
`run_report` checks a root's stamp against the bake on disk.

| Variable | Default | Names |
|---|---|---|
| `SEARCH_CITYNAV_DATA`, then `CITYNAV_ROOT` | `~/datasets/citynav-arxiv` | The upstream checkout. |
| `SEARCH_CITYNAV_SPLITS` | `$CITYNAV_ROOT/data/citynav` | The episode split files. |
| `SEARCH_CITYNAV_RGBD_DIR` | `$CITYNAV_ROOT/data/rgbd_upstream` | The nadir rasters the observation crops. |
| `SEARCH_CITYNAV_TILES` | `tiles/citynav_2m_osmv3` | The OSM prior bake. |

See [`benchmarks/citynav`](../benchmarks/citynav/README.md) for the bench
itself.

## Cal City

The Cal City corpus (`benchmarks/calcity/data/calcity.json`, 4,573 episodes)
and the positions of its reviewed targets
(`benchmarks/calcity/data_prep/authoring/ground_truth.json`) ship with the
repository. The bench needs two inputs this repository does not ship; you
supply them:

* The C-12 v6 map raster of the block, `<map>_rgb.tif`.
* A world bundle of OpenStreetMap layers over the block (`osm/<layer>.geojson`
  in WGS84 for each of the eight `bundle.OSM_LAYERS`), named by `CALCITY_BUNDLE`.

Then build the catalog, the raster pair and the tile bake:

```sh
export CALCITY_BUNDLE=<bundle dir>
python -m benchmarks.calcity.data_prep.build_objects_json
python -m benchmarks.calcity.data_prep.build_rgbd_ortho --raster <map>_rgb.tif
python -m benchmarks.calcity.data_prep.bake_tiles
```

The raster pair lands in `<root>/data/rgbd_upstream`, `<root>` being
`SEARCH_CALCITY_DATA` (default `~/datasets/calcity`), where the bench reads it.
[`benchmarks/calcity`](../benchmarks/calcity/README.md) lists what each step
writes and the variables that relocate each part.

### The DEM the flight harness reads

The flight harness also reads the block's bare-earth surface, as the flown
bake carried it: `calcity_dem.tif` beside `calcity_meta.json` under
`SEARCH_CALCITY_TILES`, heights in metres NAVD88 on 10 m cells in EPSG:32611
over the bake's bounds. It is public USGS 3DEP data (the 1/3 arc-second DEM,
from The National Map), resampled onto the bake's frame, for example:

```sh
gdalwarp -t_srs EPSG:32611 -tr 10 10 -r bilinear -te <origin_e> <origin_n> <origin_e + x_max> <origin_n + y_max> \
  USGS_13_*.tif "$SEARCH_CALCITY_TILES/calcity_dem.tif"
```

`origin_e` and `origin_n` are the bake's `georef` in `calcity_meta.json`,
`x_max` and `y_max` its `bounds`. Without the file the flight places every
frame over flat ground at the block's floor (`tile_georef.FLOOR_MSL_M`).
The twin, the live source and the frame registration stand each frame on the
DEM where it is present.

## The example runs

`MAPS-example-runs.zip`, attached to the Releases page of this repository,
holds recorded Cal City missions (each with its conversation, the pictures
the model saw and its score), the bake they were flown on and a basemap
picture. It needs none of the data above: the station opens it as it is. The
top-level README's [See it](../README.md#see-it) section gives the steps.

## The flight recordings

The camera recordings, sortie folders and analysts' adjudications of the
closed-loop flight are not distributed. The replay tiers
([`flight.md`](flight.md)) and the closed-loop measures run on recordings of
your own.
