# Third-party notices

MAPS is MIT-licensed ([`LICENSE`](LICENSE)). These files and data in the
repository come from third parties, or are derived from third-party data:

| Path | What it is | Source | License |
|---|---|---|---|
| `benchmarks/citynav/data_prep/_vendor_citynav_rasterize.py` | Upstream CityNav's PDAL/GDAL rasterizer, verbatim. | [water-cookie/citynav](https://github.com/water-cookie/citynav) `rasterize.py` | MIT, Copyright (c) 2024; [`benchmarks/citynav/data_prep/LICENSE.citynav_upstream`](benchmarks/citynav/data_prep/LICENSE.citynav_upstream). |
| `maps/perception/ortho/ground_level.py` | The CityNav blocks' ground levels. | [water-cookie/citynav](https://github.com/water-cookie/citynav) `gsamllavanav/mapdata.py` | MIT, Copyright (c) 2024; same license file. |
| `maps/perception/viz/data/w2c.txt` | The colour-name lookup table the colour extractor reads. | van de Weijer, Schmid, Verbeek and Larlus, "Learning Color Names for Real-World Applications", IEEE Transactions on Image Processing, 2009, and the table distributed with the authors' code. | Not stated by the source. |
| `benchmarks/calcity/data/calcity.json`, `benchmarks/calcity/data_prep/authoring/ground_truth.json` | The Cal City corpus and the positions of its reviewed targets. Rows name and place OpenStreetMap features. | Authored for MAPS; feature names and positions from OpenStreetMap. | OpenStreetMap data © OpenStreetMap contributors, [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). |
| `tiles/osm_cache/{birmingham,cambridge}_osm_local.geojson` | The OpenStreetMap features over each CityNav city, tags verbatim, registered into the CityNav frame; the CityNav bake's source. Each file's `citynav_osm_cache` block holds the registration fitted for MAPS. | [openstreetmap.org](https://www.openstreetmap.org), via the Overpass API. | OpenStreetMap data © OpenStreetMap contributors, [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). |

These are used and not redistributed:

| Item | Where it is used | Source | License |
|---|---|---|---|
| SAM 3.1 checkpoint `facebook/sam3.1` | The SAM serve (`serving/`) fetches it on first load. | Meta, gated on [Hugging Face](https://huggingface.co/facebook/sam3.1). | Meta's terms, accepted on the model page. |
| Meta's `sam3` package | Installed into the `sam3_1` environment from Meta's SAM 3 source; `serving/sam_engine.py` imports it. | Meta's SAM 3 source. | The license distributed with that source. |
| CityNav episodes, CityRefer catalog, SensatUrban point clouds | Fetched by `benchmarks/citynav/data_prep/fetch_data.sh`. | Their upstream releases. | Their upstream licenses. |
| OpenStreetMap | Read from the Cal City world bundle; baked into local tile directories. | [openstreetmap.org](https://www.openstreetmap.org) | © OpenStreetMap contributors, ODbL 1.0. |
