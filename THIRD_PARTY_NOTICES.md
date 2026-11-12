# Third-party notices

MAPS is MIT-licensed ([`LICENSE`](LICENSE)). These files and data in the
repository come from third parties, or are derived from third-party data:

| Path | Contents | Source | License |
|---|---|---|---|
| `benchmarks/citynav/data_prep/_vendor_citynav_rasterize.py` | Upstream CityNav's PDAL/GDAL rasterizer, verbatim. | [water-cookie/citynav](https://github.com/water-cookie/citynav) `rasterize.py` | MIT, Copyright (c) 2024; [`benchmarks/citynav/data_prep/LICENSE.citynav_upstream`](benchmarks/citynav/data_prep/LICENSE.citynav_upstream). |
| `maps/perception/ortho/ground_level.py` | CityNav blocks' ground levels. | [water-cookie/citynav](https://github.com/water-cookie/citynav), `mapdata.py` in its `gsamllavanav` package | MIT, Copyright (c) 2024; same license file. |
| `maps/perception/data/w2c.txt` | Colour-name lookup table the colour extractor reads. | van de Weijer, Schmid, Verbeek and Larlus, "Learning Color Names for Real-World Applications", IEEE Transactions on Image Processing, 2009, and the table distributed with the authors' code. | Not stated by the source. |
| `benchmarks/calcity/data/calcity.json`, `benchmarks/calcity/data_prep/authoring/ground_truth.json` | Cal City corpus and the positions of its reviewed targets. Rows name and place OpenStreetMap features. | Authored for MAPS; feature names and positions from OpenStreetMap. | OpenStreetMap data © OpenStreetMap contributors, [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). |
| `tiles/osm_cache/{birmingham,cambridge}_osm_local.geojson` | OpenStreetMap features over each CityNav city, tags verbatim, registered into the CityNav frame; the CityNav bake's source. Each file's `citynav_osm_cache` block holds the registration fitted for MAPS. | [openstreetmap.org](https://www.openstreetmap.org), via the Overpass API. | OpenStreetMap data © OpenStreetMap contributors, [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). |
| `station/web/dist/assets/` | Built station console, bundling the packages in [Station console bundle](#station-console-bundle). | Their npm packages. | Per package, in that table. |
| `docs/figures/overview.webp`, panels A and B | Photographs from the C-12 flight campaign. | U.S. Air Force Test Pilot School, test plan USAFTPS-TP-26-08. | Distribution Statement A: approved for public release. |
| `docs/figures/overview.webp`, panel C basemap | Aerial imagery of California City under the track. | USDA National Agriculture Imagery Program (NAIP), USDA Farm Service Agency. | Public domain. |

## Station console bundle

`station/web/dist/assets/` bundles these packages. Versions are the ones
`station/web/package-lock.json` pins, and licenses are as each package's
`package.json` states them. `maplibre-gl` ships prebuilt, and its build
bundles the packages listed under it.

| Package | Version | License | Copyright |
|---|---|---|---|
| `react` | 19.2.6 | MIT | Copyright (c) Meta Platforms, Inc. and affiliates. |
| `react-dom` | 19.2.6 | MIT | Copyright (c) Meta Platforms, Inc. and affiliates. |
| `scheduler` | 0.27.0 | MIT | Copyright (c) Meta Platforms, Inc. and affiliates. |
| `tailwindcss` (the stylesheet's base layer and utilities) | 4.3.3 | MIT | Copyright (c) Tailwind Labs, Inc. |
| `maplibre-gl` (script and stylesheet) | 5.24.0 | BSD-3-Clause; its glfx.js code MIT | Copyright (c) 2023, MapLibre contributors. Copyright (c) 2020, Mapbox. Copyright (C) 2011 by Evan Wallace. |
| `@maplibre/maplibre-gl-style-spec`, in `maplibre-gl` | 24.10.0 | ISC | Copyright (c) 2020, MapLibre contributors. Copyright (c) 2020, Mapbox. Copyright (C) 2011 by Evan Wallace. |
| `@maplibre/geojson-vt`, in `maplibre-gl` | 6.1.1 | ISC | Copyright (c) 2025 MapLibre contributors. |
| `@maplibre/mlt`, in `maplibre-gl` | 1.2.0 | MIT OR Apache-2.0 | Copyright (c) 2024 MapLibre contributors. |
| `@maplibre/vt-pbf`, in `maplibre-gl` | 4.3.2 | MIT | Copyright (c) 2025 MapLibre contributors. Copyright (c) 2015 Anand Thakker. |
| `@mapbox/point-geometry`, in `maplibre-gl` | 1.1.0 | ISC | Copyright (c) 2024, Mapbox. |
| `@mapbox/tiny-sdf`, in `maplibre-gl` | 2.2.0 | BSD-2-Clause | Copyright (c) 2016-2024 Mapbox, Inc. |
| `@mapbox/unitbezier`, in `maplibre-gl` and its style spec | 0.0.1, 1.0.0 | BSD-2-Clause | Copyright (C) 2008 Apple Inc. All Rights Reserved. |
| `@mapbox/vector-tile`, in `maplibre-gl` | 2.0.5 | BSD-3-Clause | Copyright (c) 2024, Mapbox. |
| `@mapbox/whoots-js`, in `maplibre-gl` | 3.1.0 | ISC | Copyright (c) 2017, Mapbox. |
| `earcut`, in `maplibre-gl` | 3.2.3 | ISC | Copyright (c) 2026, Mapbox. |
| `gl-matrix`, in `maplibre-gl` | 3.4.4 | MIT | Copyright (c) 2015-2025, Brandon Jones, Colin MacKenzie IV. |
| `json-stringify-pretty-compact`, in the style spec | 4.0.0 | MIT | Copyright (c) 2014, 2016, 2017, 2019, 2021, 2022 Simon Lydell. |
| `kdbush`, in `maplibre-gl` | 4.1.0 | ISC | Copyright (c) 2026, Vladimir Agafonkin. |
| `murmurhash-js`, in `maplibre-gl` | 1.0.0 | MIT | Copyright (c) 2011 Gary Court. |
| `pbf`, in `maplibre-gl` | 4.0.2 | BSD-3-Clause | Copyright (c) 2024, Mapbox. |
| `potpack`, in `maplibre-gl` | 2.1.0 | ISC | Copyright (c) 2025, Mapbox. |
| `quickselect`, in the style spec | 3.0.0 | ISC | Copyright (c) 2024, Vladimir Agafonkin. |
| `tinyqueue`, in `maplibre-gl` and its style spec | 3.0.0 | ISC | Copyright (c) 2017, Vladimir Agafonkin. |
| `tslib`, in `maplibre-gl` | Version of `maplibre-gl`'s build | 0BSD | Copyright (c) Microsoft Corporation. |

`@maplibre/mlt` is offered under MIT or Apache-2.0; the MIT text below
applies to it. Each text below applies with the copyright lines of its
packages in the table above in place of `<copyright>`.

MIT:

```text
<copyright>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

ISC:

```text
<copyright>

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
```

BSD-2-Clause is clauses 1 and 2; BSD-3-Clause adds clause 3:

```text
<copyright>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

0BSD:

```text
<copyright>

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
```

## Used and not redistributed

| Item | Where it is used | Source | License |
|---|---|---|---|
| SAM 3.1 checkpoint `facebook/sam3.1` | The SAM serve (`serving/`) fetches it on first load. | Meta, gated on [Hugging Face](https://huggingface.co/facebook/sam3.1). | SAM License, accepted on the model page. |
| Meta's `sam3` package at commit `2814fa6` | Installed into the `sam3_1` environment; `serving/sam_engine.py` imports it. | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | SAM License. |
| CityNav episodes, CityRefer catalog | Fetched by `benchmarks/citynav/data_prep/setup.sh` (`fetch_data.sh`) through upstream CityNav's own downloader. | [water-cookie/citynav-arxiv](https://github.com/water-cookie/citynav-arxiv) | Their upstream licenses. |
| SensatUrban point clouds | You request them from their authors; `benchmarks/citynav/data_prep/setup.sh --sensaturban <data_release.zip>` rasterizes them on your machine. | The authors' form at [QingyongHu/SensatUrban](https://github.com/QingyongHu/SensatUrban). | CC BY-NC-SA. For the terms, see [`docs/data.md`](docs/data.md#citynav). |
| OpenStreetMap | Read from the Cal City world bundle; baked into local tile directories. | [openstreetmap.org](https://www.openstreetmap.org) | © OpenStreetMap contributors, ODbL 1.0. |
| USGS 3DEP elevation (1/3 arc-second DEM) | The flight harness reads a DEM you resample from it ([`docs/data.md`](docs/data.md)). | The National Map, USGS. | Public USGS data. |

The SAM License bars use "for any activities subject to the International
Traffic in Arms Regulations (ITAR)" and end uses prohibited by trade
controls, including military or warfare purposes.

The compactor counts tokens with the served model's tokenizer, which it
fetches from the Hugging Face Hub at first use. Each model card in
[`maps/llm/model_cards.py`](maps/llm/model_cards.py) names its repository:

| Tokenizer repository | Card | License on its Hugging Face page |
|---|---|---|
| `thinkingmachines/Inkling-Small-NVFP4` | `inkling-small` | Apache-2.0 |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | `qwen38-flash-next-nvfp4` | Source model's (`Qwen/Qwen3.8-Flash-Next`): Qwen Community License 1.0 |
| `nvidia/Gemma-4-26B-A4B-NVFP4` | `gemma4-26b-a4b-nvfp4` | Apache-2.0 |
| `google/gemma-4-26B-A4B-it` | `gemma4-26b-a4b-gguf` | Apache-2.0 |
| `google/gemma-4-E4B-it` | `gemma4-e4b` | Apache-2.0 |
| `google/gemma-4-E2B-it` | `gemma4-e2b` | Apache-2.0 |
| `moonshotai/Kimi-K3.1-Pro` | `kimi-k3.1-pro` | Kimi K3.1-Pro License |
| `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `deepseek-v4.1-flash-1017` | MIT |
| `openai/gpt-oss-120b` | `gpt-5.6-sol` | Apache-2.0 |
