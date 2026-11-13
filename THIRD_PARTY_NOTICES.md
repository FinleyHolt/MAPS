# Third-party notices

MAPS is MIT-licensed ([`LICENSE`](LICENSE)). These files and data in the
repository, the README's quickstart video, and the example-runs release asset
come from third parties, or are derived from third-party data:

| Path | Contents | Source | License |
|---|---|---|---|
| `benchmarks/citynav/data_prep/_vendor_citynav_rasterize.py` | Upstream CityNav's PDAL/GDAL rasterizer, verbatim. | [water-cookie/citynav](https://github.com/water-cookie/citynav) `rasterize.py` | MIT, Copyright (c) 2024; [`benchmarks/citynav/data_prep/LICENSE.citynav_upstream`](benchmarks/citynav/data_prep/LICENSE.citynav_upstream). |
| `maps/perception/ortho/ground_level.py` | CityNav blocks' ground levels. | [water-cookie/citynav](https://github.com/water-cookie/citynav), `mapdata.py` in its `gsamllavanav` package | MIT, Copyright (c) 2024; same license file. |
| `maps/perception/data/w2c.txt` | Colour-name lookup table the colour extractor reads, byte-identical to `ColorNaming/w2c.txt` in the authors' `ColorNaming.tar` (dated 9 April 2009). | J. van de Weijer, C. Schmid, J. Verbeek and D. Larlus, "Learning Color Names for Real-World Applications", IEEE Transactions on Image Processing 18(7), 2009. The archive is the "color naming" download on the authors' [Color Naming page](https://lear.inrialpes.fr/people/vandeweijer/color_names.html). | Not stated by the source: the page and the archive's `readme.txt` name the paper and carry no license. |
| `benchmarks/calcity/data/calcity.json`, `benchmarks/calcity/data_prep/authoring/ground_truth.json` | Cal City corpus and the positions of its reviewed targets. Rows name and place OpenStreetMap features. | Authored for MAPS; feature names and positions from OpenStreetMap. | OpenStreetMap data © OpenStreetMap contributors, [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). |
| `tiles/osm_cache/{birmingham,cambridge}_osm_local.geojson` | OpenStreetMap features over each CityNav city, tags verbatim, registered into the CityNav frame; the CityNav bake's source. Each file's `citynav_osm_cache` block holds the registration fitted for MAPS. | [openstreetmap.org](https://www.openstreetmap.org), via the Overpass API. | OpenStreetMap data © OpenStreetMap contributors, [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). |
| `station/web/dist/assets/` | Built station console, bundling the packages in [Station console bundle](#station-console-bundle). | Their npm packages. | Per package, in that table. |
| `docs/figures/overview.webp`, panels A and B | Photographs from the C-12 flight campaign. | U.S. Air Force Test Pilot School, test plan USAFTPS-TP-26-08. | Distribution Statement A: approved for public release. |
| `docs/figures/overview.webp`, panel C inset crops | Crops of the C-12's map imagery. | U.S. Air Force Test Pilot School, test plan USAFTPS-TP-26-08. | Distribution Statement A: approved for public release. |
| README quickstart video (a GitHub upload and a release asset) | Console scenes over the C-12's map imagery. | U.S. Air Force Test Pilot School, test plan USAFTPS-TP-26-08. | Distribution Statement A: approved for public release. |
| `MAPS-example-runs.zip` (a release asset): `basemap.jpg` and each search's `frames/` | The C-12's map imagery and crops of it. | U.S. Air Force Test Pilot School, test plan USAFTPS-TP-26-08. | Distribution Statement A: approved for public release. |
| `MAPS-example-runs.zip` (a release asset): `bake/` | The Cal City bake's georeference and OpenStreetMap features. | [openstreetmap.org](https://www.openstreetmap.org). | OpenStreetMap data © OpenStreetMap contributors, [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). |
| `docs/figures/overview.webp`, panel C basemap | Aerial imagery of California City under the track. | USDA National Agriculture Imagery Program (NAIP), USDA Farm Service Agency. | Public domain. |

## Station console bundle

`station/web/dist/assets/` bundles these packages, and licenses are as each
package's license file states them. A package the station's build bundles
carries the version `station/web/package-lock.json` pins. `maplibre-gl`
ships prebuilt: the packages marked "in `maplibre-gl`" are bundled by its own
build, at the versions that build resolved, and their Version column gives
the version the lockfile installs beside it.

| Package | Version | License | Copyright |
|---|---|---|---|
| `react` | 19.2.6 | MIT | Copyright (c) Meta Platforms, Inc. and affiliates. |
| `react-dom` | 19.2.6 | MIT | Copyright (c) Meta Platforms, Inc. and affiliates. |
| `scheduler` | 0.27.0 | MIT | Copyright (c) Meta Platforms, Inc. and affiliates. |
| `tailwindcss` (the stylesheet's base layer and utilities) | 4.3.3 | MIT | Copyright (c) Tailwind Labs, Inc. |
| `maplibre-gl` (script and stylesheet) | 5.24.0 | BSD-3-Clause; its glfx.js code MIT; its d3-color portion BSD-3-Clause | Copyright (c) 2023, MapLibre contributors. Copyright (c) 2020, Mapbox. Copyright (C) 2011 by Evan Wallace. Copyright 2010-2016 Mike Bostock. |
| `@maplibre/maplibre-gl-style-spec`, in `maplibre-gl` | 24.10.0 | BSD-3-Clause; its glfx.js code MIT; its d3-color portion BSD-3-Clause (its `package.json` says ISC) | Copyright (c) 2020, MapLibre contributors. Copyright (c) 2020, Mapbox. Copyright (C) 2011 by Evan Wallace. Copyright 2010-2016 Mike Bostock. |
| `@maplibre/geojson-vt`, in `maplibre-gl` | 6.1.1 | MIT; its Mapbox code ISC (its `package.json` says ISC) | Copyright (c) 2025 MapLibre contributors. Copyright (c) 2024, Mapbox. |
| `@maplibre/mlt`, in `maplibre-gl` | 1.2.0 | MIT OR Apache-2.0 | Copyright (c) 2024 MapLibre contributors. |
| `@maplibre/vt-pbf`, in `maplibre-gl` | 4.3.2 | MIT; its geojson_wrapper.js BSD-3-Clause | Copyright (c) 2025 MapLibre contributors. Copyright (c) 2015 Anand Thakker. Copyright (c) 2014, Mapbox. |
| `@mapbox/point-geometry`, in `maplibre-gl` | 1.1.0 | ISC | Copyright (c) 2024, Mapbox. |
| `@mapbox/tiny-sdf`, in `maplibre-gl` | 2.2.0 | BSD-2-Clause | Copyright (c) 2016-2024 Mapbox, Inc. |
| `@mapbox/unitbezier`, in `maplibre-gl` and its style spec | 0.0.1, 1.0.0 | BSD-2-Clause | Copyright (C) 2008 Apple Inc. All Rights Reserved. |
| `@mapbox/vector-tile`, in `maplibre-gl` | 2.0.5 | BSD-3-Clause | Copyright (c) 2024, Mapbox. |
| `@mapbox/whoots-js`, in `maplibre-gl` | 3.1.0 | ISC | Copyright (c) 2017, Mapbox. |
| `earcut`, in `maplibre-gl` | 3.2.3 | ISC | Copyright (c) 2026, Mapbox. |
| `gl-matrix`, in `maplibre-gl` | 3.4.4 | MIT | Copyright (c) 2015-2025, Brandon Jones, Colin MacKenzie IV. |
| `json-stringify-pretty-compact`, in the style spec | 4.0.0 | MIT | Copyright (c) 2014, 2016, 2017, 2019, 2021, 2022 Simon Lydell. |
| `kdbush`, in `maplibre-gl` | 4.1.0 | ISC | Copyright (c) 2026, Vladimir Agafonkin. |
| `murmurhash-js`, in `maplibre-gl` | 1.0.0 | MIT, stated in its `package.json` and README | Copyright (c) 2011 Gary Court. Implements Austin Appleby's MurmurHash. |
| `pbf`, in `maplibre-gl` and `@maplibre/vt-pbf` | 4.0.2, 5.1.2 | BSD-3-Clause | Copyright (c) 2024, Mapbox (4.0.2). Copyright (c) 2026, Mapbox (5.1.2). |
| `potpack`, in `maplibre-gl` | 2.1.0 | ISC | Copyright (c) 2025, Mapbox. |
| `quickselect`, in the style spec | 3.0.0 | ISC | Copyright (c) 2024, Vladimir Agafonkin. |
| `tinyqueue`, in `maplibre-gl` and its style spec | 3.0.0 | ISC | Copyright (c) 2017, Vladimir Agafonkin. |
| `tslib`, in `maplibre-gl` | Version of `maplibre-gl`'s build | 0BSD | Copyright (c) Microsoft Corporation. |
| `vite` (module-preload polyfill and preload helper) | 8.2.2 | MIT | Copyright (c) 2019-present, VoidZero Inc. and Vite contributors. |
| `rolldown` (module interop helpers) | 1.2.5 | MIT | Copyright (c) 2024-present VoidZero Inc. & Contributors. Its THIRD-PARTY-LICENSE adds Rollup's (Copyright (c) 2017 [these people](https://github.com/rollup/rollup/graphs/contributors)) and esbuild's (Copyright (c) 2020 Evan Wallace), both MIT. |

`@maplibre/mlt` is offered under MIT or Apache-2.0; the MIT text below
applies to it. The MIT, ISC, and 0BSD texts below apply with the copyright
lines of their packages in the table above in place of `<copyright>`. The
BSD-2-Clause and BSD-3-Clause texts follow verbatim from each package's
license file, since each names its own project and holder.

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

`maplibre-gl`, its `LICENSE.txt`. The style spec's `LICENSE.txt` is the same text with `Copyright (c) 2020, MapLibre contributors` as its first line:

```text
Copyright (c) 2023, MapLibre contributors

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of MapLibre GL JS nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


-------------------------------------------------------------------------------

Contains code from mapbox-gl-js v1.13 and earlier

Version v1.13 of mapbox-gl-js and earlier are licensed under a BSD-3-Clause license

Copyright (c) 2020, Mapbox
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of Mapbox GL JS nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


-------------------------------------------------------------------------------

Contains code from glfx.js

Copyright (C) 2011 by Evan Wallace

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

--------------------------------------------------------------------------------

Contains a portion of d3-color https://github.com/d3/d3-color

Copyright 2010-2016 Mike Bostock
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the author nor the names of contributors may be used to
  endorse or promote products derived from this software without specific prior
  written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

`@maplibre/vt-pbf`, the BSD-3-Clause part of its `LICENSE`:

```text
Contains geojson_wrapper.js from https://github.com/mapbox/mapbox-gl-js

Copyright (c) 2014, Mapbox

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of Mapbox GL JS nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

`@mapbox/tiny-sdf`, its `LICENSE.txt`:

```text
BSD-2-Clause
Copyright (c) 2016-2024 Mapbox, Inc.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

`@mapbox/unitbezier` 0.0.1 and 1.0.0, the same `LICENSE` text:

```text
BSD-2-Clause

Copyright (C) 2008 Apple Inc. All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY APPLE INC. ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL APPLE INC. OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Ported from Webkit
http://svn.webkit.org/repository/webkit/trunk/Source/WebCore/platform/graphics/UnitBezier.h
```

`@mapbox/vector-tile`, its `LICENSE.txt`:

```text
Copyright (c) 2024, Mapbox

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of Mapbox nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

`pbf` 4.0.2, its `LICENSE`. The 5.1.2 `LICENSE` is the same text with `Copyright (c) 2026, Mapbox` as its first line:

```text
Copyright (c) 2024, Mapbox
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of pbf nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

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

## Used and not redistributed

| Item | Where it is used | Source | License |
|---|---|---|---|
| SAM 3.1 checkpoint `facebook/sam3.1` | The SAM serve (`serving/`) fetches it on first load. | Meta, gated on [Hugging Face](https://huggingface.co/facebook/sam3.1). | SAM License, accepted on the model page. |
| Meta's `sam3` package at commit `2814fa6` | Installed into the `sam3_1` environment; `serving/sam_engine.py` imports it. | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | SAM License. |
| CityNav episodes, CityRefer catalog | Fetched by `benchmarks/citynav/data_prep/setup.sh` (`fetch_data.sh`) through upstream CityNav's own downloader. | [water-cookie/citynav-arxiv](https://github.com/water-cookie/citynav-arxiv) | Their upstream licenses. |
| SensatUrban point clouds | You request them from their authors; `benchmarks/citynav/data_prep/setup.sh --sensaturban <data_release.zip>` rasterizes them on your machine. | The authors' form at [QingyongHu/SensatUrban](https://github.com/QingyongHu/SensatUrban). | CC BY-NC-SA. For the terms, see [`docs/data.md`](docs/data.md#citynav). |
| OpenStreetMap | Read from the Cal City world bundle; baked into local tile directories. | [openstreetmap.org](https://www.openstreetmap.org) | © OpenStreetMap contributors, ODbL 1.0. |
| USGS 3DEP elevation (1/3 arc-second DEM) | The flight harness reads a DEM you resample from it ([`docs/data.md`](docs/data.md)). | The National Map, USGS. | Public USGS data. |

Section 1.b.v of the SAM License ("SAM License, Last Updated: November 19,
2025"), verbatim:

> v. You are not the target of Trade Controls and your use of SAM Materials must comply with Trade Controls. You agree not to use, or permit others to use, SAM Materials for any activities subject to the International Traffic in Arms Regulations (ITAR) or end uses prohibited by Trade Controls, including those related to military or warfare purposes, nuclear industries or applications, espionage, or the development or use of guns or illegal weapons.

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
