/** The station's instrument: the mission over the basemap.
 *
 * Uncovered ground reads as the paper no-data tone, never black: a black
 * map cannot be told apart from a dead one.
 *
 * It carries the aircraft (the ownship and the ground its camera covers),
 * where it has been (the live trail, or a reviewed run's whole track), the
 * waypoints of the mission's flights, the target footprint, and each leg's
 * area with what the passes hold of it.
 */

import { useEffect, useRef, useState } from "react";
import type {
  CircleLayerSpecification,
  FillLayerSpecification,
  GeoJSONSource,
  LineLayerSpecification,
  Map as MapLibreMap,
  Marker as MapLibreMarker,
  StyleSpecification,
} from "maplibre-gl";

import { coverageFeatures, coverageLabel, coverageMet, maskAnchor } from "@/app/coverage";
import type { AreaPass, WaitingLeg } from "@/app/coverage";
import { fetchLocation, type DisplayLocation } from "@/app/location";
import type { Flight, Ownship } from "@/app/mission";

const NO_DATA = "#d9d4c6";

// Where the aircraft has been: the aircraft's own magenta, thin.
const TRAIL_LINE = {
  "line-color": "#ee55ba",
  "line-width": 1.5,
  "line-opacity": 0.75,
} as const;

// The target a mission is aimed at: the footprint it is scored on, in the
// console's act colour, filled thin and outlined.
const TARGET_FILL = { "fill-color": "#c95d1e", "fill-opacity": 0.22 } as const;
const TARGET_LINE = { "line-color": "#c95d1e", "line-width": 2, "line-opacity": 0.95 } as const;

// The ground the camera covers right now: a light teal wash to read imagery
// through and a deep, heavy outline.
const SENSOR_FRAME_FILL = { "fill-color": "#2fb8c6", "fill-opacity": 0.18 } as const;
const SENSOR_FRAME_LINE = { "line-color": "#0b7d8e", "line-width": 3, "line-opacity": 1 } as const;

// The ground a leg asks for, and what the passes hold of it
// (app/coverage.ts): the hull washed in the sensor frame's teal once the
// floor is met and in the warn red while it is short, every sample point a
// disc, filled where a frame holds it and hollow where none does. Each
// feature carries its own `tint`, so an update recolours the mask through
// setData and nothing is rebuilt. A leg that has ended keeps its mask, faded.
const COVERAGE_FILL: FillLayerSpecification["paint"] = {
  "fill-color": ["get", "tint"],
  "fill-opacity": ["case", ["get", "faded"], 0.08, 0.2],
};
const COVERAGE_LINE: LineLayerSpecification["paint"] = {
  "line-color": ["get", "tint"],
  "line-width": 2,
  "line-opacity": ["case", ["get", "faded"], 0.4, 1],
};
const COVERAGE_POINT: CircleLayerSpecification["paint"] = {
  "circle-radius": 3.5,
  "circle-color": ["get", "tint"],
  "circle-opacity": ["case", ["!", ["get", "held"]], 0, ["get", "faded"], 0.4, 1],
  "circle-stroke-color": ["get", "tint"],
  "circle-stroke-width": 1.5,
  "circle-stroke-opacity": ["case", ["get", "faded"], 0.4, 1],
};

// The working area, a dashed box.
const AO_LINE = {
  "line-color": "#141310",
  "line-width": 1.5,
  "line-opacity": 0.8,
  "line-dasharray": [3, 2] as number[],
} as const;

interface LayerChip {
  key: "ownship" | "frame" | "trail" | "waypoints" | "coverage" | "ao";
  label: string;
  title: string;
}

const CHIPS: LayerChip[] = [
  { key: "ownship", label: "ownship", title: "where the aircraft is now" },
  { key: "frame", label: "frame", title: "the ground the camera covers right now" },
  { key: "trail", label: "trail", title: "where it has been: the live trail, or the whole track under review" },
  { key: "waypoints", label: "waypoints", title: "where each of the mission's flights ended, in flying order" },
  { key: "coverage", label: "coverage", title: "the ground each leg asks for, and what the passes hold of it" },
  { key: "ao", label: "ao", title: "the working area of the map" },
];

type Visible = Record<LayerChip["key"], boolean>;

const DEFAULT_VISIBLE: Visible = {
  ownship: true,
  frame: true,
  trail: true,
  waypoints: true,
  coverage: true,
  ao: true,
};

export function MapPane({ missionLabel, ownship, legs, flights, target, leg, passes }: {
  missionLabel: string | null;
  // Where the aircraft is. null before the first poll answers and with no
  // mission in the seat; a pose with no lon/lat draws nothing.
  ownship?: Ownship | null;
  // The whole track a run under review flew, one leg per unbroken stretch,
  // `[lon, lat]` oldest first. Empty live, where the trail is the ownship
  // document's own.
  legs?: [number, number][][];
  // The active mission's flights: each one's end is a numbered waypoint.
  flights?: Flight[];
  // The target footprint, `[lon, lat]` ring. null draws none.
  target?: [number, number][] | null;
  // The leg the station is waiting on, with its area and the share the
  // passes hold of it. null draws no live mask.
  leg?: WaitingLeg | null;
  // The legs the active mission has finished with (`coverage.flownPasses`):
  // drawn faded under the live mask.
  passes?: AreaPass[];
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const waypointPinsRef = useRef<MapLibreMarker[]>([]);
  const [area, setArea] = useState<number[] | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const ownshipMarkerRef = useRef<MapLibreMarker | null>(null);
  const targetMarkerRef = useRef<MapLibreMarker | null>(null);
  const coverageLabelRef = useRef<MapLibreMarker | null>(null);
  const [location, setLocation] = useState<DisplayLocation | null>(null);
  const [locationChecked, setLocationChecked] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);
  // Flips once the map instance exists. Every marker effect takes this in
  // its deps, so one that ran before the map's async init draws again.
  const [mapReady, setMapReady] = useState(false);
  // Flips once the style itself has loaded: addSource/addLayer throw
  // before that.
  const [styleReady, setStyleReady] = useState(false);
  const [visible, setVisible] = useState<Visible>(DEFAULT_VISIBLE);

  useEffect(() => {
    let cancelled = false;
    fetchLocation().then((loc) => {
      if (!cancelled) {
        setLocation(loc);
        setLocationChecked(true);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // The working area, off `GET area`. A station that answers none leaves
  // the layer off.
  useEffect(() => {
    let cancelled = false;
    fetch("area")
      .then((r) => (r.ok ? r.json() : null))
      .then((body) => {
        if (!cancelled) setArea(Array.isArray(body?.bbox) && body.bbox.length === 4 ? body.bbox : null);
      })
      .catch(() => {
        // No AO is a map with none declared, not a fault.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !styleReady || !area) return;
    const [west, south, east, north] = area;
    const data = {
      type: "Feature" as const,
      properties: {},
      geometry: {
        type: "LineString" as const,
        coordinates: [[west, south], [east, south], [east, north], [west, north], [west, south]],
      },
    };
    const source = map.getSource("ao") as GeoJSONSource | undefined;
    if (source) {
      source.setData(data);
    } else {
      map.addSource("ao", { type: "geojson", data });
      map.addLayer({ id: "ao-line", type: "line", source: "ao", paint: { ...AO_LINE } });
    }
    map.setLayoutProperty("ao-line", "visibility", visible.ao ? "visible" : "none");
  }, [area, styleReady, visible.ao]);

  // The ground the camera covers at the pose the ownship marker stands on,
  // cast to flat ground by the station. Drawn under the target so a scored
  // footprint is never hidden by it.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !styleReady) return;
    const corners = ownship?.frame_corners ?? null;
    const ring = corners && corners.length >= 3 ? [...corners, corners[0]] : null;
    const data = {
      type: "FeatureCollection" as const,
      features: ring
        ? [{ type: "Feature" as const, properties: {}, geometry: { type: "Polygon" as const, coordinates: [ring] } }]
        : [],
    };
    const source = map.getSource("frame") as GeoJSONSource | undefined;
    if (source) {
      source.setData(data);
    } else {
      map.addSource("frame", { type: "geojson", data });
      map.addLayer({ id: "frame-fill", type: "fill", source: "frame", paint: { ...SENSOR_FRAME_FILL } });
      map.addLayer({ id: "frame-line", type: "line", source: "frame", paint: { ...SENSOR_FRAME_LINE } });
    }
    for (const id of ["frame-fill", "frame-line"]) {
      if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", visible.frame ? "visible" : "none");
    }
  }, [ownship, styleReady, visible.frame]);

  // The leg's mask: the waiting leg live, every ended leg faded under it.
  // Drawn after the frame and before the target, so a scored footprint
  // reads over it. The label is ONE marker moved and reworded in place: a
  // marker rebuilt on each update blinks.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !styleReady) return;
    const data = {
      type: "FeatureCollection" as const,
      features: [
        ...(passes ?? []).flatMap((pass) => coverageFeatures(pass, true)),
        ...(leg ? coverageFeatures(leg) : []),
      ],
    };
    const source = map.getSource("coverage") as GeoJSONSource | undefined;
    if (source) {
      source.setData(data);
    } else {
      map.addSource("coverage", { type: "geojson", data });
      map.addLayer({ id: "coverage-fill", type: "fill", source: "coverage",
                     filter: ["==", ["get", "part"], "hull"], paint: COVERAGE_FILL });
      map.addLayer({ id: "coverage-line", type: "line", source: "coverage",
                     filter: ["==", ["get", "part"], "hull"], paint: COVERAGE_LINE });
      map.addLayer({ id: "coverage-point", type: "circle", source: "coverage",
                     filter: ["==", ["get", "part"], "point"], paint: COVERAGE_POINT });
    }
    for (const id of ["coverage-fill", "coverage-line", "coverage-point"]) {
      map.setLayoutProperty(id, "visibility", visible.coverage ? "visible" : "none");
    }
    if (!leg || !leg.area.length || !visible.coverage) {
      coverageLabelRef.current?.remove();
      coverageLabelRef.current = null;
      return;
    }
    const label = coverageLabel(leg.coverage, leg.coverageFloor);
    const state = coverageMet(leg.coverage, leg.coverageFloor) ? "held" : "short";
    const at = maskAnchor(leg.area);
    const standing = coverageLabelRef.current;
    if (standing) {
      standing.setLngLat(at);
      const el = standing.getElement();
      if (el.textContent !== label) el.textContent = label;
      el.dataset.state = state;
      return;
    }
    let cancelled = false;
    import("maplibre-gl").then((maplibre) => {
      if (cancelled || coverageLabelRef.current) return;
      const el = document.createElement("div");
      el.className = "coverage-pin";
      el.textContent = label;
      el.dataset.state = state;
      el.title = "the share of this leg's area the passes hold, against the floor the station arrives at";
      coverageLabelRef.current = new maplibre.Marker({ element: el, anchor: "bottom", offset: [0, -6] })
        .setLngLat(at)
        .addTo(map);
    });
    return () => {
      cancelled = true;
    };
  }, [leg, passes, styleReady, visible.coverage]);

  // The target footprint.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !styleReady) return;
    const ring = target && target.length >= 3 ? [...target, target[0]] : null;
    const data = {
      type: "FeatureCollection" as const,
      features: ring
        ? [{ type: "Feature" as const, properties: {}, geometry: { type: "Polygon" as const, coordinates: [ring] } }]
        : [],
    };
    const source = map.getSource("target") as GeoJSONSource | undefined;
    if (source) {
      source.setData(data);
    } else {
      map.addSource("target", { type: "geojson", data });
      map.addLayer({ id: "target-fill", type: "fill", source: "target", paint: { ...TARGET_FILL } });
      map.addLayer({ id: "target-line", type: "line", source: "target", paint: { ...TARGET_LINE } });
    }
    // The GT label at the footprint's centroid, an HTML marker: the raster
    // style carries no glyphs for a symbol layer, and a footprint a few
    // metres wide is a dot at the area's zoom, so the word is what the eye
    // finds.
    targetMarkerRef.current?.remove();
    targetMarkerRef.current = null;
    if (!ring) return;
    let cancelled = false;
    import("maplibre-gl").then((maplibre) => {
      if (cancelled) return;
      const n = target!.length;
      const lon = target!.reduce((a, q) => a + q[0], 0) / n;
      const lat = target!.reduce((a, q) => a + q[1], 0) / n;
      const el = document.createElement("div");
      el.className = "gt-pin";
      el.textContent = "GT";
      el.title = "the mission's ground truth: the footprint it is scored on";
      targetMarkerRef.current?.remove();
      targetMarkerRef.current = new maplibre.Marker({ element: el, anchor: "bottom", offset: [0, -4] })
        .setLngLat([lon, lat])
        .addTo(map);
    });
    return () => {
      cancelled = true;
    };
  }, [target, styleReady]);

  // The waypoints: where each of the mission's flights ended, numbered in
  // flying order.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    let cancelled = false;
    import("maplibre-gl").then((maplibre) => {
      if (cancelled) return;
      waypointPinsRef.current.forEach((marker) => marker.remove());
      waypointPinsRef.current = [];
      if (!visible.waypoints) return;
      (flights ?? []).forEach((f, n) => {
        if (f.lon == null || f.lat == null) return;
        const el = document.createElement("div");
        el.className = "route-pin";
        const number = document.createElement("span");
        number.textContent = String(n + 1);
        el.append(number);
        el.title = `flight ${n + 1}: ${f.target ?? f.stop_id ?? ""}`;
        waypointPinsRef.current.push(
          new maplibre.Marker({ element: el }).setLngLat([f.lon, f.lat]).addTo(map),
        );
      });
    });
    return () => {
      cancelled = true;
    };
  }, [flights, mapReady, visible.waypoints]);

  // Init once the container exists and `location` has answered (even with
  // null: the map still mounts, on the no-data background).
  useEffect(() => {
    if (!locationChecked || containerRef.current === null || mapRef.current) return;
    let cancelled = false;
    import("maplibre-gl").then((maplibre) => {
      if (cancelled || containerRef.current === null) return;
      const map = new maplibre.Map({
        container: containerRef.current,
        style: rasterStyle(location),
        center: location ? [location.center.lon, location.center.lat] : [0, 0],
        zoom: location ? 13 : 1,
        attributionControl: false,
      });
      map.on("error", (event) => setMapError(String(event.error?.message ?? event)));
      map.on("load", () => setStyleReady(true));
      map.on("style.load", () => setStyleReady(true));
      mapRef.current = map;
      setMapReady(true);
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [locationChecked]);

  useEffect(() => () => {
    waypointPinsRef.current.forEach((marker) => marker.remove());
    ownshipMarkerRef.current?.remove();
    targetMarkerRef.current?.remove();
    coverageLabelRef.current?.remove();
    mapRef.current?.remove();
  }, []);

  // MapLibre caches its canvas size and cannot notice a CSS-driven
  // container change on its own. The divider drag resizes this container
  // without re-rendering MapPane, so a ResizeObserver is the signal.
  useEffect(() => {
    if (containerRef.current === null) return;
    const observer = new ResizeObserver(() => mapRef.current?.resize());
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // The aircraft. A DOM marker rather than a symbol layer: no glyph atlas
  // to serve, and MapLibre rotates the element for free.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    let cancelled = false;
    import("maplibre-gl").then((maplibre) => {
      if (cancelled) return;
      ownshipMarkerRef.current?.remove();
      ownshipMarkerRef.current = null;
      if (!visible.ownship || !ownship
          || typeof ownship.lon !== "number" || typeof ownship.lat !== "number") {
        return;
      }
      const el = document.createElement("div");
      el.className = "ownship";
      el.title = "the aircraft";
      ownshipMarkerRef.current = new maplibre.Marker({
        element: el,
        // The marker art points north; MapLibre turns it to the track, so
        // the aircraft on the map is pointed where it is actually going.
        rotation: ownship.track_deg,
        rotationAlignment: "map",
      })
        .setLngLat([ownship.lon, ownship.lat])
        .addTo(map);
    });
    return () => {
      cancelled = true;
    };
  }, [ownship, mapReady, visible.ownship]);

  // The trail: where the aircraft has been. Live that is the ownship
  // document's own trail; under review it is the run's whole track, drawn
  // at once so the scrub position is a place in a flight. A gap in the
  // record is a break in the line.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !styleReady) return;
    const drawn = legs?.length ? legs : [ownship?.trail ?? []].filter((l) => l.length > 1);
    const data = {
      type: "Feature" as const,
      properties: {},
      geometry: { type: "MultiLineString" as const, coordinates: drawn },
    };
    const source = map.getSource("trail") as GeoJSONSource | undefined;
    if (source) {
      source.setData(data);
    } else if (drawn.length) {
      map.addSource("trail", { type: "geojson", data });
      map.addLayer({ id: "trail-line", type: "line", source: "trail", paint: { ...TRAIL_LINE } });
    }
    if (map.getLayer("trail-line")) {
      map.setLayoutProperty("trail-line", "visibility",
                            visible.trail && drawn.length ? "visible" : "none");
    }
  }, [ownship, legs, styleReady, visible.trail]);

  // What the map frames: the reviewed run's whole track, or the basemap's
  // bounds when nothing is under review. Keyed on the track, so the
  // operator's own pan and zoom survive every seek in between.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    const points = legs?.flat() ?? [];
    const box: [number, number, number, number] | null = points.length
      ? [Math.min(...points.map(([lon]) => lon)), Math.min(...points.map(([, lat]) => lat)),
         Math.max(...points.map(([lon]) => lon)), Math.max(...points.map(([, lat]) => lat))]
      : (location?.bounds as [number, number, number, number] | undefined) ?? null;
    if (!box) return;
    map.fitBounds([[box[0], box[1]], [box[2], box[3]]], { padding: 48, animate: false });
  }, [legs, mapReady, location]);

  return (
    <div className="map-pane">
      <div ref={containerRef} className="map-canvas" />
      <div className="layer-bar">
        {CHIPS.map((chip) => (
          <button
            key={chip.key}
            type="button"
            className={visible[chip.key] ? "layer-chip on" : "layer-chip"}
            title={chip.title}
            onClick={() => setVisible((v) => ({ ...v, [chip.key]: !v[chip.key] }))}
          >
            {chip.label}
          </button>
        ))}
      </div>
      <div className="map-caption">
        {mapError
          ? `MAP ENGINE UNAVAILABLE · ${mapError}`
          : location
            ? location.label.toUpperCase()
            : "NO LOCAL IMAGERY STAGED"}
        {missionLabel ? ` · ${missionLabel}` : ""}
      </div>
    </div>
  );
}

/** The manifest's tile template as an absolute URL: MapLibre resolves
 *  nothing against the page, and the console may be mounted under a path. */
function tileUrl(template: string): string {
  return /^[a-z][a-z0-9+.-]*:/i.test(template)
    ? template
    : new URL(".", document.baseURI).href + template.replace(/^\//, "");
}

function rasterStyle(location: DisplayLocation | null): StyleSpecification {
  const sources = location
    ? {
        basemap: {
          type: "raster" as const,
          tiles: [tileUrl(location.tiles)],
          tileSize: 256,
          attribution: location.attribution,
          minzoom: location.minzoom,
          maxzoom: location.maxzoom,
          bounds: location.bounds,
        },
      }
    : {};
  return {
    version: 8,
    sources,
    layers: [
      { id: "no-data", type: "background",
        paint: { "background-color": NO_DATA } },
      ...(location
        ? [{
            id: "basemap",
            type: "raster" as const,
            source: "basemap",
            paint: {
              "raster-saturation": 0,
              "raster-contrast": 0,
              "raster-brightness-max": 1,
            },
          }]
        : []),
    ],
  } as StyleSpecification;
}
