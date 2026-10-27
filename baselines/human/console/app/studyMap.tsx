/** The capacity console's map (baselines/human/study_server.py).
 *
 * The bench's collected ortho and nothing else, as web-mercator tiles cut from it
 * (study_package.py), the paper no-data tone where it holds nothing. Over it the briefed
 * AO and the bench's OSM prior (the one MAPS reads) behind a toggle, with a search over
 * its named places. Each report opens framed on the AO.
 *
 * The style carries every layer either mode draws; a mode fills the ones it
 * uses and leaves the rest empty.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { GeoJSONSource, Map as MapLibreMap, Marker as MapLibreMarker, StyleSpecification } from "maplibre-gl";

import { boundsOf } from "@/app/fed";
import { kindMatches } from "@/app/kindWords";
import { alongStreet, axisAt, buildingsAlong, closestApproach, crossings, metres, nearestOther, type LonLat } from "@/app/studyGeometry";

export { crossings, metres };

export type { LonLat };
export type Ring = LonLat[] | null;
type Bbox = [number, number, number, number];
export interface PriorName { name: string; kind: string; lonlat: LonLat }
/** What the outline under the pointer is, and where the pointer is on the map. */
export interface Hover { text: string; x: number; y: number; left: boolean }

export const INK = { act: "#c95d1e", cite: "#1a4fa0", good: "#2e6b34", ink: "#141310", paper: "#fbfaf5" };
/** The zoom-out floor, when the server names none. */
const MIN_ZOOM = 12;
/** Below this zoom labels would only pile up: the container reads `is-far`. */
export const LABEL_MIN_ZOOM = 16;
/** The OSM prior's names drawn on the map at most, and from this zoom; one name is drawn
 * again only this many pixels from where it already is, so a long street reads along its length. */
const OSM_LABEL_CAP = 80;
const OSM_LABEL_MIN_ZOOM = 15;
const OSM_LABEL_REPEAT_PX = 260;
/** Where the browser keeps the street map's on or off. */
const OSM_KEY = "maps-study-street-map";
/** Where a found name or crossing opens: houses and the streets between them both read. */
const FIND_ZOOM = 17.5;
/** The closest a selected step's inputs open: a house and its lot. */
const FED_ZOOM = 19;
/** A wheel notch's zoom: MapLibre's default (1/450) takes forty notches from a house to a street grid. */
const WHEEL_ZOOM_RATE = 1 / 180;

export const fc = (features: GeoJSON.Feature[]): GeoJSON.FeatureCollection => ({ type: "FeatureCollection", features });
export const ringFeature = (ring: LonLat[], props: Record<string, unknown>): GeoJSON.Feature => ({
  type: "Feature", properties: props, geometry: { type: "Polygon", coordinates: [[...ring, ring[0]]] },
});
export const pointFeature = (at: LonLat, props: Record<string, unknown>): GeoJSON.Feature => ({
  type: "Feature", properties: props, geometry: { type: "Point", coordinates: at },
});

/** The paper's figure palette (MAPS-SciRob figures/style.py), for the operator console
 * whose screenshot the capacity figure prints. No orange: on imagery MAPS's commit is the
 * figures' focal mark, white over a dark halo; the person's pick is navy over a white one. */
export const PAPER = { navy: "#1F3A5F", teal: "#2F6870", sky: "#8FB0D3",
                       sand: "#E9DCC3", grey: "#6B6259", ink: "#1E1B18", white: "#FFFFFF",
                       halo: "rgba(30,27,24,0.55)" };
/** The places MAPS's reasoning names, one colour each in the order it names them: bright inks,
 * none the figure's own, each over a dark halo so it reads on imagery and street map alike. */
export const SAID = ["#ff4fd8", "#ffd60a", "#22e4ff", "#ff8c1a", "#a6ff00", "#b388ff", "#ff5c5c", "#3dffb0"];
/** What a selected step of MAPS's search used (fed.ts): the street map's rows it names, its
 * detections and its points, each over the same dark halo. */
export const FED = { prior: "#4d7dff", detected: "#a6ff00", point: "#ffd60a" };

/** The paper theme's marks, Fig. 1's: rings with a white halo, thin sky outlines, no text on the imagery. */
function paperLayers(layers: object[]): object[] {
  const ring = (id: string, source: string, colour: string, halo: string, width: number) => [
    { id: `${id}-halo`, type: "circle", source, filter: ["==", ["geometry-type"], "Point"],
      paint: { "circle-radius": 9, "circle-color": "rgba(0,0,0,0)", "circle-stroke-color": halo, "circle-stroke-width": width + 3 } },
    { id, type: "circle", source, filter: ["==", ["geometry-type"], "Point"],
      paint: { "circle-radius": 9, "circle-color": "rgba(0,0,0,0)", "circle-stroke-color": colour, "circle-stroke-width": width } },
  ];
  return layers.flatMap((l) => {
    const id = (l as { id: string }).id;
    if (id === "maps-dot") return ring("maps-dot", "maps", PAPER.white, PAPER.halo, 2);
    if (id === "sel-dot") return ring("sel-dot", "sel", PAPER.navy, PAPER.white, 2.5);
    if (id === "maps-line") {
      const area = ["==", ["get", "area"], true];
      const dash = ["case", area, ["literal", [3, 2]], ["literal", [1, 0]]];
      return [
        { ...l, id: "maps-line-halo", paint: { "line-color": PAPER.halo, "line-width": ["case", area, 2.7, 3.5], "line-dasharray": dash } },
        { ...l, paint: { "line-color": PAPER.white, "line-width": ["case", area, 1.2, 2], "line-dasharray": dash } },
      ];
    }
    if (id === "sel-line") return [{ ...l, paint: { "line-color": PAPER.navy, "line-width": 2.5 } }];
    if (id === "cands-line") return [{ ...l, paint: { "line-color": PAPER.sky, "line-width": 1.2, "line-opacity": 0.7 } }];
    if (id === "cands-fill") return [];
    if (id === "stops-dot") return [{ ...l, paint: { "circle-radius": 3.5, "circle-color": PAPER.navy, "circle-stroke-color": PAPER.white, "circle-stroke-width": 1.5 } }];
    if (id === "stops-line") return [{ ...l, paint: { "line-color": PAPER.white, "line-width": 1.2, "line-dasharray": [2, 2], "line-opacity": 0.8 } }];
    if (id === "osm-lit") return [{ ...l, paint: { "line-color": PAPER.teal, "line-width": 4, "line-opacity": 0.95 } }];
    // A search's result, a filled teal disc: never the person's point, a hollow navy ring.
    if (id === "mark-ring") return [{ ...l, paint: { "circle-radius": 13, "circle-color": "rgba(47,104,112,0.6)", "circle-stroke-color": PAPER.white, "circle-stroke-width": 2 } }];
    if (id === "osm-line") return [{ ...l, paint: { ...(l as { paint: object }).paint, "line-color": ["case", ["==", ["get", "layer"], "road"], PAPER.sand, "#f4eee1"] } }];
    return [l];
  });
}

function studyStyle(): StyleSpecification {
  const style = baseStudyStyle();
  return { ...style, layers: paperLayers(style.layers) as StyleSpecification["layers"] };
}

function baseStudyStyle(): StyleSpecification {
  return {
    version: 8,
    sources: {
      ortho: { type: "raster", tiles: [new URL("study/tiles/{z}/{x}/{y}.webp", window.location.href).href.replace(/%7B/g, "{").replace(/%7D/g, "}")],
               tileSize: 256, minzoom: 12, maxzoom: 20 },
      ao: { type: "geojson", data: fc([]) },
      osm: { type: "geojson", data: fc([]) },
      "osm-area": { type: "geojson", data: fc([]) },
      cands: { type: "geojson", data: fc([]) },
      maps: { type: "geojson", data: fc([]) },
      stops: { type: "geojson", data: fc([]) },
      sel: { type: "geojson", data: fc([]) },
      mark: { type: "geojson", data: fc([]) },
      said: { type: "geojson", data: fc([]) },
      fed: { type: "geojson", data: fc([]) },
      ruler: { type: "geojson", data: fc([]) },
    },
    layers: [
      { id: "no-data", type: "background", paint: { "background-color": "#d9d4c6" } },
      { id: "ortho", type: "raster", source: "ortho", paint: { "raster-fade-duration": 0 } },
      { id: "ao-line", type: "line", source: "ao",
        paint: { "line-color": INK.ink, "line-width": 1.5, "line-opacity": 0.8, "line-dasharray": [3, 2] } },
      // Closed outlines as invisible areas: what hover reads inside a building or a park.
      { id: "osm-area", type: "fill", source: "osm-area", paint: { "fill-color": "#000", "fill-opacity": 0 } },
      // Roads solid yellow; everything else (buildings, landuse, lots) a thinner, paler line, so
      // which lines are streets reads at a glance.
      { id: "osm-line", type: "line", source: "osm", layout: { visibility: "none" },
        paint: { "line-color": ["case", ["==", ["get", "layer"], "road"], "#ffd600", "#fff3a8"],
                 "line-opacity": ["interpolate", ["linear"], ["zoom"], 13, 0.45, 16, 0.85],
                 "line-width": ["interpolate", ["linear"], ["zoom"], 13, ["case", ["get", "lit"], 3, 0.4],
                                17, ["case", ["get", "lit"], 3.5, 1.2]] } },
      // The places MAPS's reasoning names, in their colours; the one pointed at in the text heavier.
      { id: "said-halo", type: "line", source: "said", layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": PAPER.halo, "line-width": ["case", ["get", "hot"], 10, 5.5] } },
      { id: "said-line", type: "line", source: "said", layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": ["get", "colour"], "line-width": ["case", ["get", "hot"], 6, 2.5] } },
      // What the selected step used: its rows and detections outlined, its points ringed.
      { id: "fed-halo", type: "line", source: "fed", filter: ["==", ["geometry-type"], "LineString"],
        layout: { "line-cap": "round", "line-join": "round" }, paint: { "line-color": PAPER.halo, "line-width": 6.5 } },
      { id: "fed-line", type: "line", source: "fed", filter: ["==", ["geometry-type"], "LineString"],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": ["match", ["get", "fed"], "prior", FED.prior, FED.detected], "line-width": 3 } },
      { id: "fed-dot", type: "circle", source: "fed", filter: ["==", ["geometry-type"], "Point"],
        paint: { "circle-radius": 7, "circle-color": "rgba(0,0,0,0)", "circle-stroke-width": 2.5,
                 "circle-stroke-color": ["match", ["get", "fed"], "detected", FED.detected, FED.point] } },
      { id: "stops-line", type: "line", source: "stops", filter: ["==", ["geometry-type"], "LineString"],
        paint: { "line-color": INK.paper, "line-width": 1.5, "line-dasharray": [2, 2], "line-opacity": 0.9 } },
      { id: "stops-dot", type: "circle", source: "stops", filter: ["==", ["geometry-type"], "Point"],
        paint: { "circle-radius": 4, "circle-color": INK.ink, "circle-stroke-color": INK.paper, "circle-stroke-width": 1.5 } },
      { id: "cands-fill", type: "fill", source: "cands", paint: { "fill-color": INK.cite, "fill-opacity": 0.06 } },
      { id: "cands-line", type: "line", source: "cands", paint: { "line-color": "#5fb4ff", "line-width": 1.6 } },
      // MAPS's object heavy; the mapped area its point merely falls in (a landuse, a lot) thin and dashed.
      { id: "maps-line", type: "line", source: "maps", filter: ["==", ["geometry-type"], "Polygon"],
        paint: { "line-color": INK.act, "line-width": ["case", ["==", ["get", "area"], true], 1.5, 3.5],
                 "line-dasharray": ["case", ["==", ["get", "area"], true], ["literal", [3, 2]], ["literal", [1, 0]]] } },
      { id: "maps-dot", type: "circle", source: "maps", filter: ["==", ["geometry-type"], "Point"],
        paint: { "circle-radius": 6, "circle-color": INK.act, "circle-stroke-color": INK.paper, "circle-stroke-width": 2 } },
      { id: "sel-line", type: "line", source: "sel", filter: ["==", ["geometry-type"], "Polygon"],
        paint: { "line-color": "#3cff3c", "line-width": 3.5 } },
      { id: "sel-dot", type: "circle", source: "sel", filter: ["==", ["geometry-type"], "Point"],
        paint: { "circle-radius": 6, "circle-color": INK.good, "circle-stroke-color": INK.paper, "circle-stroke-width": 2 } },
      // A searched name's pieces, in their own ink, over every other line and on with the prior off.
      { id: "osm-lit", type: "line", source: "osm", filter: ["==", ["get", "lit"], true],
        paint: { "line-color": "#ff4fd8", "line-width": 4, "line-opacity": 0.95 } },
      { id: "mark-ring", type: "circle", source: "mark",
        paint: { "circle-radius": 13, "circle-color": "rgba(0,0,0,0)", "circle-stroke-color": "#ffd600", "circle-stroke-width": 3 } },
      { id: "ruler-line", type: "line", source: "ruler", filter: ["==", ["geometry-type"], "LineString"],
        paint: { "line-color": INK.paper, "line-width": 2.5, "line-dasharray": [1.5, 1] } },
      { id: "ruler-dot", type: "circle", source: "ruler", filter: ["==", ["geometry-type"], "Point"],
        paint: { "circle-radius": 4, "circle-color": INK.paper, "circle-stroke-color": INK.ink, "circle-stroke-width": 1.5 } },
    ],
  } as StyleSpecification;
}

/** Ask a route the server builds at startup until it answers with more than "building". */
function whenBuilt<T>(path: string, then: (body: T) => void): void {
  fetch(path).then((r) => (r.ok ? r.json() : null)).then((body) => {
    if (body && !body.building) then(body as T);
    else window.setTimeout(() => whenBuilt(path, then), 3000);
  }).catch(() => window.setTimeout(() => whenBuilt(path, then), 3000));
}

export function useStudyMap(onClick: (at: LonLat) => void) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const maplibreRef = useRef<typeof import("maplibre-gl") | null>(null);
  const aoRef = useRef<Bbox | null>(null);
  const clickRef = useRef(onClick);
  clickRef.current = onClick;
  const osmMarkers = useRef<MapLibreMarker[]>([]);
  const osmData = useRef<GeoJSON.FeatureCollection | null>(null);
  /** The street map's rows by the id MAPS knew each by (``prior_2777``). */
  const priorIndex = useRef(new Map<string, GeoJSON.Feature>());
  const [ready, setReady] = useState(false);
  const [names, setNames] = useState<PriorName[]>([]);
  // The street map stays as the person left it, from one session to the next.
  const [osmOn, setOsmOn] = useState(() => { try { return localStorage.getItem(OSM_KEY) === "on"; } catch { return false; } });
  useEffect(() => { try { localStorage.setItem(OSM_KEY, osmOn ? "on" : "off"); } catch { /* storage blocked: the default next time */ } }, [osmOn]);
  const osmOnRef = useRef(osmOn);
  osmOnRef.current = osmOn;
  const [lit, setLit] = useState<string | null>(null);
  const litRef = useRef(lit);
  litRef.current = lit;
  const [hover, setHover] = useState<Hover | null>(null);
  const [ruler, setRuler] = useState<LonLat[]>([]);

  // The OSM prior's names in view, the lit one always, while the layer is on.
  const refreshOsm = useCallback(() => {
    const map = mapRef.current;
    const maplibre = maplibreRef.current;
    osmMarkers.current.forEach((m) => m.remove());
    osmMarkers.current = [];
    if (!map || !maplibre) return;
    map.setLayoutProperty("osm-line", "visibility", osmOnRef.current ? "visible" : "none");
    if (osmData.current) {
      const lit = litRef.current;
      (map.getSource("osm") as GeoJSONSource).setData(lit ? {
        ...osmData.current,
        features: osmData.current.features.map((f) => (f.properties?.name === lit ? { ...f, properties: { ...f.properties, lit: true } } : f)),
      } : osmData.current);
    }
    const label = (text: string, at: LonLat) => {
      const el = document.createElement("span");
      el.className = "judge-label paper-name";
      el.textContent = text;
      osmMarkers.current.push(new maplibre.Marker({ element: el, anchor: "left", offset: [6, 0] }).setLngLat(at).addTo(map));
    };
    if (!osmOnRef.current || map.getZoom() < OSM_LABEL_MIN_ZOOM || !osmData.current) return;
    // Every named row in view, at a vertex of it inside the view: a street broken into
    // pieces is named on each piece, and a long one again along its length.
    const bounds = map.getBounds();
    const placed: Record<string, { x: number; y: number }[]> = {};
    const litFirst = [...osmData.current.features].sort((a, b) =>
      Number(b.properties?.name === litRef.current) - Number(a.properties?.name === litRef.current));
    for (const f of litFirst) {
      if (osmMarkers.current.length >= OSM_LABEL_CAP) break;
      const name = String(f.properties?.name ?? "");
      if (!name) continue;
      // At the middle of the longest segment in view, never at a vertex: a street's vertices
      // include every point where another street meets it, and a name set there reads as the
      // crossing street's.
      const line = (f.geometry as GeoJSON.LineString).coordinates as LonLat[];
      let at: LonLat | null = null;
      let longest = 0;
      for (let i = 1; i < line.length; i++) {
        const mid: LonLat = [(line[i - 1][0] + line[i][0]) / 2, (line[i - 1][1] + line[i][1]) / 2];
        const len = metres(line[i - 1], line[i]);
        if (bounds.contains(mid) && len > longest) { longest = len; at = mid; }
      }
      if (!at) continue;
      const px = map.project(at);
      const near = (placed[name] ??= []);
      if (near.some((q) => Math.hypot(q.x - px.x, q.y - px.y) < OSM_LABEL_REPEAT_PX)) continue;
      near.push({ x: px.x, y: px.y });
      label(name, at);
    }
  }, []);

  useEffect(() => { refreshOsm(); }, [osmOn, lit, names, ready, refreshOsm]);

  useEffect(() => {
    if (containerRef.current === null || mapRef.current) return;
    let cancelled = false;
    const located = fetch("location").then((r) => (r.ok ? r.json() : null)).catch(() => null) as
      Promise<{ center: { lat: number; lon: number }; minzoom?: number } | null>;
    Promise.all([import("maplibre-gl"), located]).then(([maplibre, location]) => {
      if (cancelled || containerRef.current === null) return;
      maplibreRef.current = maplibre;
      const map = new maplibre.Map({ container: containerRef.current, style: studyStyle(), attributionControl: false,
                                     center: location ? [location.center.lon, location.center.lat] : [0, 0], zoom: 13,
                                     minZoom: location?.minzoom ?? MIN_ZOOM, maxZoom: 21, dragRotate: false, pitchWithRotate: false });
      map.addControl(new maplibre.ScaleControl({ unit: "metric" }), "bottom-left");
      map.scrollZoom.setWheelZoomRate(WHEEL_ZOOM_RATE);
      // Shift is the ruler's: MapLibre's shift-drag box zoom would swallow the click.
      map.boxZoom.disable();
      // The panel's size is the layout's, settled after the map mounts and changed with the
      // window: a map that never hears of it draws in a corner of its panel.
      const sized = new ResizeObserver(() => map.resize());
      sized.observe(containerRef.current);
      // What an OSM line under the pointer is, while the layer is on.
      // A road also says which way it runs there, for "the south side" of a diagonal street.
      map.on("mousemove", (e) => {
        if (!osmOnRef.current) return;
        // A road under the pointer first; else the smallest outline the pointer is inside.
        const roads = map.queryRenderedFeatures([[e.point.x - 10, e.point.y - 10], [e.point.x + 10, e.point.y + 10]], { layers: ["osm-line"] })
          .filter((f) => f.properties?.layer === "road");
        const areas = map.queryRenderedFeatures(e.point, { layers: ["osm-area"] });
        const size = (f: GeoJSON.Feature) => {
          const c = (f.geometry as GeoJSON.Polygon).coordinates[0];
          const xs = c.map((p) => p[0]); const ys = c.map((p) => p[1]);
          return (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
        };
        const hit = roads[0] ?? areas.sort((a, b) => size(a) - size(b))[0];
        if (!hit) { setHover(null); return; }
        const coords = hit.geometry.type === "LineString" ? (hit.geometry.coordinates as LonLat[]) : null;
        const road = hit.properties?.layer === "road";
        const axis = road && coords && coords.length > 1 ? `runs ${axisAt(coords, [e.lngLat.lng, e.lngLat.lat])}` : "";
        const text = [hit.properties?.name, hit.properties?.kind, axis].filter(Boolean).join(" · ");
        setHover(text ? { text, x: e.point.x, y: e.point.y, left: e.point.x > map.getCanvas().clientWidth / 2 } : null);
      });
      map.getCanvas().addEventListener("mouseleave", () => setHover(null));
      // Shift-click lays a ruler; a plain click is the mode's own.
      map.on("click", (e) => {
        const at: LonLat = [e.lngLat.lng, e.lngLat.lat];
        if (e.originalEvent.shiftKey) setRuler((r) => [...r, at]);
        else clickRef.current(at);
      });
      const far = () => containerRef.current?.classList.toggle("is-far", map.getZoom() < LABEL_MIN_ZOOM);
      map.on("moveend", far);
      far();
      map.on("load", () => {
        // The briefed AO, off the server's /area; none declared is no box.
        fetch("area").then((r) => (r.ok ? r.json() : null)).then((body) => {
          const b = Array.isArray(body?.bbox) && body.bbox.length === 4 ? (body.bbox as Bbox) : null;
          aoRef.current = b;
          if (b) {
            const [w, s, e, n] = b;
            (map.getSource("ao") as GeoJSONSource).setData(fc([{ type: "Feature", properties: {},
              geometry: { type: "LineString", coordinates: [[w, s], [e, s], [e, n], [w, n], [w, s]] } }]));
          }
        }).catch(() => undefined).finally(() => { mapRef.current = map; setReady(true); });
        whenBuilt<GeoJSON.FeatureCollection & { names: PriorName[] }>("study/prior.json", (p) => {
          osmData.current = { type: "FeatureCollection", features: p.features };
          priorIndex.current = new Map(p.features.map((f) => [String(f.properties?.fid ?? ""), f]));
          (map.getSource("osm") as GeoJSONSource).setData(osmData.current);
          (map.getSource("osm-area") as GeoJSONSource).setData(fc(p.features.flatMap((f) => {
            const c = (f.geometry as GeoJSON.LineString).coordinates;
            const closed = c.length > 3 && c[0][0] === c[c.length - 1][0] && c[0][1] === c[c.length - 1][1];
            return closed && f.properties?.layer !== "road"
              ? [{ type: "Feature", properties: f.properties, geometry: { type: "Polygon", coordinates: [c] } } as GeoJSON.Feature] : [];
          })));
          setNames(p.names);
        });
      });
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const onMove = () => refreshOsm();
    map.on("moveend", onMove);
    return () => { map.off("moveend", onMove); };
  }, [ready, refreshOsm]);

  const flyTo = useCallback((at: LonLat, zoom?: number) => {
    const map = mapRef.current;
    map?.easeTo({ center: at, zoom: zoom ?? Math.max(map.getZoom(), 18), duration: 400 });
  }, []);

  /** Frame the briefed AO; ``fallback`` where none is declared. */
  const frameAo = useCallback((fallback?: LonLat, zoom = 18.5) => {
    const map = mapRef.current;
    if (!map) return;
    const ao = aoRef.current;
    if (ao) map.fitBounds([[ao[0], ao[1]], [ao[2], ao[3]]], { padding: 48, animate: false });
    else if (fallback) map.jumpTo({ center: fallback, zoom });
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    (map.getSource("ruler") as GeoJSONSource | undefined)?.setData(fc([
      ...(ruler.length > 1 ? [{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: ruler } } as GeoJSON.Feature] : []),
      ...ruler.map((p) => pointFeature(p, {})),
    ]));
  }, [ruler, ready]);
  const rulerM = ruler.slice(1).reduce((sum, p, i) => sum + metres(ruler[i], p), 0);

  /** Ring every crossing a search found, numbered as the list numbers them, and fly to one; the
   * last search's street goes dark. */
  const markMarkers = useRef<MapLibreMarker[]>([]);
  const markSpots = useCallback((all: LonLat[], focus: number) => {
    const map = mapRef.current;
    const maplibre = maplibreRef.current;
    setLit(null);
    setLitPiece(null);
    (map?.getSource("mark") as GeoJSONSource | undefined)?.setData(fc(all.map((at) => pointFeature(at, {}))));
    markMarkers.current.forEach((mk) => mk.remove());
    markMarkers.current = [];
    if (map && maplibre && all.length > 1) all.forEach((at, i) => {
      const el = document.createElement("span");
      el.className = "judge-label paper-name";
      el.textContent = String(i + 1);
      markMarkers.current.push(new maplibre.Marker({ element: el, anchor: "left", offset: [14, 0] }).setLngLat(at).addTo(map));
    });
    if (all[focus]) map?.easeTo({ center: all[focus], zoom: FIND_ZOOM, duration: 400 });
  }, []);
  /** Forget the last search on the map: its rings, its numbers and the lit name. */
  const clearFound = useCallback(() => {
    (mapRef.current?.getSource("mark") as GeoJSONSource | undefined)?.setData(fc([]));
    markMarkers.current.forEach((mk) => mk.remove());
    markMarkers.current = [];
    setLit(null);
    setLitPiece(null);
    setHover(null);
  }, []);
  /** Features whose kind matches ``q`` (parking, pool, a lake is water...), nearest ``street``'s
   * pieces when one is named, else nearest the view; each with the street it is nearest. */
  const kindsNear = useCallback((q: string, street?: string) => {
    const map = mapRef.current;
    if (!map || q.length < 3) return [];
    const feats = osmData.current?.features ?? [];
    const along = street ? feats.filter((f) => f.properties?.name === street).flatMap((f) => (f.geometry as GeoJSON.LineString).coordinates as LonLat[]) : [];
    const c = map.getCenter();
    const here: LonLat = [c.lng, c.lat];
    const away = (at: LonLat) => (along.length ? Math.min(...along.map((p) => metres(p, at))) : metres(here, at));
    return feats
      .filter((f) => kindMatches(String(f.properties?.kind ?? ""), q))
      .map((f) => {
        const pts = (f.geometry as GeoJSON.LineString).coordinates as LonLat[];
        const at: LonLat = [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length];
        return { name: String(f.properties?.name ?? ""), kind: String(f.properties?.kind ?? ""), at, m: away(at) };
      })
      .sort((a, b) => a.m - b.m).slice(0, 15)
      .map((k) => ({ ...k, near: nearestOther(feats, k.at, [], 120) }));
  }, []);
  const approachOf = useCallback((a: string, b: string) => closestApproach(osmData.current?.features ?? [], a, b), []);
  const nearOf = useCallback((at: LonLat, exclude: string[]) => nearestOther(osmData.current?.features ?? [], at, exclude), []);
  const crossingsOf = useCallback((a: string, b: string) => crossings(osmData.current?.features ?? [], a, b), []);
  const piecesOf = useCallback((name: string) => (osmData.current?.features ?? []).filter((f) => f.properties?.name === name).length, []);

  /** Light every piece of a named place and frame them all: a street the prior carries in
   * seven pieces is seven pieces on the map, not one point at their mean. */
  const buildingPoints = () => (osmData.current?.features ?? []).filter((f) => f.properties?.layer === "building")
    .map((f) => (f.geometry as GeoJSON.LineString).coordinates[0] as LonLat);
  const [litPiece, setLitPiece] = useState<{ n: number; of: number; buildings: number } | null>(null);
  const frame = (map: MapLibreMap, pts: LonLat[]) => {
    const lons = pts.map((p) => p[0]);
    const lats = pts.map((p) => p[1]);
    map.fitBounds([[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]],
                  { padding: 80, maxZoom: FIND_ZOOM, duration: 400 });
  };
  /** Step through a lit name's pieces end to end; ``skipEmpty`` passes over pieces with no
   * building along them (a paper road through empty lots holds no house to find). */
  const stepPiece = useCallback((by: number, skipEmpty = false) => {
    const map = mapRef.current;
    const name = litRef.current;
    const pieces = alongStreet((osmData.current?.features ?? []).filter((f) => f.properties?.name === name)
      .map((f) => (f.geometry as GeoJSON.LineString).coordinates as LonLat[]));
    if (!map || !pieces.length) return;
    const homes = buildingPoints();
    const counts = pieces.map((p) => buildingsAlong(p, homes));
    setLitPiece((cur) => {
      let n = cur?.n ?? 0;
      for (let tries = 0; tries < pieces.length; tries++) {
        n = ((n - 1 + by) % pieces.length + pieces.length) % pieces.length + 1;
        if (!skipEmpty || counts[n - 1] > 0) break;
      }
      // At a house's scale: the buildings along the piece, not the whole piece, when it has any.
      const along = homes.filter((h) => buildingsAlong(pieces[n - 1], [h]) > 0);
      frame(map, along.length ? along : pieces[n - 1]);
      return { n, of: pieces.length, buildings: counts[n - 1] };
    });
  }, []);
  const showName = useCallback((name: string) => {
    const map = mapRef.current;
    clearFound();
    setLit(name);
    // The pieces with buildings along them, where a house or a pool can be; every piece if none has.
    const pieces = (osmData.current?.features ?? []).filter((f) => f.properties?.name === name)
      .map((f) => (f.geometry as GeoJSON.LineString).coordinates as LonLat[]);
    const homes = buildingPoints();
    const built = pieces.filter((p) => buildingsAlong(p, homes) > 0);
    const pts = (built.length ? built : pieces).flat();
    if (!map || !pts.length) return;
    frame(map, pts);
  }, [clearFound]);

  const set = useCallback((id: string, features: GeoJSON.Feature[]) =>
    (mapRef.current?.getSource(id) as GeoJSONSource | undefined)?.setData(fc(features)), []);

  const piecesNamed = (name: string) => (osmData.current?.features ?? []).filter((f) => f.properties?.name === name);
  /** Every piece of each place MAPS names, in its colour; the ``hot`` one heavier and on top. */
  const setSaid = useCallback((said: { name: string; colour: string }[], hot: string | null) => {
    const ordered = [...said].sort((a, b) => Number(a.name === hot) - Number(b.name === hot));
    set("said", ordered.flatMap(({ name, colour }) =>
      piecesNamed(name).map((f) => ({ ...f, properties: { name, colour, hot: name === hot } }))));
  }, [set]);
  /** Frame every piece of a named place, no closer than a found name opens. */
  const frameName = useCallback((name: string) => {
    const map = mapRef.current;
    const pts = piecesNamed(name).flatMap((f) => (f.geometry as GeoJSON.LineString).coordinates as LonLat[]);
    if (map && pts.length) frame(map, pts);
  }, []);

  /** Frame what a step used, or the places it names when it used nothing else. */
  const frameFed = useCallback((features: GeoJSON.Feature[], names: string[]) => {
    const map = mapRef.current;
    const b = boundsOf(features.length ? features : names.flatMap(piecesNamed));
    if (map && b) map.fitBounds(b, { padding: 80, maxZoom: FED_ZOOM, duration: 400 });
  }, []);

  return { containerRef, mapRef, maplibreRef, ready, names, osmOn, setOsmOn, lit, setLit, hover, showName, flyTo, frameAo, set,
           ruler, rulerM, setRuler, markSpots, setSaid, frameName, frameFed, priorIndex, clearFound, nearOf, crossingsOf, approachOf, kindsNear, piecesOf, litPiece, stepPiece };
}

/** The street map's toggle, a chip over the map; what the line under the pointer is, beside
 * the pointer; and north, which the map always is up. */
export function OsmToggle({ on, onChange, hover, rulerM, onClearRuler, lit, pieces, litPiece, onStep }: {
  on: boolean; onChange: (on: boolean) => void; hover?: Hover | null; rulerM?: number; onClearRuler?: () => void;
  lit?: string | null; pieces?: number; litPiece?: { n: number; of: number; buildings: number } | null;
  onStep?: (by: number, skipEmpty?: boolean) => void;
}) {
  return (
    <>
      <div className="study-layers">
        <button type="button" className="study-chip" data-on={on} aria-pressed={on} onClick={() => onChange(!on)}
                title="street and building outlines with names">
          Street map: {on ? "on" : "off"} (O)
        </button>
        {lit && onStep ? (
          <span className="study-hover">
            {lit} · <span className="study-piece">{litPiece ? `section ${litPiece.n} of ${litPiece.of} · ${litPiece.buildings} building${litPiece.buildings === 1 ? "" : "s"} along it`
              : `${pieces ?? 1} section${(pieces ?? 1) > 1 ? "s" : ""}`}</span>
            {(pieces ?? 1) > 1 ? (
              <> <button type="button" className="linkish" onClick={() => onStep(-1)}>‹</button>{" "}
                <button type="button" className="linkish" onClick={() => onStep(1)}>›</button>{" "}
                <button type="button" className="linkish" title="the next section with a building along it"
                        onClick={() => onStep(1, true)}>next with buildings</button></>
            ) : null}
          </span>
        ) : null}
        {rulerM ? (
          <button type="button" className="study-hover" onClick={onClearRuler} title="shift-click the map to lay the ruler; click here to clear it">
            ruler {rulerM < 1000 ? `${rulerM.toFixed(0)} m` : `${(rulerM / 1000).toFixed(2)} km`} ✕
          </button>
        ) : <span className="study-hover study-tip">shift-click measures</span>}
      </div>
      {on && hover ? (
        <span className="study-pointer" style={{ left: hover.x, top: hover.y, transform: hover.left ? "translate(calc(-100% - 14px), 14px)" : "translate(14px, 14px)" }}>
          {hover.text}
        </span>
      ) : null}
      <span className="study-north" title="the map is north-up, always">N ↑</span>
    </>
  );
}

/** A search over the OSM prior's named places: a match lights every piece of it and frames them.
 * Matches list exact first, then by where the words start, then the rest, each alphabetically,
 * so the list holds still while the query grows; Enter takes the first. */
export function OsmSearch({ names, lit, onPick, onCross, crossingsOf, approachOf, kindsNear, piecesOf, nearOf }: {
  names: PriorName[]; lit: string | null; onPick: (n: PriorName) => void;
  onCross: (all: LonLat[], focus: number) => void; crossingsOf: (a: string, b: string) => LonLat[];
  approachOf: (a: string, b: string) => { at: LonLat; m: number } | null;
  kindsNear: (q: string, street?: string) => { name: string; kind: string; at: LonLat; m: number; near: string | null }[];
  piecesOf: (name: string) => number; nearOf: (at: LonLat, exclude: string[]) => string | null;
}) {
  const [query, setQuery] = useState("");
  // The list closes once a result is taken (or on Esc), and opens again on typing.
  const [shut, setShut] = useState(false);
  const take = (fn: () => void) => { fn(); setShut(true); };
  // "Reed & Keller", "Reed and Keller", "Reed / Keller": where the two cross.
  const pair = query.split(/\s*(?:&|\/|\band\b)\s*/i).map((x) => x.trim()).filter(Boolean);
  const best = (part: string) => {
    const lower = part.toLowerCase();
    const r = (n: PriorName) => { const name = n.name.toLowerCase(); return name === lower ? 0 : name.startsWith(lower) ? 1 : 2; };
    return names.filter((n) => n.name.toLowerCase().includes(lower)).sort((a, b) => r(a) - r(b) || a.name.localeCompare(b.name));
  };
  const cross = pair.length === 2 ? best(pair[0]).slice(0, 3).flatMap((a) => best(pair[1]).slice(0, 3)
    .filter((b) => b.name !== a.name).flatMap((b) => crossingsOf(a.name, b.name).map((at) => ({ a: a.name, b: b.name, at })))) : [];
  // Two that never meet: where they come closest, which an instruction naming their corner means.
  const [pa, pb] = pair.length === 2 ? [best(pair[0])[0], best(pair[1])[0]] : [undefined, undefined];
  const approach = pair.length === 2 && cross.length === 0 && pa && pb && pa.name !== pb.name ? approachOf(pa.name, pb.name) : null;
  const q = pair.length === 2 ? "" : query.trim().toLowerCase();
  const rank = (n: PriorName) => {
    const name = n.name.toLowerCase();
    return name === q ? 0 : name.startsWith(q) ? 1 : name.split(/\s+/).some((w) => w.startsWith(q)) ? 2 : 3;
  };
  const hits = q ? names.filter((n) => n.name.toLowerCase().includes(q))
    .sort((a, b) => rank(a) - rank(b) || a.name.localeCompare(b.name)).slice(0, 60) : [];
  // A kind of thing, named or not ("parking", "playground"), nearest the view first; with a
  // street ("parking on Willow Avenue", "pool off Moss"), nearest that street first.
  const ctx = /^(.+?)\s+(?:on|off|near|along|by)\s+(.+)$/i.exec(q);
  const ctxStreet = ctx ? best(ctx[2])[0]?.name : undefined;
  const kinds = q ? kindsNear(ctx && ctxStreet ? ctx[1].trim() : q, ctx && ctxStreet ? ctxStreet : undefined) : [];
  return (
    <section className="study-search">
      <input className="judge-search" placeholder={names.length ? `${names.length} names: a street, a store, a park` : "loading the map's names"}
             value={query} onChange={(e) => { setQuery(e.target.value); setShut(false); }} onFocus={() => setShut(false)}
             onKeyDown={(e) => {
               if (e.key === "Escape") { setShut(true); e.currentTarget.blur(); return; }
               if (e.key !== "Enter") return;
               setShut(true);
               // The map's keys (O, S, Esc) are the map's again once a result is taken.
               e.currentTarget.blur();
               if (cross[0]) onCross(cross.map((c) => c.at), 0);
               else if (approach) onCross([approach.at], 0);
               else if (hits[0]) onPick(hits[0]);
             }} />
      <p className="judge-hint">Two names with &amp; between them find where they cross.</p>
      <div className="judge-rows" hidden={shut}>
        {cross.map((c, i) => {
          const near = cross.length > 1 ? nearOf(c.at, [c.a, c.b]) : null;
          return (
            <button type="button" key={`${c.a}|${c.b}|${c.at.join(",")}`} className="judge-row" onClick={() => take(() => onCross(cross.map((x) => x.at), i))}>
              <span>{cross.length > 1 ? `${i + 1} of ${cross.length} · ` : ""}{c.a} &amp; {c.b}</span>
              <span className="judge-dim">{near ? `near ${near}` : "crossing"}</span>
            </button>
          );
        })}
        {pair.length === 2 && cross.length === 0 ? (
          approach && pa && pb ? (
            <button type="button" className="judge-row" onClick={() => take(() => onCross([approach.at], 0))}>
              <span>{pa.name} &amp; {pb.name} never meet</span>
              <span className="judge-dim">closest {approach.m.toFixed(0)} m apart</span>
            </button>
          ) : <p className="judge-hint">No two streets match those names.</p>
        ) : null}
        {hits.map((n) => (
          <button type="button" key={`${n.name}-${n.lonlat.join(",")}`} className="judge-row" data-on={lit === n.name}
                  onClick={() => take(() => onPick(n))}>
            <span>{n.name}</span>
            <span className="judge-dim">{n.kind}{piecesOf(n.name) > 1 ? ` · ${piecesOf(n.name)} sections` : ""}</span>
          </button>
        ))}
        {kinds.map((k) => (
          <button type="button" key={`${k.kind}|${k.at.join(",")}`} className="judge-row" onClick={() => take(() => onCross([k.at], 0))}>
            <span>{k.name || k.kind}{k.name ? <span className="judge-dim"> {k.kind}</span> : null}
              {k.near && k.near !== k.name ? <span className="judge-dim"> · by {k.near}</span> : null}</span>
            <span className="judge-dim">{k.m < 1000 ? `${k.m.toFixed(0)} m` : `${(k.m / 1000).toFixed(1)} km`} from {ctxStreet && ctx ? ctxStreet : "the map's center"}</span>
          </button>
        ))}
        {q && hits.length === 0 && kinds.length === 0 ? <p className="judge-hint">No street, place or kind of place matches.</p> : null}
      </div>
    </section>
  );
}
