/** What a step of MAPS's search used, as the map draws it (CapacityConsole.tsx): the street
 * map's rows it names by id (``prior_2777``), the detections it names (``detected_11``) and the
 * points it passed or printed, each feature's ``fed`` saying which. */

type LonLat = [number, number];

/** What a step names, as the study server sends it (``review_doc``). */
export interface Refs { prior: string[]; detected: string[]; xy: LonLat[] }
export interface Detected { label: string; lonlat: LonLat; ring: LonLat[] | null }

/** The step's rows (their prior.json outlines), detections (outline, else point) and points;
 * an id the map holds no outline for is left out. */
export function fedFeatures(refs: Refs, prior: Map<string, GeoJSON.Feature>, detected: Record<string, Detected>): GeoJSON.Feature[] {
  const out: GeoJSON.Feature[] = [];
  for (const fid of refs.prior) {
    const f = prior.get(fid);
    if (f) out.push({ type: "Feature", properties: { fed: "prior", fid }, geometry: f.geometry });
  }
  for (const fid of refs.detected) {
    const d = detected[fid];
    if (!d) continue;
    out.push({ type: "Feature", properties: { fed: "detected", fid },
               geometry: d.ring ? { type: "LineString", coordinates: [...d.ring, d.ring[0]] } : { type: "Point", coordinates: d.lonlat } });
  }
  for (const at of refs.xy) out.push({ type: "Feature", properties: { fed: "point" }, geometry: { type: "Point", coordinates: at } });
  return out;
}

/** The south-west and north-east corners around every vertex of ``features``; null for none. */
export function boundsOf(features: GeoJSON.Feature[]): [LonLat, LonLat] | null {
  const pts = features.flatMap((f) => {
    const g = f.geometry;
    return g.type === "Point" ? [g.coordinates] : g.type === "LineString" ? g.coordinates : [];
  });
  if (!pts.length) return null;
  const lons = pts.map((p) => p[0]);
  const lats = pts.map((p) => p[1]);
  return [[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]];
}
