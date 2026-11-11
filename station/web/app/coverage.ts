/** The ground a leg asks for, and how much of it the passes have held.
 *
 * The station decides a leg has ARRIVED by coverage: at least
 * `coverageFloor` of the leg's area sample points must be held by frames.
 * So the area is drawn as a MASK: every sample point, filled where a frame
 * holds it and hollow where none does, inside the hull of all of them,
 * tinted for the pass once the floor is met and for the gap while it is
 * short. The station's word arrives with each state document, so the mask
 * recolours as the passes land.
 */

export type LonLat = [number, number];

/** The station's word on a leg's area (the mission's `leg.coverage`, and
 * the final block a `waypoint` row carries once its leg has ended): the
 * share held, one flag per area point in the area's own order, and how many
 * frames hold any of it. Null before any pass. */
export interface LegCoverage {
  fraction: number;
  held: boolean[];
  frames: number;
}

/** The leg the station is waiting on (the mission's `leg`); null when
 * none is. `n` and `flight` name the waypoint row it was posted as. */
export interface WaitingLeg {
  n: number;
  target: string;
  flight: number;
  packageVersion: string;
  area: LonLat[];
  coverageFloor: number;
  coverage: LegCoverage | null;
  closest_m: number;
  posted_at: number;
  short_passes: number;
}

/** A leg's area with the station's last word on it: the waiting leg, or a
 * transcript row of one that ended. What the map draws a mask from. */
export interface AreaPass {
  area: LonLat[];
  coverage: LegCoverage | null;
  coverageFloor: number;
}

/** The pass tint, the sensor frame's own teal: ground the camera has covered. */
export const COVERAGE_HELD = "#0b7d8e";
/** The gap tint, the console's warn red: ground the passes have not held. */
export const COVERAGE_SHORT = "#9e2f1c";

/** Whether the station would call this share arrived. */
export function coverageMet(coverage: LegCoverage | null | undefined, floor: number): boolean {
  return !!coverage && coverage.fraction >= floor;
}

/** The hull's ink: the pass tint at or above the floor, the gap tint below it. */
export function coverageTint(coverage: LegCoverage | null | undefined, floor: number): string {
  return coverageMet(coverage, floor) ? COVERAGE_HELD : COVERAGE_SHORT;
}

/** "84% of 90%" while short, "held" at or above the floor, "no pass yet"
 * before any pass. */
export function coverageLabel(coverage: LegCoverage | null | undefined, floor: number): string {
  if (!coverage) return "no pass yet";
  if (coverage.fraction >= floor) return "held";
  return `${Math.round(coverage.fraction * 100)}% of ${Math.round(floor * 100)}%`;
}

/** The convex hull of `points` as a closed ring, counter-clockwise, or an
 * empty ring where fewer than three points span an area. Andrew's monotone
 * chain on lon/lat directly: the hull is affine-invariant, so the metres a
 * degree of longitude is worth at this latitude does not change which
 * points are on it. */
export function convexHull(points: readonly LonLat[]): LonLat[] {
  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (sorted.length < 3) return [];
  const cross = (o: LonLat, a: LonLat, b: LonLat) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (run: LonLat[]) => {
    const chain: LonLat[] = [];
    for (const p of run) {
      while (chain.length >= 2 && cross(chain[chain.length - 2], chain[chain.length - 1], p) <= 0) chain.pop();
      chain.push(p);
    }
    chain.pop();
    return chain;
  };
  const hull = [...half(sorted), ...half([...sorted].reverse())];
  return hull.length < 3 ? [] : [...hull, hull[0]];
}

/** Where the mask's label stands: the hull's northern edge at the area's
 * centre, beside the waypoint's number rather than on it. */
export function maskAnchor(area: readonly LonLat[]): LonLat {
  const lon = area.reduce((s, p) => s + p[0], 0) / area.length;
  return [lon, Math.max(...area.map((p) => p[1]))];
}

/** One pass as features for the map's coverage source: the hull polygon
 * (`part: "hull"`), then one point per area sample (`part: "point"`) with
 * `held` as the station says. Every feature carries the pass's `tint` and
 * whether it is `faded`, a leg that has ended and stays on the map for the
 * next one to be aimed by. */
export function coverageFeatures(pass: AreaPass, faded = false): GeoJSON.Feature[] {
  const tint = coverageTint(pass.coverage, pass.coverageFloor);
  const ring = convexHull(pass.area);
  const hull: GeoJSON.Feature[] = ring.length
    ? [{ type: "Feature", properties: { part: "hull", tint, faded },
         geometry: { type: "Polygon", coordinates: [ring] } }]
    : [];
  const points: GeoJSON.Feature[] = pass.area.map((lonlat, i) => ({
    type: "Feature",
    properties: { part: "point", tint, faded, held: pass.coverage?.held[i] ?? false },
    geometry: { type: "Point", coordinates: lonlat },
  }));
  return [...hull, ...points];
}

/** The legs a transcript has finished with: every `waypoint` row carrying
 * its area and the station's final word, and every `leg` row answered off
 * a pass already held. The row of the leg still waiting (`leg`) is the live
 * mask's and not this list's, whatever it carries. */
export function flownPasses(
  transcript: readonly {
    kind: string; n?: number; flight?: number;
    area?: LonLat[]; coverage?: LegCoverage | null; coverageFloor?: number;
  }[],
  leg: { n: number; flight: number } | null | undefined,
): AreaPass[] {
  return transcript.flatMap((row) => {
    if (!row.area || row.coverage === undefined) return [];
    if (row.kind === "waypoint" && row.coverageFloor != null
        && !(leg && row.n === leg.n && row.flight === leg.flight)) {
      return [{ area: row.area, coverage: row.coverage, coverageFloor: row.coverageFloor }];
    }
    // A reused pass is one the station found held, so its floor was met by
    // construction and the row need not say what it was.
    if (row.kind === "leg") return [{ area: row.area, coverage: row.coverage, coverageFloor: row.coverageFloor ?? 0 }];
    return [];
  });
}
