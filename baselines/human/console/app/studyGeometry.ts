/** The study map's plane geometry over WGS84 points (studyMap.tsx): distances and street
 * crossings at a city's scale, where a degree is flat enough to measure in. */

export type LonLat = [number, number];

/** Metres between two points, flat at this scale (a city block, not a continent). */
export function metres(a: LonLat, b: LonLat): number {
  const k = Math.cos(((a[1] + b[1]) / 2) * Math.PI / 180);
  return Math.hypot((a[0] - b[0]) * k, a[1] - b[1]) * 111_320;
}

/** Where two named streets cross: every crossing of a piece of one with a piece of the other,
 * and where they only come within NEAR_M of each other (a T the prior drew a little short), the
 * closest approach. Each spot once. */
/** The point of segment ``a``-``b`` nearest ``x`` (planar in degrees, scaled for latitude). */
function nearestOn(x: LonLat, a: LonLat, b: LonLat): LonLat {
  const k = Math.cos(a[1] * Math.PI / 180);
  const [dx, dy] = [(b[0] - a[0]) * k, b[1] - a[1]];
  const len = dx * dx + dy * dy;
  const u = len === 0 ? 0 : Math.max(0, Math.min(1, (((x[0] - a[0]) * k) * dx + (x[1] - a[1]) * dy) / len));
  return [a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])];
}

export function crossings(features: { properties?: Record<string, unknown> | null; geometry: unknown }[], a: string, b: string): LonLat[] {
  const NEAR_M = 25;
  const pieces = (name: string) => features.filter((f) => f.properties?.name === name)
    .map((f) => (f.geometry as { coordinates: LonLat[] }).coordinates);
  const out: LonLat[] = [];
  const add = (p: LonLat) => { if (!out.some((q) => metres(p, q) < 30)) out.push(p); };
  for (const pa of pieces(a)) for (const pb of pieces(b)) {
    let best: { d: number; p: LonLat } | null = null;
    for (let i = 1; i < pa.length; i++) for (let j = 1; j < pb.length; j++) {
      const [p, q, r, t] = [pa[i - 1], pa[i], pb[j - 1], pb[j]];
      const den = (q[0] - p[0]) * (t[1] - r[1]) - (q[1] - p[1]) * (t[0] - r[0]);
      if (den !== 0) {
        const u = ((r[0] - p[0]) * (t[1] - r[1]) - (r[1] - p[1]) * (t[0] - r[0])) / den;
        const v = ((r[0] - p[0]) * (q[1] - p[1]) - (r[1] - p[1]) * (q[0] - p[0])) / den;
        if (u >= 0 && u <= 1 && v >= 0 && v <= 1) { add([p[0] + u * (q[0] - p[0]), p[1] + u * (q[1] - p[1])]); continue; }
      }
      // Each end of one piece against the other piece's whole segment.
      for (const [x, s0, s1] of [[p, r, t], [q, r, t], [r, p, q], [t, p, q]] as [LonLat, LonLat, LonLat][]) {
        const y = nearestOn(x, s0, s1);
        const d = metres(x, y);
        if (d < NEAR_M && (!best || d < best.d)) best = { d, p: y };
      }
    }
    if (best) add(best.p);
  }
  return out;
}


/** Which way a street runs at a point, as an axis: "N–S", "NE–SW", "E–W" or "NW–SE". */
export function axisAt(line: LonLat[], at: LonLat): string {
  let best = { d: Infinity, i: 1 };
  for (let i = 1; i < line.length; i++) {
    const d = metres(at, nearestOn(at, line[i - 1], line[i]));
    if (d < best.d) best = { d, i };
  }
  const [a, b] = [line[best.i - 1], line[best.i]];
  const k = Math.cos(a[1] * Math.PI / 180);
  const deg = ((Math.atan2((b[0] - a[0]) * k, b[1] - a[1]) * 180) / Math.PI + 180) % 180;
  return ["N–S", "NE–SW", "E–W", "NW–SE"][Math.round(deg / 45) % 4];
}

/** The nearest named line to ``at`` that is not one of ``exclude``, within ``within_m``: what
 * tells two crossings of the same two streets apart ("near Fuchsia Avenue"). */
export function nearestOther(features: { properties?: Record<string, unknown> | null; geometry: unknown }[],
                             at: LonLat, exclude: string[], within_m = 250): string | null {
  let best: { d: number; name: string } | null = null;
  for (const f of features) {
    const name = String(f.properties?.name ?? "");
    if (!name || exclude.includes(name)) continue;
    const line = (f.geometry as { coordinates: LonLat[] }).coordinates;
    for (let i = 1; i < line.length; i++) {
      const d = metres(at, nearestOn(at, line[i - 1], line[i]));
      if (d < within_m && (!best || d < best.d)) best = { d, name };
    }
  }
  return best?.name ?? null;
}

/** Where two named streets come closest, and how far apart, for two that never meet: the place
 * an instruction naming their "intersection" most likely means. */
export function closestApproach(features: { properties?: Record<string, unknown> | null; geometry: unknown }[],
                                a: string, b: string): { at: LonLat; m: number } | null {
  const pieces = (name: string) => features.filter((f) => f.properties?.name === name)
    .map((f) => (f.geometry as { coordinates: LonLat[] }).coordinates);
  let best: { at: LonLat; m: number } | null = null;
  for (const pa of pieces(a)) for (const pb of pieces(b)) {
    for (const [line, other] of [[pa, pb], [pb, pa]] as [LonLat[], LonLat[]][]) {
      for (const x of line) for (let j = 1; j < other.length; j++) {
        const y = nearestOn(x, other[j - 1], other[j]);
        const m = metres(x, y);
        if (!best || m < best.m) best = { at: [(x[0] + y[0]) / 2, (x[1] + y[1]) / 2], m };
      }
    }
  }
  return best;
}

/** A street's pieces in order along it: from the piece with the westmost end, each next piece the
 * one whose end is nearest the last one's far end (each piece turned to run onward). A street the
 * prior breaks into pieces is then walked end to end, not jumped about. */
export function alongStreet(pieces: LonLat[][]): LonLat[][] {
  const left = pieces.filter((p) => p.length > 1).map((p) => [...p]);
  if (!left.length) return [];
  const startAt = left.reduce((bi, p, i) => (Math.min(p[0][0], p[p.length - 1][0]) < Math.min(left[bi][0][0], left[bi][left[bi].length - 1][0]) ? i : bi), 0);
  let cur = left.splice(startAt, 1)[0];
  if (cur[cur.length - 1][0] < cur[0][0]) cur.reverse();
  const out = [cur];
  while (left.length) {
    const end = cur[cur.length - 1];
    let best = { i: 0, flip: false, d: Infinity };
    left.forEach((p, i) => {
      const a = metres(end, p[0]), b = metres(end, p[p.length - 1]);
      if (a < best.d) best = { i, flip: false, d: a };
      if (b < best.d) best = { i, flip: true, d: b };
    });
    cur = left.splice(best.i, 1)[0];
    if (best.flip) cur.reverse();
    out.push(cur);
  }
  return out;
}

/** How many of ``points`` (a building's first vertex each) lie within ``within_m`` of ``line``. */
export function buildingsAlong(line: LonLat[], points: LonLat[], within_m = 40): number {
  const xs = line.map((p) => p[0]); const ys = line.map((p) => p[1]);
  const pad = within_m / 90_000;
  const [x0, x1, y0, y1] = [Math.min(...xs) - pad, Math.max(...xs) + pad, Math.min(...ys) - pad, Math.max(...ys) + pad];
  let n = 0;
  for (const p of points) {
    if (p[0] < x0 || p[0] > x1 || p[1] < y0 || p[1] > y1) continue;
    for (let i = 1; i < line.length; i++) if (metres(p, nearestOn(p, line[i - 1], line[i])) <= within_m) { n++; break; }
  }
  return n;
}
