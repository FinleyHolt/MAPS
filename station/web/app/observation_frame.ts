/** The map's features drawn on an observation, by the page.
 *
 * The station sends the observation's ground (`world_window`, map-local
 * metres) and size (`shape`) on the record; the prior's features come once
 * per page load from `GET mission/prior.json`; the picture book lays an SVG
 * over the image, in image pixels. The SVG's viewBox is the image's pixel
 * size and `meet` letterboxes it exactly as the image's
 * `object-fit: contain` does, so no rendered size is measured.
 */

/** One feature: its legend group and its vertices in map-local metres. */
export type PriorFeature = [string, [number, number][]];
export type WorldWindow = [number, number, number, number];
/** `[h, w]` in pixels. */
export type Shape = [number, number];

/** One colour per legend group. */
export const PRIOR_COLOURS: Record<string, string> = {
  road: "rgb(255,220,0)", path: "rgb(255,140,0)", building: "rgb(0,200,255)", pitch: "rgb(0,255,120)",
  grass: "rgb(90,170,90)", parking: "rgb(150,150,255)", barrier: "rgb(255,0,255)", node: "rgb(255,255,255)",
  other: "rgb(170,170,170)",
};

/** `u = col / (w - 1)` east across the window, `v = row / (h - 1)` south down it. */
export function imagePixel([x, y]: [number, number], [x0, y0, x1, y1]: WorldWindow, [h, w]: Shape): [number, number] {
  return [(x - x0) / (x1 - x0) * (w - 1), (y1 - y) / (y1 - y0) * (h - 1)];
}

export type DrawnFeature = { colour: string; points: [number, number][]; closed: boolean };

/** The features that reach the window, in image pixels: a lone vertex is a
 * node, a ring whose first vertex is its last (four or more) is closed. */
export function framePrior(prior: PriorFeature[], window: WorldWindow, shape: Shape): DrawnFeature[] {
  const [x0, y0, x1, y1] = window;
  const out: DrawnFeature[] = [];
  for (const [group, xy] of prior) {
    if (!xy.length || xy.every(([x]) => x < x0) || xy.every(([x]) => x > x1)
        || xy.every(([, y]) => y < y0) || xy.every(([, y]) => y > y1)) continue;
    const [fx, fy] = xy[0], [lx, ly] = xy[xy.length - 1];
    out.push({ colour: PRIOR_COLOURS[group] ?? PRIOR_COLOURS.other,
               points: xy.map((p) => imagePixel(p, window, shape)),
               closed: xy.length >= 4 && Math.abs(fx - lx) < 1e-6 && Math.abs(fy - ly) < 1e-6 });
  }
  return out;
}

const fetched = new Map<string, Promise<PriorFeature[] | null>>();

/** The map's prior, fetched once per page load per map; null when the
 * station has no features for it. */
export function priorFor(map: string): Promise<PriorFeature[] | null> {
  let p = fetched.get(map);
  if (!p) {
    p = fetch(`mission/prior.json?map=${encodeURIComponent(map)}`)
      .then((r) => (r.ok ? (r.json() as Promise<PriorFeature[]>) : null))
      .catch(() => null);
    fetched.set(map, p);
  }
  return p;
}
