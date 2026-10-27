/** What a step of MAPS's search used, as the map draws it.
 *
 * Run with `npm test` in baselines/human/console.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { boundsOf, fedFeatures } from "./fed.ts";

const line = (fid, coords) => ({ type: "Feature", properties: { fid }, geometry: { type: "LineString", coordinates: coords } });
const prior = new Map([["prior_2", line("prior_2", [[0, 0], [1, 1]])]]);
const detected = { detected_4: { label: "car", lonlat: [5, 5], ring: [[4, 4], [6, 4], [6, 6]] }, detected_5: { label: "pool", lonlat: [7, 8], ring: null } };

test("each row, detection and point a step names, by what it is; an id the map lacks is left out", () => {
  const f = fedFeatures({ prior: ["prior_2", "prior_9"], detected: ["detected_4", "detected_5", "detected_6"], xy: [[2, 3]] }, prior, detected);
  assert.deepEqual(f.map((x) => [x.properties.fed, x.geometry.type]),
                   [["prior", "LineString"], ["detected", "LineString"], ["detected", "Point"], ["point", "Point"]]);
  // A detection's outline closes on itself.
  assert.deepEqual(f[1].geometry.coordinates, [[4, 4], [6, 4], [6, 6], [4, 4]]);
});

test("the bounds hold every vertex; nothing has none", () => {
  const f = fedFeatures({ prior: ["prior_2"], detected: ["detected_5"], xy: [[-1, 3]] }, prior, detected);
  assert.deepEqual(boundsOf(f), [[-1, 0], [7, 8]]);
  assert.equal(boundsOf([]), null);
});
