/** The study map's street crossings and distances.
 *
 * Run with `npm test` in baselines/human/console.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { buildingsAlong, closestApproach, crossings, metres, stretches } from "./studyGeometry.ts";

const street = (name, coordinates) => ({ properties: { name }, geometry: { type: "LineString", coordinates } });
// About 10 m a step at Cal City's latitude.
const E = 1e-4, N = 9e-5;

test("a metre is a metre at the city's latitude", () => {
  assert.ok(Math.abs(metres([-117.96, 35.12], [-117.96, 35.12 + N]) - 10.0) < 0.2);
});

test("two streets that cross are found where they cross, once per crossing", () => {
  const f = [street("Reed Place", [[0, 0], [10 * E, 0]]), street("Keller Drive", [[5 * E, -5 * N], [5 * E, 5 * N]]),
             street("Keller Drive", [[8 * E, -5 * N], [8 * E, 5 * N]])];
  const at = crossings(f, "Reed Place", "Keller Drive").sort((a, b) => a[0] - b[0]);
  assert.equal(at.length, 2);
  assert.ok(Math.abs(at[0][0] - 5 * E) < 1e-9 && Math.abs(at[0][1]) < 1e-9);
});

test("a T drawn a little short still meets its street; streets far apart do not", () => {
  const t = [street("Lime Avenue", [[0, 0], [10 * E, 0]]), street("90th Street", [[5 * E, 1.5 * N], [5 * E, 10 * N]])];
  assert.equal(crossings(t, "Lime Avenue", "90th Street").length, 1);
  const apart = [street("Lime Avenue", [[0, 0], [10 * E, 0]]), street("90th Street", [[5 * E, 6 * N], [5 * E, 10 * N]])];
  assert.equal(crossings(apart, "Lime Avenue", "90th Street").length, 0);
});

test("two streets that never meet report where they come closest and how far apart", () => {
  const f = [street("Aspen Mall", [[0, 0], [10 * E, 0]]), street("Frontage", [[12 * E, 3 * N], [20 * E, 3 * N]])];
  const near = closestApproach(f, "Aspen Mall", "Frontage");
  // The fixture sits at latitude 0: 2E of longitude is 22.3 m, 3N of latitude 30.1 m.
  assert.ok(near && Math.abs(near.m - Math.hypot(22.3, 30.1)) < 0.5);
});

test("the buildings along a piece are those within forty metres of it", () => {
  const piece = [[0, 0], [10 * E, 0]];
  assert.equal(buildingsAlong(piece, [[5 * E, 3 * N], [5 * E, 6 * N], [20 * E, 0]]), 1);
});

test("a street's pieces that touch are one stretch, and one far off is another, west first", () => {
  const far = [[100 * E, 0], [110 * E, 0]];
  const groups = stretches([far, [[0, 0], [5 * E, 0]], [[5 * E, 0], [10 * E, N]], [[10 * E + E, N], [20 * E, N]]]);
  assert.deepEqual(groups.map((g) => g.length), [3, 1]);
  assert.deepEqual(groups[1], [far]);
  assert.equal(stretches([[[0, 0], [5 * E, 0]], [[12 * E, 0], [20 * E, 0]]]).length, 2);   // 70 m apart
  const south = [[0, -100 * N], [0, -90 * N]], north = [[0, 0], [0, 10 * N]];
  assert.deepEqual(stretches([south, north]), [[north], [south]]);                     // a N–S street, north first
});
