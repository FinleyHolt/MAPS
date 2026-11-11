/** A leg's area as a mask: which points are held, what the label says,
 * which tint the hull takes, and which rows keep a faded mask.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  COVERAGE_HELD, COVERAGE_SHORT, convexHull, coverageFeatures, coverageLabel, coverageTint,
  flownPasses, maskAnchor,
} from "./coverage.ts";

// The station's 19-point disc about a point: the centre, six at one step,
// twelve at two. The three points on its south rim are the ones a pass
// flown along the north edge misses.
const CENTRE = [-117.97, 35.12];
const STEP = 0.0003;
const DISC = [CENTRE];
for (let i = 0; i < 6; i += 1) {
  const a = (Math.PI * 2 * i) / 6;
  DISC.push([CENTRE[0] + STEP * Math.sin(a), CENTRE[1] + STEP * Math.cos(a)]);
}
for (let i = 0; i < 12; i += 1) {
  const a = (Math.PI * 2 * i) / 12;
  DISC.push([CENTRE[0] + 2 * STEP * Math.sin(a), CENTRE[1] + 2 * STEP * Math.cos(a)]);
}
const SOUTH = DISC.map((p, i) => i).filter((i) => DISC[i][1] < CENTRE[1] - 1.5 * STEP);
const FLOOR = 0.9;

const short = { fraction: 16 / 19, held: DISC.map((_, i) => !SOUTH.includes(i)), frames: 8 };
const whole = { fraction: 1, held: DISC.map(() => true), frames: 11 };

test("the disc is nineteen points and three of them lie on the south rim", () => {
  assert.equal(DISC.length, 19);
  assert.deepEqual(SOUTH.length, 3);
});

test("a pass short of the floor reads its share, and the south points read unheld", () => {
  assert.equal(coverageLabel(short, FLOOR), "84% of 90%");
  assert.equal(coverageTint(short, FLOOR), COVERAGE_SHORT);
  const features = coverageFeatures({ area: DISC, coverage: short, coverageFloor: FLOOR });
  const points = features.filter((f) => f.properties.part === "point");
  assert.equal(points.length, 19);
  assert.deepEqual(points.map((f, i) => (f.properties.held ? null : i)).filter((i) => i !== null), SOUTH);
  assert.ok(points.every((f) => f.properties.tint === COVERAGE_SHORT && f.properties.faded === false));
});

test("a pass at the floor reads held, in the pass tint, every point filled", () => {
  assert.equal(coverageLabel(whole, FLOOR), "held");
  assert.equal(coverageLabel({ fraction: 0.9, held: [], frames: 1 }, FLOOR), "held");
  assert.equal(coverageTint(whole, FLOOR), COVERAGE_HELD);
  const points = coverageFeatures({ area: DISC, coverage: whole, coverageFloor: FLOOR })
    .filter((f) => f.properties.part === "point");
  assert.ok(points.every((f) => f.properties.held === true && f.properties.tint === COVERAGE_HELD));
});

test("no pass yet reads so, and no point is held", () => {
  assert.equal(coverageLabel(null, FLOOR), "no pass yet");
  assert.equal(coverageTint(null, FLOOR), COVERAGE_SHORT);
  const points = coverageFeatures({ area: DISC, coverage: null, coverageFloor: FLOOR })
    .filter((f) => f.properties.part === "point");
  assert.equal(points.length, 19);
  assert.ok(points.every((f) => f.properties.held === false));
});

test("the hull is the outer rim, closed, and the label stands on its north edge", () => {
  const ring = convexHull(DISC);
  assert.deepEqual(ring[0], ring[ring.length - 1]);
  // Twelve rim points plus the closing one; the centre and the inner six are inside.
  assert.equal(ring.length, 13);
  assert.ok(ring.every((p) => Math.hypot(p[0] - CENTRE[0], p[1] - CENTRE[1]) > 1.9 * STEP));
  const hull = coverageFeatures({ area: DISC, coverage: short, coverageFloor: FLOOR })
    .filter((f) => f.properties.part === "hull");
  assert.equal(hull.length, 1);
  assert.equal(hull[0].geometry.type, "Polygon");
  const anchor = maskAnchor(DISC);
  assert.ok(Math.abs(anchor[0] - CENTRE[0]) < 1e-9 && Math.abs(anchor[1] - (CENTRE[1] + 2 * STEP)) < 1e-9);
  // Fewer than three points span no area, and a faded pass says so on every feature.
  assert.deepEqual(convexHull([CENTRE, DISC[1]]), []);
  assert.ok(coverageFeatures({ area: DISC, coverage: whole, coverageFloor: FLOOR }, true)
    .every((f) => f.properties.faded === true));
});

test("a square's hull is its four corners", () => {
  const square = [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.2, 0.7]];
  assert.deepEqual(convexHull(square), [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]);
});

test("the faded masks are the ended legs' rows, never the one still waiting", () => {
  const rows = [
    { kind: "waypoint", flight: 0, n: 1, area: DISC, coverageFloor: FLOOR, coverage: whole },
    { kind: "leg", said: "x is held by the pass already", area: DISC, coverage: whole },
    { kind: "waypoint", flight: 1, n: 2, area: DISC, coverageFloor: FLOOR },
    { kind: "waypoint", flight: 2, n: 3, area: DISC, coverageFloor: FLOOR, coverage: null },
    { kind: "run", said: "go_to(x)" },
    { kind: "waypoint", flight: 3, n: 4, lat: 1, lon: 2 },
  ];
  const passes = flownPasses(rows, { n: 3, flight: 2 });
  assert.equal(passes.length, 2);
  assert.ok(passes.every((p) => p.coverage === whole));
  assert.equal(coverageTint(passes[1].coverage, passes[1].coverageFloor), COVERAGE_HELD);
  assert.equal(flownPasses(rows, null).length, 3);
  assert.equal(coverageLabel(flownPasses(rows, null)[2].coverage, FLOOR), "no pass yet");
});
