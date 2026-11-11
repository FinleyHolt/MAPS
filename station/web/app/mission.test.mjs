/** The reviewed route's legs, against the gaps in the record, and the
 * commit's picture.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { commitLasso, trackLegs } from "./mission.ts";

/** A track sample at `t_s`, placed unless `lon` is null. */
const at = (t_s, lon = -118 + t_s / 1000, lat = 35.1) =>
  lon === null
    ? { t_s, x: 0, y: 0, track_deg: 90, agl_m: 762 }
    : { t_s, x: 0, y: 0, track_deg: 90, agl_m: 762, lon, lat };

test("an unbroken track is one leg", () => {
  const legs = trackLegs([at(0), at(2), at(4), at(6)]);
  assert.equal(legs.length, 1);
  assert.equal(legs[0].length, 4);
  assert.deepEqual(legs[0][0], [-118, 35.1]);
});

test("a dropout breaks the line rather than crossing it", () => {
  // Sampled every 2 s, with 40 s of nothing in the middle.
  const legs = trackLegs([at(0), at(2), at(4), at(44), at(46)]);
  assert.equal(legs.length, 2);
  assert.deepEqual(legs.map((l) => l.length), [3, 2]);
});

test("the step is read off the track, not assumed", () => {
  // A long run is sampled wider; 30 s apart is no gap at that step.
  const legs = trackLegs([at(0), at(30), at(60), at(90)]);
  assert.equal(legs.length, 1);
});

test("a sample the map cannot place breaks the leg", () => {
  const legs = trackLegs([at(0), at(2), at(4, null), at(6), at(8)]);
  assert.equal(legs.length, 2);
  assert.deepEqual(legs.map((l) => l.length), [2, 2]);
});

test("nothing drawable is no legs at all", () => {
  assert.deepEqual(trackLegs([]), []);
  assert.deepEqual(trackLegs([at(0)]), []);
  assert.deepEqual(trackLegs([at(0, null), at(2, null)]), []);
  // A leg needs two points: a lone sample either side of a dropout draws
  // neither of them.
  assert.deepEqual(trackLegs([at(0), at(2), at(400), at(800), at(802)]).map((l) => l.length),
                   [2, 2]);
});

/** A mission carrying `observations`, for the commit's picture. */
const flown = (observations) => ({ path: "20260915T100000Z/run01", observations });
const shot = (sha, target, commit) => ({
  sha, target, xy: [0, 0], looks: 1, ground_m2: 1, placed: 1, t_s: 1,
  ...(commit ? { commit } : {}),
});

test("the commit's picture is the observation that carries the lasso", () => {
  const got = commitLasso(flown([shot("a", "prior_1"), shot("b", "prior_2", "b.commit.jpg")]));
  assert.equal(got.target, "prior_2");
  assert.equal(
    got.url,
    "run/file?path=" + encodeURIComponent("20260915T100000Z/run01/observations/b.commit.jpg"),
  );
});

test("no lasso on any observation is no picture, and the report still shows", () => {
  assert.equal(commitLasso(flown([shot("a", "prior_1")])), null);
  assert.equal(commitLasso(flown([])), null);
  assert.equal(commitLasso({ path: "r/w" }), null);
  // No folder to fetch from: `missionFile` refuses and so does this.
  assert.equal(commitLasso({ path: null, observations: [shot("b", "prior_2", "b.commit.jpg")] }), null);
});
