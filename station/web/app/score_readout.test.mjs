/** A run's score reads, whatever the criterion left out.
 *
 * A criterion that does not populate a field leaves the KEY OUT rather
 * than writing null, and a reader that only tests for null hands
 * `undefined` to `toFixed`.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { formatScore } from "./mission.ts";

/** A row scored under the ring criterion: no OSR, no SPL, keys absent. */
const RING = { SR: 1, NE: 2.31, scoring_basis: "final_pos_xy" };

test("a field the criterion never populated reads as an em dash, not a throw", () => {
  const line = formatScore(RING);
  assert.equal(line, "SR 1 · OSR — · NE 2.31m · SPL —");
});

test("null and absent read the same", () => {
  assert.equal(formatScore({ SR: 1, OSR: null, NE: 2.31, spl_pos: null }), formatScore(RING));
});

test("every field named, so a blank is a criterion and not a lost number", () => {
  const line = formatScore({ SR: 0, OSR: 0.5, NE: 262.254, spl_pos: 1 });
  for (const field of ["SR", "OSR", "NE", "SPL"]) assert.ok(line.includes(field), field);
  // Integers stay integers; a fraction is two places.
  assert.equal(line, "SR 0 · OSR 0.50 · NE 262.25m · SPL 1");
});

test("no score at all is no readout", () => {
  assert.equal(formatScore(null), null);
  assert.equal(formatScore(undefined), null);
});
