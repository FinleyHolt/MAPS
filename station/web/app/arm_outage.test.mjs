/** The outage chip reads under a flying mission whose last REASONER row is
 * a fault, and nowhere else: the loop is holding its turn, and the moment
 * the reasoner answers the next row of its own is the model's.
 *
 * The station's rows do not clear it: a leg wait posts waypoints and legs
 * with the model never called.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { armOutage } from "./mission.ts";

const fault = (attempt, since) => ({ kind: "transient_error", attempt, error: "ConnectError", since });
const flying = (transcript) => ({ status: "flying", transcript });
const T0 = 1700000000;

test("a flying mission whose last row is a fault reads an outage", () => {
  assert.deepEqual(armOutage(flying([{ kind: "run" }, fault(3, 1700000000)])),
    { attempt: 3, since: 1700000000 });
});

test("the model answering clears it", () => {
  assert.equal(armOutage(flying([fault(3, 1700000000), { kind: "run" }])), null);
});

test("an ended, parked, or absent mission never reads it", () => {
  assert.equal(armOutage({ status: "ended", transcript: [fault(2, 1)] }), null);
  assert.equal(armOutage({ status: "awaiting_tasking", transcript: [fault(2, 1)] }), null);
  assert.equal(armOutage(null), null);
  assert.equal(armOutage(flying([])), null);
});

test("a row the STATION wrote does not clear it", () => {
  assert.deepEqual(
    armOutage(flying([fault(1, T0), fault(2, T0), fault(3, T0), { kind: "waypoint", flight: 1, n: 0 }])),
    { attempt: 3, since: T0 });
  for (const kind of ["waypoint", "leg", "view", "retask", "summary"]) {
    assert.deepEqual(armOutage(flying([fault(3, T0), { kind }])), { attempt: 3, since: T0 },
      `a ${kind} row is the station's, not the model answering`);
  }
  // And the model answering still clears it, through the station's rows.
  assert.equal(armOutage(flying([fault(3, T0), { kind: "run" }, { kind: "waypoint" }, { kind: "leg" }])), null);
});

test("the count is the burst's own and the clock is its own since", () => {
  // The loop zeroes its failure count on success, so a second outage
  // restarts the counter at 1 while `since` repeats across one burst.
  const a = 1789343738.706697, b = a + 51;
  assert.deepEqual(armOutage(flying([
    fault(1, a), fault(2, a), fault(3, a), fault(4, a),
    { kind: "run" },
    fault(1, b), fault(2, b), fault(3, b),
    { kind: "waypoint", flight: 0, n: 1, label: "PRIOR_4533" },
  ])), { attempt: 3, since: b });
});

test("a mission with nothing but the station's own rows reads no outage", () => {
  assert.equal(armOutage(flying([{ kind: "waypoint" }, { kind: "leg" }])), null);
});

test("the console carries the chip beside the status", () => {
  const src = readFileSync(new URL("./MissionConsole.tsx", import.meta.url), "utf8");
  assert.match(src, /data-testid="arm-down"/);
  assert.match(src, /armOutage\(mission\)/);
  assert.match(src, /holding the turn/);
});
