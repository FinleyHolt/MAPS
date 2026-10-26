/** What the capacity console shows of a session and sends back to it.
 *
 * Run with `npm test` in baselines/human/console.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { actBody, header, keyCommand, mapTakesClicks, overLines, pollState } from "./capacityView.ts";

const team = { arm: "team", practice: false, remaining_s: 723, over: false, current: { episode_id: "calcity::7#6", mode: "verify" } };
const practice = { ...team, arm: "solo", practice: true, current: { episode_id: "calcity::1#0", mode: "search" } };

test("every act names the report it answers", () => {
  assert.deepEqual(actBody(team, { action: "skip" }), { action: "skip", episode_id: "calcity::7#6" });
  assert.deepEqual(actBody(null, { action: "skip" }), { action: "skip" });
});

test("a held key acts once", () => {
  assert.equal(keyCommand({ key: "Enter", repeat: false }, "verify", false), "confirm");
  assert.equal(keyCommand({ key: "Enter", repeat: true }, "verify", false), null);
  assert.equal(keyCommand({ key: "s", repeat: true }, "search", true), null);
  assert.equal(keyCommand({ key: "S", repeat: false }, "search", true), "pick");
  assert.equal(keyCommand({ key: "s", repeat: false }, "search", false), null);
  assert.equal(keyCommand({ key: "Enter", repeat: false }, "search", true), null);
});

test("a failed poll reads as lost contact, and a good one as the state", async () => {
  assert.equal(await pollState(async () => ({ ok: false, status: 409, json: async () => ({ detail: "x" }) })), null);
  assert.equal(await pollState(async () => { throw new TypeError("Failed to fetch"); }), null);
  assert.deepEqual(await pollState(async () => ({ ok: true, status: 200, json: async () => team })), team);
});

test("the header names the session; practice has no clock", () => {
  assert.deepEqual(header(team), { session: "Session: with MAPS", clock: "12:03 left" });
  assert.deepEqual(header({ ...team, arm: "solo" }), { session: "Session: alone", clock: "12:03 left" });
  assert.deepEqual(header(practice), { session: "Practice", clock: null });
});

test("practice ends without a time-ran-out sentence", () => {
  const lines = overLines({ ...practice, over: true, answered_n: 4, skipped_n: 0 });
  assert.deepEqual(lines, ["Practice finished. Press Continue to go back to the study page."]);
  const real = overLines({ ...team, over: true, answered_n: 3, skipped_n: 1 }).join(" ");
  assert.match(real, /3 answered, 1 skipped/);
  assert.match(real, /time ran out/);
});

test("the map takes no click in its first second, where a double-clicked start button lands", () => {
  assert.equal(mapTakesClicks(0, 5000), false);
  assert.equal(mapTakesClicks(4000, 4600), false);
  assert.equal(mapTakesClicks(4000, 5000), true);
});
