/** What the capacity console shows of a session and sends back to it.
 *
 * Run with `npm test` in baselines/human/console.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { actBody, keyCommand } from "./capacityView.ts";

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
