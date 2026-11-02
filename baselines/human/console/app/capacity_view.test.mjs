/** What the capacity console shows of a session and sends back to it.
 *
 * Run with `npm test` in baselines/human/console.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { actBody, aKind, header, keyCommand, mapTakesClicks, objectName, overLines, pollState, verifyHint } from "./capacityView.ts";

const team = { arm: "team", practice: false, remaining_s: 723, over: false, current: { episode_id: "calcity::7#6", mode: "verify" } };
const practice = { ...team, arm: "solo", practice: true, current: { episode_id: "calcity::1#0", mode: "search" } };

test("every act names the report it answers", () => {
  assert.deepEqual(actBody(team, { action: "skip" }), { action: "skip", episode_id: "calcity::7#6" });
  assert.equal(actBody(null, { action: "skip" }), null);
  assert.equal(actBody({ current: null }, { action: "skip" }), null);
});

test("a held key acts once", () => {
  assert.equal(keyCommand({ key: "Enter", repeat: false }, "verify", false), "confirm");
  assert.equal(keyCommand({ key: "Enter", repeat: true }, "verify", false), null);
  assert.equal(keyCommand({ key: "s", repeat: true }, "search", true), null);
  assert.equal(keyCommand({ key: "S", repeat: false }, "search", true), "pick");
  assert.equal(keyCommand({ key: "s", repeat: false }, "search", false), null);
  assert.equal(keyCommand({ key: "Enter", repeat: false }, "search", true), null);
  assert.equal(keyCommand({ key: "b", repeat: false }, "verify", false), "back");
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

test("the verify hint shows the study's own refusal, and none while the point is affordable", () => {
  const refusal = "MAPS could not have reached that spot.";
  assert.equal(verifyHint({ affordable: false, refusal }), refusal);
  assert.match(verifyHint({ affordable: true, refusal: null }), /Press Pick another to answer with it/);
  assert.match(verifyHint({ affordable: true }), /Press Pick another to answer with it/);
  assert.match(verifyHint(null), /Press Confirm if MAPS's answer is right/);
});

test("the clock counts down from the last state the study sent, and stops at zero", () => {
  assert.deepEqual(header(team, 3.4).clock, "11:59 left");
  assert.deepEqual(header(team, 5000).clock, "0:00 left");
});

test("a mapped object reads as its kind and number where the study names a kind, else as its number", () => {
  const house = { id: 9, label: "building 568", kind: "house" };
  assert.equal(objectName(house), "house 568");
  assert.equal(objectName({ ...house, kind: "apartments" }), "apartments 568");
  assert.equal(objectName({ id: 9, label: "building 568" }), "object 568");
  assert.equal(objectName({ id: 9, label: "x" }), "object 9");
});

test("MAPS's answer reads as its kind, with an article and no number", () => {
  assert.equal(aKind("residential street"), "a residential street");
  assert.equal(aKind("apartment building"), "an apartment building");
});
