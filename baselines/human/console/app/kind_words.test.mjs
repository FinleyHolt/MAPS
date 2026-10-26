/** The everyday words the street-map search takes for a kind of place.
 *
 * Run with `npm test` in baselines/human/console.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { KIND_WORDS, kindMatches } from "./kindWords.ts";

// The kinds the server names (study_server.PLAIN's words), and the OSM tags it passes through
// that the aliases rely on.
const server = readFileSync(new URL("../../study_server.py", import.meta.url), "utf8");
const plain = [...server.slice(server.indexOf("PLAIN = {"), server.indexOf("}\n", server.indexOf("PLAIN = {")))
  .matchAll(/\): "([^"]+)"/g)].map((m) => m[1]);
const tags = ["water", "swimming pool", "parking", "place of worship", "church", "school", "grass"];

test("a court is a sports field", () => {
  assert.equal(kindMatches("sports field", "court"), true);
  assert.equal(kindMatches("residential street", "court"), false);
});

test("every alias names a kind the server gives", () => {
  const kinds = [...plain, ...tags];
  const dead = Object.entries(KIND_WORDS).flatMap(([q, ws]) => ws.filter((w) => !kinds.some((k) => k.includes(w))).map((w) => `${q}: ${w}`));
  assert.deepEqual(dead, []);
});

test("a word with no alias is matched as itself", () => {
  assert.equal(kindMatches("gas station", "gas"), true);
});
