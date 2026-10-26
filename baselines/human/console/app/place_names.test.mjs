/** The places a stretch of MAPS's reasoning names.
 *
 * Run with `npm test` in baselines/human/console.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { placeSplitter } from "./placeNames.ts";

const split = placeSplitter(["Lime", "Lime Avenue", "Charlene Place", "North Charlene Place", "Ace", "Yerba Boulevard",
                             "A Street", "98th Street"]);
const names = (text) => split(text).filter((p) => typeof p !== "string").map((p) => p.name);

test("the text reads back whole, each name its own piece", () => {
  const text = "Yerba Boulevard runs N-S; the loop of Charlene Place encloses it.";
  const pieces = split(text);
  assert.equal(pieces.map((p) => (typeof p === "string" ? p : p.text)).join(""), text);
  assert.deepEqual(names(text), ["Yerba Boulevard", "Charlene Place"]);
});

test("the longest name wins: Lime Avenue over Lime, North Charlene Place over Charlene Place", () => {
  assert.deepEqual(names("on Lime Avenue, then Lime"), ["Lime Avenue", "Lime"]);
  assert.deepEqual(names("prior_6758 \"North Charlene Place\""), ["North Charlene Place"]);
});

test("whole words in any case, carrying the prior's spelling", () => {
  assert.deepEqual(split("along yerba boulevard.").filter((p) => typeof p !== "string"), [{ name: "Yerba Boulevard", text: "yerba boulevard" }]);
  assert.deepEqual(names("Limestone and Limes and Lime Avenues"), ["Lime"]);
});

test("a name shorter than four letters is a word, not a place", () => {
  assert.deepEqual(names("Ace Hardware's ace"), []);
  assert.deepEqual(placeSplitter([])("no names"), ["no names"]);
});

test("a name led by a single letter matches only as the prior capitalises it; any other in any case", () => {
  assert.deepEqual(names("the house on a street corner"), []);
  assert.deepEqual(names("A street runs north"), []);
  assert.deepEqual(names("east of A Street"), ["A Street"]);
  assert.deepEqual(names("along 98th street"), ["98th Street"]);
});

test("a name inside a longer word is not the name", () => {
  assert.deepEqual(names("Sublime views"), []);
});
