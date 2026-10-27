/** MAPS's final reply as the console reads it, and the step each tool call it cites ran in.
 *
 * Run with `npm test` in baselines/human/console.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { replyBlocks, stepOfCall } from "./reply.ts";

const reply = "\n**Answer: `prior_2777`** — the big building.\n\nEvidence gathered:\n"
  + "- `intersection(\"North Loop Boulevard\", \"95th Street\")` → a crossing at [7018, 6378].\n"
  + "* the paving *faces south*, 2 * 3 stars\n\nThe drone's last flight\nis on this building.";

test("paragraphs and bullets, with bold, code and emphasis as spans", () => {
  const blocks = replyBlocks(reply);
  assert.deepEqual(blocks.map((b) => b.kind), ["p", "p", "li", "li", "p"]);
  assert.deepEqual(blocks[0].spans, [
    { text: "Answer: ", bold: true }, { text: "prior_2777", bold: true, code: true }, { text: " — the big building." }]);
  assert.deepEqual(blocks[2].spans[0], { text: "intersection(\"North Loop Boulevard\", \"95th Street\")", code: true });
  assert.deepEqual(blocks[3].spans, [{ text: "the paving " }, { text: "faces south", em: true }, { text: ", 2 * 3 stars" }]);
  assert.equal(blocks[4].spans[0].text, "The drone's last flight\nis on this building.");
});

test("a marker never closed reads as the text it is, and markup never becomes HTML", () => {
  assert.deepEqual(replyBlocks("**half `open <b>x</b>").at(0).spans, [{ text: "**half `open <b>x</b>" }]);
  assert.deepEqual(replyBlocks(""), []);
});

test("a cited call is the first step whose code ran it, quotes and spacing aside", () => {
  const said = ["look_for(\"house\")", "r1 = intersection('North Loop Boulevard',  '95th Street')\nr1", "look(r1)"];
  assert.equal(stepOfCall("intersection(\"North Loop Boulevard\", \"95th Street\")", said), 1);
  assert.equal(stepOfCall("look(r1)", said), 2);
  // A plain word is not a call; a call no step ran has no step.
  assert.equal(stepOfCall("house", said), null);
  assert.equal(stepOfCall("explore(r9)", said), null);
});
