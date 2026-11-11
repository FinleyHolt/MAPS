/** The console says the model is looking.
 *
 * Under the last `leg` or `waypoint` row of a `flying` mission the
 * transcript says so, with a clock running from when the page first saw
 * that row. A source scan, so it fails open by construction: hence the floor.
 */

import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));
const SRC = readFileSync(here + "Transcript.tsx", "utf8");

test("the transcript is the size it is and still renders leg rows", () => {
  assert.ok(SRC.length > 8000, `Transcript.tsx is ${SRC.length} bytes`);
  assert.ok(SRC.includes('entry.kind === "leg"'), "Transcript.tsx no longer renders a leg row");
});

test("under the last leg or waypoint row of a flying mission, the model is looking, clock running", () => {
  assert.ok(SRC.includes("the model is looking at the pass · "), "the line is gone");
  assert.ok(/mission\.status === "flying" && \["leg", "waypoint"\]\.includes\(mission\.transcript\[last\]\?\.kind/.test(SRC),
            "the line is not gated on a flying mission whose last row is a leg or a waypoint");
  assert.ok(SRC.includes("{ row: last, at: Date.now() }"), "the clock is not keyed by the row index");
  assert.ok(SRC.includes("window.setInterval(() => setNow(Date.now()), 1000)"), "the clock does not run");
  assert.ok(SRC.includes("looking && saw?.row === last ?"), "the line is not tied to the row the page saw");
});
