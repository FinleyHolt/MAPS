/** Which map the console is looking at, and what an empty picture book says.
 *
 * A bench has one map per block, so the basemap and every placement follow
 * the mission on screen: a run under review over the seat, a row about to
 * fly over the mission still in the seat.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { emptyBook, viewedMap } from "./mission.ts";

const about = { map: "birmingham_block_0" };
const seat = { map_name: "cambridge_block_2" };
const row = { map: "cambridge_block_9" };

test("a run under review names the map, over everything else", () => {
  assert.equal(viewedMap({ replay: { map_name: "birmingham_block_5" }, mission: seat, picked: row, about }),
               "birmingham_block_5");
});

test("a reviewed run with no map is on no map, not on the seat's", () => {
  assert.equal(viewedMap({ replay: { map_name: null }, mission: seat, about }), null);
});

test("a row about to fly shows its own map while another mission sits in the seat", () => {
  assert.equal(viewedMap({ starting: row, mission: seat, picked: row, about }), "cambridge_block_9");
});

test("the seat's map, then the row last clicked, then the station's default", () => {
  assert.equal(viewedMap({ mission: seat, picked: row, about }), "cambridge_block_2");
  assert.equal(viewedMap({ mission: null, picked: row, about }), "cambridge_block_9");
  assert.equal(viewedMap({ about }), "birmingham_block_0");
  assert.equal(viewedMap({}), null);
});

test("a bench run's empty picture book points at the transcript; a flying twin's says not yet", () => {
  assert.match(emptyBook({ kind: "replay", status: "ended" }), /IN THE TRANSCRIPT/);
  assert.match(emptyBook({ kind: "bench", status: "flying" }), /IN THE TRANSCRIPT/);
  assert.match(emptyBook({ kind: "twin", status: "flying" }), /YET/);
  assert.match(emptyBook(null), /YET/);
});
