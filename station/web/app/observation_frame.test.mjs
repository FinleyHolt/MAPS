/** The map's features drawn on an observation by the page, off the
 * record's window.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { framePrior, imagePixel, PRIOR_COLOURS } from "./observation_frame.ts";

const WINDOW = [2990, 3990, 3022, 4014];
const SHAPE = [24, 32];

test("the window's corners land on the image's corner pixels", () => {
  assert.deepEqual(imagePixel([2990, 4014], WINDOW, SHAPE), [0, 0]);
  assert.deepEqual(imagePixel([3022, 3990], WINDOW, SHAPE), [31, 23]);
  assert.deepEqual(imagePixel([3006, 4002], WINDOW, SHAPE), [15.5, 11.5]);
});

test("a feature off the window is not drawn, a ring closes, a lone vertex is a node", () => {
  const prior = [
    ["building", [[3000, 4000], [3010, 4000], [3010, 4010], [3000, 4000]]],
    ["road", [[2000, 4000], [3500, 4000]]],
    ["node", [[3005, 4005]]],
    ["building", [[5000, 5000], [5010, 5000], [5010, 5010], [5000, 5000]]],
  ];
  const drawn = framePrior(prior, WINDOW, SHAPE);
  assert.deepEqual(drawn.map((f) => [f.colour, f.closed, f.points.length]), [
    [PRIOR_COLOURS.building, true, 4], [PRIOR_COLOURS.road, false, 2], [PRIOR_COLOURS.node, false, 1]]);
});
