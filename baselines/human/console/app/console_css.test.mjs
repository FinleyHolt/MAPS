/** Every rule in console.css names only classes the console renders.
 *
 * Run with `npm test` in baselines/human/console.
 */
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";

const here = new URL(".", import.meta.url);
const read = (name) => readFileSync(new URL(name, here), "utf8");

test("no rule in console.css is dead", () => {
  const sources = [...readdirSync(here).filter((f) => /\.tsx?$/.test(f)).map(read), read("../src/main.tsx")].join("\n");
  const rendered = new Set();
  for (const [, a, b, c] of sources.matchAll(/"([^"\n]*)"|'([^'\n]*)'|`([^`]*)`/g)) {
    for (const token of (a ?? b ?? c).split(/[\s${}]+/)) rendered.add(token);
  }
  // The list's chips: `cap-chip is-${status}`, one per status capacity.py reports.
  for (const s of ["answered", "skipped", "waiting", "not_started", "open", "searching", "ready"]) rendered.add(`is-${s}`);
  const live = (c) => rendered.has(c) || c.startsWith("maplibregl-");
  const css = read("console.css").replace(/\/\*[\s\S]*?\*\//g, "").replace(/@keyframes[^{]*\{(?:[^{}]*\{[^}]*\})*\s*\}/g, "");
  const dead = [];
  for (const [, head] of css.matchAll(/([^{}@;]+)\{[^{}]*\}/g)) {
    const selectors = head.split(",").map((s) => s.trim());
    const classes = (s) => [...s.replace(/:not\([^)]*\)/g, "").matchAll(/\.([a-zA-Z_][\w-]*)/g)].map((m) => m[1]);
    dead.push(...selectors.filter((s) => !classes(s).every(live)));
  }
  assert.deepEqual(dead, []);
});
