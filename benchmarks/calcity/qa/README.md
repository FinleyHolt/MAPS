# Cal City mask QA

This folder audits the prompt and mask pairs of the reviewed Cal City
episodes, the set that `data_prep/reviewed.py` derives and every Cal City
experiment flies.

| File | Contents |
|---|---|
| `mask_qa.csv` | One line per reviewed episode: corpus row, episode ID, where its shape comes from, the prompt, and blank `verdict` and `note` columns |
| `../data/calcity.json` | Corpus. Row `row` of this array is the pair: `descriptions[0]` is the prompt, and the shape is `target_mask` or a linked object |
| `objects.json` | Linked object contours, under `$SEARCH_CALCITY_DATA/data/cityrefer/` (built by `data_prep/build_objects_json.py`; not in git) |

## Check a line

For each line, decide whether the mask outlines exactly the thing the prompt
names, on the benchmark's own imagery. Fill `verdict` with one of the
following:

| Verdict | Meaning |
|---|---|
| `ok` | The mask covers the named thing and nothing else |
| `mask` | The mask is wrong (off target, cut short, spills over); fixed in `calcity.json` |
| `prompt` | The prompt is wrong or ambiguous for the imagery; fixed in `calcity.json` |
| `drop` | Not fixable; the row leaves the reviewed set |

The `shape` column says where each target's shape comes from:

| `shape` | Lines | Source |
|---|---|---|
| `target_mask` | 1,074 | Polygon drawn in the row |
| `object <ids>` | 26 | Contour of the linked object in `objects.json` (the "house on the corner of" prompts) |

In scoring, a linked object's contour wins over a drawn `target_mask`
(`benchmarks/citynav/eval.py`).

## Fix a row

1. Edit the row in `benchmarks/calcity/data/calcity.json` in place. Don't
   reorder or delete rows: the episode ID carries the row index.

   | Verdict | Fix |
   |---|---|
   | `mask`, drawn shape | Edit `target_mask`: a list of `[x, y]` vertices in metres, in the task frame (x east, y north). |
   | `mask`, `object` row | Point `object_ids` at the right object, or empty `object_ids` and draw a `target_mask`. |
   | `prompt` | Edit `descriptions[0]`. |
   | `drop` | Remove the row's key from `benchmarks/calcity/data_prep/authoring/ground_truth.json`, and lower `REVIEWED_N` in `benchmarks/calcity/tests/test_reviewed_episodes.py` to match. Some targets carry several prompts (1,100 prompts over 982 targets), and removing a key drops every row on that target. |

2. If the fix moves the target point, update its key in `ground_truth.json`
   to match. A row is in the reviewed set only while its
   `target_positions[0]` sits within 0.2 m of a key in that file; otherwise
   the row silently leaves the set.

3. Run the tests:

   ```bash
   python -m pytest benchmarks/calcity/tests
   ```

4. Commit the filled `mask_qa.csv` with the fixes.
