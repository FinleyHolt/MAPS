# Cal City mask QA

An audit of the prompt and mask pairs in the 1,100 reviewed Cal City episodes,
the set `data_prep/reviewed.py` derives and every Cal City experiment flies.

| File | Role |
|---|---|
| `mask_qa.csv` | One line per reviewed episode: corpus row, episode id, where its shape comes from, the prompt, and blank `verdict` and `note` columns |
| `../data/calcity.json` | The corpus. Row `row` of this array is the pair: `descriptions[0]` is the prompt, and the shape is `target_mask` or a linked object |
| `objects.json` | Linked object contours, under `$SEARCH_CALCITY_DATA/data/cityrefer/` (built by `data_prep/build_objects_json.py`; not in git) |

## What to check

For each line, does the mask outline exactly the thing the prompt names, on
the benchmark's own imagery? Fill `verdict` with one of:

| Verdict | Meaning |
|---|---|
| `ok` | The mask covers the named thing and nothing else |
| `mask` | The mask is wrong (off target, cut short, spills over); fixed in `calcity.json` |
| `prompt` | The prompt is wrong or ambiguous for the imagery; fixed in `calcity.json` |
| `drop` | Not fixable; the row should leave the reviewed set |

The `shape` column says where each target's shape comes from:

| `shape` | Lines | Source |
|---|---|---|
| `target_mask` | 1,074 | A polygon drawn in the row |
| `object <ids>` | 26 | The contour of the linked object in `objects.json` (the "house on the corner of" prompts) |

A linked object's contour wins over a drawn `target_mask` in scoring
(`benchmarks/citynav/eval.py`).

## How to fix

Edit the row in `benchmarks/calcity/data/calcity.json` in place; never
reorder or delete rows (the episode id carries the row index).

- `target_mask` is a list of `[x, y]` vertices in metres, in the task frame:
  x east, y north.
- For an `object` row, fix the shape by pointing `object_ids` at the right
  object, or by emptying `object_ids` and drawing a `target_mask`.
- A row is in the reviewed set only while its `target_positions[0]` sits
  within 0.2 m of a key in `data_prep/authoring/ground_truth.json`. If a fix
  moves the target point, update that key to match, or the row silently
  leaves the set.
- For `drop`, remove the row's key from `ground_truth.json`, and lower
  `REVIEWED_N` in `tests/test_reviewed_episodes.py` to match. Some targets
  carry several prompts (1,100 prompts over 982 targets), and removing a key
  drops every row on that target.

Before committing:

```bash
python -m pytest benchmarks/calcity/tests
```

Commit the filled `mask_qa.csv` with the fixes.
