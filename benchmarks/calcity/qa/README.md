# Cal City mask QA

An audit of the prompt and mask pairs in the 1,100 reviewed Cal City episodes,
the set `data_prep/reviewed.py` derives and every Cal City experiment flies.

| File | Role |
|---|---|
| `mask_qa.csv` | One line per reviewed episode: corpus row, episode id, whether the row has a mask, the prompt, and blank `verdict` and `note` columns |
| `../data/calcity.json` | The corpus. Row `row` of this array is the pair: `descriptions[0]` is the prompt, `target_mask` the mask polygon |

## What to check

For each line, does the mask outline exactly the thing the prompt names, on
the benchmark's own imagery? Fill `verdict` with one of:

| Verdict | Meaning |
|---|---|
| `ok` | The mask covers the named thing and nothing else |
| `mask` | The mask is wrong (off target, cut short, spills over); fixed in `calcity.json` |
| `prompt` | The prompt is wrong or ambiguous for the imagery; fixed in `calcity.json` |
| `drop` | Not fixable; the row should leave the reviewed set |

26 lines read `has_mask = NO`: reviewed rows with no `target_mask` (mostly
"the house on the corner of" prompts). Each needs a mask drawn or a `drop`.

## How to fix

Edit the row in `benchmarks/calcity/data/calcity.json` in place; never
reorder or delete rows (the episode id carries the row index).

- `target_mask` is a list of `[x, y]` vertices in metres, in the task frame:
  x east, y north.
- A row is in the reviewed set only while its `target_positions[0]` sits
  within 0.2 m of a key in `data_prep/authoring/ground_truth.json`. If a fix
  moves the target point, update that key to match, or the row silently
  leaves the set.
- For `drop`, remove the row's key from `ground_truth.json`, and lower
  `REVIEWED_N` in `tests/test_reviewed_episodes.py` to match. Some targets
  carry several prompts (1,100 prompts over 982 targets), and removing a key
  drops every row on that target.

Before opening the PR:

```bash
python -m pytest benchmarks/calcity/tests
```

Commit the filled `mask_qa.csv` with the fixes and open a PR from this branch
(`ryan-mask-qa`) to `main`.
