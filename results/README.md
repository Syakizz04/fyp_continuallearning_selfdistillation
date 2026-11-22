# Committed results

`outputs/` is gitignored, and that is how the FYP2 E2 and replicate tables
were lost with the old laptop. From now on, a result does not count until
its table is here.

The rule:
- Copy each final summary or table from `outputs/...` into `results/<experiment>/`.
- Commit it in the same commit as the code change that produced it.
- Use small files only (CSV, JSON, logs). Checkpoints and data stay out of git;
  `artifacts.json` at the repo root records their hashes instead.

| dir | what | produced by | models |
|---|---|---|---|
| `rebuild_2026-11/` | Calibration of the rebuilt base TFT and PPO, the 6-check frozen-walk comparison with the old laptop, and the rebuild report | `experiments/rebuild_base_v4.py`, `experiments/compare_rebuild.py` | rebuilt `base`, `base_v4` |
| `critic_compare/` | Pricing drift signals (revenue, critic and markup baseline) under 6 fault types, 17 non-overlapping trials, plus false alarms over 2013-2015 | `python -m experiments.critic_compare` (v2) | rebuilt `base_v4` |

The FYP1 drift-walk results predate this rule. They live in
`outputs/drift/results/`, force-added, and come from the *original* base TFT,
not the rebuilt one.

Before reproducing anything, check that the data and checkpoints on disk are the
ones these results came from:

```
python -m experiments.artifacts --verify
```
