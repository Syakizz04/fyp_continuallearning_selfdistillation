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
| `e1_sync/` | E1 manipulation check: 3 sync policies x 4 one-way delays on Redis. Fill: strong_lock 82.7%, escrow_quota 71.7%, eventual 100% (2081 units oversold). Matches the August run exactly | `bash experiments/run_model_free.sh` (step `e1`) | none (static prices) |
| `synthetic_audit/` | Honesty audit of the synthetic layer: conservation holds; elasticity is 71% prior and 29% data | step `audit` | none (data v3) |
| `e4_memory/` | E4 static memory: a full replay buffer is 3105 MB vs the SDFT teacher's 4.4 MB (706x) | step `e4` | rebuilt `base_v4` (same architecture as the original, so same bytes) |
| `forecast_signal/` | TFT h=0 vs 28-day rolling mean as the pricer's demand signal: MAE 1.539 vs 1.489 | `python -m experiments.forecast_signal_check` | rebuilt `base_v4` |
| `august_2026_original/` | The lost August 2026 FYP2 tables, recovered from the old laptop: E2 at none / strong_lock / escrow_quota (single seed), the 3-seed escrow_quota replicates (7, 42, 123) and E5. Result tables only; the E5 model weights are not included | `exp_staleness_cl`, `run_replicates.sh`, `exp_sdft_alpha`, run in August | **original** `base_v4` (calibration MASE 1.481). Today's code reproduces its 6-check walk to 2e-6 |
| `critic_compare/` | Pricing drift signals (revenue, critic and markup baseline) under 6 fault types, 17 non-overlapping trials, plus false alarms over 2013-2015 | `python -m experiments.critic_compare` (v2) | rebuilt `base_v4` |

The FYP1 drift-walk results predate this rule. They live in
`outputs/drift/results/`, force-added, and come from the *original* base TFT,
not the rebuilt one.

Before reproducing anything, check that the data and checkpoints on disk are the
ones these results came from:

```
python -m experiments.artifacts --verify
```
