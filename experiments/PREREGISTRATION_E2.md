# E2 re-run: pre-registered analysis

Written 2026-11-22, **before** any E2 result exists for the rebuilt models
(see CLAUDE.md, "The November 2026 rebuild"). The point is that the rule for
claiming "arm A beats arm B" is fixed before the numbers can influence it. If
it changes later, change it in a new commit that says why, and report both.

## What is run

- **Models:** the rebuilt `base_v4`. Run `python -m experiments.artifacts --verify` first; it must pass.
- **Data:** `processed_m5_v4`.
- **Censoring level:** `escrow_quota` (fill rate 0.717, from E1), one level.
- **Arms:** `frozen`, `naive`, `replay`, `sdft`. `periodic` is optional and secondary.
- **SDFT alpha:** 0.50 (`core_pipeline.py`'s `sdft_alpha`). The E5 alpha sweep is exploratory; see below.
- **Seeds: 5**, `42 123 7 2024 99`, the `run_replicates.sh` default. Every arm runs at every seed. Do not drop or add a seed after seeing results. If a cell fails, re-run that cell; do not replace the seed.

## Metrics

- **Primary:** `forgetting_mase_base_era`. Lower is better. This is the CL question: does the method keep the pre-drift regime?
- **Secondary:** `walk_mase_mean` (adaptation). Also report retrain count and memory bytes (E4).

## Primary comparison and decision rule

Comparisons are **paired within seed**: delta = arm − reference at the same seed. Implemented in `experiments/compare_replicates.py`.

1. **Primary contrast:** `sdft` − `replay` on the primary metric. This is the headline: is replay-free SDFT a substitute for replay?
2. **Control contrasts:**
   - `sdft` − `naive`: does the CL mechanism help, beyond retraining at the right moment?
   - `replay` − `naive`.

A contrast is **"consistent"**, and may be claimed as a difference, only if both hold:
- (a) the delta has the same sign at all 5 seeds; and
- (b) the smallest |delta| exceeds the spread (max − min) of the 5 deltas.

Otherwise it is **"not separable"**, and the report says the arms cannot be ranked at this sample size. That is a finding, not a failure.

Descriptive statistics are reported alongside every contrast and never used as the decision: mean delta, a t-based 95% interval over seeds, and an exact two-sided sign-test p-value. With 5 seeds a unanimous sign gives p = 0.0625. That is exactly why (b) exists and why no p < 0.05 claim is made from 5 seeds.

**Sanity check:** `frozen` must be identical across seeds (spread < 1e-6). If it is not, the run is broken; stop and fix it before reading anything else.

## What each outcome means for the write-up

| outcome on the primary contrast | claim |
|---|---|
| consistent, sdft < replay | SDFT forgets less than replay under endogenous censoring, while storing no past data (E4: 1× vs ~706×) |
| consistent, sdft > replay | Replay retains better. The E4 cost argument stands on its own: what retention does the 706× buy? |
| not separable | SDFT and replay cannot be told apart at this sample size. Combined with E4, SDFT reaches comparable retention at about 1/706 of the memory. This is the replay-free argument without a superiority claim. |

Separately: if `sdft` − `naive` is not separable, no claim is made that the SDFT *mechanism* helps, whatever the primary contrast says.

## Exploratory (not covered by this rule)

- E5, the `sdft_alpha` sweep (0.7, 0.85, 0.95).
- Other censoring levels.
- `periodic`.

Report these as exploratory. If one of them looks interesting, it gets its own replicate run with its own pre-registration before it is claimed.
