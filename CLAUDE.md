# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Final Year Project comparing continual-learning (CL) methods on two model types — **demand forecasting** (Temporal Fusion Transformer) and **RL-based dynamic pricing** (PPO) — over a 6-task sequence built from a synthetic Malaysian e-commerce dataset. The research question is catastrophic forgetting: tasks alternate baseline ↔ mega-sale regimes, and SDFT (self-distillation) is the proposed method benchmarked against naive fine-tuning, EWC, and replay/RECALL.

## Commands

```powershell
# Setup (Windows / PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run the full experiment as a script (writes to outputs/)
python -m initial_pipeline.experiment_runner

# Regenerate the synthetic datasets (from dataset_generator/synthetic_data/)
python pipeline.py                 # both CSVs + validation plots
python pipeline.py --stage1-only   # demand_forecasting.csv only
python pipeline.py --stage2-only   # rl_environment.csv (needs stage 1 first)
```

There is no pytest suite or linter configured. **The smoke test is `notebooks/smoke_test.ipynb`** — it imports the pipeline, then mutates `CONFIG` in place to shrink the run (2 tasks, 1 epoch, tiny timesteps, fewer CL methods) so the full loop executes in minutes. Use it to validate changes end-to-end. `notebooks/experiment.ipynb` is the full interactive run; `notebooks/colab.ipynb` is the Colab/Drive variant.

There is no installed package / `sys.path` setup — run modules from the **project root** so `initial_pipeline` / `hybrid_pipeline` imports resolve, or `sys.path.insert(0, project_dir)` as the notebooks do.

## Architecture

Each `*_pipeline/` directory is a self-contained package with the same three-module layout. Modules import each other relatively (`from .core_pipeline import ...`), so a package is run as `python -m <pkg>.experiment_runner`.

- **`core_pipeline.py`** — the single source of truth `CONFIG` dict (paths, the 6 tasks, TFT/PPO/CL hyperparameters, hardware). Importing it is side-effect-light: it seeds RNGs, detects device, and creates output dirs, but does **not** load data or train. Data flow: `prepare_data()` → `load_and_clean()` → `build_tft_dataframe()` / `build_rl_features()` → `split_tasks()` (date-range slices into 6 task DataFrames).
- **`trainers.py`** — model components and shared CL machinery: `CLTFT` (TFT subclass that injects EWC/SDFT penalties in `training_step`), `DynamicPricingEnv` (Gymnasium env, 13-dim state / 11 discrete price tiers), metric functions (`compute_mase/smape`, `evaluate_rl`, `compute_bwt_fwt`), the replay buffers / EWC engine / teacher stores, and the global `LOGGER` + `CKPT_MGR` singletons.
- **`experiment_runner.py`** — orchestration + plotting. Per-method trainer functions live in `FORECAST_TRAINERS` / `RL_TRAINERS` dicts. `run_experiment()` is the nested loop: `for model_type → for cl_method → for task: train, evaluate on all seen tasks, checkpoint`. CL methods carry state **across tasks** via module-level dicts (`_fc_model_state`, `_rl_recall_buf`, etc.), reset per model_type. Then `build_cl_summary()` / `generate_all_plots()` compute BWT/FWT and render figures.

### The two pipeline packages

- **`initial_pipeline/`** is the **base** package (the four base CL methods: naive / ewc / replay-or-recall / sdft). It writes to `outputs/`. (There was previously a byte-identical `fyp_pipeline/` copy — it has been removed; `initial_pipeline` is now canonical for the base methods.)
- **`hybrid_pipeline/`** is the extended variant — it adds **drift-adaptive** methods (`compute_distribution_drift`, `adaptive_strengths`, `train_forecast_drift_adaptive_replay_ewc`, `adaptive_drift`) that scale EWC/distillation/replay strength by measured task distribution shift. It writes to `outputs/hybrid/` instead of `outputs/`, and uses `compute_forgetting` in place of the base `compute_bwt_fwt`.

`initial_pipeline` and `hybrid_pipeline` share the same `CLTFT`/`_sdft_loss`/`training_step` core, so a change to that shared machinery in one usually needs mirroring in the other. When asked to "change the pipeline," clarify which package — base or hybrid.

## Conventions and gotchas

- **CONFIG is mutated, not subclassed.** Smoke tests and `configure_vast_ai()` adjust the experiment by editing `CONFIG` in place after import. Helpers re-read `CONFIG` at call time (e.g. `min_tft_rows()`) precisely so these post-import mutations take effect — keep that pattern; don't snapshot CONFIG values at import.
- **`# ─── Cell: ... ───` comment banners** mark what were originally notebook cells. The `.py` modules are the source of truth; the banners are just structural markers.
- **TFT data requirements** are strict: `time_idx` must be a consecutive integer per group, and each series needs `encoder_length + prediction_length` rows (`min_tft_rows()`) or it is silently dropped by `filter_tft_eval_frame`. Short eval tasks return NaN metrics rather than crashing — this is deliberate.
- **Forgetting is intended to be observable.** Tasks 1/3/5 (baseline) vs 2/4/6 (mega-sale) alternate by design; comparing Task 5 against Task 1 is the primary backward-transfer measurement. Don't "fix" the regime alternation.
- **Hardware knobs via env vars:** `FYP_TORCH_COMPILE=1` (torch.compile, off by default — unstable with repeated Lightning fits), `FYP_NUM_WORKERS` (default 0 to avoid notebook/Vast.ai fork issues). `configure_vast_ai(data_dir, output_dir, require_gpu=True)` repoints paths for rented GPU boxes.
- **Outputs** (checkpoints, results CSVs, logs, plots) land under `outputs/` (or `outputs/hybrid/`). `outputs/results/all_metrics.csv` is the long-form record every plot/table is derived from.

## Dataset generator

`dataset_generator/synthetic_data/` produces the two CSVs consumed by `data/processed/`. **All parameters live in `config/config.yaml`** (seed 42, SKUs, regions, seasonality) — edit there, not in generator code. See `dataset_generator/README.md` for the full column manifest and the demand-composition model. Stage 1 = `demand_forecasting.csv` (66 cols), Stage 2 = `rl_environment.csv` (86 cols, adds price/elasticity/inventory/reward). The pipeline only consumes a subset, enforced by `demand_required_columns()` / `RL_REQUIRED_COLUMNS` in `core_pipeline.py`.
