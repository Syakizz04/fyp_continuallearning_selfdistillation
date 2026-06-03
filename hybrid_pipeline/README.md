# Hybrid Continual Learning Pipeline

This package runs the second experiment without changing the archived
`initial_pipeline` package.

Methods compared for both forecasting and RL:

- `naive`: standard sequential fine-tuning.
- `recall_ewc`: task-balanced replay/RECALL plus EWC regularization.
- `adaptive_drift`: replay-free EWC plus distillation, with regularization
  strength reduced when distribution drift is higher.
- `drift_adaptive_replay_ewc`: task-balanced replay/RECALL plus EWC, with
  replay and EWC strength adjusted by detected drift.

Run from the repo root:

```bash
python -m hybrid_pipeline.experiment_runner
```

Notebook imports should use `hybrid_pipeline` instead of `fyp_pipeline`.
Outputs are written under `outputs/hybrid/`.
