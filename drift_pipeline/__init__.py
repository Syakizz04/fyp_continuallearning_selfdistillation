"""Drift-triggered continual-learning pipeline (M5 dataset).

Train once on a base period, walk forward through later years, detect performance
drift against realized ground truth, and retrain only when drift fires — comparing
EWC / replay / SDFT as retraining strategies against frozen and periodic baselines.

Reuses the model/CL machinery from `hybrid_pipeline.trainers`; only the config,
M5 data prep, drift monitor, and metrics are new here.
"""
