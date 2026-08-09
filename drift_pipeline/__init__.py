"""Drift-triggered continual-learning pipeline (M5 dataset).

Train once on a base period, walk forward through later years, detect performance
drift against realized ground truth, and retrain only when drift fires — comparing
replay / SDFT as retraining strategies against frozen, naive and periodic baselines.

Self-contained: the model and CL machinery live in `.trainers`, vendored here so
the deployed FYP2 stack does not import its engine from an FYP1 variant package.
"""
