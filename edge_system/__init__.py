"""
Deployed continuous-edge-learning system (FYP2).

Runtime layer on top of the offline experiment packages: edge nodes serve TFT
forecasts and PPO prices from models held **in-process**, detect drift locally,
retrain locally via SDFT, and coordinate shared inventory through an escrow
reservation service.

Subpackages
-----------
inventory : shared stock pool, escrow/bounded-counter algorithm, sync policies
edge      : per-channel node - local inference, local drift, local retrain
control   : model registry and drift/retrain event log
sim       : simulated clock, order generation, injectable network conditions
"""

__all__ = ["config"]
