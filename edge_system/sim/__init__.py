"""
Simulation driver: replays the M5 walk-forward window as a live feed.

Deliberately kept free of torch imports so E1 (which needs no models) can run
the whole stack on a laptop with nothing loaded.
"""

from .clock import SimClock, Tick
from .network import NetworkConditions, Partitioned, SimNetwork
from .order_gen import Order, OrderGenerator

__all__ = [
    "SimClock", "Tick",
    "SimNetwork", "NetworkConditions", "Partitioned",
    "OrderGenerator", "Order",
]
