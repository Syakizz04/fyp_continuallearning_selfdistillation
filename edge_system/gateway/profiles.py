"""
What kind of product each SKU is, measured from the data the simulator replays.

M5 is anonymised: Walmart never published what FOODS_3_080 is or what
FOODS_1/2/3 mean, so a name or category ("Dairy") would be invented. What the
data does carry is how each item sells, and that is what a manager needs to
read a price recommendation: how much it costs, whether demand is steady,
whether it moves with SNAP (US food-assistance payment) days, and how strongly
the simulated shoppers react to price.

Read from `rl_environment.csv`, the file the order generator replays, so the
tags describe the demo's own behaviour. The elasticity is the one the
simulator uses, which the synthetic-layer audit found to be mostly prior: the
"price-sensitive" tag says how the demo's customers react, not a measured fact
about Walmart's.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional

from ..config import SYSTEM_CONFIG

# Thresholds, chosen so each tag separates the demo's ten products rather than
# applying to all of them.
VALUE_BELOW = 1.0          # $ median price
PREMIUM_FROM = 3.0
STEADY_CV = 0.5            # day-to-day std / mean
VOLATILE_CV = 1.0
SNAP_LIFT = 1.2            # mean demand on SNAP days / other days
SENSITIVE_E = -1.5         # elasticity at or below
INSENSITIVE_E = -1.0       # elasticity at or above

_LOCK = threading.Lock()
_CACHE: Dict[str, Optional[Dict[str, Dict]]] = {}


def tags_for(price: float, cv: float, snap_lift: float, elasticity: float) -> List[str]:
    tags = ["Value" if price < VALUE_BELOW else "Premium" if price >= PREMIUM_FROM else "Everyday"]
    if cv < STEADY_CV:
        tags.append("Steady")
    elif cv > VOLATILE_CV:
        tags.append("Volatile")
    if snap_lift >= SNAP_LIFT:
        tags.append("SNAP-driven")
    if elasticity <= SENSITIVE_E:
        tags.append("Price-sensitive")
    elif elasticity >= INSENSITIVE_E:
        tags.append("Price-insensitive")
    return tags


def _build(path: Path) -> Dict[str, Dict]:
    import pandas as pd

    df = pd.read_csv(path, usecols=["product_id", "realized_demand", "base_price",
                                    "snap", "elasticity_coefficient"])
    out: Dict[str, Dict] = {}
    for sku, g in df.groupby("product_id"):
        d = g["realized_demand"]
        mean = float(d.mean())
        on, off = g.loc[g["snap"] == 1, "realized_demand"], g.loc[g["snap"] == 0, "realized_demand"]
        snap_lift = float(on.mean() / off.mean()) if len(on) and len(off) and off.mean() > 0 else 1.0
        cv = float(d.std() / mean) if mean > 0 else 0.0
        price = float(g["base_price"].median())
        el = float(g["elasticity_coefficient"].iloc[0])
        out[str(sku)] = {"price": round(price, 2), "mean_daily": round(mean, 2),
                         "cv": round(cv, 2), "snap_lift": round(snap_lift, 2),
                         "elasticity": round(el, 2), "tags": tags_for(price, cv, snap_lift, el)}
    return out


def profiles() -> Dict[str, Dict]:
    """sku -> profile; {} if the data file is missing or unreadable (tags are
    context, never a reason for the products page to fail)."""
    path = Path(SYSTEM_CONFIG["paths"]["data_dir"]) / "rl_environment.csv"
    key = str(path)
    with _LOCK:
        if key not in _CACHE:
            try:
                _CACHE[key] = _build(path)
            except Exception:  # noqa: BLE001
                _CACHE[key] = None
        return _CACHE[key] or {}
