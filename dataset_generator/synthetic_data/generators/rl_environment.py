"""
rl_environment.py
-----------------
Stage 2 dataset generator.
Reads demand_forecasting.csv and adds:
- Price columns (current, base, competitor)
- Elasticity-adjusted realized demand
- Inventory tracking
- Reward computation (revenue + profit)
- RL state/action/reward columns

This runs SEPARATELY after Stage 1 is complete, reading its CSV output.
The demand_forecast column here represents what the forecasting model
WOULD have predicted — during actual training, this will be replaced
by real model outputs. For dataset generation, we use a noisy version
of actual demand as a proxy forecast.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# PRICE DYNAMICS
# ---------------------------------------------------------------------------

def _generate_competitor_price(
    base_price: float,
    n: int,
    is_mega_sale: np.ndarray,
    rng: np.random.Generator,
    config: dict,
) -> np.ndarray:
    """
    Competitor price: tracks our base price with noise,
    undercuts occasionally, and drops during mega sales.
    """
    noise_std = config["price_tracking_noise"]
    undercut_prob = config["undercut_probability"]
    undercut_low, undercut_high = config["undercut_range"]

    prices = np.zeros(n)
    price = base_price

    for i in range(n):
        # Random walk around base price
        price += rng.normal(0, noise_std * base_price)
        price = np.clip(price, base_price * 0.80, base_price * 1.20)

        # Occasional undercut
        if rng.random() < undercut_prob:
            undercut = rng.uniform(undercut_low, undercut_high)
            price = price * (1 - undercut)

        # Mega sale: competitors slash prices
        if is_mega_sale[i] == 1:
            price = base_price * rng.uniform(0.70, 0.85)

        prices[i] = round(price, 2)

    return prices


def _compute_elasticity(
    base_elasticity: float,
    is_mega_sale: np.ndarray,
    is_ramadan: np.ndarray,
    is_pre_raya: np.ndarray,
    viral_active: np.ndarray,
    region_elasticity_mult: float,
) -> np.ndarray:
    """
    Dynamic elasticity that shifts across demand regimes.
    - Mega sale: consumers expect discounts, more elastic
    - Pre-Raya / Ramadan: necessity purchase, less elastic
    - Viral trend: hype-driven, less elastic
    """
    elasticity = np.full(len(is_mega_sale), base_elasticity * region_elasticity_mult)

    # More elastic during mega sales (shoppers are hunting deals)
    elasticity = np.where(is_mega_sale == 1, elasticity * 1.35, elasticity)

    # Less elastic during Ramadan and pre-Raya (must-buy)
    elasticity = np.where(is_ramadan == 1, elasticity * 0.65, elasticity)
    elasticity = np.where(is_pre_raya == 1, elasticity * 0.70, elasticity)

    # Less elastic during viral trends (hype > price sensitivity)
    elasticity = np.where(viral_active == 1, elasticity * 0.75, elasticity)

    return np.round(elasticity, 4)


def _price_adjusted_demand(
    base_demand: np.ndarray,
    current_price: np.ndarray,
    base_price: float,
    elasticity: np.ndarray,
) -> np.ndarray:
    """
    Apply price elasticity to compute realized demand.
    Uses constant elasticity demand model:
        realized_demand = base_demand * (current_price / base_price) ^ elasticity
    """
    price_ratio = current_price / base_price
    adjustment  = np.power(np.maximum(price_ratio, 0.01), elasticity)
    realized    = base_demand * adjustment
    return np.maximum(0, np.round(realized).astype(int))


def _simulate_inventory(
    realized_demand: np.ndarray,
    initial_stock: int,
    reorder_threshold: int,
    reorder_quantity: int,
) -> tuple:
    """
    Simple inventory simulation with reorder logic.
    Returns (inventory_levels, stockout_flags).
    """
    n = len(realized_demand)
    inventory   = np.zeros(n, dtype=int)
    stockout    = np.zeros(n, dtype=int)
    stock       = initial_stock

    for i in range(n):
        # Reorder if below threshold
        if stock <= reorder_threshold:
            stock += reorder_quantity

        sold = min(stock, realized_demand[i])
        if realized_demand[i] > stock:
            stockout[i] = 1

        stock -= sold
        inventory[i] = stock

    return inventory, stockout


# ---------------------------------------------------------------------------
# RL EPISODE STRUCTURE
# ---------------------------------------------------------------------------

def _assign_episode_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign episode IDs by task boundary.
    Each CL task = one episode for the RL agent.
    """
    df = df.copy()
    df["episode_id"] = df["task_id"].astype(str) + "_" + df["region_id"] + "_" + df["product_id"]
    return df


# ---------------------------------------------------------------------------
# PROXY DEMAND FORECAST
# ---------------------------------------------------------------------------

def _proxy_forecast(
    demand: np.ndarray,
    rng: np.random.Generator,
    noise_pct: float = 0.12,
) -> tuple:
    """
    Proxy forecast = actual demand + noise.
    Simulates what a trained forecasting model might output.
    During real training, this column is replaced by actual model output.

    Returns (forecast, uncertainty) where uncertainty is the noise magnitude.
    """
    noise = rng.normal(loc=1.0, scale=noise_pct, size=len(demand))
    noise = np.clip(noise, 0.75, 1.30)
    forecast = np.maximum(0, np.round(demand * noise).astype(int))
    uncertainty = np.abs(noise - 1.0)
    return forecast, np.round(uncertainty, 4)


# ---------------------------------------------------------------------------
# MAIN RL ENVIRONMENT GENERATOR
# ---------------------------------------------------------------------------

def generate_rl_environment(
    demand_df: pd.DataFrame,
    sku_configs: list,
    region_configs: list,
    competitor_config: dict,
    inventory_config: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate the RL environment dataset from the demand forecasting dataset.
    Processes each (product_id x region_id) group independently.
    """
    sku_lookup    = {s["id"]: s for s in sku_configs}
    region_lookup = {r["id"]: r for r in region_configs}
    all_groups    = []

    groups = demand_df.groupby(["product_id", "region_id"])

    for (sku_id, region_id), grp in groups:
        grp = grp.sort_values("date").copy().reset_index(drop=True)
        n   = len(grp)

        sku_cfg    = sku_lookup[sku_id]
        region_cfg = region_lookup[region_id]

        base_price = sku_cfg["base_price"]
        unit_cost  = sku_cfg["unit_cost"]
        price_min  = sku_cfg["price_min"]
        price_max  = sku_cfg["price_max"]
        elast_base = sku_cfg["elasticity_base"]
        elast_mult = region_cfg["elasticity_multiplier"]

        # --- Competitor price ---
        competitor_price = _generate_competitor_price(
            base_price=base_price,
            n=n,
            is_mega_sale=grp["is_mega_sale"].values,
            rng=rng,
            config=competitor_config,
        )

        # --- Dynamic elasticity ---
        elasticity = _compute_elasticity(
            base_elasticity=elast_base,
            is_mega_sale=grp["is_mega_sale"].values,
            is_ramadan=grp["is_ramadan"].values,
            is_pre_raya=grp["is_pre_raya_window"].values,
            viral_active=grp["viral_shock_active"].values,
            region_elasticity_mult=elast_mult,
        )

        # --- RL agent pricing policy (initialised to base price with noise) ---
        # During actual RL training, the agent sets this. Here we simulate
        # a naive baseline policy with slight random variation.
        current_price = np.clip(
            base_price * rng.uniform(0.90, 1.10, size=n),
            price_min,
            price_max,
        ).round(2)

        # Mega sale: simulate agent also discounting
        mega_mask = grp["is_mega_sale"].values == 1
        current_price[mega_mask] = np.clip(
            base_price * rng.uniform(0.72, 0.85, size=mega_mask.sum()),
            price_min,
            price_max,
        ).round(2)

        price_ratio = np.round(current_price / base_price, 4)
        price_gap   = np.round(current_price - competitor_price, 2)
        price_lag_1 = np.concatenate([[base_price], current_price[:-1]])
        price_change = np.round(current_price - price_lag_1, 2)

        # --- Realized demand (after price elasticity applied) ---
        realized_demand = _price_adjusted_demand(
            base_demand=grp["demand"].values.astype(float),
            current_price=current_price,
            base_price=base_price,
            elasticity=elasticity,
        )

        # --- Proxy demand forecast ---
        demand_forecast, forecast_uncertainty = _proxy_forecast(
            grp["demand"].values,
            rng=rng,
        )

        # --- Inventory ---
        initial_stock    = int(sku_cfg["base_daily_demand"] * inventory_config["initial_stock_days"])
        reorder_threshold = int(sku_cfg["base_daily_demand"] * inventory_config["reorder_threshold_days"])
        reorder_qty       = int(sku_cfg["base_daily_demand"] * inventory_config["reorder_quantity_days"])

        inventory_level, stockout_flag = _simulate_inventory(
            realized_demand=realized_demand,
            initial_stock=initial_stock,
            reorder_threshold=reorder_threshold,
            reorder_quantity=reorder_qty,
        )

        inventory_turnover = np.round(
            realized_demand / np.maximum(inventory_level, 1), 4
        )

        # --- Reward computation ---
        revenue      = np.round(current_price * realized_demand, 2)
        profit_margin = np.round((current_price - unit_cost) * realized_demand, 2)

        # Composite reward: profit margin penalised for stockouts
        stockout_penalty = stockout_flag * 0.15 * revenue
        reward           = np.round(profit_margin - stockout_penalty, 2)

        # --- Episode / done flags ---
        task_ids   = grp["task_id"].values
        done_flags = np.zeros(n, dtype=int)
        for i in range(n - 1):
            if task_ids[i] != task_ids[i + 1]:
                done_flags[i] = 1
        done_flags[-1] = 1  # last row always done

        # --- Assemble group DataFrame ---
        grp["current_price"]          = current_price
        grp["base_price"]             = base_price
        grp["price_ratio"]            = price_ratio
        grp["competitor_price"]       = competitor_price
        grp["price_gap"]              = price_gap
        grp["price_lag_1"]            = price_lag_1
        grp["price_change"]           = price_change
        grp["elasticity_coefficient"] = elasticity
        grp["demand_forecast"]        = demand_forecast
        grp["demand_forecast_uncertainty"] = forecast_uncertainty
        grp["realized_demand"]        = realized_demand
        grp["inventory_level"]        = inventory_level
        grp["inventory_turnover"]     = inventory_turnover
        grp["stockout_flag"]          = stockout_flag
        grp["revenue_myr"]            = revenue
        grp["profit_margin_myr"]      = profit_margin
        grp["reward"]                 = reward
        grp["action_price"]           = current_price   # the action taken
        grp["done"]                   = done_flags

        all_groups.append(grp)

    result = pd.concat(all_groups, ignore_index=True)
    result = _assign_episode_ids(result)
    result = result.sort_values(["product_id", "region_id", "date"]).reset_index(drop=True)

    return result
