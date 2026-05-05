"""
demand.py
---------
Generates the multi-layered demand signal for each SKU x Region combination.

Demand = base_demand
       * trend_component
       * seasonal_annual
       * seasonal_weekly
       * payday_boost
       * ramadan_multiplier
       * raya_multiplier
       * cny_multiplier
       * mega_sale_multiplier
       * viral_shock_multiplier
       * holiday_multiplier
       + gaussian_noise

Each component is stored as a separate column for interpretability
and thesis visualisation.
"""

import numpy as np
import pandas as pd
from datetime import date

from generators.calendar import (
    build_calendar_features,
    get_raya_demand_multiplier,
    get_cny_demand_multiplier,
    RAMADAN_WINDOWS,
    RAYA_DATES,
)


# ---------------------------------------------------------------------------
# VIRAL SHOCK INJECTION
# Injected at random intervals to simulate sudden social media spikes
# ---------------------------------------------------------------------------

VIRAL_SHOCKS = [
    # (start_date, duration_days, peak_magnitude, affected_skus)
    # 2023
    ("2023-04-03",  7,  3.2, ["P001"]),          # Viral Raya outfit trend
    ("2023-07-15",  5,  2.8, ["P002"]),          # Phone review goes viral
    ("2023-09-01", 10,  2.5, ["P003"]),          # Food influencer campaign
    ("2023-11-05",  3,  4.0, ["P002"]),          # Pre-11.11 hype
    # 2024
    ("2024-02-20",  6,  2.9, ["P001"]),          # CNY fashion trend
    ("2024-05-10",  8,  3.1, ["P003"]),          # Ramadan recipe viral
    ("2024-08-20",  5,  3.5, ["P002"]),          # Back to school electronics
    ("2024-10-25",  4,  3.8, ["P002"]),          # Pre-11.11 hype
    # 2025
    ("2025-02-14",  5,  2.7, ["P001", "P003"]), # Valentine + CNY convergence
    ("2025-06-01",  7,  3.3, ["P003"]),          # Raya food trend
    ("2025-09-15",  6,  3.0, ["P001", "P002"]), # Year-end shopping season early spike
    ("2025-10-28",  4,  4.2, ["P002"]),          # Pre-11.11 hype
]


def _build_viral_shock_map(date_range: pd.DatetimeIndex) -> dict:
    """
    Returns dict: {(date, sku_id): (shock_active, magnitude)}
    Uses a smooth bell-curve shape for the shock envelope.
    """
    shock_map = {}
    for start_str, duration, peak_mag, skus in VIRAL_SHOCKS:
        start = pd.Timestamp(start_str)
        for i in range(duration):
            d = start + pd.Timedelta(days=i)
            if d not in date_range:
                continue
            # Bell curve: peaks at midpoint of shock
            progress = i / max(duration - 1, 1)
            envelope = np.exp(-8 * (progress - 0.5) ** 2)
            magnitude = 1.0 + (peak_mag - 1.0) * envelope
            for sku in skus:
                shock_map[(d, sku)] = (1, round(magnitude, 4))
    return shock_map


def _trend(day_index: np.ndarray, slope: float) -> np.ndarray:
    """Slow exponential trend over the simulation horizon."""
    return 1.0 + slope * (day_index / 365.0)


def _seasonal_annual(day_of_year: np.ndarray, amplitude: float) -> np.ndarray:
    """Annual cosine seasonality — peaks mid-year."""
    return 1.0 + amplitude * np.cos(2 * np.pi * (day_of_year - 15) / 365)


def _seasonal_weekly(day_of_week: np.ndarray, amplitude: float) -> np.ndarray:
    """Weekly seasonality — weekend peaks."""
    return 1.0 + amplitude * np.sin(np.pi * day_of_week / 6)


def _payday_boost(day_of_month: np.ndarray, govt_boost: float, priv_boost: float) -> np.ndarray:
    """
    Bimodal payday effect.
    Government: 24th-26th. Private: 1st-3rd.
    """
    boost = np.ones(len(day_of_month))
    boost[np.isin(day_of_month, [24, 25, 26])] += govt_boost
    boost[np.isin(day_of_month, [1, 2, 3])]    += priv_boost
    return boost


def _ramadan_multiplier(
    dates: pd.DatetimeIndex,
    sku_config: dict,
    region_config: dict
) -> np.ndarray:
    """
    Ramadan sustained demand multiplier for food SKU.
    Other SKUs see slight suppression (consumers spending on food).
    """
    is_food = sku_config["category"] == "Food"
    region_weight = region_config.get("ramadan_weight", 1.0)
    multipliers = np.ones(len(dates))

    for year, (ram_start, ram_end) in RAMADAN_WINDOWS.items():
        for i, dt in enumerate(dates):
            d = dt.date()
            if ram_start <= d <= ram_end:
                day_in_ramadan = (d - ram_start).days + 1
                total_days = (ram_end - ram_start).days + 1
                # Ramp up over Ramadan: demand builds as Raya approaches
                ramp = 1.0 + 0.6 * (day_in_ramadan / total_days)
                if is_food:
                    food_mag = sku_config.get("primary_shock_magnitude", 4.0)
                    multipliers[i] = ramp * food_mag * region_weight
                else:
                    multipliers[i] = 0.92  # slight suppression for non-food

    return multipliers


def generate_demand(
    date_range: pd.DatetimeIndex,
    calendar_df: pd.DataFrame,
    sku_config: dict,
    region_config: dict,
    seasonality_config: dict,
    payday_config: dict,
    rng: np.random.Generator,
    viral_shock_map: dict,
) -> pd.DataFrame:
    """
    Generate the full demand time series for one SKU x Region combination.
    Returns a DataFrame with all component columns and final demand.
    """
    n = len(date_range)
    sku_id = sku_config["id"]
    base = sku_config["base_daily_demand"] * region_config["demand_multiplier"]
    is_electronics = sku_config["category"] == "Electronics"
    is_food        = sku_config["category"] == "Food"
    is_fashion     = sku_config["category"] == "Fashion"

    day_index    = np.arange(n)
    day_of_year  = calendar_df["date"].dt.dayofyear.values
    day_of_week  = calendar_df["day_of_week"].values
    day_of_month = calendar_df["day_of_month"].values

    # --- Component arrays ---
    trend_vals    = _trend(day_index, sku_config["trend_slope"])
    seas_annual   = _seasonal_annual(day_of_year, seasonality_config["annual_amplitude"])
    seas_weekly   = _seasonal_weekly(day_of_week, seasonality_config["weekly_amplitude"])
    payday_vals   = _payday_boost(
        day_of_month,
        payday_config["government_boost"],
        payday_config["private_boost"]
    )
    ramadan_vals  = _ramadan_multiplier(date_range, sku_config, region_config)

    # --- Raya multiplier ---
    raya_vals = np.ones(n)
    if is_fashion or is_food:
        raya_weight = region_config.get("raya_weight", 1.0)
        for i, dt in enumerate(date_range):
            d = dt.date()
            raya_date = RAYA_DATES.get(d.year)
            if raya_date:
                days_to_raya = (raya_date - d).days
                base_mult = get_raya_demand_multiplier(days_to_raya, sku_config)
                # Scale from 1 by the region weight
                raya_vals[i] = 1.0 + (base_mult - 1.0) * raya_weight

    # --- CNY multiplier ---
    cny_vals = np.ones(n)
    if is_electronics or is_fashion:
        cny_weight = region_config.get("cny_weight", 1.0)
        from generators.calendar import CNY_DATES, CNY_PRE_WINDOW_DAYS, CNY_POST_DROP_DAYS
        for i, dt in enumerate(date_range):
            d = dt.date()
            cny_date = CNY_DATES.get(d.year)
            if cny_date:
                days_to_cny = (cny_date - d).days
                base_mult = get_cny_demand_multiplier(days_to_cny, sku_config)
                cny_vals[i] = 1.0 + (base_mult - 1.0) * cny_weight

    # --- Mega sale multiplier ---
    mega_vals = np.ones(n)
    mega_sensitivity = sku_config.get("mega_sale_sensitivity", 1.0)
    mega_adoption    = region_config.get("mega_sale_adoption", 1.0)
    is_1111 = calendar_df["mega_sale_name"].values == "11.11"
    is_1212 = calendar_df["mega_sale_name"].values == "12.12"

    for i in range(n):
        base_mag = calendar_df["mega_sale_base_magnitude"].values[i]
        if base_mag > 0:
            # Electronics gets extra boost on 11.11
            if is_electronics and is_1111[i]:
                adj_mag = base_mag * mega_sensitivity * mega_adoption
            elif is_electronics and is_1212[i]:
                adj_mag = base_mag * 0.85 * mega_sensitivity * mega_adoption
            else:
                adj_mag = base_mag * mega_sensitivity * mega_adoption
            mega_vals[i] = max(1.0, adj_mag)

    # --- Viral shock multiplier ---
    viral_vals = np.ones(n)
    viral_active = np.zeros(n, dtype=int)
    for i, dt in enumerate(date_range):
        key = (dt, sku_id)
        if key in viral_shock_map:
            active, mag = viral_shock_map[key]
            viral_vals[i]  = mag
            viral_active[i] = active

    # --- Federal holiday multiplier ---
    holiday_vals = np.ones(n)
    from generators.calendar import FEDERAL_HOLIDAYS
    for i, dt in enumerate(date_range):
        d = dt.date()
        if d in FEDERAL_HOLIDAYS:
            _, mult = FEDERAL_HOLIDAYS[d]
            if mult > 0:
                holiday_vals[i] = mult

    # --- Combine all multiplicative components ---
    combined = (
        base
        * trend_vals
        * seas_annual
        * seas_weekly
        * payday_vals
        * ramadan_vals
        * raya_vals
        * cny_vals
        * mega_vals
        * viral_vals
        * holiday_vals
    )

    # --- Gaussian noise (multiplicative) ---
    noise_std = seasonality_config["noise_std"]
    noise = rng.normal(loc=1.0, scale=noise_std, size=n)
    noise = np.clip(noise, 0.7, 1.5)  # bound to prevent extreme outliers

    demand_raw = combined * noise
    demand_int = np.maximum(0, np.round(demand_raw).astype(int))

    # --- Assemble output DataFrame ---
    df = calendar_df.copy()
    df["product_id"]            = sku_config["id"]
    df["product_name"]          = sku_config["name"]
    df["product_category"]      = sku_config["category"]
    df["region_id"]             = region_config["id"]
    df["region_name"]           = region_config["name"]

    # Ground-truth components (for debugging / thesis diagrams)
    df["base_demand"]           = base
    df["trend_component"]       = np.round(trend_vals, 6)
    df["seasonal_annual"]       = np.round(seas_annual, 6)
    df["seasonal_weekly"]       = np.round(seas_weekly, 6)
    df["payday_multiplier"]     = np.round(payday_vals, 6)
    df["ramadan_multiplier"]    = np.round(ramadan_vals, 6)
    df["raya_multiplier"]       = np.round(raya_vals, 6)
    df["cny_multiplier"]        = np.round(cny_vals, 6)
    df["mega_sale_multiplier"]  = np.round(mega_vals, 6)
    df["viral_shock_active"]    = viral_active
    df["viral_shock_multiplier"]= np.round(viral_vals, 6)
    df["holiday_multiplier"]    = np.round(holiday_vals, 6)
    df["noise_factor"]          = np.round(noise, 6)
    df["demand_before_noise"]   = np.round(combined, 2)

    # Target variable
    df["demand"]                = demand_int

    return df
