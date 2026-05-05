"""
features.py
-----------
Computes all derived features on top of the raw demand signal:
- Lag features
- Rolling statistics
- CL task labels
- Shock summary columns
- Normalised / encoded features
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# LAG FEATURES
# ---------------------------------------------------------------------------

LAG_DAYS = [1, 7, 14, 30]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged demand columns per product x region group.
    Lags are computed within each (product_id, region_id) group
    so that lag_7 for product P001 in WCP does not bleed into P002.
    """
    df = df.sort_values(["product_id", "region_id", "date"]).copy()

    for lag in LAG_DAYS:
        col_name = f"demand_lag_{lag}"
        df[col_name] = (
            df.groupby(["product_id", "region_id"])["demand"]
            .shift(lag)
        )

    return df


# ---------------------------------------------------------------------------
# ROLLING STATISTICS
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean and standard deviation features.
    """
    df = df.sort_values(["product_id", "region_id", "date"]).copy()

    windows = [7, 14, 30]
    for w in windows:
        df[f"demand_rolling_mean_{w}"] = (
            df.groupby(["product_id", "region_id"])["demand"]
            .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
        )
        df[f"demand_rolling_std_{w}"] = (
            df.groupby(["product_id", "region_id"])["demand"]
            .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).std().fillna(0))
        )

    # Demand momentum: ratio of 7-day to 30-day rolling mean
    df["demand_momentum"] = (
        df["demand_rolling_mean_7"] / df["demand_rolling_mean_30"].replace(0, np.nan)
    ).fillna(1.0).round(4)

    return df


# ---------------------------------------------------------------------------
# SHOCK SUMMARY COLUMNS
# ---------------------------------------------------------------------------

def add_shock_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate individual shock flags into unified shock columns
    for easier use during model training.
    """
    df = df.copy()

    # Any festive or shock event active
    df["any_shock_active"] = (
        (df["is_pre_raya_window"] == 1) |
        (df["is_pre_cny_window"] == 1)  |
        (df["is_mega_sale"] == 1)        |
        (df["viral_shock_active"] == 1)  |
        (df["is_ramadan"] == 1)
    ).astype(int)

    # Composite shock type label (priority order for overlaps)
    def classify_shock(row):
        if row["is_mega_sale"] == 1:
            return f"mega_sale_{row['mega_sale_name']}"
        elif row["viral_shock_active"] == 1:
            return "viral"
        elif row["is_pre_raya_window"] == 1:
            return "pre_raya"
        elif row["is_post_raya_window"] == 1:
            return "post_raya"
        elif row["is_pre_cny_window"] == 1:
            return "pre_cny"
        elif row["is_post_cny_window"] == 1:
            return "post_cny"
        elif row["is_ramadan"] == 1:
            return "ramadan"
        elif row["is_federal_holiday"] == 1:
            return "federal_holiday"
        else:
            return "normal"

    df["shock_type"] = df.apply(classify_shock, axis=1)

    # Effective demand multiplier (combined effect, for reference)
    df["effective_multiplier"] = np.round(
        df["trend_component"]
        * df["seasonal_annual"]
        * df["seasonal_weekly"]
        * df["ramadan_multiplier"]
        * df["raya_multiplier"]
        * df["cny_multiplier"]
        * df["mega_sale_multiplier"]
        * df["viral_shock_multiplier"]
        * df["holiday_multiplier"],
        4
    )

    # Days since last shock (any type)
    df = df.sort_values(["product_id", "region_id", "date"])
    df["days_since_last_shock"] = (
        df.groupby(["product_id", "region_id"])["any_shock_active"]
        .transform(lambda x: x[::-1].cumsum()[::-1].where(x == 0, 0))
    )

    return df


# ---------------------------------------------------------------------------
# CONTINUAL LEARNING TASK LABELS
# ---------------------------------------------------------------------------

def add_task_labels(df: pd.DataFrame, task_configs: list) -> pd.DataFrame:
    """
    Assign CL task_id and task_name to each row based on date ranges
    defined in config.yaml.
    """
    df = df.copy()
    df["task_id"]   = -1
    df["task_name"] = "unassigned"

    for task in task_configs:
        mask = (
            (df["date"] >= pd.Timestamp(task["start_date"])) &
            (df["date"] <= pd.Timestamp(task["end_date"]))
        )
        df.loc[mask, "task_id"]   = task["task_id"]
        df.loc[mask, "task_name"] = task["task_name"]

    return df


# ---------------------------------------------------------------------------
# SOCIAL MEDIA INDEX (exogenous proxy signal)
# ---------------------------------------------------------------------------

def add_social_media_index(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Synthetic social media trend index (0-1).
    Peaks before viral shocks and mega sales with some lead time,
    simulating how social buzz precedes actual purchases.
    Includes baseline random fluctuation.
    """
    df = df.copy()
    n = len(df)

    # Base random social noise
    base = rng.beta(a=2, b=5, size=n)  # right-skewed, mostly low values

    # Boost pre-mega-sale (3 days before)
    mega_boost = np.zeros(n)
    for i in range(n):
        if df["is_mega_sale"].iloc[i] == 1:
            # boost the 3 days before
            start = max(0, i - 3)
            mega_boost[start:i+1] = np.linspace(0.2, 0.6, i+1-start)

    # Boost aligned with viral shock
    viral_boost = np.minimum(
        (df["viral_shock_multiplier"].values - 1.0) / 3.0,
        0.8
    )

    smi = np.clip(base + mega_boost + viral_boost, 0.0, 1.0)
    df["social_media_index"] = np.round(smi, 4)

    return df


# ---------------------------------------------------------------------------
# COMPETITOR ACTIVITY INDEX
# ---------------------------------------------------------------------------

def add_competitor_index(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Synthetic competitor promotional activity index (0-1).
    Higher during mega sale periods (competitors also run sales).
    """
    df = df.copy()
    n = len(df)

    base = rng.beta(a=1.5, b=4, size=n)

    # Competitors also ramp up on mega sale days
    mega_boost = df["mega_sale_base_magnitude"].values / 15.0  # normalise to 0-1

    ci = np.clip(base + mega_boost, 0.0, 1.0)
    df["competitor_activity_index"] = np.round(ci, 4)

    return df


# ---------------------------------------------------------------------------
# MARKETING SPEND (simulated internal signal)
# ---------------------------------------------------------------------------

def add_marketing_spend(
    df: pd.DataFrame,
    sku_config: dict,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Simulated daily marketing spend (MYR).
    Higher during festive seasons and mega sales.
    Base spend varies by SKU.
    """
    df = df.copy()
    base_spend = {
        "P001": 500,
        "P002": 1200,
        "P003": 300,
    }.get(sku_config["id"], 500)

    # Scale spend with mega sale and festive periods
    scale = (
        1.0
        + 2.0 * (df["is_mega_sale"].values)
        + 1.0 * (df["is_pre_raya_window"].values)
        + 0.8 * (df["is_pre_cny_window"].values)
        + 0.3 * (df["is_ramadan"].values)
    )

    noise = rng.normal(loc=1.0, scale=0.15, size=len(df))
    noise = np.clip(noise, 0.6, 1.6)

    spend = base_spend * scale * noise
    df["marketing_spend_myr"] = np.round(spend, 2)

    return df


# ---------------------------------------------------------------------------
# MASTER FEATURE PIPELINE
# ---------------------------------------------------------------------------

def build_all_features(
    df: pd.DataFrame,
    sku_config: dict,
    task_configs: list,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Run all feature engineering steps in order.
    Input df is the raw output from demand.py.
    """
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_shock_summary(df)
    df = add_task_labels(df, task_configs)
    df = add_social_media_index(df, rng)
    df = add_competitor_index(df, rng)
    df = add_marketing_spend(df, sku_config, rng)

    # Final sort
    df = df.sort_values(["product_id", "region_id", "date"]).reset_index(drop=True)

    return df
