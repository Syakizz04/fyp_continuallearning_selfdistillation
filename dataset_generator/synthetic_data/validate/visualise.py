"""
visualise.py
------------
Validation plots for the synthetic dataset.
Generates a multi-panel figure per SKU showing:
  1. Full demand curve with annotated shock events and task boundaries
  2. Seasonal decomposition (trend, weekly, annual components)
  3. Price vs realized demand scatter (elasticity verification)
  4. Regional demand comparison
  5. CL task distribution summary

Run: python validate/visualise.py
Outputs saved to output/plots/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ---------------------------------------------------------------------------
DEMAND_CSV = Path("output/demand_forecasting.csv")
RL_CSV     = Path("output/rl_environment.csv")
PLOT_DIR   = Path("output/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TASK_COLORS = {
    1: "#4C72B0", 2: "#DD8452", 3: "#55A868",
    4: "#C44E52", 5: "#8172B3", 6: "#937860",
}

REGION_COLORS = {
    "WCP": "#2196F3",
    "ECP": "#FF9800",
    "BRN": "#4CAF50",
}

SKU_LABELS = {
    "P001": "Baju Raya Fashion",
    "P002": "Consumer Electronics",
    "P003": "Festive Food & Groceries",
}

# ---------------------------------------------------------------------------

def load_data():
    print("Loading datasets...")
    demand_df = pd.read_csv(DEMAND_CSV, parse_dates=["date"])
    rl_df     = pd.read_csv(RL_CSV,     parse_dates=["date"])
    print(f"  Demand dataset : {len(demand_df):,} rows")
    print(f"  RL dataset     : {len(rl_df):,} rows")
    return demand_df, rl_df


# ---------------------------------------------------------------------------
# PLOT 1: Full demand curve per SKU (all regions overlaid)
# ---------------------------------------------------------------------------

def plot_demand_overview(demand_df: pd.DataFrame, sku_id: str):
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    fig.suptitle(
        f"Demand Overview — {SKU_LABELS[sku_id]} ({sku_id})",
        fontsize=15, fontweight="bold", y=0.98
    )

    skudf = demand_df[demand_df["product_id"] == sku_id].copy()

    # --- Panel 1: Raw demand by region ---
    ax = axes[0]
    for rid, rname in [("WCP", "West Coast"), ("ECP", "East Coast"), ("BRN", "Borneo")]:
        rdf = skudf[skudf["region_id"] == rid]
        ax.plot(rdf["date"], rdf["demand"],
                color=REGION_COLORS[rid], alpha=0.85, linewidth=0.9, label=rname)

    # Annotate task boundaries using WCP
    wcp = skudf[skudf["region_id"] == "WCP"].sort_values("date")
    _annotate_tasks(ax, wcp)
    _annotate_mega_sales(ax, wcp)
    ax.set_ylabel("Units Sold (demand)", fontsize=10)
    ax.set_title("Raw Demand — All Regions", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Demand components (WCP only) ---
    ax = axes[1]
    wcp = wcp.sort_values("date")
    ax.plot(wcp["date"], wcp["base_demand"] * wcp["trend_component"],
            label="Trend", linewidth=1.5, color="#333333")
    ax.plot(wcp["date"], wcp["demand_before_noise"],
            label="Demand before noise", linewidth=0.8, color="#E91E63", alpha=0.7)
    ax.fill_between(
        wcp["date"],
        wcp["demand_before_noise"] * 0.9,
        wcp["demand_before_noise"] * 1.1,
        alpha=0.15, color="#E91E63", label="±10% noise band"
    )
    _annotate_tasks(ax, wcp)
    ax.set_ylabel("Units", fontsize=10)
    ax.set_title("Demand Signal Decomposition (WCP)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Shock multipliers ---
    ax = axes[2]
    ax.plot(wcp["date"], wcp["raya_multiplier"],   label="Raya",     color="#FF5722", linewidth=1.2)
    ax.plot(wcp["date"], wcp["cny_multiplier"],    label="CNY",      color="#FF9800", linewidth=1.2)
    ax.plot(wcp["date"], wcp["ramadan_multiplier"],label="Ramadan",  color="#9C27B0", linewidth=1.2)
    ax.plot(wcp["date"], wcp["mega_sale_multiplier"], label="Mega Sale", color="#2196F3", linewidth=1.0)
    ax.plot(wcp["date"], wcp["viral_shock_multiplier"], label="Viral",  color="#4CAF50", linewidth=1.0)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Demand Multiplier", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_title("Shock Multipliers Over Time (WCP)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOT_DIR / f"demand_overview_{sku_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# PLOT 2: Seasonal decomposition
# ---------------------------------------------------------------------------

def plot_seasonal_decomposition(demand_df: pd.DataFrame, sku_id: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Seasonal Decomposition — {SKU_LABELS[sku_id]} (WCP Only)",
        fontsize=13, fontweight="bold"
    )

    wcp = demand_df[
        (demand_df["product_id"] == sku_id) &
        (demand_df["region_id"] == "WCP")
    ].sort_values("date")

    # Trend
    axes[0,0].plot(wcp["date"], wcp["trend_component"], color="#333", linewidth=1.5)
    axes[0,0].set_title("Trend Component")
    axes[0,0].set_ylabel("Multiplier")
    axes[0,0].grid(True, alpha=0.3)

    # Annual seasonality
    axes[0,1].plot(wcp["date"], wcp["seasonal_annual"], color="#2196F3", linewidth=1.2)
    axes[0,1].set_title("Annual Seasonality")
    axes[0,1].grid(True, alpha=0.3)

    # Weekly seasonality (average by day of week)
    weekly_avg = wcp.groupby("day_of_week")["seasonal_weekly"].mean()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[1,0].bar(days, weekly_avg.values, color="#FF9800", edgecolor="black", linewidth=0.5)
    axes[1,0].set_title("Average Weekly Seasonality")
    axes[1,0].set_ylabel("Multiplier")
    axes[1,0].grid(True, alpha=0.3, axis="y")

    # Monthly average demand
    wcp["month_label"] = wcp["date"].dt.to_period("M").astype(str)
    monthly = wcp.groupby("month")["demand"].mean()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    axes[1,1].bar(month_names, monthly.values, color="#4CAF50", edgecolor="black", linewidth=0.5)
    axes[1,1].set_title("Average Monthly Demand")
    axes[1,1].set_ylabel("Units")
    axes[1,1].tick_params(axis="x", rotation=45)
    axes[1,1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = PLOT_DIR / f"seasonal_decomposition_{sku_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# PLOT 3: Price-Demand elasticity scatter
# ---------------------------------------------------------------------------

def plot_price_demand(rl_df: pd.DataFrame, sku_id: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(
        f"Price vs Realized Demand — {SKU_LABELS[sku_id]}\n"
        "(Elasticity Verification — downward slope confirms price sensitivity)",
        fontsize=12, fontweight="bold"
    )

    skudf = rl_df[rl_df["product_id"] == sku_id]

    for ax, (rid, rname) in zip(axes, [("WCP","West Coast"),("ECP","East Coast"),("BRN","Borneo")]):
        rdf = skudf[skudf["region_id"] == rid]
        scatter = ax.scatter(
            rdf["current_price"], rdf["realized_demand"],
            c=rdf["elasticity_coefficient"], cmap="RdYlGn_r",
            alpha=0.4, s=8, vmin=-2.8, vmax=-1.0
        )
        # Trend line
        if len(rdf) > 10:
            z = np.polyfit(rdf["current_price"], rdf["realized_demand"], 1)
            p = np.poly1d(z)
            xline = np.linspace(rdf["current_price"].min(), rdf["current_price"].max(), 100)
            ax.plot(xline, p(xline), "r--", linewidth=1.5, label="OLS trend")

        ax.set_title(rname, fontsize=11)
        ax.set_xlabel("Current Price (MYR)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Realized Demand (units)")
    cbar = fig.colorbar(scatter, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Elasticity Coefficient", fontsize=9)

    plt.tight_layout()
    path = PLOT_DIR / f"price_demand_scatter_{sku_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# PLOT 4: CL Task distribution summary
# ---------------------------------------------------------------------------

def plot_task_summary(demand_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Continual Learning Task Structure", fontsize=13, fontweight="bold")

    # Task row counts
    task_counts = demand_df.groupby(["task_id", "task_name"]).size().reset_index(name="rows")
    task_counts["label"] = task_counts["task_id"].astype(str) + ": " + task_counts["task_name"]
    colors = [TASK_COLORS.get(t, "#999") for t in task_counts["task_id"]]
    axes[0].barh(task_counts["label"], task_counts["rows"], color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Number of Rows")
    axes[0].set_title("Row Count per CL Task")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Mean demand per task per SKU
    task_demand = demand_df.groupby(["task_id", "product_id"])["demand"].mean().reset_index()
    for sku_id in ["P001","P002","P003"]:
        sub = task_demand[task_demand["product_id"] == sku_id]
        axes[1].plot(
            sub["task_id"], sub["demand"],
            marker="o", linewidth=2, label=SKU_LABELS[sku_id]
        )
    axes[1].set_xticks(list(TASK_COLORS.keys()))
    axes[1].set_xlabel("Task ID")
    axes[1].set_ylabel("Mean Daily Demand (units, all regions)")
    axes[1].set_title("Mean Demand Shift Across CL Tasks")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOT_DIR / "cl_task_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# PLOT 5: Regional comparison heatmap
# ---------------------------------------------------------------------------

def plot_regional_heatmap(demand_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Monthly Average Demand by Region (All SKUs)", fontsize=13, fontweight="bold")

    demand_df["year_month"] = demand_df["date"].dt.to_period("M").astype(str)

    for ax, (sku_id, sku_name) in zip(axes, SKU_LABELS.items()):
        pivot = demand_df[demand_df["product_id"] == sku_id].pivot_table(
            index="year_month", columns="region_id", values="demand", aggfunc="mean"
        )
        im = ax.imshow(pivot.T.values, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(pivot.columns)))
        ax.set_yticklabels(pivot.columns)
        # Show every 3rd month label to avoid crowding
        xticks = range(0, len(pivot.index), 3)
        ax.set_xticks(list(xticks))
        ax.set_xticklabels([pivot.index[i] for i in xticks], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{sku_name}\n({sku_id})", fontsize=10)
        plt.colorbar(im, ax=ax, label="Avg Daily Demand")

    plt.tight_layout()
    path = PLOT_DIR / "regional_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _annotate_tasks(ax, wcp_df: pd.DataFrame):
    """Shade background by CL task."""
    tasks = wcp_df.groupby("task_id")["date"].agg(["min","max"])
    for task_id, row in tasks.iterrows():
        color = TASK_COLORS.get(task_id, "#cccccc")
        ax.axvspan(row["min"], row["max"], alpha=0.07, color=color)
        ax.axvline(row["min"], color=color, linewidth=0.8, linestyle="--", alpha=0.6)
        y_pos = ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 1
        ax.text(row["min"], y_pos, f"T{task_id}", fontsize=7, color=color, va="top")


def _annotate_mega_sales(ax, wcp_df: pd.DataFrame):
    """Mark 11.11 and 12.12 with vertical lines."""
    big_sales = wcp_df[wcp_df["mega_sale_name"].isin(["11.11","12.12"])]
    for _, row in big_sales.iterrows():
        ax.axvline(row["date"], color="#E91E63", linewidth=1.0, linestyle=":", alpha=0.8)


# ---------------------------------------------------------------------------
# DATASET STATISTICS SUMMARY
# ---------------------------------------------------------------------------

def print_summary(demand_df: pd.DataFrame, rl_df: pd.DataFrame):
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"\nDemand Forecasting Dataset")
    print(f"  Total rows       : {len(demand_df):,}")
    print(f"  Date range       : {demand_df['date'].min().date()} → {demand_df['date'].max().date()}")
    print(f"  SKUs             : {demand_df['product_id'].nunique()}")
    print(f"  Regions          : {demand_df['region_id'].nunique()}")
    print(f"  Columns          : {len(demand_df.columns)}")
    print(f"  Null values      : {demand_df.isnull().sum().sum():,}")

    print(f"\nRL Environment Dataset")
    print(f"  Total rows       : {len(rl_df):,}")
    print(f"  Columns          : {len(rl_df.columns)}")
    print(f"  Null values      : {rl_df.isnull().sum().sum():,}")

    print(f"\nDemand Statistics by SKU x Region (demand_forecasting.csv)")
    summary = demand_df.groupby(["product_id","region_id"])["demand"].agg(
        ["mean","std","min","max"]
    ).round(1)
    print(summary.to_string())

    print(f"\nCL Task Row Distribution")
    task_dist = demand_df.groupby(["task_id","task_name"]).size()
    print(task_dist.to_string())

    print(f"\nShock Type Distribution (WCP, P001 only)")
    shock_dist = demand_df[
        (demand_df["region_id"]=="WCP") &
        (demand_df["product_id"]=="P001")
    ]["shock_type"].value_counts()
    print(shock_dist.to_string())
    print("="*60 + "\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    demand_df, rl_df = load_data()
    print_summary(demand_df, rl_df)

    print("\nGenerating plots...")
    for sku_id in ["P001","P002","P003"]:
        print(f"\n  SKU: {sku_id} — {SKU_LABELS[sku_id]}")
        plot_demand_overview(demand_df, sku_id)
        plot_seasonal_decomposition(demand_df, sku_id)
        plot_price_demand(rl_df, sku_id)

    plot_task_summary(demand_df)
    plot_regional_heatmap(demand_df)

    print(f"\nAll plots saved to: {PLOT_DIR.resolve()}")


if __name__ == "__main__":
    main()
