"""Render the drift pipeline's final summary table to PNG.

Standalone (pandas + matplotlib only; no torch / pipeline import). Reads the
efficiency metric CSV and writes a styled table image to
outputs/drift/plots/final_summary_table.png.

Styled to match the synthetic pipeline's final_summary_table.png: navy header,
white bold header text, light-blue SDFT row highlight, centered bold title.

    python drift_pipeline/summary_table.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "drift" / "results"
OUT = ROOT / "outputs" / "drift" / "plots" / "final_summary_table.png"

HEADER = "#2C3E50"      # navy header (matches synthetic table)
HILITE = "#EBF5FB"      # light-blue highlight for the SDFT row
LABEL = {"frozen": "Frozen", "ewc": "EWC", "replay": "Replay", "sdft": "SDFT"}
ORDER = ["frozen", "ewc", "replay", "sdft"]

COLUMNS = ["Method", "FC Retrains", "RL Retrains", "Total Retrains",
           "Walk MASE ↓", "Profit Index ↑", "FC Forgetting ↓", "RL Forgetting ↓"]


DASH = "—"


def _rows() -> list[list[str]]:
    eff = pd.read_csv(RES / "metrics_efficiency.csv").set_index("arm")
    # RL forgetting (profit drop on base-era probes) is computed in the forgetting
    # table; older efficiency CSVs may not carry it, so join it in if missing.
    if "forgetting_profit_base_era" not in eff.columns:
        fg = pd.read_csv(RES / "metrics_forgetting.csv").set_index("arm")
        if "forgetting_profit_base_era" in fg.columns:
            eff = eff.join(fg[["forgetting_profit_base_era"]], how="left")
    eff = eff[~eff.index.isin(["periodic"])]
    out = []
    for arm in [a for a in ORDER if a in eff.index]:
        r = eff.loc[arm]
        if arm == "frozen":
            # Anchor: never retrains, and profit/forgetting are 1.0/0.0 by
            # definition (measured vs itself). Only Walk MASE is informative.
            out.append(["Frozen", DASH, DASH, DASH,
                        f"{r['walk_mase_mean']:.3f}", DASH, DASH, DASH])
            continue
        out.append([
            LABEL.get(arm, arm),
            f"{int(r['n_fc_retrains'])}",
            f"{int(r['n_rl_retrains'])}",
            f"{int(r['n_retrains_total'])}",
            f"{r['walk_mase_mean']:.3f}",
            f"{r['profit_index_vs_frozen']:.3f}",
            f"{r['forgetting_mase_base_era']:+.3f}",
            f"{r['forgetting_profit_base_era']:+.3f}",
        ])
    return out


def render() -> Path:
    cell_text = _rows()

    fig, ax = plt.subplots(figsize=(17, 3.8))
    ax.axis("off")

    table = ax.table(cellText=cell_text, colLabels=COLUMNS,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Header style
    for j in range(len(COLUMNS)):
        table[0, j].set_facecolor(HEADER)
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight the SDFT row
    for i, row in enumerate(cell_text, 1):
        if "SDFT" in row[0]:
            for j in range(len(COLUMNS)):
                table[i, j].set_facecolor(HILITE)

    plt.title("Drift Experiment Results Summary",
              fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    p = render()
    print(f"wrote {p}")
