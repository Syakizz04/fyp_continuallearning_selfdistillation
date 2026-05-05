# Synthetic Malaysian E-Commerce Dataset Generator

**Final Year Project — Continuous Edge Learning for Mitigating Catastrophic Forgetting in Dynamic Pricing and Real-Time Inventory Sync**

This pipeline generates a fully synthetic, Malaysia-specific e-commerce dataset for training and evaluating continual learning (CL) methods in demand forecasting and reinforcement learning-based dynamic pricing. It produces two datasets: a demand forecasting dataset (Stage 1) and an RL environment dataset (Stage 2).

---

## Project Context

This dataset supports a comparison of four continual learning methods:

| Method | Type | Role |
|---|---|---|
| Naïve Fine-tuning | Baseline | Lower bound — no forgetting mitigation |
| EWC (Elastic Weight Consolidation) | Regularisation | Mid-tier baseline |
| RECALL | Replay-based | Current SOTA anchor |
| SDFT (Self-Distillation Fine-Tuning) | Distillation-based | Proposed method |

The dataset is designed so that catastrophic forgetting is **observable and measurable** — models trained on Task 1 baseline demand patterns will degrade when exposed to Task 2 mega sale shocks, and backward transfer can be measured when demand returns to baseline in Tasks 3 and 5.

---

## Directory Structure

```
synthetic_data/
│
├── config/
│   └── config.yaml              # All parameters — edit here, not in code
│
├── generators/
│   ├── calendar.py              # Malaysian holidays, Ramadan/Raya/CNY dates, mega sales
│   ├── demand.py                # Multi-layer demand signal generator
│   ├── features.py              # Lag features, rolling stats, CL task labels
│   └── rl_environment.py        # Price elasticity, inventory, reward computation
│
├── validate/
│   └── visualise.py             # Diagnostic plots for dataset validation
│
├── pipeline.py                  # Master entry point — run this
│
└── output/
    ├── demand_forecasting.csv   # Stage 1 output
    ├── rl_environment.csv       # Stage 2 output
    └── plots/                   # Validation figures
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install numpy pandas pyyaml matplotlib
```

### 2. Run the full pipeline

```bash
python pipeline.py
```

This generates both CSV datasets and all validation plots in one command (~20 seconds).

### 3. Other run modes

```bash
# Stage 1 only (demand forecasting dataset)
python pipeline.py --stage1-only

# Stage 2 only (RL dataset — Stage 1 must exist first)
python pipeline.py --stage2-only

# Re-run validation plots without regenerating data
python pipeline.py --validate

# Print full column manifest after generation
python pipeline.py --manifest
```

---

## Dataset Specifications

| Parameter | Value |
|---|---|
| Simulation period | 2023-01-01 → 2025-12-31 (1,096 days) |
| Time granularity | Daily |
| SKUs | 3 (Fashion, Electronics, Food) |
| Regions | 3 (West Coast, East Coast, Borneo) |
| Total rows (Stage 1) | 9,864 |
| Total columns (Stage 1) | 66 |
| Total columns (Stage 2) | 86 |
| Random seed | 42 (reproducible) |

### Products (SKUs)

| ID | Name | Category | Base Price | Primary Shock | Peak Multiplier |
|---|---|---|---|---|---|
| P001 | Baju Raya Fashion | Fashion | MYR 89 | Hari Raya Aidilfitri | 8× pre-Raya |
| P002 | Consumer Electronics | Electronics | MYR 1,299 | 11.11 Mega Sale | 10× |
| P003 | Festive Food & Groceries | Food | MYR 18 | Ramadan | 4× sustained |

### Regions

| ID | Name | States Covered | Demand Scale | Elasticity |
|---|---|---|---|---|
| WCP | West Coast Peninsular | Perlis, Kedah, Penang, Perak, Selangor, KL, Putrajaya, N9, Melaka, Johor | 1.0× (reference) | Moderate |
| ECP | East Coast Peninsular | Kelantan, Terengganu, Pahang | 0.6× | High (price sensitive) |
| BRN | Borneo | Sabah, Sarawak, Labuan | 0.75× | Moderate-high |

> **Methodology note:** Weekend structure is standardised to Saturday–Sunday across all regions. State-level holidays beyond federal public holidays are not modelled. These are stated simplifying assumptions documented in the thesis methodology.

### Continual Learning Task Sequence

| Task | Name | Period | Description |
|---|---|---|---|
| 1 | Baseline_2023_H1 | Jan–May 2023 | Normal baseline with CNY and pre-Raya |
| 2 | MegaSale_2023 | Jun–Dec 2023 | Mega sale regime including 11.11 and 12.12 |
| 3 | Baseline_2024_H1 | Jan–May 2024 | Baseline with shifted lunar calendar (drift +1) |
| 4 | MegaSale_2024 | Jun–Dec 2024 | Mega sale regime 2024 |
| 5 | Baseline_2025_H1 | Jan–May 2025 | Second lunar drift — primary backward transfer window |
| 6 | MegaSale_2025 | Jun–Dec 2025 | Mega sale regime 2025 |

Tasks 1, 3, and 5 are structurally identical in regime type but differ due to lunar calendar drift (Raya shifts ~10 days earlier each year). Comparing model performance on Task 5 against Task 1 is the primary backward transfer measurement.

---

## Dataset Columns

### Stage 1 — `demand_forecasting.csv` (66 columns)

**Identity**
- `date`, `product_id`, `product_name`, `product_category`, `region_id`, `region_name`

**Temporal features**
- `day_of_week`, `day_of_month`, `week_of_year`, `month`, `quarter`
- `is_weekend`, `is_month_start`, `is_month_end`

**Malaysian calendar features**
- `is_federal_holiday`, `holiday_name`
- `is_government_payday_window`, `is_private_payday_window`
- `is_ramadan`, `ramadan_day`
- `days_to_raya`, `is_pre_raya_window`, `is_post_raya_window`
- `days_to_cny`, `is_pre_cny_window`, `is_post_cny_window`
- `is_mega_sale`, `mega_sale_name`, `mega_sale_base_magnitude`
- `is_school_holiday`
- `viral_shock_active`, `viral_shock_magnitude`

**Ground-truth demand components** *(for debugging and thesis diagrams)*
- `base_demand`, `trend_component`, `seasonal_annual`, `seasonal_weekly`
- `payday_multiplier`, `ramadan_multiplier`, `raya_multiplier`, `cny_multiplier`
- `mega_sale_multiplier`, `viral_shock_multiplier`, `holiday_multiplier`
- `noise_factor`, `demand_before_noise`

**Target variable**
- `demand` — units sold (integer)

**Lag and rolling features**
- `demand_lag_1`, `demand_lag_7`, `demand_lag_14`, `demand_lag_30`
- `demand_rolling_mean_7/14/30`, `demand_rolling_std_7/14/30`
- `demand_momentum`

**Shock summary**
- `any_shock_active`, `shock_type`, `effective_multiplier`, `days_since_last_shock`

**Continual learning labels**
- `task_id`, `task_name`

**Exogenous signals**
- `social_media_index` — proxy for viral/trend intensity (0–1)
- `competitor_activity_index` — competitor promotional activity (0–1)
- `marketing_spend_myr` — simulated internal marketing spend (MYR)

---

### Stage 2 — `rl_environment.csv` (86 columns)

Contains all 66 Stage 1 columns plus:

**Price columns**
- `current_price`, `base_price`, `price_ratio`
- `competitor_price`, `price_gap`
- `price_lag_1`, `price_change`

**Elasticity and demand response**
- `elasticity_coefficient` — dynamic, shifts across demand regimes
- `demand_forecast` — proxy for Stage 1 model output (replaced by real model output during training)
- `demand_forecast_uncertainty` — forecast confidence proxy
- `realized_demand` — actual units sold after price elasticity applied

**Inventory**
- `inventory_level`, `inventory_turnover`, `stockout_flag`

**Reward and business metrics**
- `revenue_myr`, `profit_margin_myr`, `reward`

**RL structure**
- `action_price` — price action taken at each timestep
- `done` — episode boundary flag (1 at task transitions and final row)
- `episode_id` — unique episode identifier per task × region × SKU

---

## Malaysian Calendar Events Encoded

### Federal Public Holidays
All federal public holidays for 2023–2025 are encoded with demand multipliers, including New Year's Day, Thaipusam, Labour Day, Wesak Day, Agong's Birthday, Merdeka Day, Malaysia Day, Deepavali, and Christmas.

### Lunar Calendar Events
Ramadan, Hari Raya Aidilfitri, and Chinese New Year dates are encoded with their correct Gregorian dates for each year, reflecting the ~10-day annual lunar drift:

| Event | 2023 | 2024 | 2025 |
|---|---|---|---|
| Chinese New Year | Jan 22 | Feb 10 | Jan 29 |
| Ramadan start | Mar 23 | Mar 11 | Mar 1 |
| Hari Raya Aidilfitri | Apr 21 | Apr 10 | Mar 30 |

### Mega Sales (Shopee / Lazada)
All 12 double-digit mega sales are encoded per year (1.1 through 12.12), with 11.11 carrying the highest base magnitude (10×) and 12.12 second (7.5×).

### Viral Shock Events
12 synthetic viral shock events are injected across 2023–2025, each targeting specific SKUs with a bell-curve demand envelope lasting 3–10 days.

---

## Demand Generation Model

Demand is generated as a multiplicative composition of components:

```
demand = base_demand
       × trend_component
       × seasonal_annual
       × seasonal_weekly
       × payday_multiplier
       × ramadan_multiplier
       × raya_multiplier
       × cny_multiplier
       × mega_sale_multiplier
       × viral_shock_multiplier
       × holiday_multiplier
       × noise_factor
```

Each component is stored as a separate column, enabling full interpretability and ground-truth access for ablation studies.

---

## Configuration

All parameters are controlled from `config/config.yaml`. Key sections:

```yaml
simulation:
  start_date: "2023-01-01"
  end_date:   "2025-12-31"
  random_seed: 42          # Change for different realisations

skus:
  - id: "P001"
    primary_shock_magnitude: 8.0   # Raya peak multiplier
    elasticity_base: -1.6          # Price sensitivity

seasonality:
  weekly_amplitude: 0.15
  annual_amplitude: 0.20
  noise_std: 0.08
```

Changing `random_seed` produces a different noise realisation while preserving all structural patterns. This can be used to generate multiple dataset realisations for robustness testing.

---

## Validation Plots

Running the pipeline generates 11 diagnostic plots in `output/plots/`:

| Plot | Description |
|---|---|
| `demand_overview_P00X.png` | Full demand curve per SKU with task boundaries, shock annotations, and component breakdown |
| `seasonal_decomposition_P00X.png` | Trend, annual seasonality, weekly seasonality, and monthly average demand |
| `price_demand_scatter_P00X.png` | Price vs realized demand scatter by region — confirms elasticity relationship |
| `cl_task_summary.png` | Row distribution and mean demand shift across CL tasks |
| `regional_heatmap.png` | Monthly average demand heatmap across all regions and SKUs |

---

## Reproducibility

- Fixed random seed (`42`) in `config.yaml` ensures identical output on every run
- All parameters are externalised to `config/config.yaml` — no magic numbers in code
- Malaysian holiday dates are hardcoded from official sources, not computed programmatically
- All ground-truth demand components are stored as columns for full auditability

---

## Dependencies

```
numpy >= 1.24
pandas >= 2.0
pyyaml >= 6.0
matplotlib >= 3.7
```

Install with:
```bash
pip install numpy pandas pyyaml matplotlib
```

---

## Citing This Dataset

If you use this dataset in your thesis or any subsequent publication, reference it as:

> Synthetic Malaysian E-Commerce Demand and Pricing Dataset (2023–2025). Generated for: *Continuous Edge Learning for Mitigating Catastrophic Forgetting in Dynamic Pricing and Real-Time Inventory Sync*. Ahmad Syakir Izzuan bin Hashim. [Institution], [Year].
