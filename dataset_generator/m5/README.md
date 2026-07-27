# M5 → FYP adapter

Turns the raw Kaggle **M5 Forecasting – Accuracy** data into the two CSVs the
continual-learning pipeline consumes, for the drift-triggered experiment.

**Scope (locked):** `cat_id == FOODS`, stores `CA_1..CA_4`.

## 1. Get the raw data

Accept the competition rules, then:

```powershell
kaggle competitions download -c m5-forecasting-accuracy
# unzip into data/m5_raw/
```

Expected files in `data/m5_raw/`:

| file | role |
|------|------|
| `sales_train_evaluation.csv` (or `..._validation.csv`) | daily unit sales, wide `d_1..d_N` |
| `calendar.csv` | date ↔ `d`, `wm_yr_wk`, events, `snap_CA` |
| `sell_prices.csv` | weekly `sell_price` per (store, item) |

## 2. Build

```powershell
python -m dataset_generator.m5.build_m5                 # data/m5_raw -> data/processed_m5
python -m dataset_generator.m5.build_m5 --self-test     # validate adapter on a synthetic mini-M5
python -m dataset_generator.m5.build_m5 --raw-dir D:\m5 --out-dir data\processed_m5
```

## 3. Outputs (`data/processed_m5/`)

### `demand_forecasting.csv` — real M5

`date, product_id(item_id), region_id(store_id), product_category(dept_id),
demand, day_of_week, day_of_month, week_of_year, month, quarter, is_weekend,
sell_price, snap, is_event, event_type_code`

`sell_price`/`snap`/`is_event`/`event_type_code` are the M5-native known-reals
that replace the synthetic dataset's Malaysia features. The drift pipeline's
CONFIG `known_reals` points at these.

### `rl_environment.csv` — semi-synthetic

Real, calibrated from M5: `elasticity_coefficient` (see below),
`sell_price`-derived `base_price` (28-day rolling median), `competitor_price`
(dept median per store/week).

#### Elasticity estimation (`elasticity.py`)

Estimated by a three-level empirical-Bayes hierarchy — item IV estimate,
department pool, external prior — replacing the old flat
"clip to [-3.0, -0.2], else fallback to -1.5". On the 100-item CA_1 scope that
old scheme left **57% of elasticities arbitrary** (25% at the constant, 32%
pinned at a clip bound). The estimator now reports **0% of each**.

Three things worth knowing, all visible in `elasticity_report.csv`:

1. **Price is instrumented.** OLS is biased here: promotions cut price and run
   display/feature advertising at the same time, so `log(price)` correlates with
   the demand shock. 31% of items came out with a *positive* elasticity under
   OLS. The price is instrumented with the same item's price in the stores
   outside the modelling scope (a Hausman instrument), built from the unfiltered
   `sell_prices.csv` — so it costs no extra data. First-stage F is reported per
   item; below 10 (Staiger-Stock) the estimate is down-weighted, not trusted.

2. **The instrument does not fully solve it, and that is reported.** Wrong-sign
   estimates only fall 31% → 28%, because Walmart prices chain-wide: other
   stores carry the *same* national promotion shock, so the exclusion
   restriction fails. This is Bresnahan's known critique of Hausman instruments.
   Adding snap/event controls (31% → 32%) and aggregating to the weekly price
   frequency (31% → 41%) were both tested and did not help. M5 records no
   promotion flag, so the confounder is not observable in this dataset.

3. **Wrong-sign estimates are discarded by sign restriction, not clipped.** A
   positive elasticity violates the law of demand, so it is treated as evidence
   of confounding: the item is marked unidentified and takes its department's
   pooled elasticity instead. Current split: 54 items IV-identified, 21
   discarded as wrong-sign, 25 with no usable price variation.

**Honest limitation:** the 46 prior-shrunk items do not get individually
distinct values — they take their department's pooled mean, so items within a
department share a number. That is the correct posterior when a series carries
no price information, and unlike the old `-1.5` it is estimated from data rather
than assumed, but it is a shared value and should be described as one.

#### The external prior (dunnhumby)

Fitted from **dunnhumby "The Complete Journey"** — 2,500 households, two years,
with `causal_data.csv` giving the `display` and `mailer` promotion flags that M5
lacks entirely. It supplies **(mu, tau) only — never a row-level join**; there is
no shared item or store key with M5.

```powershell
kaggle datasets download -d frtgnn/dunnhumby-the-complete-journey `
    -p data/external_pricing --unzip
python -c "from dataset_generator.m5.elasticity import load_dunnhumby; \
    load_dunnhumby('data/external_pricing','data/external_pricing/dunnhumby_panel.csv')"
# fit once, cache, then reuse (the fit takes minutes)
python -m dataset_generator.m5.build_m5 ... --external-prior data/external_pricing/prior.json
```

Fitted prior: **mu = −0.524, tau = 0.313**, from 270 identified products.

**Grain matters, and getting it wrong is silent.** dunnhumby must be aggregated
to **product × week pooled across stores**, not product × store × week — even
though the latter matches `causal_data.csv`'s key exactly. It is a household
panel, not a scanner census: at store grain 74% of cells hold a single unit, the
median product-store has ONE week of data, and within-product-store price
variation is exactly zero. Fitting there produces textbook attenuation —
elasticities collapse to a pooled −0.15 and the sign becomes a coin flip. That
prior would have dragged every M5 item toward "price does not affect demand" and
quietly destroyed the RL environment. `load_dunnhumby` also requires ≥5 baskets
behind each weekly price for the same reason.

**Cross-dataset validation.** After the sign restriction, the two datasets agree
on central tendency despite being different retailers, decades and methods:

| | n | mean | median |
|---|---|---|---|
| dunnhumby (promotion-controlled) | 260 | −1.02 | −0.62 |
| M5 (IV-identified) | 54 | −1.12 | −0.85 |

**A negative result worth reporting:** controlling for `display`/`mailer` barely
moves the dunnhumby prior (−0.555 → −0.524) and does not reduce its wrong-sign
rate (44% → 43%). So the promotion confound identified on M5 is *not* mainly
display and feature advertising — even when those are observed directly, the
wrong-sign problem persists. Whatever drives it is more fundamental than the
promotion variables retailers record, which further justifies handling it by
sign restriction rather than by hunting for more controls.

Without `--external-prior` the build falls back to a documented placeholder and
says so loudly.

Modeled: `inventory_level` + `stockout_flag` (simple (s,S) reorder policy),
`demand_forecast` (28-day trailing mean reference — replaced by the TFT forecast
at runtime).

`snap` / `is_event` carry the regime signal that `is_mega_sale` carried in the
synthetic env; `DynamicPricingEnv` reads regime flags via `df.get(col, 0)`, so
the env's state vector is remapped to these in the drift pipeline (later phase).

## Notes

- Rows before an item's first sale (no `sell_price`) are dropped — standard M5.
- FOODS × 4 CA stores ≈ 5.7k item×store series over ~2011-01-29 → 2016-06-19.
- Forecasting CSV is fully real M5; only the RL economic layer is semi-synthetic.
