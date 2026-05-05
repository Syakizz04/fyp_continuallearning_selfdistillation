"""
calendar.py
-----------
Malaysian public holiday calendar, mega sale dates, lunar calendar events,
school holidays, and payday windows for 2023-2025.

Simplifying assumptions (stated in thesis methodology):
- Weekend standardised to Saturday-Sunday across all regions
- State-level holidays not modelled; federal holidays only
- Borneo-specific cultural holidays (Gawai, Kaamatan) not modelled
- East Coast weekend structure (Fri-Sat) not modelled
"""

from datetime import date, timedelta
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. FEDERAL PUBLIC HOLIDAYS (West Malaysia, applies all regions)
# ---------------------------------------------------------------------------

FEDERAL_HOLIDAYS = {
    # 2023
    date(2023, 1, 1):  ("New Year's Day", 1.3),
    date(2023, 1, 22): ("Chinese New Year Day 1", 0.0),   # handled separately
    date(2023, 1, 23): ("Chinese New Year Day 2", 0.0),
    date(2023, 2, 5):  ("Thaipusam", 1.15),
    date(2023, 4, 7):  ("Hari Raya Aidilfitri Day 1", 0.0),  # handled separately
    date(2023, 4, 8):  ("Hari Raya Aidilfitri Day 2", 0.0),
    date(2023, 4, 22): ("Hari Raya Aidiladha", 1.2),
    date(2023, 5, 1):  ("Labour Day", 1.1),
    date(2023, 5, 4):  ("Wesak Day", 1.1),
    date(2023, 6, 5):  ("Agong's Birthday", 1.1),
    date(2023, 7, 19): ("Awal Muharram", 1.05),
    date(2023, 8, 31): ("Merdeka Day", 1.25),
    date(2023, 9, 16): ("Malaysia Day", 1.25),
    date(2023, 9, 27): ("Nuzul Quran", 1.05),
    date(2023, 10, 28): ("Prophet's Birthday", 1.05),
    date(2023, 11, 12): ("Deepavali", 1.3),
    date(2023, 12, 25): ("Christmas", 1.2),

    # 2024
    date(2024, 1, 1):  ("New Year's Day", 1.3),
    date(2024, 2, 10): ("Chinese New Year Day 1", 0.0),
    date(2024, 2, 11): ("Chinese New Year Day 2", 0.0),
    date(2024, 2, 24): ("Thaipusam", 1.15),
    date(2024, 4, 10): ("Hari Raya Aidilfitri Day 1", 0.0),
    date(2024, 4, 11): ("Hari Raya Aidilfitri Day 2", 0.0),
    date(2024, 5, 1):  ("Labour Day", 1.1),
    date(2024, 5, 22): ("Wesak Day", 1.1),
    date(2024, 6, 3):  ("Agong's Birthday", 1.1),
    date(2024, 6, 17): ("Hari Raya Aidiladha", 1.2),
    date(2024, 7, 7):  ("Awal Muharram", 1.05),
    date(2024, 8, 31): ("Merdeka Day", 1.25),
    date(2024, 9, 16): ("Malaysia Day", 1.25),
    date(2024, 10, 16): ("Prophet's Birthday", 1.05),
    date(2024, 11, 1): ("Deepavali", 1.3),
    date(2024, 12, 25): ("Christmas", 1.2),

    # 2025
    date(2025, 1, 1):  ("New Year's Day", 1.3),
    date(2025, 1, 29): ("Chinese New Year Day 1", 0.0),
    date(2025, 1, 30): ("Chinese New Year Day 2", 0.0),
    date(2025, 3, 11): ("Thaipusam", 1.15),
    date(2025, 3, 30): ("Hari Raya Aidilfitri Day 1", 0.0),
    date(2025, 3, 31): ("Hari Raya Aidilfitri Day 2", 0.0),
    date(2025, 5, 1):  ("Labour Day", 1.1),
    date(2025, 5, 12): ("Wesak Day", 1.1),
    date(2025, 6, 2):  ("Agong's Birthday", 1.1),
    date(2025, 6, 6):  ("Hari Raya Aidiladha", 1.2),
    date(2025, 6, 26): ("Awal Muharram", 1.05),
    date(2025, 8, 31): ("Merdeka Day", 1.25),
    date(2025, 9, 5):  ("Prophet's Birthday", 1.05),
    date(2025, 9, 16): ("Malaysia Day", 1.25),
    date(2025, 11, 20): ("Deepavali", 1.3),
    date(2025, 12, 25): ("Christmas", 1.2),
}


# ---------------------------------------------------------------------------
# 2. LUNAR CALENDAR EVENTS — Ramadan, Raya, CNY windows
# ---------------------------------------------------------------------------

# Ramadan: (start_date, end_date)
RAMADAN_WINDOWS = {
    2023: (date(2023, 3, 23), date(2023, 4, 20)),
    2024: (date(2024, 3, 11), date(2024, 4, 9)),
    2025: (date(2025, 3, 1),  date(2025, 3, 29)),
}

# Hari Raya Aidilfitri: actual day (demand peak in PRE-window)
RAYA_DATES = {
    2023: date(2023, 4, 21),
    2024: date(2024, 4, 10),
    2025: date(2025, 3, 30),
}

# Chinese New Year: actual day (demand peak in PRE-window)
CNY_DATES = {
    2023: date(2023, 1, 22),
    2024: date(2024, 2, 10),
    2025: date(2025, 1, 29),
}

# Raya demand build-up: starts N days before, peaks on eve
RAYA_PRE_WINDOW_DAYS = 42    # 6 weeks of baju raya shopping
RAYA_POST_DROP_DAYS  = 14    # demand crashes after Raya

# CNY demand build-up
CNY_PRE_WINDOW_DAYS  = 21
CNY_POST_DROP_DAYS   = 7


# ---------------------------------------------------------------------------
# 3. MEGA SALE DATES (Shopee / Lazada double-digit sales)
# ---------------------------------------------------------------------------

def get_mega_sales(years):
    """Return list of (date, sale_name, base_magnitude) for all mega sales."""
    sales = []
    monthly = [
        (1,  1,  "1.1",  4.5),
        (2,  2,  "2.2",  3.5),
        (3,  3,  "3.3",  3.8),
        (4,  4,  "4.4",  3.8),
        (5,  5,  "5.5",  4.5),
        (6,  6,  "6.6",  4.5),
        (7,  7,  "7.7",  3.5),
        (8,  8,  "8.8",  4.5),
        (9,  9,  "9.9",  5.5),
        (10, 10, "10.10", 5.5),
        (11, 11, "11.11", 10.0),   # Singles Day — biggest
        (12, 12, "12.12", 7.5),
    ]
    for year in years:
        for month, day, name, magnitude in monthly:
            try:
                sales.append((date(year, month, day), name, magnitude))
            except ValueError:
                pass  # skip invalid dates
    return sales


# ---------------------------------------------------------------------------
# 4. SCHOOL HOLIDAYS (approximate West Malaysia windows)
# ---------------------------------------------------------------------------

SCHOOL_HOLIDAYS = [
    # 2023
    (date(2023, 3, 11), date(2023, 3, 19),  "School Holiday March 2023"),
    (date(2023, 5, 13), date(2023, 5, 21),  "School Holiday May 2023"),
    (date(2023, 8, 12), date(2023, 8, 20),  "School Holiday Aug 2023"),
    (date(2023, 11, 18), date(2023, 12, 31), "Year-End School Holiday 2023"),
    # 2024
    (date(2024, 3, 9),  date(2024, 3, 17),  "School Holiday March 2024"),
    (date(2024, 5, 25), date(2024, 6, 2),   "School Holiday May 2024"),
    (date(2024, 8, 10), date(2024, 8, 18),  "School Holiday Aug 2024"),
    (date(2024, 11, 16), date(2024, 12, 31), "Year-End School Holiday 2024"),
    # 2025
    (date(2025, 3, 15), date(2025, 3, 23),  "School Holiday March 2025"),
    (date(2025, 5, 24), date(2025, 6, 1),   "School Holiday May 2025"),
    (date(2025, 8, 9),  date(2025, 8, 17),  "School Holiday Aug 2025"),
    (date(2025, 11, 22), date(2025, 12, 31), "Year-End School Holiday 2025"),
]


# ---------------------------------------------------------------------------
# 5. CALENDAR FEATURE BUILDER
# ---------------------------------------------------------------------------

def build_calendar_features(date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build a DataFrame of all calendar features for the given date range.
    One row per date. This is merged into the demand dataset.
    """
    years = list(range(date_range.year.min(), date_range.year.max() + 1))
    mega_sales = get_mega_sales(years)
    mega_sale_lookup = {d: (name, mag) for d, name, mag in mega_sales}

    records = []
    for dt in date_range:
        d = dt.date()
        year = d.year

        # --- Basic temporal ---
        row = {
            "date": dt,
            "day_of_week": dt.dayofweek,        # 0=Mon, 6=Sun
            "day_of_month": dt.day,
            "week_of_year": dt.isocalendar()[1],
            "month": dt.month,
            "quarter": dt.quarter,
            "is_weekend": int(dt.dayofweek >= 5),
            "is_month_start": int(dt.day <= 3),
            "is_month_end": int(dt.day >= 28),
        }

        # --- Federal holiday ---
        if d in FEDERAL_HOLIDAYS:
            hname, _ = FEDERAL_HOLIDAYS[d]
            row["is_federal_holiday"] = 1
            row["holiday_name"] = hname
        else:
            row["is_federal_holiday"] = 0
            row["holiday_name"] = None

        # --- Payday windows ---
        row["is_government_payday_window"] = int(d.day in [24, 25, 26])
        row["is_private_payday_window"]    = int(d.day in [1, 2, 3])

        # --- Ramadan ---
        ram_start, ram_end = RAMADAN_WINDOWS.get(year, (None, None))
        in_ramadan = (ram_start is not None) and (ram_start <= d <= ram_end)
        row["is_ramadan"] = int(in_ramadan)
        row["ramadan_day"] = (d - ram_start).days + 1 if in_ramadan else 0

        # --- Days to Raya ---
        raya_date = RAYA_DATES.get(year)
        if raya_date:
            days_to_raya = (raya_date - d).days
            row["days_to_raya"] = max(-RAYA_POST_DROP_DAYS, min(RAYA_PRE_WINDOW_DAYS, days_to_raya))
            row["is_pre_raya_window"]  = int(0 < days_to_raya <= RAYA_PRE_WINDOW_DAYS)
            row["is_post_raya_window"] = int(-RAYA_POST_DROP_DAYS <= days_to_raya <= 0)
        else:
            row["days_to_raya"] = 999
            row["is_pre_raya_window"]  = 0
            row["is_post_raya_window"] = 0

        # --- Days to CNY ---
        cny_date = CNY_DATES.get(year)
        if cny_date:
            days_to_cny = (cny_date - d).days
            row["days_to_cny"] = max(-CNY_POST_DROP_DAYS, min(CNY_PRE_WINDOW_DAYS, days_to_cny))
            row["is_pre_cny_window"]  = int(0 < days_to_cny <= CNY_PRE_WINDOW_DAYS)
            row["is_post_cny_window"] = int(-CNY_POST_DROP_DAYS <= days_to_cny <= 0)
        else:
            row["days_to_cny"] = 999
            row["is_pre_cny_window"]  = 0
            row["is_post_cny_window"] = 0

        # --- Mega sale ---
        if d in mega_sale_lookup:
            sale_name, sale_mag = mega_sale_lookup[d]
            row["is_mega_sale"]   = 1
            row["mega_sale_name"] = sale_name
            row["mega_sale_base_magnitude"] = sale_mag
        else:
            row["is_mega_sale"]   = 0
            row["mega_sale_name"] = None
            row["mega_sale_base_magnitude"] = 0.0

        # --- School holiday ---
        in_school_holiday = any(start <= d <= end for start, end, _ in SCHOOL_HOLIDAYS)
        row["is_school_holiday"] = int(in_school_holiday)

        # --- Viral shock placeholder (injected by demand.py) ---
        row["viral_shock_active"]     = 0
        row["viral_shock_magnitude"]  = 0.0

        records.append(row)

    return pd.DataFrame(records)


def get_raya_demand_multiplier(days_to_raya: int, sku_config: dict) -> float:
    """
    Smooth Raya demand multiplier based on countdown to Raya day.
    Peaks on Raya eve, drops sharply after.
    """
    if days_to_raya > RAYA_PRE_WINDOW_DAYS or days_to_raya < -RAYA_POST_DROP_DAYS:
        return 1.0

    if days_to_raya >= 0:
        # Build-up phase: sigmoid curve peaking at day 0
        peak = sku_config.get("primary_shock_magnitude", 5.0)
        progress = 1 - (days_to_raya / RAYA_PRE_WINDOW_DAYS)
        multiplier = 1.0 + (peak - 1.0) * (progress ** 2.2)
    else:
        # Post-Raya drop: linear decay back to baseline
        peak = sku_config.get("primary_shock_magnitude", 5.0)
        decay = 1 + days_to_raya / RAYA_POST_DROP_DAYS  # days_to_raya is negative
        multiplier = 1.0 + (peak - 1.0) * max(0, decay)

    return max(1.0, multiplier)


def get_cny_demand_multiplier(days_to_cny: int, sku_config: dict) -> float:
    """Smooth CNY demand multiplier."""
    if days_to_cny > CNY_PRE_WINDOW_DAYS or days_to_cny < -CNY_POST_DROP_DAYS:
        return 1.0

    secondary_mag = sku_config.get("secondary_shock_magnitude", 2.0)

    if days_to_cny >= 0:
        progress = 1 - (days_to_cny / CNY_PRE_WINDOW_DAYS)
        multiplier = 1.0 + (secondary_mag - 1.0) * (progress ** 1.8)
    else:
        decay = 1 + days_to_cny / CNY_POST_DROP_DAYS
        multiplier = 1.0 + (secondary_mag - 1.0) * max(0, decay)

    return max(1.0, multiplier)
