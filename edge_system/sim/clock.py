"""
Simulated clock. One tick = one M5 day.

The dataset is a daily panel, so the tick granularity is fixed by the data, not
chosen: there is exactly one `realized_demand` observation per SKU per day, and
a finer tick would have to invent intra-day structure that M5 does not record.

Wall-clock pacing is decoupled from simulated time (`wall_clock_per_tick_s`).
Experiments run it at 0 - as fast as the services answer - while a viva demo can
set 0.5 s so the dashboard visibly moves. Nothing downstream reads real time for
anything except latency measurement, so the two are safe to separate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

import pandas as pd


@dataclass(frozen=True)
class Tick:
    """One simulated day."""

    index: int              # 0-based tick counter
    date: pd.Timestamp      # the M5 day being replayed
    is_first: bool
    is_last: bool

    @property
    def date_str(self) -> str:
        return self.date.strftime("%Y-%m-%d")


class SimClock:
    """
    Iterates the walk-forward window a day at a time.

    Constructed from an explicit date range rather than from the data so a
    scenario can shorten a run without the driver having to slice frames; the
    driver intersects the clock's dates with what the data actually has.
    """

    def __init__(self, start: str, end: str, *, tick_days: int = 1,
                 wall_clock_per_tick_s: float = 0.0,
                 max_ticks: Optional[int] = None,
                 available_dates: Optional[pd.Series] = None) -> None:
        if tick_days < 1:
            raise ValueError(f"tick_days must be >= 1, got {tick_days}")

        dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end),
                              freq=f"{tick_days}D")
        if available_dates is not None:
            # Keep only days the dataset can actually serve. A tick with no rows
            # would silently generate zero orders and dilute every per-tick mean
            # in the results - better to not have the tick at all.
            have = set(pd.to_datetime(pd.Series(available_dates)).dt.normalize())
            dates = pd.DatetimeIndex([d for d in dates if d.normalize() in have])

        if max_ticks is not None:
            dates = dates[:max_ticks]
        if len(dates) == 0:
            raise ValueError(
                f"no simulable days between {start} and {end}"
                + ("" if available_dates is None else " that exist in the data")
            )

        self.dates: List[pd.Timestamp] = list(dates)
        self.wall_clock_per_tick_s = float(wall_clock_per_tick_s)
        self.started_at: Optional[float] = None
        self.current: Optional[Tick] = None

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def span(self) -> str:
        return f"{self.dates[0]:%Y-%m-%d} -> {self.dates[-1]:%Y-%m-%d}"

    def __iter__(self) -> Iterator[Tick]:
        self.started_at = time.perf_counter()
        last = len(self.dates) - 1
        for i, date in enumerate(self.dates):
            tick_started = time.perf_counter()
            self.current = Tick(index=i, date=date, is_first=i == 0, is_last=i == last)
            yield self.current

            if self.wall_clock_per_tick_s > 0:
                # Pace on elapsed time, not a flat sleep, so a slow tick does not
                # compound into drift against the requested rate.
                slack = self.wall_clock_per_tick_s - (time.perf_counter() - tick_started)
                if slack > 0:
                    time.sleep(slack)

    @property
    def elapsed_s(self) -> float:
        return 0.0 if self.started_at is None else time.perf_counter() - self.started_at

    @classmethod
    def from_config(cls, cfg: dict, *,
                    available_dates: Optional[pd.Series] = None) -> "SimClock":
        """Build from `SYSTEM_CONFIG['sim']`, re-read at call time."""
        return cls(
            start=cfg["start_date"], end=cfg["end_date"],
            tick_days=cfg.get("tick_days", 1),
            wall_clock_per_tick_s=cfg.get("wall_clock_per_tick_s", 0.0),
            max_ticks=cfg.get("max_ticks"),
            available_dates=available_dates,
        )
