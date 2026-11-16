"""
Operator price guardrails: limits and minimum margins applied to the PPO's price.

Pure functions, no I/O, so every edge case is unit-testable.

The PPO chooses a tier in [-10%, +10%] of base price. Guardrails never change
what the agent decided - they record it as `raw_price` - and clamp the price the
operator is asked to approve into the band the operator set:

    floor   = max(min_price, margin_floor)
    ceiling = max_price
    final   = clamp(raw_price, floor, ceiling)

Margin is GROSS MARGIN ON PRICE, (price - cost) / price, the retail convention.
So a minimum margin m implies a price floor of cost / (1 - m) - not
cost * (1 + m), which is markup and gives a lower floor for the same number.

Unit cost is an assumption: M5 records sell prices and no costs, so cost defaults
to base_price x unit_cost_ratio unless a per-SKU rule supplies one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class GlobalRule:
    """Defaults for every SKU. Relative to base price, so one rule fits a
    catalogue whose prices span two orders of magnitude."""

    min_margin_pct: Optional[float] = 0.20    # gross margin floor, fraction
    max_discount_pct: Optional[float] = None  # floor = base x (1 - this)
    max_markup_pct: Optional[float] = None    # ceiling = base x (1 + this)
    unit_cost_ratio: float = 0.70             # cost = base x this, if no SKU cost


@dataclass(frozen=True)
class SkuRule:
    """Per-SKU overrides, in absolute currency. None = inherit."""

    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_margin_pct: Optional[float] = None
    unit_cost: Optional[float] = None


@dataclass(frozen=True)
class EffectiveRule:
    """What actually applies to one SKU at one base price."""

    unit_cost: float
    min_margin_pct: Optional[float]
    min_price: Optional[float]
    max_price: Optional[float]


@dataclass
class GuardResult:
    raw_price: float
    final_price: float
    base_price: float
    unit_cost: float
    floor: Optional[float]
    ceiling: Optional[float]
    #: Which rules moved the price: subset of min_price / max_price / min_margin.
    bound_by: List[str] = field(default_factory=list)
    #: The rules contradict (floor above ceiling). Margin protection wins.
    conflict: bool = False

    @property
    def margin_pct(self) -> Optional[float]:
        return gross_margin(self.final_price, self.unit_cost)

    @property
    def raw_margin_pct(self) -> Optional[float]:
        return gross_margin(self.raw_price, self.unit_cost)

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["margin_pct"] = self.margin_pct
        d["raw_margin_pct"] = self.raw_margin_pct
        return d


def gross_margin(price: float, cost: float) -> Optional[float]:
    return None if price <= 0 else (price - cost) / price


def validate_margin(m: Optional[float]) -> None:
    if m is not None and not (0.0 <= m < 1.0):
        raise ValueError(f"min_margin_pct must be in [0, 1), got {m}: a 100% "
                         f"gross margin needs a zero cost, and above that a price "
                         f"floor does not exist")


def resolve(base_price: float, glob: GlobalRule, sku: Optional[SkuRule] = None) -> EffectiveRule:
    """Merge the global rule and an optional SKU override for one base price.

    A SKU's absolute limit and the global relative limit both apply, and the
    TIGHTER one wins - an override can narrow the band but cannot silently widen
    it past a catalogue-wide policy.
    """
    sku = sku or SkuRule()
    unit_cost = sku.unit_cost if sku.unit_cost is not None else base_price * glob.unit_cost_ratio
    margin = sku.min_margin_pct if sku.min_margin_pct is not None else glob.min_margin_pct

    floors = [v for v in (sku.min_price,
                          None if glob.max_discount_pct is None
                          else base_price * (1.0 - glob.max_discount_pct))
              if v is not None]
    ceilings = [v for v in (sku.max_price,
                            None if glob.max_markup_pct is None
                            else base_price * (1.0 + glob.max_markup_pct))
                if v is not None]
    return EffectiveRule(unit_cost=unit_cost, min_margin_pct=margin,
                         min_price=max(floors) if floors else None,
                         max_price=min(ceilings) if ceilings else None)


def _cents_up(x: float) -> float:
    """Round a floor UP to the cent - rounding to nearest can land a cent below
    a margin floor, which is exactly the breach the floor exists to prevent."""
    return math.ceil(x * 100.0 - 1e-6) / 100.0


def _cents_down(x: float) -> float:
    return math.floor(x * 100.0 + 1e-6) / 100.0


def apply(raw_price: float, base_price: float, rule: EffectiveRule) -> GuardResult:
    """Clamp the agent's price into the operator's band and say why."""
    validate_margin(rule.min_margin_pct)

    margin_floor = (rule.unit_cost / (1.0 - rule.min_margin_pct)
                    if rule.min_margin_pct is not None else None)
    candidates = {k: v for k, v in (("min_price", rule.min_price),
                                    ("min_margin", margin_floor)) if v is not None}
    floor = max(candidates.values()) if candidates else None
    floor_by = [k for k, v in candidates.items() if floor is not None and abs(v - floor) < 1e-9]
    ceiling = rule.max_price

    final, bound_by, conflict = raw_price, [], False
    if floor is not None and ceiling is not None and floor > ceiling:
        # Contradictory rules. Selling below the margin floor loses money on
        # every unit; exceeding a max price only forgoes demand - so the floor
        # wins, and the conflict is surfaced rather than resolved silently.
        final, bound_by, conflict = _cents_up(floor), floor_by + ["max_price"], True
    elif floor is not None and raw_price < floor:
        final, bound_by = _cents_up(floor), floor_by
    elif ceiling is not None and raw_price > ceiling:
        final, bound_by = _cents_down(ceiling), ["max_price"]
    # An unbound price is left exactly as the agent produced it: rounding it
    # would show a "change" that no rule made.
    return GuardResult(raw_price=raw_price, final_price=final, base_price=base_price,
                       unit_cost=rule.unit_cost, floor=floor, ceiling=ceiling,
                       bound_by=sorted(set(bound_by)), conflict=conflict)
