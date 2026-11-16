// Business readings of the data the v1 pages show raw. Kept apart so each
// estimate is defined once and the pages only lay them out.
import type { NodeHealth, Product, SimDay } from '../../api'

/** Below this many days of stock a product is "running low". */
export const LOW_DAYS = 7
/** Below this, it is at risk of running out before a delivery could land. */
export const RISK_DAYS = 3

/** Sales refused for want of stock (or of a reachable stock count), at the
 *  price the channel was selling at that day. An estimate: a refused customer
 *  might have paid a different price, or bought something else. */
export function lostSales(day: SimDay | undefined): number | null {
  if (!day) return null
  const avg = day.sold ? day.revenue / day.sold : 0
  return day.channels.reduce((s, c) => s + c.refused * (c.mean_price || avg), 0)
}

export interface StockLine {
  sku: string
  inStock: number | null
  perDay: number          // mean units sold a day over the last 28 days
  daysLeft: number | null // inStock / perDay; null when it does not sell
  refused28: number
}

export function stockLine(p: Product): StockLine {
  const last = p.daily.slice(-28)
  const perDay = last.length ? last.reduce((s, d) => s + d.sold, 0) / last.length : 0
  const inStock = p.true_available
  return {
    sku: p.sku, inStock, perDay,
    daysLeft: inStock === null || perDay <= 0 ? null : inStock / perDay,
    refused28: last.reduce((s, d) => s + d.refused, 0),
  }
}

/** Profit achieved / profit expected on the pricing model's latest daily check,
 *  averaged over the channels that have one. */
export function profitVsExpected(nodes: NodeHealth[]): number | null {
  const v = nodes.map(n => n.drift?.status.last_profit_index).filter((x): x is number => typeof x === 'number')
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null
}

/** A plain-words line for a drift warning. */
export function driftWords(kind: 'forecasting' | 'rl'): string {
  return kind === 'forecasting' ? 'Demand moved away from the forecast' : 'Prices earning less than expected'
}
