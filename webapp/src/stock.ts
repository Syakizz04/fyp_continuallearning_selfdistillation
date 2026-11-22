import type { Product } from './api'

// ── derived per-product figures ─────────────────────────────────────────────

export interface Row extends Product {
  sold28: number
  refused28: number
  avgDaily: number
  stockNow: number | null
  cover: number | null          // days of stock at the recent sales rate
  price: number | null          // live in-store price, else base
}

export function derive(p: Product): Row {
  const sold28 = p.daily.reduce((a, d) => a + d.sold, 0)
  const refused28 = p.daily.reduce((a, d) => a + d.refused, 0)
  const avgDaily = p.daily.length ? sold28 / p.daily.length : 0
  const stockNow = p.true_available
  const cover = stockNow !== null && avgDaily > 0 ? stockNow / avgDaily : null
  const price = p.price_history.length ? p.price_history[p.price_history.length - 1].price : p.base_price
  return { ...p, sold28, refused28, avgDaily, stockNow, cover, price }
}

export const LOW_COVER_DAYS = 5
