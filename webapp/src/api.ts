// Typed client for the decision-support gateway (edge_system/gateway/service.py).
// Every route is under /api; in dev Vite proxies it to :8020.

export type Channel = 'pos' | 'web' | 'marketplace' | string

export interface Status {
  sim_date: string | null
  display_date: string | null
  offset_days: number | null
  services: Record<string, boolean>
  last_generation: { sim_date: string; created: number; skus: number; n_errors: number; errors: string[]; seconds: number } | null
  counts: Record<'pending' | 'approved' | 'rejected' | 'superseded', number>
  channels: Channel[]
  sim: SimBrief | null
}

export type SimMode = 'starting' | 'paused' | 'playing' | 'finished' | 'resetting' | 'error'
export interface Shock { multiplier: number; sku: string | null; days: number; from_tick: number; until_tick: number; until_date: string; display_until?: string | null }
export interface SimBrief {
  mode: SimMode; error: string | null; speed_s: number; require_approval: boolean
  tick: number; total: number; sim_date: string | null; last_traded: string | null
  policy: string; delay_ms: number; partition: Channel[]; shock: Shock | null
  /** channels whose model update is holding the next day from opening */
  waiting_for?: Channel[]
}
export interface SimDay {
  tick: number; sim_date: string; display_date: string | null; seconds: number
  orders: number; units_requested: number; sold: number; refused: number; oversold: number
  revenue: number; fill_rate: number | null; shock: number | null; partition: Channel[]
  prices: Partial<Record<'approved' | 'rejected' | 'auto' | 'unreviewed', number>>
  channels: { channel: Channel; sold: number; refused: number; revenue: number; mean_price: number }[]
}
export interface SimEvent { ts: number; kind: string; text: string; sim_date: string | null; display_date: string | null; tick: number }
export interface SimState extends SimBrief {
  display_sim_date?: string | null; display_last_traded?: string | null
  channels: Channel[]; policies: string[]; speeds: number[]; skus: string[]
  totals: { days: number; sold: number; refused: number; oversold: number; revenue: number; fill_rate: number | null
            prices: Record<'approved' | 'rejected' | 'auto' | 'unreviewed', number> }
  days: SimDay[]; log: SimEvent[]
  nodes: Record<Channel, { generation: number; retraining: boolean; events: number }>
  last_generation: { created: number; seconds: number; n_errors: number } | null
}

export interface Product {
  sku: string
  dept: string
  base_price: number | null
  true_available: number | null
  latest_sim_date: string | null
  pending: number
  has_rule: boolean
  /** measured from the replayed data; null when the data file is unavailable */
  profile: ProductProfile | null
  daily: { sim_date: string; sold: number; refused: number; stock: number | null }[]
  price_history: { sim_date: string; price: number }[]
}

export interface ProductProfile { price: number; mean_daily: number; cv: number; snap_lift: number; elasticity: number; tags: string[] }

export interface ForecastDay { date: string; display_date: string | null; p10: number; p50: number; p90: number }
export interface Forecast {
  sku: string
  node: string
  as_of: string
  display_as_of: string | null
  horizon: number
  total_p50: number
  daily: ForecastDay[]
  history: { date: string; display_date: string | null; actual: number }[]
  model_generation: number
  forecaster_generation: number
  forecaster_updated_at: number | null
  forecaster_updated_sim_date: string | null
  forecaster_updated_display_date: string | null
  reason?: string
}

export type Bound = 'min_price' | 'max_price' | 'min_margin'
/** One row of the inventory service's event log (GET /inventory/events). */
export interface InventoryEvent {
  id: number; ts: number; kind: string
  node: string | null; sku: string | null; qty: number | null
  granted?: number | null; node_view?: number | null; true_available?: number | null
  [extra: string]: unknown
}

export type RecStatus = 'pending' | 'approved' | 'rejected' | 'superseded'

export interface Recommendation {
  id: number
  sku: string
  channel: Channel
  sim_date: string
  display_date: string | null
  created_at: number
  model_generation: number | null
  tier: number | null
  raw_price: number
  base_price: number
  final_price: number
  unit_cost: number
  margin_pct: number | null
  raw_margin_pct: number | null
  floor: number | null
  ceiling: number | null
  bound_by: Bound[]
  conflict: boolean
  inventory_view: number | null
  true_available: number | null
  forecast_p50_total: number | null
  /** pending rows only: the price on the shelf now (gateway `shelf`) */
  shelf_price?: number | null
  status: RecStatus
  decision: 'approve' | 'reject' | null
  decided_by: string | null
  decision_note: string | null
  decided_at: number | null
  applied_price: number | null
  live_price: number | null
  live_source: 'approved' | 'rejected' | 'auto' | 'unreviewed' | null
}

export interface Decision {
  id: number
  recommendation_id: number
  action: 'approve' | 'reject'
  user: string
  note: string | null
  ts: number
  applied_price: number | null
  sku: string
  channel: Channel
  sim_date: string
  display_date: string | null
  raw_price: number
  final_price: number
  base_price: number
  bound_by: Bound[]
}

export interface GlobalRule {
  min_margin_pct: number | null
  max_discount_pct: number | null
  max_markup_pct: number | null
  unit_cost_ratio: number
  updated_at?: number
  updated_by?: string | null
}
export interface SkuRule {
  sku: string
  min_price: number | null
  max_price: number | null
  min_margin_pct: number | null
  unit_cost: number | null
  updated_at?: number
  updated_by?: string | null
}
export interface Rules { global: GlobalRule; skus: SkuRule[]; unit_cost_is_assumed: boolean }
export interface Preview { pending: number; changed: number; examples: { id: number; sku: string; channel: Channel; current: number; draft: number; bound_by: Bound[] }[] }

export interface StockRow {
  sku: string
  true_available: number
  total: number
  committed: number
  reserved: number
  central_free: number
  node_views: Record<Channel, number>
  staleness: Record<Channel, number>
  stock_estimates: Record<Channel, number>
  estimate_errors: Record<Channel, number>
}
export interface ChannelSync {
  channel: Channel
  partitioned: boolean
  quota_units: number
  mean_abs_estimate_error: number | null
  max_abs_estimate_error: number | null
  skus_with_stale_view: number
  sync_hops: number
}
export interface Inventory {
  sim_date: string | null
  display_date: string | null
  policy: string
  backend: string
  effective: Record<string, unknown>
  metrics: Record<string, number | null>
  channels: ChannelSync[]
  skus: StockRow[]
}

export interface DriftEvent { tick: number; sim_date: string; display_date: string | null; kind: 'forecasting' | 'rl'; value: number; threshold: number; node: string; ts: number }
/** A line put up after a model swap, before its warm-up completes: level so far + the previous spread. */
export interface Provisional { mu: number; sigma: number; k: number; days: number }
export interface CheckRow {
  tick: number; date: string; display_date: string | null; mase: number | null; profit_index: number | null
  /** the line this check was judged against (moves at each recalibration); null while warming up */
  fc_threshold?: number | null; rl_threshold?: number | null
}
export interface NodeHealth {
  node: Channel
  online: boolean
  error?: string
  health?: {
    ok: boolean; strategy: string; uptime_s: number; model_generation: number | null
    retrain_in_flight: boolean; sdft_alpha: number | null
    forecaster: { forecaster_generation: number; forecaster_updated_at: number | null; forecaster_updated_sim_date: string | null } | null
    last_fc_check: { tick: number; sim_date: string; mase: number | null; seconds: number; ts: number } | null
  }
  model?: { version: { generation: number; source: string; retrained_at: number | null; trigger_reason: string | null }; history: unknown[] }
  drift?: {
    status: { checks: number; fc_threshold: number | null; fc_signal?: 'dataset' | 'realised'; fc_warmup_left?: number; fc_calibration?: { mu: number; sigma: number; k: number; warmup_days: number; consecutive: number } | null; rl_threshold: number | null; fc_streak: number; rl_streak: number; rl_detector?: 'fixed' | 'rolling'; rl_warmup_left?: number; rl_calibration?: { mu: number; sigma: number; k: number; window_days: number; warmup_days: number } | null; rl_provisional?: Provisional | null; fc_provisional?: Provisional | null; last_mase: number | null; last_profit_index: number | null; n_events: number }
    events: DriftEvent[]
    stream: CheckRow[]
  }
}
export interface ModelVersion { id: number; ts: number; node: string; strategy: string; generation: number; kind: 'forecasting' | 'rl'; sim_date: string; display_date: string | null; reason: string; duration_s: number | null; ok: number | null; skipped: number | null; error: string | null }
export interface Models { nodes: NodeHealth[]; versions: ModelVersion[]; sdft_alpha: number }

export type HistoryItem =
  | { kind: 'decision'; ts: number; user: string; sim_date: string; display_date: string | null; sku: string; channel: Channel; action: 'approve' | 'reject'; note: string | null; price: number; recommendation_id: number }
  | { kind: 'rule'; ts: number; user: string; scope: 'global' | 'sku'; sku: string | null; before: Record<string, number | null> | null; after: Record<string, number | null> | null; display_date: string | null }
  | { kind: 'recommendations'; ts: number; sim_date: string; display_date: string | null; count: number; bound: number }
  | { kind: 'model'; ts: number; node: string; sim_date: string; display_date: string | null; model_kind: 'forecasting' | 'rl'; generation: number; reason: string; ok: number | null; duration_s: number | null }

// ── transport ──────────────────────────────────────────────────────────────

export class ApiError extends Error {
  status: number
  service?: string
  constructor(status: number, message: string, service?: string) {
    super(message)
    this.status = status
    this.service = service
  }
}

export type Role = 'operator' | 'viewer'
export interface Me { username: string; role: Role; created_at: number; created_by: string | null }
/** Fired on any 401: the session is gone, so the app returns to sign-in. */
export const SIGNED_OUT = 'shelfsense:signed-out'

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  let res: Response
  try {
    res = await fetch(`/api${path}`, {
      method,
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
  } catch {
    throw new ApiError(0, 'Gateway unreachable - is the system running?', 'gateway')
  }
  if (res.status === 401 && !path.startsWith('/auth/')) window.dispatchEvent(new Event(SIGNED_OUT))
  if (!res.ok) {
    let detail = res.statusText
    let service: string | undefined
    try {
      const j = await res.json()
      detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
      service = j.service
    } catch { /* not JSON */ }
    throw new ApiError(res.status, detail, service)
  }
  return res.json() as Promise<T>
}

const qs = (p: Record<string, string | number | undefined | null>) => {
  const s = Object.entries(p).filter(([, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`).join('&')
  return s ? `?${s}` : ''
}

export const api = {
  authStatus: () => call<{ setup_required: boolean; user: Me | null; roles: Role[] }>('GET', '/auth/status'),
  login: (username: string, password: string) => call<{ user: Me }>('POST', '/auth/login', { username, password }),
  setup: (username: string, password: string) => call<{ user: Me }>('POST', '/auth/setup', { username, password }),
  logout: () => call<{ ok: boolean }>('POST', '/auth/logout'),
  users: () => call<Me[]>('GET', '/auth/users'),
  addUser: (username: string, password: string, role: Role) => call<Me>('POST', '/auth/users', { username, password, role }),
  removeUser: (username: string) => call<{ ok: boolean }>('DELETE', `/auth/users/${encodeURIComponent(username)}`),
  status: () => call<Status>('GET', '/status'),
  products: () => call<Product[]>('GET', '/products'),
  forecast: (sku: string, node?: string) => call<Forecast>('GET', `/products/${encodeURIComponent(sku)}/forecast${qs({ node })}`),

  rules: () => call<Rules>('GET', '/rules'),
  putGlobal: (r: GlobalRule) => call<Rules>('PUT', '/rules/global', r),
  putSku: (sku: string, r: Omit<SkuRule, 'sku'>) => call<Rules>('PUT', `/rules/sku/${encodeURIComponent(sku)}`, r),
  deleteSku: (sku: string) => call<Rules>('DELETE', `/rules/sku/${encodeURIComponent(sku)}`),
  preview: (body: { global_rule?: GlobalRule; sku?: string; sku_rule?: Omit<SkuRule, 'sku'> }) =>
    call<Preview>('POST', '/rules/preview', body),

  recommendations: (p: { status?: RecStatus; sku?: string; channel?: string; sim_date?: string; limit?: number } = {}) =>
    call<Recommendation[]>('GET', `/recommendations${qs(p)}`),
  recommendationsPage: (p: { status?: RecStatus; channel?: string; q?: string; bound?: boolean; limit: number; offset: number }) =>
    call<{ items: Recommendation[]; total: number; limit: number; offset: number }>('GET',
      `/recommendations/page${qs({ ...p, bound: p.bound ? 'true' : undefined })}`),
  approve: (id: number, note?: string) => call<Recommendation>('POST', `/recommendations/${id}/approve`, { note }),
  reject: (id: number, note?: string) => call<Recommendation>('POST', `/recommendations/${id}/reject`, { note }),
  bulk: (ids: number[], action: 'approve' | 'reject', note?: string) =>
    call<{ decided: number[]; failed: { id: number; status: number; detail: string }[] }>('POST', '/recommendations/bulk', { ids, action, note }),
  decisions: (user?: string) => call<Decision[]>('GET', `/decisions${qs({ user })}`),

  inventory: () => call<Inventory>('GET', '/inventory'),
  inventoryEvents: (limit = 60) => call<InventoryEvent[]>('GET', `/inventory/events${qs({ limit })}`),
  models: (stream = 120) => call<Models>('GET', `/models${qs({ stream })}`),
  history: (limit = 300) => call<HistoryItem[]>('GET', `/history${qs({ limit })}`),

  sim: () => call<SimState>('GET', '/sim'),
  simAction: (action: 'play' | 'pause' | 'step' | 'speed' | 'outage' | 'latency' | 'shock' | 'clear-shock' | 'restock' | 'settings' | 'reset',
              body?: Record<string, unknown>) => call<SimState>('POST', `/sim/${action}`, body ?? {}),
}
