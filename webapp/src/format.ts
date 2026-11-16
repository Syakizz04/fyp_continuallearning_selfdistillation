// One place for how numbers and dates read, so every page agrees.

export const money = (v: number | null | undefined) =>
  v === null || v === undefined || !Number.isFinite(v) ? '—' : `$${v.toFixed(2)}`

export const num = (v: number | null | undefined, dp = 0) =>
  v === null || v === undefined || !Number.isFinite(v) ? '—'
    : v.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp })

/** fraction -> "12.5%" */
export const pct = (v: number | null | undefined, dp = 1) =>
  v === null || v === undefined || !Number.isFinite(v) ? '—' : `${(v * 100).toFixed(dp)}%`

/** fraction -> "+4.0%" / "−6.0%" */
export const signedPct = (v: number | null | undefined, dp = 1) => {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  const s = (v * 100).toFixed(dp)
  return v > 0 ? `+${s}%` : v < 0 ? `−${s.slice(1)}%` : `${s}%`
}

/** Demand forecasts for slow movers come back as 1e-30; show them as people read them. */
export const units = (v: number | null | undefined) => {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  if (v < 0.05) return '0'
  return v < 10 ? v.toFixed(1) : Math.round(v).toLocaleString('en-US')
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

/** "2026-11-04" -> "Wed 4 Nov 2026" (parsed as a calendar date, never shifted by timezone) */
export const longDate = (iso: string | null | undefined) => {
  if (!iso) return '—'
  const [y, m, d] = iso.split('-').map(Number)
  const wd = new Date(Date.UTC(y, m - 1, d)).getUTCDay()
  return `${WEEKDAYS[wd]} ${d} ${MONTHS[m - 1]} ${y}`
}
/** "2026-11-04" -> "4 Nov" */
export const shortDate = (iso: string | null | undefined) => {
  if (!iso) return '—'
  const [, m, d] = iso.split('-').map(Number)
  return `${d} ${MONTHS[m - 1]}`
}

/** epoch seconds -> "3 min ago" */
export const ago = (ts: number | null | undefined) => {
  if (!ts) return '—'
  const s = Math.max(0, Date.now() / 1000 - ts)
  if (s < 45) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} d ago`
}
export const clockTime = (ts: number | null | undefined) =>
  !ts ? '—' : new Date(ts * 1000).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })

export const CHANNEL_LABEL: Record<string, string> = { pos: 'In-store till', web: 'Web store', marketplace: 'Marketplace' }
export const channelLabel = (c: string) => CHANNEL_LABEL[c] ?? c
// Hex, not var(--ch-*): Recharts sets fill/stroke as SVG presentation
// attributes, which do not resolve CSS custom properties. Same values as the
// --ch-* tokens in styles.css (validated categorical set).
// pos was teal (#008C82), then blue (#1F4D9A) - each too close to the UI
// accent of its day - and is now indigo; marketplace was purple
// (#7A5BC9) and too close to the blue. All three pass the palette validator
// (all pairs, light surface) and stay clear of the status red/green/amber.
const CHANNEL_HEX: Record<string, string> = { pos: '#3D3FAA', web: '#D06A2C', marketplace: '#B04BC0' }
export const channelColor = (c: string) => CHANNEL_HEX[c] ?? '#8A8A84'

export const BOUND_LABEL: Record<string, string> = {
  min_margin: 'Margin floor', min_price: 'Min price', max_price: 'Max price',
}
