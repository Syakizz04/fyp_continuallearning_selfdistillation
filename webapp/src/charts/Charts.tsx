/*
 * Charts, in the dashboard/theme.py idiom: white plot face, one faint
 * horizontal rule, no x grid, quiet axes, a unified crosshair readout.
 * Marks follow the dataviz spec: 2px lines, >=8px ringed markers, a 10% wash
 * for bands, <=24px bars with 4px rounded data-ends and a 2px gap; values lead
 * in tooltips with a line key; alert colour only on drift triggers.
 */
import {
  Area, Bar, BarChart, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine,
  ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis,
} from 'recharts'
import type { CheckRow, DriftEvent, ForecastDay, SimDay } from '../api'
import { channelColor, channelLabel, num, shortDate, units } from '../format'

const INK2 = '#5A5A55'
const ACCENT = '#3A6A82'
const ALERT = '#A8322D'
const LEARN = '#EEECE5'   // checks with no threshold yet: the channel is learning its normal level
const HISTORY = '#6B7177'          // theme.py "frozen" grey: observed, not modelled

const axis = { stroke: '#D8D6CE', tickLine: false, tick: { fill: INK2 } }

// Chart rows mix the x label, numbers and [lo, hi] band arrays per series key,
// and recharts reads them by dynamic dataKey, so no narrower type fits.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Row = Record<string, any>
/** What recharts hands a Scatter `shape`: the point's pixel position (absent for a null value). */
type DotProps = { cx?: number; cy?: number | null }

function TipShell({ head, rows }: { head: string; rows: { key: React.ReactNode; name: string; value: string }[] }) {
  return (
    <div className="tip">
      <div className="tip-head">{head}</div>
      {rows.map((r, i) => <div className="tip-row" key={i}>{r.key}<span className="v">{r.value}</span><span className="n">{r.name}</span></div>)}
    </div>
  )
}
const lineKey = (c: string) => <span className="key-line" style={{ background: c }} />
const bandKey = <span className="key-line" style={{ background: ACCENT, opacity: .25, height: 8 }} />

// ── Forecast: 28 days observed, 14 days ahead with a p10–p90 band ───────────

export function ForecastChart({ history, daily, height = 300 }: {
  history: { date: string; display_date: string | null; actual: number }[]
  daily: ForecastDay[]
  height?: number
}) {
  const rows: Row[] = [
    ...history.map(h => ({ x: h.display_date ?? h.date, actual: h.actual })),
    ...daily.map(d => ({ x: d.display_date ?? d.date, p50: d.p50, band: [d.p10, d.p90], p10: d.p10, p90: d.p90 })),
  ]
  // Bridge the observed line into the forecast so the two read as one timeline.
  if (history.length && daily.length) {
    const last = rows[history.length - 1]
    last.p50 = last.actual
    last.band = [last.actual, last.actual]
  }
  const today = history.length ? (history[history.length - 1].display_date ?? history[history.length - 1].date) : undefined

  return (
    <div className="chart-frame">
      <div className="legend">
        <span className="key">{lineKey(HISTORY)}Units sold</span>
        <span className="key">{lineKey(ACCENT)}Forecast (median)</span>
        <span className="key">{bandKey}80% range (p10–p90)</span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: -12 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="x" {...axis} tickFormatter={shortDate} minTickGap={28} />
          <YAxis {...axis} axisLine={false} width={48} allowDecimals={false} />
          {today && <ReferenceLine x={today} stroke="#1A1A1A" strokeWidth={1}
            label={{ value: 'today', position: 'insideTopRight', fill: INK2, fontSize: 12 }} />}
          <Area dataKey="band" stroke="none" fill={ACCENT} fillOpacity={0.1} isAnimationActive={false} activeDot={false} />
          <Line dataKey="actual" stroke={HISTORY} strokeWidth={2} dot={false} isAnimationActive={false}
            activeDot={{ r: 4, stroke: '#fff', strokeWidth: 2 }} connectNulls={false} />
          <Line dataKey="p50" stroke={ACCENT} strokeWidth={2} dot={false} isAnimationActive={false}
            activeDot={{ r: 4, stroke: '#fff', strokeWidth: 2 }} />
          <Tooltip cursor={{ stroke: '#1A1A1A', strokeWidth: 1 }} isAnimationActive={false}
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null
              const r = payload[0].payload as Row
              const rows = r.actual !== undefined && r.p10 === undefined
                ? [{ key: lineKey(HISTORY), name: 'units sold', value: units(r.actual) }]
                : [{ key: lineKey(ACCENT), name: 'forecast', value: units(r.p50) },
                   { key: bandKey, name: 'range', value: `${units(r.p10)}–${units(r.p90)}` }]
              return <TipShell head={String(label)} rows={rows} />
            }} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Daily checks: profit index per node against the drift floor ────────────

export function CheckChart({ series, threshold, metric, events, height = 260, label, fmt, thresholdLabel = 'Drift threshold' }: {
  series: { node: string; rows: CheckRow[] }[]
  threshold: number | null
  metric: 'profit_index' | 'mase'
  events: DriftEvent[]
  height?: number
  label: string
  /** how a value reads; default 2 dp on the axis, 3 in the tooltip */
  fmt?: (v: number) => string
  thresholdLabel?: string
}) {
  // One row per date, one column per node: the crosshair reads every node at once.
  const byDate = new Map<string, Row>()
  for (const s of series) for (const r of s.rows) {
    const v = r[metric]
    if (v === null || v === undefined) continue
    const x = r.display_date ?? r.date
    const row = byDate.get(x) ?? { x, _sim: r.date }
    row[s.node] = v
    byDate.set(x, row)
  }
  const kind = metric === 'mase' ? 'forecasting' : 'rl'
  // Warnings are kept far longer than the check window, so an old one would
  // stretch the axis back to a stretch with no line under it. Plot only the
  // ones the lines cover; the warnings table still lists every one.
  const first = [...byDate.keys()].sort()[0]
  for (const e of events.filter(e => e.kind === kind)) {
    const x = e.display_date ?? e.sim_date
    if (first === undefined || x < first) continue
    const row = byDate.get(x) ?? { x }
    row[`ev_${e.node}`] = e.value
    byDate.set(x, row)
  }
  const rows = [...byDate.values()].sort((a, b) => String(a.x).localeCompare(String(b.x)))
  const nodes = series.map(s => s.node)
  const sparse = rows.length <= 40           // few (weekly) points: show markers, not just a line
  // "2026-11-04" -> "04/11/26"
  const tick = dayMonth

  if (!rows.length) return null
  return (
    <div className="chart-frame">
      <div className="legend">
        {nodes.map(n => <span key={n} className="key">{lineKey(channelColor(n))}{channelLabel(n)}</span>)}
        {threshold !== null && <span className="key">{lineKey(ALERT)}{thresholdLabel}</span>}
        <span className="key"><span className="key-dot" style={{ background: ALERT }} />Drift warning</span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: -8 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="x" {...axis} tickFormatter={tick} minTickGap={32} />
          <YAxis {...axis} axisLine={false} width={52} domain={['auto', 'auto']} tickFormatter={v => fmt ? fmt(v) : num(v, 2)} />
          {threshold !== null && <ReferenceLine y={threshold} stroke={ALERT} strokeWidth={1}
            label={{ value: metric === 'mase' ? 'retrain above' : 'retrain below', position: 'insideBottomRight', fill: ALERT, fontSize: 12 }} />}
          {nodes.map(n => (
            <Line key={n} dataKey={n} stroke={channelColor(n)} strokeWidth={2} isAnimationActive={false} connectNulls
              dot={sparse ? { r: 4, fill: channelColor(n), stroke: '#fff', strokeWidth: 2 } : false}
              activeDot={{ r: 4, stroke: '#fff', strokeWidth: 2 }} />
          ))}
          {nodes.map(n => (
            <Scatter key={`ev_${n}`} dataKey={`ev_${n}`} fill={ALERT} isAnimationActive={false}
              shape={(p: DotProps) => p.cy == null ? <g /> : <circle cx={p.cx} cy={p.cy} r={5} fill={ALERT} stroke="#fff" strokeWidth={2} />} />
          ))}
          <Tooltip cursor={{ stroke: '#1A1A1A', strokeWidth: 1 }} isAnimationActive={false}
            content={({ active, payload, label: x }) => {
              if (!active || !payload?.length) return null
              const r = payload[0].payload as Row
              const out = nodes.filter(n => r[n] !== undefined)
                .map(n => ({ key: lineKey(channelColor(n)), name: channelLabel(n), value: fmt ? fmt(r[n] as number) : num(r[n], 3) }))
              for (const n of nodes) if (r[`ev_${n}`] !== undefined)
                out.push({ key: <span className="key-dot" style={{ background: ALERT }} />, name: `${channelLabel(n)} drift warning`, value: fmt ? fmt(r[`ev_${n}`] as number) : num(r[`ev_${n}`], 3) })
              return <TipShell head={`${tick(String(x))} · ${label}`} rows={out} />
            }} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Daily checks, one small panel per channel ──────────────────────────────
//
// Three noisy lines on one plot read as clutter, and with per-channel
// (rolling) floors there is no single threshold to draw. Each channel gets its
// own panel with its own floor; the panels share one y-scale so a channel that
// sits lower is still visibly lower.

/** "2026-11-04" -> "04/11" */
const dayMonth = (x: string) => { const [, m, d] = String(x).split('-'); return d ? `${d}/${m}` : String(x) }

export function CheckPanels({ series, metric, events, height = 150, label, fmt = v => num(v, 2), thresholdLabel = 'Drift threshold' }: {
  series: { node: string; rows: CheckRow[]; threshold: number | null }[]
  metric: 'profit_index' | 'mase'
  events: DriftEvent[]
  height?: number
  label: string
  fmt?: (v: number) => string
  thresholdLabel?: string
}) {
  const kind = metric === 'mase' ? 'forecasting' : 'rl'
  const firstDay = (s: { rows: CheckRow[] }) => s.rows.find(r => r[metric] !== null)?.display_date ?? ''
  const shown = (e: DriftEvent) => e.kind === kind && series.some(s => s.node === e.node && (e.display_date ?? e.sim_date) >= firstDay(s))
  // The line a check was judged against, as recorded with it. Rows from a
  // node without that record fall back to the channel's current line.
  const thrKey = metric === 'mase' ? 'fc_threshold' : 'rl_threshold'
  const recordedIn = (s: { rows: CheckRow[] }) => s.rows.some(r => thrKey in r)
  const thrOf = (s: { rows: CheckRow[]; threshold: number | null }) => {
    const recorded = recordedIn(s)
    return (r: CheckRow) => (recorded ? (r[thrKey] ?? null) : s.threshold)
  }
  // Checks with no line: the channel is learning its normal level, at the
  // start of a run and again after each model update (the node discards the
  // old line, which belonged to the old model). Spans run from the day the old
  // line stops to the day the new one starts, so the shading meets both.
  const learningSpans = (data: Row[], recorded: boolean) => {
    const out: [string, string][] = []
    if (!recorded) return out
    let start: string | null = null
    data.forEach((r, i) => {
      const off = r.v !== undefined && (r.thr === null || r.thr === undefined)
      if (off && start === null) start = String((data[i - 1] ?? r).x)   // from where the old line stops
      if (start !== null && (!off || i === data.length - 1)) {
        out.push([start, String(r.x)])
        start = null
      }
    })
    return out
  }
  const vals = [...series.flatMap(s => [...s.rows.map(r => r[metric]), ...s.rows.map(thrOf(s)), s.threshold]),
                ...events.filter(shown).flatMap(e => [e.value, e.threshold])]
    .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  if (!vals.length) return null
  // Round ticks (0.7, 0.8, 0.9 ...), so no axis label sits a hair off the
  // threshold's own number.
  const lo = Math.min(...vals), hi = Math.max(...vals)
  const raw = (hi - lo || Math.abs(hi) || 1) / 3
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map(f => f * mag).find(s => s >= raw) ?? 10 * mag
  const domain: [number, number] = [+(Math.floor(lo / step) * step).toFixed(10), +(Math.ceil(hi / step) * step).toFixed(10)]
  const ticks: number[] = []
  for (let t = domain[0]; t <= domain[1] + step / 2; t += step) ticks.push(+t.toFixed(10))
  const panels = series.map(s => {
    const thr = thrOf(s)
    const rows: Row[] = s.rows.filter(r => r[metric] !== null && r[metric] !== undefined)
      .map(r => ({ x: r.display_date ?? r.date, v: r[metric], thr: thr(r) }))
    const first = rows[0]?.x
    const byX = new Map(rows.map(r => [r.x, r]))
    for (const e of events.filter(e => e.kind === kind && e.node === s.node)) {
      const x = e.display_date ?? e.sim_date
      if (first === undefined || x < first) continue
      const r = byX.get(x) ?? { x }
      r.ev = e.value
      // A warning carries the line it crossed; draw that one at its day.
      if (e.threshold !== null && e.threshold !== undefined) r.thr = e.threshold
      byX.set(x, r)
    }
    const data = [...byX.values()].sort((a, b) => String(a.x).localeCompare(String(b.x)))
    return { s, data, learning: learningSpans(data, recordedIn(s)) }
  })
  const anyLearning = panels.some(p => p.learning.length > 0)
  return (
    <div className="chart-frame">
      <div className="legend">
        <span className="key">{lineKey(ALERT)}{thresholdLabel}</span>
        <span className="key"><span className="key-dot" style={{ background: ALERT }} />Drift warning</span>
        {anyLearning && <span className="key" title="No threshold yet: the channel is learning what normal looks like - 4 weeks at the start of a run, 1 week after a model update (its first line then is refined daily for a month). Nothing is checked meanwhile">
          <span className="key-dot" style={{ background: LEARN, border: '1px solid #D9D6CC' }} />Learning normal level</span>}
      </div>
      <div className="grid grid-3 grid-gap" style={{ marginTop: '.4rem' }}>
        {panels.map(({ s, data, learning }) => {
          const lastThr = [...data].reverse().find(r => r.thr !== null && r.thr !== undefined)?.thr as number | undefined
          const color = channelColor(s.node)
          return (
            <div key={s.node} style={{ minWidth: 0 }}>
              <div className="caption" style={{ display: 'flex', alignItems: 'center', gap: '.4rem', marginBottom: '.2rem' }}>
                <span className="key-dot" style={{ background: color, width: 8, height: 8, display: 'inline-block' }} />
                <span style={{ color: 'var(--ink)' }}>{channelLabel(s.node)}</span>
              </div>
              {!data.length ? <div className="caption faint" style={{ height, display: 'grid', placeItems: 'center' }}>no checks yet</div> : (
                <ResponsiveContainer width="100%" height={height}>
                  <ComposedChart data={data} margin={{ top: 6, right: 6, bottom: 0, left: 0 }}>
                    {learning.map(([a, b], i) => <ReferenceArea key={`l${i}`} x1={a} x2={b} fill={LEARN} fillOpacity={1} stroke="none" />)}
                    <CartesianGrid vertical={false} />
                    <XAxis dataKey="x" {...axis} tickFormatter={dayMonth} minTickGap={28} />
                    <YAxis {...axis} axisLine={false} width={42} domain={domain} ticks={ticks} interval={0} tickFormatter={v => fmt(v)} />
                    {/* Stepped: the line moves where the channel recalibrated after a retrain. */}
                    <Line dataKey="thr" type="stepAfter" stroke={ALERT} strokeWidth={1} strokeDasharray="3 3"
                      dot={false} activeDot={false} isAnimationActive={false} connectNulls={false} />
                    {lastThr !== undefined && <ReferenceLine y={lastThr} stroke="none"
                      label={{ value: fmt(lastThr),
                               position: metric === 'mase' ? 'insideTopRight' : 'insideBottomRight', fill: ALERT, fontSize: 11 }} />}
                    <Line dataKey="v" stroke={color} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls
                      activeDot={{ r: 4, stroke: '#fff', strokeWidth: 2 }} />
                    <Scatter dataKey="ev" fill={ALERT} isAnimationActive={false}
                      shape={(p: DotProps) => p.cy == null ? <g /> : <circle cx={p.cx} cy={p.cy} r={4.5} fill={ALERT} stroke="#fff" strokeWidth={2} />} />
                    <Tooltip cursor={{ stroke: '#1A1A1A', strokeWidth: 1 }} isAnimationActive={false}
                      content={({ active, payload, label: x }) => {
                        if (!active || !payload?.length) return null
                        const r = payload[0].payload as Row
                        const out = []
                        if (r.v !== undefined) out.push({ key: lineKey(color), name: label, value: fmt(r.v) })
                        if (r.ev !== undefined) out.push({ key: <span className="key-dot" style={{ background: ALERT }} />, name: 'drift warning', value: fmt(r.ev) })
                        if (r.thr !== null && r.thr !== undefined) out.push({ key: lineKey(ALERT), name: thresholdLabel.toLowerCase(), value: fmt(r.thr) })
                        else if (learning.length && r.v !== undefined) out.push({ key: lineKey(ALERT), name: thresholdLabel.toLowerCase(), value: 'learning' })
                        return <TipShell head={`${dayMonth(String(x))} · ${channelLabel(s.node)}`} rows={out} />
                      }} />
                  </ComposedChart>
                </ResponsiveContainer>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Stock belief error: how far each channel's view is behind the truth ────

export function BeliefErrorChart({ rows, channels, height = 240 }: {
  rows: { sku: string; errors: Record<string, number> }[]
  channels: string[]
  height?: number
}) {
  const data = rows.map(r => ({ sku: r.sku, ...Object.fromEntries(channels.map(c => [c, Math.abs(r.errors[c] ?? 0)])) }))
  return (
    <div className="chart-frame">
      <div className="legend">{channels.map(c => (
        <span key={c} className="key"><span style={{ width: 10, height: 10, borderRadius: 2, background: channelColor(c) }} />{channelLabel(c)}</span>
      ))}</div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }} barGap={2} barCategoryGap="22%">
          <CartesianGrid vertical={false} />
          <XAxis dataKey="sku" {...axis} tickFormatter={s => String(s).replace('FOODS_', 'F')} interval={0} />
          <YAxis {...axis} axisLine={false} width={44} allowDecimals={false} />
          {channels.map(c => <Bar key={c} dataKey={c} fill={channelColor(c)} maxBarSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false} />)}
          <Tooltip cursor={{ fill: 'rgba(0,0,0,.04)' }} isAnimationActive={false}
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null
              return <TipShell head={String(label)} rows={payload.map(p => ({
                key: <span style={{ width: 10, height: 10, borderRadius: 2, background: channelColor(String(p.dataKey)) }} />,
                name: channelLabel(String(p.dataKey)), value: `${num(Number(p.value))} units`,
              }))} />
            }} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Simulation: units sold / refused per trading day, disruptions shaded ───

function ranges(days: { x: string; on: boolean }[]) {
  const out: [string, string][] = []
  let start: string | null = null, prev: string | null = null
  for (const d of days) {
    if (d.on && start === null) start = d.x
    if (!d.on && start !== null) { out.push([start, prev!]); start = null }
    prev = d.x
  }
  if (start !== null && prev !== null) out.push([start, prev])
  return out
}

export function SimDaysChart({ days, height = 220 }: { days: SimDay[]; height?: number }) {
  const rows = days.map(d => ({ x: d.display_date ?? d.sim_date, sold: d.sold, refused: d.refused,
    shock: d.shock, partition: d.partition }))
  const shock = ranges(rows.map(r => ({ x: r.x, on: r.shock !== null })))
  const outage = ranges(rows.map(r => ({ x: r.x, on: r.partition.length > 0 })))
  return (
    <div className="chart-frame">
      <div className="legend">
        <span className="key"><span style={{ width: 10, height: 10, background: ACCENT }} />Units sold</span>
        <span className="key"><span style={{ width: 10, height: 10, background: '#8A8A84' }} />Refused</span>
        <span className="key"><span style={{ width: 14, height: 10, background: '#F7EFDD', border: '1px solid #9A6A12' }} />Demand shock</span>
        <span className="key"><span style={{ width: 14, height: 10, background: '#F6E7E5', border: '1px solid #A8322D' }} />Channel offline</span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: -12 }} barCategoryGap={1}>
          <CartesianGrid vertical={false} />
          {shock.map(([a, b], i) => <ReferenceArea key={`s${i}`} x1={a} x2={b} fill="#F7EFDD" fillOpacity={1} stroke="none" />)}
          {outage.map(([a, b], i) => <ReferenceArea key={`o${i}`} x1={a} x2={b} fill="#F6E7E5" fillOpacity={.8} stroke="none" />)}
          <XAxis dataKey="x" {...axis} tickFormatter={shortDate} minTickGap={36} />
          <YAxis {...axis} axisLine={false} width={40} allowDecimals={false} />
          <Bar dataKey="sold" stackId="u" fill={ACCENT} isAnimationActive={false} maxBarSize={18} />
          <Bar dataKey="refused" stackId="u" fill="#8A8A84" isAnimationActive={false} maxBarSize={18} />
          <Tooltip cursor={{ fill: 'rgba(0,0,0,.05)' }} isAnimationActive={false}
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null
              const r = payload[0].payload as Row
              const out = [
                { key: <span style={{ width: 10, height: 10, background: ACCENT }} />, name: 'sold', value: num(r.sold) },
                { key: <span style={{ width: 10, height: 10, background: '#8A8A84' }} />, name: 'refused', value: num(r.refused) },
              ]
              if (r.shock !== null) out.push({ key: <span />, name: 'demand shock', value: `×${r.shock}` })
              if (r.partition.length) out.push({ key: <span />, name: 'offline', value: r.partition.map(channelLabel).join(', ') })
              return <TipShell head={shortDate(String(label))} rows={out} />
            }} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
