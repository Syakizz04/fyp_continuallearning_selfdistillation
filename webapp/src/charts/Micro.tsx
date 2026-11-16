/*
 * Microcharts: inline SVG, no library, cheap enough for every table row.
 * Word-sized - they show shape and direction; the number beside them carries
 * the value, so none of them needs an axis. Hover gives the exact reading.
 */
import { useState } from 'react'

const INK3 = '#8A8A84'
const ACCENT = '#3A6A82'
const ALERT = '#A8322D'
const GRID = '#ECEAE3'

type Pt = { label?: string; v: number | null }

function extent(vals: number[], pad = 0): [number, number] {
  if (!vals.length) return [0, 1]
  let lo = Math.min(...vals), hi = Math.max(...vals)
  if (lo === hi) { lo -= 1; hi += 1 }
  const p = (hi - lo) * pad
  return [lo - p, hi + p]
}

/** A line over time; optional dashed reference (e.g. a drift threshold). */
export function Sparkline({ data, width = 96, height = 26, color = ACCENT, reference, zero, fmt = (v: number) => v.toFixed(1) }: {
  data: Pt[]; width?: number; height?: number; color?: string
  reference?: number | null; zero?: boolean; fmt?: (v: number) => string
}) {
  const [hover, setHover] = useState<number | null>(null)
  const vals = data.map(d => d.v).filter((v): v is number => v !== null && Number.isFinite(v))
  if (vals.length < 2) return <svg className="spark" width={width} height={height}><line x1={0} x2={width} y1={height / 2} y2={height / 2} stroke={GRID} /></svg>
  const all = reference !== undefined && reference !== null ? [...vals, reference] : vals
  const [lo0, hi] = extent(all, 0.08)
  let lo = lo0
  if (zero) lo = Math.min(lo, 0)
  const x = (i: number) => (data.length === 1 ? width / 2 : (i / (data.length - 1)) * (width - 4) + 2)
  const y = (v: number) => height - 2 - ((v - lo) / (hi - lo)) * (height - 4)
  let path = '', pen = false
  data.forEach((d, i) => {
    if (d.v === null || !Number.isFinite(d.v)) { pen = false; return }
    path += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(d.v).toFixed(1)}`
    pen = true
  })
  let lastI = -1
  for (let i = data.length - 1; i >= 0; i--) if (data[i].v !== null && Number.isFinite(data[i].v)) { lastI = i; break }
  const h = hover !== null ? data[hover] : null
  return (
    <span style={{ position: 'relative', display: 'inline-block' }}>
      <svg className="spark" width={width} height={height}
        onMouseMove={e => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect()
          setHover(Math.max(0, Math.min(data.length - 1, Math.round(((e.clientX - r.left - 2) / (width - 4)) * (data.length - 1)))))
        }}
        onMouseLeave={() => setHover(null)}>
        {reference !== undefined && reference !== null &&
          <line x1={0} x2={width} y1={y(reference)} y2={y(reference)} stroke={ALERT} strokeWidth={1} strokeDasharray="2 2" />}
        <path d={path} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
        {lastI >= 0 && <rect x={x(lastI) - 2} y={y(data[lastI].v!) - 2} width={4} height={4} fill={color} />}
        {hover !== null && h?.v !== null && h?.v !== undefined &&
          <>
            <line x1={x(hover)} x2={x(hover)} y1={0} y2={height} stroke="#1A1A1A" strokeWidth={1} />
            <rect x={x(hover) - 2.5} y={y(h.v) - 2.5} width={5} height={5} fill={color} stroke="#fff" strokeWidth={1} />
          </>}
      </svg>
      {h && h.v !== null && <span className="tip" style={{ position: 'absolute', bottom: height + 4, left: '50%', transform: 'translateX(-50%)', minWidth: 0, whiteSpace: 'nowrap', zIndex: 5, pointerEvents: 'none' }}>
        <span className="mono">{h.label ? `${h.label} · ` : ''}{fmt(h.v)}</span>
      </span>}
    </span>
  )
}

/** Daily bars from a zero baseline; `alt` stacks a second series in grey. */
export function SparkBars({ data, width = 96, height = 26, color = ACCENT, fmt = (v: number) => String(v) }: {
  data: { label?: string; v: number; alt?: number }[]; width?: number; height?: number; color?: string; fmt?: (v: number) => string
}) {
  const [hover, setHover] = useState<number | null>(null)
  const n = Math.max(data.length, 1)
  const hi = Math.max(1, ...data.map(d => d.v + (d.alt ?? 0)))
  // Always fit `width`: drop the 1px gaps first, then thin the bars.
  const gap = width - (n - 1) >= n ? 1 : 0
  const bw = Math.max(0.5, (width - gap * (n - 1)) / n)
  const h = hover !== null ? data[hover] : null
  return (
    <span style={{ position: 'relative', display: 'inline-block' }}>
      <svg className="spark" width={width} height={height} onMouseLeave={() => setHover(null)}>
        <line x1={0} x2={width} y1={height - .5} y2={height - .5} stroke={GRID} />
        {data.map((d, i) => {
          const x = i * (bw + gap)
          const hv = (d.v / hi) * (height - 1)
          const ha = ((d.alt ?? 0) / hi) * (height - 1)
          return (
            <g key={i} onMouseEnter={() => setHover(i)}>
              <rect x={x} y={0} width={bw + gap} height={height} fill="transparent" />
              {d.v > 0 && <rect x={x} y={height - hv} width={bw} height={hv} fill={color} opacity={hover === null || hover === i ? 1 : .45} />}
              {ha > 0 && <rect x={x} y={height - hv - ha} width={bw} height={ha} fill={INK3} opacity={hover === null || hover === i ? .8 : .35} />}
            </g>
          )
        })}
      </svg>
      {h && <span className="tip" style={{ position: 'absolute', bottom: height + 4, left: '50%', transform: 'translateX(-50%)', minWidth: 0, whiteSpace: 'nowrap', zIndex: 5, pointerEvents: 'none' }}>
        <span className="mono">{h.label ? `${h.label} · ` : ''}{fmt(h.v)}{h.alt ? ` · ${h.alt} refused` : ''}</span>
      </span>}
    </span>
  )
}

/**
 * Where a price sits in its allowed band: the band as a bar, the model's own
 * price as a hollow tick, the recommended price as a solid one, base as a
 * faint centre mark. Shows at a glance whether a rule moved the price.
 */
export function PriceBand({ raw, final, base, floor, ceiling, width = 110 }: {
  raw: number; final: number; base: number; floor: number | null; ceiling: number | null; width?: number
}) {
  // The model can move ±10% of base; draw that span, then the operator band inside it.
  const lo = Math.min(base * 0.88, raw, final, floor ?? Infinity)
  const hi = Math.max(base * 1.12, raw, final, ceiling ?? -Infinity)
  const x = (v: number) => 2 + ((v - lo) / (hi - lo)) * (width - 4)
  const f = floor ?? lo, c = ceiling ?? hi
  const moved = Math.abs(raw - final) > 1e-9
  return (
    <svg className="spark" width={width} height={16}
      aria-label={`band ${floor ?? 'none'} to ${ceiling ?? 'none'}, model ${raw.toFixed(2)}, recommended ${final.toFixed(2)}`}>
      <title>{`allowed ${floor !== null ? '$' + floor.toFixed(2) : '—'} – ${ceiling !== null ? '$' + ceiling.toFixed(2) : '—'} · model $${raw.toFixed(2)} · recommended $${final.toFixed(2)}`}</title>
      <line x1={2} x2={width - 2} y1={8} y2={8} stroke={GRID} strokeWidth={2} />
      <rect x={x(f)} y={5} width={Math.max(1, x(c) - x(f))} height={6} fill="#E4EFEE" />
      <line x1={x(base)} x2={x(base)} y1={3} y2={13} stroke={INK3} strokeWidth={1} />
      {moved && <rect x={x(raw) - 2.5} y={4.5} width={5} height={7} fill="#fff" stroke="#1A1A1A" strokeWidth={1} />}
      <rect x={x(final) - 2} y={3} width={4} height={10} fill={moved ? ACCENT : '#1A1A1A'} />
    </svg>
  )
}

/** A filled share of a whole, e.g. stock left against the restock target. */
export function Meter({ value, max, width = 64, color = ACCENT, warnBelow }: {
  value: number; max: number; width?: number; color?: string; warnBelow?: number
}) {
  const frac = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0
  const low = warnBelow !== undefined && frac < warnBelow
  return (
    <svg className="spark" width={width} height={8}>
      <rect x={0} y={0} width={width} height={8} fill={GRID} />
      <rect x={0} y={0} width={width * frac} height={8} fill={low ? ALERT : color} />
    </svg>
  )
}
