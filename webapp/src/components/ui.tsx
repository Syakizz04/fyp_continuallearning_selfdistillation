import { useState, type ReactNode } from 'react'
import { IconEye } from './icons'
import type { ApiError, Bound, ProductProfile, RecStatus } from '../api'
import { BOUND_LABEL, channelColor, channelLabel, money, signedPct } from '../format'

export function PageHead({ title, sub, actions, eyebrow }: { title: ReactNode; sub?: ReactNode; actions?: ReactNode; eyebrow?: ReactNode }) {
  return (
    <div className="page-head">
      <div className="grow">
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>{title}</h1>
        {sub && <div className="page-sub">{sub}</div>}
      </div>
      {actions && <div className="btn-row">{actions}</div>}
    </div>
  )
}

export function Section({ title, caption, children, actions }: { title: ReactNode; caption?: ReactNode; actions?: ReactNode; children: ReactNode }) {
  return (
    <section className="section">
      <div className="section-head">
        <h2>{title}</h2>
        {actions}
        {caption && <span className="caption">{caption}</span>}
      </div>
      {children}
    </section>
  )
}

/** A metric tile: label, mono value, an optional microchart, an optional foot.
 *  `flag` paints a 2px top rule in a status colour - state, never decoration. */
export function Metric({ label, value, foot, footMono, chart, flag, title }: {
  label: string; value: ReactNode; foot?: ReactNode; footMono?: boolean; chart?: ReactNode
  flag?: 'alert' | 'warn' | 'good' | 'accent'; title?: string
}) {
  return (
    <div className="tile" title={title}>
      {flag && <span className="tile-flag" style={{ background: `var(--${flag})` }} />}
      <div className="metric-label">{label}</div>
      <div className="metric-row">
        <div className="metric-value">{value}</div>
        {chart}
      </div>
      {foot !== undefined && <div className={`metric-foot${footMono ? ' mono' : ''}`}>{foot}</div>}
    </div>
  )
}

/** Tiles that share borders. */
/** What kind of product a SKU is, from how it sells (M5 has no product names).
 *  Each tag's tooltip gives the measurement behind it. */
export function ProductTags({ profile }: { profile: ProductProfile | null | undefined }) {
  if (!profile) return null
  const why = (t: string) => {
    switch (t) {
      case 'Value': case 'Everyday': case 'Premium': return `Usual shelf price $${profile.price.toFixed(2)}`
      case 'Steady': case 'Volatile': return `Day-to-day swing in demand: ${profile.cv.toFixed(2)} (std ÷ mean)`
      case 'SNAP-driven': return `Sells ${Math.round((profile.snap_lift - 1) * 100)}% more on SNAP (US food-assistance) days`
      default: return `Simulated customers' price elasticity ${profile.elasticity.toFixed(2)}: a 10% price rise cuts demand by about ${Math.abs(Math.round(profile.elasticity * 10))}%`
    }
  }
  return <span className="ptags">{profile.tags.map(t => <span key={t} className="ptag" title={why(t)}>{t}</span>)}</span>
}

export function Strip({ cols, children }: { cols: number; children: ReactNode }) {
  return <div className={`strip grid-${cols}`}>{children}</div>
}

export function ChannelKey({ channel, short }: { channel: string; short?: boolean }) {
  return (
    <span className="key">
      <span className="key-dot" style={{ background: channelColor(channel) }} />
      {short ? channel : channelLabel(channel)}
    </span>
  )
}

const STATUS: Record<RecStatus, { cls: string; label: string }> = {
  pending:    { cls: 'badge-accent', label: 'Pending' },
  approved:   { cls: 'badge-good',   label: 'Approved' },
  rejected:   { cls: 'badge-alert',  label: 'Rejected' },
  superseded: { cls: '',             label: 'Expired' },
}
/** A superseded (undecided) price either went on sale as recommended or, when
 * approval is required, never did - `live_source` says which. "Expired" was
 * the label for both, which read as "not used" for prices that were sold at. */
const UNDECIDED: Record<string, { cls: string; label: string; title: string }> = {
  auto:       { cls: '', label: 'Applied automatically', title: 'Nobody decided, so the recommended price went on sale' },
  unreviewed: { cls: '', label: 'Not applied',           title: 'Approval was required and nobody decided, so the shelf price stayed' },
}
export function StatusBadge({ status, liveSource }: { status: RecStatus; liveSource?: string | null }) {
  const u = status === 'superseded' && liveSource ? UNDECIDED[liveSource] : undefined
  if (u) return <span className={`badge ${u.cls}`} title={u.title}>{u.label}</span>
  const s = STATUS[status]
  return <span className={`badge ${s.cls}`}>{s.label}</span>
}

/** Which operator rule moved the price - always a label, never colour alone. */
export function BoundBadges({ bound, conflict }: { bound: Bound[]; conflict?: boolean }) {
  if (conflict) return <span className="badge badge-warn" title="Margin floor is above the max price; the floor wins">Rules conflict</span>
  if (!bound.length) return <span className="faint">—</span>
  return <span className="btn-row" style={{ gap: '.3rem' }}>{bound.map(b => <span key={b} className="badge">{BOUND_LABEL[b] ?? b}</span>)}</span>
}

/** "$9.00 → $10.00" when a rule moved it; just the price when not. */
export function PriceMove({ raw, final, base }: { raw: number; final: number; base?: number }) {
  const moved = Math.abs(raw - final) > 1e-9
  return (
    <span className="price-move">
      {moved && <span className="from" title="The pricing agent's own price">{money(raw)}</span>}
      <strong style={{ fontWeight: 500 }}>{money(final)}</strong>
      {base !== undefined && <span className="faint" style={{ marginLeft: '.45rem', fontSize: '.8rem' }}>{signedPct(final / base - 1)}</span>}
    </span>
  )
}

export function Callout({ kind = 'info', children }: { kind?: 'info' | 'warn' | 'alert'; children: ReactNode }) {
  return <div className={`callout callout-${kind}`}><div>{children}</div></div>
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return <div className="empty"><div className="t1">{title}</div>{children && <div style={{ fontSize: '.9rem' }}>{children}</div>}</div>
}

/** A failed fetch, named by the service that failed - never a blank page. */
export function ErrorBox({ error, what }: { error: ApiError; what?: string }) {
  const svc = error.service ? `${error.service[0].toUpperCase()}${error.service.slice(1)}` : 'Gateway'
  return (
    <Callout kind="alert">
      <strong>{what ? `Couldn't load ${what}.` : 'Something failed.'}</strong>{' '}
      {error.status === 409 ? error.message : `${svc}: ${error.message}`}
    </Callout>
  )
}

export function Tabs<T extends string>({ value, onChange, items }: { value: T; onChange: (v: T) => void; items: { id: T; label: string; count?: number }[] }) {
  return (
    <div className="tabs" role="tablist">
      {items.map(i => (
        <button key={i.id} role="tab" aria-selected={value === i.id} className={value === i.id ? 'on' : ''} onClick={() => onChange(i.id)}>
          {i.label}{i.count !== undefined && <span className="count">{i.count}</span>}
        </button>
      ))}
    </div>
  )
}

/** Page controls: range readout, first/prev/next/last, and page size. */
export function Pager({ total, offset, limit, onOffset, onLimit, sizes = [25, 50, 100] }: {
  total: number; offset: number; limit: number
  onOffset: (o: number) => void; onLimit?: (l: number) => void; sizes?: number[]
}) {
  const pages = Math.max(1, Math.ceil(total / limit))
  const page = Math.min(Math.floor(offset / limit), pages - 1)
  const go = (p: number) => onOffset(Math.max(0, Math.min(p, pages - 1)) * limit)
  return (
    <div className="pager">
      <span className="caption mono">
        {total ? `${(offset + 1).toLocaleString()}–${Math.min(offset + limit, total).toLocaleString()}` : '0'} of {total.toLocaleString()}
      </span>
      <span className="pager-btns">
        <button className="btn btn-sm" disabled={page === 0} onClick={() => go(0)} aria-label="First page">«</button>
        <button className="btn btn-sm" disabled={page === 0} onClick={() => go(page - 1)} aria-label="Previous page">‹</button>
        <span className="caption mono pager-page">{page + 1} / {pages}</span>
        <button className="btn btn-sm" disabled={page >= pages - 1} onClick={() => go(page + 1)} aria-label="Next page">›</button>
        <button className="btn btn-sm" disabled={page >= pages - 1} onClick={() => go(pages - 1)} aria-label="Last page">»</button>
      </span>
      {onLimit && (
        <select className="select" value={limit} aria-label="Rows per page"
          onChange={e => { onLimit(Number(e.target.value)); onOffset(0) }}>
          {sizes.map(s => <option key={s} value={s}>{s} / page</option>)}
        </select>
      )}
    </div>
  )
}

/** Page-head tag for a viewer: the page is for looking, by design. */
export function ReadOnly() {
  return <span className="badge" title="Viewer account: operators make these decisions">Read only</span>
}

/** A password input that shows its text only while the eye is held down
 *  (pointer, or Space / Enter from the keyboard) - it can't be left revealed. */
export function PasswordInput({ id, value, onChange, autoComplete }: {
  id: string; value: string; onChange: (v: string) => void; autoComplete: string
}) {
  const [shown, setShown] = useState(false)
  return (
    <div className="pw">
      <input id={id} className="input" type={shown ? 'text' : 'password'} autoComplete={autoComplete}
        value={value} onChange={e => onChange(e.target.value)} />
      <button type="button" className={`pw-toggle${shown ? ' on' : ''}`} tabIndex={-1}
        aria-label="Hold to show password" title="Hold to show"
        onPointerDown={e => { e.preventDefault(); setShown(true) }}
        onPointerUp={() => setShown(false)} onPointerLeave={() => setShown(false)} onPointerCancel={() => setShown(false)}
        onKeyDown={e => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); setShown(true) } }}
        onKeyUp={() => setShown(false)} onBlur={() => setShown(false)}>
        <IconEye width={15} height={15} strokeWidth={1.4} />
      </button>
    </div>
  )
}

export function Seg<T extends string>({ value, onChange, items, label }: { value: T; onChange: (v: T) => void; items: { id: T; label: ReactNode }[]; label: string }) {
  return (
    <div className="seg" role="radiogroup" aria-label={label}>
      {items.map(i => (
        <button key={i.id} role="radio" aria-checked={value === i.id} className={value === i.id ? 'on' : ''} onClick={() => onChange(i.id)}>{i.label}</button>
      ))}
    </div>
  )
}
