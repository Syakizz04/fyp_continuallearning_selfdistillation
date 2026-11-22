import { Fragment, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { DLink } from '../components/DLink'
import { useTo } from '../design'
import { api, ApiError, type Recommendation } from '../api'
import { ForecastChart } from '../charts/Charts'
import { PriceBand, SparkBars, Sparkline } from '../charts/Micro'
import { Decide } from '../components/Decide'
import { BoundBadges, Callout, ChannelKey, Empty, ErrorBox, Metric, PageHead, Pager, PriceMove, ProductTags, Section, Seg, Strip } from '../components/ui'
import { useApp } from '../context'
import { ago, channelLabel, longDate, money, num, pct, shortDate, signedPct, units } from '../format'
import { usePoll } from '../hooks'
import { derive, LOW_COVER_DAYS, type Row } from '../stock'

const PER_PAGE = 10

// Per-product cover figures live in ../stock so the Overview's stock-risk
// card and this page use the same numbers.

type Filter = 'all' | 'pending' | 'low' | 'refused' | 'rule'
type SortKey = 'sku' | 'price' | 'sold28' | 'stockNow' | 'cover' | 'pending'

const FILTERS: { id: Filter; label: string; test: (r: Row) => boolean }[] = [
  { id: 'all', label: 'All', test: () => true },
  { id: 'pending', label: 'Needs decision', test: r => r.pending > 0 },
  { id: 'low', label: 'Low stock', test: r => r.cover !== null && r.cover < LOW_COVER_DAYS },
  { id: 'refused', label: 'Orders refused', test: r => r.refused28 > 0 },
  { id: 'rule', label: 'Own rule', test: r => r.has_rule },
]

function SortTh({ k, sort, setSort, children, num: isNum, center }: { k: SortKey; sort: [SortKey, 1 | -1]; setSort: (s: [SortKey, 1 | -1]) => void; children: React.ReactNode; num?: boolean; center?: boolean }) {
  const on = sort[0] === k
  return (
    <th className={`sortable${isNum ? ' num' : ''}`} style={center ? { textAlign: 'center' } : undefined} aria-sort={on ? (sort[1] === 1 ? 'ascending' : 'descending') : 'none'}
      onClick={() => setSort([k, on ? (sort[1] === 1 ? -1 : 1) : (k === 'sku' ? 1 : -1)])}>
      {children}{on && <span className="arrow">{sort[1] === 1 ? '↑' : '↓'}</span>}
    </th>
  )
}

/** Today's three channel recommendations for one product, decided in place. */
function RowDetail({ sku, simDate }: { sku: string; simDate?: string | null }) {
  const { toast, bump, version, canEdit } = useApp()
  const recs = usePoll(() => api.recommendations({ sku, sim_date: simDate ?? undefined, limit: 10 }), 5000, [sku, simDate, version])
  const pending = (recs.data ?? []).filter(r => r.status === 'pending')
  const approveAll = async () => {
    try {
      const out = await api.bulk(pending.map(r => r.id), 'approve')
      toast(`Approved ${out.decided.length} for ${sku}`); bump()
    } catch (e) { toast(e instanceof ApiError ? e.message : String(e), 'err') }
  }
  if (recs.error) return <ErrorBox error={recs.error} what="recommendations" />
  if (!recs.data) return <div className="caption" style={{ padding: '.6rem 0' }}>Loading…</div>
  if (!recs.data.length) return <div className="caption" style={{ padding: '.6rem 0' }}>No recommendation for this day.</div>
  return (
    <div style={{ border: '1px solid var(--border)', background: 'var(--panel)' }}>
      <table className="t compact">
        <tbody>{recs.data.map((r: Recommendation) => (
          <tr key={r.id}>
            <td style={{ width: '22%' }}><ChannelKey channel={r.channel} /></td>
            <td><PriceBand raw={r.raw_price} final={r.final_price} base={r.base_price} floor={r.floor} ceiling={r.ceiling} /></td>
            <td className="num"><PriceMove raw={r.raw_price} final={r.final_price} base={r.base_price} /></td>
            <td><BoundBadges bound={r.bound_by} conflict={r.conflict} /></td>
            <td className="num faint" title="stock this channel sees / actual">{num(r.inventory_view)} / {num(r.true_available)}</td>
            <td className="tight"><Decide rec={r} compact /></td>
          </tr>
        ))}</tbody>
      </table>
      <div className="btn-row" style={{ padding: '.5rem .6rem', borderTop: '1px solid var(--grid)' }}>
        <DLink className="btn btn-sm" to={`/products/${sku}`}>Open product</DLink>
        {canEdit && pending.length > 1 && <button className="btn btn-sm btn-primary" onClick={approveAll}>Approve all {pending.length}</button>}
      </div>
    </div>
  )
}

export function ProductList() {
  const { version, status } = useApp()
  const products = usePoll(api.products, 5000, [version])
  const [q, setQ] = useState('')
  const [filter, setFilter] = useState<Filter>('all')
  const [sort, setSort] = useState<[SortKey, 1 | -1]>(['pending', -1])
  const [open, setOpen] = useState<string | null>(null)
  const [cursor, setCursor] = useState(0)
  const nav = useNavigate()
  const to = useTo()
  const tableRef = useRef<HTMLTableElement>(null)

  const all = useMemo(() => (products.data ?? []).map(derive), [products.data])
  const rows = useMemo(() => {
    const f = FILTERS.find(x => x.id === filter)!
    const [k, dir] = sort
    return all.filter(r => f.test(r) && r.sku.toLowerCase().includes(q.toLowerCase()))
      .sort((a, b) => {
        const av = a[k], bv = b[k]
        if (av === bv) return a.sku.localeCompare(b.sku)
        if (av === null) return 1
        if (bv === null) return -1
        return (av < bv ? -1 : 1) * dir
      })
  }, [all, filter, q, sort])
  useEffect(() => { setCursor(c => Math.min(c, Math.max(rows.length - 1, 0))) }, [rows.length])
  // The cursor indexes the whole filtered list; the page follows it, so arrow
  // keys walk straight across page boundaries.
  const offset = Math.floor(cursor / PER_PAGE) * PER_PAGE
  const pageRows = rows.slice(offset, offset + PER_PAGE)
  useEffect(() => { setCursor(0) }, [filter, q, sort])

  const onKey = (e: React.KeyboardEvent) => {
    if ((e.target as HTMLElement).tagName === 'INPUT') return
    const r = rows[cursor]
    if (e.key === 'ArrowDown' || e.key === 'j') { setCursor(c => Math.min(c + 1, rows.length - 1)); e.preventDefault() }
    else if (e.key === 'ArrowUp' || e.key === 'k') { setCursor(c => Math.max(c - 1, 0)); e.preventDefault() }
    else if (e.key === 'ArrowRight' || e.key === ' ') { if (r) setOpen(o => (o === r.sku && e.key === ' ' ? null : r.sku)); e.preventDefault() }
    else if (e.key === 'ArrowLeft' || e.key === 'Escape') setOpen(null)
    else if (e.key === 'Enter' && r) nav(to(`/products/${r.sku}`))
  }

  return (
    <>
      <PageHead title="Products" eyebrow={`${all.length} products · 3 channels`} />
      <div className="filters">
        <input className="input" placeholder="Search SKU" value={q} onChange={e => setQ(e.target.value)} style={{ width: 220 }} />
        {FILTERS.map(f => {
          const n = all.filter(f.test).length
          return <button key={f.id} className={`chip${filter === f.id ? ' on' : ''}`} onClick={() => setFilter(f.id)}>{f.label}<span className="n">{n}</span></button>
        })}
        <span className="caption" style={{ marginLeft: 'auto' }} title="↑ ↓ move · → or space expand · Enter open">↑↓ → ⏎</span>
      </div>
      {products.error ? <ErrorBox error={products.error} what="products" /> : (
        <div className="table-wrap" tabIndex={0} onKeyDown={onKey} style={{ outline: 'none' }}>
          <table className="t" ref={tableRef}>
            <thead><tr>
              <th style={{ width: 28 }} />
              <SortTh k="sku" sort={sort} setSort={setSort}>SKU</SortTh>
              <SortTh k="price" sort={sort} setSort={setSort} num>Price</SortTh>
              <SortTh k="sold28" sort={sort} setSort={setSort} num>Sold · 28d</SortTh>
              <SortTh k="stockNow" sort={sort} setSort={setSort} num>Stock · 28d</SortTh>
              <SortTh k="cover" sort={sort} setSort={setSort} num>Cover</SortTh>
              <SortTh k="pending" sort={sort} setSort={setSort} num center>To decide</SortTh>
              <th>Rule</th>
            </tr></thead>
            <tbody>
              {pageRows.map((p, j) => {
                const i = offset + j
                const expanded = open === p.sku
                return (
                  <Fragment key={p.sku}>
                    <tr className={`clickable${expanded ? ' expanded' : ''}${i === cursor ? ' selected' : ''}`}
                      onClick={() => { setCursor(i); setOpen(expanded ? null : p.sku) }}>
                      <td className="mono faint" style={{ textAlign: 'center' }}>{expanded ? '−' : '+'}</td>
                      <td>
                        <DLink className="mono" to={`/products/${p.sku}`} onClick={e => e.stopPropagation()}>{p.sku}</DLink>
                        <div style={{ lineHeight: 1 }}>{p.profile
                          ? <ProductTags profile={p.profile} />
                          : <span className="faint" style={{ fontSize: '.74rem' }}>{p.dept}</span>}</div>
                      </td>
                      <td className="num" title="In-store price now; the % is against the usual shelf price">
                        <div>{money(p.price)}</div>
                        <div className="faint" style={{ fontSize: '.74rem' }}>
                          {p.price !== null && p.base_price ? signedPct(p.price / p.base_price - 1, 0) : ''}</div></td>
                      <td className="num"><span className="spark-cell">
                        <SparkBars data={p.daily.map(d => ({ v: d.sold, alt: d.refused }))} width={84} height={20} />
                        <span className="v">{num(p.sold28)}</span></span></td>
                      <td className="num"><span className="spark-cell">
                        <Sparkline data={p.daily.map(d => ({ v: d.stock }))} width={72} height={20} zero fmt={v => num(v)} />
                        <span className="v">{num(p.stockNow)}</span></span></td>
                      <td className="num" title={p.cover !== null ? `${p.cover.toFixed(1)} days at ${p.avgDaily.toFixed(1)} sold a day` : 'no recent sales'}
                        style={p.cover !== null && p.cover < LOW_COVER_DAYS ? { color: 'var(--alert)' } : undefined}>
                        {p.cover === null ? '—' : `${Math.round(p.cover)}d`}</td>
                      <td className="num" style={{ textAlign: 'center' }}>{p.pending ? <strong style={{ color: 'var(--ink)', fontWeight: 700 }}>{p.pending}</strong> : <span className="faint">0</span>}</td>
                      <td>{p.has_rule ? <span className="badge">Custom</span> : <span className="faint">—</span>}</td>
                    </tr>
                    {expanded && <tr className="detail"><td colSpan={8}><RowDetail sku={p.sku} simDate={status.data?.sim_date} /></td></tr>}
                  </Fragment>
                )
              })}
            </tbody>
          </table>
          {!products.loading && !rows.length && <Empty title="No products match" />}
          {rows.length > 0 && <Pager total={rows.length} offset={offset} limit={PER_PAGE} onOffset={o => { setCursor(o); setOpen(null) }} />}
        </div>
      )}
    </>
  )
}

export function ProductDetail() {
  const { sku = '' } = useParams()
  const { status, version, canEdit } = useApp()
  const channels = status.data?.channels ?? ['pos', 'web', 'marketplace']
  const [node, setNode] = useState('pos')
  const simDate = status.data?.sim_date
  const fc = usePoll(() => api.forecast(sku, node), 15000, [sku, node, simDate])
  const recs = usePoll(() => api.recommendations({ sku, limit: 2000 }), 4000, [sku, version])
  const [earlierAt, setEarlierAt] = useState(0)
  useEffect(() => setEarlierAt(0), [sku])
  const rules = usePoll(api.rules, null, [version])
  const products = usePoll(api.products, 10000, [version])

  const today = (recs.data ?? []).filter(r => r.sim_date === simDate)
  const earlier = (recs.data ?? []).filter(r => r.sim_date !== simDate)
  const prod = products.data?.find(p => p.sku === sku)
  const row = prod ? derive(prod) : undefined
  const stock = today[0]?.true_available ?? recs.data?.[0]?.true_available
  const base = today[0]?.base_price ?? recs.data?.[0]?.base_price
  const skuRule = rules.data?.skus.find(r => r.sku === sku)
  const f = fc.data

  return (
    <>
      <div style={{ marginBottom: '.5rem' }}><DLink to="/products" className="caption">← Products</DLink></div>
      <PageHead title={<span className="mono">{sku}</span>} eyebrow={<>{prod?.profile ? <ProductTags profile={prod.profile} /> : sku.split('_').slice(0, 2).join('_')} · {longDate(status.data?.display_date)}</>}
        actions={canEdit && <DLink className="btn" to={`/rules?sku=${sku}`}>{skuRule ? 'Edit price rule' : 'Set a price rule'}</DLink>} />

      <Strip cols={4}>
        <Metric label="In stock" value={num(stock)}
          chart={row && <Sparkline data={row.daily.map(d => ({ label: shortDate(d.sim_date), v: d.stock }))} zero width={84} fmt={v => num(v)} />}
          foot={row?.cover !== null && row?.cover !== undefined ? `≈${Math.round(row.cover)} days at the recent sales rate` : 'all channels, one pool'} />
        <Metric label="Forecast · 14 days" value={f?.daily.length ? units(f.total_p50) : '—'}
          foot={f?.daily.length ? `${units(f.daily.reduce((a, d) => a + d.p10, 0))}–${units(f.daily.reduce((a, d) => a + d.p90, 0))} units (p10–p90)` : undefined} footMono />
        <Metric label="Sold · 28 days" value={num(row?.sold28)}
          chart={row && <SparkBars data={row.daily.map(d => ({ label: shortDate(d.sim_date), v: d.sold, alt: d.refused }))} width={64} />}
          foot={row?.refused28 ? `${num(row.refused28)} refused` : `base ${money(base)}`} />
        <Metric label="Forecaster updated" value={f ? (f.forecaster_updated_display_date ? shortDate(f.forecaster_updated_display_date) : 'Base') : '—'}
          foot={f ? (f.forecaster_generation ? `retrained ${f.forecaster_generation}× · ${ago(f.forecaster_updated_at)}` : `not retrained · loaded ${ago(f.forecaster_updated_at)}`) : undefined} />
      </Strip>

      <Section title="Demand forecast"
        actions={<Seg label="Forecast from channel" value={node} onChange={setNode}
          items={channels.map(c => ({ id: c, label: <><span className="key-dot" style={{ background: `var(--ch-${c})` }} />{channelLabel(c)}</> }))} />}>
        <div className={`card${fc.refreshing && f ? ' stale' : ''}`}>
          {fc.error ? <ErrorBox error={fc.error} what="the forecast" />
            : !f ? <Empty title="Loading forecast…" />
            : !f.daily.length ? <Empty title="Not enough history">{f.reason}</Empty>
            : <ForecastChart history={f.history} daily={f.daily} />}
        </div>
      </Section>

      <Section title="Today's recommendations">
        {recs.error ? <ErrorBox error={recs.error} what="recommendations" />
          : !today.length ? <Empty title="No recommendation yet today" />
          : <div className="strip grid-3">
              {today.map(r => (
                <div className="tile" key={r.id}>
                  <div className="card-title"><ChannelKey channel={r.channel} /><span className="right"><BoundBadges bound={r.bound_by} conflict={r.conflict} /></span></div>
                  <div className="metric-row" style={{ margin: '.7rem 0 .2rem' }}>
                    <div className="metric-value"><PriceMove raw={r.raw_price} final={r.final_price} /></div>
                    <PriceBand raw={r.raw_price} final={r.final_price} base={r.base_price} floor={r.floor} ceiling={r.ceiling} width={96} />
                  </div>
                  <div className="metric-foot">margin {pct(r.margin_pct)} · {r.final_price >= r.base_price ? '+' : '−'}{pct(Math.abs(r.final_price / r.base_price - 1))} vs base</div>
                  <table className="kv" style={{ margin: '.7rem 0' }}><tbody>
                    <tr><td>Stock this channel sees</td><td>{num(r.inventory_view)}</td></tr>
                    <tr><td>Actual stock</td><td>{num(r.true_available)}</td></tr>
                    <tr><td>Unit cost (assumed)</td><td>{money(r.unit_cost)}</td></tr>
                  </tbody></table>
                  <Decide rec={r} onDone={recs.refresh} />
                </div>
              ))}
            </div>}
        {today.some(r => r.inventory_view !== null && r.true_available !== null && r.inventory_view !== r.true_available) && (
          <div style={{ marginTop: '.8rem' }}>
            <Callout>A channel whose stock view lags the real count can price differently. <DLink to="/inventory">Stock &amp; sync</DLink></Callout>
          </div>
        )}
      </Section>

      {earlier.length > 0 && (
        <Section title="Earlier">
          <div className="table-wrap">
            <table className="t">
              <thead><tr><th>Day</th><th>Channel</th><th className="num">Recommended</th><th className="num">Sold at</th><th>Rule</th><th>Outcome</th><th>By</th></tr></thead>
              <tbody>{earlier.slice(earlierAt, earlierAt + PER_PAGE).map(r => (
                <tr key={r.id}>
                  <td className="mono">{shortDate(r.display_date)}</td>
                  <td><ChannelKey channel={r.channel} /></td>
                  <td className="num"><PriceMove raw={r.raw_price} final={r.final_price} /></td>
                  <td className="num" title={r.live_source ?? undefined}>{money(r.live_price)}</td>
                  <td><BoundBadges bound={r.bound_by} conflict={r.conflict} /></td>
                  <td><Decide rec={r} /></td>
                  <td>{r.decided_by ?? <span className="faint">—</span>}</td>
                </tr>
              ))}</tbody>
            </table>
            {earlier.length > PER_PAGE && <Pager total={earlier.length} offset={earlierAt} limit={PER_PAGE} onOffset={setEarlierAt} />}
          </div>
        </Section>
      )}
    </>
  )
}
