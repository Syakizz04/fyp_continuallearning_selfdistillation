import { useMemo, useState } from 'react'
import { api } from '../../api'
import { SparkBars } from '../../charts/Micro'
import { DLink } from '../../components/DLink'
import { Empty, ErrorBox, Metric, PageHead, Section, Seg, Strip } from '../../components/ui'
import { money, num, shortDate, units } from '../../format'
import { usePoll } from '../../hooks'
import { LOW_DAYS, lostSales, RISK_DAYS, stockLine } from './business'

type Filter = 'all' | 'low' | 'refused'

/** v2 stock: what is on the shelf, how long it lasts, and what was turned away.
 *  How each channel's stock view is kept in sync lives on "Sync details". */
export default function Stock() {
  const products = usePoll(api.products, 5000)
  const sim = usePoll(api.sim, 5000)
  const [filter, setFilter] = useState<Filter>('all')

  const lines = useMemo(() => (products.data ?? []).map(p => ({ ...stockLine(p), daily: p.daily.slice(-28) }))
    .sort((a, b) => (a.daysLeft ?? Infinity) - (b.daysLeft ?? Infinity)), [products.data])
  const shown = lines.filter(l => filter === 'all'
    || (filter === 'low' && l.daysLeft !== null && l.daysLeft < LOW_DAYS)
    || (filter === 'refused' && l.refused28 > 0))
  const low = lines.filter(l => l.daysLeft !== null && l.daysLeft < LOW_DAYS).length
  const risk = lines.filter(l => l.daysLeft !== null && l.daysLeft < RISK_DAYS).length
  const days = (sim.data?.days ?? []).slice(-28)
  const lost28 = days.reduce((a, d) => a + (lostSales(d) ?? 0), 0)
  const onHand = lines.reduce((a, l) => a + (l.inStock ?? 0), 0)

  return (
    <>
      <PageHead title="Stock" eyebrow={`${lines.length} products`} />
      {products.error && <ErrorBox error={products.error} what="stock" />}

      <Strip cols={4}>
        <Metric label="Units on hand" value={num(onHand)} foot="across all products" />
        <Metric label="Running low" value={num(low)} flag={low ? 'warn' : undefined} foot={`under ${LOW_DAYS} days of stock`} />
        <Metric label="At risk of running out" value={num(risk)} flag={risk ? 'alert' : undefined} foot={`under ${RISK_DAYS} days of stock`} />
        <Metric label="Lost sales · 28 days" value={money(lost28)} flag={lost28 ? 'warn' : undefined}
          title="Orders turned away for lack of stock, at the selling price that day. An estimate."
          foot={`${num(days.reduce((a, d) => a + d.refused, 0))} units turned away`} />
      </Strip>

      <Section title="By product" actions={
        <Seg label="Show" value={filter} onChange={setFilter}
          items={[{ id: 'all', label: 'All' }, { id: 'low', label: 'Running low' }, { id: 'refused', label: 'Turned orders away' }]} />}>
        {!products.data ? <Empty title="Loading…" />
          : !shown.length ? <Empty title="No products match" />
          : <div className="table-wrap">
              <table className="t">
                <thead><tr><th>Product</th><th className="num">In stock</th><th className="num">Sells / day</th>
                  <th className="num">Days left</th><th className="num">Turned away · 28d</th><th>Sales · 28 days</th></tr></thead>
                <tbody>{shown.map(l => (
                  <tr key={l.sku}>
                    <td className="mono"><DLink to={`/products/${l.sku}`}>{l.sku}</DLink></td>
                    <td className="num">{num(l.inStock)}</td>
                    <td className="num">{units(l.perDay)}</td>
                    <td className="num">{l.daysLeft === null ? '—'
                      : l.daysLeft < RISK_DAYS ? <span className="badge badge-alert">{num(l.daysLeft, 1)}</span>
                      : l.daysLeft < LOW_DAYS ? <span className="badge badge-warn">{num(l.daysLeft, 1)}</span>
                      : num(l.daysLeft, 1)}</td>
                    <td className="num">{l.refused28 ? num(l.refused28) : '—'}</td>
                    <td><SparkBars data={l.daily.map(d => ({ label: shortDate(d.sim_date), v: d.sold, alt: d.refused }))} width={120} /></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>}
        <div className="caption" style={{ marginTop: '.6rem' }}>
          Restocking runs on a fixed reorder rule. How each channel's view of stock is kept in step: <DLink to="/inventory">Sync details →</DLink>
        </div>
      </Section>
    </>
  )
}
