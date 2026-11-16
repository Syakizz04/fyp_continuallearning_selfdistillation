import { useMemo, useState } from 'react'
import { DLink } from '../components/DLink'
import { api } from '../api'
import { BeliefErrorChart } from '../charts/Charts'
import { Sparkline } from '../charts/Micro'
import { ChannelKey, Empty, ErrorBox, Metric, PageHead, Pager, Section, Seg, Strip } from '../components/ui'
import { ago, channelLabel, num, pct, shortDate } from '../format'
import { usePaged, usePoll } from '../hooks'

const POLICY_LABEL: Record<string, string> = {
  escrow_quota: 'Quota: each channel holds a share and refills from central',
  strong_lock: 'Central lock: every sale checks central first',
  eventual: 'Eventual: channels sell on their own count and reconcile later',
}

const EVENT_LABEL: Record<string, string> = {
  reserve: 'Reserved', commit: 'Sold', release: 'Released', expire: 'Reservation expired',
  refill: 'Quota refilled', oversell: 'Oversold', replenish: 'Restocked',
}

export default function Inventory({ title = 'Stock & sync' }: { title?: string }) {
  const inv = usePoll(api.inventory, 3000)
  const [kind, setKind] = useState<string>('all')
  const events = usePoll(() => api.inventoryEvents(80), 3000)
  const products = usePoll(api.products, 6000)
  const daily = Object.fromEntries((products.data ?? []).map(p => [p.sku, p.daily]))
  const d = inv.data
  const channels = d?.channels.map(c => c.channel) ?? []
  const m = d?.metrics ?? {}

  const worst = useMemo(() => [...(d?.skus ?? [])]
    .sort((a, b) => Math.max(...Object.values(b.estimate_errors).map(Math.abs), 0) - Math.max(...Object.values(a.estimate_errors).map(Math.abs), 0))
    .slice(0, 14), [d])
  const ev = (events.data ?? []).filter(e => kind === 'all' || (kind === 'refused' ? e.kind === 'reserve' && e.granted === 0 : e.kind === kind))
  const evPage = usePaged(ev, 5)

  return (
    <>
      <PageHead title={title}
        eyebrow={d ? <>policy <strong>{d.policy}</strong> · {POLICY_LABEL[d.policy] ?? ''}</> : undefined} />

      {inv.error && <ErrorBox error={inv.error} what="stock" />}

      <Strip cols={4}>
        <Metric label="Orders accepted" value={pct(1 - (m.rejection_rate ?? 0))}
          foot={`${num(m.reserve_granted)} of ${num(m.reserve_attempts)} reservations`} />
        <Metric label="Oversold" value={num(m.oversell_units)} foot={`units · ${num(m.oversell_events)} events`} flag={m.oversell_units ? 'alert' : undefined} />
        <Metric label="Quota refills" value={num(m.quota_refills)} foot={`${num(m.central_roundtrips)} trips to central`} />
        <Metric label="Reserve latency" value={`${num(m.latency_p50_ms, 1)} ms`} foot={`p99 ${num(m.latency_p99_ms, 1)} ms`} footMono />
      </Strip>

      <Section title="Channels">
        <div className="strip grid-3">
          {(d?.channels ?? []).map(c => (
            <div className="tile" key={c.channel}>
              <div className="card-title">
                <ChannelKey channel={c.channel} />
                <span className="right">{c.partitioned
                  ? <span className="badge badge-warn">Offline</span>
                  : <span className="badge badge-good">Online</span>}</span>
              </div>
              <table className="kv" style={{ marginTop: '.6rem' }}>
                <tbody>
                  <tr><td>Quota held</td><td>{num(c.quota_units)}</td></tr>
                  <tr><td>Stale views</td><td>{num(c.skus_with_stale_view)} / {num(d?.skus.length)}</td></tr>
                  <tr><td>View error · mean</td><td>{num(c.mean_abs_estimate_error, 1)} u</td></tr>
                  <tr><td>View error · worst</td><td>{num(c.max_abs_estimate_error)} u</td></tr>
                  <tr><td>Trips to central</td><td>{num(c.sync_hops)}</td></tr>
                </tbody>
              </table>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Stock-view error by product">
        <div className="card">
          {!worst.length ? <Empty title="No stock loaded" /> :
            <BeliefErrorChart channels={channels} rows={worst.map(r => ({ sku: r.sku, errors: r.estimate_errors }))} />}
        </div>
      </Section>

      <Section title="Stock by product">
        <div className="table-wrap">
          <table className="t">
            <thead><tr>
              <th>SKU</th><th className="num">Stock · 28d</th><th className="num">On hand</th><th className="num">Reserved</th>
              {channels.map(c => <th key={c} className="num">{channelLabel(c)} sees</th>)}
            </tr></thead>
            <tbody>{(d?.skus ?? []).map(r => (
              <tr key={r.sku}>
                <td className="mono"><DLink to={`/products/${r.sku}`}>{r.sku}</DLink></td>
                <td className="num"><span className="spark-cell">
                  <Sparkline data={(daily[r.sku] ?? []).map(x => ({ label: shortDate(x.sim_date), v: x.stock }))} zero width={72} height={18} fmt={v => num(v)} />
                  <span className="v">{num(r.true_available)}</span></span></td>
                <td className="num">{num(r.total)}</td>
                <td className="num">{num(r.reserved)}</td>
                {channels.map(c => {
                  const err = r.estimate_errors[c] ?? 0
                  return <td key={c} className="num">{num(r.stock_estimates[c] ?? r.node_views[c])}
                    {err !== 0 && <span className="faint" style={{ marginLeft: '.35rem', fontSize: '.8rem' }}>{err > 0 ? '+' : '−'}{num(Math.abs(err))}</span>}</td>
                })}
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Section>

      <Section title="Stock events"
        actions={<Seg label="Event type" value={kind} onChange={k => { setKind(k); evPage.pager.onOffset(0) }} items={[
          { id: 'all', label: 'All' }, { id: 'commit', label: 'Sales' }, { id: 'refused', label: 'Refused' },
          { id: 'refill', label: 'Refills' }, { id: 'oversell', label: 'Oversold' },
        ]} />}>
        {events.error ? <ErrorBox error={events.error} what="stock events" /> : !ev.length ? <Empty title="No events of this type yet" /> : (
          <div className="table-wrap">
            <table className="t">
              <thead><tr><th>When</th><th>Event</th><th>Channel</th><th>SKU</th><th className="num">Qty</th><th className="num">Channel saw / actual</th></tr></thead>
              <tbody>{evPage.rows.map(e => (
                <tr key={e.id}>
                  <td className="mono faint">{ago(e.ts)}</td>
                  <td>{e.kind === 'reserve' && e.granted === 0 ? <span className="badge badge-alert">Refused</span>
                    : e.kind === 'oversell' ? <span className="badge badge-alert">Oversold</span>
                    : EVENT_LABEL[e.kind] ?? e.kind}</td>
                  <td>{e.node ? <ChannelKey channel={e.node} /> : <span className="faint">central</span>}</td>
                  <td className="mono">{e.sku ?? '—'}</td>
                  <td className="num">{num(e.qty)}</td>
                  <td className="num">{e.node_view === null || e.node_view === undefined ? '—' : <>{num(e.node_view)} <span className="faint">/ {num(e.true_available)}</span></>}</td>
                </tr>
              ))}</tbody>
            </table>
            {evPage.many && <Pager {...evPage.pager} />}
          </div>
        )}
      </Section>
    </>
  )
}
