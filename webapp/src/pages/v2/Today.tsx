import { api } from '../../api'
import { SparkBars } from '../../charts/Micro'
import { DLink } from '../../components/DLink'
import { BoundBadges, ChannelKey, Empty, ErrorBox, Metric, PageHead, Pager, PriceMove, Section, Strip } from '../../components/ui'
import { useApp } from '../../context'
import { channelLabel, longDate, money, num, pct, shortDate, units } from '../../format'
import { usePaged, usePoll } from '../../hooks'
import { driftWords, LOW_DAYS, lostSales, profitVsExpected, RISK_DAYS, stockLine } from './business'

/** v2 home: money, risk and what is waiting on a decision. */
export default function Today() {
  const { status, version } = useApp()
  const pending = usePoll(() => api.recommendations({ status: 'pending', limit: 500 }), 5000, [version])
  const models = usePoll(() => api.models(30), 5000)
  const sim = usePoll(api.sim, 4000, [version])
  const products = usePoll(api.products, 6000, [version])

  const s = status.data
  const recs = pending.data ?? []
  const bound = recs.filter(r => r.bound_by.length || r.conflict)
  const nodes = models.data?.nodes ?? []
  const events = nodes.flatMap(n => n.drift?.events ?? []).sort((a, b) => b.tick - a.tick || a.node.localeCompare(b.node))
  const boundPage = usePaged(bound, 5)
  const eventPage = usePaged(events, 5)
  const updating = nodes.filter(n => n.health?.retrain_in_flight).map(n => n.node)
  const days = (sim.data?.days ?? []).slice(-28)
  const last = days[days.length - 1]
  const lost28 = days.reduce((a, d) => a + (lostSales(d) ?? 0), 0)
  const refused28 = days.reduce((a, d) => a + d.refused, 0)
  const pve = profitVsExpected(nodes)
  const low = (products.data ?? []).map(stockLine)
    .filter(l => l.daysLeft !== null && l.daysLeft < LOW_DAYS)
    .sort((a, b) => (a.daysLeft ?? 0) - (b.daysLeft ?? 0))

  return (
    <>
      <PageHead title="Today" eyebrow={longDate(s?.display_date)} />
      {status.error && !s && <ErrorBox error={status.error} what="the system status" />}

      <Strip cols={5}>
        <Metric label="Prices awaiting decision" value={num(s?.counts.pending)} flag={s?.counts.pending ? 'accent' : undefined}
          foot={bound.length ? `${bound.length} held by a rule` : 'none held by a rule'} />
        <Metric label="Revenue" value={money(last?.revenue)}
          foot={last ? `${num(last.sold)} units · ${shortDate(last.display_date)}` : 'no trading yet'} />
        <Metric label="Lost sales" value={money(lostSales(last))} flag={lostSales(last) ? 'warn' : undefined}
          title="Orders turned away for lack of stock, at that day's selling price. An estimate."
          foot={days.length ? `${money(lost28)} · ${num(refused28)} units over ${days.length} days` : 'no trading yet'} />
        <Metric label="Profit vs expected" value={pct(pve, 0)} flag={pve !== null && pve < 0.9 ? 'warn' : undefined}
          title="Profit the prices earned, against what the pricing model expected them to earn, on its latest daily check"
          foot="latest daily check, all channels" />
        <Metric label="Drift warnings" value={num(events.length)} flag={updating.length ? 'warn' : undefined}
          foot={updating.length ? `updating: ${updating.map(channelLabel).join(', ')}` : 'models up to date'} />
      </Strip>

      <div className="grid grid-2 grid-gap section">
        <div className="card">
          <div className="card-title">Running low<span className="right caption">{low.length ? `under ${LOW_DAYS} days of stock` : ''}</span></div>
          {products.error ? <ErrorBox error={products.error} what="stock" />
            : !low.length ? <Empty title={`Every product has ${LOW_DAYS}+ days of stock`} />
            : <div style={{ overflowX: 'auto', marginTop: '.5rem' }}><table className="t compact">
                <thead><tr><th>Product</th><th className="num">In stock</th><th className="num">Sells / day</th><th className="num">Days left</th></tr></thead>
                <tbody>{low.slice(0, 7).map(l => (
                  <tr key={l.sku}>
                    <td className="mono"><DLink to={`/products/${l.sku}`}>{l.sku}</DLink></td>
                    <td className="num">{num(l.inStock)}</td>
                    <td className="num">{units(l.perDay)}</td>
                    <td className="num">{(l.daysLeft ?? 0) < RISK_DAYS
                      ? <span className="badge badge-alert">{num(l.daysLeft, 1)}</span> : num(l.daysLeft, 1)}</td>
                  </tr>
                ))}</tbody>
              </table></div>}
          <div style={{ marginTop: '.6rem' }}><DLink to="/stock">All stock →</DLink></div>
        </div>

        <div className="card">
          <div className="card-title">Prices held by a rule<span className="right caption">{bound.length || ''}</span></div>
          {pending.error ? <ErrorBox error={pending.error} what="recommendations" />
            : !bound.length ? <Empty title="No price is held back by a rule" />
            : <div style={{ overflowX: 'auto', marginTop: '.5rem' }}><table className="t compact">
                <tbody>{boundPage.rows.map(r => (
                  <tr key={r.id}>
                    <td className="mono"><DLink to={`/products/${r.sku}`}>{r.sku}</DLink></td>
                    <td><ChannelKey channel={r.channel} /></td>
                    <td><BoundBadges bound={r.bound_by} conflict={r.conflict} /></td>
                    <td className="num"><PriceMove raw={r.raw_price} final={r.final_price} base={r.base_price} /></td>
                  </tr>
                ))}</tbody>
              </table></div>}
          {boundPage.many && <Pager {...boundPage.pager} />}
          <div style={{ marginTop: '.6rem' }}><DLink to="/recommendations">Decide prices →</DLink></div>
        </div>
      </div>

      <div className="grid grid-2 grid-gap section">
        <div className="card">
          <div className="card-title">Drift warnings<span className="right caption">{events.length || ''}</span></div>
          {models.error ? <ErrorBox error={models.error} what="model health" />
            : !events.length ? <Empty title="No drift detected" />
            : <div className="list" style={{ marginTop: '.3rem' }}>
                {eventPage.rows.map(e => (
                  <div className="list-row" key={`${e.node}-${e.tick}-${e.kind}`}>
                    <div className="list-kind" style={{ ['--k' as string]: 'var(--alert)' }}>{e.kind === 'forecasting' ? 'Demand' : 'Pricing'}</div>
                    <div className="list-main">
                      <ChannelKey channel={e.node} />
                      <span className="list-meta" style={{ marginLeft: '.6rem' }}>{driftWords(e.kind)}</span>
                    </div>
                    <div className="list-time">{shortDate(e.display_date)}</div>
                  </div>
                ))}
              </div>}
          {eventPage.many && <Pager {...eventPage.pager} />}
          <div style={{ marginTop: '.6rem' }}><DLink to="/models">Model health →</DLink></div>
        </div>

        <div className="card">
          <div className="card-title">By channel<span className="right caption">{last ? shortDate(last.display_date) : ''}</span></div>
          {!last ? <Empty title="No trading yet" /> : (
            <div style={{ overflowX: 'auto', marginTop: '.5rem' }}><table className="t compact">
              <thead><tr><th>Channel</th><th className="num">Units</th><th className="num">Revenue</th><th className="num">Lost</th><th className="num">Profit vs exp.</th></tr></thead>
              <tbody>{last.channels.map(c => {
                const n = nodes.find(x => x.node === c.channel)
                const off = s?.sim?.partition.includes(c.channel)
                return (
                  <tr key={c.channel}>
                    <td><ChannelKey channel={c.channel} />{off && <span className="badge badge-warn" style={{ marginLeft: '.4rem' }}>Offline</span>}
                      {n?.health?.retrain_in_flight && <span className="badge" style={{ marginLeft: '.4rem' }}>Updating</span>}</td>
                    <td className="num">{num(c.sold)}</td>
                    <td className="num">{money(c.revenue)}</td>
                    <td className="num">{c.refused ? money(c.refused * (c.mean_price || 0)) : '—'}</td>
                    <td className="num">{pct(n?.drift?.status.last_profit_index, 0)}</td>
                  </tr>
                )
              })}</tbody>
            </table></div>
          )}
        </div>
      </div>

      <Section title="Last 28 days">
        <div className="strip grid-3">
          <Metric label="Units sold" value={num(days.reduce((a, d) => a + d.sold, 0))}
            chart={<SparkBars data={days.map(d => ({ label: shortDate(d.display_date), v: d.sold, alt: d.refused }))} width={120} />}
            foot={`${num(refused28)} turned away`} />
          <Metric label="Revenue" value={money(days.reduce((a, d) => a + d.revenue, 0))}
            foot={sim.data ? `${money(sim.data.totals.revenue)} this run` : undefined} />
          <Metric label="Orders filled" value={pct(sim.data?.totals.fill_rate, 1)}
            foot="share of units asked for that were sold, this run" />
        </div>
      </Section>
    </>
  )
}
