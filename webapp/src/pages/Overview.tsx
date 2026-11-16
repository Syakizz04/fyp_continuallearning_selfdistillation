import { Link } from 'react-router-dom'
import { api } from '../api'
import { PriceBand, SparkBars, Sparkline } from '../charts/Micro'
import { BoundBadges, ChannelKey, Empty, ErrorBox, Metric, PageHead, Pager, PriceMove, Section, Strip } from '../components/ui'
import { useApp } from '../context'
import { channelLabel, longDate, money, num, pct, shortDate } from '../format'
import { usePaged, usePoll } from '../hooks'

export default function Overview() {
  const { status, version } = useApp()
  const pending = usePoll(() => api.recommendations({ status: 'pending', limit: 500 }), 5000, [version])
  const models = usePoll(() => api.models(60), 5000)
  const inv = usePoll(api.inventory, 5000)
  const sim = usePoll(api.sim, 4000, [version])

  const s = status.data
  const recs = pending.data ?? []
  const bound = recs.filter(r => r.bound_by.length || r.conflict)
  const nodes = models.data?.nodes ?? []
  const events = nodes.flatMap(n => n.drift?.events ?? [])
    .sort((a, b) => b.tick - a.tick || a.node.localeCompare(b.node))
  const boundPage = usePaged(bound, 5)
  const eventPage = usePaged(events, 5)
  const retraining = nodes.filter(n => n.health?.retrain_in_flight).map(n => n.node)
  const partitioned = (inv.data?.channels ?? []).filter(c => c.partitioned)
  const days = (sim.data?.days ?? []).slice(-28)
  const last = days[days.length - 1]

  return (
    <>
      <PageHead title="Overview" eyebrow={longDate(s?.display_date)} />

      {status.error && !s && <ErrorBox error={status.error} what="the system status" />}

      <Strip cols={5}>
        <Metric label="Awaiting decision" value={num(s?.counts.pending)} flag={s?.counts.pending ? 'accent' : undefined}
          foot={bound.length ? `${bound.length} held by a rule` : 'none held by a rule'} />
        <Metric label="Units sold" value={num(last?.sold)}
          chart={<SparkBars data={days.map(d => ({ label: shortDate(d.display_date), v: d.sold, alt: d.refused }))} width={84} />}
          foot={last ? `${num(last.refused)} refused · ${shortDate(last.display_date)}` : 'no trading yet'} />
        <Metric label="Fill rate" value={pct(last?.fill_rate, 0)}
          chart={<Sparkline data={days.map(d => ({ label: shortDate(d.display_date), v: d.fill_rate }))} width={84} fmt={v => pct(v, 0)} />}
          foot={`${pct(sim.data?.totals.fill_rate, 1)} this run`} />
        <Metric label="Drift warnings" value={num(events.length)} flag={retraining.length ? 'warn' : undefined}
          foot={retraining.length ? `retraining: ${retraining.map(channelLabel).join(', ')}` : 'no retrain running'} />
        <Metric label="Channels offline" value={num(partitioned.length)} flag={partitioned.length ? 'alert' : undefined}
          foot={partitioned.length ? partitioned.map(c => channelLabel(c.channel)).join(', ') : `policy ${inv.data?.policy ?? '—'}`} />
      </Strip>


      <div className="grid grid-2 grid-gap section">
        <div className="card">
          <div className="card-title">Held by a rule<span className="right caption">{bound.length || ''}</span></div>
          {pending.error ? <ErrorBox error={pending.error} what="recommendations" />
            : !bound.length ? <Empty title="Nothing held" />
            : <table className="t compact" style={{ marginTop: '.5rem' }}>
                <tbody>{boundPage.rows.map(r => (
                  <tr key={r.id}>
                    <td className="mono"><Link to={`/products/${r.sku}`}>{r.sku}</Link></td>
                    <td><ChannelKey channel={r.channel} /></td>
                    <td><BoundBadges bound={r.bound_by} conflict={r.conflict} /></td>
                    <td><PriceBand raw={r.raw_price} final={r.final_price} base={r.base_price} floor={r.floor} ceiling={r.ceiling} width={84} /></td>
                    <td className="num"><PriceMove raw={r.raw_price} final={r.final_price} /></td>
                  </tr>
                ))}</tbody>
              </table>}
          {boundPage.many && <Pager {...boundPage.pager} />}
          <div style={{ marginTop: '.6rem' }}><Link to="/recommendations">Recommendations →</Link></div>
        </div>

        <div className="card">
          <div className="card-title">Drift warnings<span className="right caption">{events.length || ''}</span></div>
          {models.error ? <ErrorBox error={models.error} what="model health" />
            : !events.length ? <Empty title="No drift detected" />
            : <div className="list" style={{ marginTop: '.3rem' }}>
                {eventPage.rows.map(e => (
                  <div className="list-row" key={`${e.node}-${e.tick}-${e.kind}`}>
                    <div className="list-kind" style={{ ['--k' as string]: 'var(--alert)' }}>{e.kind === 'forecasting' ? 'Forecast' : 'Pricing'}</div>
                    <div className="list-main">
                      <ChannelKey channel={e.node} />
                      <span className="list-meta mono" style={{ marginLeft: '.6rem' }}>
                        {num(e.value, 2)} {e.kind === 'forecasting' ? '>' : '<'} {num(e.threshold, 2)}
                      </span>
                    </div>
                    <div className="list-time">{shortDate(e.display_date)}</div>
                  </div>
                ))}
              </div>}
          {eventPage.many && <Pager {...eventPage.pager} />}
          <div style={{ marginTop: '.6rem' }}><Link to="/models">Model health →</Link></div>
        </div>
      </div>

      <Section title="Channels">
        <div className="strip grid-3">
          {(inv.data?.channels ?? []).map(c => {
            const n = nodes.find(x => x.node === c.channel)
            const stream = n?.drift?.stream ?? []
            const ch = last?.channels.find(x => x.channel === c.channel)
            return (
              <div className="tile" key={c.channel}>
                <div className="card-title" style={{ marginBottom: '.5rem' }}>
                  <ChannelKey channel={c.channel} />
                  <span className="right">
                    {n?.online === false ? <span className="badge badge-alert">Node down</span>
                      : c.partitioned ? <span className="badge badge-warn">Offline</span>
                      : <span className="badge badge-good">Online</span>}
                  </span>
                </div>
                <table className="kv"><tbody>
                  <tr><td>Sold last day</td><td>{num(ch?.sold)} · {money(ch?.revenue)}</td></tr>
                  <tr><td>Model</td><td>v{n?.health?.model_generation ?? '—'}{n?.health?.retrain_in_flight ? ' · retraining' : ''}</td></tr>
                  <tr><td>Stock view off by</td><td>{num(c.mean_abs_estimate_error, 1)} u</td></tr>
                  <tr><td title="Profit achieved ÷ profit expected, per day; dashed line is the retrain threshold">Profit index</td>
                    <td><Sparkline data={stream.filter(r => r.profit_index != null).slice(-28).map(r => ({ label: shortDate(r.display_date), v: r.profit_index }))}
                      reference={n?.drift?.status.rl_threshold} width={90} height={18} fmt={v => v.toFixed(2)} color="#1A1A1A" /></td></tr>
                </tbody></table>
              </div>
            )
          })}
        </div>
      </Section>
    </>
  )
}
