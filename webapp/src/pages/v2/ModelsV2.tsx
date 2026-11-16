import { api } from '../../api'
import { CheckPanels } from '../../charts/Charts'
import { ChannelKey, Empty, ErrorBox, PageHead, Pager, Section, Strip } from '../../components/ui'
import { num, pct, shortDate } from '../../format'
import { usePaged, usePoll } from '../../hooks'
import { driftWords } from './business'

const KIND = { forecasting: 'Demand forecast', rl: 'Pricing' } as const

/** v2 model health: is each channel's forecasting and pricing still earning its
 *  keep, and when did it last adapt. The raw readings stay one hover away. */
export default function ModelsV2() {
  const models = usePoll(() => api.models(240), 4000)
  const d = models.data
  const nodes = d?.nodes ?? []
  const online = nodes.filter(n => n.online && n.drift)
  const events = online.flatMap(n => n.drift!.events).sort((a, b) => b.tick - a.tick || a.node.localeCompare(b.node))
  const eventPage = usePaged(events, 5)
  const fcLive = online.some(n => n.drift?.status.fc_signal === 'realised')
  const fcThr = fcLive ? null : online[0]?.drift?.status.fc_threshold ?? null
  const rlThr = online.some(n => n.drift?.status.rl_detector === 'rolling') ? null : online[0]?.drift?.status.rl_threshold ?? null
  const series = online.map(n => ({ node: n.node, rows: n.drift!.stream }))
  // Each channel's own floor (rolling mode calibrates one per channel).
  const panels = online.map(n => ({ node: n.node, rows: n.drift!.stream, threshold: n.drift!.status.rl_threshold ?? rlThr }))
  const fcPanels = online.map(n => ({ node: n.node, rows: n.drift!.stream, threshold: n.drift!.status.fc_threshold ?? fcThr }))
  const hasMase = series.some(s => s.rows.some(r => r.mase !== null))
  const updates = (d?.versions ?? []).filter(v => !v.error && !v.skipped && v.ok !== 0)

  return (
    <>
      <PageHead title="Model health" eyebrow="models adapt on their own when demand or prices drift" />
      {models.error && <ErrorBox error={models.error} what="model health" />}

      <Strip cols={3}>
        {nodes.map(n => {
          const st = n.drift?.status, h = n.health
          const fcBad = st?.last_mase != null && st?.fc_threshold != null && st.last_mase > st.fc_threshold
          const rlBad = st?.last_profit_index != null && st?.rl_threshold != null && st.last_profit_index < st.rl_threshold
          const state = !n.online ? 'Unreachable' : h?.retrain_in_flight ? 'Updating' : (fcBad || rlBad) ? 'Drifting' : 'Healthy'
          return (
            <div className="tile" key={n.node}>
              <span className="tile-flag" style={{ background: `var(--${state === 'Unreachable' ? 'alert' : state === 'Healthy' ? 'good' : 'warn'})` }} />
              <div className="card-title"><ChannelKey channel={n.node} />
                <span className="right"><span className={`badge ${state === 'Healthy' ? 'badge-good' : state === 'Unreachable' ? 'badge-alert' : 'badge-warn'}`}>{state}</span></span></div>
              {!n.online ? <div className="caption" style={{ marginTop: '.5rem' }}>{n.error}</div> : <table className="kv" style={{ marginTop: '.6rem' }}><tbody>
                <tr><td title="Profit the prices earned against what the pricing model expected">Profit vs expected</td><td>{pct(st?.last_profit_index, 0)}</td></tr>
                <tr><td title={`Forecast error relative to a simple "same as last period" guess (MASE ${num(st?.last_mase, 3)}); below 1 is better than that guess`}>Forecast error</td>
                  <td>{st?.last_mase == null ? 'not checked yet' : `${num(st.last_mase, 2)}×`}</td></tr>
                <tr><td>Forecast last updated</td><td>{h?.forecaster?.forecaster_generation ? shortDate(h.forecaster.forecaster_updated_sim_date) : 'never (original)'}</td></tr>
                <tr><td>Updates so far</td><td>{num(updates.filter(v => v.node === n.node).length)}</td></tr>
              </tbody></table>}
            </div>
          )
        })}
      </Strip>

      <Section title="Profit vs expected, daily" caption="a channel below its line two days running updates its pricing model">
        <div className="card">
          {!series.length ? <Empty title="No checks yet" />
            : <CheckPanels series={panels} metric="profit_index" events={events}
                label="profit vs expected" fmt={v => pct(v, 0)} thresholdLabel="Update threshold" />}
        </div>
      </Section>

      <Section title="Forecast error, daily" caption="× a simple guess; a channel above its alarm several days running updates its forecast">
        <div className="card">
          {!hasMase ? <Empty title="No forecast checks yet">The first runs once a channel has traded a full 14-day forecast horizon.</Empty>
            : <CheckPanels series={fcPanels} metric="mase" events={events}
                label="forecast error" fmt={v => `${num(v, 2)}×`} thresholdLabel="Update threshold" />}
        </div>
      </Section>

      <Section title="Drift warnings">
        {!events.length ? <Empty title="No drift detected" /> : (
          <div className="table-wrap">
            <table className="t">
              <thead><tr><th>Day</th><th>Channel</th><th>What happened</th></tr></thead>
              <tbody>{eventPage.rows.map(e => (
                <tr key={`${e.node}-${e.tick}-${e.kind}`}>
                  <td className="mono">{shortDate(e.display_date ?? e.sim_date)}</td>
                  <td><ChannelKey channel={e.node} /></td>
                  <td title={`reading ${num(e.value, 3)} against ${num(e.threshold, 3)}`}>{driftWords(e.kind)}</td>
                </tr>
              ))}</tbody>
            </table>
            {eventPage.many && <Pager {...eventPage.pager} />}
          </div>
        )}
      </Section>

      <Section title="Model updates">
        {!d?.versions.length ? <Empty title="No updates yet" /> : (
          <div className="table-wrap">
            <table className="t">
              <thead><tr><th>Day</th><th>Channel</th><th>Model</th><th>Result</th></tr></thead>
              <tbody>{d.versions.map(v => (
                <tr key={v.id}>
                  <td className="mono">{shortDate(v.display_date ?? v.sim_date)}</td>
                  <td><ChannelKey channel={v.node} /></td>
                  <td title={v.reason}>{KIND[v.kind] ?? v.kind}</td>
                  <td>{v.error ? <span className="badge badge-alert" title={v.error}>Failed, kept the old model</span>
                    : v.skipped ? <span className="badge">Skipped</span>
                    : v.ok === 0 ? <span className="badge badge-warn">Kept the old model</span>
                    : <span className="badge badge-good">Updated</span>}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </Section>
    </>
  )
}
