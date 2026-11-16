import { api } from '../api'
import { CheckPanels } from '../charts/Charts'
import { Sparkline } from '../charts/Micro'
import { ChannelKey, Empty, ErrorBox, PageHead, Pager, Section } from '../components/ui'
import { ago, channelColor, num, shortDate } from '../format'
import { usePaged, usePoll } from '../hooks'

const KIND = { forecasting: 'Demand forecaster', rl: 'Pricing model' } as const

export default function Models() {
  const models = usePoll(() => api.models(240), 4000)
  const d = models.data
  const nodes = d?.nodes ?? []
  const online = nodes.filter(n => n.online && n.drift)
  const events = online.flatMap(n => n.drift!.events).sort((a, b) => b.tick - a.tick || a.node.localeCompare(b.node))
  const eventPage = usePaged(events, 5)
  // A realised-demand threshold is calibrated per node, like the rolling floor.
  const fcLive = online.some(n => n.drift?.status.fc_signal === 'realised')
  const fcThr = fcLive ? null : online[0]?.drift?.status.fc_threshold ?? null
  // A rolling floor is calibrated per node, so one line cannot stand for all.
  const rlThr = online.some(n => n.drift?.status.rl_detector === 'rolling') ? null
    : online[0]?.drift?.status.rl_threshold ?? null
  const series = online.map(n => ({ node: n.node, rows: n.drift!.stream }))
  // Each channel's own floor (rolling mode calibrates one per channel).
  const panels = online.map(n => ({ node: n.node, rows: n.drift!.stream, threshold: n.drift!.status.rl_threshold ?? rlThr }))
  const fcPanels = online.map(n => ({ node: n.node, rows: n.drift!.stream, threshold: n.drift!.status.fc_threshold ?? fcThr }))
  const hasMase = series.some(s => s.rows.some(r => r.mase !== null))

  return (
    <>
      <PageHead title="Model health" eyebrow={d ? `retraining: SDFT · α ${num(d.sdft_alpha, 2)}` : undefined} />

      {models.error && <ErrorBox error={models.error} what="model health" />}

      <div className="strip grid-3">
        {nodes.map(n => {
          const h = n.health, st = n.drift?.status, fc = h?.forecaster
          return (
            <div className="tile" key={n.node}>
              <div className="card-title">
                <ChannelKey channel={n.node} />
                <span className="right">{!n.online ? <span className="badge badge-alert">Unreachable</span>
                  : h?.retrain_in_flight ? <span className="badge badge-warn">Retraining</span>
                  : <span className="badge badge-good">Serving</span>}</span>
              </div>
              {!n.online ? <div className="caption" style={{ marginTop: '.5rem' }}>{n.error}</div> : (
                <>
                  <div className="metric-row" style={{ margin: '.7rem 0 .5rem' }}>
                    <div>
                      <div className="metric-label" title="Profit achieved / profit expected. Below the dashed line two days running triggers a retrain.">Profit index</div>
                      <div className="metric-value">{num(st?.last_profit_index, 2)}</div>
                    </div>
                    <Sparkline data={(n.drift?.stream ?? []).filter(r => r.profit_index != null).slice(-40).map(r => ({ label: shortDate(r.display_date), v: r.profit_index }))}
                      reference={st?.rl_threshold} width={120} height={34} fmt={v => v.toFixed(3)} color={channelColor(n.node)} />
                  </div>
                  <table className="kv"><tbody>
                    <tr><td title="Retrain when the profit index is below this two days running. Provisional: set from the new model's first week and refined daily until its full calibration">Pricing floor</td><td>{st?.rl_provisional ? `${num(st.rl_threshold, 3)} · provisional` : st?.rl_warmup_left ? `calibrating · ${st.rl_warmup_left}d` : num(st?.rl_threshold, 3)}</td></tr>
                    <tr><td>Version</td><td>v{num(h?.model_generation)}</td></tr>
                    <tr><td>Forecaster</td><td>{fc?.forecaster_generation ? `${shortDate(fc.forecaster_updated_sim_date)} · ${fc.forecaster_generation}×` : 'base'}</td></tr>
                    <tr><td title="Mean absolute scaled error; below 1 beats a naive forecast">Forecast MASE</td><td>{num(st?.last_mase, 3)}</td></tr>
                    <tr><td title="Retrain the forecaster when MASE is above this several checks running">Forecast alarm</td><td>{st?.fc_provisional ? `${num(st.fc_threshold, 3)} · provisional` : st?.fc_warmup_left ? `calibrating · ${st.fc_warmup_left}d` : num(st?.fc_threshold, 3)}</td></tr>
                    <tr><td>Last forecast check</td><td>{h?.last_fc_check ? ago(h.last_fc_check.ts) : 'not yet'}</td></tr>
                    <tr><td>Checks · warnings</td><td>{num(st?.checks)} · {num(st?.n_events)}</td></tr>
                  </tbody></table>
                </>
              )}
            </div>
          )
        })}
      </div>


      <Section title="Daily pricing check · profit index">
        <div className="card">
          {!series.length ? <Empty title="No checks yet" />
            : <CheckPanels series={panels} metric="profit_index" events={events} label="profit index" />}
        </div>
      </Section>

      <Section title={fcLive ? 'Daily forecast check · MASE on realised demand' : 'Weekly forecast check · MASE'}>
        <div className="card">
          {!hasMase ? <Empty title="No forecast checks yet">{fcLive
              ? 'The first runs once a channel has traded a full 14-day forecast horizon.'
              : 'The first runs after a week of trading.'}</Empty>
            : <CheckPanels series={fcPanels} metric="mase" events={events} label="MASE" fmt={v => num(v, 2)} />}
        </div>
      </Section>

      <Section title="Drift warnings">
        {!events.length ? <Empty title="No drift detected" /> : (
          <div className="table-wrap">
            <table className="t">
              <thead><tr><th>Trading day</th><th>Channel</th><th>Model</th><th className="num">Value</th><th className="num">Threshold</th></tr></thead>
              <tbody>{eventPage.rows.map(e => (
                <tr key={`${e.node}-${e.tick}-${e.kind}`}>
                  <td className="mono">{shortDate(e.display_date ?? e.sim_date)}</td>
                  <td><ChannelKey channel={e.node} /></td>
                  <td>{KIND[e.kind]}</td>
                  <td className="num">{num(e.value, 3)}</td>
                  <td className="num faint">{e.kind === 'rl' ? '< ' : '> '}{num(e.threshold, 3)}</td>
                </tr>
              ))}</tbody>
            </table>
            {eventPage.many && <Pager {...eventPage.pager} />}
          </div>
        )}
      </Section>

      <Section title="Model versions">
        {!d?.versions.length ? <Empty title="No retrains yet" /> : (
          <div className="table-wrap">
            <table className="t">
              <thead><tr><th>Trading day</th><th>Channel</th><th>Model</th><th className="num">Version</th><th>Why</th><th className="num">Took</th><th>Result</th></tr></thead>
              <tbody>{d.versions.map(v => (
                <tr key={v.id}>
                  <td className="mono">{shortDate(v.display_date ?? v.sim_date)}</td>
                  <td><ChannelKey channel={v.node} /></td>
                  <td>{KIND[v.kind] ?? v.kind}</td>
                  <td className="num">v{v.generation}</td>
                  <td className="muted">{v.reason}</td>
                  <td className="num">{v.duration_s ? `${num(v.duration_s, 0)} s` : '—'}</td>
                  <td>{v.error ? <span className="badge badge-alert" title={v.error}>Failed</span>
                    : v.skipped ? <span className="badge">Skipped</span>
                    : v.ok === 0 ? <span className="badge badge-warn">Kept old model</span>
                    : <span className="badge badge-good">Deployed</span>}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </Section>
    </>
  )
}
