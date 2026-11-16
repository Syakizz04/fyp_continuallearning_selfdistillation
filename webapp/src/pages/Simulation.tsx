import { useState } from 'react'
import { DLink } from '../components/DLink'
import { api, ApiError, type SimState } from '../api'
import { SimDaysChart } from '../charts/Charts'
import { SparkBars, Sparkline } from '../charts/Micro'
import { Callout, ChannelKey, Empty, ErrorBox, Metric, PageHead, Section, Seg, Strip } from '../components/ui'
import { useApp } from '../context'
import { clockTime, longDate, money, num, pct, shortDate } from '../format'
import { usePoll } from '../hooks'

const MODE: Record<string, { label: string; cls: string }> = {
  starting: { label: 'Setting up', cls: 'badge-warn' }, resetting: { label: 'Resetting', cls: 'badge-warn' },
  paused: { label: 'Paused', cls: '' }, playing: { label: 'Running', cls: 'badge-accent' },
  finished: { label: 'Run complete', cls: 'badge-good' }, error: { label: 'Error', cls: 'badge-alert' },
}
const SPEED_LABEL: Record<string, string> = { '5': '5 s', '2': '2 s', '1': '1 s', '0': 'Max' }
const POLICY_LABEL: Record<string, string> = {
  escrow_quota: 'Quota (escrow)', strong_lock: 'Central lock', eventual: 'Eventual',
}
const LOG_COLOR: Record<string, string> = {
  drift: 'var(--alert)', retrain: 'var(--warn)', model: 'var(--good)', outage: 'var(--alert)',
  shock: 'var(--warn)', warn: 'var(--alert)', run: 'var(--ink)', restock: 'var(--accent)',
}
const SHOCKS = [
  { id: '0.3', label: '×0.3' }, { id: '0.6', label: '×0.6' }, { id: '1.5', label: '×1.5' }, { id: '2.5', label: '×2.5' },
]

export default function Simulation() {
  const { toast, bump, status } = useApp()
  const playing = status.data?.sim?.mode === 'playing'
  const sim = usePoll(api.sim, playing ? 1200 : 2500)
  const s = sim.data
  const [busy, setBusy] = useState(false)
  const [shockMult, setShockMult] = useState('0.3')
  const [shockDays, setShockDays] = useState('10')
  const [shockSku, setShockSku] = useState('')
  const [restockSku, setRestockSku] = useState('')
  const [restockQty, setRestockQty] = useState('')
  const [policy, setPolicy] = useState('')

  const act = async (action: Parameters<typeof api.simAction>[0], body?: Record<string, unknown>, ok?: string) => {
    setBusy(true)
    try { await api.simAction(action, body); if (ok) toast(ok); bump(); sim.refresh() }
    catch (e) { toast(e instanceof ApiError ? e.message : String(e), 'err') }
    finally { setBusy(false) }
  }

  if (sim.error && !s) return <><PageHead title="Simulation" /><ErrorBox error={sim.error} what="the simulator" /></>
  if (!s) return <><PageHead title="Simulation" /><Empty title="Connecting…" /></>

  const settingUp = s.mode === 'starting' || s.mode === 'resetting'
  const locked = settingUp || s.mode === 'error'
  const days = s.days
  const last = days[days.length - 1]
  const recent = days.slice(-28)
  const m = MODE[s.mode] ?? { label: s.mode, cls: '' }

  return (
    <>
      <PageHead title="Simulation" eyebrow={<>replaying Walmart M5 sales · <span className="mono">{s.sim_date ?? '—'}</span></>} />

      {s.error && <div style={{ marginBottom: '1rem' }}><Callout kind="alert"><strong>Simulator error.</strong> <span className="mono">{s.error}</span></Callout></div>}
      {settingUp && <div style={{ marginBottom: '1rem' }}><Callout kind="warn"><strong>{m.label}…</strong> reloading stock and base models on every channel. This takes about a minute.</Callout></div>}

      {/* ── clock ── */}
      <div className="strip" style={{ gridTemplateColumns: '1.25fr 1fr 1fr' }}>
        <div className="tile">
          <div className="card-title"><span className="metric-label">Trading day</span><span className="right"><span className={`badge ${m.cls}`}>{m.label}</span></span></div>
          <div className="metric-value" style={{ marginTop: '.35rem' }}>{longDate(s.display_sim_date)}</div>
          <div className="metric-foot mono">day {num(s.tick + 1)} of {num(s.total)} · {s.tick} traded</div>
          <div className="progress" style={{ marginTop: '.6rem' }}><span style={{ width: `${(s.tick / Math.max(s.total, 1)) * 100}%` }} /></div>
        </div>
        <div className="tile">
          <div className="metric-label">Clock</div>
          <div className="btn-row" style={{ marginTop: '.6rem' }}>
            <div className="btn-join">
              {s.mode === 'playing'
                ? <button className="btn" disabled={busy} onClick={() => act('pause')}>Pause</button>
                : <button className="btn btn-primary" disabled={busy || locked || s.mode === 'finished'} onClick={() => act('play')}>Play</button>}
              <button className="btn" disabled={busy || locked || s.mode !== 'paused'} onClick={() => act('step')}>Next day</button>
            </div>
          </div>
          <div style={{ marginTop: '.7rem', display: 'flex', alignItems: 'center', gap: '.6rem' }}>
            <span className="label">Pace</span>
            <Seg label="Seconds per trading day" value={String(s.speed_s)} onChange={v => act('speed', { speed_s: Number(v) })}
              items={s.speeds.map(v => ({ id: String(v), label: SPEED_LABEL[String(v)] ?? `${v} s` }))} />
          </div>
        </div>
        <div className="tile">
          <div className="metric-label">Going live</div>
          <label className="switch" style={{ marginTop: '.7rem' }}>
            <input type="checkbox" checked={s.require_approval} disabled={busy}
              onChange={e => act('settings', { require_approval: e.target.checked })} />
            Only approved prices go live
          </label>
          <div className="metric-foot" style={{ marginTop: '.5rem' }}>
            {s.require_approval
              ? 'Undecided prices keep the shelf price.'
              : 'Undecided prices go live as recommended.'} Rejected ones always keep the shelf price.
          </div>
        </div>
      </div>

      {/* ── readout ── */}
      <Section title={last ? `Last trading day · ${shortDate(last.display_date)}` : 'Last trading day'}>
        <Strip cols={5}>
          <Metric label="Units sold" value={num(last?.sold)}
            chart={<SparkBars data={recent.map(d => ({ label: shortDate(d.display_date), v: d.sold, alt: d.refused }))} width={80} />}
            foot={`${num(s.totals.sold)} this run`} />
          <Metric label="Refused" value={num(last?.refused)} flag={last?.refused ? 'warn' : undefined}
            chart={<SparkBars data={recent.map(d => ({ label: shortDate(d.display_date), v: d.refused }))} width={80} color="#8A8A84" />}
            foot={`${num(s.totals.refused)} this run`} />
          <Metric label="Fill rate" value={pct(last?.fill_rate, 0)}
            chart={<Sparkline data={recent.map(d => ({ label: shortDate(d.display_date), v: d.fill_rate }))} width={80} fmt={v => pct(v, 0)} />}
            foot={`${pct(s.totals.fill_rate, 1)} this run`} />
          <Metric label="Revenue" value={money(last?.revenue)}
            chart={<Sparkline data={recent.map(d => ({ label: shortDate(d.display_date), v: d.revenue }))} width={80} zero fmt={v => money(v)} />}
            foot={`${money(s.totals.revenue)} this run`} />
          <Metric label="Oversold" value={num(last?.oversold)} flag={s.totals.oversold ? 'alert' : undefined}
            foot={`${num(s.totals.oversold)} this run`} />
        </Strip>
        <div className="card" style={{ marginTop: '1rem' }}>
          {days.length ? <SimDaysChart days={days} /> : <Empty title="No trading days yet">Press Play or Next day.</Empty>}
        </div>
      </Section>

      {/* ── disruptions ── */}
      <Section title="Disrupt">
        <div className="strip grid-4">
          <div className="tile">
            <div className="metric-label">Channel links</div>
            <div style={{ marginTop: '.6rem', display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
              {s.channels.map(ch => {
                const off = s.partition.includes(ch)
                return (
                  <label key={ch} className="switch" style={{ justifyContent: 'space-between' }}>
                    <ChannelKey channel={ch} />
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '.45rem' }}>
                      <span className={off ? 'badge badge-alert' : 'faint'} style={{ fontSize: '.72rem' }}>{off ? 'Offline' : 'Online'}</span>
                      <input type="checkbox" checked={!off} disabled={busy || locked}
                        onChange={e => act('outage', { channel: ch, offline: !e.target.checked })} />
                    </span>
                  </label>
                )
              })}
            </div>
            <div className="metric-foot" style={{ marginTop: '.6rem' }}>An offline channel sells from the quota it holds.</div>
          </div>

          <div className="tile">
            <div className="metric-label">Network delay</div>
            <div className="metric-value" style={{ marginTop: '.35rem' }}>{num(s.delay_ms)}<span className="faint" style={{ fontSize: '.9rem' }}> ms</span></div>
            <div style={{ marginTop: '.6rem' }}>
              <Seg label="Network delay" value={String(s.delay_ms)} onChange={v => act('latency', { delay_ms: Number(v) })}
                items={[0, 50, 200, 1000].map(v => ({ id: String(v), label: String(v) }))} />
            </div>
            <div className="metric-foot" style={{ marginTop: '.6rem' }}>Each trip from a channel to central stock.</div>
          </div>

          <div className="tile">
            <div className="metric-label">Demand shock</div>
            {s.shock ? (
              <>
                <div className="metric-value" style={{ marginTop: '.35rem' }}>×{s.shock.multiplier}</div>
                <div className="metric-foot">{s.shock.sku ?? 'every product'} · until {shortDate(s.shock.display_until)}</div>
                <button className="btn btn-sm" style={{ marginTop: '.7rem' }} disabled={busy} onClick={() => act('clear-shock', {}, 'Demand back to normal')}>End shock</button>
              </>
            ) : (
              <div style={{ marginTop: '.6rem', display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
                <Seg label="Demand multiplier" value={shockMult} onChange={setShockMult} items={SHOCKS} />
                <div style={{ display: 'flex', gap: '.4rem' }}>
                  <div className="input-affix" style={{ width: 92 }}>
                    <input className="input mono" value={shockDays} onChange={e => setShockDays(e.target.value)} inputMode="numeric" aria-label="Days" />
                    <span>days</span>
                  </div>
                  <select className="select mono" value={shockSku} onChange={e => setShockSku(e.target.value)} style={{ flex: 1 }} aria-label="Product">
                    <option value="">All products</option>
                    {s.skus.map(k => <option key={k} value={k}>{k}</option>)}
                  </select>
                </div>
                <button className="btn btn-sm" disabled={busy || locked || !(Number(shockDays) >= 1)}
                  onClick={() => act('shock', { multiplier: Number(shockMult), days: Number(shockDays), sku: shockSku || null })}>Start</button>
              </div>
            )}
          </div>

          <div className="tile">
            <div className="metric-label">Restock</div>
            <div style={{ marginTop: '.6rem', display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
              <select className="select mono" value={restockSku} onChange={e => setRestockSku(e.target.value)} aria-label="Product to restock">
                <option value="">All products</option>
                {s.skus.map(k => <option key={k} value={k}>{k}</option>)}
              </select>
              <div className="input-affix">
                <input className="input mono" placeholder="to target" value={restockQty} onChange={e => setRestockQty(e.target.value)} inputMode="numeric" aria-label="Units" />
                <span>units</span>
              </div>
              <button className="btn btn-sm" disabled={busy || locked}
                onClick={() => act('restock', { sku: restockSku || null, qty: restockQty ? Number(restockQty) : null }, 'Delivered')}>Deliver</button>
            </div>
          </div>
        </div>
      </Section>

      {/* ── what happened ── */}
      <div className="grid grid-2 grid-gap section">
        <div className="card">
          <div className="card-title">Event log</div>
          {!s.log.length ? <Empty title="Nothing yet" /> : (
            <div className="list" style={{ marginTop: '.3rem', maxHeight: 360, overflowY: 'auto' }}>
              {s.log.map((e, i) => (
                <div className="list-row" key={i} style={{ gridTemplateColumns: '5.5rem 1fr auto' }}>
                  <div className="list-kind" style={{ ['--k' as string]: LOG_COLOR[e.kind] ?? 'var(--ink-3)' }}>{e.kind}</div>
                  <div className="list-main">{e.text}</div>
                  <div className="list-time">{shortDate(e.display_date)} <span className="faint">{clockTime(e.ts)}</span></div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="card">
          <div className="card-title">Prices that went live<span className="right caption">this run</span></div>
          <PriceSources s={s} />
          {last && (
            <div style={{ overflowX: 'auto', marginTop: '1rem' }}>
            <table className="t compact">
              <thead><tr><th>Channel · last day</th><th className="num">Sold</th><th className="num">Refused</th><th className="num">Revenue</th><th className="num">Avg price</th></tr></thead>
              <tbody>{last.channels.map(c => (
                <tr key={c.channel}>
                  <td><ChannelKey channel={c.channel} /></td>
                  <td className="num">{num(c.sold)}</td><td className="num">{num(c.refused)}</td>
                  <td className="num">{money(c.revenue)}</td><td className="num">{money(c.mean_price)}</td>
                </tr>
              ))}</tbody>
            </table>
            </div>
          )}
          <div style={{ marginTop: '.8rem' }}><DLink to="/recommendations">Decide today's prices →</DLink></div>
        </div>
      </div>

      {/* ── run ── */}
      <Section title="Run">
        <div className="card">
          <div className="grid grid-gap" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)', alignItems: 'start' }}>
            <div style={{ fontSize: '.88rem', color: 'var(--ink-2)', maxWidth: '68ch' }}>
              <p style={{ marginTop: 0 }}>Each trading day replays one day of real Walmart (M5) sales across the in-store till, web store and marketplace, which share one stock pool. A day opens with a recommended price per product per channel; when it trades, the prices that went live meet simulated shoppers.</p>
              <p>Each channel checks its pricing model every day (profit index) and its forecaster every week (MASE). Two bad checks in a row retrain that model locally with SDFT, which learns the new regime while distilling from the current model so it doesn't forget the old one.</p>
              <p style={{ marginBottom: 0 }}>A demand shock changes what shoppers buy, so it moves the daily pricing check; the forecaster is scored against recorded sales.</p>
            </div>
            <div className="field">
              <label htmlFor="policy">Sync policy for the next run</label>
              <select id="policy" className="select" value={policy || s.policy} onChange={e => setPolicy(e.target.value)}>
                {s.policies.map(p => <option key={p} value={p}>{POLICY_LABEL[p] ?? p}</option>)}
              </select>
              <span className="hint">current: {POLICY_LABEL[s.policy] ?? s.policy}</span>
              <button className="btn btn-danger" style={{ marginTop: '.5rem' }} disabled={busy || settingUp}
                onClick={() => {
                  if (window.confirm('Start a new run? Stock, models, recommendations and decisions reset. Decisions are archived, rules are kept.'))
                    act('reset', { policy: policy || s.policy }, 'Resetting…')
                }}>Reset run</button>
            </div>
          </div>
        </div>
      </Section>
    </>
  )
}

function PriceSources({ s }: { s: SimState }) {
  const p = s.totals.prices
  const total = p.approved + p.rejected + p.auto + p.unreviewed
  const items = [
    { k: 'approved', label: 'Approved', color: 'var(--good)' },
    { k: 'auto', label: 'Recommended, undecided', color: 'var(--accent)' },
    { k: 'rejected', label: 'Rejected → shelf price', color: 'var(--alert)' },
    { k: 'unreviewed', label: 'Waiting → shelf price', color: 'var(--ink-3)' },
  ] as const
  if (!total) return <Empty title="No trading days yet" />
  return (
    <div style={{ marginTop: '.6rem' }}>
      <div style={{ display: 'flex', height: 10, gap: 2 }}>
        {items.filter(i => p[i.k]).map(i => <span key={i.k} title={`${i.label}: ${p[i.k]}`} style={{ flex: p[i.k], background: i.color }} />)}
      </div>
      <table className="kv" style={{ marginTop: '.6rem' }}><tbody>
        {items.map(i => (
          <tr key={i.k}><td><span className="key"><span className="key-dot" style={{ background: i.color }} />{i.label}</span></td>
            <td>{num(p[i.k])} <span className="faint">· {pct(p[i.k] / total, 0)}</span></td></tr>
        ))}
      </tbody></table>
    </div>
  )
}
