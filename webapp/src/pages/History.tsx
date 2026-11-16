import { useEffect, useState } from 'react'
import { DLink } from '../components/DLink'
import { api, type HistoryItem } from '../api'
import { ChannelKey, Empty, ErrorBox, PageHead, Pager } from '../components/ui'
import { useApp } from '../context'
import { channelLabel, clockTime, money, pct, shortDate } from '../format'
import { usePoll } from '../hooks'

type Kind = 'all' | HistoryItem['kind']
const PER_PAGE = 10

const FIELD: Record<string, string> = {
  min_margin_pct: 'min margin', max_discount_pct: 'max discount', max_markup_pct: 'max markup',
  unit_cost_ratio: 'unit cost', min_price: 'min price', max_price: 'max price', unit_cost: 'unit cost',
}
const fmtField = (k: string, v: number | null | undefined) =>
  v === null || v === undefined ? 'none' : k.endsWith('_pct') || k === 'unit_cost_ratio' ? pct(v) : money(v)

function ruleDiff(before: Record<string, number | null> | null, after: Record<string, number | null> | null) {
  if (!after) return 'removed'
  const keys = Object.keys(FIELD).filter(k => (before && k in before) || k in after)
  const changed = keys.filter(k => (before?.[k] ?? null) !== (after[k] ?? null))
  if (!changed.length) return 'no change'
  return changed.map(k => `${FIELD[k]} ${fmtField(k, before?.[k])} → ${fmtField(k, after[k])}`).join(' · ')
}

const KIND: Record<HistoryItem['kind'], { label: string; color: string }> = {
  decision: { label: 'Decision', color: 'var(--accent)' },
  rule: { label: 'Rule', color: 'var(--ink)' },
  recommendations: { label: 'Prices', color: 'var(--ink-3)' },
  model: { label: 'Model', color: 'var(--warn)' },
}

function Row({ it }: { it: HistoryItem }) {
  let main
  switch (it.kind) {
    case 'decision':
      main = <><strong>{it.user}</strong> {it.action === 'approve' ? 'approved' : 'rejected'}{' '}
        <DLink className="mono" to={`/products/${it.sku}`}>{it.sku}</DLink> · <ChannelKey channel={it.channel} /> · <span className="mono">{money(it.price)}</span>
        {it.note && <span className="muted"> · “{it.note}”</span>}</>
      break
    case 'rule':
      main = <><strong>{it.user}</strong> changed {it.scope === 'global' ? 'the global rule' : <>the rule for <DLink className="mono" to={`/products/${it.sku}`}>{it.sku}</DLink></>}
        <div className="list-meta mono">{ruleDiff(it.before, it.after)}</div></>
      break
    case 'recommendations':
      main = <><span className="mono">{it.count}</span> prices proposed{it.bound ? <> · <span className="mono">{it.bound}</span> held by a rule</> : ''}</>
      break
    case 'model':
      main = <>{channelLabel(it.node)} {it.model_kind === 'forecasting' ? 'forecaster' : 'pricing model'} → <strong className="mono">v{it.generation}</strong>
        <span className="muted"> · {it.reason}{it.ok === 0 ? ' · kept old model' : ''}</span></>
      break
  }
  return (
    <div className="list-row">
      <div className="list-kind" style={{ ['--k' as string]: KIND[it.kind].color }}>{KIND[it.kind].label}</div>
      <div className="list-main">{main}</div>
      <div className="list-time">{it.display_date ? shortDate(it.display_date) : ''} <span className="faint">{clockTime(it.ts)}</span></div>
    </div>
  )
}

export default function History() {
  const { version } = useApp()
  const [kind, setKind] = useState<Kind>('all')
  const [offset, setOffset] = useState(0)
  useEffect(() => setOffset(0), [kind])
  const hist = usePoll(() => api.history(400), 5000, [version])
  const all = hist.data ?? []
  const items = all.filter(i => kind === 'all' || i.kind === kind)
  const n = (k: Kind) => (k === 'all' ? all.length : all.filter(i => i.kind === k).length)
  const chips: { id: Kind; label: string }[] = [
    { id: 'all', label: 'Everything' }, { id: 'decision', label: 'Decisions' }, { id: 'recommendations', label: 'Prices' },
    { id: 'rule', label: 'Rules' }, { id: 'model', label: 'Models' },
  ]

  return (
    <>
      <PageHead title="History" eyebrow="signed and append-only" />
      <div className="filters">
        {chips.map(c => <button key={c.id} className={`chip${kind === c.id ? ' on' : ''}`} onClick={() => setKind(c.id)}>{c.label}<span className="n">{n(c.id)}</span></button>)}
      </div>
      {hist.error ? <ErrorBox error={hist.error} what="history" />
        : !items.length ? <Empty title="Nothing here yet" />
        : <div className="card" style={{ padding: '.2rem 1.1rem' }}>
            <div className="list">{items.slice(offset, offset + PER_PAGE).map((it, i) => <Row key={`${it.kind}-${it.ts}-${offset + i}`} it={it} />)}</div>
            <Pager total={items.length} offset={Math.min(offset, Math.max(0, items.length - 1))} limit={PER_PAGE} onOffset={setOffset} />
          </div>}
    </>
  )
}
