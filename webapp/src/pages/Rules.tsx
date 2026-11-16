import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, ApiError, type GlobalRule, type Preview, type SkuRule } from '../api'
import { Callout, Empty, ErrorBox, Metric, PageHead, ReadOnly, Section, Strip } from '../components/ui'
import { useApp } from '../context'
import { ago, money, pct } from '../format'
import { usePoll } from '../hooks'

// Forms hold percentages as the operator types them ("20"), the API takes
// fractions (0.2). Empty = no limit.
const toPct = (v: number | null | undefined) => (v === null || v === undefined ? '' : String(+(v * 100).toFixed(4)))
const fromPct = (s: string) => (s.trim() === '' ? null : Number(s) / 100)
const toNum = (v: number | null | undefined) => (v === null || v === undefined ? '' : String(v))
const fromNum = (s: string) => (s.trim() === '' ? null : Number(s))

function PctField({ id, label, hint, value, onChange }: { id: string; label: string; hint?: string; value: string; onChange: (v: string) => void }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <div className="input-affix"><input id={id} className="input mono" inputMode="decimal" placeholder="no limit" value={value} onChange={e => onChange(e.target.value)} /><span>%</span></div>
      {hint && <span className="hint">{hint}</span>}
    </div>
  )
}
function MoneyField({ id, label, hint, value, onChange, placeholder }: { id: string; label: string; hint?: string; value: string; onChange: (v: string) => void; placeholder?: string }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <div className="input-affix"><input id={id} className="input mono" inputMode="decimal" placeholder={placeholder ?? 'inherit'} value={value} onChange={e => onChange(e.target.value)} /><span>$</span></div>
      {hint && <span className="hint">{hint}</span>}
    </div>
  )
}

function PreviewLine({ p }: { p: Preview | undefined }) {
  if (!p) return null
  if (!p.pending) return <span className="caption">No pending prices to compare.</span>
  return (
    <span className="caption">
      Changes <strong style={{ color: 'var(--ink)' }}>{p.changed}</strong> of {p.pending} pending prices
      {p.examples.length > 0 && <> · e.g. <span className="mono">{p.examples[0].sku}</span> {p.examples[0].channel} {money(p.examples[0].current)} → {money(p.examples[0].draft)}</>}
    </span>
  )
}

export default function Rules() {
  const { version, toast, bump, canEdit } = useApp()
  const rules = usePoll(api.rules, null, [version])
  const products = usePoll(api.products, null)
  const [params, setParams] = useSearchParams()

  // ── global ───────────────────────────────────────────────────────────────
  const [g, setG] = useState({ margin: '', discount: '', markup: '', cost: '' })
  const [gPreview, setGPreview] = useState<Preview>()
  useEffect(() => {
    const r = rules.data?.global
    if (r) setG({ margin: toPct(r.min_margin_pct), discount: toPct(r.max_discount_pct), markup: toPct(r.max_markup_pct), cost: toPct(r.unit_cost_ratio) })
  }, [rules.data?.global])
  const draftGlobal = (): GlobalRule => ({
    min_margin_pct: fromPct(g.margin), max_discount_pct: fromPct(g.discount),
    max_markup_pct: fromPct(g.markup), unit_cost_ratio: fromPct(g.cost) ?? 0.7,
  })
  useEffect(() => {
    if (!rules.data || !canEdit) return
    const t = window.setTimeout(() => {
      api.preview({ global_rule: draftGlobal() }).then(setGPreview).catch(() => setGPreview(undefined))
    }, 300)
    return () => window.clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [g, rules.data])

  const save = async (fn: () => Promise<unknown>, msg: string) => {
    try { await fn(); toast(msg); bump() } catch (e) { toast(e instanceof ApiError ? e.message : String(e), 'err') }
  }

  // ── per-SKU ──────────────────────────────────────────────────────────────
  const [sku, setSku] = useState(params.get('sku') ?? '')
  const [s, setS] = useState({ min: '', max: '', margin: '', cost: '' })
  const [sPreview, setSPreview] = useState<Preview>()
  const existing = rules.data?.skus.find(r => r.sku === sku)
  useEffect(() => {
    setS(existing ? { min: toNum(existing.min_price), max: toNum(existing.max_price), margin: toPct(existing.min_margin_pct), cost: toNum(existing.unit_cost) }
      : { min: '', max: '', margin: '', cost: '' })
  }, [sku, existing])
  const draftSku = (): Omit<SkuRule, 'sku'> => ({
    min_price: fromNum(s.min), max_price: fromNum(s.max), min_margin_pct: fromPct(s.margin), unit_cost: fromNum(s.cost),
  })
  useEffect(() => {
    if (!sku || !canEdit) { setSPreview(undefined); return }
    const t = window.setTimeout(() => {
      api.preview({ sku, sku_rule: draftSku() }).then(setSPreview).catch(() => setSPreview(undefined))
    }, 300)
    return () => window.clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s, sku])
  const base = products.data?.find(p => p.sku === sku)?.base_price
  const ratio = rules.data?.global.unit_cost_ratio ?? 0.7

  const gr = rules.data?.global
  const skuTable = (edit: boolean) => (
    <Section title="Products with their own rule">
      {!rules.data?.skus.length ? <Empty title="No product rules" /> : (
        <div className="table-wrap">
          <table className="t">
            <thead><tr><th>SKU</th><th className="num">Min price</th><th className="num">Max price</th><th className="num">Min margin</th><th className="num">Unit cost</th><th>Changed</th>{edit && <th />}</tr></thead>
            <tbody>{rules.data.skus.map(r => (
              <tr key={r.sku}>
                <td className="mono">{r.sku}</td>
                <td className="num">{money(r.min_price)}</td><td className="num">{money(r.max_price)}</td>
                <td className="num">{r.min_margin_pct === null ? <span className="faint">global</span> : pct(r.min_margin_pct)}</td>
                <td className="num">{r.unit_cost === null ? <span className="faint">assumed</span> : money(r.unit_cost)}</td>
                <td className="caption">{r.updated_by} · {ago(r.updated_at)}</td>
                {edit && <td><button className="btn btn-sm" onClick={() => { setSku(r.sku); setParams({ sku: r.sku }); window.scrollTo({ top: 0, behavior: 'smooth' }) }}>Edit</button></td>}
              </tr>
            ))}</tbody>
          </table>
        </div>
      )}
    </Section>
  )

  if (!canEdit) {
    const limit = (v: number | null | undefined) => (v === null || v === undefined ? 'none' : pct(v))
    return (
      <>
        <PageHead title="Pricing rules" actions={<ReadOnly />} />
        {rules.error && <ErrorBox error={rules.error} what="pricing rules" />}
        <Section title="All products" caption={gr?.updated_by ? `last changed by ${gr.updated_by} · ${ago(gr.updated_at)}` : undefined}>
          <Strip cols={4}>
            <Metric label="Minimum margin" value={limit(gr?.min_margin_pct)} foot="floor = cost ÷ (1 − margin)" />
            <Metric label="Max discount" value={limit(gr?.max_discount_pct)} foot="below base price" />
            <Metric label="Max markup" value={limit(gr?.max_markup_pct)} foot="above base price" />
            <Metric label="Assumed unit cost" value={limit(gr?.unit_cost_ratio)} foot="share of base price · an assumption" />
          </Strip>
        </Section>
        {skuTable(false)}
      </>
    )
  }

  return (
    <>
      <PageHead title="Pricing rules" />

      {rules.error && <ErrorBox error={rules.error} what="pricing rules" />}

      <Callout>
        <strong>Unit cost is assumed</strong> at a fixed share of base price; the sales data has no costs. Set a real one per product below. Margin = (price − cost) ÷ price.
      </Callout>

      <Section title="All products" caption={gr?.updated_by ? `last changed by ${gr.updated_by} · ${ago(gr.updated_at)}` : undefined}>
        <div className="card">
          <div className="grid grid-4 grid-gap">
            <PctField id="g-margin" label="Minimum margin" hint="floor = cost ÷ (1 − margin)" value={g.margin} onChange={v => setG({ ...g, margin: v })} />
            <PctField id="g-discount" label="Max discount" hint="model range: −10%" value={g.discount} onChange={v => setG({ ...g, discount: v })} />
            <PctField id="g-markup" label="Max markup" hint="model range: +10%" value={g.markup} onChange={v => setG({ ...g, markup: v })} />
            <PctField id="g-cost" label="Assumed unit cost" hint="share of base price" value={g.cost} onChange={v => setG({ ...g, cost: v })} />
          </div>
          <div className="btn-row" style={{ marginTop: '1rem' }}>
            <button className="btn btn-primary" onClick={() => save(() => api.putGlobal(draftGlobal()), 'Rules saved')}>Save rules</button>
            <PreviewLine p={gPreview} />
          </div>
        </div>
      </Section>

      <Section title="One product">
        <div className="card">
          <div className="grid grid-4 grid-gap" style={{ alignItems: 'start' }}>
            <div className="field">
              <label htmlFor="s-sku">Product</label>
              <select id="s-sku" className="select mono" value={sku} onChange={e => { setSku(e.target.value); setParams(e.target.value ? { sku: e.target.value } : {}) }}>
                <option value="">Choose a product</option>
                {(products.data ?? []).map(p => <option key={p.sku} value={p.sku}>{p.sku}{p.has_rule ? '  •' : ''}</option>)}
              </select>
              {sku && <span className="hint">base price {money(base)}{existing ? ' · has a custom rule' : ''}</span>}
            </div>
            <MoneyField id="s-min" label="Minimum price" value={s.min} onChange={v => setS({ ...s, min: v })} />
            <MoneyField id="s-max" label="Maximum price" value={s.max} onChange={v => setS({ ...s, max: v })} />
            <PctField id="s-margin" label="Minimum margin" hint="blank = global" value={s.margin} onChange={v => setS({ ...s, margin: v })} />
          </div>
          <div className="grid grid-4 grid-gap" style={{ marginTop: '1rem' }}>
            <MoneyField id="s-cost" label="Unit cost" value={s.cost} onChange={v => setS({ ...s, cost: v })}
              placeholder={base ? `${(base * ratio).toFixed(2)} (assumed)` : 'assumed'} hint="blank = assumed" />
          </div>
          <div className="btn-row" style={{ marginTop: '1rem' }}>
            <button className="btn btn-primary" disabled={!sku} onClick={() => save(() => api.putSku(sku, draftSku()), `Rule saved for ${sku}`)}>Save product rule</button>
            {existing && <button className="btn btn-danger" onClick={() => save(() => api.deleteSku(sku), `Removed the rule for ${sku}`)}>Remove rule</button>}
            {sku && <PreviewLine p={sPreview} />}
          </div>
        </div>
      </Section>

      {skuTable(true)}
    </>
  )
}
