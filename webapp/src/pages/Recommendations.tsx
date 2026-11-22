import { useEffect, useState } from 'react'
import { DLink } from '../components/DLink'
import { api, ApiError, type RecStatus, type Recommendation } from '../api'
import { Decide } from '../components/Decide'
import { PriceBand } from '../charts/Micro'
import { BoundBadges, Callout, ChannelKey, Empty, ErrorBox, PageHead, Pager, PriceMove, ReadOnly, StatusBadge, Tabs } from '../components/ui'
import { useApp } from '../context'
import { ago, channelLabel, longDate, money, num, pct, recContext, shortDate } from '../format'
import { usePoll } from '../hooks'

type Tab = 'pending' | 'mine' | 'all'

export default function Recommendations({ title = 'Recommendations' }: { title?: string }) {
  const { status, user, version, toast, bump, canEdit } = useApp()
  const [tab, setTab] = useState<Tab>('pending')
  const [channel, setChannel] = useState('')
  const [q, setQ] = useState('')
  const [onlyBound, setOnlyBound] = useState(false)
  const [statusFilter, setStatusFilter] = useState<RecStatus | ''>('')
  const [picked, setPicked] = useState<Set<number>>(new Set())
  const [busy, setBusy] = useState(false)
  const [offset, setOffset] = useState(0)
  const [limit, setLimit] = useState(10)
  // Rows decided one at a time on this view, kept where they were (showing
  // their new status) so the row below does not slide up under the pointer
  // and catch a second click. Cleared whenever the view itself changes.
  const [held, setHeld] = useState<Map<number, { rec: Recommendation; index: number }>>(new Map())

  // Any filter change starts again from the first page.
  useEffect(() => { setOffset(0); setPicked(new Set()) }, [tab, channel, statusFilter, q, onlyBound])
  useEffect(() => { setHeld(new Map()) }, [tab, channel, statusFilter, q, onlyBound, offset, limit, status.data?.display_date])

  const recs = usePoll(() => api.recommendationsPage({
    status: tab === 'pending' ? 'pending' : (statusFilter || undefined),
    channel: channel || undefined, q: q.trim() || undefined, bound: onlyBound, limit, offset,
  }), 4000, [tab, channel, statusFilter, q, onlyBound, limit, offset, version])
  const mine = usePoll(() => api.decisions(user || undefined), tab === 'mine' ? 5000 : null, [tab, user, version])

  const fetched = recs.data?.items ?? []
  const rows = (() => {
    if (!held.size) return fetched
    const out = fetched.filter(r => !held.has(r.id))
    ;[...held.values()].sort((a, b) => a.index - b.index)
      .forEach(h => out.splice(Math.min(h.index, out.length), 0, h.rec))
    return out
  })()
  const hold = (d: Recommendation) => setHeld(m => {
    const n = new Map(m); n.set(d.id, { rec: d, index: rows.findIndex(x => x.id === d.id) }); return n
  })
  const total = recs.data?.total ?? 0
  // Deciding rows on the last page can empty it: step back rather than show nothing.
  useEffect(() => { if (recs.data && offset > 0 && offset >= total) setOffset(Math.max(0, Math.ceil(total / limit) - 1) * limit) },
    [recs.data, offset, total, limit])
  const pendingIds = canEdit ? rows.filter(r => r.status === 'pending').map(r => r.id) : []
  const allPicked = pendingIds.length > 0 && pendingIds.every(id => picked.has(id))
  const counts = status.data?.counts
  // Undecided prices are not left waiting: when the next trading day opens they
  // go on sale as recommended, unless the simulator requires approval. The tab
  // used to say "Awaiting decision" either way, which read as "nothing happens
  // until I act".
  const gated = status.data?.sim?.require_approval ?? false
  const pendingLabel = gated ? 'Needs approval' : 'Goes live unless you act'

  const toggle = (id: number) => setPicked(p => { const n = new Set(p); if (n.has(id)) n.delete(id); else n.add(id); return n })
  const bulk = async (action: 'approve' | 'reject') => {
    setBusy(true)
    try {
      const out = await api.bulk([...picked], action)
      toast(`${action === 'approve' ? 'Approved' : 'Rejected'} ${out.decided.length}` +
        (out.failed.length ? ` · ${out.failed.length} skipped (already decided or expired)` : ''), out.failed.length ? 'err' : 'ok')
      setPicked(new Set()); bump()
    } catch (e) { toast(e instanceof ApiError ? e.message : String(e), 'err') }
    finally { setBusy(false) }
  }

  return (
    <>
      <PageHead title={title} eyebrow={longDate(status.data?.display_date)} actions={!canEdit && <ReadOnly />} />

      <Tabs value={tab} onChange={t => { setTab(t); setPicked(new Set()) }} items={[
        { id: 'pending', label: pendingLabel, count: counts?.pending },
        ...(canEdit ? [{ id: 'mine' as Tab, label: 'My decisions' }] : []),
        { id: 'all', label: 'All' },
      ]} />

      {tab === 'mine' ? (
        mine.error ? <ErrorBox error={mine.error} what="your decisions" />
        : !mine.data?.length ? <Empty title="No decisions yet">Approve or reject a recommendation and it appears here.</Empty>
        : <div className="table-wrap">
            <table className="t">
              <thead><tr><th>Decided</th><th>Trading day</th><th>SKU</th><th>Channel</th><th>Decision</th><th className="num">Price</th><th>Rule</th><th>Note</th></tr></thead>
              <tbody>{mine.data.map(d => (
                <tr key={d.id}>
                  <td className="mono faint">{ago(d.ts)}</td>
                  <td className="mono">{shortDate(d.display_date)}</td>
                  <td className="mono"><DLink to={`/products/${d.sku}`}>{d.sku}</DLink></td>
                  <td><ChannelKey channel={d.channel} /></td>
                  <td><StatusBadge status={d.action === 'approve' ? 'approved' : 'rejected'} /></td>
                  <td className="num"><PriceMove raw={d.raw_price} final={d.final_price} /></td>
                  <td><BoundBadges bound={d.bound_by} /></td>
                  <td className="muted">{d.note ?? ''}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
      ) : (
        <>
          {tab === 'pending' && (
            <div style={{ marginBottom: '.75rem' }}>
              <Callout kind="info">
                {gated
                  ? <>Only approved prices go on sale. Anything you leave undecided keeps today's shelf price when the next trading day opens.</>
                  : <>Anything you leave undecided <strong>goes on sale as recommended</strong> when the next trading day opens. Reject a price to keep the shelf price.</>}
              </Callout>
            </div>
          )}
          <div className="filters">
            <input className="input" placeholder="Search SKU" value={q} onChange={e => setQ(e.target.value)} style={{ width: 200 }} />
            <select className="select" value={channel} onChange={e => setChannel(e.target.value)} aria-label="Channel">
              <option value="">All channels</option>
              {(status.data?.channels ?? []).map(c => <option key={c} value={c}>{channelLabel(c)}</option>)}
            </select>
            {tab === 'all' && (
              <select className="select" value={statusFilter} onChange={e => setStatusFilter(e.target.value as RecStatus | '')} aria-label="Status">
                <option value="">Any status</option>
                <option value="pending">Pending</option><option value="approved">Approved</option>
                <option value="rejected">Rejected</option><option value="superseded">Superseded</option>
              </select>
            )}
            <label className="caption" style={{ display: 'inline-flex', gap: '.35rem', alignItems: 'center', cursor: 'pointer' }}>
              <input type="checkbox" checked={onlyBound} onChange={e => setOnlyBound(e.target.checked)} /> Held by a rule
            </label>
            <span className="caption">{num(total)} match</span>
            {picked.size > 0 && (
              <span className="btn-row" style={{ marginLeft: 'auto' }}>
                <span className="caption">{picked.size} selected</span>
                <button className="btn btn-sm btn-primary" disabled={busy} onClick={() => bulk('approve')}>Approve selected</button>
                <button className="btn btn-sm btn-danger" disabled={busy} onClick={() => bulk('reject')}>Reject selected</button>
              </span>
            )}
          </div>

          {recs.error ? <ErrorBox error={recs.error} what="recommendations" />
            : !recs.loading && !rows.length
              ? <Empty title={tab === 'pending' ? 'Nothing awaiting a decision' : 'No recommendations match'}>
                  {tab === 'pending' ? 'The next trading day brings new ones.' : 'Try clearing the filters.'}
                </Empty>
              : <div className="table-wrap">
                  <table className="t">
                    <thead><tr>
                      {canEdit && <th style={{ width: 32 }}>
                        <input type="checkbox" aria-label="Select all pending on this page" checked={allPicked} disabled={!pendingIds.length}
                          onChange={() => setPicked(allPicked ? new Set() : new Set(pendingIds))} />
                      </th>}
                      <th title="Trading day">Day</th><th>SKU</th><th>Channel</th>
                      <th className="num" title="On the shelf now (pending), or what went on sale (decided)">Shelf now</th>
                      <th className="num" style={{ textAlign: 'center' }}>Recommended</th><th style={{ textAlign: 'center' }} title="Allowed band (shaded), model's own price (hollow), recommendation (solid), base (tick)">In band</th><th className="num">Margin</th><th>Rule</th>
                      <th className="num" title="Stock this channel believes is available / actual stock">Stock</th>
                      <th>Decision</th>
                    </tr></thead>
                    <tbody>{rows.map(r => (
                      <tr key={r.id} className={picked.has(r.id) ? 'selected' : ''}>
                        {canEdit && <td>{r.status === 'pending' && <input type="checkbox" aria-label={`Select ${r.sku} ${r.channel}`} checked={picked.has(r.id)} onChange={() => toggle(r.id)} />}</td>}
                        <td className="mono">{shortDate(r.display_date)}</td>
                        <td>
                          <div className="mono"><DLink to={`/products/${r.sku}`}>{r.sku}</DLink></div>
                          {recContext(r) && <div className="caption" style={{ maxWidth: 280, whiteSpace: 'normal' }}>{recContext(r)}</div>}
                        </td>
                        <td><ChannelKey channel={r.channel} /></td>
                        <td className="num">{money(r.status === 'pending' ? r.shelf_price : r.live_price)}</td>
                        <td className="num" style={{ textAlign: 'center' }}><PriceMove raw={r.raw_price} final={r.final_price} base={r.base_price} /></td>
                        <td style={{ textAlign: 'center' }}><PriceBand raw={r.raw_price} final={r.final_price} base={r.base_price} floor={r.floor} ceiling={r.ceiling} width={84} /></td>
                        <td className="num">{pct(r.margin_pct)}</td>
                        <td><BoundBadges bound={r.bound_by} conflict={r.conflict} /></td>
                        <td className="num">
                          {num(r.inventory_view)} <span className="faint">/ {num(r.true_available)}</span>
                        </td>
                        <td>{r.status === 'pending' ? <Decide rec={r} compact onDone={hold} /> :
                          <span className="btn-row" style={{ gap: '.4rem' }}><StatusBadge status={r.status} liveSource={r.live_source} />{r.decided_by && <span className="caption">{r.decided_by}</span>}</span>}</td>
                      </tr>
                    ))}</tbody>
                  </table>
                  <Pager total={total} offset={offset} limit={limit} onOffset={o => { setOffset(o); setPicked(new Set()) }} onLimit={setLimit} sizes={[10, 25, 50]} />
                </div>}
        </>
      )}
    </>
  )
}
