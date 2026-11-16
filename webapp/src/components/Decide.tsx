import { useState } from 'react'
import { api, ApiError, type Recommendation } from '../api'
import { useApp } from '../context'
import { money } from '../format'
import { StatusBadge } from './ui'

/**
 * Approve / reject one recommendation, one click each. Signed by the session;
 * a viewer sees the status but no buttons.
 */
export function Decide({ rec, onDone }: { rec: Recommendation; onDone?: () => void; compact?: boolean }) {
  const { toast, bump, canEdit } = useApp()
  const [busy, setBusy] = useState(false)

  if (rec.status !== 'pending' || !canEdit) return <StatusBadge status={rec.status} />

  const go = async (action: 'approve' | 'reject') => {
    setBusy(true)
    try {
      if (action === 'approve') await api.approve(rec.id)
      else await api.reject(rec.id)
      toast(action === 'approve'
        ? `Approved ${rec.sku} on ${rec.channel} at ${money(rec.final_price)}`
        : `Rejected ${rec.sku} on ${rec.channel}`)
      bump(); onDone?.()
    } catch (e) {
      toast(e instanceof ApiError ? e.message : String(e), 'err')
    } finally { setBusy(false) }
  }

  return (
    <div className="btn-row" style={{ flexWrap: 'nowrap' }}>
      <button className="btn btn-sm btn-primary" disabled={busy} onClick={() => go('approve')}>Approve</button>
      <button className="btn btn-sm btn-danger" disabled={busy} onClick={() => go('reject')}>Reject</button>
    </div>
  )
}
