import { useState } from 'react'
import { api, ApiError, type Recommendation } from '../api'
import { useApp } from '../context'
import { channelLabel, money } from '../format'
import { StatusBadge } from './ui'

/**
 * Approve / reject one recommendation, one click each. Signed by the session;
 * a viewer sees the status but no buttons. `onDone` receives the decided
 * recommendation, so a list can keep the row where it was instead of letting
 * the next one slide up under the pointer.
 */
export function Decide({ rec, onDone }: { rec: Recommendation; onDone?: (decided: Recommendation) => void; compact?: boolean }) {
  const { toast, bump, canEdit } = useApp()
  const [busy, setBusy] = useState(false)

  if (rec.status !== 'pending' || !canEdit) return <StatusBadge status={rec.status} liveSource={rec.live_source} />

  const go = async (action: 'approve' | 'reject') => {
    setBusy(true)
    try {
      const decided = action === 'approve' ? await api.approve(rec.id) : await api.reject(rec.id)
      const where = channelLabel(rec.channel)
      toast(action === 'approve'
        ? `Approved ${rec.sku} on ${where} at ${money(rec.final_price)}`
        : `Rejected ${rec.sku} on ${where}`)
      onDone?.(decided); bump()
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
