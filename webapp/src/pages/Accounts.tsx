import { useState, type FormEvent } from 'react'
import { api, ApiError, type Role } from '../api'
import { useAuth } from '../auth'
import { Empty, ErrorBox, PageHead, PasswordInput, Section } from '../components/ui'
import { useApp } from '../context'
import { ago } from '../format'
import { usePoll } from '../hooks'

const ROLE_NOTE: Record<Role, string> = {
  operator: 'decides prices, edits rules, runs the simulation, manages accounts',
  viewer: 'sees everything, changes nothing',
}

export default function Accounts() {
  const { me } = useAuth()
  const { toast, canEdit } = useApp()
  const users = usePoll(api.users, null)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState<Role>('viewer')
  const [busy, setBusy] = useState(false)

  const add = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    try {
      await api.addUser(username.trim(), password, role)
      toast(`Added ${username.trim()} (${role})`)
      setUsername(''); setPassword(''); users.refresh()
    } catch (err) { toast(err instanceof ApiError ? err.message : String(err), 'err') }
    finally { setBusy(false) }
  }
  const remove = async (name: string) => {
    if (!window.confirm(`Remove ${name}? Their past decisions stay in the history.`)) return
    try { await api.removeUser(name); toast(`Removed ${name}`); users.refresh() }
    catch (err) { toast(err instanceof ApiError ? err.message : String(err), 'err') }
  }

  return (
    <>
      <PageHead title="Accounts" eyebrow={`signed in as ${me.username} · ${me.role}`} />

      <Section title="People">
        {users.error ? <ErrorBox error={users.error} what="accounts" />
          : !users.data?.length ? <Empty title="Loading…" />
          : <div className="table-wrap">
              <table className="t">
                <thead><tr><th>Username</th><th>Role</th><th>Can</th><th>Added</th><th>By</th><th /></tr></thead>
                <tbody>{users.data.map(u => (
                  <tr key={u.username}>
                    <td className="mono">{u.username}{u.username === me.username && <span className="faint"> (you)</span>}</td>
                    <td><span className={`badge${u.role === 'operator' ? ' badge-good' : ''}`}>{u.role}</span></td>
                    <td className="muted">{ROLE_NOTE[u.role]}</td>
                    <td className="mono faint">{ago(u.created_at)}</td>
                    <td className="mono faint">{u.created_by ?? '—'}</td>
                    <td style={{ textAlign: 'right' }}>
                      {canEdit && u.username !== me.username && <button className="btn btn-sm btn-danger" onClick={() => remove(u.username)}>Remove</button>}
                    </td>
                  </tr>
                ))}</tbody>
              </table>
            </div>}
      </Section>

      {canEdit && (
        <Section title="Add someone">
          <form className="card" onSubmit={add} style={{ display: 'flex', gap: '.8rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
            <div className="field"><label htmlFor="nu">Username</label>
              <input id="nu" className="input" value={username} autoComplete="off" onChange={e => setUsername(e.target.value)} /></div>
            <div className="field"><label htmlFor="np">Password</label>
              <PasswordInput id="np" autoComplete="new-password" value={password} onChange={setPassword} /></div>
            <div className="field"><label htmlFor="nr">Role</label>
              <select id="nr" className="select" value={role} onChange={e => setRole(e.target.value as Role)}>
                <option value="viewer">Viewer</option><option value="operator">Operator</option>
              </select></div>
            <button className="btn btn-primary" disabled={busy || !username.trim() || password.length < 8}>Add account</button>
          </form>
        </Section>
      )}
    </>
  )
}
