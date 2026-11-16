import { createContext, useCallback, useContext, useEffect, useState, type FormEvent, type ReactNode } from 'react'
import { api, ApiError, SIGNED_OUT, type Me } from './api'
import { PasswordInput } from './components/ui'

interface Auth {
  me: Me
  signOut: () => Promise<void>
}

const Ctx = createContext<Auth | null>(null)

/** Nothing below this renders until there is a session: the gateway refuses
 *  every /api call without one, so the app would only show errors. */
export function AuthGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<{ loading: boolean; setup: boolean; me: Me | null; error?: string }>(
    { loading: true, setup: false, me: null })

  const check = useCallback(() => {
    api.authStatus()
      .then(s => setState({ loading: false, setup: s.setup_required, me: s.user }))
      .catch(e => setState({ loading: false, setup: false, me: null, error: e instanceof ApiError ? e.message : String(e) }))
  }, [])
  useEffect(check, [check])
  // Any 401 (session expired, signed out elsewhere) drops back to the sign-in screen.
  useEffect(() => {
    const onOut = () => setState(s => (s.me ? { ...s, me: null } : s))
    window.addEventListener(SIGNED_OUT, onOut)
    return () => window.removeEventListener(SIGNED_OUT, onOut)
  }, [])

  const signOut = useCallback(async () => {
    try { await api.logout() } finally { setState({ loading: false, setup: false, me: null }) }
  }, [])

  if (state.loading) return null
  if (!state.me) return <SignIn setup={state.setup} unreachable={state.error} onDone={me => setState({ loading: false, setup: false, me })} />
  return <Ctx.Provider value={{ me: state.me, signOut }}>{children}</Ctx.Provider>
}

export function useAuth(): Auth {
  const c = useContext(Ctx)
  if (!c) throw new Error('useAuth outside AuthGate')
  return c
}

function SignIn({ setup, unreachable, onDone }: { setup: boolean; unreachable?: string; onDone: (me: Me) => void }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [again, setAgain] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (setup && password !== again) { setError('The passwords do not match.'); return }
    setBusy(true); setError(null)
    try {
      const out = setup ? await api.setup(username.trim(), password) : await api.login(username.trim(), password)
      onDone(out.user)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally { setBusy(false) }
  }

  return (
    <div className="signin">
      <form className="signin-card" onSubmit={submit}>
        <div className="brand" style={{ padding: 0, border: 0, marginBottom: '1.4rem' }}>
          <div className="brand-mark" aria-hidden />
          <div className="brand-name">Shelfsense</div>
        </div>
        <h1 className="signin-title">{setup ? 'Create the operator account' : 'Sign in'}</h1>
        {unreachable && <p className="signin-error">Gateway unreachable: {unreachable}</p>}
        <div className="field">
          <label htmlFor="su">Username</label>
          <input id="su" className="input" autoFocus autoComplete="username" value={username} onChange={e => setUsername(e.target.value)} />
        </div>
        <div className="field" style={{ marginTop: '.8rem' }}>
          <label htmlFor="sp">Password</label>
          <PasswordInput id="sp" autoComplete={setup ? 'new-password' : 'current-password'} value={password} onChange={setPassword} />
          {setup && <span className="hint">At least 8 characters.</span>}
        </div>
        {setup && (
          <div className="field" style={{ marginTop: '.8rem' }}>
            <label htmlFor="sp2">Confirm password</label>
            <PasswordInput id="sp2" autoComplete="new-password" value={again} onChange={setAgain} />
          </div>
        )}
        {error && <p className="signin-error">{error}</p>}
        <button className="btn btn-primary signin-btn" disabled={busy || !username.trim() || !password}>
          {setup ? 'Create account' : 'Sign in'}
        </button>
      </form>
    </div>
  )
}
