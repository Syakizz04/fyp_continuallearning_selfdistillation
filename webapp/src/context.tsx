import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from 'react'
import { api, type Status } from './api'
import { useAuth } from './auth'
import { usePoll, type Poll } from './hooks'

interface App {
  /** the signed-in username; the gateway signs every change with it */
  user: string
  /** operators change things; viewers only look */
  canEdit: boolean
  /** the gateway's view of the system, polled every 2 s for the whole app */
  status: Poll<Status>
  toast: (message: string, kind?: 'ok' | 'err') => void
  /** bumped after any write, so open pages refetch at once instead of waiting a poll */
  version: number
  bump: () => void
}

const Ctx = createContext<App | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const { me } = useAuth()
  const user = me.username
  const canEdit = me.role === 'operator'
  const [version, setVersion] = useState(0)
  const [toastState, setToast] = useState<{ message: string; kind: 'ok' | 'err' } | null>(null)
  const timer = useRef<number>(undefined)
  const status = usePoll(api.status, 2000)

  const toast = useCallback((message: string, kind: 'ok' | 'err' = 'ok') => {
    setToast({ message, kind })
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setToast(null), kind === 'err' ? 6000 : 3000)
  }, [])
  const bump = useCallback(() => { setVersion(v => v + 1); status.refresh() }, [status])

  return (
    <Ctx.Provider value={{ user, canEdit, status, toast, version, bump }}>
      {children}
      {toastState && <div className={`toast${toastState.kind === 'err' ? ' err' : ''}`} role="status">{toastState.message}</div>}
    </Ctx.Provider>
  )
}

export function useApp(): App {
  const c = useContext(Ctx)
  if (!c) throw new Error('useApp outside AppProvider')
  return c
}
